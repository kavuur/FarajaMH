#!/usr/bin/env python3
import json
import pickle
import faiss
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import logging
from dataclasses import dataclass
from pathlib import Path
# top of file, add:
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QuestionSearchResult:
    """Data class for question search results"""
    question_id: str
    question: Dict[str, str]  # {"english": str, "swahili": str}
    category: str | None
    tags: List[str]
    similarity_score: float


def _item_to_text(it: dict) -> str:
    q = (it.get("question") or {})
    en = (q.get("english") or "").strip()
    sw = (q.get("swahili") or "").strip()
    cat = (it.get("category") or "").strip()
    tags = it.get("tags") or []
    parts = []
    if en: parts.append(en)
    if sw: parts.append(sw)
    if cat: parts.append(f"Category: {cat}")
    if isinstance(tags, list) and tags:
        parts.append("Tags: " + " ".join(t for t in tags if t))
    if not parts:
        parts.append(it.get("id", ""))
    return " ".join(parts).strip()

class MentalHealthQuestionsFAISS:
    """
    FAISS index for mental health questions only
    """

    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.questions: List[Dict[str, Any]] = []
        self.embeddings = None
        self.dimension = None
        logger.info(f"Initialized MentalHealthQuestionsFAISS with model: {model_name}")

    # replace the entire suggest_questions() with this:
    def suggest_questions(self, query_text: str, k: int = 3, threshold: float = 0.38):
        """
        Return top-k FAISS questions (bilingual text) most relevant to query_text.
        Works with both dict and QuestionSearchResult objects.
        """
        hits = self.search(query_text, k=k * 2, threshold=threshold) or []
        out, seen = [], set()

        for h in hits:
            # --- object style (QuestionSearchResult) ---
            if hasattr(h, "question_id"):
                qid = getattr(h, "question_id", None)
                if not qid or qid in seen:
                    continue
                seen.add(qid)
                out.append({
                    "id": qid,
                    "question": dict(getattr(h, "question", {}) or {}),
                    "category": (getattr(h, "category", None) or None),
                    "tags": list(getattr(h, "tags", []) or []),
                    "similarity": float(getattr(h, "similarity_score", 0.0) or 0.0),
                })
                if len(out) >= k:
                    break
                continue

            # --- dict style (legacy/other callers) ---
            if isinstance(h, dict):
                item = h.get("item") or h.get("data") or h
                qid = item.get("id")
                if not qid or qid in seen:
                    continue
                seen.add(qid)
                out.append({
                    "id": qid,
                    "question": dict(item.get("question", {}) or {}),
                    "category": (item.get("category") or "").lower() or None,
                    "tags": item.get("tags", []) or [],
                    "similarity": float(h.get("similarity") or h.get("score") or item.get("similarity") or 0.0),
                })
                if len(out) >= k:
                    break

        return out

    def label_scores_from_text(self, text: str):
        """
        Soft label weights from text via FAISS (normalized 0..1).
        """
        hits = self.search(text, k=24, threshold=0.30) or []
        agg = {"depression": 0.0, "anxiety": 0.0, "psychosis": 0.0}
        for h in hits:
            item = h.get("item") or {}
            cat = (item.get("category") or "").lower()
            if cat in agg:
                agg[cat] += float(h.get("similarity") or 0.0)
        s = sum(agg.values()) or 1.0
        return {k: (v / s) for k, v in agg.items()}

    def _extract_text(self, q: Dict[str, Any]) -> str:
        """Combine English/Swahili text for embedding; accept legacy strings too."""
        # Legacy: q may be a plain string
        if isinstance(q, str):
            s = q.strip()
            return f"English: {s}" if s else ""

        text_parts = []
        if "question" in q and isinstance(q["question"], dict):
            eng = q["question"].get("english", "")
            swa = q["question"].get("swahili", "")
            if eng: text_parts.append(f"English: {eng}")
            if swa: text_parts.append(f"Swahili: {swa}")
        elif "question" in q and isinstance(q["question"], str):
            s = q["question"].strip()
            if s: text_parts.append(f"English: {s}")
        return " ".join([t.strip() for t in text_parts if t.strip()])

    def build_database(self, json_file: str):
        """Build FAISS index from questions.json"""
        logger.info(f"Loading questions from {json_file}")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Expected a list of questions in JSON")

        processed = []
        texts = []

        for i, q in enumerate(data):
            if "id" not in q:
                q["id"] = f"q_{i+1}"
            text = self._extract_text(q)
            if text:
                processed.append(q)
                texts.append(text)
            else:
                logger.warning(f"Skipping {q.get('id','(no id)')} (no text)")

        if not texts:
            raise ValueError("No valid questions to index")

        logger.info(f"Building embeddings for {len(texts)} questions…")
        emb = self.model.encode(texts, show_progress_bar=True)
        faiss.normalize_L2(emb)

        self.dimension = emb.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(emb.astype("float32"))

        self.questions = processed
        self.embeddings = emb

        logger.info(f"FAISS index built with {self.index.ntotal} questions")

    def search(self, query: str, k: int = 5, threshold: float = 0.4) -> List[QuestionSearchResult]:
        """Search most similar questions (robust to legacy entries)."""
        if self.index is None:
            raise RuntimeError("Index not built/loaded")

        q_emb = self.model.encode([query])
        faiss.normalize_L2(q_emb)

        k = min(k, len(self.questions))
        sims, idxs = self.index.search(q_emb.astype("float32"), k)

        results: List[QuestionSearchResult] = []
        for sim, idx in zip(sims[0], idxs[0]):
            if sim < threshold:
                continue

            q = self.questions[idx]
            if isinstance(q, str):
                q_id = f"q_{idx + 1}"
                q_question = {"english": q, "swahili": ""}
                q_cat = None
                q_tags = []
            else:
                q_id = str(q.get("id", f"q_{idx + 1}"))
                # ensure dict form for question
                if isinstance(q.get("question"), dict):
                    q_question = {
                        "english": (q["question"].get("english") or "").strip(),
                        "swahili": (q["question"].get("swahili") or "").strip(),
                    }
                else:
                    q_question = {"english": str(q.get("question", "")).strip(), "swahili": ""}
                q_cat = q.get("category")
                q_tags = list(q.get("tags", []))

            results.append(
                QuestionSearchResult(
                    question_id=q_id,
                    question=q_question,
                    category=q_cat,
                    tags=q_tags,
                    similarity_score=float(sim),
                )
            )
        return results

    def save_index(self, index_path: str, meta_path: str):
        
        faiss.write_index(self.index, str(index_path))
        meta = {
            "questions": list(self.questions),  # <-- REQUIRED
            "embedding_model": getattr(self, "model_name", None),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "dim": getattr(self.index, "d", None),
            "count": len(self.questions),
        }
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)

    def load_index(self, index_path: str, meta_path: str):
        ip, mp = Path(index_path), Path(meta_path)
        if not ip.exists() or not mp.exists():
            raise FileNotFoundError(f"Missing FAISS files:\n- {ip}\n- {mp}")

        self.index = faiss.read_index(str(ip))
        with mp.open("rb") as f:
            meta = pickle.load(f)

        # Extract questions from multiple possible legacy shapes
        questions_raw = None
        if isinstance(meta, dict):
            if isinstance(meta.get("questions"), (list, tuple)):
                questions_raw = list(meta["questions"])
            elif isinstance(meta.get("items"), (list, tuple)):
                questions_raw = list(meta["items"])
        elif isinstance(meta, (list, tuple)):
            questions_raw = list(meta)

        if not questions_raw:
            keys = list(meta.keys()) if isinstance(meta, dict) else type(meta).__name__
            raise KeyError(
                f"Could not find questions in metadata. Saw keys/type: {keys}. "
                "Expected one of: 'questions' (list[dict|str]) or 'items' (list[dict|str])."
            )

        # --- normalize to dict shape expected downstream ---
        normalized: List[Dict[str, Any]] = []
        for i, it in enumerate(questions_raw):
            if isinstance(it, dict):
                qid = it.get("id") or f"q_{i + 1}"
                qqq = it.get("question")
                if isinstance(qqq, str):
                    qqq = {"english": qqq, "swahili": ""}
                elif not isinstance(qqq, dict):
                    qqq = {"english": str(qqq or ""), "swahili": ""}
                normalized.append({
                    "id": str(qid),
                    "question": {
                        "english": (qqq.get("english") or "").strip(),
                        "swahili": (qqq.get("swahili") or "").strip(),
                    },
                    "category": (it.get("category") or None),
                    "tags": list(it.get("tags") or []),
                })
            else:
                s = str(it).strip()
                normalized.append({
                    "id": f"q_{i + 1}",
                    "question": {"english": s, "swahili": ""},
                    "category": None,
                    "tags": [],
                })

        self.questions = normalized
        self.dimension = self.index.d
        logger.info(f"Loaded FAISS index {ip.name}; questions={len(self.questions)}; dim={self.dimension}")

    def get_stats(self):
        return {
            "total_questions": len(self.questions),
            "index_built": self.index is not None,
            "dimension": self.dimension,
        }

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python mental_health_questions_faiss.py questions.json")
        sys.exit(1)

    json_file = sys.argv[1]
    out_index = "questions.index"
    out_meta = "questions_metadata.pkl"

    faiss_sys = MentalHealthQuestionsFAISS()
    faiss_sys.build_database(json_file)
    faiss_sys.save_index(out_index, out_meta)
    print("✅ Questions FAISS index built and saved.")
