# screening.py
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import math
import os
import re

# Optional semantic boost if the model is available; otherwise keywords only.
try:
    from sentence_transformers import SentenceTransformer, util
    _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
except Exception:
    _embedder = None

# Optional FAISS integration (lazy / best-effort; safe fallbacks kept)
try:
    from mental_health_faiss import MentalHealthQuestionsFAISS  # type: ignore
except Exception:
    MentalHealthQuestionsFAISS = None  # type: ignore

# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------
@dataclass
class Evidence:
    text: str
    feature: str
    weight: float
    source: str

@dataclass
class ConditionResult:
    name: str                      # "depression" | "anxiety" | "psychosis"
    severity: str                  # PHQ/GAD bucket or "positive"/"negative" for psychosis
    score: float                   # normalized 0..1 (PHQ/GAD) or 0/1 (psychosis)
    confidence: float              # 0..1
    rationale: List[Evidence]
    next_steps: List[str]
    referral: Optional[str]        # None | "routine" | "urgent"

@dataclass
class ScreeningOutput:
    results: List[ConditionResult]
    overall_flag: str              # "ok" | "monitor" | "refer_routine" | "refer_urgent"


# ---------------------------------------------------------------------
# Thresholds / mappings
# ---------------------------------------------------------------------
PHQ9_THRESH = [(0,4,"none"),(5,9,"mild"),(10,14,"moderate"),(15,19,"moderately_severe"),(20,27,"severe")]
GAD7_THRESH = [(0,4,"none"),(5,9,"mild"),(10,14,"moderate"),(15,21,"severe")]

PHQ9_MAP = {
    "phq9_q01":"anhedonia","phq9_q02":"low_mood","phq9_q03":"sleep_change","phq9_q04":"fatigue",
    "phq9_q05":"appetite_change","phq9_q06":"worthlessness_guilt","phq9_q07":"concentration",
    "phq9_q08":"psychomotor","phq9_q09":"suicidality"
}
GAD7_MAP = {
    "gad7_q01":"excess_worry","gad7_q02":"excess_worry","gad7_q03":"restlessness",
    "gad7_q04":"poor_concentration","gad7_q05":"irritability","gad7_q06":"muscle_tension",
    "gad7_q07":"sleep_disturbance"
}

# Bilingual symptom lexicon (short, expandable)
SYMPTOMS = {
    "depression": {
        "anhedonia": [
            "lost interest","no interest","lack of pleasure","nothing is enjoyable",
            "sina hamu","sitaki kufanya chochote"
        ],
        "low_mood": [
            "feeling down","sad","depressed","blue","hopeless","helpless",
            "huzuni","moyo chini","sina matumaini","sina msaada"
        ],
        "sleep_change": [
            "trouble sleeping","insomnia","sleeping too much","kulala vibaya",
            "usingizi duni","ninalala sana","usingizi mwingi"
        ],
        "fatigue": [
            "tired","fatigue","exhausted","drained","no energy",
            "uchovu","kuchoka","ninaishiwa nguvu"
        ],
        "appetite_change": [
            "poor appetite","overeating","weight loss","weight gain",
            "hamu ya kula imepungua","nakula sana","kupoteza uzito","kuongeza uzito"
        ],
        "worthlessness_guilt": [
            "worthless","guilty","failure","self blame","ashamed",
            "najilaumu","sina thamani","nimefeli","nina aibu"
        ],
        "concentration": [
            "trouble concentrating","hard to focus","mind is slow",
            "siwezi kuzingatia","mawazo hayakai","ninakosa umakini"
        ],
        "psychomotor": [
            "moving slowly","slowed down","restless","fidgety",
            "mwendo polepole","nasitasita","nasisimka","natembea polepole"
        ],
        "suicidality": [
            "suicidal","suicide","want to die","kill myself","end my life",
            "self harm","self-harm","hurt myself","harm myself","cut myself",
            "kujiua","nataka kufa","najiumiza","kujiumiza"
        ],
        "crying": [
            "crying a lot","tearful","weeping","breaking down",
            "nalia mara nyingi","kulia","nalia sana"
        ],
        "social_withdrawal": [
            "avoiding people","don’t want to see anyone","isolated",
            "nimejitenga","sitaki kuonana na mtu","niko peke yangu"
        ],
        "low_self_esteem": [
            "I feel like a failure","not good enough","I am useless",
            "mimi ni bure","sina thamani","nimekosa maana"
        ]
    },

    "anxiety": {
        "excess_worry": [
            "worry a lot","cannot control worry","always worried","overthinking",
            "nina wasiwasi","sidhhibiti wasiwasi","siwezi kudhibiti wasiwasi","kuwaza sana"
        ],
        "restlessness": [
            "on edge","restless","uneasy","fidgety","nervous",
            "nasisimka","msononeko","nahisi wasi"
        ],
        "poor_concentration": [
            "mind goes blank","hard to focus","cannot concentrate",
            "siwezi kuzingatia","mawazo hayakai"
        ],
        "irritability": [
            "irritable","short tempered","easily annoyed","snappy",
            "nakasirika haraka","nakasirika kwa urahisi"
        ],
        "muscle_tension": [
            "muscle tension","tense","tight muscles","jaw clenching",
            "misuli ngumu","mwili umekakamaa","natetemeka"
        ],
        "sleep_disturbance": [
            "trouble sleeping","difficulty staying asleep","usingizi duni","kulala vibaya",
            "ninaamka usiku","sioni usingizi"
        ],
        "panic": [
            "heart racing","palpitations","short of breath","chest tightness",
            "panic attack","suffocating","nashindwa kupumua","moyo wangu unapiga haraka"
        ],
        "somatic": [
            "stomach upset","nausea","sweating","shaking","trembling",
            "kichefuchefu","jasho jingi","ninatetemeka","maumivu ya tumbo"
        ],
        "phobias": [
            "fear of leaving home","afraid of crowds","fear of public places",
            "naogopa kutoka nje","naogopa watu","naogopa umati"
        ]
    },

    "psychosis": {
        "auditory_hallucination": [
            "hearing voices","voices talking","voices commenting",
            "sauti ambazo wengine hawasikii","ninasikia sauti"
        ],
        "visual_hallucination": [
            "seeing things","visions","shadows moving",
            "naona vitu wengine hawaoni","naona kivuli kikisogea"
        ],
        "delusions_persecution": [
            "out to get me","people following me","being spied on","poisoned",
            "wananifuatilia","wamenilenga","wamenipelelea"
        ],
        "delusions_reference": [
            "messages for me","TV talking to me","radio sending messages",
            "ujumbe kupitia tv","ujumbe kupitia radio","wananiambia kwa matangazo"
        ],
        "thought_disorganization": [
            "thoughts mixed up","cannot think straight","speech disorganized",
            "mawazo yamechanganyikana","nashindwa kufikiri","maneno hayana mpangilio"
        ],
        "bizarre_behavior": [
            "strange behavior","acting odd","weird actions",
            "tabia ya ajabu","nafanya mambo yasiyoelezeka"
        ],
        "paranoia": [
            "people are watching me","they put cameras","spying on me",
            "wananichunguza","wamenifunga kamera","wananifuata"
        ],
        "negative_symptoms": [
            "lack of motivation","flat affect","emotionless","not talking",
            "hakuna motisha","hakuna hisia","sionyeshi uso","siongei"
        ],
        "catatonia": [
            "staring blankly","not moving","frozen","repetitive movements",
            "natizama bila kuelewa","nimeganda","narudia tabia moja"
        ]
    }
}

RED_FLAGS = set(SYMPTOMS["psychosis"].keys())


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))

def _sum_scale(responses: Dict[str, int], prefix: str):
    items = [(k, int(v)) for k, v in (responses or {}).items() if k.startswith(prefix)]
    return sum(v for _, v in items), items

def _bucket(score: int, table):
    for lo, hi, name in table:
        if lo <= score <= hi:
            return name
    return table[-1][2]


# ---------------------------------------------------------------------
# Evidence extraction & signals
# ---------------------------------------------------------------------
def extract_symptoms(text: str) -> List[Evidence]:
    t = (text or "").lower()
    ev: List[Evidence] = []
    if not t:
        return ev  # always a list

    # keyword hits
    for cond, feats in SYMPTOMS.items():
        for feat, kws in feats.items():
            for kw in kws:
                if kw in t:
                    base_w = 0.7
                    if feat == "suicidality":
                        base_w = 1.6
                    elif cond == "psychosis":
                        base_w = 1.2
                    ev.append(Evidence(text=kw, feature=feat, weight=base_w, source="transcript"))

    # semantic paraphrase assist (best-effort; never crash)
    try:
        if _embedder and t.strip():
            phrases = [p for feats in SYMPTOMS.values() for lst in feats.values() for p in lst]
            emb_t = _embedder.encode([t], normalize_embeddings=True)
            emb_p = _embedder.encode(phrases, normalize_embeddings=True)
            sims = util.cos_sim(emb_t, emb_p).cpu().tolist()[0]
            for i, s in enumerate(sims):
                if s >= 0.42:
                    ev.append(Evidence(text=phrases[i], feature="semantic_hit", weight=0.4, source="transcript"))
    except Exception:
        # swallow embedder issues; keep whatever we already found
        pass

    return ev  # ← don’t forget this!


def evidence_signal(evs: Optional[List[Evidence]], condition: str) -> float:
    evs = evs or []  # ← key fix
    relevant = 0.0
    for e in evs:
        if e.feature == "semantic_hit" and condition != "psychosis":
            relevant += 0.25
        elif e.feature in SYMPTOMS.get(condition, {}):
            relevant += e.weight
    # saturating growth: 1 - exp(-x)
    return 1.0 - math.exp(-relevant)


def screen_psychosis(evidence: List[Evidence], safety: bool = False):
    """Keyword/semantic rule for psychosis red flags."""
    strong = [e for e in evidence if e.feature in RED_FLAGS and e.weight >= 0.6]
    if strong:
        urgent = bool(safety)
        conf = min(0.75 + 0.1 * (len(strong) - 1) + (0.1 if urgent else 0.0), 0.98)
        return True, conf, ("urgent" if urgent else "routine"), strong
    return False, 0.05, None, strong


# ---------------------------------------------------------------------
# Light duration extraction (English + Swahili)
# ---------------------------------------------------------------------
_SW_DUR = r"(wiki|miezi|mwezi|siku)"
_EN_DUR = r"(day|days|week|weeks|month|months|year|years)"

def _duration_weeks(text: str) -> float:
    """Very light extractor: returns approx duration in weeks if found, else 0."""
    t = (text or "").lower()
    # english: "for about three months", "two weeks", "3 months"
    m = re.search(r"(?:for\s+about\s+|for\s+|around\s+)?(\d+)\s*(" + _EN_DUR + r")", t)
    if not m:
        # swahili: "kwa takriban miezi mitatu", "wiki mbili"
        m = re.search(r"(?:kwa\s+(?:takriban|karibu)\s+)?(\d+)\s*(" + _SW_DUR + r")", t)
    if not m:
        return 0.0
    n = float(m.group(1))
    unit = m.group(2)
    if unit.startswith('day') or unit == 'siku':   return n/7.0
    if unit.startswith('week') or unit == 'wiki':  return n
    if unit.startswith('month') or unit in ('mwezi','miezi'): return n*4.0
    if unit.startswith('year'):                   return n*52.0
    return 0.0


# ---------------------------------------------------------------------
# Text → proxy PHQ/GAD if scales absent
# ---------------------------------------------------------------------
def _proxy_phq_from_text(evs: List[Evidence], transcript: str) -> int:
    feats = {e.feature for e in evs if e.feature}
    pts = 0

    # core (heavier)
    if 'anhedonia' in feats:  pts += 4
    if 'low_mood' in feats:   pts += 4

    # supporting
    for f in ('sleep_change','appetite_change','fatigue','concentration','psychomotor',
              'worthlessness_guilt','crying','social_withdrawal','low_self_esteem'):
        if f in feats: pts += 3

    # suicidality dominates
    if 'suicidality' in feats:
        pts = max(pts, 18)   # ensure ≥ moderately severe range
        pts += 4

    # duration criterion
    if _duration_weeks(transcript) >= 2:
        pts += 4

    return min(int(pts), 27)


def _proxy_gad_from_text(evs: List[Evidence], transcript: str) -> int:
    feats = {e.feature for e in evs if e.feature}
    pts = 0

    for f in ('excess_worry','restlessness','poor_concentration','irritability',
              'muscle_tension','sleep_disturbance'):
        if f in feats: pts += 4

    # panic & somatic symptoms push harder
    if 'panic' in feats:   pts += 4
    if 'somatic' in feats: pts += 3

    if _duration_weeks(transcript) >= 4:
        pts += 3

    return min(int(pts), 21)


# ---------------------------------------------------------------------
# FAISS grounding (surgical, best-effort)
# ---------------------------------------------------------------------
_FAISS_INSTANCE = None  # holds a MentalHealthQuestionsFAISS, if available

def set_faiss_instance(faiss_obj) -> None:
    """
    Allows the app to inject its already-loaded FAISS instance.
    Call once from app.py after initialize_faiss(): screening.set_faiss_instance(faiss_system)
    """
    global _FAISS_INSTANCE
    _FAISS_INSTANCE = faiss_obj

def _lazy_load_faiss_from_env() -> None:
    """As a fallback, try to load FAISS using env/config paths once."""
    global _FAISS_INSTANCE
    if _FAISS_INSTANCE is not None:
        return
    try:
        if MentalHealthQuestionsFAISS is None:
            return
        index_path = os.getenv("FAISS_INDEX_PATH") or ""
        meta_path  = os.getenv("FAISS_METADATA_PATH") or ""
        if not (index_path and meta_path and os.path.exists(index_path) and os.path.exists(meta_path)):
            return
        fs = MentalHealthQuestionsFAISS()
        fs.load_index(index_path, meta_path)
        _FAISS_INSTANCE = fs
    except Exception:
        # stay None; we’ll fall back to keyword/semantic signals
        _FAISS_INSTANCE = None

def _faiss_label_scores(text: str) -> Dict[str, float]:
    """
    Returns normalized label weights (0..1) per {depression, anxiety, psychosis}
    using FAISS matches against your curated question DB. Safe fallback: {}.
    """
    if not text or not text.strip():
        return {}
    # Prefer injected instance; else lazy-load from env once.
    if _FAISS_INSTANCE is None:
        _lazy_load_faiss_from_env()
    try:
        if _FAISS_INSTANCE is None:
            return {}
        # We assume the instance exposes label_scores_from_text(text) as suggested.
        scores = _FAISS_INSTANCE.label_scores_from_text(text)  # type: ignore[attr-defined]
        if not isinstance(scores, dict):
            return {}
        # Normalize / clamp defensively
        out = {}
        total = sum(float(v) for v in scores.values() if isinstance(v, (int, float))) or 1.0
        for k in ("depression","anxiety","psychosis"):
            v = float(scores.get(k, 0.0))
            out[k] = _clamp(v / total, 0.0, 1.0)
        return out
    except Exception:
        return {}


# ---------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------
def run_screening(
    transcript: str,
    responses: Dict[str, int],
    safety_concerns: bool = False
) -> ScreeningOutput:
    # 1) Free-text evidence (keyword + semantic)
    text_evidence = extract_symptoms(transcript or "") or []  # never None

    # 2) Questionnaire totals
    phq_total, phq_items = _sum_scale(responses, "phq9_")
    gad_total, gad_items = _sum_scale(responses, "gad7_")

    # 3) Proxy from text if scales are empty
    if phq_total == 0:
        phq_proxy = _proxy_phq_from_text(text_evidence, transcript or "")
        if phq_proxy > 0:
            phq_total = phq_proxy
            # mark text-derived rationale
            for e in [e for e in text_evidence if e.feature in SYMPTOMS["depression"]]:
                e.source = (e.source or "text") + "→PHQ-proxy"

    if gad_total == 0:
        gad_proxy = _proxy_gad_from_text(text_evidence, transcript or "")
        if gad_proxy > 0:
            gad_total = gad_proxy
            for e in [e for e in text_evidence if e.feature in SYMPTOMS["anxiety"]]:
                e.source = (e.source or "text") + "→GAD-proxy"

    # 4) Severity buckets and normalized scores
    phq_sev = _bucket(phq_total, PHQ9_THRESH)
    gad_sev = _bucket(gad_total, GAD7_THRESH)
    phq_norm = min(phq_total / 27.0, 1.0)
    gad_norm = min(gad_total / 21.0, 1.0)

    # 5) Build item-level rationale from scale answers
    phq_ev = [
        Evidence(text=f"{qid}={val}", feature=PHQ9_MAP.get(qid, "phq9_item"),
                 weight=(val / 3) * 0.6, source="PHQ-9")
        for qid, val in phq_items
    ]
    gad_ev = [
        Evidence(text=f"{qid}={val}", feature=GAD7_MAP.get(qid, "gad7_item"),
                 weight=(val / 3) * 0.6, source="GAD-7")
        for qid, val in gad_items
    ]

    # 6) FAISS label weights from the transcript (primary conversational signal)
    faiss_lbl = _faiss_label_scores(transcript or "")
    # Fall back to keyword/semantic text signal if FAISS not available
    if faiss_lbl:
        dep_text = float(faiss_lbl.get("depression", 0.0))
        anx_text = float(faiss_lbl.get("anxiety",    0.0))
        psy_text = float(faiss_lbl.get("psychosis",  0.0))
    else:
        dep_text = evidence_signal(text_evidence, "depression")
        anx_text = evidence_signal(text_evidence, "anxiety")
        psy_text = evidence_signal(text_evidence, "psychosis")  # weaker than FAISS, used only as fallback

    # 7) Psychosis rule (keywords) and FAISS combo
    psy_flag_kw, psy_conf_kw, psy_ref_kw, psy_ev_kw = screen_psychosis(text_evidence, safety_concerns)
    # FAISS-derived psychosis confidence (if available): base + safety bump
    psy_conf_faiss = _clamp(psy_text + (0.10 if safety_concerns and psy_text > 0 else 0.0), 0.05, 0.99)
    # Final flag/conf/referral: take the stronger signal
    psy_flag = psy_flag_kw or (psy_text >= 0.60)  # FAISS high score can flag it
    psy_conf = max(psy_conf_kw, psy_conf_faiss)
    psy_ref = "urgent" if (psy_flag and safety_concerns) else ("routine" if psy_flag else None)
    psy_ev = psy_ev_kw  # keep rationale list from keyword/semantic hits

    # 8) Dynamic confidence blending for depression/anxiety
    no_phq = (len(phq_items) == 0)
    no_gad = (len(gad_items) == 0)

    # If no scale items answered, weight conversation (FAISS/text) more heavily
    w_scale_dep, w_text_dep = (0.3, 0.7) if no_phq else (0.5, 0.5)
    w_scale_anx,  w_text_anx  = (0.3, 0.7) if no_gad else (0.5, 0.5)

    lower_bound_dep = 0.35 if no_phq else 0.25
    lower_bound_anx = 0.35 if no_gad else 0.25

    dep_conf = _clamp(w_scale_dep * phq_norm + w_text_dep * dep_text, lower_bound_dep, 0.99)
    anx_conf = _clamp(w_scale_anx * gad_norm + w_text_anx * anx_text, lower_bound_anx, 0.99)

    # 9) Next steps
    dep_steps = {
        "none": ["Maintain healthy routines (sleep, activity, social).",
                 "Monitor mood; reach out if things change."],
        "mild": ["Self-help: activity plan, light exercise, journaling.",
                 "Talk with trusted family/friends or faith leader (church/mosque)."],
        "moderate": ["Consider structured therapy referral (CBT).",
                     "If worsening or suicidal thoughts, seek urgent care."],
        "moderately_severe": ["Refer to mental health professional for evaluation.",
                              "Safety check; crisis plan if suicidal thoughts."],
        "severe": ["Urgent referral to mental health services.",
                   "Immediate safety assessment if suicidality or inability to function."]
    }[phq_sev]

    anx_steps = {
        "none": ["Continue stress-reducing routines.",
                 "Practice brief breathing/relaxation when needed."],
        "mild": ["Daily breathing 5–10 min; sleep hygiene.",
                 "Lean on social/faith support (family/church)."],
        "moderate": ["Therapy referral (CBT), psychoeducation on worry cycles.",
                     "Limit caffeine; schedule 'worry time'."],
        "severe": ["Refer to mental health services; structured therapy.",
                   "If panic/safety concerns, urgent evaluation."]
    }[gad_sev]

    psy_steps = (["Refer for assessment to rule out psychosis.",
                  "If safety concerns (self/others), urgent/emergency care."]
                 if psy_flag else ["No clear psychosis signals detected; continue observation."])

    # 10) Assemble condition results
    dep = ConditionResult(
        name="depression",
        severity=phq_sev,
        score=round(phq_norm, 3),
        confidence=round(dep_conf, 3),
        rationale=(phq_ev + [e for e in text_evidence if e.feature in SYMPTOMS["depression"]])[:12],
        next_steps=dep_steps,
        referral=("urgent" if (phq_sev in ["moderately_severe", "severe"] or
                               any(e.feature == "suicidality" for e in (phq_ev + text_evidence)))
                  else None)
    )

    anx = ConditionResult(
        name="anxiety",
        severity=gad_sev,
        score=round(gad_norm, 3),
        confidence=round(anx_conf, 3),
        rationale=(gad_ev + [e for e in text_evidence if e.feature in SYMPTOMS["anxiety"]])[:12],
        next_steps=anx_steps,
        referral=("urgent" if gad_sev == "severe" and safety_concerns else None)
    )

    psy = ConditionResult(
        name="psychosis",
        severity=("positive" if psy_flag else "negative"),
        score=(1.0 if psy_flag else 0.0),
        confidence=round(psy_conf, 3),
        rationale=psy_ev[:12],
        next_steps=psy_steps,
        referral=psy_ref
    )

    # 11) Overall triage flag
    if any(r.referral == "urgent" for r in (dep, anx, psy)):
        overall = "refer_urgent"
    elif any(r.referral == "routine" for r in (dep, anx, psy)):
        overall = "refer_routine"
    elif any(r.severity in ["mild", "moderate", "moderately_severe", "severe", "positive"]
             for r in (dep, anx, psy)):
        overall = "monitor"
    else:
        overall = "ok"

    return ScreeningOutput(results=[dep, anx, psy], overall_flag=overall)


def screening_to_dict(out: ScreeningOutput) -> Dict:
    return {
        "results": [{
            "name": r.name,
            "severity": r.severity,
            "score": r.score,
            "confidence": r.confidence,
            "rationale": [e.__dict__ for e in r.rationale][:12],
            "next_steps": r.next_steps,
            "referral": r.referral
        } for r in out.results],
        "overall_flag": out.overall_flag
    }
