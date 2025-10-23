# admin.py
from flask import Blueprint, jsonify, request, current_app
from flask_login import login_required, current_user
from sqlalchemy import func, desc, or_
from collections import Counter, defaultdict
import re

from models import (
    SessionLocal,
    Conversation,
    Message,
    User,
    Role,
    user_roles,
    ConversationOwner,
)

# Optional: FAISS-driven disease likelihoods
from mental_health_faiss import MentalHealthQuestionsFAISS

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")

# --------------------------
# Auth guards
# --------------------------
def _require_admin():
    return current_user.is_authenticated and any(r.name == "admin" for r in current_user.roles)

def admin_guard():
    if not _require_admin():
        return jsonify({"ok": False, "error": "Admin only"}), 403

# --------------------------
# Helpers: text cleaning & symptom extraction
# --------------------------
# For pulling a single target symptom from recommender text (your existing heuristic)
SYM_RE = re.compile(r"(?:symptom|target|focus)\s*:\s*([A-Za-z][\w\s/-]{1,80})", re.IGNORECASE)

def _extract_symptom(text: str) -> str | None:
    if not text:
        return None
    m = SYM_RE.search(text)
    if m:
        return m.group(1).strip()
    for kw in (
        "Sadness", "Loss of interest", "Irritability", "Hopelessness", "Guilt", "Mood swings",
        "Worry", "Panic attacks", "Restlessness", "Muscle tension", "Avoidance", "Phobias", 
        "Poor concentration", "Memory problems", "Racing thoughts", "Indecisiveness", "Rumination",
        "Intrusive thoughts", "Hallucinations", "Delusions", "Disorganized thinking", "Derealization",
        "Depersonalization", "Social withdrawal", "Sleep changes", "Appetite changes", "Reckless behavior",
        "Aggression", "Self-harm", "Fatigue", "Headaches", "Gastrointestinal issues", "Palpitations",
        "Weight changes", "Substance misuse", "Obsessions", "Compulsions", "Impaired functioning", "Emotional numbness"

    ):
        if kw in text.lower():
            return kw
    return None

# Strip legacy HTML if any
TAG_RE = re.compile(r"<[^>]+>")

def _safe_text(m: Message) -> str:
    msg = getattr(m, "message", "") or ""
    return TAG_RE.sub("", msg)

# Counter-based symptom extraction for tallies/graphs
SYMPTOM_LEXICON = [
    "Sadness", "Loss of interest", "Irritability", "Hopelessness", "Guilt", "Mood swings",
    "Worry", "Panic attacks", "Restlessness", "Muscle tension", "Avoidance", "Phobias", 
    "Poor concentration", "Memory problems", "Racing thoughts", "Indecisiveness", "Rumination",
    "Intrusive thoughts", "Hallucinations", "Delusions", "Disorganized thinking", "Derealization",
    "Depersonalization", "Social withdrawal", "Sleep changes", "Appetite changes", "Reckless behavior",
    "Aggression", "Self-harm", "Fatigue", "Headaches", "Gastrointestinal issues", "Palpitations",
    "Weight changes", "Substance misuse", "Obsessions", "Compulsions", "Impaired functioning", "Emotional numbness"
]
CANON = {s: s for s in SYMPTOM_LEXICON}
CANON.update({
    "sob": "shortness of breath",
    "dyspnea": "shortness of breath",
    "tiredness": "fatigue",
    "lightheadedness": "dizziness",
    "chest tightness": "chest pain",
    "loose stools": "diarrhea",
    "constipated": "constipation",
    "weightloss": "weight loss",
})

def extract_symptoms(text: str) -> Counter:
    t = " " + (text or "").lower() + " "
    counts = Counter()
    # phrase-first to catch multi-word entries
    for phrase in sorted(CANON.keys(), key=len, reverse=True):
        pattern = r'\b' + re.escape(phrase) + r'\b'
        hits = re.findall(pattern, t)
        if hits:
            counts[CANON[phrase]] += len(hits)
            # remove to avoid double-counting overlaps
            t = re.sub(pattern, " ", t)
    return counts

# Lazy FAISS loader for disease likelihoods
_faiss = None
def get_faiss():
    global _faiss
    if _faiss is None:
        idx_path  = current_app.config.get('FAISS_INDEX_PATH', 'mental_health_questions.index')
        meta_path = current_app.config.get('FAISS_METADATA_PATH', 'mental_health_questions_metadata.pkl')
        f = MentalHealthQuestionsFAISS()
        f.load_index(idx_path, meta_path)
        _faiss = f
    return _faiss

# --------------------------
# Overview stats
# --------------------------
@admin_bp.get("/api/summary")
@login_required
def summary():
    if not _require_admin():
        return admin_guard()

    db = SessionLocal()
    try:
        total_users = db.query(User).count()
        counselors = (
            db.query(User)
              .join(user_roles, user_roles.c.user_id == User.id)
              .join(Role, Role.id == user_roles.c.role_id)
              .filter(Role.name == "counselor")
              .count()
        )
        admins = (
            db.query(User)
              .join(user_roles, user_roles.c.user_id == User.id)
              .join(Role, Role.id == user_roles.c.role_id)
              .filter(Role.name == "admin")
              .count()
        )
        total_convos = db.query(Conversation).count()
        total_messages = db.query(Message).count()
        actor_msgs = db.query(Message).filter(Message.role == "patient" or Message.role == "counselor").count()
        rec_questions = db.query(Message).filter(Message.type == "question_recommender").count()

        # Conversations per day (last 30 rows by date asc)
        convs_per_day = (
            db.query(func.date(Conversation.created_at), func.count(Conversation.id))
              .group_by(func.date(Conversation.created_at))
              .order_by(func.date(Conversation.created_at))
              .limit(30)
              .all()
        )

        # Top counselors by # of owned conversations
        top_counselors = (
            db.query(User.email, func.count(ConversationOwner.conversation_id))
              .join(ConversationOwner, ConversationOwner.owner_user_id == User.id)
              .group_by(User.email)
              .order_by(desc(func.count(ConversationOwner.conversation_id)))
              .limit(10)
              .all()
        )

        # --- in admin.py -> summary()
        return jsonify({
            "ok": True,
            "users": {
                "total": total_users,
                "counselors": counselors,
                "clinicians": counselors,  # ✅ add alias for the UI
                "admins": admins
            },
            "conversations": {"total": total_convos},
            "messages": {
                "total": total_messages,
                "actor": actor_msgs,
                "recommended": rec_questions
            },
            "series": {
                "conversations_per_day": [[str(d), c] for d, c in convs_per_day],
                "top_counselors": [{"email": e, "count": c} for e, c in top_counselors],
            }
        })

    finally:
        db.close()

# --------------------------
# List counselors (with conversation counts)
# --------------------------
@admin_bp.get("/api/counselors")
@login_required
def counselors():
    if not _require_admin():
        return admin_guard()

    db = SessionLocal()
    try:
        rows = (
            db.query(User.id, User.email, func.count(ConversationOwner.conversation_id).label("convos"))
              .join(user_roles, user_roles.c.user_id == User.id)
              .join(Role, Role.id == user_roles.c.role_id)
              .filter(Role.name == "counselor")
              .outerjoin(ConversationOwner, ConversationOwner.owner_user_id == User.id)
              .group_by(User.id, User.email)
              .order_by(desc("convos"))
              .all()
        )
        return jsonify({"ok": True, "counselors": [
            {"id": i, "email": e, "conversations": c} for i, e, c in rows
        ]})
    finally:
        db.close()

# --------------------------
# Paginated conversations (includes owner email)
# --------------------------
# --- Paginated conversations (owner via conversations.owner_user_id) ---
@admin_bp.get("/api/conversations")
@login_required
def conversations():
    if not _require_admin():
        return admin_guard()

    page = int(request.args.get("page", 1))
    size = min(int(request.args.get("size", 20)), 100)
    offset = (page - 1) * size

    db = SessionLocal()
    try:
        total = db.query(Conversation).count()

        rows = (
            db.query(
                Conversation.id,
                Conversation.created_at,
                User.email,                 # may be None
                Conversation.owner_user_id  # may be None
            )
            .outerjoin(User, User.id == Conversation.owner_user_id)
            .order_by(Conversation.created_at.desc())
            .offset(offset).limit(size)
            .all()
        )

        convs = [{
            "id": cid,
            "created_at": created.isoformat(),
            # convenience field for legacy UI: prefer email, then ID, else None
            "owner": email or (str(owner_id) if owner_id is not None else None),
            "owner_email": email,
            "owner_user_id": owner_id,
        } for (cid, created, email, owner_id) in rows]

        return jsonify({"ok": True, "page": page, "size": size, "total": total, "conversations": convs})
    finally:
        db.close()


# --------------------------
# Conversation detail (messages + recommended questions)
# --------------------------
@admin_bp.get("/api/conversation/<cid>")
@login_required
def conversation_detail(cid):
    if not _require_admin():
        return admin_guard()

    db = SessionLocal()
    try:
        msgs = (
            db.query(Message)
              .filter(Message.conversation_id == cid)
              .order_by(Message.created_at.asc())
              .all()
        )

        out_msgs, recos = [], []
        for m in msgs:
            text = _safe_text(m)
            out_msgs.append({
                "id": m.id,
                "role": m.role,
                "type": m.type,
                "text": text,
                "timestamp": m.timestamp,
                "created_at": m.created_at.isoformat(),
            })
            if (m.type == "question_recommender") or (m.role == "Question Recommender"):
                recos.append({
                    "id": m.id,
                    "question": text,
                    "symptom": _extract_symptom(text)
                })

        return jsonify({"ok": True, "messages": out_msgs, "recommended_questions": recos})
    finally:
        db.close()

# --------------------------
# Symptom tallies (global + per-conversation)
# --------------------------
@admin_bp.get("/api/symptoms")
@login_required
def symptoms_api():
    if not _require_admin():
        return admin_guard()

    db = SessionLocal()
    try:
        # Pull all conversations + owners in one pass
        convo_rows = (
            db.query(
                Conversation.id,
                Conversation.created_at,
                User.email,
                Conversation.owner_user_id,
            )
            .outerjoin(User, User.id == Conversation.owner_user_id)
            .order_by(Conversation.created_at.desc())
            .all()
        )
        conv_ids = [cid for (cid, _created, _email, _uid) in convo_rows]

        owner_map = {
            cid: {
                "email": email,
                "id": uid,
                "created_at": created.isoformat(),
            }
            for (cid, created, email, uid) in convo_rows
        }

        # No conversations yet
        if not conv_ids:
            return jsonify({"ok": True, "global": {}, "by_conversation": []})

        # Only patient utterances for counting (be forgiving on casing)
        from sqlalchemy import or_
        msgs = (
            db.query(Message)
              .filter(Message.conversation_id.in_(conv_ids))
              .filter(or_(Message.role == "patient", Message.role == "Patient"))
              .order_by(Message.created_at.asc())
              .all()
        )

        from collections import Counter, defaultdict
        global_counts = Counter()
        per_conv = defaultdict(Counter)

        for m in msgs:
            counts = extract_symptoms(m.message or "")  # uses your helper defined above
            global_counts.update(counts)
            per_conv[m.conversation_id].update(counts)

        by_conv = []
        for cid in conv_ids:
            meta = owner_map.get(cid, {})
            owner_email = meta.get("email")
            owner_id = meta.get("id")
            by_conv.append({
                "conversation_id": cid,
                # convenience + explicit fields
                "owner": owner_email or (str(owner_id) if owner_id is not None else ""),
                "owner_email": owner_email,
                "owner_user_id": owner_id,
                "created_at": meta.get("created_at"),
                "symptoms": dict(per_conv[cid].most_common()),
            })

        return jsonify({
            "ok": True,
            "global": dict(global_counts.most_common()),
            "by_conversation": by_conv
        })
    finally:
        db.close()


# --------------------------
# Disease likelihoods per conversation (FAISS-weighted)
# --------------------------
# --------------------------
# Disease likelihoods per conversation (FAISS-weighted, questions index)
# --------------------------
@admin_bp.get("/api/conversation/<cid>/disease_likelihoods")
@login_required
def conversation_disease_likelihoods(cid):
    if not _require_admin():
        return admin_guard()

    db = SessionLocal()
    try:
        msgs = (
            db.query(Message)
              .filter(Message.conversation_id == cid)
              .order_by(Message.created_at.asc())
              .all()
        )
        if not msgs:
            return jsonify({"ok": False, "error": "No messages for conversation"}), 404

        # Prefer patient-only text; fall back to all text
        patient_text = " ".join((m.message or "") for m in msgs if (m.role or "").lower() == "patient").strip()
        if not patient_text:
            patient_text = " ".join((m.message or "") for m in msgs if m.message).strip()

        f = get_faiss()  # MentalHealthQuestionsFAISS instance

        # If a cases index exists, use it; otherwise use questions index
        if hasattr(f, "search_similar_cases"):
            results = f.search_similar_cases(patient_text, k=8, similarity_threshold=0.05)
            # Expecting results already bucketed—wrap to current response shape
            ranked = sorted(
                ({"disease": r["label"], "weight": float(r["score"])} for r in results),
                key=lambda x: (-x["weight"], x["disease"])
            )
            total = sum(r["weight"] for r in ranked) or 1.0
            top_diseases = [
                {**r, "likelihood_pct": round(100.0 * r["weight"] / total, 1)}
                for r in ranked[:5]
            ]
            faiss_matches = []  # cases search doesn't expose question rows
        else:
            # Questions FAISS flow
            qhits = f.search(patient_text, k=20, threshold=0.05)

            from collections import defaultdict
            weights = defaultdict(float)
            labels = ("depression", "anxiety", "psychosis")
            for r in qhits:
                sim = max(float(r.similarity_score), 0.0)
                cat = (r.category or "").strip().lower()
                tags = [t.strip().lower() for t in (r.tags or [])]
                for lab in labels:
                    if cat == lab or lab in tags:
                        weights[lab] += sim

            # Light heuristic backoff if nothing matched
            if not weights:
                lt = patient_text.lower()
                if any(w in lt for w in ("sad", "hopeless", "phq", "anhedonia", "self-harm", "suicid")):
                    weights["depression"] += 0.4
                if any(w in lt for w in ("worry", "panic", "gad", "tension", "restless", "phobia")):
                    weights["anxiety"] += 0.4
                if any(w in lt for w in ("hallucinat", "delusion", "voices", "thought disorder", "psychosis")):
                    weights["psychosis"] += 0.4

            total = sum(weights.values()) or 1.0
            top_diseases = sorted(
                ({"disease": k, "weight": v, "likelihood_pct": round(100.0 * v / total, 1)} for k, v in weights.items()),
                key=lambda x: (-x["weight"], x["disease"])
            )[:5]

            faiss_matches = [{
                "question_id": r.question_id,
                "question": r.question,         # {"english", "swahili"}
                "category": r.category,
                "tags": r.tags,
                "similarity": round(float(r.similarity_score), 4),
            } for r in qhits]

        # Symptom extraction (your project’s helper)
        sym = extract_symptoms(patient_text)

        return jsonify({
            "ok": True,
            "conversation_id": cid,
            "symptoms": dict(sym.most_common()),
            "top_diseases": top_diseases,
            "faiss_matches": faiss_matches
        })
    finally:
        db.close()

