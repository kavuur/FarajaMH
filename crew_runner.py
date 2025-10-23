# crew_runner.py
from crewai import Crew, Task
from agent_loader import load_llm, load_agents_from_yaml, load_tasks_from_yaml
from datetime import datetime
import json
import re
import os
import logging
from time import time

# Per-session throttling & dedupe
_SUGGEST_STATE = {}  # session_id -> {"last_chars": int, "last_ts": float, "seen": set[str]}


logger = logging.getLogger(__name__)

# ---------- FAISS bootstrap (robust to either index type) ----------
faiss_system = None
_has_case_search = False

# Try CASES index first (MedicalCaseFAISS). If not available, try QUESTIONS index (MentalHealthQuestionsFAISS).
try:
    from medical_case_faiss import MedicalCaseFAISS as _CasesFAISS
    _cases = _CasesFAISS()
    # Try common filename pairs
    for idx, meta in [
        ("medical_cases.index", "medical_cases_metadata.pkl"),
        ("mental_health_cases.index", "mental_health_cases_metadata.pkl"),
    ]:
        if os.path.exists(idx) and os.path.exists(meta):
            _cases.load_index(idx, meta)
            faiss_system = _cases
            _has_case_search = hasattr(faiss_system, "search_similar_cases")
            logger.info(f"[crew_runner] Loaded CASES FAISS: {idx} / {meta}")
            break
except Exception:
    logger.exception("[crew_runner] CASES FAISS init failed; will try QUESTIONS FAISS")

# If we didn't get a cases index, fall back to QUESTIONS-only FAISS (no retrieval used here)
if faiss_system is None:
    try:
        from mental_health_faiss import MentalHealthQuestionsFAISS as _QFAISS
        _q = _QFAISS()
        # Try common questions filenames
        for idx, meta in [
            ("mental_q.index", "mental_q_meta.pkl"),
            ("questions.index", "questions_meta.pkl"),
        ]:
            if os.path.exists(idx) and os.path.exists(meta):
                _q.load_index(idx, meta)
                faiss_system = _q
                _has_case_search = False  # questions-only class has no search_similar_cases
                logger.info(f"[crew_runner] Loaded QUESTIONS FAISS: {idx} / {meta}")
                break
    except Exception:
        logger.exception("[crew_runner] QUESTIONS FAISS init failed; continuing without FAISS")

AGENT_PATH = "config/agents.yaml"
TASK_PATH = "config/tasks.yaml"


def run_task(agent, input_text, name="Step"):
    crew = Crew(
        agents=[agent],
        tasks=[Task(
            name=name,
            description=input_text,
            expected_output="Give your response as if you were in the middle of the diagnostic session.",
            agent=agent
        )],
        verbose=False
    )
    result = crew.kickoff()
    return result if isinstance(result, str) else getattr(result, "final_output", str(result))


def recommend_question_bilingual(context_text, faiss_system=None, threshold=0.25, k=5):
    # Try FAISS first (if available and has suggest_questions)
    if faiss_system and hasattr(faiss_system, "suggest_questions"):
        try:
            faiss_suggestions = faiss_system.suggest_questions(
                context_text, k=k, max_questions=1, similarity_threshold=threshold
            ) or []
            if faiss_suggestions:
                q = faiss_suggestions[0]["question"]
                # Return bilingual fields if present; fall back to a single string
                en = (q.get("english") if isinstance(q, dict) else str(q)) or ""
                sw = (q.get("swahili") if isinstance(q, dict) else "") or ""
                return en, sw
        except Exception:
            pass  # fall through to LLM

    # Fallback to LLM (existing logic condensed)
    return None, None  # signal caller to use LLM path


# ---------- SSE helpers ----------
def sse_message(role, message, log_hook=None, session_id=None):
    ts = datetime.now().strftime("%H:%M:%S")
    payload = {"role": role, "message": (message or "").strip(), "timestamp": ts}
    if log_hook:
        log_hook(session_id, role, payload["message"], ts, "message")
    return "data: " + json.dumps(payload) + "\n\n"


def sse_recommender(english, swahili, log_hook=None, session_id=None):
    ts = datetime.now().strftime("%H:%M:%S")
    payload = {
        "type": "question_recommender",
        "question": {"english": (english or "").strip(), "swahili": (swahili or "").strip()},
        "timestamp": ts,
    }
    if log_hook:
        msg = f"Recommended Q | EN: {payload['question']['english']} | SW: {payload['question']['swahili']}"
        log_hook(session_id, "Question Recommender", msg, ts, "question_recommender")
    return "data: " + json.dumps(payload) + "\n\n"


def _case_snippet(r):
    """Build a short one-liner from a MedicalCaseFAISS result."""
    try:
        cc = (r.chief_complaint or {}).get("english") if isinstance(r.chief_complaint, dict) else None
        os_ = (r.opening_statement or {}).get("english") if isinstance(r.opening_statement, dict) else None
        bg = (r.patient_background or {}).get("english") if isinstance(r.patient_background, dict) else None
        txt = (cc or os_ or bg or "")[:120]
        return (txt + "…") if (cc or os_ or bg) and len((cc or os_ or bg)) > 120 else txt
    except Exception:
        return ""


# ---------- Mode 1: Fully simulated ----------
def simulate_agent_chat_stepwise(initial_message: str, turns: int = 6, language_mode: str = "bilingual", log_hook=None, session_id=None):
    llm = load_llm()
    agents = load_agents_from_yaml(AGENT_PATH, llm)

    # Always emit the patient's seed
    yield sse_message("Patient", initial_message, log_hook, session_id)

    # Retrieval context (only if we have a CASES index with search_similar_cases)
    similar_bullets = ""
    if faiss_system and _has_case_search:
        try:
            similar_cases = faiss_system.search_similar_cases(initial_message, k=5, similarity_threshold=0.19) or []
            similar_bullets = "\n".join(
                f"- {getattr(r, 'case_id', 'case')}: {_case_snippet(r)} (sim={getattr(r, 'similarity_score', 0):.2f})"
                for r in similar_cases if r
            )
        except Exception:
            logger.exception("FAISS search failed during simulated mode; continuing without retrieval")

    context_log = [f"Patient says: {initial_message}"]
    if similar_bullets:
        context_log.append("Similar cases (context):\n" + similar_bullets)

    for turn in range(turns):
        # question recommender
        if language_mode == "english":
            recommender_input = "\n".join(context_log) + "\n\nSuggest the next most relevant diagnostic question. Format: English: ..."
        elif language_mode == "swahili":
            recommender_input = "\n".join(context_log) + "\n\nPendekeza swali fupi la uchunguzi linalofuata. Format: Swahili: ..."
        else:
            recommender_input = "\n".join(context_log) + "\n\nSuggest the next most relevant bilingual question only. Format as:\nEnglish: ...\n\nSwahili: ..."

        recommended = run_task(agents["question_recommender_agent"], recommender_input, f"Question Suggestion {turn+1}")

        if language_mode == "english":
            english_q, swahili_q = recommended.strip(), ""
        elif language_mode == "swahili":
            english_q, swahili_q = "", recommended.strip()
        else:
            match = re.search(r"English:\s*(.+?)\n+Swahili:\s*(.+)", recommended, re.DOTALL)
            if match:
                english_q, swahili_q = match.group(1).strip(), match.group(2).strip()
            else:
                english_q, swahili_q = recommended.strip(), ""

        plain_q = english_q if language_mode == "english" else (swahili_q if language_mode == "swahili" else f"{english_q}\n\n{swahili_q}")

        yield sse_recommender(english_q, swahili_q, log_hook, session_id)
        yield sse_message("Clinician", plain_q, log_hook, session_id)
        context_log.append(f"Clinician: {plain_q}")

        # simulated patient reply
        if language_mode == "english":
            patient_input = f"Clinician: {english_q}\n\nRespond in English as the patient. Be short and realistic."
        elif language_mode == "swahili":
            patient_input = f"Clinician: {swahili_q}\n\nJibu kwa Kiswahili kama mgonjwa. Toa jibu fupi na halisi."
        else:
            patient_input = f"Clinician: English: {english_q} Swahili: {swahili_q}\n\nRespond as the patient. Answer both languages if possible. Be short and realistic."

        patient_response = run_task(agents["patient_agent"], patient_input, f"Patient Response {turn+1}")
        yield sse_message("Patient", patient_response, log_hook, session_id)
        context_log.append(f"Patient: {patient_response}")

    # Finalize
    listener_input = "\n".join(context_log) + "\n\nSummarize the conversation in two parts:\n**English Summary:**\n- ...\n**Swahili Summary:**\n- ..."
    listener_summary = run_task(agents["listener_agent"], listener_input, "Listener Summary")
    yield sse_message("Listener", listener_summary, log_hook, session_id)

    final_input = listener_input + "\n\nProvide a FINAL PLAN clearly structured as bullet points. Format like:\n**FINAL PLAN:**\n- Step 1: ...\n- Step 2: ..."
    final_plan = run_task(agents["clinician_agent"], final_input, "Final Plan")
    yield sse_message("Clinician", f"**FINAL PLAN:**\n\n{final_plan}", log_hook, session_id)


# ---------- Mode 2: Real actors ----------
def real_actor_chat_stepwise(
    initial_message: str,
    language_mode: str = "bilingual",
    speaker_role: str = "Patient",
    conversation_history: list | None = None,
    log_hook=None,
    session_id=None
):
    llm = load_llm()
    agents = load_agents_from_yaml(AGENT_PATH, llm)
    history = conversation_history or []

    if speaker_role.lower() == "finalize":
        transcript_lines = [f"{m.get('role','')}: {m.get('message','')}" for m in history]
        convo_text = "\n".join(transcript_lines)

        listener_input = convo_text + "\n\nSummarize the conversation in two parts:\n**English Summary:**\n- ...\n**Swahili Summary:**\n- ..."
        listener_summary = run_task(agents["listener_agent"], listener_input, "Listener Summary")
        yield sse_message("Listener", listener_summary, log_hook, session_id)

        final_input = listener_input + "\n\nProvide a FINAL PLAN clearly structured as bullet points. Format like:\n**FINAL PLAN:**\n- Step 1: ...\n- Step 2: ..."
        final_plan = run_task(agents["clinician_agent"], final_input, "Final Plan")
        yield sse_message("Clinician", f"**FINAL PLAN**\n\n{final_plan}", log_hook, session_id)
        return

    # echo the role-tagged line
    yield sse_message(speaker_role, initial_message, log_hook, session_id)

    # Only patient turns trigger recommendations in real-actors mode
    if speaker_role.lower() == "patient":
        context_lines = [f"{m.get('role')}: {m.get('message')}" for m in history]
        context_text = "\n".join(context_lines)

        if language_mode == "english":
            recommender_input = context_text + "\n\nSuggest the next most relevant diagnostic question. Format: English: ..."
        elif language_mode == "swahili":
            recommender_input = context_text + "\n\nPendekeza swali fupi la uchunguzi linalofuata. Format: Swahili: ..."
        else:
            recommender_input = context_text + "\n\nSuggest the next most relevant bilingual question only. Format as:\nEnglish: ...\n\nSwahili: ..."

        # FAISS first, LLM fallback
        en, sw = recommend_question_bilingual(context_text, faiss_system)
        if en is None and sw is None:
            rec = run_task(agents["question_recommender_agent"], recommender_input, "Question Suggestion")
            if language_mode == "english":
                english_q, swahili_q = rec.strip(), ""
            elif language_mode == "swahili":
                english_q, swahili_q = "", rec.strip()
            else:
                match = re.search(r"English:\s*(.+?)\n+Swahili:\s*(.+)", rec, re.DOTALL)
                english_q, swahili_q = (match.group(1).strip(), match.group(2).strip()) if match else (rec.strip(), "")
        else:
            english_q, swahili_q = en.strip(), (sw or "").strip()

        yield sse_recommender(english_q, swahili_q, log_hook, session_id)



# ---------- Mode 3: Live transcription (final-driven) ----------
# ---------- Mode 3: Live transcription (final-driven) ----------
def live_transcription_stream(
    initial_message: str,
    language_mode: str = "bilingual",
    speaker_role: str = "live",
    suggest_mode: str = "stream",  # "stream" | "final"
    conversation_history: list | None = None,
    log_hook=None,
    session_id=None,
):
    """
    Live transcription (final-driven):
      • Always emits the transcript line.
      • If suggest_mode == "stream": MAY emit one question (gated: substance, cues, cooldown, dedupe).
      • If suggest_mode == "final": never suggest mid-chat; buffer transcript and on speaker_role=="finalize"
        emit MULTIPLE tailored follow-up questions derived from the conversation (not a fixed list).
    """
    from time import time

    # ---------- tiny per-session state ----------
    def _state_for(sid: str):
        return _SUGGEST_STATE.setdefault(
            sid or "default",
            {"last_chars": 0, "last_ts": 0.0, "seen": set(), "buffer": []},
        )

    # ---------- helpers (light, local) ----------
    def _is_backchannel(t: str) -> bool:
        t = (t or "").strip().lower()
        if len(t) < 25:
            return True
        bc = (
            "okay", "ok", "sure", "mm", "hmm", "uh huh", "yes", "no",
            "go on", "continue", "carry on", "ndio", "sawa", "poa", "yeah", "yep"
        )
        return any(re.fullmatch(rf"(?:{w})[.!?]?", t) for w in bc)

    def _live_chars(hist: list) -> int:
        total = 0
        for m in hist or []:
            role = (m.get("role") or "").lower()
            if role in ("live", "transcript"):
                total += len(m.get("message") or "")
        return total

    def _has_meaningful_cue(text: str) -> bool:
        t = (text or "").lower()
        cue_any = [
            # safety / risk
            "self-harm", "suicide", "suicidal", "harm myself", "hurt myself",
            "kill myself", "ending my life", "harming others",
            # mood
            "hopeless", "hopelessness", "worthless", "worthlessness",
            "anxious", "anxiety", "panic", "worry", "tearful", "sad",
            # depression signs
            "lost interest", "no interest", "anhedonia",
            "sleep", "insomnia", "appetite", "tired", "fatigue",
            # somatic/psychotic cues
            "headache", "hallucination", "hearing voices", "paranoid",
        ]
        neg_noise = [
            "i did with my pasta", "some good", "yeah yeah", "about yeah",
            "i hope you are doing well"
        ]
        if any(n in t for n in neg_noise):
            return False
        return any(c in t for c in cue_any)

    def _canon_q(q: str) -> str:
        q = (q or "").strip()
        q = re.sub(r"^\s*(English:)\s*(English:)\s*", r"\1 ", q, flags=re.I)  # fix double prefix
        q = re.sub(r"^\s*(english:|swahili:)\s*", "", q, flags=re.I)
        q = re.sub(r"\s+", " ", q).strip(" ?.!").lower()
        return q

    def _pick_one_question(candidate_blob: str) -> tuple[str, str]:
        lines = [l.strip(" -•\t") for l in (candidate_blob or "").splitlines() if l.strip()]
        if not lines and candidate_blob:
            lines = [candidate_blob.strip()]
        eng, swa = "", ""
        m = re.search(r"English:\s*(.+?)(?:\n|$)", candidate_blob, flags=re.I)
        if m: eng = m.group(1).strip()
        m = re.search(r"Swahili:\s*(.+?)(?:\n|$)", candidate_blob, flags=re.I)
        if m: swa = m.group(1).strip()
        if not eng:
            for l in lines:
                if len(l) > 8:
                    eng = re.sub(r"^\s*(English:)\s*", "", l, flags=re.I)
                    break
        return eng, swa

    # -------- dynamic end-of-session question generation ----------
    # Uses screening.SYMPTOMS if available to steer the agent to depression/anxiety/psychosis signals
    try:
        from screening import SYMPTOMS  # {condition: {feature: [keywords...]}}
    except Exception:
        SYMPTOMS = {
            "depression": {"hopelessness": ["hopeless", "worthless", "lost interest", "tearful"]},
            "anxiety":    {"worry": ["worry", "anxious", "panic", "restless"]},
            "psychosis":  {"hallucinations": ["hearing voices", "seeing things", "paranoid"]},
        }

    def _salient_conditions(context_text: str) -> list[str]:
        t = (context_text or "").lower()
        scores = {"depression": 0, "anxiety": 0, "psychosis": 0}
        for cond, feats in SYMPTOMS.items():
            for _, kws in feats.items():
                for k in kws:
                    if k.lower() in t:
                        scores[cond] += 1
        # order by score desc
        return [k for k, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]

    def _ask_agent_for_multi(context_text: str, lang_mode: str, focus_conditions: list[str]) -> str:
        """
        Get a *list* of tailored questions from the question_recommender_agent.
        Output contract (strict): each question on its own line, formatted as:
          English: <short question> | Swahili: <short question>
        If Swahili is unavailable, still include the 'Swahili:' label with an empty value.
        Return raw text blob; we'll parse/clean/dedupe after.
        """
        conditions_hint = ", ".join([c.capitalize() for c in focus_conditions if c])
        if not conditions_hint:
            conditions_hint = "Depression, Anxiety, Psychosis"

        instructions = (
            "You are assisting a clinician at session end. Based on the conversation below, "
            "produce a concise, ranked LIST of tailored diagnostic follow-up questions that "
            "advance assessment of the most salient issues (focus on: "
            f"{conditions_hint}).\n"
            "- Keep them short, specific, and non-repetitive.\n"
            "- Cover risk/safety first if present; then mood/anxiety; then psychosis if relevant.\n"
            "- Output 4 to 7 items MAX.\n"
            "- STRICT FORMAT: one item per line as:\n"
            "  English: <question> | Swahili: <question>\n"
            "- Do NOT include numbering or bullets.\n"
            "- Do NOT repeat identical questions with minor wording changes.\n"
        )

        if lang_mode == "english":
            instructions += "- If Swahili is not requested, still include an empty Swahili field.\n"
        elif lang_mode == "swahili":
            instructions += "- You may leave English empty if needed but keep both fields present.\n"

        prompt = (
            f"{instructions}\n\n=== Conversation Context ===\n{context_text}\n=== End Context ===\n"
        )

        # Use your existing agent plumbing
        llm = load_llm()
        agents = load_agents_from_yaml(AGENT_PATH, llm)
        return run_task(agents["question_recommender_agent"], prompt, "End-of-Session Questions") or ""

    def _parse_multi(blob: str) -> list[tuple[str, str]]:
        """
        Parse lines of 'English: ... | Swahili: ...' into list of (en, sw).
        Tolerant to missing swahili part.
        """
        items = []
        for raw in (blob or "").splitlines():
            line = raw.strip(" -•\t")
            if not line:
                continue
            # try strict split
            m = re.match(r"(?i)english:\s*(.*?)\s*\|\s*swahili:\s*(.*)$", line, flags=re.I)
            if m:
                en, sw = m.group(1).strip(), m.group(2).strip()
            else:
                # try separate labels on the same line
                en = ""
                sw = ""
                me = re.search(r"(?i)english:\s*(.+?)($|\||$)", line)
                ms = re.search(r"(?i)swahili:\s*(.+?)$", line)
                if me: en = me.group(1).strip(" |")
                if ms: sw = ms.group(1).strip()
                if not en and not sw:
                    # fallback: treat the whole line as English
                    en = line
            items.append((en, sw))
        return items

    # ---------- setup ----------
    llm = load_llm()
    agents = load_agents_from_yaml(AGENT_PATH, llm)

    history = conversation_history or []
    final_text = (initial_message or "").strip()
    if not final_text:
        return

    sid = session_id or "default"
    state = _state_for(sid)

    # Always emit the transcript line
    yield sse_message("Transcript", final_text, log_hook, sid)

    # -------- end-mode path: buffer & finalize with MULTI tailored questions --------
    if suggest_mode != "stream":
        # keep buffering normal lines; do not suggest mid-chat
        if speaker_role != "finalize":
            state["buffer"].append(final_text)
            return

        # On finalize, compile context
        context_lines = [f"{m.get('role','')}: {m.get('message','')}" for m in history]
        if state["buffer"]:
            context_lines.append("Transcript:\n" + "\n".join(state["buffer"]))
        if final_text and final_text != "[Finalize]":
            context_lines.append(f"Transcript: {final_text}")
        context_text = "\n".join(context_lines)

        # Figure salient conditions from transcript to *steer* the agent
        focus = _salient_conditions(context_text)

        # Ask the agent for a short ranked list (dynamic, conversation-aware)
        raw_list = _ask_agent_for_multi(context_text, language_mode, focus)
        pairs = _parse_multi(raw_list)

        # Deduplicate vs session and vs anything already asked by assistant/clinician
        already_asked = set()
        for m in history:
            role = (m.get("role") or "").lower()
            if role in ("assistant", "recommender", "clinician"):
                already_asked.add(_canon_q(m.get("message") or ""))

        emitted = 0
        MAX_EMIT = 7  # hard cap
        for en, sw in pairs:
            # Normalize and check
            can_en = _canon_q(en)
            can_sw = _canon_q(sw)
            if can_en and can_en in state["seen"]:
                continue
            if can_en and can_en in already_asked:
                continue
            if can_sw and can_sw in state["seen"]:
                continue
            if can_sw and can_sw in already_asked:
                continue

            # Prefer bilingual when available, else whichever exists
            out_en = (en or "").strip()
            out_sw = (sw or "").strip()

            # Avoid empty-empties
            if not out_en and not out_sw:
                continue

            # Mark seen for dedupe
            if can_en: state["seen"].add(can_en)
            if can_sw: state["seen"].add(can_sw)

            # Emit
            yield sse_recommender(out_en, out_sw, log_hook, sid)
            emitted += 1
            if emitted >= MAX_EMIT:
                break

        # Clear buffer for a fresh session
        state["buffer"].clear()
        return

    # -------- streaming suggestion path (gated, one-at-a-time) --------
    # Build context (history + latest line)
    context_lines = [f"{m.get('role','')}: {m.get('message','')}" for m in history]
    context_lines.append(f"Transcript: {final_text}")
    context_text = "\n".join(context_lines)

    # 1) Per-line substance and cue check
    if len(final_text) < 40 or _is_backchannel(final_text):
        return
    if not _has_meaningful_cue(final_text):
        return

    # 2) Total context threshold
    total_chars = _live_chars(history)
    if total_chars < 200:
        return

    # 3) Cooldown + new content since last suggestion
    now = time()
    MIN_COOLDOWN_SEC = 18
    MIN_NEW_CHARS = 160
    if (now - state["last_ts"]) < MIN_COOLDOWN_SEC:
        return
    if (total_chars - state["last_chars"]) < MIN_NEW_CHARS:
        return

    # 4) Avoid re-asking what assistant/clinician already asked
    already_asked = set()
    for m in history:
        role = (m.get("role") or "").lower()
        if role in ("assistant", "recommender", "clinician"):
            already_asked.add(_canon_q(m.get("message") or ""))

    # ---- compute a single question suggestion (FAISS first, LLM fallback) ----
    en_faiss, sw_faiss = recommend_question_bilingual(context_text, faiss_system)

    if en_faiss or sw_faiss:
        english_q, swahili_q = (en_faiss or "").strip(), (sw_faiss or "").strip()
    else:
        if language_mode == "english":
            prompt = context_text + "\n\nSuggest exactly ONE next diagnostic question. Format: English: ..."
        elif language_mode == "swahili":
            prompt = context_text + "\n\nPendekeza SWALI MOJA tu la uchunguzi linalofuata. Format: Swahili: ..."
        else:
            prompt = (
                context_text
                + "\n\nSuggest exactly ONE next diagnostic question in bilingual format:\n"
                  "English: ...\nSwahili: ..."
            )
        rec = run_task(load_agents_from_yaml(AGENT_PATH, load_llm())["question_recommender_agent"],
                       prompt, "Question Suggestion")
        english_q, swahili_q = _pick_one_question(rec or "")

    can_eng = _canon_q(english_q)
    can_swa = _canon_q(swahili_q)
    seen = state["seen"]

    out_eng, out_swa = "", ""
    if can_eng and can_eng not in seen and can_eng not in already_asked:
        out_eng = english_q.strip()
        seen.add(can_eng)
    elif can_swa and can_swa not in seen and can_swa not in already_asked:
        out_swa = swahili_q.strip()
        seen.add(can_swa)
    else:
        return

    state["last_ts"] = now
    state["last_chars"] = total_chars

    if out_eng:
        out_eng = re.sub(r"^\s*(English:)\s*(English:)\s*", r"\1 ", out_eng, flags=re.I)

    yield sse_recommender(out_eng, out_swa, log_hook, sid)
    return

