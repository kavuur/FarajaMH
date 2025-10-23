# app.py — keep live STT & crew flows; add question-bank + safer search fallback

from __future__ import annotations

import os
import re
import io
import csv
import json
import time
import uuid
import wave
import shutil
import logging
import tempfile
import threading
import queue
import subprocess
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, Dict

import numpy as np
import torch
import webrtcvad
from flask import (
    Flask, render_template, request, jsonify, session, Response, stream_with_context,
    send_file
)
from flask_login import current_user, login_required
from flask_wtf import CSRFProtect
from flask_wtf.csrf import generate_csrf
from flask_sock import Sock
from dotenv import load_dotenv

# Engines
from faster_whisper import WhisperModel
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor

# Project modules
from config import Config
from mental_health_faiss import MentalHealthQuestionsFAISS
from crew_runner import (
    simulate_agent_chat_stepwise,
    real_actor_chat_stepwise,
    live_transcription_stream,
)
from models import init_db, create_conversation, log_message, SessionLocal, Message
from screening import run_screening, screening_to_dict, set_faiss_instance  # ← add set_faiss_instance

# Optional blueprints
from auth import auth_bp, login_manager
from admin import admin_bp

# -------------------
# ENV / CONFIG
# -------------------
load_dotenv()

JACARANDA_MODEL_ID  = os.getenv("JACARANDA_MODEL_ID", "Jacaranda-Health/ASR-STT").strip()
JACARANDA_MODEL_DIR = os.getenv("JACARANDA_MODEL_DIR", "").strip()

WHISPER_MODEL_SIZE   = os.getenv("WHISPER_MODEL_SIZE", "small").strip()
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8_float16").strip()

SAMPLE_RATE             = int(os.getenv("SAMPLE_RATE", "16000"))
FFMPEG_BIN              = os.getenv("FFMPEG_BIN", r"C:\\ffmpeg\\ffmpeg-7.1.1-full_build\\bin\\ffmpeg.exe" if os.name == "nt" else "ffmpeg")
VAD_AGGR                = int(os.getenv("STT_VAD_AGGRESSIVENESS", "3"))
VAD_FRAME_MS            = int(os.getenv("VAD_FRAME_MS", "30"))
VAD_RATIO_MIN           = float(os.getenv("VAD_VOICED_RATIO_MIN", "0.65"))

EMIT_PARTIALS           = (os.getenv("EMIT_PARTIALS", "false").lower() == "true")
PARTIAL_MIN_INTERVAL_MS = int(os.getenv("STT_PARTIAL_MIN_INTERVAL_MS", "600"))
PARTIAL_WINDOW_SEC      = float(os.getenv("PARTIAL_WINDOW_SEC", "2.2"))
SEGMENT_SILENCE_MS      = int(os.getenv("STT_SEGMENT_SILENCE_MS", "1200"))
MAX_SEGMENT_SEC         = float(os.getenv("STT_MAX_SEGMENT_SEC", "10.0"))
RMS_MIN                 = float(os.getenv("STT_RMS_MIN", "250.0"))

os.environ['CREWAI_TELEMETRY_DISABLED'] = '1'

# -------------------
# APP + Security
# -------------------
app = Flask(__name__)
app.config.from_object(Config)

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-change-me")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = not app.debug

login_manager.init_app(app)
csrf = CSRFProtect(app)
app.register_blueprint(auth_bp)
app.register_blueprint(admin_bp)
sock = Sock(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------
# Utilities
# -------------------
@contextmanager
def tempenv(env: Dict[str, Optional[str]]):
    old: Dict[str, Optional[str]] = {k: os.getenv(k) for k in env}
    try:
        for k, v in env.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v

def _debabble(s: str) -> str:
    if not s: return s
    s = re.sub(r'\b(\w{1,3})(?:\s+\1){4,}\b', r'\1 \1', s, flags=re.IGNORECASE)
    s = re.sub(r'\b(\w+)(?:\s+\1){2,}\b', r'\1 \1', s, flags=re.IGNORECASE)
    return s.strip()

def _squash_runs(s: str) -> str:
    if not s: return s
    return re.sub(r'(\b(?:\w+[\s,;:.!?-]+){1,6})\1{2,}', r'\1\1', s, flags=re.IGNORECASE)

def _clean_text(s: str) -> str:
    if not s: return ''
    s = _debabble(_squash_runs(s.replace('\uFFFd','').strip()))
    bad = ['nigga','nigger']
    if any(b in s.lower() for b in bad): return ''
    return s

def vad_voiced_ratio(pcm_bytes: bytes, sr: int, frame_ms: int = 30, aggressiveness: int = 3) -> float:
    try:
        vad = webrtcvad.Vad(int(aggressiveness))
        frame_len = int(sr * (frame_ms / 1000.0)) * 2
        if frame_len <= 0 or len(pcm_bytes) < frame_len: return 0.0
        voiced, total = 0, 0
        for i in range(0, len(pcm_bytes) - frame_len + 1, frame_len):
            chunk = pcm_bytes[i:i+frame_len]; total += 1
            if vad.is_speech(chunk, sr): voiced += 1
        return voiced / max(total, 1)
    except Exception:
        return 1.0

def rms_level(pcm_bytes: bytes) -> float:
    if not pcm_bytes: return 0.0
    arr = np.frombuffer(pcm_bytes, dtype=np.int16)
    if arr.size == 0: return 0.0
    return float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))

def _bytes_to_temp_wav(pcm_bytes: bytes, sr: int) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav_path = tf.name
    with wave.open(wav_path, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(pcm_bytes)
    return wav_path

# -------------------
# Engines
# -------------------
class FasterWhisperTranscriber:
    _model = None
    _lock = threading.Lock()

    @classmethod
    def _get_model(cls):
        if cls._model is not None: return cls._model
        with cls._lock:
            if cls._model is None:
                cls._model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type=WHISPER_COMPUTE_TYPE)
        return cls._model

    @classmethod
    def transcribe_wav(cls, wav_path: str, lang: str | None = None, initial_prompt: str | None = None) -> str:
        model = cls._get_model()
        segments, _ = model.transcribe(
            wav_path,
            language=(lang if lang in ("en","sw") else None),
            task="transcribe",
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 600},
            temperature=0.0, best_of=1, beam_size=1,
            no_speech_threshold=0.7, compression_ratio_threshold=2.4,
            without_timestamps=True,
            initial_prompt=initial_prompt
        )
        return " ".join(s.text.strip() for s in segments if s.text).strip()

class JacarandaTranscriber:
    _pipe = None
    _lock = threading.Lock()

    @classmethod
    def _resolve(cls) -> str:
        if JACARANDA_MODEL_DIR and os.path.isdir(JACARANDA_MODEL_DIR):
            return os.path.abspath(JACARANDA_MODEL_DIR)
        return JACARANDA_MODEL_ID

    @classmethod
    def get_pipeline(cls):
        if cls._pipe is not None: return cls._pipe
        with cls._lock:
            if cls._pipe is not None: return cls._pipe
            model_ref = cls._resolve()
            use_cuda = (os.getenv("USE_CUDA", "0") == "1") and torch.cuda.is_available()
            device = 0 if use_cuda else -1
            dtype = torch.float16 if use_cuda else torch.float32

            offline_env = {
                "HF_HUB_OFFLINE": "1" if os.path.isdir(model_ref) else None,
                "TRANSFORMERS_OFFLINE": "1" if os.path.isdir(model_ref) else None,
                "HF_HUB_ENABLE_XET": "0",
                "HF_HUB_DISABLE_TELEMETRY": "1",
            }
            with tempenv(offline_env):
                processor = AutoProcessor.from_pretrained(model_ref)
                model = AutoModelForSpeechSeq2Seq.from_pretrained(model_ref, dtype=dtype)
                try:
                    model.generation_config.forced_decoder_ids = None
                except Exception:
                    pass
                cls._pipe = pipeline(
                    task="automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    device=device,
                    ignore_warning=True
                )
        return cls._pipe

    @classmethod
    def transcribe_wav(cls, wav_path: str) -> str:
        pipe = cls.get_pipeline()
        out = pipe(wav_path, return_timestamps=False, generate_kwargs={"task": "transcribe"})
        return (out.get("text") if isinstance(out, dict) else out or "").strip()

# -------------------
# FAISS init
# -------------------
faiss_system: Optional[MentalHealthQuestionsFAISS] = None

def initialize_faiss() -> bool:
    """
    Loads a Questions-FAISS index (bilingual questions).
    If you later add a Cases-FAISS with `search_similar_cases`, /search will detect it.
    """
    global faiss_system
    try:
        fs = MentalHealthQuestionsFAISS()
        index_path = app.config.get('FAISS_INDEX_PATH')
        meta_path  = app.config.get('FAISS_METADATA_PATH')
        if index_path and meta_path and os.path.exists(index_path) and os.path.exists(meta_path):
            logger.info("Loading FAISS questions index…")
            fs.load_index(index_path, meta_path)
            faiss_system = fs
            # Hand the same FAISS to the screening module for label weights
            try:
                set_faiss_instance(fs)
            except Exception:
                logger.exception("Failed to inject FAISS into screening")
            logger.info("FAISS loaded OK")
            return True
        else:
            logger.error("FAISS index files not found. Build the database first.")
            return False
    except Exception as e:
        logger.exception(f"FAISS init failed: {e}")
        return False

# -------------------
# FFmpeg helpers
# -------------------
if shutil.which(FFMPEG_BIN) is None:
    alt = shutil.which("ffmpeg")
    if alt:
        FFMPEG_BIN = alt

def convert_to_wav_16k(src_path: str) -> str:
    dst_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.wav")
    cmd = [FFMPEG_BIN, "-hide_banner", "-loglevel", "error",
           "-y", "-i", src_path, "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "wav", dst_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return dst_path

def start_ffmpeg_decoder():
    cmd = [
        FFMPEG_BIN, '-y',
        '-f', 'matroska,webm',
        '-err_detect', 'ignore_err',
        '-analyzeduration', '0',
        '-probesize', '32',
        '-fflags', '+genpts+igndts',
        '-re',
        '-i', 'pipe:0',
        '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', '1', '-acodec', 'pcm_s16le',
        'pipe:1'
    ]
    try:
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=1024)
    except FileNotFoundError:
        raise RuntimeError(f"FFmpeg not found at {FFMPEG_BIN!r}")

# -------------------
# Routes: CSRF & Health
# -------------------
@app.get('/csrf-token')
def get_csrf_token():
    return jsonify({'csrfToken': generate_csrf()})

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'faiss_loaded': faiss_system is not None})

# -------------------
# WebSocket: Live STT — finals-first
# -------------------
@sock.route('/ws/stt')
def ws_stt(ws):
    client_lang = (request.args.get('lang', 'bilingual') or 'bilingual').strip().lower()
    use_jacaranda = (client_lang == 'swahili')

    if use_jacaranda and (JACARANDA_MODEL_DIR and not os.path.isdir(JACARANDA_MODEL_DIR)):
        ws.send(json.dumps({"type": "error", "message": "Jacaranda model directory not found"}))
        return

    BYTES_PER_SAMPLE = 2
    FRAME_BYTES = int(SAMPLE_RATE * (VAD_FRAME_MS / 1000.0)) * BYTES_PER_SAMPLE
    RING_SECONDS = 14
    RING_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE * RING_SECONDS

    def send_json(obj):
        try: ws.send(json.dumps(obj))
        except Exception: pass

    try:
        ff = start_ffmpeg_decoder()
    except Exception as e:
        send_json({"type": "error", "message": str(e)}); return

    ring = bytearray()
    ws_buf = b''
    stop = threading.Event()
    MIN_WEBSOCKET_CHUNK = 512

    in_speech = False
    seg_buf = bytearray()
    seg_start_ts = None
    last_voiced_ts = None
    last_emit_partial_ts = 0.0

    job_q: "queue.Queue[bytes]" = queue.Queue(maxsize=6)

    def worker():
        engine = "jacaranda" if use_jacaranda else "whisper"
        while not stop.is_set():
            try:
                pcm = job_q.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                    wav_path = tf.name
                with wave.open(wav_path, 'wb') as wf:
                    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE); wf.writeframes(pcm)

                if use_jacaranda:
                    text = JacarandaTranscriber.transcribe_wav(wav_path)
                else:
                    lang_hint = "en" if client_lang == "english" else None
                    prompt = ("Clinician-patient conversation in Kenya. Transcribe literally and keep any code-switching.")
                    text = FasterWhisperTranscriber.transcribe_wav(wav_path, lang=lang_hint, initial_prompt=prompt)

                text = _clean_text(text or "")
                if text:
                    send_json({"type": "final", "text": text, "engine": engine})
            except Exception:
                logging.exception("ASR worker failed")
            finally:
                try:
                    os.unlink(wav_path)
                except Exception:
                    pass
                job_q.task_done()

    threading.Thread(target=worker, daemon=True).start()

    def ingest():
        nonlocal ws_buf
        try:
            send_json({"type": "meter", "bytes_in": 0, "bytes_pcm": 0})
            while not stop.is_set():
                frame = ws.receive()
                if frame is None: break
                if isinstance(frame, str) or not frame: continue
                ws_buf += frame
                if len(ws_buf) >= MIN_WEBSOCKET_CHUNK:
                    try:
                        ff.stdin.write(ws_buf); ff.stdin.flush()
                        ws_buf = b''
                    except Exception:
                        logger.exception("FFmpeg stdin write failed"); break
        except Exception:
            logger.debug("WS ingest ended")
        finally:
            stop.set()
            try:
                if ws_buf:
                    ff.stdin.write(ws_buf); ff.stdin.flush()
                ff.stdin.close()
            except Exception:
                pass

    threading.Thread(target=ingest, daemon=True).start()

    last_meter_ts = time.time()

    try:
        while not stop.is_set():
            try:
                chunk = ff.stdout.read(FRAME_BYTES) if hasattr(ff.stdout, "read") else b""
            except Exception:
                chunk = b""

            if chunk:
                ring.extend(chunk)
                if len(ring) > RING_BYTES:
                    del ring[: -RING_BYTES]

                if not in_speech:
                    tail_len = FRAME_BYTES * max(int(1000 / VAD_FRAME_MS), 1)
                    tail = bytes(ring[-tail_len:]) if len(ring) >= tail_len else bytes(ring)
                    if len(tail) >= FRAME_BYTES:
                        vr = vad_voiced_ratio(tail, SAMPLE_RATE, frame_ms=VAD_FRAME_MS, aggressiveness=VAD_AGGR)
                        if vr >= VAD_RATIO_MIN and rms_level(tail) >= RMS_MIN:
                            in_speech = True
                            seg_buf.extend(tail)
                            seg_start_ts = time.time()
                            last_voiced_ts = time.time()
                else:
                    seg_buf.extend(chunk)
                    vr_frame = vad_voiced_ratio(chunk, SAMPLE_RATE, frame_ms=VAD_FRAME_MS, aggressiveness=VAD_AGGR)
                    if vr_frame >= VAD_RATIO_MIN or rms_level(chunk) >= RMS_MIN:
                        last_voiced_ts = time.time()

                    if EMIT_PARTIALS and (time.time() - last_emit_partial_ts) * 1000.0 >= PARTIAL_MIN_INTERVAL_MS:
                        tail_win = int(SAMPLE_RATE * 2.0) * 2
                        tail = bytes(seg_buf[-tail_win:]) if len(seg_buf) > tail_win else bytes(seg_buf)
                        if len(tail) >= 3 * FRAME_BYTES:
                            try:
                                if use_jacaranda:
                                    ptext = JacarandaTranscriber.transcribe_wav(_bytes_to_temp_wav(tail, SAMPLE_RATE))
                                    engine = "jacaranda"
                                else:
                                    lang_hint = "en" if client_lang == "english" else None
                                    ptext = FasterWhisperTranscriber.transcribe_wav(_bytes_to_temp_wav(tail, SAMPLE_RATE), lang=lang_hint)
                                    engine = "whisper"
                                ptext = _clean_text(ptext or "")
                                if ptext:
                                    send_json({"type": "partial", "text": ptext, "engine": engine})
                            except Exception:
                                pass
                            finally:
                                last_emit_partial_ts = time.time()

                    if seg_start_ts and (time.time() - seg_start_ts) >= MAX_SEGMENT_SEC:
                        if not job_q.full() and len(seg_buf) > FRAME_BYTES * 5:
                            job_q.put(bytes(seg_buf))
                        seg_buf.clear()
                        in_speech = False
                        seg_start_ts = None
                        last_voiced_ts = None

                    if last_voiced_ts and (time.time() - last_voiced_ts) * 1000.0 >= SEGMENT_SILENCE_MS:
                        if not job_q.full() and len(seg_buf) > FRAME_BYTES * 5:
                            job_q.put(bytes(seg_buf))
                        seg_buf.clear()
                        in_speech = False
                        seg_start_ts = None
                        last_voiced_ts = None

            if (time.time() - last_meter_ts) > 1.0:
                send_json({"type": "meter", "status": "ok"})
                last_meter_ts = time.time()

            time.sleep(0.005)

    except Exception:
        logger.exception("/ws/stt loop error")
    finally:
        stop.set()
        try:
            if ff: ff.terminate()
        except Exception: pass
        try:
            if len(seg_buf) > FRAME_BYTES * 5 and not job_q.full():
                job_q.put(bytes(seg_buf))
        except Exception:
            pass
        t0 = time.time()
        while not job_q.empty() and (time.time() - t0) < 2.0:
            time.sleep(0.05)

# -------------------
# SSE agent chat stream (kept)
# -------------------
@app.route('/agent_chat_stream')
@login_required
def agent_chat_stream():
    if not current_user.is_authenticated:
        return "Forbidden", 403

    message = request.args.get('message', '').strip()
    language = request.args.get('lang', 'bilingual').strip().lower()
    role = request.args.get('role', 'patient').strip().lower()
    mode = request.args.get('mode', 'real').strip().lower()
    suggest = request.args.get('suggest', 'stream').strip().lower()

    if not message:
        return jsonify({'error': 'No message provided'}), 400

    if not session.get('id'):
        session['id'] = create_conversation(owner_user_id=current_user.id)
        session['conv'] = []
    sid = session['id']

    conv = session.get('conv', [])
    conv.append({"role": role, "message": message})
    session['conv'] = conv

    def _log_hook(session_id, role_, message_, timestamp_, type_="message"):
        try:
            log_message(session_id, role_, message_, timestamp_, type_)
        except Exception:
            logger.exception("DB log failed")

    if mode == "simulated":
        generator = simulate_agent_chat_stepwise(
            message,
            language_mode=language,
            log_hook=_log_hook,
            session_id=sid,
        )
    elif mode == "live":
        generator = live_transcription_stream(
            message,
            language_mode=language,
            speaker_role=role,
            suggest_mode=suggest,
            conversation_history=conv,
            log_hook=_log_hook,
            session_id=sid,
        )
    else:
        generator = real_actor_chat_stepwise(
            message,
            language_mode=language,
            speaker_role=role,
            conversation_history=conv,
            log_hook=_log_hook,
            session_id=sid,
        )

    return Response(stream_with_context(generator), mimetype='text/event-stream')

# -------------------
# FAISS suggest/answer endpoints (kept)
# -------------------
_PENDING_FAISS_Q: dict[str, tuple[str | None, str | None]] = {}

def _set_pending_faiss_q(cid: str, qid: str | None, cat: str | None):
    if cid:
        _PENDING_FAISS_Q[cid] = (qid, cat)

def _pop_pending_faiss_q(cid: str) -> tuple[str | None, str | None]:
    return _PENDING_FAISS_Q.pop(cid, (None, None))

@app.post("/faiss/suggest_question")
@csrf.exempt
@login_required
def faiss_suggest_question():
    if faiss_system is None:
        return jsonify({"error": "FAISS not loaded"}), 503

    data = request.get_json(silent=True) or {}
    query_text = (data.get("text") or "").strip()
    k = int(data.get("k", 1))
    if not query_text:
        return jsonify({"error": "text is required"}), 400

    try:
        results = faiss_system.suggest_questions(query_text, k=max(1, k), threshold=0.38) or []
    except Exception as e:
        logger.exception("FAISS suggest failed")
        return jsonify({"error": f"FAISS suggest failed: {e}"}), 500

    if not results:
        return jsonify({"question": None, "reason": "no_match"}), 200

    q = results[0]
    if not session.get('id'):
        session['id'] = create_conversation(owner_user_id=current_user.id)
        session['conv'] = []
    sid = session['id']

    eng_text = (q.get("question", {}) or {}).get("english") or ""
    try:
        log_message(
            sid,
            role="question_recommender",
            message=eng_text.strip(),
            timestamp=datetime.utcnow().isoformat(timespec="seconds"),
            type_="question_recommender",
            faiss_question_id=q.get("id"),
            faiss_category=q.get("category"),
            faiss_is_answer=False,
        )
    except Exception:
        logger.exception("DB log for FAISS question failed")

    _set_pending_faiss_q(sid, q.get("id"), q.get("category"))

    payload = {
        "question": q.get("question"),
        "id": q.get("id"),
        "category": q.get("category"),
        "similarity": q.get("similarity"),
    }
    return jsonify(payload), 200

@app.post("/faiss/mark_answer")
@csrf.exempt
@login_required
def faiss_mark_answer():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400

    if not session.get('id'):
        session['id'] = create_conversation(owner_user_id=current_user.id)
        session['conv'] = []
    sid = session['id']

    qid, qcat = _pop_pending_faiss_q(sid)

    try:
        log_message(
            sid,
            role="patient",
            message=text,
            timestamp=datetime.utcnow().isoformat(timespec="seconds"),
            type_="message",
            faiss_question_id=qid,
            faiss_category=qcat,
            faiss_is_answer=bool(qid),
        )
    except Exception:
        logger.exception("DB log for patient answer failed")

    return jsonify({"ok": True, "linked_to_faiss": bool(qid), "faiss_question_id": qid, "faiss_category": qcat}), 200

# -------------------
# /search — robust fallback to Questions-FAISS
# -------------------
@app.route('/search', methods=['POST'])
@csrf.exempt
@login_required
def search():
    """
    If a Cases-FAISS with `.search_similar_cases` exists, use it.
    Otherwise, gracefully fall back to Questions-FAISS and return top matching questions.
    """
    try:
        data = request.get_json() or {}
        query = (data.get('query') or '').strip()
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400

        k = int(min(data.get('max_results', 10), app.config.get('MAX_RESULTS', 25)))
        similarity_threshold = float(data.get('similarity_threshold', app.config.get('DEFAULT_SIMILARITY_THRESHOLD', 0.19)))

        # CASES path (only if the loaded FAISS actually supports it)
        if hasattr(faiss_system, "search_similar_cases"):
            results = faiss_system.search_similar_cases(query, k=k, similarity_threshold=similarity_threshold) or []
            out = []
            for r in results:
                out.append({
                    'case_id': getattr(r, 'case_id', ''),
                    'similarity_score': round(getattr(r, 'similarity_score', 0.0), 4),
                    'patient_background': getattr(r, 'patient_background', {}),
                    'chief_complaint': getattr(r, 'chief_complaint', {}),
                    'medical_history': getattr(r, 'medical_history', {}),
                    'opening_statement': getattr(r, 'opening_statement', {}),
                    'recommended_questions': (getattr(r, 'recommended_questions', []) or [])[:5],
                    'red_flags': getattr(r, 'red_flags', {}),
                    'Suspected_illness': getattr(r, 'Suspected_illness', ''),
                })
            return jsonify({
                'query': query,
                'results': out,
                'suggested_questions': [],
                'total_results': len(out),
            })

        # QUESTIONS fallback
        hits = faiss_system.suggest_questions(query, k=k, threshold=max(0.30, similarity_threshold)) or []
        out_q = [{
            "question_id": h.get("id"),
            "question": h.get("question"),
            "category": h.get("category"),
            "similarity": float(h.get("similarity", 0.0)),
            "tags": h.get("tags", []),
        } for h in hits]
        return jsonify({
            'query': query,
            'results': [],
            'suggested_questions': out_q,
            'total_results': len(out_q),
        })

    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': 'An error occurred during search'}), 500

# -------------------
# Question-bank endpoints (browse/search/print/export)
# -------------------
def _ensure_qfaiss():
    if not faiss_system:
        raise RuntimeError("Questions FAISS not available")
    return faiss_system

def _norm_cat(c):
    c = (c or '').strip().lower()
    return c if c in ('depression', 'anxiety', 'psychosis') else None

@app.get("/questions/meta")
@login_required
def questions_meta():
    try:
        f = _ensure_qfaiss()
        cats = {"depression":0,"anxiety":0,"psychosis":0,"other":0}
        for q in f.questions:
            cat = (q.get("category") or "").lower()
            if cat in cats: cats[cat]+=1
            else: cats["other"]+=1
        return jsonify({"categories": cats, "total": len(f.questions)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/questions/list")
@login_required
def questions_list():
    try:
        f = _ensure_qfaiss()
        cat = _norm_cat(request.args.get("category"))
        qtext = (request.args.get("q") or "").strip().lower()
        items = []
        for it in f.questions:
            c = (it.get("category") or "").lower() or None
            if cat and c != cat: continue
            en = (it.get("question",{}).get("english") or "")
            sw = (it.get("question",{}).get("swahili") or "")
            blob = (en + " " + sw).lower()
            if qtext and qtext not in blob: continue
            items.append({
                "id": it.get("id"),
                "category": c,
                "english": en,
                "swahili": sw,
                "tags": it.get("tags", [])
            })
        return jsonify({"count": len(items), "items": items})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/questions/search")
@csrf.exempt
@login_required
def questions_search():
    try:
        f = _ensure_qfaiss()
        data = request.get_json(force=True) or {}
        query = (data.get("query") or "").strip()
        cat = _norm_cat(data.get("category"))
        k = int(data.get("k", 25))
        if not query:
            return jsonify({"count": 0, "items": []})

        hits = f.suggest_questions(query, k=k, threshold=0.30) or []

        def _norm(h):
            # dict-style (unchanged) ...
            if isinstance(h, dict):
                item = h.get("item") or h.get("data") or h
                q = item.get("question") if isinstance(item, dict) else {}
                return {
                    "id": item.get("id") or h.get("id"),
                    "category": (item.get("category") or h.get("category") or "").lower() or None,
                    "english": (q or {}).get("english") or h.get("question", {}).get("english") or "",
                    "swahili": (q or {}).get("swahili") or h.get("question", {}).get("swahili") or "",
                    "similarity": float(h.get("similarity") or item.get("similarity") or 0.0),
                    "tags": item.get("tags") or h.get("tags") or [],
                }

            # object-style (QuestionSearchResult dataclass)
            # attributes: question_id, question (dict), category, tags, similarity_score
            q = getattr(h, "question", {}) or {}
            return {
                "id": getattr(h, "question_id", None),
                "category": (getattr(h, "category", "") or "").lower() or None,
                "english": (q.get("english") if isinstance(q, dict) else "") or "",
                "swahili": (q.get("swahili") if isinstance(q, dict) else "") or "",
                "similarity": float(getattr(h, "similarity_score", 0.0) or 0.0),
                "tags": list(getattr(h, "tags", []) or []),
            }

        out = []
        for h in hits:
            row = _norm(h)
            if cat and row["category"] != cat:
                continue
            out.append(row)

        return jsonify({"count": len(out), "items": out})
    except Exception as e:
        app.logger.exception("questions_search failed")
        return jsonify({"error": str(e)}), 500


@app.get("/questions/print")
@login_required
def questions_print():
    try:
        f = _ensure_qfaiss()
        cat = _norm_cat(request.args.get("category"))
        items = []
        for it in f.questions:
            c = (it.get("category") or "").lower() or None
            if cat and c != cat: continue
            items.append({
                "id": it.get("id"),
                "category": c or "",
                "english": (it.get("question",{}).get("english") or ""),
                "swahili": (it.get("question",{}).get("swahili") or "")
            })
        html = ["<html><head><meta charset='utf-8'><title>Question Bank</title>",
                "<style>body{font-family:sans-serif} .q{margin:10px 0;padding:8px;border-bottom:1px solid #ddd}</style>",
                "</head><body>"]
        html.append(f"<h2>Question Bank{(' — ' + cat.capitalize()) if cat else ''}</h2>")
        html.append("<p><em>English and Swahili</em></p>")
        for x in items:
            html.append(
                f"<div class='q'><div><strong>{x['id']}</strong> · "
                f"<span style='color:#555'>{x['category']}</span></div>"
                f"<div><strong>English:</strong> {x['english']}</div>"
                f"<div><strong>Swahili:</strong> {x['swahili']}</div></div>"
            )
        html.append("<script>window.print()</script></body></html>")
        return Response("\n".join(html), mimetype="text/html")
    except Exception as e:
        return Response(f"<pre>Error: {e}</pre>", mimetype="text/html", status=500)

@app.get("/questions/export")
@login_required
def questions_export():
    try:
        f = _ensure_qfaiss()
        cat = _norm_cat(request.args.get("category"))
        qtext = (request.args.get("q") or "").strip().lower()

        output = io.StringIO()
        w = csv.writer(output)
        w.writerow(["id","category","english","swahili","tags"])
        for it in f.questions:
            c = (it.get("category") or "").lower() or None
            if cat and c != cat: continue
            en = (it.get("question",{}).get("english") or "")
            sw = (it.get("question",{}).get("swahili") or "")
            blob = (en + " " + sw).lower()
            if qtext and qtext not in blob: continue
            w.writerow([it.get("id"), c or "", en, sw, " ".join(it.get("tags",[]))])

        mem = io.BytesIO(output.getvalue().encode("utf-8"))
        filename = f"questions{('-' + cat) if cat else ''}.csv"
        return send_file(mem, mimetype="text/csv", as_attachment=True, download_name=filename)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------
# Case details (return 404 if using Questions-FAISS)
# -------------------
@app.route('/case/<case_id>')
@login_required
def get_case_details(case_id):
    try:
        if hasattr(faiss_system, "get_case_details"):
            case_details = faiss_system.get_case_details(case_id)
            if case_details:
                return jsonify(case_details)
            else:
                return jsonify({'error': 'Case not found'}), 404
        return jsonify({'error': 'Cases index not available'}), 404
    except Exception as e:
        logger.error(f"Error getting case details: {e}")
        return jsonify({'error': 'An error occurred'}), 500

# -------------------
# Admin: FAISS answered summary (kept)
# -------------------
@app.get("/admin/api/faiss_answered_summary")
@login_required
def admin_faiss_answered_summary():
    if not any(r.name == "admin" for r in current_user.roles):
        return "Forbidden", 403
    db = SessionLocal()
    try:
        from sqlalchemy import func
        rows = (
            db.query(Message.faiss_category, func.count())
              .filter(Message.faiss_is_answer.is_(True))
              .group_by(Message.faiss_category)
              .all()
        )
        counts = {"depression": 0, "anxiety": 0, "psychosis": 0}
        for cat, c in rows:
            if cat in counts:
                counts[cat] = int(c)
        total = sum(counts.values()) or 1
        pct = {k: round(100.0 * v / total, 1) for k, v in counts.items()}
        return jsonify({"counts": counts, "percents": pct})
    finally:
        db.close()

# -------------------
# Basic pages
# -------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/demo')
def demo():
    demo_queries = [
        "finger pain stiffness morning",
        "breathing difficulty night cough",
        "joint pain swelling",
        "wheezing chest whistling sound",
        "fatigue hand pain work difficulty",
        "headache fever nausea",
        "chest pain shortness breath",
        "dizziness balance problems",
    ]
    return jsonify({'demo_queries': demo_queries})

@app.route('/admin')
@login_required
def admin_page():
    if not any(r.name == "admin" for r in current_user.roles):
        return "Forbidden", 403
    return render_template('admin.html')

@app.errorhandler(404)
def not_found(error):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# -------------------
# Conversation helpers
# -------------------
@app.route('/reset_conv', methods=['POST'])
@csrf.exempt
@login_required
def reset_conv():
    session['conv'] = []
    cid = create_conversation(owner_user_id=current_user.id)
    session['id'] = cid
    return jsonify({'ok': True, 'conversation_id': cid})

# -------------------
# Batch Transcribe (voice note)
# -------------------
@app.route('/transcribe_audio', methods=['POST'])
@csrf.exempt
def transcribe_audio():
    try:
        audio = request.files.get('audio')
        lang = (request.form.get('lang') or 'bilingual').strip().lower()
        if not audio:
            return jsonify({'error': 'No audio uploaded'}), 400

        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_audio:
            audio.save(temp_audio.name)
        wav_path = convert_to_wav_16k(temp_audio.name)

        if lang == "swahili":
            text = JacarandaTranscriber.transcribe_wav(wav_path)
            engine = "jacaranda"
        else:
            prompt = ("Clinician-patient conversation in Kenya. Transcribe literally; keep any code-switching.")
            lang_hint = "en" if lang == "english" else None
            text = FasterWhisperTranscriber.transcribe_wav(wav_path, lang=lang_hint, initial_prompt=prompt)
            engine = "whisper"

        text = _clean_text(text or '')
        return jsonify({'text': text, 'engine': engine})
    except Exception:
        logger.exception("Error during audio transcription")
        return jsonify({'error': 'Audio transcription failed'}), 500

# -------------------
# Screening API
# -------------------
@csrf.exempt
@app.post("/mh/screen")
def mh_screen():
    data = request.get_json(force=True, silent=True) or {}
    transcript = data.get("transcript") or ""
    responses = data.get("responses") or {}
    safety = bool(data.get("safety_concerns", False))
    app.logger.debug(
        "MH screen payload: transcript=%r, responses_keys=%s, safety=%s",
        (transcript or "")[:160], list((responses or {}).keys()), safety,
    )
    out = run_screening(transcript, responses, safety_concerns=safety)
    return jsonify(screening_to_dict(out))

# -------------------
# Main
# -------------------
if __name__ == '__main__':
    if initialize_faiss():
        try:
            init_db()
        except Exception:
            logger.exception("DB init failed")
        logger.info("Starting Flask application…")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("Failed to initialize FAISS system. Application cannot start.")
        logger.error("Please ensure the following files exist:")
        logger.error(f"- {app.config.get('FAISS_INDEX_PATH')}")
        logger.error(f"- {app.config.get('FAISS_METADATA_PATH')}")
