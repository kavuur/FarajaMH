import os
import json
import re
from docx import Document
from dotenv import load_dotenv, find_dotenv
from langdetect import detect
import pyttsx3
import speech_recognition as sr

# === ENV HANDLING ===
def load_env():
    _ = load_dotenv(find_dotenv())

def get_openai_api_key():
    load_env()
    return os.getenv("OPENAI_API_KEY")

# === READER ===
def read_docx(file_path):
    doc = Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

# === LANGUAGE DETECTION UTILS ===
def detect_lang(text):
    try:
        return detect(text)
    except:
        return "unknown"

# === SECTION EXTRACTOR ===
def extract_section_lines(lines, start_headers, stop_headers):
    section_lines = []
    in_section = False
    for line in lines:
        if any(h.lower() in line.lower() for h in start_headers):
            in_section = True
            continue
        if in_section and any(h.lower() in line.lower() for h in stop_headers):
            break
        if in_section:
            section_lines.append(line)
    return section_lines

def split_by_language_block(section_lines):
    """Split a block of lines into English and Swahili assuming sequential EN then SW paragraphs."""
    lines = [l.strip() for l in section_lines if l.strip()]
    if not lines:
        return "", ""
    midpoint = len(lines) // 2
    english = " ".join(lines[:midpoint])
    swahili = " ".join(lines[midpoint:])
    return english.strip(), swahili.strip()

# === CASE SPLITTER ===
def split_cases(full_text):
    # Match headings like: Standardized Patient Case 10
    cases = re.split(r'Standardized Patient Case\s+(\d+)', full_text, flags=re.IGNORECASE)
    cases = cases[1:]  # remove anything before the first match
    return [{'case_id': cases[i], 'content': cases[i+1]} for i in range(0, len(cases), 2)]

# === PARSER ===
def extract_case_fields(case_data):
    content = case_data['content']
    lines = [line.strip() for line in content.split('\n') if line.strip()]

    # Patient Background
    pb_lines = extract_section_lines(lines, ["Patient Background", "Asili ya Mgonjwa"], ["Chief Complaint", "Malalamiko makuu"])
    pb_en, pb_sw = split_by_language_block(pb_lines)

    # Chief Complaint & History of Present Illness
    cc_lines = extract_section_lines(
        lines,
        ["Chief Complaint", "History of Present Illness", "Malalamiko makuu", "Historia ya Ugonjwa wa Sasa"],
        ["Medical & Social History", "Historia ya Matibabu", "Opening Statement", "Taarifa ya ufunguzi"]
    )
    cc_en, cc_sw = split_by_language_block(cc_lines)

    # Medical & Social History
    ms_lines = extract_section_lines(
        lines,
        ["Medical & Social History", "Historia ya Matibabu na Jamii"],
        ["Opening Statement", "Taarifa ya ufunguzi"]
    )
    ms_en, ms_sw = split_by_language_block(ms_lines)

    # Opening Statement
    op_lines = extract_section_lines(
        lines,
        ["Opening statement:", "Taarifa ya ufunguzi:"],
        ["Provider Questions", "Maswali ya Mtoa Huduma"]
    )
    op_en, op_sw = split_by_language_block(op_lines)

    # Extract Provider Questions and SP Responses
    questions = extract_questions_bilingual(lines)

    return {
        "case_id": case_data['case_id'],
        "Suspected_illness": "", 
        "red_flags": [],
        "patient_background": {
            "english": pb_en,
            "swahili": pb_sw
        },
        "chief_complaint_history": {
            "english": cc_en,
            "swahili": cc_sw
        },
        "medical_social_history": {
            "english": ms_en,
            "swahili": ms_sw
        },
        "opening_statement": {
            "english": op_en,
            "swahili": op_sw
        },
        "recommended_questions": questions
    }

def extract_questions_bilingual(lines):
    questions = []
    in_section = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if "Provider Questions" in line or "Maswali ya Mtoa Huduma" in line:
            in_section = True
            i += 1
            continue

        if in_section:
            if i + 3 < len(lines):
                q_en = lines[i].strip()
                q_sw = lines[i+1].strip()
                a_en_line = lines[i+2].strip()
                a_sw = lines[i+3].strip()

                a_en = ""
                if a_en_line.lower().startswith('a.'):
                    a_en = a_en_line[2:].strip()
                else:
                    a_en = a_en_line

                questions.append({
                    "question": {"english": q_en, "swahili": q_sw},
                    "response": {"english": a_en, "swahili": a_sw}
                })
                i += 4
            else:
                break
        else:
            i += 1

    return questions

# === JSON WRITER ===
def write_to_json(cases, filename="cases.json"):
    # Convert red_flags list to dictionary format if needed
    for case in cases:
        if isinstance(case.get("red_flags"), list):
            red_dict = {}
            for flag in case["red_flags"]:
                if ">" in flag:
                    key, val = flag.split(">", 1)
                    red_dict[key.strip()] = f">{val.strip()}"
                elif ":" in flag:
                    key, val = flag.split(":", 1)
                    red_dict[key.strip()] = val.strip()
                else:
                    red_dict[flag] = True  # fallback
            case["red_flags"] = red_dict

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

#def write_to_json(cases, filename="cases_new.jsonl"):
 #   with open(filename, "w", encoding="utf-8") as f:
  #      for case in cases:
   #         json.dump(case, f, ensure_ascii=False, indent=2)
    #        f.write("\n")

#def write_to_json(cases, filename="cases.jsonl"):
#    with open(filename, "w", encoding="utf-8") as f:
 #       for case in cases:
  #          json.dump(case, f, ensure_ascii=False, indent=2)
   #         f.write("\n")
    #return filename

# === RED FLAG TAGGER ===
def label_red_flags(case_data):
    red_flags = []
    for section_key in ["patient_background", "chief_complaint_history", "medical_social_history"]:
        section_data = case_data.get(section_key, {})
        en = section_data.get("english", "").lower()
        sw = section_data.get("swahili", "").lower()
        combined = en + " " + sw
        if "months" in combined and ("pain" in combined or "bleeding" in combined):
            red_flags.append("Symptom duration > 3 months")
        if "weight loss" in combined:
            red_flags.append("Unintentional weight loss")
        if "blood" in combined:
            red_flags.append("Possible cancer-related bleeding")
    case_data["red_flags"] = red_flags
    return case_data


# === SPEECH TO TEXT ===
def speak(text, language_hint="en"):
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    chosen = None
    for v in engine.getProperty('voices'):
        name = (v.name or "").lower()
        lang = " ".join(v.languages or [])
        # pick by hint
        if language_hint.startswith("sw") and ("swahili" in name or "sw" in lang):
            chosen = v.id; break
        if language_hint.startswith("en") and ("english" in name or "en" in lang):
            chosen = v.id; break
    if chosen: engine.setProperty('voice', chosen)
    engine.say(text)
    engine.runAndWait()

    
def listen(language="en-KE", role="patient"):
    import speech_recognition as sr
    r = sr.Recognizer()
    r.energy_threshold = 300       # tune for your environment
    r.pause_threshold = 0.6
    with sr.Microphone() as source:
        print(f"🎤 Listening as {role} [{language}]...")
        r.adjust_for_ambient_noise(source, duration=0.5)
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio, language=language)
        return role, text
    except sr.UnknownValueError:
        return role, ""
    except sr.RequestError as e:
        return role, f"[Speech error: {e}]"
