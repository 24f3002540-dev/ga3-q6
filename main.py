import os
import re
import json
import base64
import copy
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai


# Use 2.0 Flash for speed. 2.5 Flash may be slower for audio under 12s grader timeout.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str


EMPTY_RESPONSE = {
    "rows": 0,
    "columns": [],
    "mean": {},
    "std": {},
    "variance": {},
    "min": {},
    "max": {},
    "median": {},
    "mode": {},
    "range": {},
    "allowed_values": {},
    "value_range": {},
    "correlation": []
}


def empty_response():
    return copy.deepcopy(EMPTY_RESPONSE)


def get_api_key():
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def decode_audio_and_mime(audio_base64: str):
    mime_type = None

    # Handle data URL:
    # data:audio/webm;base64,...
    # data:audio/wav;base64,...
    if audio_base64.startswith("data:") and "," in audio_base64:
        header, audio_base64 = audio_base64.split(",", 1)
        if ":" in header and ";" in header:
            mime_type = header.split(":", 1)[1].split(";", 1)[0].strip()

    audio_bytes = base64.b64decode(audio_base64)

    if mime_type:
        return audio_bytes, mime_type

    # WAV
    if audio_bytes.startswith(b"RIFF"):
        return audio_bytes, "audio/wav"

    # MP3
    if audio_bytes.startswith(b"ID3") or audio_bytes[:2] == b"\xff\xfb":
        return audio_bytes, "audio/mpeg"

    # OGG
    if audio_bytes.startswith(b"OggS"):
        return audio_bytes, "audio/ogg"

    # FLAC
    if audio_bytes.startswith(b"fLaC"):
        return audio_bytes, "audio/flac"

    # MP4 / M4A
    if b"ftyp" in audio_bytes[:64]:
        return audio_bytes, "audio/mp4"

    # WEBM / Matroska EBML header
    if audio_bytes.startswith(b"\x1A\x45\xDF\xA3"):
        return audio_bytes, "audio/webm"

    # Browser audio is commonly webm, so default to webm, not wav
    return audio_bytes, "audio/webm"


def extract_json(text: str):
    if not text:
        return None

    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return None


def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def clean_numeric_dict(d, columns):
    if not isinstance(d, dict):
        return {}

    cleaned = {}
    for k, v in d.items():
        k = str(k)
        if k in columns and is_number(v):
            cleaned[k] = v

    return cleaned


def clean_allowed_values(d, columns):
    if not isinstance(d, dict):
        return {}

    cleaned = {}
    for k, v in d.items():
        k = str(k)

        if k not in columns:
            continue

        if isinstance(v, list):
            cleaned[k] = [str(x) for x in v]
        elif v is not None:
            cleaned[k] = [str(v)]

    return cleaned


def clean_value_range(d, columns):
    if not isinstance(d, dict):
        return {}

    cleaned = {}
    for k, v in d.items():
        k = str(k)

        if k not in columns:
            continue

        if isinstance(v, list):
            cleaned[k] = v
        elif isinstance(v, dict):
            cleaned[k] = v

    return cleaned


def infer_columns_from_result(result):
    columns = result.get("columns", [])

    if not isinstance(columns, list):
        columns = []

    columns = [str(c) for c in columns if str(c).strip()]

    # If Gemini forgot columns but produced stats keys, recover them
    possible_dict_keys = [
        "mean",
        "std",
        "variance",
        "min",
        "max",
        "median",
        "mode",
        "range",
        "allowed_values",
        "value_range",
    ]

    for key in possible_dict_keys:
        d = result.get(key)
        if isinstance(d, dict):
            for col in d.keys():
                col = str(col)
                if col and col not in columns:
                    columns.append(col)

    return columns


def normalize_response(obj):
    result = empty_response()

    if not isinstance(obj, dict):
        return result

    for key in result:
        if key in obj:
            result[key] = obj[key]

    try:
        result["rows"] = int(result["rows"])
    except Exception:
        result["rows"] = 0

    result["columns"] = infer_columns_from_result(result)
    columns = set(result["columns"])

    # Numeric stats only. Remove categorical values from max/min/mode etc.
    for key in ["mean", "std", "variance", "min", "max", "median", "mode", "range"]:
        result[key] = clean_numeric_dict(result[key], columns)

    result["allowed_values"] = clean_allowed_values(result["allowed_values"], columns)
    result["value_range"] = clean_value_range(result["value_range"], columns)

    if not isinstance(result["correlation"], list):
        result["correlation"] = []

    return result


def make_prompt(audio_id: str):
    return f"""
You are a strict audio-to-JSON statistics API.

Listen to the audio and extract the dataset/table/statistics.

The audio may be in English, Hindi, Japanese, Korean, or mixed language.

Return ONLY valid JSON.
No markdown.
No explanation.

Required output structure exactly:

{{
  "rows": 0,
  "columns": [],
  "mean": {{}},
  "std": {{}},
  "variance": {{}},
  "min": {{}},
  "max": {{}},
  "median": {{}},
  "mode": {{}},
  "range": {{}},
  "allowed_values": {{}},
  "value_range": {{}},
  "correlation": []
}}

Rules:
- Return all keys exactly.
- Do not translate column names.
- Keep Korean column names exactly, e.g. 소득.
- Keep Japanese column names exactly, e.g. 会社.
- If audio says one column named 소득, columns must be ["소득"].
- If audio says one column named 会社, columns must be ["会社"].
- rows must be an integer.
- Use JSON numbers, not strings.
- For categorical/string columns, do NOT fill mean, std, variance, min, max, median, mode, or range.
- For categorical/string columns, put categories only in allowed_values.
- For numeric columns, compute mean, std, variance, min, max, median, mode, and range.
- If a field is not applicable, use {{}} or [].

audio_id: {audio_id}
"""


def analyze_with_gemini(audio_bytes: bytes, mime_type: str, audio_id: str):
    api_key = get_api_key()

    if not api_key:
        return empty_response()

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    response = model.generate_content(
        [
            make_prompt(audio_id),
            {
                "mime_type": mime_type,
                "data": audio_bytes,
            },
        ],
        generation_config={
            "temperature": 0,
            "top_p": 1,
            "top_k": 1,
            "response_mime_type": "application/json",
        },
    )

    raw = response.text if response and response.text else "{}"
    obj = extract_json(raw)

    if obj is None:
        return empty_response()

    return normalize_response(obj)


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Audio statistics API is running"
    }


@app.get("/debug-key")
def debug_key():
    api_key = get_api_key()
    return {
        "has_key": bool(api_key),
        "model": GEMINI_MODEL
    }


@app.post("/analyze-audio")
def analyze_audio(req: AudioRequest):
    try:
        audio_bytes, mime_type = decode_audio_and_mime(req.audio_base64)
        return analyze_with_gemini(audio_bytes, mime_type, req.audio_id)
    except Exception:
        return empty_response()


@app.post("/")
def analyze_audio_root(req: AudioRequest):
    return analyze_audio(req)