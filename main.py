import os
import re
import json
import base64
import copy
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


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

LAST_DEBUG = {
    "error": None,
    "mime_type": None,
    "model": GEMINI_MODEL,
    "raw_text": None,
}


def empty_response():
    return copy.deepcopy(EMPTY_RESPONSE)


def get_api_key():
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def decode_base64_and_mime(audio_base64: str):
    mime_type = None

    if audio_base64.startswith("data:") and "," in audio_base64:
        header, audio_base64 = audio_base64.split(",", 1)

        if ":" in header and ";" in header:
            mime_type = header.split(":", 1)[1].split(";", 1)[0].strip().lower()

    clean_b64 = audio_base64.strip()
    audio_bytes = base64.b64decode(clean_b64)

    if mime_type:
        if mime_type in ["audio/mp3", "audio/x-mp3"]:
            mime_type = "audio/mpeg"
        if mime_type in ["audio/x-wav", "audio/wave"]:
            mime_type = "audio/wav"
        return clean_b64, audio_bytes, mime_type

    if audio_bytes.startswith(b"RIFF"):
        return clean_b64, audio_bytes, "audio/wav"

    if audio_bytes.startswith(b"ID3") or audio_bytes[:2] == b"\xff\xfb":
        return clean_b64, audio_bytes, "audio/mpeg"

    if audio_bytes.startswith(b"OggS"):
        return clean_b64, audio_bytes, "audio/ogg"

    if audio_bytes.startswith(b"fLaC"):
        return clean_b64, audio_bytes, "audio/flac"

    if b"ftyp" in audio_bytes[:80]:
        return clean_b64, audio_bytes, "audio/mp4"

    if audio_bytes.startswith(b"\x1A\x45\xDF\xA3"):
        return clean_b64, audio_bytes, "audio/webm"

    return clean_b64, audio_bytes, "audio/webm"


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


def infer_columns(result):
    columns = result.get("columns", [])

    if not isinstance(columns, list):
        columns = []

    columns = [str(c) for c in columns if str(c).strip()]

    for key in [
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
    ]:
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

    result["columns"] = infer_columns(result)
    columns = set(result["columns"])

    for key in ["mean", "std", "variance", "min", "max", "median", "mode", "range"]:
        result[key] = clean_numeric_dict(result[key], columns)

    result["allowed_values"] = clean_allowed_values(result["allowed_values"], columns)
    result["value_range"] = clean_value_range(result["value_range"], columns)

    if not isinstance(result["correlation"], list):
        result["correlation"] = []

    return result


def make_prompt(audio_id: str):
    return f"""
You are an audio-to-JSON statistics extraction API.

Listen carefully to the audio. The audio may be in English, Hindi, Japanese, Korean, or mixed language.

Return ONLY valid JSON. No markdown. No explanation.

Return exactly this JSON structure:

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
- Keep Korean column names exactly, for example 소득.
- Keep Japanese column names exactly, for example 会社.
- If the audio says the column is 소득, columns must be ["소득"].
- If the audio says the column is 会社, columns must be ["会社"].
- rows must be an integer.
- columns must list every column found in the audio.
- Numeric values must be JSON numbers, not strings.
- For numeric columns, compute mean, std, variance, min, max, median, mode, and range.
- For categorical/string columns, do not fill mean/std/variance/min/max/median/mode/range.
- For categorical/string columns, put categories only in allowed_values.
- If a field is not applicable, use {{}} or [].

audio_id: {audio_id}
"""


def call_gemini_rest(audio_base64: str, mime_type: str, audio_id: str):
    api_key = get_api_key()

    if not api_key:
        LAST_DEBUG["error"] = "Missing API key"
        return empty_response()

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={api_key}"
    )

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": make_prompt(audio_id)},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": audio_base64,
                        }
                    },
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "topP": 1,
            "topK": 1,
            "responseMimeType": "application/json",
        },
    }

    LAST_DEBUG["mime_type"] = mime_type
    LAST_DEBUG["model"] = GEMINI_MODEL
    LAST_DEBUG["error"] = None
    LAST_DEBUG["raw_text"] = None

    try:
        response = requests.post(url, json=payload, timeout=10)

        if response.status_code != 200:
            LAST_DEBUG["error"] = f"HTTP {response.status_code}: {response.text[:500]}"
            return empty_response()

        data = response.json()

        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "{}")
        )

        LAST_DEBUG["raw_text"] = text[:1000]

        obj = extract_json(text)

        if obj is None:
            LAST_DEBUG["error"] = "Could not parse JSON"
            return empty_response()

        return normalize_response(obj)

    except Exception as e:
        LAST_DEBUG["error"] = str(e)
        return empty_response()


@app.get("/")
def root():
    return {"status": "ok", "message": "Audio statistics API is running"}


@app.get("/debug-key")
def debug_key():
    api_key = get_api_key()
    return {
        "has_key": bool(api_key),
        "model": GEMINI_MODEL
    }


@app.get("/debug-last")
def debug_last():
    return LAST_DEBUG


@app.post("/analyze-audio")
def analyze_audio(req: AudioRequest):
    try:
        clean_b64, _audio_bytes, mime_type = decode_base64_and_mime(req.audio_base64)
        return call_gemini_rest(clean_b64, mime_type, req.audio_id)

    except Exception as e:
        LAST_DEBUG["error"] = str(e)
        return empty_response()


@app.post("/")
def analyze_audio_root(req: AudioRequest):
    return analyze_audio(req)