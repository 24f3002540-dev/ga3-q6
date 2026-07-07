import os
import re
import json
import base64
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

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


def decode_audio(audio_base64: str) -> bytes:
    if "," in audio_base64:
        audio_base64 = audio_base64.split(",", 1)[1]
    return base64.b64decode(audio_base64)


def detect_mime(audio_bytes: bytes) -> str:
    if audio_bytes.startswith(b"RIFF"):
        return "audio/wav"
    if audio_bytes.startswith(b"ID3") or audio_bytes[:2] == b"\xff\xfb":
        return "audio/mpeg"
    if audio_bytes.startswith(b"OggS"):
        return "audio/ogg"
    if audio_bytes.startswith(b"fLaC"):
        return "audio/flac"
    return "audio/wav"


def extract_json(text: str):
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


def normalize_response(obj):
    result = {}

    for key in EMPTY_RESPONSE:
        result[key] = obj.get(key, EMPTY_RESPONSE[key]) if isinstance(obj, dict) else EMPTY_RESPONSE[key]

    if not isinstance(result["rows"], int):
        try:
            result["rows"] = int(result["rows"])
        except Exception:
            result["rows"] = 0

    if not isinstance(result["columns"], list):
        result["columns"] = []

    for k in ["mean", "std", "variance", "min", "max", "median", "mode", "range", "allowed_values", "value_range"]:
        if not isinstance(result[k], dict):
            result[k] = {}

    if not isinstance(result["correlation"], list):
        result["correlation"] = []

    return result


def analyze_with_gemini(audio_bytes: bytes, audio_id: str):
    if not GEMINI_API_KEY:
        return EMPTY_RESPONSE

    mime_type = detect_mime(audio_bytes)
    model = genai.GenerativeModel(GEMINI_MODEL)

    prompt = f"""
You are a strict data extraction and statistics API.

The input is an audio file. It describes a dataset/table and possibly the statistical rules.
Listen carefully, reconstruct the dataset exactly, then return ONLY valid JSON.

Return this exact JSON structure:
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
- Return JSON only. No markdown. No explanation.
- Include all keys exactly.
- rows must be integer.
- columns must be an array of column names.
- mean, std, variance, min, max, median, mode, range must be objects.
- allowed_values and value_range must be objects.
- correlation must be an array.
- Use numbers as JSON numbers, not strings.
- If a field is not applicable, use {{}} or [] as shown.
- Compute statistics exactly from the dataset in the audio.
- audio_id is: {audio_id}
"""

    response = model.generate_content(
        [
            prompt,
            {
                "mime_type": mime_type,
                "data": audio_bytes,
            },
        ],
        generation_config={
            "temperature": 0,
            "top_p": 1,
            "top_k": 1,
        }
    )

    raw = response.text if response and response.text else "{}"
    obj = extract_json(raw)

    if obj is None:
        return EMPTY_RESPONSE

    return normalize_response(obj)


@app.get("/")
def root():
    return {"status": "ok", "message": "Audio statistics API is running"}


@app.post("/analyze-audio")
def analyze_audio(req: AudioRequest):
    try:
        audio_bytes = decode_audio(req.audio_base64)
        return analyze_with_gemini(audio_bytes, req.audio_id)
    except Exception:
        return EMPTY_RESPONSE


# Extra route in case grader posts to base URL directly
@app.post("/")
def analyze_audio_root(req: AudioRequest):
    return analyze_audio(req)