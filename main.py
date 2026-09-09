"""
TSC Backend
-----------
Receives raw Telegram message text from the companion app, uses an LLM
(Groq) to turn it into a strict structured trading signal, validates it,
and returns it. This is the piece that gives you a real edge over
regex-based competitors: it tolerates typos, abbreviations, and messy
human formatting instead of breaking on anything unexpected.

Run:
    pip install -r requirements.txt
    cp .env.example .env   # fill in your values
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
import json
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tsc")

app = FastAPI(title="TSC Backend")

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# Symbols you actually support - keep this tight and explicit.
# The model is instructed to only ever pick from this list.
ALLOWED_SYMBOLS = [
    "XAUUSD", "XAGUSD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
    "NZDUSD", "USDCAD", "USDCHF", "BTCUSD", "ETHUSD",
    "US30", "NAS100", "SP500",
]

ALLOWED_ACTIONS = [
    "BUY", "SELL", "BUY_LIMIT", "SELL_LIMIT", "BUY_STOP", "SELL_STOP",
    "CLOSE", "MODIFY", "NONE",
]

SYSTEM_PROMPT = f"""You extract structured trading signals from raw, messy
Telegram messages written by human signal providers. They frequently
contain typos, inconsistent formatting, emojis, and shorthand.

Return ONLY a single JSON object, no prose, no markdown fences, matching
exactly this schema:
{{
  "valid": boolean,               // true only if this message is an actionable trade signal
  "symbol": string,                // one of: {", ".join(ALLOWED_SYMBOLS)}
  "action": string,                // one of: {", ".join(ALLOWED_ACTIONS)}
  "entry": number,                 // 0 if market order / not specified
  "sl": number,                    // 0 if not specified
  "tp": number,                    // nearest/first take-profit; 0 if not specified
  "lot_override": number,          // 0 unless the message explicitly states lot size
  "comment": string                // short human-readable summary, max 40 chars
}}

Rules:
- If the message is chit-chat, a recap, an image caption with no new
  instruction, or anything not actionable, set "valid": false and
  "action": "NONE".
- Tolerate typos in symbol names (e.g. "GOLD", "XAUUSD", "gld" all mean
  XAUUSD; "GBPUSD", "gbp usd", "cable" all mean GBPUSD).
- If the message says to move stop loss to break-even or update an
  existing trade, use action "MODIFY" and put the new sl/tp in those
  fields (0 for the one not mentioned).
- If it says to close a trade, use action "CLOSE".
- Never invent numbers that are not present or clearly implied in the
  message. If a field isn't specified, use 0.
- Output raw JSON only.
"""


class ParseRequest(BaseModel):
    text: str
    license_key: str


class ParseResponse(BaseModel):
    valid: bool
    signal: Optional[dict] = None


def check_license(license_key: str) -> bool:
    """MVP license check. Replace with a real Supabase lookup:

        from supabase import create_client
        sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
        row = sb.table("licenses").select("*").eq("key", license_key).execute()
        return bool(row.data) and row.data[0]["status"] == "active"

    For now, checks against a comma-separated allow-list in the env so
    you can test end-to-end before wiring up Supabase.
    """
    allowed = os.environ.get("VALID_LICENSE_KEYS", "")
    return license_key in [k.strip() for k in allowed.split(",") if k.strip()]


@app.post("/parse", response_model=ParseResponse)
def parse(req: ParseRequest):
    if not check_license(req.license_key):
        raise HTTPException(status_code=403, detail="Invalid or inactive license key")

    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": req.text},
        ],
        response_format={"type": "json_object"},
    )

    raw = completion.choices[0].message.content
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("Model returned non-JSON: %s", raw)
        return ParseResponse(valid=False)

    if not data.get("valid"):
        return ParseResponse(valid=False)

    # Hard validation - never trust the model blindly.
    if data.get("symbol") not in ALLOWED_SYMBOLS:
        log.info("Rejected signal, symbol not allowed: %s", data.get("symbol"))
        return ParseResponse(valid=False)
    if data.get("action") not in ALLOWED_ACTIONS or data.get("action") == "NONE":
        return ParseResponse(valid=False)

    for numeric_field in ("entry", "sl", "tp", "lot_override"):
        if not isinstance(data.get(numeric_field), (int, float)):
            data[numeric_field] = 0

    return ParseResponse(valid=True, signal=data)


@app.get("/health")
def health():
    return {"status": "ok"}
