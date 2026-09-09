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

SYSTEM_PROMPT = f"""You are an expert trading signal extraction system. Your task is to parse raw, messy Telegram messages and convert them into a strict, structured trading signal that MetaTrader 5 can execute without errors.

You MUST be EXTREMELY TOLERANT to:
- Typos: "xauusd", "xau usd", "gold", "gld", "eur usd", "cable" (for GBPUSD)
- Case sensitivity: "buy", "Buy", "BUY"
- Inconsistent formats: "at 2030.50", "2030.50", "en 2030.50", "price 2030.50"
- Abbreviations: "sl" (stop loss), "tp" (take profit), "lot" (lot size), "stp" (stop), "lmt" (limit)
- Currency symbols: "$", "€", "£" before numbers
- Extra text: "Vamos a comprar", "Entrada larga", "Objetivo", "Stop loss", "Take profit"
- Emojis and special characters: 🚀, 📈, 🔴, 🟢

**SYMBOL MAPPING (BE FLEXIBLE):**
- XAUUSD: "gold", "xau", "xauusd", "oro", "gld"
- XAGUSD: "silver", "xag", "xagusd", "plata"
- EURUSD: "eur", "euro", "eurusd", "cable" (note: cable is GBP, but be tolerant)
- GBPUSD: "gbp", "libra", "gbpusd", "cable"
- USDJPY: "jpy", "yen", "usdjpy"
- AUDUSD: "aud", "australian", "audusd"
- NZDUSD: "nzd", "new zealand", "nzdusd"
- USDCAD: "cad", "canadian", "usdcad"
- USDCHF: "chf", "swiss", "usdchf"
- BTCUSD: "btc", "bitcoin", "btcusd"
- ETHUSD: "eth", "ethereum", "ethusd"
- US30: "dow", "us30", "dow jones"
- NAS100: "nas", "nas100", "nasdaq"
- SP500: "s&p", "sp500", "s&p 500"

**ACTION MAPPING (BE FLEXIBLE):**
- BUY: "buy", "comprar", "long", "call", "sube", "compra", "entrada larga"
- SELL: "sell", "vender", "short", "put", "baja", "venta", "entrada corta"
- BUY_LIMIT: "buy limit", "comprar limite", "lmt compra", "limite de compra"
- SELL_LIMIT: "sell limit", "vender limite", "lmt venta", "limite de venta"
- BUY_STOP: "buy stop", "comprar stop", "stp compra", "stop de compra"
- SELL_STOP: "sell stop", "vender stop", "stp venta", "stop de venta"
- CLOSE: "close", "cerrar", "salir", "cerrar posicion", "close position", "cerrar todo"
- MODIFY: "modify", "modificar", "cambiar sl", "mover sl", "actualizar sl", "nuevo sl"

**EXTRACTION RULES (CRITICAL):**
1. **valid:** TRUE if the message appears to be a trading instruction. FALSE if it's a greeting, market analysis without an order, profit/loss report, or anything not actionable.
2. **symbol:** Extract the trading symbol using the mapping above. If you're unsure, choose the most likely one.
3. **action:** Extract the action using the mapping above. If it's a market order (no entry price specified), use "BUY" or "SELL".
4. **entry:** 
   - If price is specified (e.g., "at 2030.50", "2030.50", "price 2030.50"), extract it as a number.
   - If it's a market order (e.g., "buy gold now", "sell eur"), set to 0.0.
   - If multiple prices are mentioned, the entry is usually the first number.
5. **sl (Stop Loss):** 
   - Extract the number following "sl", "stop loss", "stop", "stoploss".
   - If not specified, set to 0.0.
6. **tp (Take Profit):** 
   - Extract the number following "tp", "take profit", "target", "profit".
   - If not specified, set to 0.0.
7. **lot_override:** 
   - Extract if the message mentions "lot", "lote", "tamaño", "size", "volume".
   - If not specified, set to 0.0 (the EA will calculate it).
8. **comment:** 
   - A short summary (max 40 chars) of the signal for MT5.
   - Example: "BUY GOLD 2030.5", "SELL EUR 1.1050"

**OUTPUT FORMAT (STRICT):**
- Output ONLY a valid JSON object.
- NO prose, NO markdown fences, NO explanations outside the JSON.
- Use double quotes (") for keys and strings.
- Numbers must be numbers, not strings.

**EXAMPLES OF INPUT AND EXPECTED OUTPUT:**

Input: "BUY XAUUSD at 2030.50 sl 2020.00 tp 2050.00"
Output: {{"valid": true, "symbol": "XAUUSD", "action": "BUY", "entry": 2030.5, "sl": 2020.0, "tp": 2050.0, "lot_override": 0.0, "comment": "BUY XAUUSD 2030.5"}}

Input: "SELL EURUSD 1.1050 SL 1.1100 TP 1.0950"
Output: {{"valid": true, "symbol": "EURUSD", "action": "SELL", "entry": 1.105, "sl": 1.11, "tp": 1.095, "lot_override": 0.0, "comment": "SELL EURUSD 1.105"}}

Input: "Buy gold 2030.5 sl 2020 tp 2050 lot 0.5"
Output: {{"valid": true, "symbol": "XAUUSD", "action": "BUY", "entry": 2030.5, "sl": 2020.0, "tp": 2050.0, "lot_override": 0.5, "comment": "BUY XAUUSD 2030.5"}}

Input: "close xauusd now"
Output: {{"valid": true, "symbol": "XAUUSD", "action": "CLOSE", "entry": 0.0, "sl": 0.0, "tp": 0.0, "lot_override": 0.0, "comment": "CLOSE XAUUSD"}}

Input: "modify gold sl 2015 tp 2060"
Output: {{"valid": true, "symbol": "XAUUSD", "action": "MODIFY", "entry": 0.0, "sl": 2015.0, "tp": 2060.0, "lot_override": 0.0, "comment": "MODIFY XAUUSD"}}

Input: "Hola, ¿cómo están? Buen mercado hoy."
Output: {{"valid": false, "symbol": "", "action": "NONE", "entry": 0.0, "sl": 0.0, "tp": 0.0, "lot_override": 0.0, "comment": ""}}

Input: "Vamos a comprar XAUUSD en 2030.50, stop en 2020 y objetivo en 2050"
Output: {{"valid": true, "symbol": "XAUUSD", "action": "BUY", "entry": 2030.5, "sl": 2020.0, "tp": 2050.0, "lot_override": 0.0, "comment": "BUY XAUUSD 2030.5"}}

Input: "SHORT EURUSD 1.1050 STOP 1.1100 TARGET 1.0950"
Output: {{"valid": true, "symbol": "EURUSD", "action": "SELL", "entry": 1.105, "sl": 1.11, "tp": 1.095, "lot_override": 0.0, "comment": "SELL EURUSD 1.105"}}

Input: "BTCUSD buy limit at 40000 sl 39000 tp 42000"
Output: {{"valid": true, "symbol": "BTCUSD", "action": "BUY_LIMIT", "entry": 40000.0, "sl": 39000.0, "tp": 42000.0, "lot_override": 0.0, "comment": "BUY_LIMIT BTCUSD 40000"}}

Now, process the following user message and return ONLY the JSON object:"""


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
