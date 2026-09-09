"""
TSC Backend v3.0
Sistema de procesamiento de señales con IA
Con soporte para licencias Supabase y heartbeat
"""

import os
import json
import logging
from typing import Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

# Intentar importar Supabase (opcional)
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("⚠️ Supabase no instalado - usando licencia por variable de entorno")

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tsc")

app = FastAPI(title="TSC Backend", version="3.0.0")

# CORS para permitir conexiones desde el companion app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================================================
# CONFIGURACION
# ================================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# Inicializar Groq
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    log.error("GROQ_API_KEY no configurada")
    groq_client = None

# Inicializar Supabase si esta disponible
if SUPABASE_AVAILABLE:
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    if supabase_url and supabase_key:
        supabase: Client = create_client(supabase_url, supabase_key)
        log.info("✅ Supabase inicializado")
    else:
        SUPABASE_AVAILABLE = False
        log.warning("⚠️ Variables de Supabase no configuradas")
else:
    supabase = None

# ================================================================
# LISTAS BLANCAS
# ================================================================

ALLOWED_SYMBOLS = [
    "XAUUSD", "XAGUSD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
    "NZDUSD", "USDCAD", "USDCHF", "BTCUSD", "ETHUSD",
    "US30", "NAS100", "SP500",
]

ALLOWED_ACTIONS = [
    "BUY", "SELL", "BUY_LIMIT", "SELL_LIMIT", "BUY_STOP", "SELL_STOP",
    "CLOSE", "CLOSE_ALL", "MODIFY", "NONE",
]

# ================================================================
# PROMPT DE IA
# ================================================================

SYSTEM_PROMPT = f"""You are an expert trading signal extraction system. Your task is to parse raw, messy Telegram messages and convert them into a strict, structured trading signal that MetaTrader 5 can execute without errors.

You MUST be EXTREMELY TOLERANT to:
- Typos: "xauusd", "xau usd", "gold", "gld", "eur usd"
- Case sensitivity: "buy", "Buy", "BUY"
- Inconsistent formats: "at 2030.50", "2030.50", "en 2030.50"
- Abbreviations: "sl" (stop loss), "tp" (take profit), "lot" (lot size)
- Currency symbols: "$", "€", "£" before numbers
- Extra text: "Vamos a comprar", "Entrada larga", "Objetivo"
- Emojis and special characters: 🚀, 📈, 🔴, 🟢

**SYMBOL MAPPING:**
- XAUUSD: "gold", "xau", "xauusd", "oro", "gld"
- XAGUSD: "silver", "xag", "xagusd", "plata"
- EURUSD: "eur", "euro", "eurusd"
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

**ACTION MAPPING:**
- BUY: "buy", "comprar", "long", "call", "sube", "compra", "entrada larga"
- SELL: "sell", "vender", "short", "put", "baja", "venta", "entrada corta"
- BUY_LIMIT: "buy limit", "comprar limite", "lmt compra", "limite de compra"
- SELL_LIMIT: "sell limit", "vender limite", "lmt venta", "limite de venta"
- BUY_STOP: "buy stop", "comprar stop", "stp compra", "stop de compra"
- SELL_STOP: "sell stop", "vender stop", "stp venta", "stop de venta"
- CLOSE: "close", "cerrar", "salir", "cerrar posicion", "close position"
- CLOSE_ALL: "close all", "cerrar todo", "close all positions"
- MODIFY: "modify", "modificar", "cambiar sl", "mover sl", "actualizar sl"

**EXTRACTION RULES:**
1. **valid:** TRUE if message is a trading instruction. FALSE for greetings or analysis.
2. **symbol:** Extract using mapping above.
3. **action:** Extract using mapping above.
4. **entry:** Price if specified, 0.0 for market orders.
5. **sl:** Stop loss price, 0.0 if not specified.
6. **tp:** Take profit price, 0.0 if not specified.
7. **lot_override:** Lot size if mentioned, 0.0 if not.
8. **comment:** Short summary (max 40 chars).

**OUTPUT FORMAT (STRICT JSON):**
{{"valid": true/false, "symbol": "", "action": "", "entry": 0.0, "sl": 0.0, "tp": 0.0, "lot_override": 0.0, "comment": ""}}

Now process the following message and return ONLY the JSON:"""

# ================================================================
# MODELOS
# ================================================================

class ParseRequest(BaseModel):
    text: str
    license_key: str

class ParseResponse(BaseModel):
    valid: bool
    signal: Optional[dict] = None

class HeartbeatRequest(BaseModel):
    license_key: str
    timestamp: int
    status: str
    processed: int = 0
    errors: int = 0
    version: str = "3.0.0"

class LicenseStatus(BaseModel):
    status: str
    expires_at: Optional[str] = None
    current_activations: int = 0
    max_activations: int = 3

# ================================================================
# FUNCIONES DE LICENCIA
# ================================================================

def check_license(license_key: str) -> bool:
    """Verifica la licencia (Supabase o variable de entorno)"""
    
    # Si Supabase esta disponible, usarlo
    if SUPABASE_AVAILABLE and supabase:
        try:
            result = supabase.table("licenses")\
                .select("*")\
                .eq("key", license_key)\
                .execute()
            
            if not result.data:
                log.warning(f"Licencia no encontrada: {license_key}")
                return False
            
            license_data = result.data[0]
            
            # Verificar estado
            if license_data.get("status") != "active":
                log.warning(f"Licencia inactiva: {license_key}")
                return False
            
            # Verificar expiracion
            if license_data.get("expires_at"):
                expiry = datetime.fromisoformat(license_data["expires_at"].replace("Z", "+00:00"))
                if datetime.now() > expiry:
                    log.warning(f"Licencia expirada: {license_key}")
                    return False
            
            # Verificar activaciones
            max_activations = license_data.get("max_activations", 3)
            current = license_data.get("current_activations", 0)
            if current >= max_activations:
                log.warning(f"Limite de activaciones alcanzado: {license_key}")
                return False
            
            # Actualizar ultimo uso
            supabase.table("licenses")\
                .update({"last_used_at": datetime.now().isoformat()})\
                .eq("key", license_key)\
                .execute()
            
            return True
            
        except Exception as e:
            log.error(f"Error verificando licencia en Supabase: {e}")
            # Fallback a variable de entorno
    
    # Fallback: variable de entorno
    allowed = os.environ.get("VALID_LICENSE_KEYS", "")
    return license_key in [k.strip() for k in allowed.split(",") if k.strip()]

# ================================================================
# ENDPOINTS
# ================================================================

@app.post("/parse", response_model=ParseResponse)
def parse(req: ParseRequest):
    """Endpoint principal para procesar señales"""
    
    # Verificar licencia
    if not check_license(req.license_key):
        raise HTTPException(status_code=403, detail="Invalid or inactive license key")
    
    # Verificar Groq
    if not groq_client:
        raise HTTPException(status_code=503, detail="IA no disponible")
    
    # Procesar con IA
    try:
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
        log.info(f"Respuesta de IA: {raw[:200]}...")
        
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            log.warning(f"IA devolvio JSON invalido: {raw}")
            return ParseResponse(valid=False)
        
        # Validar
        if not data.get("valid"):
            return ParseResponse(valid=False)
        
        # Validacion estricta
        symbol = data.get("symbol")
        action = data.get("action")
        
        if symbol not in ALLOWED_SYMBOLS:
            log.info(f"Simbolo no permitido: {symbol}")
            return ParseResponse(valid=False)
        
        if action not in ALLOWED_ACTIONS or action == "NONE":
            log.info(f"Accion no permitida: {action}")
            return ParseResponse(valid=False)
        
        # Normalizar numeros
        for field in ("entry", "sl", "tp", "lot_override"):
            if not isinstance(data.get(field), (int, float)):
                data[field] = 0.0
        
        return ParseResponse(valid=True, signal=data)
        
    except Exception as e:
        log.error(f"Error en parse: {e}")
        raise HTTPException(status_code=500, detail="Error procesando la señal")

@app.post("/heartbeat")
def heartbeat(req: HeartbeatRequest):
    """Recibe heartbeats del companion app"""
    
    if not check_license(req.license_key):
        raise HTTPException(status_code=403, detail="Invalid license")
    
    # Actualizar en Supabase si esta disponible
    if SUPABASE_AVAILABLE and supabase:
        try:
            supabase.table("licenses")\
                .update({
                    "last_heartbeat": datetime.now().isoformat(),
                    "status": req.status,
                    "processed_count": req.processed,
                    "error_count": req.errors,
                    "version": req.version
                })\
                .eq("key", req.license_key)\
                .execute()
        except Exception as e:
            log.error(f"Error actualizando heartbeat: {e}")
    
    return {"status": "ok", "timestamp": req.timestamp}

@app.get("/license_status")
def get_license_status(license_key: str):
    """Obtiene el estado de una licencia"""
    
    if not check_license(license_key):
        raise HTTPException(status_code=403, detail="Invalid license")
    
    if SUPABASE_AVAILABLE and supabase:
        try:
            result = supabase.table("licenses")\
                .select("status, expires_at, current_activations, max_activations")\
                .eq("key", license_key)\
                .execute()
            
            if result.data:
                data = result.data[0]
                return LicenseStatus(
                    status=data.get("status", "unknown"),
                    expires_at=data.get("expires_at"),
                    current_activations=data.get("current_activations", 0),
                    max_activations=data.get("max_activations", 3)
                )
        except Exception as e:
            log.error(f"Error obteniendo estado: {e}")
    
    return {"status": "active", "expires_at": None, "current_activations": 1, "max_activations": 3}

@app.get("/health")
def health():
    """Health check para Render"""
    return {"status": "ok", "version": "3.0.0"}

@app.get("/")
def root():
    """Raiz del servicio"""
    return {
        "service": "TSC Backend",
        "version": "3.0.0",
        "status": "online",
        "endpoints": ["/health", "/parse", "/heartbeat", "/license_status"]
    }
