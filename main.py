"""
TSC Backend v5.1
Sistema de procesamiento de señales con IA
Con soporte para licencias Supabase, heartbeat y demo automática
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
    print("WARNING: Supabase no instalado - usando licencia por variable de entorno")

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tsc")

app = FastAPI(title="TSC Backend", version="5.1.0")

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
    log.info("Groq inicializado")
else:
    log.error("GROQ_API_KEY no configurada")
    groq_client = None

# Inicializar Supabase si esta disponible
supabase = None
if SUPABASE_AVAILABLE:
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    if supabase_url and supabase_key:
        try:
            supabase: Client = create_client(supabase_url, supabase_key)
            log.info("Supabase inicializado")
        except Exception as e:
            log.error(f"Error inicializando Supabase: {e}")
            supabase = None
    else:
        log.warning("Variables de Supabase no configuradas")

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
    version: str = "5.1.0"

class LicenseStatus(BaseModel):
    status: str
    expires_at: Optional[str] = None
    current_activations: int = 0
    max_activations: int = 5

class DemoLicenseRequest(BaseModel):
    metaquotes_id: str

# ================================================================
# FUNCIONES DE LICENCIA
# ================================================================

def get_license_from_supabase(license_key: str) -> Optional[dict]:
    """Obtiene una licencia de Supabase"""
    if not supabase:
        return None
    
    try:
        result = supabase.table("licenses")\
            .select("*")\
            .eq("key", license_key)\
            .execute()
        
        if result.data and len(result.data) > 0:
            return result.data[0]
        return None
    except Exception as e:
        log.error(f"Error obteniendo licencia de Supabase: {e}")
        return None

def check_license(license_key: str) -> bool:
    """Verifica la licencia (Supabase o variable de entorno)"""
    
    # Si Supabase esta disponible, usarlo
    if supabase:
        lic = get_license_from_supabase(license_key)
        
        if lic is None:
            log.warning(f"Licencia no encontrada en Supabase: {license_key}")
            # Fallback a variable de entorno
            allowed = os.environ.get("VALID_LICENSE_KEYS", "")
            return license_key in [k.strip() for k in allowed.split(",") if k.strip()]
        
        # Verificar estado
        if lic.get("status") != "active":
            log.warning(f"Licencia inactiva: {license_key} (status: {lic.get('status')})")
            return False
        
        # Verificar expiracion
        if lic.get("expires_at"):
            try:
                expiry_str = lic["expires_at"]
                if isinstance(expiry_str, str):
                    expiry = datetime.fromisoformat(expiry_str.replace("Z", "+00:00"))
                    if expiry.tzinfo is None:
                        expiry = expiry.replace(tzinfo=datetime.now().astimezone().tzinfo)
                    if datetime.now(expiry.tzinfo) > expiry:
                        log.warning(f"Licencia expirada: {license_key}")
                        try:
                            supabase.table("licenses")\
                                .update({"status": "expired"})\
                                .eq("key", license_key)\
                                .execute()
                        except:
                            pass
                        return False
            except Exception as e:
                log.error(f"Error verificando expiracion: {e}")
        
        # Verificar activaciones
        max_activations = lic.get("max_activations", 5)
        current = lic.get("current_activations", 0)
        if current >= max_activations:
            log.warning(f"Limite de activaciones alcanzado: {license_key}")
            return False
        
        # Actualizar ultimo uso
        try:
            supabase.table("licenses")\
                .update({"last_used_at": datetime.now().isoformat()})\
                .eq("key", license_key)\
                .execute()
        except Exception as e:
            log.error(f"Error actualizando last_used_at: {e}")
        
        return True
    
    # Fallback: variable de entorno
    allowed = os.environ.get("VALID_LICENSE_KEYS", "")
    return license_key in [k.strip() for k in allowed.split(",") if k.strip()]

def create_demo_license_for_metaquotes(metaquotes_id: str) -> dict:
    """
    Crea una licencia demo para un MetaQuotes ID.
    Retorna un dict con success, license_key, expires_at, status, message.
    """
    metaquotes_id = metaquotes_id.strip().upper()
    
    if not metaquotes_id or len(metaquotes_id) < 6:
        return {
            "success": False,
            "message": "Invalid MetaQuotes ID (must be at least 6 characters)"
        }
    
    # Crear la clave de licencia demo
    license_key = f"DEMO-{metaquotes_id}"
    
    # Si no hay Supabase, usar fallback
    if not supabase:
        expires_at = (datetime.now() + timedelta(days=3)).isoformat()
        return {
            "success": True,
            "license_key": license_key,
            "expires_at": expires_at,
            "status": "active",
            "message": "Demo created (fallback mode - no Supabase)"
        }
    
    try:
        # Verificar si ya existe
        result = supabase.table("licenses")\
            .select("*")\
            .eq("key", license_key)\
            .execute()
        
        if result.data:
            lic = result.data[0]
            return {
                "success": True,
                "license_key": license_key,
                "expires_at": lic.get("expires_at"),
                "status": lic.get("status"),
                "message": "Demo already exists"
            }
        
        # Crear nueva licencia demo (14 días)
        expires_at = (datetime.now() + timedelta(days=3)).isoformat()
        
        supabase.table("licenses").insert({
            "key": license_key,
            "status": "active",
            "customer_email": f"demo_{metaquotes_id}@tsc.app",
            "customer_name": f"Demo User {metaquotes_id}",
            "notes": f"Demo license for MetaQuotes ID: {metaquotes_id}",
            "plan": "demo",
            "expires_at": expires_at,
            "max_activations": 1
        }).execute()
        
        log.info(f"Demo license created: {license_key}")
        
        return {
            "success": True,
            "license_key": license_key,
            "expires_at": expires_at,
            "status": "active",
            "message": "Demo license created successfully"
        }
        
    except Exception as e:
        log.error(f"Error creating demo license: {e}")
        return {
            "success": False,
            "message": f"Error creating demo license: {str(e)}"
        }

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
        raise HTTPException(status_code=503, detail="AI service unavailable")
    
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
        log.info(f"AI response: {raw[:200]}...")
        
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            log.warning(f"Invalid JSON: {raw}")
            return ParseResponse(valid=False)
        
        # Validar
        if not data.get("valid"):
            return ParseResponse(valid=False)
        
        # Validacion estricta
        symbol = data.get("symbol")
        action = data.get("action")
        
        if symbol not in ALLOWED_SYMBOLS:
            log.info(f"Symbol not allowed: {symbol}")
            return ParseResponse(valid=False)
        
        if action not in ALLOWED_ACTIONS or action == "NONE":
            log.info(f"Action not allowed: {action}")
            return ParseResponse(valid=False)
        
        # Normalizar numeros
        for field in ("entry", "sl", "tp", "lot_override"):
            if not isinstance(data.get(field), (int, float)):
                data[field] = 0.0
        
        return ParseResponse(valid=True, signal=data)
        
    except Exception as e:
        log.error(f"Parse error: {e}")
        raise HTTPException(status_code=500, detail="Error processing signal")

@app.post("/heartbeat")
def heartbeat(req: HeartbeatRequest):
    """Recibe heartbeats del companion app"""
    
    if not check_license(req.license_key):
        raise HTTPException(status_code=403, detail="Invalid license")
    
    if supabase:
        try:
            supabase.table("licenses")\
                .update({
                    "last_heartbeat": datetime.now().isoformat(),
                    "processed_count": req.processed,
                    "error_count": req.errors,
                    "version": req.version
                })\
                .eq("key", req.license_key)\
                .execute()
        except Exception as e:
            log.error(f"Error updating heartbeat: {e}")
    
    return {"status": "ok", "timestamp": req.timestamp}

@app.get("/license_status")
def get_license_status(license_key: str):
    """Obtiene el estado de una licencia (sin verificar, para el companion)"""
    
    if supabase:
        lic = get_license_from_supabase(license_key)
        
        if lic is None:
            allowed = os.environ.get("VALID_LICENSE_KEYS", "")
            if license_key in [k.strip() for k in allowed.split(",") if k.strip()]:
                return {
                    "status": "active",
                    "expires_at": None,
                    "current_activations": 0,
                    "max_activations": 5,
                    "plan": "basic",
                    "message": "License valid (fallback mode)"
                }
            raise HTTPException(status_code=404, detail="License not found")
        
        status = lic.get("status", "unknown")
        
        if status == "active" and lic.get("expires_at"):
            try:
                expiry_str = lic["expires_at"]
                if isinstance(expiry_str, str):
                    expiry = datetime.fromisoformat(expiry_str.replace("Z", "+00:00"))
                    if expiry.tzinfo is None:
                        expiry = expiry.replace(tzinfo=datetime.now().astimezone().tzinfo)
                    if datetime.now(expiry.tzinfo) > expiry:
                        status = "expired"
                        try:
                            supabase.table("licenses")\
                                .update({"status": "expired"})\
                                .eq("key", license_key)\
                                .execute()
                        except:
                            pass
            except Exception as e:
                log.error(f"Error checking expiry: {e}")
        
        return {
            "status": status,
            "expires_at": lic.get("expires_at"),
            "current_activations": lic.get("current_activations", 0),
            "max_activations": lic.get("max_activations", 5),
            "plan": lic.get("plan", "basic"),
            "message": f"License status: {status}"
        }
    
    allowed = os.environ.get("VALID_LICENSE_KEYS", "")
    if license_key in [k.strip() for k in allowed.split(",") if k.strip()]:
        return {
            "status": "active",
            "expires_at": None,
            "current_activations": 0,
            "max_activations": 5,
            "plan": "basic",
            "message": "License valid (env mode)"
        }
    
    raise HTTPException(status_code=404, detail="License not found")

@app.post("/create_demo_license")
def create_demo_license(req: DemoLicenseRequest):
    """Crea una licencia demo para un MetaQuotes ID"""
    
    result = create_demo_license_for_metaquotes(req.metaquotes_id)
    
    if not result.get("success"):
        raise HTTPException(
            status_code=400,
            detail=result.get("message", "Error creating demo license")
        )
    
    return {
        "license_key": result.get("license_key"),
        "expires_at": result.get("expires_at"),
        "status": result.get("status"),
        "message": result.get("message")
    }

@app.get("/health")
def health():
    """Health check para Render"""
    return {
        "status": "ok",
        "version": "5.1.0",
        "supabase": "connected" if supabase else "not_configured",
        "groq": "connected" if groq_client else "not_configured"
    }

@app.get("/")
def root():
    """Raiz del servicio"""
    return {
        "service": "TSC Backend",
        "version": "5.1.0",
        "status": "online",
        "endpoints": [
            "/health",
            "/parse",
            "/heartbeat",
            "/license_status",
            "/create_demo_license"
        ]
    }
