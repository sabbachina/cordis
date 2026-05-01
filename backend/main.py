import math
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from api.router import router


def _sanitize(obj):
    """Recursively replace nan/inf with None for JSON safety."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


class SafeJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        import json
        return json.dumps(_sanitize(content), ensure_ascii=False).encode("utf-8")


app = FastAPI(
    title="ECG/PPG Analysis API",
    description="Piattaforma per l'estrazione di digital biomarker clinici da segnali ECG e PPG",
    version="1.0.0",
    default_response_class=SafeJSONResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/")
def root():
    return {"message": "ECG/PPG Analysis Platform API", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}
