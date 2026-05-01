from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.router import router

app = FastAPI(
    title="ECG/PPG Analysis API",
    description="Piattaforma per l'estrazione di digital biomarker clinici da segnali ECG e PPG",
    version="1.0.0",
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
