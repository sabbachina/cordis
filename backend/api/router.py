from fastapi import APIRouter
from api.endpoints import signals, analysis

router = APIRouter()
router.include_router(signals.router)
router.include_router(analysis.router)
