"""
AI conversational assistant for ECG/PPG biomarker analysis.
Calls the Google Gemini REST API directly with httpx — no google-genai SDK
dependency, avoiding pydantic_core version conflicts in the Streamlit env.
"""
from __future__ import annotations

import json
from typing import Iterator

import httpx

GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-thinking-exp",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _fmt(v, decimals: int = 3) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def _risk_symbol(level: str) -> str:
    return {"normal": "✓", "borderline": "⚠", "abnormal": "✗", "unknown": "?"}.get(level, "?")


def _section_lines(section: dict) -> list[str]:
    lines = []
    for k, bm in section.items():
        if isinstance(bm, dict) and "name" in bm:
            sym = _risk_symbol(bm.get("risk_level", "unknown"))
            val = _fmt(bm.get("value"))
            unit = bm.get("unit", "")
            lines.append(f"  {sym} {bm['name']}: {val} {unit}")
    return lines


def build_context(session_state: dict) -> str:
    parts: list[str] = []

    sig_type = session_state.get("signal_type") or "Unknown"
    fs = session_state.get("sampling_rate") or "Unknown"
    parts.append("=== ECG/PPG Analysis Platform — Session Context ===")
    parts.append(f"Signal type : {sig_type}")
    parts.append(f"Sampling rate: {fs} Hz")

    report = session_state.get("biomarker_report")
    if report is None:
        parts.append("\n[No biomarker analysis has been run yet in this session.]")
        return "\n".join(parts)

    parts.append(f"Duration     : {report.get('duration_seconds', 'N/A')} s")
    parts.append(f"Peaks        : {report.get('n_peaks_detected', 'N/A')}")

    for section_key, label in [
        ("hrv_time",       "HRV Time Domain"),
        ("hrv_freq",       "HRV Frequency Domain"),
        ("hrv_nonlinear",  "HRV Nonlinear"),
        ("hrv_advanced",   "HRV Advanced (Kubios Scientific)"),
        ("autonomic",      "Autonomic Indices"),
        ("ecg_morphology", "ECG Morphology"),
        ("ppg_vascular",   "PPG Vascular"),
    ]:
        sec = report.get(section_key)
        if sec:
            parts.append(f"\n--- {label} ---")
            parts.extend(_section_lines(sec))

    arr = report.get("arrhythmia")
    if arr:
        parts.append("\n--- Arrhythmia Screening ---")
        parts.append(f"  AFib suspected : {arr.get('afib_suspected')}")
        parts.append(f"  Ectopic beats  : {arr.get('ectopic_beats')} ({arr.get('ectopic_ratio', 0)*100:.1f}%)")
        parts.append(f"  HR class       : {arr.get('heart_rate_class')}")
        if arr.get("afib_evidence"):
            parts.append("  Evidence: " + "; ".join(arr["afib_evidence"]))

    sq = report.get("signal_quality")
    if sq:
        parts.append("\n--- Signal Quality ---")
        parts.append(f"  SQI score  : {sq.get('overall_score', 'N/A'):.0f}/100 [{sq.get('quality_label', '')}]")
        parts.append(f"  SNR        : {_fmt(sq.get('snr_db'))} dB")

    ml = report.get("ml_anomaly")
    if ml:
        parts.append("\n--- ML Anomaly Detection ---")
        parts.append(f"  Anomalous  : {ml.get('is_anomalous')}")
        parts.append(f"  Score      : {_fmt(ml.get('anomaly_score'))}")
        if ml.get("flags"):
            parts.append("  Flags: " + "; ".join(ml["flags"]))

    if report.get("warnings"):
        parts.append("\n--- Analysis Warnings ---")
        for w in report["warnings"]:
            parts.append(f"  ⚠ {w}")

    return "\n".join(parts)


def build_system_prompt(context: str, lang: str = "it") -> str:
    if lang == "en":
        role = (
            "You are an expert biomedical AI assistant specialised in ECG and PPG signal analysis "
            "and interpretation of Heart Rate Variability (HRV) biomarkers. "
            "You support clinicians and researchers in understanding the clinical significance "
            "of the analysis results shown below.\n\n"
            "Guidelines:\n"
            "- Interpret biomarkers using Task Force 1996, ESC/AHA guidelines, and Kubios HRV references.\n"
            "- When a value is flagged ABNORMAL or BORDERLINE, explain the clinical meaning concisely.\n"
            "- Always remind the user that results are FOR RESEARCH USE ONLY and do not constitute a diagnosis.\n"
            "- Be concise but precise. Use bullet points when listing multiple findings.\n"
            "- If no analysis has been run yet, guide the user through the 5-step wizard.\n"
        )
    else:
        role = (
            "Sei un assistente AI biomedico esperto nell'analisi di segnali ECG e PPG "
            "e nell'interpretazione dei biomarker di Heart Rate Variability (HRV). "
            "Supporti clinici e ricercatori nel comprendere il significato clinico "
            "dei risultati mostrati di seguito.\n\n"
            "Linee guida:\n"
            "- Interpreta i biomarker secondo le linee guida Task Force 1996, ESC/AHA e Kubios HRV.\n"
            "- Quando un valore è ANOMALO o BORDERLINE, spiega il significato clinico in modo conciso.\n"
            "- Ricorda sempre che i risultati sono SOLO PER RICERCA e non costituiscono diagnosi.\n"
            "- Sii conciso ma preciso. Usa elenchi puntati per più osservazioni.\n"
            "- Se non è stata ancora eseguita un'analisi, guida l'utente attraverso i 5 step del wizard.\n"
        )
    return f"{role}\n\n{context}"


# ---------------------------------------------------------------------------
# Gemini REST client — no SDK, no pydantic dependency
# ---------------------------------------------------------------------------

class GeminiAssistant:
    """
    Calls the Gemini generateContent REST API directly using httpx.
    Avoids any pydantic/protobuf version conflicts with the Streamlit env.

    History format: [{"role": "user"|"model", "parts": [{"text": "..."}]}, ...]
    """

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self._key = api_key
        self.model = model

    def _url(self, method: str, stream: bool = False) -> str:
        suffix = "streamGenerateContent?alt=sse" if stream else "generateContent"
        return f"{_BASE}/{self.model}:{suffix}&key={self._key}" if stream else \
               f"{_BASE}/{self.model}:{suffix}?key={self._key}"

    def _build_body(self, history: list[dict], system_prompt: str) -> dict:
        return {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [
                {
                    "role": msg["role"],
                    "parts": [{"text": p["text"]} for p in msg["parts"]],
                }
                for msg in history
            ],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 2048,
            },
        }

    def stream(self, history: list[dict], system_prompt: str) -> Iterator[str]:
        """Yield text chunks via SSE streaming."""
        body = self._build_body(history, system_prompt)
        url = self._url("streamGenerateContent", stream=True)

        with httpx.Client(timeout=60.0) as client:
            with client.stream("POST", url, json=body) as resp:
                if resp.status_code != 200:
                    err = resp.read().decode()
                    raise RuntimeError(f"HTTP {resp.status_code}: {err[:300]}")
                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                        for cand in obj.get("candidates", []):
                            for part in cand.get("content", {}).get("parts", []):
                                text = part.get("text", "")
                                if text:
                                    yield text
                    except json.JSONDecodeError:
                        continue

    @staticmethod
    def validate_key(api_key: str, model: str = "gemini-2.0-flash") -> tuple[bool, str]:
        """Quick connectivity check. Returns (ok, error_message)."""
        url = f"{_BASE}/{model}:generateContent?key={api_key}"
        body = {
            "contents": [{"role": "user", "parts": [{"text": "Reply with exactly: OK"}]}],
            "generationConfig": {"maxOutputTokens": 5},
        }
        try:
            resp = httpx.post(url, json=body, timeout=15.0)
            if resp.status_code == 200:
                return True, ""
            err_body = resp.json()
            msg = err_body.get("error", {}).get("message", resp.text[:200])
            return False, msg
        except Exception as exc:
            return False, str(exc)
