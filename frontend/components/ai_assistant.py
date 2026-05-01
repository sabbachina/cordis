"""
AI conversational assistant for ECG/PPG biomarker analysis.
Uses Google Gemini API (google-genai SDK >= 1.0).
"""
from __future__ import annotations

import json
from typing import Iterator, Optional

GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-thinking-exp",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

# ---------------------------------------------------------------------------
# Context builder — converts session-state data into a readable summary
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
    """
    Serialise the platform session state into a compact text block
    that Gemini can use as clinical context.
    """
    parts: list[str] = []

    sig_type = session_state.get("signal_type") or "Unknown"
    fs = session_state.get("sampling_rate") or "Unknown"
    parts.append(f"=== ECG/PPG Analysis Platform — Session Context ===")
    parts.append(f"Signal type : {sig_type}")
    parts.append(f"Sampling rate: {fs} Hz")

    report = session_state.get("biomarker_report")
    if report is None:
        parts.append("\n[No biomarker analysis has been run yet in this session.]")
        return "\n".join(parts)

    dur = report.get("duration_seconds", "N/A")
    n_peaks = report.get("n_peaks_detected", "N/A")
    parts.append(f"Duration     : {dur} s")
    parts.append(f"Peaks        : {n_peaks}")

    if report.get("hrv_time"):
        parts.append("\n--- HRV Time Domain ---")
        parts.extend(_section_lines(report["hrv_time"]))

    if report.get("hrv_freq"):
        parts.append("\n--- HRV Frequency Domain ---")
        parts.extend(_section_lines(report["hrv_freq"]))

    if report.get("hrv_nonlinear"):
        parts.append("\n--- HRV Nonlinear ---")
        parts.extend(_section_lines(report["hrv_nonlinear"]))

    if report.get("hrv_advanced"):
        parts.append("\n--- HRV Advanced (Kubios Scientific) ---")
        parts.extend(_section_lines(report["hrv_advanced"]))

    if report.get("autonomic"):
        parts.append("\n--- Autonomic Indices ---")
        parts.extend(_section_lines(report["autonomic"]))

    if report.get("ecg_morphology"):
        parts.append("\n--- ECG Morphology ---")
        parts.extend(_section_lines(report["ecg_morphology"]))

    if report.get("ppg_vascular"):
        parts.append("\n--- PPG Vascular ---")
        parts.extend(_section_lines(report["ppg_vascular"]))

    if report.get("arrhythmia"):
        arr = report["arrhythmia"]
        parts.append("\n--- Arrhythmia Screening ---")
        parts.append(f"  AFib suspected : {arr.get('afib_suspected')}")
        parts.append(f"  Ectopic beats  : {arr.get('ectopic_beats')} ({arr.get('ectopic_ratio', 0)*100:.1f}%)")
        parts.append(f"  HR class       : {arr.get('heart_rate_class')}")
        if arr.get("afib_evidence"):
            parts.append("  Evidence: " + "; ".join(arr["afib_evidence"]))

    if report.get("signal_quality"):
        sq = report["signal_quality"]
        parts.append("\n--- Signal Quality ---")
        parts.append(f"  SQI score  : {sq.get('overall_score', 'N/A'):.0f}/100 [{sq.get('quality_label', '')}]")
        parts.append(f"  SNR        : {_fmt(sq.get('snr_db'))} dB")

    if report.get("ml_anomaly"):
        ml = report["ml_anomaly"]
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
# Gemini client wrapper
# ---------------------------------------------------------------------------

class GeminiAssistant:
    """
    Thin wrapper around google-genai for multi-turn chat with streaming.
    History is stored externally (in st.session_state) as a list of dicts:
      [{"role": "user"|"model", "parts": [{"text": "..."}]}, ...]
    """

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        from google import genai  # deferred — only when needed
        self._client = genai.Client(api_key=api_key)
        self.model = model

    def stream(
        self,
        history: list[dict],
        system_prompt: str,
    ) -> Iterator[str]:
        """
        Yield text chunks for the latest user turn.
        *history* must already include the new user message as the last entry.
        """
        from google.genai import types

        contents = []
        for msg in history:
            contents.append(
                types.Content(
                    role=msg["role"],
                    parts=[types.Part(text=p["text"]) for p in msg["parts"]],
                )
            )

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.4,
            max_output_tokens=2048,
        )

        for chunk in self._client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config,
        ):
            if chunk.text:
                yield chunk.text

    @staticmethod
    def validate_key(api_key: str, model: str = "gemini-2.0-flash") -> tuple[bool, str]:
        """Quick connectivity check. Returns (ok, error_message)."""
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=api_key)
            resp = client.models.generate_content(
                model=model,
                contents="Reply with exactly: OK",
                config=types.GenerateContentConfig(max_output_tokens=5),
            )
            _ = resp.text
            return True, ""
        except Exception as exc:
            return False, str(exc)
