import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.i18n import t, render_language_selector
from components.ai_assistant import (
    GeminiAssistant,
    GEMINI_MODELS,
    build_context,
    build_system_prompt,
)

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")

# ---- Sidebar: language + Gemini config ----
render_language_selector()
st.sidebar.markdown("---")
st.sidebar.subheader(t("ai_sidebar_header"))

api_key = st.sidebar.text_input(
    t("ai_key_label"),
    type="password",
    placeholder=t("ai_key_placeholder"),
    help=t("ai_key_help"),
    key="gemini_api_key",
)

model_choice = st.sidebar.selectbox(
    t("ai_model_label"),
    GEMINI_MODELS,
    index=0,
    help=t("ai_model_help"),
    key="gemini_model",
)

if st.sidebar.button(t("ai_validate_btn"), key="validate_key_btn"):
    if not api_key:
        st.sidebar.error("❌ API Key vuota / empty.")
    else:
        with st.sidebar.spinner("..."):
            ok, err = GeminiAssistant.validate_key(api_key, model_choice)
        if ok:
            st.sidebar.success(t("ai_validate_ok", model=model_choice))
            st.session_state.gemini_key_valid = True
        else:
            st.sidebar.error(t("ai_validate_fail", err=err))
            st.session_state.gemini_key_valid = False

# ---- Main area ----
st.title(t("ai_page_title"))
st.markdown(t("ai_page_subtitle"))
st.markdown(t("ai_disclaimer"))
st.markdown("---")

if not api_key:
    st.info(t("ai_key_missing"))
    st.stop()

# Warn if no report is available
report = st.session_state.get("biomarker_report")
if report is None:
    st.warning(t("ai_no_report_warn"))

# Build context from current session state
context = build_context(dict(st.session_state))
lang = st.session_state.get("lang", "it")
system_prompt = build_system_prompt(context, lang)

with st.expander(t("ai_context_expander"), expanded=False):
    st.code(context, language="text")

# ---- Chat history ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": str, "parts": [{"text": str}]}

# Clear button
col_clear, _ = st.columns([1, 5])
with col_clear:
    if st.button(t("ai_clear_btn"), key="clear_chat"):
        st.session_state.chat_history = []
        st.success(t("ai_clear_confirm"))
        st.rerun()

# Suggested questions (shown when chat is empty)
if not st.session_state.chat_history:
    st.markdown(f"**{t('ai_suggestions_header')}**")
    suggestions = [t(f"ai_q{i}") for i in range(1, 6)]
    cols = st.columns(len(suggestions))
    for i, (col, q) in enumerate(zip(cols, suggestions)):
        if col.button(q, key=f"sugg_{i}"):
            st.session_state.chat_history.append({"role": "user", "parts": [{"text": q}]})
            st.rerun()

# Render existing conversation
for msg in st.session_state.chat_history:
    role = msg["role"]
    text = msg["parts"][0]["text"]
    avatar = "🧑‍💻" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(text)

# Chat input
user_input = st.chat_input(t("ai_input_placeholder"))

if user_input:
    # Append user message
    st.session_state.chat_history.append({"role": "user", "parts": [{"text": user_input}]})

    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    # Stream model response
    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        full_response = ""
        try:
            assistant = GeminiAssistant(api_key=api_key, model=model_choice)
            with st.spinner(t("ai_thinking")):
                for chunk in assistant.stream(
                    history=st.session_state.chat_history,
                    system_prompt=system_prompt,
                ):
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
        except Exception as exc:
            full_response = t("ai_stream_err", err=exc)
            placeholder.error(full_response)

    # Save model turn
    if full_response:
        st.session_state.chat_history.append(
            {"role": "model", "parts": [{"text": full_response}]}
        )
