# styling.py

from pathlib import Path
import streamlit as st

CSS_FILE = Path(__file__).parent / "styles" / "app.css"


def configure_page():
    st.set_page_config(
        page_title="Japanese Speaking Practice",
        page_icon="JP",
        layout="wide"
    )


def init_theme_state():
    if "theme" not in st.session_state:
        st.session_state.theme = "day"


def toggle_theme():
    st.session_state.theme = (
        "night" if st.session_state.theme == "day" else "day"
    )


def load_css():
    if CSS_FILE.exists():
        st.markdown(
            f"<style>{CSS_FILE.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True
        )


def apply_theme():
    if st.session_state.theme == "night":
        st.markdown(
            """
            <style>
            :root {
                --bg-start: var(--night-bg-start);
                --bg-mid: var(--night-bg-mid);
                --bg-end: var(--night-bg-end);
                --text: var(--night-text);
                --subtext: var(--night-subtext);
                --border: var(--night-border);
                --brand: var(--night-brand);
                --brand-soft: var(--night-brand-soft);
                --panel: var(--night-panel);
                --result-bg: var(--night-result-bg);
                --timer-bg: var(--night-timer-bg);
                --timer-text: var(--night-timer-text);
                --sidebar-bg: var(--night-sidebar-bg);
                --sidebar-text: var(--night-sidebar-text);
                --button-bg: var(--night-button-bg);
                --button-border: var(--night-button-border);
                --button-hover: var(--night-button-hover);
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            :root {
                --bg-start: var(--day-bg-start);
                --bg-mid: var(--day-bg-mid);
                --bg-end: var(--day-bg-end);
                --text: var(--day-text);
                --subtext: var(--day-subtext);
                --border: var(--day-border);
                --brand: var(--day-brand);
                --brand-soft: var(--day-brand-soft);
                --panel: var(--day-panel);
                --result-bg: var(--day-result-bg);
                --timer-bg: var(--day-timer-bg);
                --timer-text: var(--day-timer-text);
                --sidebar-bg: var(--day-sidebar-bg);
                --sidebar-text: var(--day-sidebar-text);
                --button-bg: var(--day-button-bg);
                --button-border: var(--day-button-border);
                --button-hover: var(--day-button-hover);
            }
            </style>
            """,
            unsafe_allow_html=True,
        )