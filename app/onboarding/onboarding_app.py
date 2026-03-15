"""Enterprise IDP Onboarding Studio — Main Entry Point.

Run with:
    streamlit run app/onboarding/onboarding_app.py
"""

import streamlit as st

st.set_page_config(
    page_title="IDP Onboarding Studio",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from app.onboarding.components.theme import inject_global_css
from app.onboarding.components.layout import render_header, render_stepper
from app.onboarding.pages import (
    step1_workspace,
    step2_intake,
    step3_schema,
    step4_ground_truth,
    step5_pipeline,
    step6_evaluation,
    step7_readiness,
    step8_enablement,
    step9_monitoring,
)

# ── Theme ────────────────────────────────────────────────────────────
inject_global_css()

# ── Navigation State ─────────────────────────────────────────────────
if "current_step" not in st.session_state:
    st.session_state.current_step = 1
if "view" not in st.session_state:
    st.session_state.view = "wizard"

# Allow URL query params to override step (for screenshots)
params = st.query_params
if "step" in params:
    try:
        step_val = int(params["step"])
        if 1 <= step_val <= 8:
            st.session_state.current_step = step_val
            st.session_state.view = "wizard"
        elif step_val == 9:
            st.session_state.view = "monitoring"
    except (ValueError, TypeError):
        pass

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Navigation")
    st.markdown("---")

    st.markdown("**Onboarding Wizard**")
    step_labels = {
        1: "1. Workspace Setup",
        2: "2. Document Intake",
        3: "3. Task & Schema",
        4: "4. Ground Truth",
        5: "5. Pipeline Settings",
        6: "6. Evaluation Plan",
        7: "7. Review & Readiness",
        8: "8. Production Enablement",
    }
    for num, label in step_labels.items():
        if st.button(label, key=f"nav_{num}", use_container_width=True):
            st.session_state.current_step = num
            st.session_state.view = "wizard"
            st.rerun()

    st.markdown("---")
    st.markdown("**Operations**")
    if st.button("📊 Post-Go-Live Monitoring", key="nav_monitoring", use_container_width=True):
        st.session_state.view = "monitoring"
        st.rerun()

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.72rem;color:#ABB2B9;margin-top:1rem">'
        "IDP Platform v2.0<br>© 2026 Enterprise Operations"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Page Dispatch ────────────────────────────────────────────────────
WIZARD_PAGES = {
    1: step1_workspace,
    2: step2_intake,
    3: step3_schema,
    4: step4_ground_truth,
    5: step5_pipeline,
    6: step6_evaluation,
    7: step7_readiness,
    8: step8_enablement,
}

render_header()

if st.session_state.view == "monitoring":
    step9_monitoring.render()
else:
    current = st.session_state.current_step
    render_stepper(current)
    page_module = WIZARD_PAGES.get(current, step1_workspace)
    page_module.render(current_step=current)
