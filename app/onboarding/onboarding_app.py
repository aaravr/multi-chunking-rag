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
from app.onboarding.components.layout import (
    render_top_nav, render_breadcrumb, render_sidebar_workspace,
)
from app.onboarding.views import (
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

# ── Top Navigation Bar ───────────────────────────────────────────────
render_top_nav(active_section="Workspaces")

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

if st.session_state.view == "monitoring":
    step9_monitoring.render()
else:
    current = st.session_state.current_step

    # ── 3-column layout: sidebar nav | breadcrumb + content | (built into content) ──
    sidebar_col, content_col = st.columns([1, 5])

    with sidebar_col:
        render_sidebar_workspace()

        if st.session_state.view == "wizard":
            st.markdown(
                f'<div class="sidebar-step-badge">Step {current}</div>',
                unsafe_allow_html=True,
            )

        step_nav = {
            1: "Workspace",
            2: "Documents",
            3: "Schema",
            4: "Ground Truth",
            5: "Pipeline",
            6: "Evaluation",
            7: "Review",
            8: "Enablement",
        }
        for num, label in step_nav.items():
            is_active = (num == current)
            if st.button(label, key=f"nav_{num}", use_container_width=True,
                         type="primary" if is_active else "secondary"):
                st.session_state.current_step = num
                st.session_state.view = "wizard"
                st.rerun()

        st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
        if st.button("Monitoring", key="nav_monitoring", use_container_width=True):
            st.session_state.view = "monitoring"
            st.rerun()

        # Progress indicator
        st.markdown(f"""
        <div style="margin-top:1.5rem;padding:0 0.25rem">
            <div style="font-size:0.68rem;color:#94a3b8;margin-bottom:0.3rem">PROGRESS</div>
            <div class="progress-bar-track" style="height:6px">
                <div class="progress-bar-fill" style="width:{int((current/8)*100)}%;background:#3b82f6"></div>
            </div>
            <div style="font-size:0.68rem;color:#94a3b8;margin-top:0.2rem">{current} of 8 steps</div>
        </div>
        """, unsafe_allow_html=True)

    with content_col:
        render_breadcrumb(current)
        page_module = WIZARD_PAGES.get(current, step1_workspace)
        page_module.render(current_step=current)
