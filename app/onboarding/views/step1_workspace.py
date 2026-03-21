"""Step 1 — Workspace / Use Case Setup."""

import streamlit as st
from app.onboarding.mock_data import WORKSPACE
from app.onboarding.components.layout import (
    render_page_title, render_section_header, render_recommendation,
    render_summary_card, render_action_bar,
)


def render(current_step: int = 1):
    render_page_title(
        "Workspace & Use Case Setup",
        "Define the business context, ownership, and regulatory profile for this onboarding workspace.",
    )

    col_main, col_side = st.columns([3, 1])

    with col_main:
        render_section_header("Use Case Definition")
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("Use Case Name", value=WORKSPACE["name"], key="ws_name")
            st.text_input("Department / Business Unit", value=WORKSPACE["department"], key="ws_dept")
            st.selectbox("Jurisdiction / Region", [
                "United States — OCC Regulated",
                "United Kingdom — FCA Regulated",
                "European Union — ECB Regulated",
                "Canada — OSFI Regulated",
                "Asia-Pacific — Multi-jurisdiction",
            ], index=0, key="ws_jurisdiction")
        with c2:
            st.selectbox("Sensitivity Classification", [
                "Public", "Internal", "Confidential — No PII",
                "Confidential — PII Present", "Restricted",
            ], index=3, key="ws_sensitivity")
            st.selectbox("Regulatory Profile", [
                "SOX / OCC / FDIC", "MiFID II / FCA",
                "Basel III / IV", "GDPR / Privacy", "None",
            ], index=0, key="ws_regulatory")
            st.selectbox("Target Outcome Type", [
                "Field Extraction + Validation",
                "Document Classification Only",
                "Full Extract + Compare",
                "Compliance Screening",
                "Data Migration QA",
            ], index=0, key="ws_outcome")

        render_section_header("Ownership & Governance")
        c3, c4 = st.columns(2)
        with c3:
            st.text_input("Operations Owner", value=WORKSPACE["owner"], key="ws_owner")
            st.text_input("Approver", value=WORKSPACE["approver"], key="ws_approver")
        with c4:
            st.text_area("Reviewers", value="\n".join(WORKSPACE["reviewers"]), height=100, key="ws_reviewers")

        render_section_header("Workspace Metadata")
        c5, c6, c7 = st.columns(3)
        with c5:
            st.text_input("Workspace ID", value=WORKSPACE["workspace_id"], disabled=True, key="ws_id")
        with c6:
            st.text_input("Created", value=WORKSPACE["created_at"], disabled=True, key="ws_created")
        with c7:
            st.text_input("Business Unit", value=WORKSPACE["business_unit"], disabled=True, key="ws_bu")

    with col_side:
        render_recommendation(
            "Onboarding Mode: Guided",
            "Based on your regulatory profile (SOX / OCC) and sensitivity classification "
            "(Confidential — PII Present), we recommend <strong>Guided Onboarding</strong> with "
            "mandatory compliance review gates and full audit trail."
        )

        render_recommendation(
            "Evaluation Strategy",
            "For lending document extraction with regulatory requirements, we recommend "
            "<strong>Shadow deployment</strong> as the initial target — parallel processing "
            "with no downstream impact until quality gates are met.",
        )

        render_summary_card("Workspace Status", [
            ("Status", "Active — Setup", "#16a34a"),
            ("Step", "1 of 8"),
            ("Mode", "Guided Onboarding"),
            ("Workspace ID", WORKSPACE["workspace_id"]),
        ])

    render_action_bar(current_step)
