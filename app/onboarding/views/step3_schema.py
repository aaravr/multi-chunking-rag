"""Step 3 — Task & Schema Definition."""

import streamlit as st
from app.onboarding.mock_data import SCHEMA_FIELDS
from app.onboarding.components.layout import (
    render_page_title, render_section_header, render_recommendation,
    render_badge, render_summary_card, render_action_bar,
)


def render(current_step: int = 3):
    render_page_title(
        "Task & Schema Definition",
        "Define the extraction task, target schema, field validation rules, and reference data requirements.",
    )

    col_main, col_side = st.columns([3, 1])

    with col_main:
        render_section_header("Task Mode")
        c1, c2, c3, c4 = st.columns(4)
        modes = [
            ("📋", "Classify", "Categorize documents by type and subtype", False),
            ("🔍", "Extract Fields", "Extract structured fields from documents", True),
            ("✓", "Validate Fields", "Verify pre-extracted values against source", False),
            ("⇄", "Compare", "Compare field values across document versions", False),
        ]
        for col, (icon, title, desc, selected) in zip([c1, c2, c3, c4], modes):
            with col:
                cls = "selected" if selected else ""
                st.markdown(f"""
                <div class="mode-card {cls}">
                    <div class="mode-icon">{icon}</div>
                    <div class="mode-title">{title}</div>
                    <div class="mode-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

        render_section_header("Extraction Schema")
        st.markdown(
            '<p style="color:#666666;font-size:0.8rem">12 fields defined &nbsp;&bull;&nbsp; '
            '5 critical &nbsp;&bull;&nbsp; 3 with reference resolution</p>',
            unsafe_allow_html=True,
        )

        rows_html = ""
        for f in SCHEMA_FIELDS:
            crit_variant = {
                "Critical": "fail", "High": "warn", "Medium": "info", "Low": "pending",
            }.get(f["criticality"], "pending")
            crit_badge = render_badge(f["criticality"], crit_variant)
            req = "✓" if f["required"] else "—"
            ref = '<span style="color:#E60000">🔗</span>' if f["reference_resolver"] else ""
            rows_html += f"""
            <tr>
                <td style="font-weight:500;font-family:monospace;font-size:0.78rem">{f['name']}</td>
                <td>{f['type']}</td>
                <td style="text-align:center">{req}</td>
                <td>{crit_badge}</td>
                <td style="font-size:0.75rem;color:#666666">{f['validation']}</td>
                <td style="text-align:center">{ref}</td>
                <td style="font-size:0.75rem;color:#666666;max-width:180px">{f['description']}</td>
            </tr>"""

        st.markdown(f"""
        <table class="clean-table">
            <thead><tr>
                <th>Field Name</th><th>Type</th><th style="text-align:center">Req</th>
                <th>Criticality</th><th>Validation Rule</th><th style="text-align:center">Ref</th>
                <th>Description</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c5, c6, c7 = st.columns(3)
        with c5:
            st.button("+ Add Field", key="add_field", use_container_width=True)
        with c6:
            st.button("Import Schema (JSON/CSV)", key="import_schema", use_container_width=True)
        with c7:
            st.button("Copy from Template", key="copy_template", use_container_width=True)

    with col_side:
        render_recommendation(
            "Schema Complexity: Medium-High",
            "This schema has <strong>12 fields</strong> with <strong>5 critical fields</strong> and "
            "<strong>3 fields requiring reference resolution</strong> (entity lookup, collateral type mapping, "
            "risk rating normalization). We recommend enabling the MCP reference data server."
        )

        render_summary_card("Schema Summary", [
            ("Total Fields", "12"),
            ("Required", "8"),
            ("Critical", "5", "#dc2626"),
            ("Reference Lookup", "3"),
            ("Field Types", "6 distinct"),
        ])

        render_recommendation(
            "Template Available",
            "A pre-built schema template for <strong>Commercial Lending Annual Reviews</strong> is available. "
            "It covers 10 of your 12 fields with pre-configured validation rules."
        )

    render_action_bar(current_step)
