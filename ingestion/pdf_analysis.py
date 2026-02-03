from __future__ import annotations

from typing import List, Tuple

import fitz
import numpy as np

from core.contracts import TriageDecision, TriageMetrics

LOW_TEXT_THRESHOLD = 50
HIGH_IMAGE_COVERAGE_THRESHOLD = 0.35
HIGH_LAYOUT_COMPLEXITY_THRESHOLD = 0.6


def analyze_page(page: fitz.Page) -> TriageDecision:
    text = page.get_text("text") or ""
    text_length = len(text.strip())
    page_area = float(page.rect.width * page.rect.height)
    text_density = (text_length / page_area) if page_area else 0.0

    image_coverage_ratio = _estimate_image_coverage(page)
    layout_complexity_score = _estimate_layout_complexity(page)

    metrics = TriageMetrics(
        text_length=text_length,
        text_density=text_density,
        image_coverage_ratio=image_coverage_ratio,
        layout_complexity_score=layout_complexity_score,
    )

    reason_codes: List[str] = []
    if text_length < LOW_TEXT_THRESHOLD:
        reason_codes.append("low_text")
    if image_coverage_ratio > HIGH_IMAGE_COVERAGE_THRESHOLD:
        reason_codes.append("high_image_coverage")
    if layout_complexity_score > HIGH_LAYOUT_COMPLEXITY_THRESHOLD:
        reason_codes.append("high_layout_complexity")

    decision = "di_required" if reason_codes else "native_only"
    return TriageDecision(metrics=metrics, decision=decision, reason_codes=reason_codes)


def _estimate_image_coverage(page: fitz.Page, zoom: float = 0.4) -> float:
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    non_white = np.any(img < 245, axis=2)
    return float(non_white.mean())


def _estimate_layout_complexity(page: fitz.Page) -> float:
    words = page.get_text("words")
    if not words:
        return 0.0
    line_keys = [(w[5], w[6]) for w in words]
    total_lines = len(set(line_keys))
    if total_lines == 0:
        return 0.0
    words_per_line: dict[Tuple[int, int], int] = {}
    for line_key in line_keys:
        words_per_line[line_key] = words_per_line.get(line_key, 0) + 1
    short_lines = sum(1 for count in words_per_line.values() if count <= 3)
    short_line_ratio = short_lines / total_lines

    page_area = float(page.rect.width * page.rect.height)
    line_density = total_lines / max(page_area / 100000.0, 1.0)
    density_score = min(1.0, line_density / 3.0)

    complexity = (0.6 * short_line_ratio) + (0.4 * density_score)
    return min(1.0, complexity)
