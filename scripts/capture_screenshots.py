"""Capture screenshots of all 9 Onboarding Studio screens.

Prerequisites:
    pip install playwright
    playwright install chromium

Usage:
    # Start the app first in another terminal:
    #   streamlit run app/onboarding/onboarding_app.py --server.port 8501
    #
    # Then run this script:
    python scripts/capture_screenshots.py

    # Or specify a custom URL / output directory:
    python scripts/capture_screenshots.py --url http://localhost:8501 --out screenshots/
"""

import argparse
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Capture Onboarding Studio screenshots")
    parser.add_argument("--url", default="http://localhost:8501", help="Streamlit app URL")
    parser.add_argument("--out", default="screenshots", help="Output directory")
    parser.add_argument("--width", type=int, default=1920, help="Viewport width")
    parser.add_argument("--height", type=int, default=1080, help="Viewport height")
    parser.add_argument("--wait", type=float, default=3.0, help="Seconds to wait for page render")
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: playwright not installed. Run:")
        print("  pip install playwright && playwright install chromium")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    screens = [
        (1, "step1_workspace"),
        (2, "step2_intake"),
        (3, "step3_schema"),
        (4, "step4_ground_truth"),
        (5, "step5_pipeline"),
        (6, "step6_evaluation"),
        (7, "step7_readiness"),
        (8, "step8_enablement"),
        (9, "step9_monitoring"),
    ]

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": args.width, "height": args.height})

        for step_num, name in screens:
            url = f"{args.url}/?step={step_num}"
            print(f"Capturing {name} ({url})...")
            page.goto(url, wait_until="networkidle")
            time.sleep(args.wait)

            # Scroll to top
            page.evaluate("window.scrollTo(0, 0)")
            time.sleep(0.5)

            filepath = out_dir / f"{name}.png"
            page.screenshot(path=str(filepath), full_page=True)
            print(f"  Saved: {filepath}")

        browser.close()

    print(f"\nAll {len(screens)} screenshots saved to {out_dir}/")


if __name__ == "__main__":
    main()
