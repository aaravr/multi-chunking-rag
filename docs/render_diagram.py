"""Render architecture-diagram.html to JPG using Playwright."""
from playwright.sync_api import sync_playwright
from pathlib import Path
from PIL import Image

html_path = Path(__file__).parent / "architecture-diagram.html"
png_path = Path(__file__).parent / "architecture-diagram.png"
jpg_path = Path(__file__).parent / "architecture-diagram.jpg"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1920, "height": 1280})
    page.goto(f"file://{html_path.resolve()}")
    page.wait_for_timeout(1000)  # let fonts load

    # Screenshot the canvas element
    canvas = page.locator(".canvas")
    canvas.screenshot(path=str(png_path), type="png")
    browser.close()

# Convert PNG to JPG with high quality
img = Image.open(png_path)
img = img.convert("RGB")
img.save(jpg_path, "JPEG", quality=95)
png_path.unlink()  # remove intermediate PNG

print(f"Saved: {jpg_path}")
print(f"Size: {jpg_path.stat().st_size / 1024:.0f} KB")
