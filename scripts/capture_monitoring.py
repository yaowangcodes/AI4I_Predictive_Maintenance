"""Optional screenshot helper: pip install playwright==1.56.0 (uses installed Chrome)."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["normal", "drift"], required=True)
    parser.add_argument("--chrome-path", default="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    args = parser.parse_args()
    evidence = json.loads((ROOT / "artifacts/monitoring" / f"demo-latest-{args.mode}.json").read_text())
    destination = ROOT / "docs/screenshots" / f"monitoring-{args.mode}.png"
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.chrome_path, headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 2000}, device_scale_factor=1)
        page.goto(evidence["dashboard_url"] + "&kiosk", wait_until="networkidle")
        page.get_by_text("DRIFT" if args.mode == "drift" else "NORMAL", exact=True).first.wait_for(timeout=30000)
        page.wait_for_timeout(6000)
        print(page.locator("body").inner_text()[:12000])
        page.screenshot(path=str(destination), full_page=True)
        browser.close()
    print(f"Saved real Grafana screenshot: {destination}")


if __name__ == "__main__":
    main()
