from __future__ import annotations

import asyncio
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
DEFAULT_MD = ROOT / "Phase2_Final_Report.md"
THEME_CSS = ROOT / "report_theme.css"
TEMPLATE_HTML = ROOT / "report_template.html"
OUT_HTML = ROOT / "Phase2_Final_Report.themed.html"
OUT_PDF = ROOT / "Phase2_Final_Report.themed.pdf"


def _fail(msg: str) -> None:
    print(f"[error] {msg}")
    raise SystemExit(1)


def _read(path: Path) -> str:
    if not path.exists():
        _fail(f"Missing required file: {path}")
    return path.read_text(encoding="utf-8")


def build_html(markdown_path: Path) -> str:
    try:
        import markdown  # type: ignore
    except Exception:
        _fail(
            "Python package 'markdown' is not installed.\n"
            "Install with: py -m pip install markdown"
        )

    md_text = _read(markdown_path)
    css = _read(THEME_CSS)
    template = _read(TEMPLATE_HTML)

    html_body = markdown.markdown(
        md_text,
        extensions=[
            "fenced_code",
            "tables",
            "toc",
            "sane_lists",
            "admonition",
        ],
    )

    title = markdown_path.stem
    return (
        template.replace("{{title}}", title)
        .replace("{{css}}", css)
        .replace("{{content}}", html_body)
    )


async def html_to_pdf(html_path: Path, pdf_path: Path) -> None:
    try:
        from playwright.async_api import async_playwright  # type: ignore
    except Exception:
        _fail(
            "Python package 'playwright' is not installed.\n"
            "Install with: py -m pip install playwright\n"
            "Then install Chromium: py -m playwright install chromium"
        )

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(html_path.resolve().as_uri(), wait_until="networkidle")
        await page.pdf(
            path=str(pdf_path),
            format="A4",
            print_background=True,
            margin={
                "top": "12mm",
                "right": "10mm",
                "bottom": "12mm",
                "left": "10mm",
            },
        )
        await browser.close()


def main() -> None:
    md_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_MD
    if not md_path.exists():
        _fail(f"Markdown file not found: {md_path}")

    print(f"[info] Building themed HTML from: {md_path.name}")
    html = build_html(md_path)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"[ok] Wrote HTML: {OUT_HTML.name}")

    print("[info] Rendering PDF with Chromium...")
    asyncio.run(html_to_pdf(OUT_HTML, OUT_PDF))
    print(f"[ok] Wrote PDF: {OUT_PDF.name}")


if __name__ == "__main__":
    main()
