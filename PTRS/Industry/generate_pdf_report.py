import base64
import os
import re
import subprocess
import sys
from pathlib import Path
import markdown_it

SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_MD = SCRIPT_DIR / "FINAL_PROJECT_REPORT.md"
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
HTML_OUT = SCRIPT_DIR / "FINAL_PROJECT_REPORT.html"
PDF_OUT = SCRIPT_DIR / "FINAL_PROJECT_REPORT.pdf"
ROOT_PDF_OUT = SCRIPT_DIR.parent.parent / "FINAL_PROJECT_REPORT.pdf"


def img_to_base64(path: Path) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    ext = path.suffix.lower().replace(".", "")
    mime = f"image/{ext}" if ext in ["png", "jpeg", "jpg", "webp"] else "image/png"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def embed_all_images(html_text: str, base_dir: Path) -> str:
    """Finds all <img src="..."> tags and replaces local file paths with base64 data URIs."""
    def replacer(match):
        full_tag = match.group(0)
        src = match.group(1)
        if src.startswith("data:") or src.startswith("http://") or src.startswith("https://"):
            return full_tag

        # Clean query strings if any
        clean_src = src.split("?")[0]
        candidate_paths = [
            (base_dir / clean_src).resolve(),
            (OUTPUTS_DIR / clean_src).resolve(),
            (OUTPUTS_DIR / Path(clean_src).name).resolve(),
            (base_dir / "outputs" / clean_src).resolve(),
        ]

        for p in candidate_paths:
            if p.exists() and p.is_file():
                b64 = img_to_base64(p)
                print(f"  [Embed Image] {clean_src} -> {p.name} ({p.stat().st_size / 1024:.1f} KB)")
                return full_tag.replace(src, b64)

        print(f"  [WARN] Image not found for: {src}")
        return full_tag

    pattern = r'<img\s+[^>]*?src=["\']([^"\']+)["\'][^>]*?>'
    return re.sub(pattern, replacer, html_text)


def build_html_report():
    md_content = REPORT_MD.read_text(encoding="utf-8")

    # Double check for dollar signs
    dollar_count = md_content.count("$")
    if dollar_count > 0:
        print(f"[WARN] Found {dollar_count} dollar signs in markdown! Removing them...")
        md_content = md_content.replace("$", "")

    # Render Markdown to HTML using markdown-it
    md_parser = (
        markdown_it.MarkdownIt("commonmark", {"html": True})
        .enable("table")
        .enable("strikethrough")
    )
    rendered_body = md_parser.render(md_content)

    # Embed all local images as base64
    rendered_body = embed_all_images(rendered_body, SCRIPT_DIR)

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>LOCO-Inspect AI — NUS-ISS Practice Module Final Report</title>
    <style>
        @page {{
            size: A4;
            margin: 18mm 16mm 18mm 16mm;
            @bottom-right {{
                content: counter(page);
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 8.5pt;
                color: #64748b;
            }}
        }}

        * {{
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Helvetica, Arial, sans-serif;
            color: #0f172a;
            line-height: 1.5;
            font-size: 9.5pt;
            background: #ffffff;
            margin: 0;
            padding: 0;
        }}

        /* Header / Banner */
        .header-banner {{
            border-bottom: 3px solid #1e3a8a;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}

        .nus-title {{
            font-size: 12.5pt;
            font-weight: 800;
            color: #1e3a8a;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin: 0 0 3px 0;
        }}

        .nus-sub {{
            font-size: 9pt;
            color: #475569;
            font-weight: 600;
            margin: 0 0 8px 0;
        }}

        h1 {{
            font-size: 16.5pt;
            font-weight: 800;
            color: #0f172a;
            line-height: 1.25;
            margin: 14px 0 6px 0;
            border: none;
        }}

        .meta-line {{
            font-size: 9pt;
            color: #334155;
            background: #f1f5f9;
            padding: 6px 10px;
            border-radius: 5px;
            margin-bottom: 16px;
            border-left: 4px solid #3b82f6;
        }}

        h2 {{
            font-size: 12pt;
            font-weight: 700;
            color: #1e3a8a;
            border-bottom: 1.5px solid #cbd5e1;
            padding-bottom: 3px;
            margin-top: 22px;
            margin-bottom: 8px;
            page-break-after: avoid;
            break-after: avoid;
        }}

        h3 {{
            font-size: 10.5pt;
            font-weight: 700;
            color: #1e293b;
            margin-top: 14px;
            margin-bottom: 5px;
            page-break-after: avoid;
            break-after: avoid;
        }}

        h4 {{
            font-size: 9.5pt;
            font-weight: 600;
            color: #334155;
            margin-top: 10px;
            margin-bottom: 4px;
            page-break-after: avoid;
            break-after: avoid;
        }}

        p, ul, ol {{
            margin-top: 0;
            margin-bottom: 7px;
            text-align: justify;
        }}

        li {{
            margin-bottom: 2px;
        }}

        /* Tables */
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 8.5pt;
            margin: 10px 0 14px 0;
            page-break-inside: avoid;
            break-inside: avoid;
        }}

        th, td {{
            padding: 5px 7px;
            border: 1px solid #cbd5e1;
            text-align: left;
        }}

        th {{
            background-color: #1e293b;
            color: #ffffff;
            font-weight: 600;
        }}

        tr:nth-child(even) {{
            background-color: #f8fafc;
        }}

        /* Visual Figures */
        .figure-container {{
            margin: 12px 0 14px 0;
            text-align: center;
            page-break-inside: avoid;
            break-inside: avoid;
        }}

        .figure-row {{
            display: flex;
            gap: 12px;
            justify-content: center;
            align-items: flex-start;
            margin: 12px 0 14px 0;
            page-break-inside: avoid;
            break-inside: avoid;
        }}

        .figure-item {{
            flex: 1;
            text-align: center;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            page-break-inside: avoid;
            break-inside: avoid;
        }}

        .figure-container img, .figure-item img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            display: block;
            margin: 0 auto;
            image-rendering: -webkit-optimize-contrast;
            image-rendering: crisp-edges;
        }}

        .caption {{
            font-size: 8pt;
            color: #475569;
            margin-top: 5px;
            margin-bottom: 2px;
            text-align: center;
            line-height: 1.35;
        }}

        .caption strong {{
            color: #1e293b;
        }}

        /* Code & Pre */
        pre {{
            background: #0f172a;
            color: #f8fafc;
            padding: 8px 12px;
            border-radius: 5px;
            font-size: 8pt;
            font-family: 'Consolas', 'Courier New', monospace;
            overflow-x: auto;
            margin: 8px 0;
            page-break-inside: avoid;
            break-inside: avoid;
            line-height: 1.4;
        }}

        code {{
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 8.5pt;
            background: #f1f5f9;
            color: #0f172a;
            padding: 1px 4px;
            border-radius: 3px;
        }}

        pre code {{
            background: transparent;
            color: inherit;
            padding: 0;
        }}

        blockquote {{
            border-left: 3px solid #3b82f6;
            margin: 8px 0;
            padding-left: 10px;
            color: #475569;
            font-style: italic;
        }}

        hr {{
            border: none;
            border-top: 1px solid #e2e8f0;
            margin: 18px 0;
        }}

        .page-break {{
            page-break-before: always;
            break-before: always;
        }}

        .appendix-section {{
            background: #fafaf9;
            border: 1px dashed #d6d3d1;
            padding: 12px 16px;
            border-radius: 6px;
            margin-top: 18px;
            page-break-inside: avoid;
            break-inside: avoid;
        }}
    </style>
</head>
<body>
    <div class="header-banner">
        <div class="nus-title">National University of Singapore • Institute of Systems Science (NUS-ISS)</div>
        <div class="nus-sub">Graduate Certificate in Pattern Recognition Systems — Practice Module Final Deliverable</div>
    </div>

    {rendered_body}
</body>
</html>
"""

    HTML_OUT.write_text(html_template, encoding="utf-8")
    print(f"[OK] Generated HTML report at: {HTML_OUT}")
    return HTML_OUT


def convert_html_to_pdf(html_path: Path) -> Path:
    edge_paths = [
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    ]
    browser_bin = next((p for p in edge_paths if p.exists()), None)
    if not browser_bin:
        raise FileNotFoundError("Could not find Edge or Chrome executable for PDF generation.")

    print(f"Using browser: {browser_bin}")
    cmd = [
        str(browser_bin),
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF_OUT}",
        str(html_path.resolve()),
    ]
    subprocess.run(cmd, check=True)

    if not PDF_OUT.exists():
        raise RuntimeError("PDF output file was not created.")

    # Also copy to workspace root
    ROOT_PDF_OUT.write_bytes(PDF_OUT.read_bytes())
    print(f"[OK] Generated PDF report ({PDF_OUT.stat().st_size / 1024:.1f} KB) at:")
    print(f"  1. {PDF_OUT}")
    print(f"  2. {ROOT_PDF_OUT}")
    return PDF_OUT


if __name__ == "__main__":
    h = build_html_report()
    p = convert_html_to_pdf(h)
