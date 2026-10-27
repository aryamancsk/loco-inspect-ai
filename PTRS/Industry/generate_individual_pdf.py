import os
import subprocess
from pathlib import Path
import markdown_it
import pypdf

SCRIPT_DIR = Path(__file__).resolve().parent
MD_PATH = SCRIPT_DIR / "INDIVIDUAL_PROJECT_REPORT.md"
HTML_PATH = SCRIPT_DIR / "INDIVIDUAL_PROJECT_REPORT.html"
PDF_PATH = SCRIPT_DIR / "INDIVIDUAL_PROJECT_REPORT.pdf"
ROOT_PDF = SCRIPT_DIR.parent.parent / "INDIVIDUAL_PROJECT_REPORT.pdf"

md_text = MD_PATH.read_text(encoding="utf-8")
parser = markdown_it.MarkdownIt("commonmark").enable("table").enable("strikethrough")
body_html = parser.render(md_text)

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Individual Project Report — NUS-ISS</title>
    <style>
        @page {{
            size: A4;
            margin: 14mm 14mm 14mm 14mm;
            @bottom-right {{
                content: "Page " counter(page) " of " counter(pages);
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 8pt;
                color: #64748b;
            }}
        }}

        body {{
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Helvetica, Arial, sans-serif;
            color: #0f172a;
            line-height: 1.34;
            font-size: 8.8pt;
            background: #ffffff;
            margin: 0;
            padding: 0;
        }}

        .top-banner {{
            border-bottom: 2px solid #1e3a8a;
            padding-bottom: 4px;
            margin-bottom: 8px;
        }}

        .inst-title {{
            font-size: 10pt;
            font-weight: 800;
            color: #1e3a8a;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            margin: 0 0 2px 0;
        }}

        .doc-sub {{
            font-size: 8pt;
            color: #475569;
            font-weight: 600;
            margin: 0;
        }}

        h1 {{
            font-size: 13pt;
            font-weight: 800;
            color: #0f172a;
            margin: 6px 0 6px 0;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 8.5pt;
            margin: 6px 0 10px 0;
        }}

        th, td {{
            padding: 4px 8px;
            border: 1px solid #cbd5e1;
            text-align: left;
        }}

        th {{
            background: #f1f5f9;
            color: #1e293b;
            font-weight: 600;
            width: 25%;
        }}

        td {{
            background: #ffffff;
        }}

        h2 {{
            font-size: 10pt;
            font-weight: 700;
            color: #1e3a8a;
            border-bottom: 1px solid #cbd5e1;
            padding-bottom: 2px;
            margin-top: 10px;
            margin-bottom: 5px;
            page-break-after: avoid;
        }}

        p, ul, ol {{
            margin-top: 0;
            margin-bottom: 5px;
            text-align: justify;
        }}

        li {{
            margin-bottom: 2px;
        }}

        strong {{
            color: #0f172a;
        }}
    </style>
</head>
<body>
    <div class="top-banner">
        <div class="inst-title">National University of Singapore • Institute of Systems Science (NUS-ISS)</div>
        <div class="doc-sub">Graduate Certificate in Pattern Recognition Systems — Practice Module Individual Report</div>
    </div>
    {body_html}
</body>
</html>
"""

HTML_PATH.write_text(html_content, encoding="utf-8")

edge_path = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
if not edge_path.exists():
    edge_path = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")

cmd = [
    str(edge_path),
    "--headless=new",
    "--disable-gpu",
    "--no-pdf-header-footer",
    f"--print-to-pdf={PDF_PATH}",
    str(HTML_PATH.resolve()),
]
subprocess.run(cmd, check=True)

if PDF_PATH.exists():
    ROOT_PDF.write_bytes(PDF_PATH.read_bytes())
    reader = pypdf.PdfReader(str(PDF_PATH))
    print(f"[OK] Generated {PDF_PATH} ({PDF_PATH.stat().st_size / 1024:.1f} KB) - Page count: {len(reader.pages)}")
