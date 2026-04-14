#!/usr/bin/env python3
"""Convert report.md to PDF with embedded images."""
import markdown
from weasyprint import HTML
import os, base64, re

def embed_images(html_content, base_dir):
    def replace_img(match):
        src = match.group(1)
        img_path = os.path.join(base_dir, src)
        if os.path.exists(img_path):
            with open(img_path, 'rb') as f:
                data = base64.b64encode(f.read()).decode()
            ext = os.path.splitext(src)[1].lower().strip('.')
            mime = {'png': 'image/png', 'jpg': 'image/jpeg'}.get(ext, 'image/png')
            return f'src="data:{mime};base64,{data}"'
        return match.group(0)
    return re.sub(r'src="([^"]+)"', replace_img, html_content)

with open('report.md', 'r') as f:
    md = f.read()

html_body = markdown.markdown(md, extensions=['tables', 'fenced_code', 'codehilite', 'toc'])

html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
@page {{ size: A4; margin: 2cm; }}
body {{ font-family: 'Segoe UI', Tahoma, sans-serif; line-height: 1.6; color: #000; font-size: 11pt; }}
h1 {{ color: #000; border-bottom: 3px solid #000; padding-bottom: 10px; font-size: 22pt; }}
h2 {{ color: #000; border-bottom: 1px solid #999; padding-bottom: 5px; margin-top: 30px; font-size: 16pt; }}
h3 {{ color: #000; font-size: 13pt; }}
code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-family: Consolas, monospace; font-size: 10pt; color: #000; }}
pre {{ background: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 5px; font-size: 9pt; line-height: 1.4; }}
pre code {{ background: none; padding: 0; color: #d4d4d4; }}
table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 10pt; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
th {{ background: #333; color: white; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
img {{ max-width: 100%; height: auto; display: block; margin: 20px auto; border: 1px solid #ddd; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
blockquote {{ border-left: 4px solid #333; margin: 10px 0; padding: 10px 20px; background: #f5f5f5; color: #000; }}
hr {{ border: none; border-top: 2px solid #eee; margin: 30px 0; }}
</style></head><body>{html_body}
<div style="text-align:center;color:#999;font-size:9pt;margin-top:40px;border-top:1px solid #ddd;padding-top:10px;">
Jayant Batra — UCS645: Parallel & Distributed Computing — Assignment 4</div>
</body></html>"""

html = embed_images(html, '.')
HTML(string=html).write_pdf('Assignment_4_Report.pdf')
print("PDF generated: Assignment_4_Report.pdf")
