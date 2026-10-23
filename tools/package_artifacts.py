"""Create checksums and a static, readable copy of the executed notebook."""
from pathlib import Path
import hashlib
import html
import json
import nbformat

root=Path(__file__).resolve().parents[1]
book=nbformat.read(root/'notebooks/Seattle_Parking_Reconstruction.ipynb',as_version=4)
assert all(c.get('execution_count') is not None for c in book.cells if c.cell_type=='code')
assert not any(o.output_type=='error' for c in book.cells for o in c.get('outputs',[]))
parts=[]
for cell in book.cells:
    if cell.cell_type=='markdown':
        parts.append('<section class="markdown"><pre>'+html.escape(cell.source)+'</pre></section>')
    else:
        parts.append('<section><div class="cell-label">Executed cell '+str(cell.execution_count)+'</div><pre class="code">'+html.escape(cell.source)+'</pre>')
        for out in cell.get('outputs',[]):
            data=out.get('data',{})
            if 'image/png' in data:
                parts.append('<img alt="Saved notebook figure" src="data:image/png;base64,'+data['image/png']+'">')
            elif 'text/html' in data:
                parts.append('<div class="output">'+data['text/html']+'</div>')
            elif 'text/plain' in data:
                parts.append('<pre>'+html.escape(data['text/plain'])+'</pre>')
            elif 'text' in out:
                parts.append('<pre>'+html.escape(out['text'])+'</pre>')
        parts.append('</section>')
page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Executed Seattle Parking Notebook</title><style>body{margin:0;background:#f6f7f5;color:#20352b;font:15px/1.6 system-ui}main{max-width:1100px;margin:auto;padding:35px 22px}a{color:#496a28}section{background:white;margin:20px 0;padding:22px;border:1px solid #d5dfd5;border-radius:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:10px 0}.markdown pre{font:inherit}.code{font:13px/1.6 ui-monospace,monospace;background:#edf2ed;padding:18px}.cell-label{font-size:12px;color:#607b65}.output{overflow:auto}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #d5dfd5;padding:7px}img{max-width:100%}</style></head><body><main><a href="../review.html">Back to project review</a><h1>Executed Seattle parking notebook</h1><p>Saved code, outputs and figures from the new reconstruction. This static view does not execute code in your browser.</p>'''+''.join(parts)+'</main></body></html>'
(root/'notebooks/Seattle_Parking_Reconstruction.html').write_text(page,encoding='utf-8')
files=[root/'experiment.py',root/'requirements-lock.txt',root/'data/processed/sample.csv.gz',root/'notebooks/Seattle_Parking_Reconstruction.ipynb']
files+=list((root/'models').glob('*.joblib'))
manifest={str(p.relative_to(root)).replace('\\','/'):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}
(root/'reports/artifact_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Packaged executed notebook HTML and checksums for',len(files),'artifacts.')
