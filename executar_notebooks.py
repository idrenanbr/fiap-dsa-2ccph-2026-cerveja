"""Reexecuta os três notebooks existentes e registra validação. Use na raiz do projeto."""
from pathlib import Path
import json
import nbformat
from nbclient import NotebookClient

root=Path(__file__).resolve().parent
report=[]
for name in ['CP1_corrigido.ipynb','CP2_corrigido.ipynb','CP3_corrigido.ipynb']:
    path=root/name
    nb=nbformat.read(path,as_version=4)
    client=NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(root)}})
    try: client.execute()
    finally: nbformat.write(nb,path)
    cells=[c for c in nb.cells if c.cell_type=='code']
    errors=[o for c in cells for o in c.get('outputs',[]) if o.output_type=='error']
    assert all(c.execution_count is not None for c in cells) and not errors
    report.append({'notebook':name,'celulas_codigo':len(cells),'executadas':len(cells),'erros':len(errors)})
    print(name,'OK',flush=True)
(root/'artifacts/notebook_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
