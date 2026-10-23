"""Execute the notebook in the isolated project kernel."""
from pathlib import Path
import os
import nbformat
from nbclient import NotebookClient

root=Path(__file__).resolve().parents[1]
os.environ['JUPYTER_PATH']=str(root/'.jupyter/share/jupyter')
os.environ['MPLCONFIGDIR']=str(root/'tmp/matplotlib')
path=root/'notebooks/Seattle_Parking_Project.ipynb'
book=nbformat.read(path,as_version=4)
NotebookClient(book,timeout=600,kernel_name='seattle-parking',resources={'metadata':{'path':str(root)}}).execute()
nbformat.write(book,path)
print('Executed all notebook cells without errors:',path)
