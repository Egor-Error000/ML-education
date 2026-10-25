"""Arrange browser screenshots for manual inspection; not course illustrations."""
from pathlib import Path
from PIL import Image,ImageDraw
root=Path(__file__).resolve().parents[2]
files=sorted((root/'.qa-results/browser/qa').glob('batch01-*.png'))
for n in range(0,len(files),4):
    sheet=Image.new('RGB',(1800,1900),'#dce3eb');draw=ImageDraw.Draw(sheet)
    for i,p in enumerate(files[n:n+4]):
        im=Image.open(p).convert('RGB');im.thumbnail((880,890))
        x=(i%2)*900+10;y=(i//2)*950+35
        sheet.paste(im,(x,y));draw.text((x,y-22),p.stem,fill='black')
    sheet.save(root/f'.qa-results/batch01-sheet-{n//4+1}.png')
print(len(files),'screenshots;', (len(files)+3)//4,'inspection sheets')
