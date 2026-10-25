"""Render the reviewed CV chapter pipelines from their editorial manifest.

Only original explanatory diagrams are generated here. Paper figures are kept
separately with their source, page, figure, extraction bounds and checksums.
"""
from pathlib import Path
from html import escape
import json,textwrap

ROOT=Path(__file__).resolve().parents[1]

def draw(title,question,rows):
    height=118+len(rows)*165
    desc='; '.join(name+': '+' → '.join(nodes) for name,nodes in rows)
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 {height}" role="img" aria-labelledby="title desc">',
         '<title id="title">'+escape(title)+'</title><desc id="desc">'+escape(desc)+'</desc>',
         '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#416474"/></marker></defs>',
         f'<rect width="1120" height="{height}" rx="14" fill="#f6f8fb"/>']
    for i,line in enumerate(textwrap.wrap(title,85)):
        out.append(f'<text x="30" y="{36+i*27}" font-family="Arial,sans-serif" font-size="23" font-weight="bold" fill="#173b50">{escape(line)}</text>')
    for r,(name,nodes) in enumerate(rows):
        y=112+r*165
        out.append(f'<text x="30" y="{y}" font-family="Arial,sans-serif" font-size="18" fill="#315a67">{escape(name)}</text>')
        for i,label in enumerate(nodes):
            x=30+i*275
            out.append(f'<rect x="{x}" y="{y+19}" width="235" height="94" rx="9" fill="{["#dceafa","#e1eee8","#fff0d8","#ede5f7"][i]}" stroke="#718b9b"/>')
            lines=textwrap.wrap(label,23,break_long_words=False)
            for j,line in enumerate(lines):
                out.append(f'<text x="{x+117.5}" y="{y+56+j*23-(len(lines)-1)*6}" text-anchor="middle" font-family="Arial,sans-serif" font-size="17" fill="#173b50">{escape(line)}</text>')
            if i<3:out.append(f'<path d="M{x+238},{y+66} H{x+266}" stroke="#416474" stroke-width="2" marker-end="url(#arrow)"/>')
    out.append('</svg>')
    return '\n'.join(out)+'\n'

def main():
    data=json.loads((ROOT/'editorial/cv-expansion.json').read_text(encoding='utf8'))
    for c in data['chapters']:
        path=ROOT/c['diagram_path'];path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(draw(c['title'],c['question'],c['diagram']),encoding='utf8',newline='\n')
    print('CV original diagrams:',len(data['chapters']))

if __name__=='__main__':main()
