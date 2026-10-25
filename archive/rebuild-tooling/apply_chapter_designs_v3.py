"""Apply the authored diagram narratives, remove repeated boilerplate, fix metadata."""
from pathlib import Path
import sys,json,re,os
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools'))
from edit_chapter import load,save,transform
from draw_revision_figures import Diagram
def main():
    cov=load();p=ROOT/'course.json';course=json.loads(p.read_text(encoding='utf8'))
    designs={}
    for line in (ROOT/'editorial/chapter-designs.txt').read_text(encoding='utf8').splitlines():
        if not line or line.startswith('#'):continue
        key,steps,caption=[s.strip() for s in line.split('|')]
        designs[key]={'steps':steps.split(';'),'caption':caption}
    assert set(designs)=={c['id'] for c in course['chapters']}
    by={c['id']:c for c in course['chapters']}
    for c in course['chapters']:
        spec=designs[c['id']]
        def edit(text):
            text=re.sub(r'Читайте схему по направлению стрелок:[^\n]+',spec['caption'],text)
            text=re.sub(r'Ответьте на вопрос: \*\*[^\n]+Обоснуйте ответ конкретным вычислением[^\n]+\n','',text)
            if not c['used_by']:
                use='[Итоговый конвейер признаков](<feat-pipeline.md>); применение к пространственным данным и проверка переноса на новые территории.' if c['id']=='feat-geography' else 'Завершение выбранной ветки: самостоятельный эксперимент по практике этой главы.'
                text=re.sub(r'\*\*Используют:\*\*[^\n]+','**Используют:** '+use,text)
            # Collapse only extra empty lines, not lists, formula lines, or code.
            text=re.sub(r'\n{3,}','\n\n',text)
            return text
        result=transform(c['path'],edit,'Убрана универсальная инструкция, добавлен предметный разбор схемы; метаданные конечной ветки уточнены.',cov)
        save(cov)
        # Regenerate the generic overview as a subject-specific flow. Detailed
        # architecture figures remain in the chapter and are not substituted.
        name=ROOT/c['figures'][0]
        if name.name not in ('arch-attention.svg',):
            d=Diagram(c['title'],spec['caption'],h=460)
            count=len(spec['steps']);width=188;gap=28;start=30
            for i,label in enumerate(spec['steps']):
                x=start+i*(width+gap)
                # Explicit line wrapping only affects captions inside this SVG.
                import textwrap
                wrapped='\n'.join(textwrap.wrap(label,21,break_long_words=False))
                d.box(x,145,width,105,wrapped,'#e4edf7' if i<count-1 else '#e3f2e7',16)
                if i:d.arrow([(x-gap,198),(x,198)])
            import textwrap
            d.text(30,320,'\n'.join(textwrap.wrap(spec['caption'],105,break_long_words=False)),17)
            d.save(name.name)
        # Registry enumerates actually used SVGs, including the new detailed ones.
        figures=[]
        for match in re.finditer(r'!\[[^\]]*\]\((?:<([^>]+)>|([^\n)]+))\)',result):
            raw=match[1] or match[2];asset=((ROOT/c['path']).parent/raw).resolve()
            if asset.suffix=='.svg':figures.append(asset.relative_to(ROOT).as_posix())
        c['figures']=list(dict.fromkeys(figures))
        if c['id']=='feat-geography':c['used_by']=['feat-pipeline']
    save(cov);p.write_text(json.dumps(course,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    # Markdown links cannot address build-only anchors. Point the coverage index
    # at generated HTML; source records remain precise in the sidecar.
    p=ROOT/'editorial/coverage.md';text=p.read_text(encoding='utf8')
    text=re.sub(r'\]\(<\.\./(?:0[1-5]-[^>]+\.md)#(u-[^>]+)>\)',r'](<../Читать конспекты.html#\1>)',text)
    p.write_text(text,encoding='utf8')
    print('Applied subject-specific designs:',len(designs))
if __name__=='__main__':main()
