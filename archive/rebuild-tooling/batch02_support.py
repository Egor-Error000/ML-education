"""Explicit rewrite operations for batch 02; normal builds never run this file."""
import json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from edit_chapter import load,save,digest,sections
from editorial_io import verify_span
def read(p):return json.loads((ROOT/p).read_text(encoding='utf8'))
def write(p,v):(ROOT/p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')

def apply(ident,title,question,body,mapping,example,practice,limits):
    registry=read('course.json');by={c['id']:c for c in registry['chapters']};chapter=by[ident]
    rel=chapter['path'];old=(ROOT/rel).read_text(encoding='utf8');cov=load()
    dest=sections(body)
    assert set(re.findall(r'^## (.+)$',old,re.M))==set(mapping)-{'__intro__'},(ident,'Every old H2 needs an explicit destination')
    for key,s in cov['spans'].items():
        if s['path']!=rel:continue
        assert verify_span(old,s),(ident,key,'stale before author edit')
    def link(key):
        import os
        path=os.path.relpath(ROOT/by[key]['path'],(ROOT/rel).parent).replace('\\','/')
        return f'[{by[key]["title"]}](<{path}>)'
    intro=f'# {title}\n\n**Уровень:** {chapter["level"]}.\n\n**Предпосылки:** '+', '.join(link(k) for k in chapter['prerequisites'])+'.\n\n'
    consumers=chapter.get('used_by',[])
    intro+='**Используют:** '+(', '.join(link(k) for k in consumers[:3]) if consumers else 'прикладная проверка выбранной ветки; переходы указаны в конце главы')+'.\n\n'
    intro+=f'**Учебный вопрос:** {question}\n\n'
    new=intro+body.strip()+'\n\n[Маршрут модуля](<../README.md>) · [Карта происхождения](<../../editorial/coverage.md>)\n'
    dest=sections(new)
    # A source span retains its old section identity. Authored mapping can split
    # a composite section into several destinations. No text matching is used.
    assignments={}
    for key,s in list(cov['spans'].items()):
        if s['path']!=rel:continue
        before='\n'.join(old.splitlines()[:s['start_line']])
        hs=re.findall(r'^## (.+)$',before,re.M)
        heading=hs[-1] if hs else '__intro__'
        targets=mapping.get(heading,mapping['__intro__'])
        if isinstance(targets,str):targets=[targets]
        assignments[key]=targets
        a,b,part=dest[targets[0]]
        s.update(start_line=a,end_line=b,sha256=digest(part),destination_section=targets[0],status='revised',review_status='pending',
            rationale='Пакет 02: цельная редактура раздела; исходные вопросы, вычисления и оговорки имеют явные назначения. Содержательная приёмка выполняется отдельно.')
    for u in cov['units']:
        if u.get('chapter')!=ident or u.get('provenance_only'):continue
        previous=list(u['targets']);titles=list(dict.fromkeys(t for k in previous for t in assignments[k]))
        targets=[]
        for i,t in enumerate(titles):
            key=u['anchor'] if i==0 else f't-batch02-{u["id"]}-{i}'
            a,b,part=dest[t]
            cov['spans'][key]=dict(path=rel,start_line=a,end_line=b,sha256=digest(part),destination_section=t,status='revised',review_status='pending',
                rationale='Составной исходный случай сопоставлен автором через явно заданные разделы; объединение и исправления требуют итогового чтения.')
            targets.append(key)
        u.update(targets=targets,status='semantic-review-pending',action='объединить с сохранением нюансов')
    (ROOT/rel).write_text(new,encoding='utf8',newline='\n');save(cov)
    chapter.update(title=title,question=question)
    refs=[a or b for a,b in re.findall(r'!\[[^\]]*\]\((?:<([^>]+\.svg)>|([^()\s]+\.svg))\)',new)]
    chapter['figures']=list(dict.fromkeys((ROOT/rel).parent.joinpath(ref).resolve().relative_to(ROOT).as_posix() for ref in refs))
    write('course.json',registry)
    scenarios=read('editorial/chapter-scenarios.json');s=scenarios['chapters'][ident]
    steps=[k for k in dest if k not in ['Практика и проверка']]
    s.update(status='authored — пакет 02, чтение отдельно',steps=steps,question=question,example=example,practice=practice,limitations=limits,
        diagram_narrative='Вход, промежуточные объекты, изменяемые параметры и выход показаны на предметной схеме и объяснены в тексте.')
    s['markdown']=f'**Учебный вопрос:** {question}\n\n**Результат:** объяснить перечисленные механизмы, воспроизвести расчёт и выбрать проверяемое вмешательство.\n\n**Уровень:** {chapter["level"]}. **Предпосылки:** '+', '.join(chapter['prerequisites'])+'\n\n**Потребители:** '+(', '.join(consumers) or 'завершение прикладной ветки')+'\n\n**Порядок объяснения:**\n\n'+''.join(f'{i}. {x}.\n' for i,x in enumerate(steps,1))+f'\n**Определения, формулы и предпосылки:** вводятся перед применением в указанном порядке; формы, фиксируемые величины и нормировки проверяются при чтении.\n\n**Разобранный пример:** {example}\n\n**Схемы и разбор:** {s["diagram_narrative"]}\n\n**Практика и проверка:** {practice}\n\n**Ограничения:** {limits}\n'
    write('editorial/chapter-scenarios.json',scenarios)
    cards=read('editorial/chapter-reviews.json');card=cards['chapters'][ident]
    card.update(text_sha256=digest(new),prose='rewritten',content_acceptance='open',remaining=['Итоговая сверка исходных случаев, вычислений и схем пакета 02.'])
    units=[u for u in cov['units'] if u.get('chapter')==ident and not u.get('provenance_only')]
    card.update(source_units=len(units),source_units_pending_semantic_review=len(units))
    write('editorial/chapter-reviews.json',cards)
    print(ident,len(units),'source units; pending acceptance')
