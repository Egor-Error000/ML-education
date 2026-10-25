"""Explicit author operations for the ten-card batch; never a normal build step."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from edit_chapter import ROOT,load,save,rewrite,sections,digest

def apply(chapter,new,mapping,groups,extras=None):
    c=load();rel='05-language-models/02-bert-encoders/'+chapter+'.md'
    rewrite(rel,new,mapping,'Пакет 01: сквозная редактура, явная карта отдельных исходных случаев, сохранение вычислений и ограничений; приёмка отдельно от написания.',c);save(c)
    parts=sections(new);extras=extras or {}
    dest={ident:[title]+extras.get(ident,[]) for title,ids in groups.items() for ident in ids.split()}
    units=[u for u in c['units'] if u.get('chapter')==chapter and not u.get('provenance_only')]
    assert set(dest)=={u['id'] for u in units},(set(dest)^{u['id'] for u in units})
    for u in units:
        targets=[]
        for i,title in enumerate(dest[u['id']]):
            a,b,body=parts[title];key=u['anchor'] if i==0 else f"t-batch1-{u['id']}-{i}"
            c['spans'][key]={'path':rel,'start_line':a,'end_line':b,'sha256':digest(body),'destination_section':title,'status':'revised','review_status':'pending','rationale':'Исходная единица прочитана и сопоставлена конкретному разделу; исправления и объединения требуют итоговой содержательной приёмки.'}
            targets.append(key)
        u.update(targets=targets,status='semantic-review-pending',action='объединить с сохранением нюансов')
    save(c)
    p=ROOT/'course.json';registry=json.loads(p.read_text(encoding='utf8'))
    for ch in registry['chapters']:
        if ch['id']==chapter:ch['title']=new.splitlines()[0][2:]
    p.write_text(json.dumps(registry,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
