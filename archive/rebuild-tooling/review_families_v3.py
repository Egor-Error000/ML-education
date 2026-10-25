"""Explicit review of the 23 architecture-family source units."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from edit_chapter import ROOT,load,save,sections,digest
c=load();rel='05-language-models/01-foundations/lm-families.md'
groups={
'Три семейства и доступная информация':'''20abdfaa1c44976236 8b95d7ea26cf69a00a 8a4b4890dc710f5abe 30445ef84e04fcec79 d630c0dc861da64dd6''',
'От маски к задаче обучения':'''616966f0384ed8b762 7e6f7d72ea108ae542 149dd9d2ddcea30f76''',
'Перекрёстное внимание: кто спрашивает и кто передаёт':'''dfb4e308178bb51b34 a4200a35167b80094a 8239ed363f03f976b7 f7068995609eb9c07e 8b917f2a34746ea085 074fd53d4e23248e13 6cb0a4d9cba5be56f6 a50a83c38b98e694fc 74c006f11f53fe89a9''',
'Два запроса к трём источникам: полный расчёт':'''b51eda5dcea7c3f4c6 41af9f19766359ed4a e9431dd78ba1c5cb8d e47719fc0aeb37f160 c2bf96aa6c048f8a29''',
'Слой кодировщика–декодера и проверка':'''a7fc116c773f51bf89'''}
dest={ident:[title] for title,ids in groups.items() for ident in ids.split()}
dest['8b95d7ea26cf69a00a'].append('От маски к задаче обучения')
dest['8a4b4890dc710f5abe'].append('Перекрёстное внимание: кто спрашивает и кто передаёт')
parts=sections((ROOT/rel).read_text(encoding='utf8'))
units=[u for u in c['units'] if u.get('chapter')=='lm-families' and not u.get('provenance_only')]
assert set(dest)=={u['id'] for u in units}
for u in units:
    targets=[]
    for i,title in enumerate(dest[u['id']]):
        a,b,body=parts[title];key=u['anchor'] if i==0 else f"t-lm-families-{u['id']}-{i}"
        c['spans'][key]={'path':rel,'start_line':a,'end_line':b,'sha256':digest(body),'destination_section':title,'status':'revised','review_status':'verified',
          'rationale':'Прочитана исходная единица; сверены доступная информация, прямоугольные формы, оба расчёта, маска PAD и градиент в источник. Самопроверка ошибки явно отложена до её определения.',
          'evidence':'editorial/review-evidence.md#families'}
        targets.append(key)
    u.update(targets=targets,status='semantic-reviewed',action='объединить с сохранением нюансов')
save(c)
