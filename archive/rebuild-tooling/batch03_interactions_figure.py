"""Connect the field-aware factors diagram to its explanation."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools'))
from edit_chapter import load,save,transform
def op(t):
 old='![Взаимодействия признаков, FM и FFM: входы, преобразования и выходы]'
 new='![Ручной ключ пары, общий фактор FM и зависящий от поля фактор FFM]'
 assert old in t;t=t.replace(old,new,1)
 old='FM заменяет отдельный вес каждой пары скалярным произведением факторов. FFM дополнительно учитывает поле второго признака; это меняет число параметров и способ разделения статистики между парами.'
 new='Слева каждая пара получает свой вес. В середине FM использует один общий вектор A для взаимодействий с днём и товаром; справа FFM выбирает разные векторы A для этих двух полей. Дальше проверим, что даёт разделение факторов, и отдельно рассмотрим новую пару и новый ID.'
 assert old in t;return t.replace(old,new,1)
c=load();transform('01-data-features/02-features/feat-interactions.md',op,
 'Пакет 03: схема показывает отдельные веса, общие факторы и факторы по полю; содержательная приёмка ожидается.',c);save(c)
p=ROOT/'editorial/illustrations.json';data=json.loads(p.read_text(encoding='utf8'))
entry={'path':'оформление/course/feat-interactions.svg','kind':'original-course-diagram',
 'external_figure':'none','authors':'Авторская схема курса','license':'Лицензия проекта; внешние рисунки не копируются.',
 'production':'tools/draw_batch03_features.py','chapters':['feat-interactions'],'checked_on':'2026-10-28',
 'source_of_mechanism':'Сравнение факторных правил и полей разобрано в главе; исходная работа FFM приведена рядом.',
 'changes':'Предметная схема отдельных пар, общих факторов FM и факторов FFM для полей партнёров.'}
data['figures']=[x for x in data['figures'] if x['path']!=entry['path']];data['figures'].append(entry)
p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
print('Interaction figure updated; review remains open.')
