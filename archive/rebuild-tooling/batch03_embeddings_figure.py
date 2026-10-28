"""Connect the numerical lookup example to its original diagram."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools'))
from edit_chapter import load,save,transform
def op(t):
 old='![Обучаемые embeddings и перенос представлений: входы, преобразования и выходы]'
 new='![Пакет A, A, B выбирает строки таблицы; вклады градиентов одного ID складываются]'
 assert old in t;t=t.replace(old,new,1)
 old='Таблица векторов является параметром, а выбранные строки — входом текущего примера. Близость строк отражает обучение на конкретной задаче и не гарантирует универсального смысла категорий.'
 new='Схема показывает таблицу как обучаемый параметр: пакет A,A,B выбирает три строки, а после вычисления потерь оба вклада A складываются в один градиент. Числа ниже выведены для среднего значения половины квадрата ошибки. Близость строк зависит от задачи и не гарантирует универсального смысла категорий.'
 assert old in t;return t.replace(old,new,1)
c=load();transform('01-data-features/02-features/feat-embeddings.md',op,
 'Пакет 03: схема выбора строк и накопления градиентов связана с числовым примером; приёмка ожидается.',c);save(c)
p=ROOT/'editorial/illustrations.json';data=json.loads(p.read_text(encoding='utf8'))
entry={'path':'оформление/course/feat-embeddings.svg','kind':'original-course-diagram',
 'external_figure':'none','authors':'Авторская схема курса','license':'Лицензия проекта; внешние рисунки не копируются.',
 'production':'tools/draw_batch03_features.py','chapters':['feat-embeddings'],'checked_on':'2026-10-28',
 'source_of_mechanism':'Выбор строк и сложение градиентов описаны в главе; API реализации указан рядом с утверждениями.',
 'changes':'Предметная схема таблицы, пакета и накопления градиентов с расчётом.'}
data['figures']=[x for x in data['figures'] if x['path']!=entry['path']];data['figures'].append(entry)
p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
print('Embedding illustration and explanation updated; review remains open.')
