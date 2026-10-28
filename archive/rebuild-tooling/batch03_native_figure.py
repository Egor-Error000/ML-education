"""Connect the figure to the one-node comparison; retain review status."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools'))
from edit_chapter import load,save,transform
def op(t):
 old='![Нативные категории в бустинге: входы, преобразования и выходы]'
 new='![Сравнение числовых порогов и категориального подмножества в одном узле]'
 assert old in t;t=t.replace(old,new,1)
 old='Нативная обработка категорий зависит от алгоритма бустинга. Нужно отличать поиск групп категорий от целевых статистик и проверить, какие данные используются при вычислении каждой величины.'
 new='На схеме одинаковый узел проверяется двумя способами: числовые пороги делят категории по кодам, а категориальный вариант допускает множество AC. Справа его выигрыш равен 7.2; слева лучший порог даёт около 3.81. Ниже разберём расчёт, а затем различия библиотек. Группы категорий и целевые статистики — разные механизмы с разными правилами использования меток.'
 assert old in t;return t.replace(old,new,1)
c=load();transform('01-data-features/02-features/feat-native.md',op,
 'Пакет 03: новая схема сравнивает допустимые разбиения и связана с расчётом; содержательная приёмка ожидается.',c);save(c)
p=ROOT/'editorial/illustrations.json';data=json.loads(p.read_text(encoding='utf8'))
entry={'path':'оформление/course/feat-native.svg','kind':'original-course-diagram',
 'external_figure':'none','authors':'Авторская схема курса','license':'Лицензия проекта; внешние рисунки не копируются.',
 'production':'tools/draw_batch03_features.py','chapters':['feat-native'],'checked_on':'2026-10-28',
 'source_of_mechanism':'Учебный пример функции потерь и разбиения подробно рассчитан в главе.',
 'changes':'Предметная схема сравнения порога и подмножества с градиентами, весами и выигрышем.'}
data['figures']=[x for x in data['figures'] if x['path']!=entry['path']];data['figures'].append(entry)
p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
print('Native split diagram updated; review remains open.')
