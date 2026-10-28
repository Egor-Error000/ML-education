"""Make the target-statistics route and new diagram explicit."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools'))
from edit_chapter import load,save,transform
rel='01-data-features/02-features/feat-target.md'
changes={
 '![Target statistics, smoothing, cross-fitting и WoE: входы, преобразования и выходы]':'![Внешнее разбиение, внефолдовые обучающие признаки и таблица для новых объектов]',
 'Признаки обучающих строк строятся вне собственного фолда. Для новых объектов используется отдельная карта по всей разрешённой обучающей части; эти две карты выполняют разные роли.':'Фолд — одна из частей обучающих данных, которую временно откладывают при построении признаков. Слева на схеме каждая часть T кодируется по остальным частям T, после чего OOF-признаки подаются модели. Справа отдельная таблица по всем разрешённым данным T кодирует новые объекты V. Метки V используют только после прогноза, для оценки. Границы по пациентам и времени соблюдают на обоих уровнях.',
 '### Собственная метка остаётся даже после сильного shrinkage':'### Собственная метка остаётся даже после сильного сглаживания',
 '### Почему OOF по всему dataset перед внешней CV недостаточно':'### Почему OOF до внешней проверки недостаточно',
 '### Несовпадение reference size — ожидаемая часть метода':'### Разный размер справочной выборки',
 '## Ordered statistics':'## Упорядоченные статистики',
 '## WoE и multiclass':'## WoE и несколько классов',
 '### Distribution вместо среднего номера класса':'### Доли классов вместо среднего номера',
 '### Числовое различие WoE и posterior odds':'### Числовое различие WoE и условных шансов',
 '1. Убирает ли smoothing собственный label? **Ответ:** нет; нужна смена способа вычисления train feature.':'1. Убирает ли сглаживание собственную метку? **Ответ:** нет; нужно изменить способ вычисления обучающего признака.',
 '2. Почему нельзя один OOF feature использовать до внешней CV? **Ответ:** внешний validation может уже участвовать в features, по которым обучается модель.':'2. Почему нельзя заранее вычислить OOF по всем данным? **Ответ:** внешняя проверка может уже участвовать в признаках, по которым обучается модель.',
 '3. Чем WoE отличается от posterior log-odds? **Ответ:** prior odds constant при указанной convention.':'3. Чем WoE отличается от логарифма условных шансов? **Ответ:** логарифмом априорных шансов при указанном выборе положительного класса.',
 '**Практика:** досчитайте OOF values всех десяти строк при $\\alpha=2$. Добавьте случайный unique ID и сравните naive, smoothed и OOF representations. Нарисуйте два уровня split с точным указанием labels, доступных каждому mapping. Новые measured scores (оценки) сообщайте только после реального запуска.':'**Практика:** пересчитайте OOF-значения всех десяти строк при $\\alpha=2$. Добавьте случайный уникальный ID и сравните простую, сглаженную и OOF-кодировки. Нарисуйте два уровня разбиения и укажите, какие метки доступны каждой таблице. Сообщайте измеренные метрики только после реального эксперимента.'
}
def op(t):
 for old,new in changes.items():
  if old not in t:
   if new in t:continue
   raise ValueError('Missing exact text: '+old[:90])
  t=t.replace(old,new,1)
 return t
c=load();transform(rel,op,'Пакет 03: новая схема вложенной оценки и конкретные задания; содержательная проверка источников продолжается.',c)
renames={'Ordered statistics':'Упорядоченные статистики','WoE и multiclass':'WoE и несколько классов',
 'Distribution вместо среднего номера класса':'Доли классов вместо среднего номера','Числовое различие WoE и posterior odds':'Числовое различие WoE и условных шансов'}
for span in c['spans'].values():
 if span['path']==rel and span.get('destination_section') in renames:span['destination_section']=renames[span['destination_section']]
save(c)
p=ROOT/'editorial/illustrations.json';data=json.loads(p.read_text(encoding='utf8'))
for fig in data['figures']:
 if fig['path']=='оформление/course/feat-target.svg':
  fig.update(kind='original-course-diagram',external_figure='none',authors='Авторская схема курса',license='Лицензия проекта; внешние рисунки не копируются.',production='tools/draw_batch03_features.py',chapters=['feat-target'],checked_on='2026-10-28',source_of_mechanism='Вложенный протокол и обработка меток описаны в главе; первичные источники приведены рядом с утверждениями.',changes='Новая схема различает внутренние OOF-признаки, полную таблицу и внешний тест.')
  break
else:
 data['figures'].append({'path':'оформление/course/feat-target.svg','kind':'original-course-diagram',
  'external_figure':'none','authors':'Авторская схема курса','license':'Лицензия проекта; внешние рисунки не копируются.',
  'production':'tools/draw_batch03_features.py','chapters':['feat-target'],'checked_on':'2026-10-28',
  'source_of_mechanism':'Вложенный протокол и обработка меток описаны в главе; первичные источники приведены рядом с утверждениями.',
  'changes':'Новая схема различает внутренние OOF-признаки, полную таблицу и внешний тест.'})
p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
print('Target-statistics diagram and headings updated; review stays open.')
