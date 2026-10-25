"""Refresh factual inventories without automatically accepting open chapters."""
import json,sys,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from edit_chapter import ROOT,load,digest
cov=load()
registry_path=ROOT/'course.json';registry=json.loads(registry_path.read_text(encoding='utf8'))
for chapter in registry['chapters']:
    text=(ROOT/chapter['path']).read_text(encoding='utf8')
    refs=[a or b for a,b in re.findall(r'!\[[^\]]*\]\((?:<([^>]+\.svg)>|([^()\s]+\.svg))\)',text)]
    chapter['figures']=list(dict.fromkeys((Path(chapter['path']).parent/ref).resolve().relative_to(ROOT).as_posix() for ref in refs))
registry_path.write_text(json.dumps(registry,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
p=ROOT/'editorial/chapter-reviews.json';cards=json.loads(p.read_text(encoding='utf8'))
rewritten={'lm-attention','lm-bert','bert-interface','bert-objectives','lm-tokenizer','lm-states','lm-output','lm-position','lm-generation','lm-transformer','lm-families','lm-training','feat-numeric','feat-selection'}
for ident,card in cards['chapters'].items():
    card['text_sha256']=digest((ROOT/card['path']).read_text(encoding='utf8'))
    if ident in rewritten or ident.startswith('cl-'):card['prose']='rewritten'
    units=[u for u in cov['units'] if u.get('chapter')==ident and not u.get('provenance_only')]
    card['source_units']=len(units)
    card['source_units_pending_semantic_review']=sum(any(cov['spans'][k].get('review_status')=='pending' for k in u['targets']) for u in units)
    evidence={
      'feat-numeric':'numeric-selection','feat-selection':'numeric-selection',
      'cl-kmeans':'cluster-algorithms','cl-hierarchy':'cluster-algorithms',
      'cl-density':'cluster-algorithms','cl-gmm':'cluster-algorithms','lm-bert':'bert-core',
      'bert-interface':'bert-interface','bert-objectives':'bert-objectives','cl-frozen':'frozen','lm-transformer':'transformer',
      'lm-families':'families','lm-training':'training',
      'cl-geometry':'geometry-projection','cl-projection':'geometry-projection',
      'cl-metrics':'cluster-metrics','cl-stability':'cluster-stability',
      'cl-deep':'deep-clustering','cl-images':'image-clustering',
      'cl-text':'text-systems-clustering','cl-systems':'text-systems-clustering'}
    if ident in evidence:
        if card['source_units_pending_semantic_review']:raise ValueError('Pending source review: '+ident)
        card.update(content_acceptance='closed',evidence=['editorial/review-evidence.md#'+evidence[ident]],remaining=[])
p.write_text(json.dumps(cards,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
print('Updated hashes; closed cards:',sum(x['content_acceptance']=='closed' for x in cards['chapters'].values()))

p=ROOT/'editorial/chapter-scenarios.json';scenarios=json.loads(p.read_text(encoding='utf8'))
updates={
'feat-numeric':('Единицы и расстояния; стандартизация, min–max и IQR; применение к новому объекту; log1p и интервалы; сохранение состояния',
'A=(0,0), B=(1,100), C=(10,0): смена единиц меняет соседа. Обучающие 1 и 3 дают новому 100 координаты 98 и 49.5; IQR=0 не доказывает постоянство поля.',
'Отдельные потоки fit и transform; новые значения не возвращаются в оценку среднего и масштаба.'),
'feat-selection':('Семейства отбора; сохраняемое состояние; дубликаты и XOR; вложенная оценка; бюджет и генерация; предметная проверка',
'Средние ошибки k=1: 0.22, k=2: 0.24; выбор k=1 внутри внешней обучающей части. Для поиска 20×3×5 требуется 300 обучений.',
'Внутренние складки выбирают k. Процедура переобучается на полном outer train и один раз применяется к outer test.'),
'lm-transformer':('Форма блока; Pre-Norm/Post-Norm; LayerNorm/RMSNorm; формы внимания; MLP и управляющая ветвь; остаточная сумма; место в модели',
'h=(1,2,3,4): LayerNorm и RMSNorm дают разные состояния. Управляющая MLP D=2,F=3 проходит up/gate/SiLU/произведение/down и возвращает (5.284782,−3.822665).',
'Две остаточные связи имеют отдельные обходы. В управляющей MLP ветви up и gate сходятся в произведение; down возвращает D.'),
'lm-families':('Три семейства и доступная информация; маска и задача обучения; источники Q/K/V; формы T×S; полный расчёт и маска PAD; слой и проверка',
'Q=(ln2,0), K=(1,0,−1), три двумерных V: веса первой строки 4/7,2/7,1/7 дают (1,5/7). Исключение третьего источника даёт (2/3,2/3).',
'Три самостоятельные ветви архитектур. В encoder–decoder состояния источника дают K/V, состояния целевого префикса — Q; выход имеет T строк.'),
'cl-geometry':('Класс и кластер; представление и контракт; расстояния и центр; масштаб и соседи; косинус и нормировка; корреляции и отбеливание; предметная проверка',
'A=(0,0), B=(1,100), C=(10,0): смена масштаба меняет соседа. Три единичных вектора нарушают неравенство треугольника для 1−cos.',
'Контракт представления и координаты задают расстояние до выбора алгоритма; результат проверяется на соседях и полезных различиях.'),
'cl-projection':('Зачем менять размерность; PCA через SVD; пример восстановления; условные и симметричные сходства t-SNE; UMAP и плотность; проверка в исходном пространстве',
'Точка (5,1), среднее (2,2), направление (1,1)/√2: координата √2, восстановление (3,3), квадрат ошибки 8.',
'Ветви PCA, t-SNE и UMAP имеют разные цели; двумерная карта и рабочие координаты кластеризации оцениваются отдельно.'),
'cl-metrics':('Тип результата; таблица сопряжённости; purity и сопоставление; пары и ARI; энтропии и NMI; DB и CH; шум, охват и независимая оценка',
'Таблица [[1,1],[1,1]] даёт ARI=−0.5; отдельная группа на объект даёт purity=1 без полезного обобщения. CH=32 и DB=0.25 вычисляются в заданной геометрии.',
'Разбиение расходится на внешнюю оценку с независимыми классами и внутреннюю с координатами; общий протокол фиксирует охват и шум.'),
'cl-stability':('Ошибка при разных K; силуэт и крайние случаи; Gap и опорная модель; соглашения реализации; смеси и разрешение; сравнение общих объектов; выбор по полезности',
'Для [0,1,4,5] J=17/1/0 при K=1/2/4. ARI=1 на общих A/B/C/D не подтверждает устойчивость исключённых E/F.',
'Два запуска сохраняют идентификаторы объектов, сравнивают состав групп на пересечении и отдельно сообщают долю пересечения.'),
'cl-deep':('Автокодировщик и инициализация; назначения Стьюдента; фиксированная вспомогательная цель; равные и неравные частоты; коллапс; IDEC с восстановлением; раздельные сравнения',
'q=(0.6,0.4) при равных частотах даёт p≈(0.6923,0.3077), при f=(9,1) — p=(0.2,0.8); однообъектный случай даёт p=q.',
'DEC разделён на предобучение и кластерную фазу без декодера. IDEC имеет общий кодировщик и две ветви ошибки с отдельными путями обновления.'),
'cl-images':('Снимок и пиксель как разные объекты; карта фиксированной CNN; среднее и максимум; цели предобучения; аугментации и представители; Lab и SLIC; поток и неизвестные',
'Карта 2×2×2 даёт среднее (4,2), максимум (7,4) и нормированный вектор (0.894427,0.447214). Два веса компактности SLIC дают 14.142 и 10.308.',
'Верхняя ветвь даёт номер группы снимка, нижняя — карту номеров связных областей; показан повтор назначения и обновления центров SLIC.'),
'cl-text':('Нужное различие и TF-IDF; готовые векторы; LDA и порождение слов; NMF и масштаб множителей; Top2Vec в двух пространствах; определения оценок и срезы',
'TF-IDF до нормировки 3.386294; смесь LDA даёт 0.073. NMF (2,1) и (4,1) с разными H восстанавливают одно (2,3); разнообразие списков слов 3/4.',
'Три независимые схемы различают порождение слова LDA, произведение обучаемых матриц NMF и группировку/описание Top2Vec.'),
'cl-systems':('Редкость и неисправность; псевдометки и перестановка; построение IVF; отдельный путь запроса; кандидаты, затраты и полнота; новый объект и версия; три метрики применения',
'Миллион векторов, 1000 ячеек и 10 списков дают около 10000 кандидатов; при 20 списках — 20000. Пересечение точного {a,b,c} и ответа {a,c,d} даёт полноту 2/3.',
'Построение индекса отделено от запроса; фиксированные центры и списки подаются в маршрутизацию кандидатов, точный поиск задаёт контроль полноты.'),
'lm-training':('Сдвинутые цели; вероятность и ошибка; две маски и усреднение; путь градиента; общая таблица весов; обновление и режим оценки; перплексия и практика',
'Вероятности 0.5,0.25,0.8 дают среднюю CE 0.767528 и perplexity 2.154435. Исключение второй цели меняет среднее на 0.458145.',
'Входы и цели хранятся раздельно и сходятся в функции потерь. Backward вычисляет производные; оптимизатор отдельно изменяет параметры. Маска контекста не заменяет маску ошибки.')}
for ident,(order,example,diagram) in updates.items():
    s=scenarios['chapters'][ident];s['steps']=order.split('; ');s['example']=example;s['diagram_narrative']=diagram
    md=s['markdown'];import re
    if ident=='cl-text':md=md.replace('Как различать тему документа, embedding и кластер?','Как различать тему документа, векторное представление и кластер?')
    if ident=='cl-deep':md=md.replace('Как self-training изменяет представление','Как самообучение изменяет представление')
    md=re.sub(r'\*\*Порядок объяснения:\*\*[\s\S]*?(?=\*\*Разобранный пример:)', '**Порядок объяснения:**\n\n'+''.join(f'{i}. {x}.\n' for i,x in enumerate(s['steps'],1))+'\n',md)
    md=re.sub(r'\*\*Разобранный пример:\*\*[^\n]*','**Разобранный пример:** '+example,md)
    md=re.sub(r'\*\*Схемы и разбор:\*\*[^\n]*','**Схемы и разбор:** '+diagram,md)
    s['markdown']=md
p.write_text(json.dumps(scenarios,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
