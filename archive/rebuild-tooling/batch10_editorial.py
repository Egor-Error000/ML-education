"""Author scenarios and register original drawings; acceptance is a later action."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from edit_chapter import load,save,transform
def read(p):return json.loads((ROOT/p).read_text(encoding='utf8'))
def write(p,v):(ROOT/p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
data={
'lm-tokenizer':(
 'Как превратить текст в IDs, сохранив связь с исходными символами?',
 'Различить нормализацию, подслова, словарь, специальные позиции и маски; воспроизвести сохранённый контракт токенизатора.',
 'Нормализация; сегментация; IDs и словарь; специальные позиции; маски и offsets; длина и воспроизводимость',
 'Пройти «кот спит» от строки к условным подсловам и полуоткрытым интервалам; сравнить разные нормализаторы без заявления о качестве.',
 'Сопоставить подслова с символьной разметкой и объяснить, почему специальная позиция не обозначает символ документа.',
 'Учебное разбиение не подменяет вывод реального tokenizer; offsets зависят от реализации.'),
'lm-states':(
 'Чем отличаются ID, строка таблицы embeddings, контекстное состояние и оценка слова?',
 'Проследить оси V и D, повторные lookup и выход hEᵀ; отделить изменение активаций от обновления весов.',
 'ID как индекс; lookup; сумма градиентов повторного ID; контекстные состояния; выходная проекция; обучение и инференс',
 'E:[5,3], h=(0.2,0.9,0.1) дают logits=(0.1,0.2,0.9,1.1,−0.1); argmax по D не предсказывает ID слова.',
 'Вычислить выходную проекцию и сумму градиентов двух вхождений одного ID; назвать изменяемые объекты в eval и train.',
 'Ширина состояния не равна размеру словаря; совпадение ширины двух моделей не согласует пространства.'),
'lm-position':(
 'Как модель различает расположение токенов и ограничивает доступ к контексту?',
 'Построить позиционные сигналы и маски; отличить геометрию порядка от запрета чтения будущего.',
 'Перестановки и причинность; обучаемые позиции; синусные позиции; RoPE; padding и causal masks; случаи пустого контекста',
 'Причинная матрица для трёх позиций имеет 1,2,3 разрешённых ключа; относительный поворот Q/K сохраняет зависимость от разности углов.',
 'Нарисовать маску с padding, указать ось softmax и показать, почему маска ошибки не заменяет маску внимания.',
 'Причинность уже нарушает перестановочную симметрию, но не задаёт полный позиционный сигнал; длинный контекст требует проверки.'),
'bert-adaptation':(
 'Какие данные, выход и параметры следует выбрать для адаптации BERT?',
 'Согласовать единицу разметки, loss, набор обновляемых параметров и независимую оценку.',
 'Единица разметки; независимый split; выход и loss; путь градиента; frozen и full; необязательная LoRA; накопление; диагностика и словарь',
 'CE/BCE/Huber и их градиенты; 2307 параметров головы и 294912 LoRA; последняя группа 16+16+8 взвешивается 0.4/0.4/0.2.',
 'Составить контракт задачи, пересчитать неполную группу и отличить frozen от detach и eval; сравнить baseline при том же split.',
 'LoRA — пропускаемое углубление; адаптация словаря и DAPT не доказывают улучшения задачи или усвоения факта.'),
'bert-tasks':(
 'Как разметка превращается в голову классификации, сущностей или извлечения ответа?',
 'Выбрать выход и декодирование, восстановить интервалы текста и отделить метрики отдельных стадий.',
 'Классы и независимые labels; NER и подслова; CRF; вложенные интервалы и связи; пары текстов; QA; zero-shot и SetFit; проекты P1 P2 P4',
 'Softmax(2,1,0), четыре CRF-пути 2/3/1/4 и пять допустимых QA-интервалов; [7,13) в «Анна в Москве».',
 'Выполнить P1/P2/P4 с независимыми документами; проверить границы сущности, допустимость QA и вклад отказов.',
 'CRF не гарантирует смысловую корректность; пары и окна не создают новых независимых документов; NLI-шаблоны требуют оценки.'),
'bert-embeddings':(
 'Как обучить и проверить векторы предложений для сравнения текстов?',
 'Различить независимый encoder, pooling, нормировку и цель обучения; сохранить совместимость индекса.',
 'Контракт и интерфейс reader; dot и cosine; triplet и InfoNCE; несколько positives и пул; SimCSE; дистилляция; Matryoshka; применения',
 'Масса двух positives даёт 0.223144, средний CE — 0.948560; поворот нового encoder ломает сравнение со старым индексом.',
 'Сравнить общий пул и накопление микропакетов; найти ложный negative; проверить цепочку дубликатов и нулевой префикс.',
 'Средний вектор MLM не гарантирует качество retrieval; геометрия и контракт требуют самостоятельной оценки.'),
'bert-retrieval':(
 'Как устроены кандидаты, переранжирование и разные представления поискового документа?',
 'Разделить offline-кодирование и запрос, оценить потолок кандидатов и затраты; понимать BM25, SPLADE и MaxSim.',
 'Плотный индекс; точный и приближённый поиск; BM25; cross-encoder; SPLADE; RRF; ColBERT; BGE-M3 и хранение; P3',
 'BM25=11.710677 с полными промежуточными значениями; одинаковые средние документов дают MaxSim 2 и 1; RRF рангов (1,100) и (10,10).',
 'В P3 отдельно оценить первый этап и reranker, сравнить с точным поиском; посчитать сырой массив без обещания задержки.',
 'SPLADE и ColBERT пропускаются в базовом маршруте; ANN-полнота и релевантность имеют разные эталоны.'),
'bert-domains':(
 'Что меняется при переносе encoder между языками, доменами и видами документа?',
 'Различить потерю контекста, смену корпуса, вёрстку и OCR; выбрать независимую проверку переноса.',
 'Русский и многоязычие; усечение окна и иерархия; гарантия перекрытия; Longformer; SciBERT CodeBERT; аспекты и linking; LayoutLM; P5',
 'Взвешенная точность языков 0.915 против среднего 0.775; O≥A−1; рамка (100,400,500,600) преобразуется в (100,200,500,300).',
 'Проверить граничный ответ и дальнюю связь отдельно; построить языковые срезы; разделить ошибки OCR и привязки поля; выполнить P5.',
 'Видимость ответа не гарантирует его извлечение; доменная модель и большая длина не обещают качество без сравнения.'),
'bert-evaluation':(
 'Какие выводы позволяют метрики, калибровка и объяснения encoder?',
 'Выбрать единицу оценки и раздельно проверить качество задачи, уверенность, устойчивость и интерпретацию.',
 'Метрики и дисбаланс; ranking и qrels; парная неопределённость; калибровка; риск и покрытие; устойчивость; IG и probes; доступ; контроль',
 '95/5 и постоянный класс: accuracy .95, macro-F1 .487179; MRR .5; IG=(3,3) для F=x₁x₂ против gradient×input=(6,6).',
 'Сравнить модели парным bootstrap по независимым группам; выбрать T на calibration; сформулировать ожидаемую реакцию на отрицание.',
 'Калибровка не исправляет ranking; attribution не доказывает причинность; неполные qrels и выбор test ограничивают вывод.'),
'bert-operations':(
 'Как уменьшить стоимость encoder и безопасно для качества обновить служебный комплект?',
 'Различить дистилляцию, квантизацию, кеш, очередь и версионный индекс; составить проверяемый протокол эксплуатации.',
 'Учитель и ученик; квантизация; память; bidirectional cache; очередь и пакеты; экспорт; мониторинг; миграция; P6',
 'Квантизация .26 при шаге .1 даёт .3; новый токен меняет старый выход 2→4; 100 запросов дают throughput 50/с и среднюю задержку 1.1 с.',
 'В P6 проверить пакет encoder/tokenizer/head/index, задержки от прихода и откат; отличить рассчитанный payload от замера памяти.',
 'Уменьшение числа параметров не задаёт ускорение; очередь влияет на latency; новый encoder требует совместимого индекса.')}
course=read('course.json');scenarios=read('editorial/chapter-scenarios.json')
definitions={
'lm-tokenizer':('Токен, ID, нормализация, подслово, специальная позиция, символьное смещение.','IDs задают последовательность индексов; числовой порядок IDs не является метрикой смысла.'),
'lm-states':('Словарь V, ширина D, lookup, параметр, активация, контекстное состояние, логит.','E:[V,D], IDs:[B,L], X:[B,L,D]; hEᵀ:[V]; сумма lookup-градиентов повторного ID.'),
'lm-position':('Позиционное представление, маска внимания, маска ошибки, padding, причинность, RoPE.','Синусы для чётного D; R(iθ)ᵀR(jθ)=R((j−i)θ); запрет до softmax.'),
'bert-adaptation':('Единица разметки, наблюдаемая метка, голова, CE/BCE, заморозка, накопление, DAPT/TAPT.','Z=UW+b; CE/BCE/Huber и производные; нормировка по числу наблюдаемых целей; подсчёт обучаемых весов.'),
'bert-tasks':('Класс, независимая метка, BIO, CRF, фрагмент, NLI, символьная граница, отказ.','Баллы и нормировка CRF; logsumexp и max; L(L+1)/2; QA start/end и допустимые пары.'),
'bert-embeddings':('Bi-encoder, pooling, косинус, контрастивный пул, false negative, dropout-view, префикс вектора.','Triplet/InfoNCE с τ>0; производная dot-сравнения; две цели нескольких positives; проекция P; вложенные ошибки.'),
'bert-retrieval':('ANN, qrels, BM25, reranker, словарная разреженность, RRF, MaxSim.','BM25 с k1,b; попарная softplus; SPLADE max/sum; RRF; ColBERT и градиенты; Nd·байт.'),
'bert-domains':('Перенос языка и домена, окно, шаг, перекрытие, глобальная позиция, NIL, OCR, рамка.','O≥A−1 при целых границах и A≤C; поле 1+2rK; нормировка x/width и y/height.'),
'bert-evaluation':('Единица оценки, macro/micro, qrels, bootstrap, температура, охват/риск, attribution.','MRR, DCG/nDCG, NLL/Brier; T>0; риск при непустом принятом множестве; интегральные градиенты.'),
'bert-operations':('Дистилляция, квантизация, параметрический payload, кеш функции, очередь, shadow/canary, комплект версии.','T²KL, round/clip и восстановление; 4P/16P; сумма времён завершения и throughput; согласование версий.')}
for chapter in course['chapters']:
    if chapter['id'] not in data:continue
    ident=chapter['id'];q,result,steps,example,practice,limits=data[ident]
    chapter['question']=q
    s=scenarios['chapters'][ident]
    s.update(status='authored — сценарий пакета 01; приёмка текста отдельно',question=q,result=result,
             steps=steps.split('; '),example=example,practice=practice,limitations=limits,
             diagram_narrative='Проследить вход, промежуточные состояния, ветви и выход локальных предметных схем; формы и числовые подписи разобраны в тексте.')
    s['definitions'],s['formulas']=definitions[ident]
    consumers=', '.join(chapter.get('used_by',[])) or 'Завершение прикладной ветки; проект P6.'
    s['markdown']=f'**Учебный вопрос:** {q}\n\n**Результат:** {result}\n\n**Уровень:** {chapter.get("level","по реестру курса")}. **Предпосылки:** '+', '.join(chapter.get('depends',chapter.get('prerequisites',[])))+f'\n\n**Потребители:** {consumers}\n\n**Порядок объяснения:**\n\n'+''.join(f'{i}. {v}.\n' for i,v in enumerate(s['steps'],1))+f'\n**Определения:** {s["definitions"]}\n\n**Формулы и условия:** {s["formulas"]}\n\n**Разобранный пример:** {example}\n\n**Схемы и разбор:** {s["diagram_narrative"]}\n\n**Практика и проверка:** {practice}\n\n**Ограничения:** {limits}\n\n**Редакторская проверка:** происхождение, смысл формул, промежуточные вычисления, визуальная связность и самостоятельное чтение; статус фиксируется отдельно.\n'
write('course.json',course);write('editorial/chapter-scenarios.json',scenarios)
cov=load()
for chapter in course['chapters']:
    if chapter['id'] not in data:continue
    transform(chapter['path'],lambda text:text.replace('микрошагa','микрошага'),
              'Исправлена латинская буква в русском окончании; смысл и привязка сохранены.',cov)
    save(cov)
registry=read('editorial/illustrations.json')
sources={
 'lm-tokenizer':'https://github.com/google-research/bert/blob/master/tokenization.py',
 'lm-states':'https://github.com/google-research/bert/blob/master/modeling.py',
 'lm-position':'https://arxiv.org/abs/2104.09864',
 'bert-adaptation':'https://docs.pytorch.org/docs/2.14/notes/autograd.html',
 'bert-crf':'https://docs.pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html',
 'bert-span-relations':'https://aclanthology.org/N19-1423/',
 'bert-qa-decode':'https://aclanthology.org/N19-1423/',
 'bert-setfit':'https://arxiv.org/abs/2209.11055',
 'bert-embeddings':'https://aclanthology.org/D19-1410/',
 'bert-contrastive-pool':'https://sbert.net/docs/package_reference/sentence_transformer/losses.html',
 'bert-simcse':'https://arxiv.org/abs/2104.08821',
 'bert-embedding-distill':'https://arxiv.org/abs/1503.02531',
 'bert-matryoshka':'https://arxiv.org/abs/2205.13147',
 'bert-retrieval':'https://arxiv.org/html/2005.11401v4#S2.SS3',
 'bert-splade':'https://arxiv.org/abs/2109.10086',
 'bert-colbert':'https://arxiv.org/abs/2112.01488',
 'bert-bge-m3':'https://arxiv.org/html/2402.03216v5',
 'bert-domains':'https://arxiv.org/abs/2004.05150',
 'bert-longformer':'https://arxiv.org/abs/2004.05150',
 'bert-domain-pretraining':'https://aclanthology.org/D19-1371/ ; https://arxiv.org/abs/2002.08155',
 'bert-linking':'Авторская схема разложения задачи: упоминание, кандидаты, ранжирование и NIL.',
 'bert-layout':'https://arxiv.org/abs/1912.13318',
 'bert-evaluation':'https://arxiv.org/abs/1706.04599',
 'bert-attribution':'https://arxiv.org/abs/1703.01365',
 'bert-operations':'https://arxiv.org/abs/1503.02531',
 'bert-encoder-cache':'https://github.com/google-research/bert/blob/master/modeling.py',
 'bert-serving-queue':'Авторская арифметическая модель очереди: 100 запросов, пакет 10, обслуживание 0.2 с.',
 'bert-rollout':'Авторский протокол согласованного encoder/index и отката.'}
key=next(k for k in registry if isinstance(registry[k],list))
for name,source in sources.items():
    path='оформление/course/'+name+'.svg'
    registry[key]=[v for v in registry[key] if v.get('path')!=path]
    registry[key].append(dict(path=path,kind='original-course-diagram',source_of_mechanism=source,
        external_figure='none',production='tools/draw_batch10_figures.py',
        changes='Собственный русскоязычный рисунок; внешняя иллюстрация не копировалась. Числа — учебные вычисления, не benchmark.'))
write('editorial/illustrations.json',registry)
print('Authored ten scenarios and registered',len(sources),'original diagrams; acceptance unchanged.')
