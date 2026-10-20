"""One-time editorial blueprint. Published Markdown is edited directly after migration."""
MODULES = [
    ('data', '01-data-features', 'Данные и признаки', [('reliability','01-reliability','Достоверность эксперимента'),('features','02-features','Представление и создание признаков'),('modalities','03-modalities','Изображения и текст')]),
    ('reg', '02-regularization', 'Регуляризация', [('main','','Механизмы и выбор')]),
    ('cluster', '03-clustering', 'Кластеризация', [('main','','Геометрия, алгоритмы и применение')]),
    ('cv', '04-computer-vision', 'Компьютерное зрение', [('main','','Основы и задачи')]),
    ('lm', '05-language-models', 'LLM и языковые модели', [('found','01-foundations','Основы'),('bert','02-bert-encoders','BERT и encoder-модели'),('adapt','03-adaptation','Адаптация'),('infer','04-inference','Инференс'),('app','05-applications','Приложения'),('agent','06-agent-learning','Обучение агентов')]),
]
CHAPTERS=[]
def add(module, section, id, title, question, prerequisites=(), level='общее ядро', source=None):
    m=next(x for x in MODULES if x[0]==module)
    s=next(x for x in m[3] if x[0]==section)
    CHAPTERS.append(dict(id=id,module=module,section=section,title=title,question=question,
        prerequisites=list(prerequisites),level=level,path='/'.join(x for x in [m[1],s[1],id+'.md'] if x),source=source))

# The question is an observable learning objective, not a restatement of the title.
add('data','reliability','df-task','Постановка задачи и паспорт данных','Что является независимым объектом и какая информация доступна в момент прогноза?',source='data-А')
add('data','reliability','df-split','Split, cross-validation и утечки','Как разделить данные так, чтобы оценить именно будущий сценарий применения?',['df-task'],source='data-Б')
add('data','reliability','df-quality','Качество данных, пропуски, выбросы и разметка','Как отличить ошибку данных от редкого корректного наблюдения?',['df-task','df-split'],source='data-В')
add('data','reliability','df-metrics','Дисбаланс, метрики, порог и калибровка','Как связать ошибки, вероятности и цену решения?',['df-task','df-split'],source='data-Г')
add('data','reliability','df-augmentation','Дубликаты, аугментации и синтетика','Как расширять выборку, сохраняя смысл метки и независимость оценки?',['df-task','df-split','df-metrics'],source='data-Д')
add('data','reliability','df-loading','Dataset, batching и воспроизводимость','Как проследить путь одного примера от хранилища до функции потерь?',['df-task','df-split'],source='data-Е')
add('data','reliability','df-drift','Drift и надёжность оценки','Как отличить изменение данных от доказанного изменения качества?',['df-split','df-metrics'],level='основа ветки',source='data-И')
add('data','features','feat-types','Типы признаков и контракты представлений','Почему одинаковый dtype требует разных преобразований?',['df-task','df-split'])
add('data','features','feat-numeric','Числовые преобразования и масштабирование','Как преобразование меняет единицы, расстояния и поведение нового объекта?',['feat-types','df-split'])
add('data','features','feat-missing','Missing, unknown, rare и стабильный словарь','Как различать отсутствие значения, новую категорию и редкую известную категорию?',['feat-types','df-quality'],source='cat-З')
add('data','features','feat-categorical','Nominal, ordinal, one-hot и dummy encoding','Как представить категорию, не приписав ей ложный числовой порядок?',['feat-types','feat-missing'],source='cat-А')
add('data','features','feat-hashing','Кардинальность, частотные кодировки и hashing','Как ограничить размер представления и измерить цену столкновений?',['feat-categorical','df-split'],source='cat-Б')
add('data','features','feat-target','Target statistics, smoothing, cross-fitting и WoE','Как использовать целевую статистику, не раскрывая собственную метку?',['feat-categorical','df-split','df-metrics'],source='cat-В')
add('data','features','feat-embeddings','Обучаемые embeddings и перенос представлений','Как обучается вектор объекта и при каких условиях его можно использовать в другой задаче?',['feat-categorical','df-split'],level='основа ветки',source='cat-Г')
add('data','features','feat-native','Нативные категории в бустинге','Как модель выбирает разбиение категорий и что входит в сохранённый контракт?',['feat-categorical','feat-target'],source='cat-Д')
add('data','features','feat-interactions','Взаимодействия признаков, FM и FFM','Когда эффект признака зависит от другого поля и как это представить?',['feat-categorical','feat-embeddings'],level='основа ветки',source='cat-Е')
add('data','features','feat-sets','Наборы категорий и pooling','Как отличить множество от последовательности и что теряет агрегирование?',['feat-categorical','feat-embeddings'],level='основа ветки',source='cat-Ж')
add('data','features','feat-time','Временные признаки и доступность метки','Как вычислить признак на исторический момент с учётом задержки данных?',['df-split','feat-types'],level='основа ветки',source='cat-И')
add('data','features','feat-geography','Географические признаки и пространственная валидация','Как измерять близость на сфере и проверять перенос на новые территории?',['feat-types','df-split','feat-numeric'],level='основа ветки')
add('data','features','feat-selection','Отбор признаков и устойчивость выбора','Как оценить пользу отбора вместе со всей процедурой настройки?',['df-split','feat-numeric','feat-categorical'],level='основа ветки')
add('data','features','feat-reduction','Сокращение размерности: PCA и SVD','Чем новая координата отличается от выбранного исходного признака?',['feat-numeric','df-split'],level='основа ветки')
add('data','features','feat-autofe','Автоматическая генерация признаков','Как искать признаки, контролируя время доступности и переобучение на validation?',['feat-time','feat-selection'],level='углубление')
add('data','features','feat-pipeline','Единый pipeline и применение к новым объектам','Какие состояния сохраняются после fit и как доказать одинаковое преобразование при обучении и применении?',['df-split','feat-numeric','feat-missing','feat-categorical','feat-target'])
add('data','modalities','df-images','Изображения, аннотации и синхронные преобразования','Как преобразовать изображение и все его пространственные метки согласованно?',['df-task','df-split','df-loading'],level='основа ветки',source='data-Ж')
add('data','modalities','df-text','Текст, токенизация и разбиение документов','Как сохранить происхождение, границы документов и правильные targets?',['df-task','df-split','df-loading'],level='основа ветки',source='data-З')

reg=[('diagnosis','Обобщение, диагностика и границы теории','Какой разрыв между обучением и будущей ошибкой требует вмешательства?','А'),('penalties','Штрафы, ограничения, L1/L2 и AdamW','Как штраф изменяет оптимальное решение и зависит от масштаба признаков?','Б'),('training','Early stopping, learning rate, SWA и EMA','Как траектория обучения и выбор checkpoint меняют итоговую модель?','В'),('stochastic','Dropout, DropConnect, DropBlock и случайные ветви','Как структура случайной маски меняет вычисление и objective?','Г'),('normalization','Нормализации и состояния train/eval','По каким осям вычисляются статистики и какие из них сохраняются?','Д'),('augmentation','Label smoothing, MixUp, CutMix и аугментации','Как изменения inputs и targets задают предположения о задаче?','Ж'),('robustness','Adversarial training, consistency и spectral norm','Относительно каких допустимых изменений требуется устойчивость?','Е'),('structure','Структурные ограничения: время, графы, физика и ранг','Как выразить известную структуру задачи в допустимых решениях?','И'),('sequences','Регуляризация последовательностей, NLP и RL','Чем различаются шум представлений, entropy и ограничение обновления политики?','З'),('selection','SAM, Lookahead и итоговый выбор метода','Как проверить гипотезу о регуляризации при фиксированном бюджете настройки?','К')]
for i,(slug,title,q,letter) in enumerate(reg):
    req=['df-task','df-split','df-metrics'] if i==0 else ['reg-diagnosis']
    if slug=='penalties':req+=['feat-numeric']
    if slug=='selection':req+=['reg-penalties','reg-training','reg-stochastic']
    add('reg','main','reg-'+slug,title,q,req,level='углубление' if slug in ['structure','sequences'] else 'общее ядро',source='reg-'+letter)
add('reg','main','reg-theory','Углубление: VC, Rademacher и PAC-Bayes','При каких предпосылках оценка сложности ограничивает риск?',['reg-diagnosis'],level='углубление')
CHAPTERS.insert(next(i for i,c in enumerate(CHAPTERS) if c['id']=='reg-selection'),CHAPTERS.pop())

add('cluster','main','cl-geometry','Похожесть, расстояния и постановка кластеризации','Какую структуру данных мы называем кластером и почему это зависит от представления?',['df-task','df-split'],source='cluster-А')
add('cluster','main','cl-projection','Масштабирование, PCA, t-SNE и UMAP','Какие отношения сохраняет проекция и как проверить искажения?',['cl-geometry','feat-numeric'],source='cluster-З')
add('cluster','main','cl-kmeans','Центроидные методы: K-Means и K-Means++','Как assignment и обновление центров уменьшают один objective?',['cl-geometry','cl-projection'])
add('cluster','main','cl-hierarchy','Иерархическая кластеризация','Что определяет последовательность слияний и смысл среза дендрограммы?',['cl-geometry','cl-projection'])
add('cluster','main','cl-density','Плотностные методы: DBSCAN и HDBSCAN','Как связность плотных областей отличается от близости к центру?',['cl-geometry','cl-projection'])
add('cluster','main','cl-gmm','Вероятностные смеси и EM','Как мягкая принадлежность и обновление параметров повышают likelihood?',['cl-geometry','cl-projection'])
add('cluster','main','cl-metrics','Качество разбиения: внутренние и внешние метрики','Что измеряет метрика и как noise/coverage меняют сравнение?',['cl-kmeans','cl-hierarchy','cl-density','cl-gmm'],source='cluster-Г')
add('cluster','main','cl-stability','Выбор числа кластеров и устойчивость','Как выбрать разрешение разбиения без универсального оптимального K?',['cl-metrics'],source='cluster-В')
add('cluster','main','cl-frozen','Кластеризация замороженных представлений','Как разделить качество представления и качество алгоритма группировки?',['cl-metrics','cl-stability'],level='основа ветки')
add('cluster','main','cl-deep','Совместное обучение: DEC и IDEC','Как self-training изменяет представление и когда усиливает собственные ошибки?',['cl-frozen'],level='углубление')
add('cluster','main','cl-images','Кластеризация изображений и superpixels','Как различаются группировка изображений и группировка пикселей?',['cl-frozen'],level='основа ветки',source='cluster-Е')
add('cluster','main','cl-text','Текстовые представления, LDA, NMF и Top2Vec','Как различать тему документа, embedding и кластер?',['cl-frozen'],level='основа ветки',source='cluster-Ж')
add('cluster','main','cl-systems','Аномалии, псевдометки, потоки и IVF','Как использовать кластеризацию как компонент системы и обрабатывать новый объект?',['cl-metrics','cl-stability'],level='основа ветки',source='cluster-И')

cv=[('input','Изображение, карты признаков и блоки модели','Как меняются форма, разрешение и рецептивное поле по пути изображения?','00'),('classification','Классификация, CNN и residual learning','Как изображение превращается в logits класса и что меняет backbone?','А'),('vit','Vision Transformer: patches, attention и обучение','Как patch projection, позиции и attention преобразуют изображение?',None),('hierarchy','Углубление: Swin и иерархические представления','Как окна и patch merging создают пространственную иерархию?',None),('pretraining','Углубление: SSL, MAE, DeiT и DINOv3','Как получить и проверить представление без полного набора task labels?',None),('detection','Detection: множества объектов, IoU, AP и архитектуры','Как получить множество рамок и оценить его с учётом matching?','Б'),('semantic','Semantic segmentation: FCN, U-Net и DeepLab','Как предсказывать класс пикселя и восстанавливать пространственную сетку?','В'),('instance','Instance segmentation: Mask R-CNN и prototype masks','Как отличить экземпляры одного класса и получить маску каждого?','Г'),('pose','Pose: keypoints, heatmaps и группировка','Как связать положение ключевых точек с конкретным объектом?','Д'),('obb','Ориентированные рамки и периодичность угла','Как сравнивать геометрически одинаковые рамки с разными параметрами?','Е'),('tracking','Tracking: Kalman, association и идентичность','Как сохранить идентичность объекта между кадрами?','Ж'),('depth','Depth: геометрия, масштаб и DPT','Что можно восстановить из одного изображения и какие допущения нужны для масштаба?','З'),('open-vocabulary','CLIP, открытый словарь и YOLO-World','Как связать текстовое название с локализованным объектом?','И')]
for i,(slug,title,q,letter) in enumerate(cv):
 req=['df-images'] if i==0 else ['cv-input','cv-classification'] if i>1 else ['cv-input']
 if slug in ['instance','obb','tracking','open-vocabulary']:req+=['cv-detection']
 if slug=='instance':req+=['cv-semantic']
 if slug in ['hierarchy','pretraining']:req=['cv-vit']
 core=slug in ['input','classification','detection','semantic','instance']
 add('cv','main','cv-'+slug,title,q,req,level='общее ядро' if core else 'углубление' if slug in ['hierarchy','pretraining'] else 'основа ветки',source='cv-'+letter if letter else None)

found=[('task','Языковая модель и её задачи','Что моделирует вероятность последовательности и какие задачи используют это распределение?'),('tokenizer','Токенизация: от текста к индексам','Как строка становится последовательностью ID и где заканчивается токенизатор?'),('states','Embeddings и контекстные состояния','Чем различаются ID, embedding и скрытое состояние?'),('attention','Attention, Q/K/V и несколько голов','Как одна позиция получает взвешенную информацию от других?'),('position','Позиции, causal mask и padding','Как независимо задать порядок токенов и доступ к информации?'),('transformer','Transformer-блок: residual, normalization и MLP','Как attention, MLP и остаточные связи образуют повторяемый блок?'),('families','Encoder, decoder-only и encoder–decoder','Как допустимые связи и источник Q/K/V различают семейства архитектур?'),('output','Выход модели: LM head, logits и вероятности','Как скрытое состояние превращается в распределение следующего токена?'),('training','Обучение: targets, loss и обновление весов','Как построить targets и проследить путь ошибки до обучаемых параметров?'),('generation','Обычная генерация по одному токену','Как распределение превращается в продолжение и почему обучение отличается от применения?'),('bert','BERT в общем ядре: encoder, MLM и задачи','Как двунаправленный encoder обучается на скрытых позициях и используется в задачах?'),('rnn','Углубление: RNN и рекуррентное состояние','Как одна и та же ячейка связывает состояния и градиенты во времени?'),('backward','Углубление: точный backward attention','Как производные проходят через softmax, Q/K/V и общие параметры?'),('interpretation','Углубление: интерпретация представлений','Какие проверки нужны для утверждения о смысле слоя или embedding?')]
for i,(slug,title,q) in enumerate(found):
 if i==0:req=[]
 elif slug=='position':req=['lm-states','lm-attention']
 elif slug=='rnn':req=['lm-states','lm-training']
 elif slug=='backward':req=['lm-attention','lm-training']
 elif slug=='interpretation':req=['lm-states','lm-output','lm-training']
 else:req=['lm-'+found[i-1][0]]
 # Encoder users do not need an autoregressive-generation prerequisite.
 if slug=='bert':req=['lm-families','lm-training']
 add('lm','found','lm-'+slug,title,q,req,level='углубление' if i>=11 else 'общее ядро',source='found-Б' if slug=='rnn' else None)

bert=[('interface','Encoder, tokenizer, attention и pooling','Как получить состояния и корректно агрегировать их в представление текста?','А'),('objectives','MLM, альтернативные objectives и семейства encoder','Какие позиции дают loss и как цель обучения меняет представления?','Б'),('adaptation','Данные, task heads, fine-tuning и DAPT','Как согласовать разметку, форму выхода и objective encoder?','В'),('tasks','Classification, NER, spans и extractive QA','Как преобразовать текстовую задачу в targets и восстановить ответ в исходных координатах?','Г'),('embeddings','Sentence embeddings и contrastive learning','Как обучить пространство, где близость полезна для конкретной задачи?','Д'),('retrieval','Dense retrieval, cross-encoder, SPLADE и ColBERT','Как выбрать место взаимодействия query и document и оценить полный budget?','Е'),('domains','Языки, длинные документы, домены и layout','Какие части контракта меняются при переносе на новый язык или тип документа?','Ж'),('evaluation','Метрики, калибровка, OOD и интерпретация','Как оценить encoder на правильной единице задачи и проверить отказ?','З'),('operations','Сжатие, сервис encoder и жизненный цикл индекса','Как обновить encoder и индекс, сохраняя согласованность представлений?','И')]
for i,(slug,title,q,letter) in enumerate(bert):
 req=['lm-bert','df-split','df-text'] if i==0 else ['bert-interface']
 if slug=='adaptation':req+=['bert-objectives']
 if slug in ['tasks','embeddings','domains','evaluation']:req+=['bert-adaptation']
 if slug=='retrieval':req+=['bert-embeddings']
 if slug=='operations':req+=['bert-evaluation']
 add('lm','bert','bert-'+slug,title,q,req,level='основа ветки' if slug!='operations' else 'углубление',source='bert-'+letter)

adapt=[('choice','Задача адаптации, baseline и выбор вмешательства','Как отличить нехватку знаний, ошибку поведения и проблему исполнения?','А'),('sft','SFT: данные, шаблоны, masks и objective','Какие позиции становятся целями supervised fine-tuning?','Д'),('regimes','Полное обучение или PEFT','Какие параметры обновляются и почему это не меняет смысл SFT?',None),('lora','LoRA: low-rank обновление и градиенты','Как low-rank ветвь обновляет конкретную линейную проекцию?','Б'),('variants','DoRA, AdaLoRA, LoRA+ и QLoRA','Как различаются изменяемый объект, бюджет ранга и формат хранения базы?','В'),('alternatives','Adapters, prefix tuning, soft prompts и BitFit','Где добавляются обучаемые параметры и что меняется при inference?','Г'),('preference','RLHF, PPO, DPO, IPO, KTO и ORPO','Как предпочтения превращаются в objective и каких данных он требует?','Е'),('merging','Model soup, TIES, DARE и переключение адаптеров','Когда операции над весами совместимы и как проверить итоговую функцию?','Ж'),('distillation','Дистилляция и обучение на траекториях','Какой сигнал teacher передаёт student и как проверить перенос?','К'),('continual','AI feedback и continual learning','Как обновлять поведение, контролируя источник feedback и забывание?','И'),('evaluation','Оценка, perplexity, judge и экспорт','Как сравнить candidate с baseline и подтвердить сохранение поведения после экспорта?','З')]
for i,(slug,title,q,letter) in enumerate(adapt):
 req=['lm-training','lm-generation','df-split','df-text'] if i==0 else ['adapt-choice','adapt-sft'] if i>1 else ['adapt-choice']
 if slug=='lora':req+=['adapt-regimes','lm-transformer']
 if slug=='variants':req+=['adapt-lora']
 if slug=='continual':req+=['adapt-preference']
 add('lm','adapt','adapt-'+slug,title,q,req,level='углубление' if slug in ['variants','alternatives','merging','distillation','continual'] else 'основа ветки',source='adapt-'+letter if letter else None)

infer=[('generation','От prompt к обычному циклу генерации','Какие состояния меняются при генерации и какие веса остаются фиксированными?',None),('kv','Prefill, decode, KV-cache, MQA и GQA','Какие вычисления и тензоры можно повторно использовать?','В'),('context','Контекст, sliding window и prefix caching','Что меняется при ограничении истории и повторном использовании prefix?','Ж'),('decoding','Greedy, temperature, top-p, beam и ограничения','Как decoding policy выбирает токен из logits?','Е'),('serving','Serving, batching и PagedAttention','Как совместно обслуживать запросы при ограничениях памяти и задержки?','Г'),('speculative','Speculative decoding и rejection sampling','Почему проверка draft сохраняет заданное распределение?','Д'),('quantization','Квантизация: PTQ, QAT, GPTQ и AWQ','Какие массивы и ошибки меняются при сокращении precision?','А'),('pruning','Pruning и границы сжатия','Как удаление структуры связано с реальным ускорением?','Б'),('deployment','Deployment, cold start, очереди и масштабирование','Как согласовать admission, startup и воспроизводимый rollout?','И'),('monitoring','Кэш ответов, monitoring и качество','Как доказать допустимость reuse и заметить регрессию качества?','К')]
for i,(slug,title,q,letter) in enumerate(infer):
 req=['lm-generation'] if i==0 else ['infer-generation','infer-kv'] if i>1 else ['infer-generation']
 if slug=='speculative':req+=['infer-decoding']
 if slug in ['deployment','monitoring']:req+=['infer-serving']
 add('lm','infer','infer-'+slug,title,q,req,level='основа ветки' if slug in ['serving','deployment','monitoring'] else 'углубление' if slug in ['speculative','quantization','pruning'] else 'общее ядро',source='infer-'+letter if letter else None)

apps=[('structured','Structured output: schema и смысл результата','Что гарантирует синтаксис результата и что ещё должно проверить приложение?','А'),('tools','Вызов инструментов и исполнение','Где заканчивается выбор модели и начинается действие приложения?','Б'),('state','Состояние, orchestration и граф управления','Какие данные и переходы нужны для остановки, восстановления и повторного исполнения?','В'),('rag','RAG: BM25, dense, hybrid и multi-vector retrieval','Как найти evidence и связать его с ответом без обучения собственного retriever?','Г'),('evaluation','Оценка retrieval, ответов и task success','Как локализовать ошибку приложения и измерить стоимость успешной задачи?','Д')]
for i,(slug,title,q,letter) in enumerate(apps):
 req=['lm-generation','df-task'] if i==0 else ['app-'+apps[i-1][0]]
 if slug=='rag':req+=['df-split','df-text']
 add('lm','app','app-'+slug,title,q,req,level='основа ветки',source='app-'+letter)

agents=[('task','Постановка обучения агента и baseline','Как выбрать вмешательство по измеренному типу ошибки?','А'),('environment','Среда, observations, actions и horizon','Как описать доступную агенту информацию и последствия действия?','Б'),('evaluation','Оценка траекторий и воспроизводимость','Что считать успехом целой попытки и как учитывать retries и стоимость?','К'),('demonstrations','Behavior cloning и SFT на демонстрациях','Как из траектории получить targets для обучения политики?','В'),('recovery','Distribution shift, recovery и DAgger','Как научиться действовать в состояниях, порождённых собственными ошибками?','Г'),('rl','Обучение через взаимодействие: PPO и GRPO','Как отложенный reward меняет вероятность последовательности действий?','Е'),('reasoning','Углубление: reasoning и верификация','Когда дополнительный поиск и проверка улучшают результат?','Д'),('memory','Углубление: память и длинный горизонт','Какую информацию сохранить и как измерить её полезность?','Ж'),('preferences','Углубление: предпочтения траекторий','Как сравнивать полные попытки с разным бюджетом и исходом?','З'),('coordination','Углубление: роли и обучение координации','При каких условиях несколько участников дают измеримый выигрыш?','И')]
for i,(slug,title,q,letter) in enumerate(agents):
 req=['app-state','app-evaluation'] if i==0 else ['agent-task'] if i==1 else ['agent-environment'] if i==2 else ['agent-evaluation']
 if slug=='demonstrations':req+=['adapt-sft']
 if slug=='recovery':req+=['agent-demonstrations']
 if slug=='rl':req+=['agent-recovery']
 add('lm','agent','agent-'+slug,title,q,req,level='углубление' if i>=6 else 'основа ветки',source='agent-'+letter)

ROUTES = [
 dict(id='tabular',title='Табличные задачи',chapters=['df-task','df-split','df-quality','df-metrics','feat-types','feat-numeric','feat-missing','feat-categorical','feat-hashing','feat-target','feat-native','feat-pipeline']),
 dict(id='clustering',title='Кластеризация без Transformer',chapters=['df-task','df-split','feat-types','feat-numeric','cl-geometry','cl-projection','cl-kmeans','cl-hierarchy','cl-density','cl-gmm','cl-metrics','cl-stability','cl-systems']),
 dict(id='vision',title='CV без LLM',chapters=['df-task','df-split','df-loading','df-images','cv-input','cv-classification','cv-detection','cv-semantic','cv-instance']),
 dict(id='language-core',title='Общее ядро языковых моделей',chapters=['lm-'+x[0] for x in found[:11]]),
 dict(id='encoder',title='BERT без генеративной адаптации',chapters=['df-task','df-split','df-loading','df-text']+['lm-'+x[0] for x in found[:9]]+['lm-bert','bert-interface','bert-objectives','bert-adaptation','bert-tasks','bert-evaluation']),
 dict(id='application',title='Приложение без LoRA и обучения retrieval',chapters=['df-task','df-split','df-loading','df-text']+['lm-'+x[0] for x in found[:10]]+['app-'+x[0] for x in apps]),
 dict(id='adaptation',title='SFT перед LoRA',chapters=['df-task','df-split','df-loading','df-text']+['lm-'+x[0] for x in found[:10]]+['adapt-choice','adapt-sft','adapt-regimes','adapt-lora','adapt-evaluation']),
 dict(id='inference',title='От генерации к serving',chapters=['lm-'+x[0] for x in found[:10]]+['infer-generation','infer-kv','infer-context','infer-decoding','infer-serving','infer-deployment','infer-monitoring']),
 dict(id='agents',title='Оценка агента перед обучением',chapters=['agent-task','agent-environment','agent-evaluation','agent-demonstrations','agent-recovery','agent-rl'],entry=['app-state','app-evaluation','adapt-sft']),
]
