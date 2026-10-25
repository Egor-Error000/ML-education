# Происхождение учебных материалов

Редакционные вступления сохранены отдельно от учебного объяснения. Полный неизменённый срез — в архиве; отдельные содержательные блоки связаны с новой редакцией картой покрытия.

**Источник:** `01-data/Учебные конспекты/А — Постановка задачи и паспорт данных.md:1`

# А. Что предсказывает модель и на каких данных проверяется вывод

**Источник:** `01-data/Учебные конспекты/А — Постановка задачи и паспорт данных.md:3`

[Оглавление](../01-data-features/README.md). Пункт 1 плана. **Происхождение:** дополнительный материал по действующему плану. В предоставленных чатах нет отдельного полного обсуждения этого курса; связанные примеры взяты из задач CV и LLM, без приписывания вам пройденных занятий.

**Источник:** `01-data/Учебные конспекты/А — Постановка задачи и паспорт данных.md:5`

**Уровень:** общее ядро. **Предпосылки:** таблица и основные понятия supervised learning. **Используют:** все модули.

**Источник:** `01-data/Учебные конспекты/Б — Split, cross-validation и утечки.md:1`

# Б. Как отделить обучение от честной проверки

**Источник:** `01-data/Учебные конспекты/Б — Split, cross-validation и утечки.md:3`

[Оглавление](../01-data-features/README.md). Пункт 2 плана. **Происхождение:** дополнительная глава по плану; в чатах встречаются held-out evaluation и contamination, но полного разбора split нет.

**Источник:** `01-data/Учебные конспекты/Б — Split, cross-validation и утечки.md:5`

**Уровень:** общее ядро. **Предпосылки:** D1, индексированные примеры. **Используют:** вся практика; F2/F9, CV/LLM preprocessing и model-selection protocols.

**Источник:** `01-data/Учебные конспекты/В — Пропуски, выбросы и качество разметки.md:1`

# В. Очистка данных без уничтожения полезного сигнала

**Источник:** `01-data/Учебные конспекты/В — Пропуски, выбросы и качество разметки.md:3`

[Оглавление](../01-data-features/README.md). Пункт 3 плана.

**Источник:** `01-data/Учебные конспекты/В — Пропуски, выбросы и качество разметки.md:5`

**Происхождение:** дополнительный материал по действующему плану. В чатах CV/LLM обсуждались observations и annotations, но полного курса очистки табличных данных нет. Примеры ниже — учебные расчёты, а не результаты аудита пользовательского набора.

**Источник:** `01-data/Учебные конспекты/В — Пропуски, выбросы и качество разметки.md:11`

<!-- source-depth-data-В -->

**Источник:** `01-data/Учебные конспекты/В — Пропуски, выбросы и качество разметки.md:13`

**Уровень:** общее ядро. **Предпосылки:** D1–D2. **Используют:** F3, F7 и анализ ошибок в каждом модуле.

**Источник:** `01-data/Учебные конспекты/Г — Дисбаланс, метрики, порог и калибровка.md:1`

# Г. Редкий класс: как связать вероятности с решением

**Источник:** `01-data/Учебные конспекты/Г — Дисбаланс, метрики, порог и калибровка.md:3`

[Оглавление](../01-data-features/README.md). Пункт 4 плана. **Происхождение:** дополнение по плану; CV-чат использует precision/recall/AP, но табличный дисбаланс и калибровку здесь раскрываем отдельно.

**Источник:** `01-data/Учебные конспекты/Г — Дисбаланс, метрики, порог и калибровка.md:5`

**Уровень:** общее ядро. **Предпосылки:** D1–D2 и вероятность. **Используют:** классификация CV/BERT, target encoding и production monitoring.

**Источник:** `01-data/Учебные конспекты/Д — Аугментации, синтетика и дедупликация.md:1`

# Д. Как расширять набор и не подменять независимую проверку

**Источник:** `01-data/Учебные конспекты/Д — Аугментации, синтетика и дедупликация.md:3`

[Оглавление](../01-data-features/README.md). Пункт 5 плана. **Происхождение:** дополнение по плану с опорой на CV-задачи S7 и teacher/distillation из S4. Здесь рассматривается качество данных; формулы MixUp/CutMix находятся в курсе регуляризации.

**Источник:** `01-data/Учебные конспекты/Д — Аугментации, синтетика и дедупликация.md:5`

**Уровень:** общее ядро; генерация синтетики — основа ветки по задаче. **Предпосылки:** D1–D4. **Используют:** D7–D8, CV и генерация признаков.

**Источник:** `01-data/Учебные конспекты/Е — Dataset, batching, загрузка и воспроизводимость.md:1`

# Е. Как доставить правильные данные модели и восстановить эксперимент

**Источник:** `01-data/Учебные конспекты/Е — Dataset, batching, загрузка и воспроизводимость.md:3`

[Оглавление](../01-data-features/README.md). Пункт 6 плана. **Происхождение:** дополнительный учебный материал; чат о serving описывает другие очереди, а здесь рассматривается обучающий data pipeline. Термины API сверены с [PyTorch DataLoader](https://docs.pytorch.org/docs/stable/data.html).

**Источник:** `01-data/Учебные конспекты/Е — Dataset, batching, загрузка и воспроизводимость.md:5`

**Уровень:** общее ядро. **Предпосылки:** D1–D2. **Используют:** каждый практический pipeline.

**Источник:** `01-data/Учебные конспекты/Ж — CV-данные, координаты и синхронные transforms.md:1`

# Ж. Почему правильная картинка может иметь неправильную аннотацию

**Источник:** `01-data/Учебные конспекты/Ж — CV-данные, координаты и синхронные transforms.md:3`

[Оглавление](../01-data-features/README.md). Пункт 7 плана. **Основа:** S7:1618–3712 — detection/segmentation, S7:3713–5276 — pose/OBB, S7:5277–6186 — tracking. Здесь собрана data-сторона этих задач; дополнены конвертации и streaming. [CV-конспекты](../04-computer-vision/README.md) раскрывают модели и метрики.

**Источник:** `01-data/Учебные конспекты/Ж — CV-данные, координаты и синхронные transforms.md:5`

**Уровень:** основа ветки CV; достаточно локального введения. **Предпосылки:** D1–D2; CV-архитектура не нужна. **Используют:** CV B–I.

**Источник:** `01-data/Учебные конспекты/З — Текстовые данные, токенизация и chunking.md:1`

# З. Как превратить документы и диалоги в корректные примеры

**Источник:** `01-data/Учебные конспекты/З — Текстовые данные, токенизация и chunking.md:3`

[Оглавление](../01-data-features/README.md). Пункт 8 плана. **Основа:** S2:1817–3404 — токены и обучение LLM; S3:228–339 — SFT на книгах и masks; S6:182–1373 — RAG, embeddings и хранение. Здесь объединены data-протоколы; детали моделей находятся в LLM-конспектах.

**Источник:** `01-data/Учебные конспекты/З — Текстовые данные, токенизация и chunking.md:5`

**Уровень:** основа ветки NLP/LLM; достаточно локального введения. **Предпосылки:** D1–D2. **Используют:** BERT и LLM apps/adaptation.

**Источник:** `01-data/Учебные конспекты/И — Drift, интервалы и достоверность вывода.md:1`

# И. Когда вчерашняя оценка перестаёт описывать будущее

**Источник:** `01-data/Учебные конспекты/И — Drift, интервалы и достоверность вывода.md:3`

[Оглавление](../01-data-features/README.md). Пункт 9 плана. **Происхождение:** дополнение к плану, связанное с evaluation и monitoring из LLM-чатов. Здесь статистический смысл отделён от устройства serving.

**Источник:** `01-data/Учебные конспекты/И — Drift, интервалы и достоверность вывода.md:5`

**Уровень:** углубление; для меняющихся production domains — основа ветки. **Предпосылки:** D1–D4, временные данные для time series. **Используют:** оценки всех модулей.

**Источник:** `02-regularization/Учебные конспекты/А — Обобщение, bias-variance и ограничения теории.md:1`

# А. Зачем ограничивать модель, которая уже хорошо обучается

**Источник:** `02-regularization/Учебные конспекты/А — Обобщение, bias-variance и ограничения теории.md:3`

[Оглавление](../02-regularization/README.md). Пункт 1 плана. **Происхождение:** дополнительный материал по плану; отдельного полного чата по регуляризации нет. В LLM-чатах встречаются KL, PEFT и generalization, но не этот теоретический курс.

**Источник:** `02-regularization/Учебные конспекты/А — Обобщение, bias-variance и ограничения теории.md:5`

**Уровень:** общее ядро. **Предпосылки:** empirical risk, train/test. **Используют:** R2–R10.

**Источник:** `02-regularization/Учебные конспекты/Б — L1, L2, группы и AdamW.md:1`

# Б. Штрафы на параметры: от геометрии до шага optimizer

**Источник:** `02-regularization/Учебные конспекты/Б — L1, L2, группы и AdamW.md:3`

[Оглавление](../02-regularization/README.md). Пункт 2 и практика AdamW из плана. **Происхождение:** дополнительный разбор; в чатах PEFT нет полного вывода ridge/lasso. AdamW сверён с [первоисточником](https://arxiv.org/abs/1711.05101).

**Источник:** `02-regularization/Учебные конспекты/Б — L1, L2, группы и AdamW.md:5`

**Уровень:** общее ядро. **Предпосылки:** R1, norms/gradients. **Используют:** R10, адаптация LLM.

**Источник:** `02-regularization/Учебные конспекты/В — Early stopping, learning rate и усреднение весов.md:1`

# В. Ограничивать обучение можно через его траекторию

**Источник:** `02-regularization/Учебные конспекты/В — Early stopping, learning rate и усреднение весов.md:3`

[Оглавление](../02-regularization/README.md). Пункт 3 и практика early stopping. **Происхождение:** дополнение по плану; обсуждённое в S3 merging не заменяет разбор checkpoints одного training run.

**Источник:** `02-regularization/Учебные конспекты/В — Early stopping, learning rate и усреднение весов.md:5`

**Уровень:** общее ядро. **Предпосылки:** R1–R2, training loop. **Используют:** R10, CV/NLP adaptation.

**Источник:** `02-regularization/Учебные конспекты/Г — Dropout, DropConnect, DropBlock и случайные ветви.md:1`

# Г. Шум в forward pass: какие части сети временно выключаются

**Источник:** `02-regularization/Учебные конспекты/Г — Dropout, DropConnect, DropBlock и случайные ветви.md:3`

[Оглавление](../02-regularization/README.md). Пункт 4 плана и практика dropout. **Происхождение:** дополнение; в чатах LoRA/Transformer dropout встречается контекстно, но семейство методов там не разобрано.

**Источник:** `02-regularization/Учебные конспекты/Г — Dropout, DropConnect, DropBlock и случайные ветви.md:5`

**Уровень:** общее ядро. **Предпосылки:** R1–R3. **Используют:** R10, CNN/Transformer branches.

**Источник:** `02-regularization/Учебные конспекты/Д — Нормализации, оси и train-eval statistics.md:1`

# Д. Нормализация: что усредняется и почему это важно

**Источник:** `02-regularization/Учебные конспекты/Д — Нормализации, оси и train-eval statistics.md:3`

[Оглавление](../02-regularization/README.md). Пункт 5 плана. **Основа:** блоки Transformer из S2 и backbones CV из S7; систематическое сравнение семейств — дополнение. Нормализация прежде всего меняет оптимизацию, а шум статистик может дополнительно регуляризовать.

**Источник:** `02-regularization/Учебные конспекты/Д — Нормализации, оси и train-eval statistics.md:5`

**Уровень:** общее ядро для deep nets. **Предпосылки:** R1/R4 и tensor shapes. **Используют:** Transformer/CV architectures, R10.

**Источник:** `02-regularization/Учебные конспекты/Е — Adversarial training, consistency и spectral norm.md:1`

# Е. Устойчивость predictions к допустимым возмущениям

**Источник:** `02-regularization/Учебные конспекты/Е — Adversarial training, consistency и spectral norm.md:3`

[Оглавление](../02-regularization/README.md). Пункт 6 плана. **Происхождение:** новое углубление; оно не приписывается исходным CV/LLM-чатам как пройденное занятие.

**Источник:** `02-regularization/Учебные конспекты/Е — Adversarial training, consistency и spectral norm.md:5`

**Уровень:** основа robustness ветки; spectral norm — углубление. **Предпосылки:** R1, R2, differentiable loss. **Используют:** R10, CV and NLP experiments.

**Источник:** `02-regularization/Учебные конспекты/Ж — Label smoothing, MixUp и CV-аугментации.md:1`

# Ж. Регуляризация через входы, targets и промежуточные representations

**Источник:** `02-regularization/Учебные конспекты/Ж — Label smoothing, MixUp и CV-аугментации.md:3`

[Оглавление](../02-regularization/README.md). Объединены близкие пункты 7–8 плана. **Основа:** CV-чат S7 задаёт задачи и annotations; полный вывод data-level regularization — дополнение. [Курс данных](../01-data-features/README.md) отвечает за provenance, leakage и проверку меток.

**Источник:** `02-regularization/Учебные конспекты/Ж — Label smoothing, MixUp и CV-аугментации.md:5`

**Уровень:** общее ядро для supervised deep learning; specialized policies are branch enrichment. **Предпосылки:** R1, task labels, D5. **Используют:** CV classifier/detection/segmentation and R10.

**Источник:** `02-regularization/Учебные конспекты/З — NLP, R-Drop и ограничения политики в RL.md:1`

# З. Регуляризация дискретных последовательностей и policy distributions

**Источник:** `02-regularization/Учебные конспекты/З — NLP, R-Drop и ограничения политики в RL.md:3`

[Оглавление](../02-regularization/README.md). Объединены пункты 9–10. **Основа:** S2 — embeddings/attention, S3 — SFT/RLHF; остальное отмечено как новое углубление. Полный RL-course находится в [lifecycle](../05-language-models/06-agent-learning/README.md).

**Источник:** `02-regularization/Учебные конспекты/З — NLP, R-Drop и ограничения политики в RL.md:5`

**Уровень:** основа branches NLP/RL; details can be skipped otherwise. **Предпосылки:** R1, R2; sequence masking/conditional probability; RL objective for RL portion. **Используют:** LLM adaptation/lifecycle.

**Источник:** `02-regularization/Учебные конспекты/И — Структурные prior, графы, физика и PEFT.md:1`

# И. Ограничение должно соответствовать структуре задачи

**Источник:** `02-regularization/Учебные конспекты/И — Структурные prior, графы, физика и PEFT.md:3`

[Оглавление](../02-regularization/README.md). Объединены пункты 11 и 13. **Основа:** S1–S3 подробно объясняют PEFT/LoRA; временные, графовые и физические penalties — дополнение по плану. Ссылка на FRESNEL/NICA в исходном плане не используется как доказательство вашего конкретного опыта или готового эксперимента.

**Источник:** `02-regularization/Учебные конспекты/И — Структурные prior, графы, физика и PEFT.md:5`

**Уровень:** основа named domain branches; PEFT is a cross-reference to adaptation, not a universal regularizer. **Предпосылки:** R1–R2; time series/graph/constraint context. **Используют:** domain-specific projects and model adaptation.

**Источник:** `02-regularization/Учебные конспекты/К — SAM, Lookahead и протокол выбора регуляризации.md:1`

# К. Как выбрать следующий эксперимент и не переобучиться на validation

**Источник:** `02-regularization/Учебные конспекты/К — SAM, Lookahead и протокол выбора регуляризации.md:3`

[Оглавление](../02-regularization/README.md). Объединены пункты 12 и 14 и итоговая практика плана. **Происхождение:** дополнительный материал; названия «передовые/SOTA» в плане рассматриваются как исследовательский обзор, а не обещание лучшего текущего метода для любой задачи.

**Источник:** `02-regularization/Учебные конспекты/К — SAM, Lookahead и протокол выбора регуляризации.md:5`

**Уровень:** общее ядро для выбора; optimizer variants — углубление. **Предпосылки:** R1–R9 by applicable intervention. **Используют:** finishing all branches.

**Источник:** `03-categorical-features/Учебные конспекты/А — Nominal, ordinal, one-hot и dummy encoding.md:1`

# А. Как превратить категорию в числа без выдуманного порядка

**Источник:** `03-categorical-features/Учебные конспекты/А — Nominal, ordinal, one-hot и dummy encoding.md:3`

[Оглавление](../01-data-features/README.md). Пункт 1 плана. **Происхождение:** самостоятельное дополнение по плану; полного чата по категориальным данным среди источников нет. Принцип embeddings связан с основами LLM, но табличные encoding protocols здесь новые.

**Источник:** `03-categorical-features/Учебные конспекты/А — Nominal, ordinal, one-hot и dummy encoding.md:5`

**Уровень:** общее ядро для табличного маршрута. **Предпосылки:** F1–F2, D2. **Используют:** F1/F3–F6 и все табличные models.

**Источник:** `03-categorical-features/Учебные конспекты/Б — Кардинальность, frequency и feature hashing.md:1`

# Б. Когда категорий слишком много для отдельных столбцов

**Источник:** `03-categorical-features/Учебные конспекты/Б — Кардинальность, frequency и feature hashing.md:3`

[Оглавление](../01-data-features/README.md). Пункт 2 плана. **Происхождение:** дополнение; технические приёмы не выдаются за уже обсуждённые в чатах.

**Источник:** `03-categorical-features/Учебные конспекты/Б — Кардинальность, frequency и feature hashing.md:5`

**Уровень:** общее ядро для табличного маршрута. **Предпосылки:** F1–F2, D2. **Используют:** F4–F6 и все табличные models.

**Источник:** `03-categorical-features/Учебные конспекты/В — Target encoding, cross-fitting, smoothing и WoE.md:1`

# В. Как использовать target statistics, не подглядывая в ответ

**Источник:** `03-categorical-features/Учебные конспекты/В — Target encoding, cross-fitting, smoothing и WoE.md:3`

[Оглавление](../01-data-features/README.md). Пункт 3 и практика вложенной валидации. **Происхождение:** подробное дополнение по плану. Это центральная глава курса: неправильная подготовка features делает последующее сравнение недостоверным.

**Источник:** `03-categorical-features/Учебные конспекты/В — Target encoding, cross-fitting, smoothing и WoE.md:5`

**Уровень:** общее ядро после F3. **Предпосылки:** D2–D4, условные средние/логарифмы. **Используют:** native boosting, production risk.

**Источник:** `03-categorical-features/Учебные конспекты/Г — Embeddings, gradients и перенос между задачами.md:1`

# Г. Когда representations категорий обучаются вместе с задачей

**Источник:** `03-categorical-features/Учебные конспекты/Г — Embeddings, gradients и перенос между задачами.md:3`

[Оглавление](../01-data-features/README.md). Пункт 4 и расчёт параметров из практики. **Основа:** S2:1621–1708 и 1830–1880 объясняет lookup token embeddings и переход к contextual states; перенос принципа на табличные entities и transfer между tasks — дополнение.

**Источник:** `03-categorical-features/Учебные конспекты/Г — Embeddings, gradients и перенос между задачами.md:5`

**Уровень:** основа ветки deep tabular. **Предпосылки:** F3, neural embeddings, backprop; D2. **Используют:** F5/F8, clustering, retrieval и downstream tasks.

**Источник:** `03-categorical-features/Учебные конспекты/Д — Нативные категории в LightGBM, CatBoost и XGBoost.md:1`

# Д. Как дерево делит категории на группы

**Источник:** `03-categorical-features/Учебные конспекты/Д — Нативные категории в LightGBM, CatBoost и XGBoost.md:3`

[Оглавление](../01-data-features/README.md). Пункт 5 плана. **Происхождение:** дополнение по плану. API-границы сверены с официальными страницами 17 октября 2026; при реализации нужно фиксировать installed version и device/mode.

**Источник:** `03-categorical-features/Учебные конспекты/Д — Нативные категории в LightGBM, CatBoost и XGBoost.md:5`

**Уровень:** общее ядро табличной ветки; interaction networks — углубление. **Предпосылки:** F3–F4 и дерево решений. **Используют:** F5/F8, dataset baseline.

**Источник:** `03-categorical-features/Учебные конспекты/Е — Feature crosses, FM и field-aware взаимодействия.md:1`

# Е. Когда сигнал живёт в сочетании категорий

**Источник:** `03-categorical-features/Учебные конспекты/Е — Feature crosses, FM и field-aware взаимодействия.md:3`

[Оглавление](../01-data-features/README.md). Пункт 6 плана. **Происхождение:** дополнение по плану; FFM — обзор с точной механикой, а не заявленный вами реализованный recommender.

**Источник:** `03-categorical-features/Учебные конспекты/Е — Feature crosses, FM и field-aware взаимодействия.md:5`

**Уровень:** общее ядро табличной ветки; interaction networks — углубление. **Предпосылки:** F3–F4 и дерево решений. **Используют:** F5/F8, dataset baseline.

**Источник:** `03-categorical-features/Учебные конспекты/Ж — Multi-hot, counts и pooling наборов категорий.md:1`

# Ж. Если одно наблюдение имеет несколько значений field

**Источник:** `03-categorical-features/Учебные конспекты/Ж — Multi-hot, counts и pooling наборов категорий.md:3`

[Оглавление](../01-data-features/README.md). Пункт 7 плана. **Происхождение:** дополнение. Сходство с token embeddings из LLM относится к lookup, но set pooling и sequence encoding задают разные модели.

**Источник:** `03-categorical-features/Учебные конспекты/Ж — Multi-hot, counts и pooling наборов категорий.md:5`

**Уровень:** основа ветки по типу данных. **Предпосылки:** F1–F3, D2/D9. **Используют:** recommendation, event, geographic and temporal tasks.

**Источник:** `03-categorical-features/Учебные конспекты/З — Missing, unknown, rare и стабильный словарь.md:1`

# З. Отсутствующая категория — отдельное состояние данных

**Источник:** `03-categorical-features/Учебные конспекты/З — Missing, unknown, rare и стабильный словарь.md:3`

[Оглавление](../01-data-features/README.md). Пункт 8 плана и inference edge cases. **Происхождение:** дополнение; общая теория missingness находится в [данных](../01-data-features/README.md).

**Источник:** `03-categorical-features/Учебные конспекты/З — Missing, unknown, rare и стабильный словарь.md:5`

**Уровень:** основа ветки для missing/unknown handling. **Предпосылки:** D3, F1–F3. **Используют:** D3, F3/F7 и production vocabulary/fallback monitoring.

**Источник:** `03-categorical-features/Учебные конспекты/И — Временные статистики и доступность метки.md:1`

# И. Encoding по прошлому: учитывать нужно и время прихода labels

**Источник:** `03-categorical-features/Учебные конспекты/И — Временные статистики и доступность метки.md:3`

[Оглавление](../01-data-features/README.md). Пункт 9 и итоговый protocol. **Происхождение:** дополнение по плану; random permutation CatBoost из главы В не заменяет временную постановку.

**Источник:** `03-categorical-features/Учебные конспекты/И — Временные статистики и доступность метки.md:5`

**Уровень:** основа ветки по типу данных. **Предпосылки:** F1–F3, D2/D9. **Используют:** F7/D9, временные feature pipelines и drift monitoring.

**Источник:** `04-clustering/Учебные конспекты/А — Похожесть, расстояния и постановка кластеризации.md:1`

# А. Кластер существует относительно выбранного представления

**Источник:** `04-clustering/Учебные конспекты/А — Похожесть, расстояния и постановка кластеризации.md:3`

[Оглавление](../03-clustering/README.md). Пункт 1 плана. **Происхождение:** дополнение по плану; в чатах есть embeddings/retrieval, но отдельного полного курса кластеризации нет.

**Источник:** `04-clustering/Учебные конспекты/А — Похожесть, расстояния и постановка кластеризации.md:5`

**Уровень:** общее ядро. **Предпосылки:** vectors/matrices. **Используют:** все остальные главы и прикладные модели.

**Источник:** `04-clustering/Учебные конспекты/Б — K-Means, иерархии, плотность и GMM.md:1`

# Б. Разные алгоритмы ищут разные виды групп

**Источник:** `04-clustering/Учебные конспекты/Б — K-Means, иерархии, плотность и GMM.md:3`

[Оглавление](../03-clustering/README.md). Пункт 2 и практики K-Means/GMM/плотности. **Происхождение:** дополнительная глава по плану; все числа ниже — toy calculations.

**Источник:** `04-clustering/Учебные конспекты/Б — K-Means, иерархии, плотность и GMM.md:5`

**Уровень:** общее ядро. **Предпосылки:** C1, expectation/variance. **Используют:** C3–C4/C9.

**Источник:** `04-clustering/Учебные конспекты/В — Выбор K, gap statistic и устойчивость.md:1`

# В. Сколько групп нужно задаче

**Источник:** `04-clustering/Учебные конспекты/В — Выбор K, gap statistic и устойчивость.md:3`

[Оглавление](../03-clustering/README.md). Пункт 3 и практика stability. **Происхождение:** дополнение; ни одно число K не объявляется истинным только по красивой картинке.

**Источник:** `04-clustering/Учебные конспекты/В — Выбор K, gap statistic и устойчивость.md:5`

**Уровень:** общее ядро. **Предпосылки:** C2 and sampling. **Используют:** any result reporting.

**Источник:** `04-clustering/Учебные конспекты/Г — Purity, ARI, NMI и внутренние метрики.md:1`

# Г. Что именно измеряет качество partition

**Источник:** `04-clustering/Учебные конспекты/Г — Purity, ARI, NMI и внутренние метрики.md:3`

[Оглавление](../03-clustering/README.md). Пункт 4 и практика оценки. **Происхождение:** дополнение по плану; labels можно использовать для независимой проверки, даже если fit был unsupervised.

**Источник:** `04-clustering/Учебные конспекты/Г — Purity, ARI, NMI и внутренние метрики.md:5`

**Уровень:** общее ядро. **Предпосылки:** C1–C3. **Используют:** C5–C9.

**Источник:** `04-clustering/Учебные конспекты/Д — Frozen embeddings, DEC и IDEC.md:1`

# Д. Как обучать представление под clustering и не принять уверенность за истину

**Источник:** `04-clustering/Учебные конспекты/Д — Frozen embeddings, DEC и IDEC.md:3`

[Оглавление](../03-clustering/README.md). Пункт 5 плана. **Связь с чатами:** S2 объясняет hidden states, S6 — retrieval embeddings; distinction между representation и clustering раскрывается здесь как дополнение.

**Источник:** `04-clustering/Учебные конспекты/Д — Frozen embeddings, DEC и IDEC.md:5`

**Уровень:** основа ветки. **Предпосылки:** C1–C4, neural training basics. **Используют:** C6/C7 and research route.

**Источник:** `04-clustering/Учебные конспекты/Е — Image clustering, superpixels и поток новых объектов.md:1`

# Е. Clustering изображений и pixels: разные уровни задачи

**Источник:** `04-clustering/Учебные конспекты/Е — Image clustering, superpixels и поток новых объектов.md:3`

[Оглавление](../03-clustering/README.md). Пункт 6 плана. **Основа:** S7:207–1617 — backbones/ViT, S7:2469–3080 — segmentation. Clustering/SSL/SLIC — дополнение. Упоминание «вашего SSL-проекта» в плане не означает подтверждённый здесь выполненный проект.

**Источник:** `04-clustering/Учебные конспекты/Е — Image clustering, superpixels и поток новых объектов.md:5`

**Уровень:** основа CV branch. **Предпосылки:** C1–C5 or suitable frozen encoder. **Используют:** downstream CV discovery.

**Источник:** `04-clustering/Учебные конспекты/Ж — Текстовые representations, LDA, NMF и Top2Vec.md:1`

# Ж. Темы, смысл и lexical similarity документов

**Источник:** `04-clustering/Учебные конспекты/Ж — Текстовые representations, LDA, NMF и Top2Vec.md:3`

[Оглавление](../03-clustering/README.md). Пункт 7 плана. **Основа:** S6:182–1373 различает retrieval embeddings и внутренние LLM states; topic models здесь — дополнение.

**Источник:** `04-clustering/Учебные конспекты/Ж — Текстовые representations, LDA, NMF и Top2Vec.md:5`

**Уровень:** основа NLP branch; Transformer training not required. **Предпосылки:** C1, bag-of-words and linear algebra. **Используют:** discovery/topic work.

**Источник:** `04-clustering/Учебные конспекты/З — PCA, t-SNE, UMAP и геометрия проекции.md:1`

# З. Снижение размерности может помочь и может создать иллюзию

**Источник:** `04-clustering/Учебные конспекты/З — PCA, t-SNE, UMAP и геометрия проекции.md:3`

[Оглавление](../03-clustering/README.md). Пункт 8 плана; читать после А до основных algorithms. **Происхождение:** дополнение к обсуждённым embeddings, с отдельным разбором preprocessing и visualization.

**Источник:** `04-clustering/Учебные конспекты/З — PCA, t-SNE, UMAP и геометрия проекции.md:5`

**Уровень:** общее ядро как diagnostic; t-SNE/UMAP theory — углубление. **Предпосылки:** C1, covariance/SVD. **Используют:** C5–C7 and reports.

**Источник:** `04-clustering/Учебные конспекты/И — Аномалии, псевдометки, IVF и итоговый pipeline.md:1`

# И. Clustering как часть решения, а не самоцель

**Источник:** `04-clustering/Учебные конспекты/И — Аномалии, псевдометки, IVF и итоговый pipeline.md:3`

[Оглавление](../03-clustering/README.md). Пункт 9 и итоговая практика. **Основа:** S6 обсуждает retrieval и embeddings; использование IVF/псевдометок/режимов — дополнительное углубление. Исторические ссылки плана на проекты не считаются подтверждением ваших выполненных результатов.

**Источник:** `04-clustering/Учебные конспекты/И — Аномалии, псевдометки, IVF и итоговый pipeline.md:5`

**Уровень:** основа ветки by task. **Предпосылки:** C1–C4 and chosen application. **Используют:** anomaly discovery, pseudo-label, ANN routing.

**Источник:** `05-computer-vision/Учебные конспекты/00 — Изображение, тензоры и общий путь модели.md:1`

# V0. Изображение, тензоры и общий путь модели

**Источник:** `05-computer-vision/Учебные конспекты/00 — Изображение, тензоры и общий путь модели.md:3`

[Оглавление](../04-computer-vision/README.md) · [Сценарий модуля](../archive/2026-10-20-before-rebuild/05-computer-vision/План — компьютерное зрение.md)

**Источник:** `05-computer-vision/Учебные конспекты/00 — Изображение, тензоры и общий путь модели.md:5`

**Уровень:** общее ядро. **Предпосылки:** массивы/тензоры, умножение матриц и базовый gradient descent. **Используют:** V1–V9. **Задача главы:** проследить изображение от файла до выхода модели и научиться проверять формы данных на каждом этапе. Эта глава дополняет девять задач CV; она не предполагает знания LLM или BERT.

**Источник:** `05-computer-vision/Учебные конспекты/А — Классификация, backbone и математика ViT.md:1`

# А. Классификация изображений, backbone и ViT

**Источник:** `05-computer-vision/Учебные конспекты/А — Классификация, backbone и математика ViT.md:3`

[Оглавление](../04-computer-vision/README.md). Пункт 1 плана; отдельное углубление из чата о patch embedding.

**Источник:** `05-computer-vision/Учебные конспекты/А — Классификация, backbone и математика ViT.md:5`

**Основа:** S7, строки 207–1617: классификация, вопрос «что такое backbone», просьба подробно разобрать шаг проекции патчей с математикой. Из экспорта исключены английские служебные вставки и дубли повреждённых формул. **Углубление/исправления:** единая NCHW-нотация, точная эквивалентность Conv2d, equivariance вместо ошибочного «permutation-invariant patch embedding», ограничения интуиции ResNet и ViT.

**Источник:** `05-computer-vision/Учебные конспекты/А — Классификация, backbone и математика ViT.md:7`

**Уровень:** общее ядро. **Предпосылки:** V0, D2/D4/D5. **Используют:** V2–V9 and clustering.

**Источник:** `05-computer-vision/Учебные конспекты/Б — Detection, IoU, AP, NMS и архитектуры.md:1`

# Б. Object detection: от boxes к AP и устройству детектора

**Источник:** `05-computer-vision/Учебные конспекты/Б — Detection, IoU, AP, NMS и архитектуры.md:3`

[Оглавление](../04-computer-vision/README.md). Пункт 2 плана.

**Источник:** `05-computer-vision/Учебные конспекты/Б — Detection, IoU, AP, NMS и архитектуры.md:5`

**Основа:** S7, строки 1618–2467. Сохранён порядок: постановка → IoU/matching/AP → NMS → two-stage/one-stage → YOLO/DETR. **Исправления:** YOLOv1 не anchor-based как v2/v3; one-stage не всегда быстрее любой two-stage модели; наличие версии YOLO26 не означает обязательный NMS-free режим. **Углубление:** ручной AP с явной convention, assignment и box losses.

**Источник:** `05-computer-vision/Учебные конспекты/Б — Detection, IoU, AP, NMS и архитектуры.md:7`

**Уровень:** общее ядро. **Предпосылки:** V0–V1, D7 box coordinates. **Используют:** V4–V7/V9.

**Источник:** `05-computer-vision/Учебные конспекты/В — Semantic segmentation, FCN, U-Net и DeepLab.md:1`

# В. Semantic segmentation: классы пикселей и восстановление деталей

**Источник:** `05-computer-vision/Учебные конспекты/В — Semantic segmentation, FCN, U-Net и DeepLab.md:3`

[Оглавление](../04-computer-vision/README.md). Пункт 3 плана.

**Источник:** `05-computer-vision/Учебные конспекты/В — Semantic segmentation, FCN, U-Net и DeepLab.md:5`

**Основа:** S7, строки 2469–3080. Сохранена последовательность pixel task → masks/metrics → FCN → U-Net → dilated convolution/DeepLab. **Углубление:** ручная confusion matrix, CE/Dice, ignore-index и разница semantic auxiliary loss/отдельной semantic head.

**Источник:** `05-computer-vision/Учебные конспекты/В — Semantic segmentation, FCN, U-Net и DeepLab.md:7`

**Уровень:** общее ядро. **Предпосылки:** V0–V1, pixel labels and D7 mask transforms. **Используют:** V4/V8 and spatial outputs.

**Источник:** `05-computer-vision/Учебные конспекты/Г — Instance segmentation, RoIAlign и prototype masks.md:1`

# Г. Instance segmentation: отдельная маска для каждого объекта

**Источник:** `05-computer-vision/Учебные конспекты/Г — Instance segmentation, RoIAlign и prototype masks.md:3`

[Оглавление](../04-computer-vision/README.md). Пункт 4 плана.

**Источник:** `05-computer-vision/Учебные конспекты/Г — Instance segmentation, RoIAlign и prototype masks.md:5`

**Основа:** S7, строки 3081–3712. Сохранены Mask R-CNN, RoIAlign, YOLACT и YOLO-seg. **Исправления:** one-stage mask prediction может включать boxes; RoIAlign не выдаёт автоматически один vector; совместное обучение не гарантирует улучшение каждой ветви. **Углубление:** числовая bilinear interpolation, матричные формы prototypes, box/mask AP и panoptic quality.

**Источник:** `05-computer-vision/Учебные конспекты/Г — Instance segmentation, RoIAlign и prototype masks.md:7`

**Уровень:** общее ядро. **Предпосылки:** V2–V3. **Используют:** V7/V9 and object-centric analysis.

**Источник:** `05-computer-vision/Учебные конспекты/Д — Pose, heatmaps, OKS и группировка keypoints.md:1`

# Д. Pose estimation: ключевые точки и принадлежность человеку

**Источник:** `05-computer-vision/Учебные конспекты/Д — Pose, heatmaps, OKS и группировка keypoints.md:3`

[Оглавление](../04-computer-vision/README.md). Пункт 5 плана.

**Источник:** `05-computer-vision/Учебные конспекты/Д — Pose, heatmaps, OKS и группировка keypoints.md:5`

**Основа:** S7, строки 3713–4523. **Исправления:** YOLO-pose one-stage не равен «detector → отдельный crop model»; COCO v>0 включает размеченные occluded points, а не только видимые; OKS sigma conventions не смешиваются. **Углубление:** ручной Gaussian target, PCK/OKS и inverse crop transform.

**Источник:** `05-computer-vision/Учебные конспекты/Д — Pose, heatmaps, OKS и группировка keypoints.md:7`

**Уровень:** основа pose branch. **Предпосылки:** V0–V2 and coordinate geometry. **Используют:** V7 and action tasks.

**Источник:** `05-computer-vision/Учебные конспекты/Е — OBB, поворот, periodicity и rotated IoU.md:1`

# Е. Oriented boxes: геометрия, угол и неоднозначность

**Источник:** `05-computer-vision/Учебные конспекты/Е — OBB, поворот, periodicity и rotated IoU.md:3`

[Оглавление](../04-computer-vision/README.md). Пункт 6 плана.

**Источник:** `05-computer-vision/Учебные конспекты/Е — OBB, поворот, periodicity и rotated IoU.md:5`

**Основа:** S7, строки 4524–5276. Сохранены representations OBB, corners, rotated IoU, periodicity, Gaussian losses и YOLO-OBB. **Исправления:** axis-aligned box применим, но может быть неэффективен; rotated IoU не «недифференцируем по углу везде»; произвольный четырёхугольник нельзя без потерь заменить пятью параметрами прямоугольника. **Углубление:** ручной поворот и cyclic loss.

**Источник:** `05-computer-vision/Учебные конспекты/Е — OBB, поворот, periodicity и rotated IoU.md:7`

**Уровень:** основа detection branch. **Предпосылки:** V2 and coordinate transforms. **Используют:** aerial/document/video detection.

**Источник:** `05-computer-vision/Учебные конспекты/Ж — Tracking, Kalman, association, MOTA, IDF1 и HOTA.md:1`

# Ж. Multi-object tracking: сохранить identity во времени

**Источник:** `05-computer-vision/Учебные конспекты/Ж — Tracking, Kalman, association, MOTA, IDF1 и HOTA.md:3`

[Оглавление](../04-computer-vision/README.md). Пункт 7 плана.

**Источник:** `05-computer-vision/Учебные конспекты/Ж — Tracking, Kalman, association, MOTA, IDF1 и HOTA.md:5`

**Основа:** S7, строки 5277–6186. Сохранены detector→association, SORT/DeepSORT, ByteTrack, BoT-SORT и trajectory metrics. **Исправления:** MOTA учитывает IDS, IDF1 зависит и от detection errors; HOTA содержит квадратный корень и protocol aggregation; tracker не может гарантированно восстановить ID после любой occlusion. **Углубление:** ручной assignment, Kalman correction и разбор ID metrics.

**Источник:** `05-computer-vision/Учебные конспекты/Ж — Tracking, Kalman, association, MOTA, IDF1 и HOTA.md:7`

**Уровень:** основа temporal CV branch. **Предпосылки:** V2, V0, D7 sequence splits. **Используют:** V5/V9 and video systems.

**Источник:** `05-computer-vision/Учебные конспекты/З — Depth, scale ambiguity, DPT и метрики.md:1`

# З. Depth estimation: глубина, масштаб и протокол оценки

**Источник:** `05-computer-vision/Учебные конспекты/З — Depth, scale ambiguity, DPT и метрики.md:3`

[Оглавление](../04-computer-vision/README.md). Пункт 8 плана.

**Источник:** `05-computer-vision/Учебные конспекты/З — Depth, scale ambiguity, DPT и метрики.md:5`

**Основа:** S7, строки 6187–6969. Сохранены monocular cues, ambiguity, relative/metric output, MiDaS/DPT, Depth Anything и regression metrics. **Исправления:** disparity не буквально 1/Z без scale, δ2/δ3 не «25²%/25³%», log-scale invariant loss не тождественна произвольной affine alignment в depth. **Углубление:** camera geometry, ручной пример raw/aligned metrics и validity protocol.

**Источник:** `05-computer-vision/Учебные конспекты/З — Depth, scale ambiguity, DPT и метрики.md:7`

**Уровень:** основа depth branch. **Предпосылки:** V0–V1, regression and geometry. **Используют:** 3D and robotics branches.

**Источник:** `05-computer-vision/Учебные конспекты/И — CLIP, open-vocabulary detection и YOLO-World.md:1`

# И. Open-vocabulary detection: текст задаёт, что искать

**Источник:** `05-computer-vision/Учебные конспекты/И — CLIP, open-vocabulary detection и YOLO-World.md:3`

[Оглавление](../04-computer-vision/README.md). Пункт 9 плана.

**Источник:** `05-computer-vision/Учебные конспекты/И — CLIP, open-vocabulary detection и YOLO-World.md:5`

**Основа:** S7, строки 6970–7816: shared image-text space, CLIP, OVD metrics, region methods, GLIP, Grounding DINO и YOLO-World. **Исправления:** open vocabulary не гарантирует любой произвольный объект/язык; global CLIP не локализует автоматически; название «YOLO26-OV» не заменяет проверку конкретной модели. **Углубление:** symmetric contrastive loss, prompt sets, base/novel protocol и threshold diagnosis.

**Источник:** `05-computer-vision/Учебные конспекты/И — CLIP, open-vocabulary detection и YOLO-World.md:7`

**Уровень:** углубление after V2; basis for multimodal branch. **Предпосылки:** V2, text/image embeddings, D2/D4. **Используют:** promptable annotation and open-world systems.

**Источник:** `06-llm/01-foundations/Учебные конспекты/А — Токены, embeddings и обучение следующему токену.md:1`

# А. От текста к обучающей задаче: токены, embeddings, позиции

**Источник:** `06-llm/01-foundations/Учебные конспекты/А — Токены, embeddings и обучение следующему токену.md:3`

[Оглавление](../05-language-models/01-foundations/README.md). Соответствует пункту 1 плана.

**Источник:** `06-llm/01-foundations/Учебные конспекты/А — Токены, embeddings и обучение следующему токену.md:5`

**Основа:** S1, `blank.pdf`, страницы 4–5 и 13–15; S2, `chat-План изучения темы.txt`, строки 1817–3404. В обсуждении устройства LLM неоднократно смешивались слова «токен», «вектор» и «состояние». Здесь они разделены. **Углубление:** подробный сдвиг targets, числовой CE и вывод градиента добавлены для самостоятельного чтения. Ссылки на оригиналы — в [оглавлении](../05-language-models/01-foundations/README.md).

**Источник:** `06-llm/01-foundations/Учебные конспекты/А — Токены, embeddings и обучение следующему токену.md:7`

**Уровень:** общее ядро. **Предпосылки:** векторы, матричное умножение, условная вероятность. **Используют:** A2–A5, все ветки ниже.

**Источник:** `06-llm/01-foundations/Учебные конспекты/Б — RNN и ограничения рекуррентного состояния.md:1`

# Б. RNN: как хранится история и почему это трудно

**Источник:** `06-llm/01-foundations/Учебные конспекты/Б — RNN и ограничения рекуррентного состояния.md:3`

[Оглавление](../05-language-models/01-foundations/README.md). Соответствует пункту 2 плана.

**Источник:** `06-llm/01-foundations/Учебные конспекты/Б — RNN и ограничения рекуррентного состояния.md:5`

**Основа:** S1, страницы 4–7; S2, строки 1817–3404. В чатах переход от RNN к Transformer понадобился для ответа «что находится внутри LLM и куда вставлять LoRA». **Углубление:** ручной recurrent forward, произведение Jacobians и связь attention с bottleneck. Исходная фраза о том, что RNN «невозможно распараллелить», уточнена: зависимость касается последовательных шагов обычной рекурренции.

**Источник:** `06-llm/01-foundations/Учебные конспекты/Б — RNN и ограничения рекуррентного состояния.md:7`

**Уровень:** углубление; можно пропустить. **Предпосылки:** A1. **Используют:** историческое сравнение с Transformer.

**Источник:** `06-llm/01-foundations/Учебные конспекты/В — Transformer, causal mask и устройство блока.md:1`

# В. Transformer: из каких узлов состоит LLM

**Источник:** `06-llm/01-foundations/Учебные конспекты/В — Transformer, causal mask и устройство блока.md:3`

[Оглавление](../05-language-models/01-foundations/README.md). Пункт 3 плана; математический разбор attention — в следующей главе.

**Источник:** `06-llm/01-foundations/Учебные конспекты/В — Transformer, causal mask и устройство блока.md:5`

**Основа:** S1, страницы 4–14; S2, строки 1817–3404 и 3413–3487. Главный вопрос чата: «Что представляет из себя LLM, где attention-слои и куда мы вставляем матрицы?» **Углубление:** путь тензора через residual/normalization/MLP, различие двух масок и пример gated MLP. Формы в примерах учебные; названия модулей не универсальны для всех архитектур.

**Источник:** `06-llm/01-foundations/Учебные конспекты/В — Transformer, causal mask и устройство блока.md:7`

**Уровень:** общее ядро. **Предпосылки:** A1; A2 не обязателен. **Используют:** A4–A5 и все генеративные ветки; BERT использует encoder часть.

**Источник:** `06-llm/01-foundations/Учебные конспекты/Г — QKV, multi-head attention и точный backward.md:1`

# Г. Q/K/V: полный маленький forward и backward

**Источник:** `06-llm/01-foundations/Учебные конспекты/Г — QKV, multi-head attention и точный backward.md:3`

[Оглавление](../05-language-models/01-foundations/README.md). Пункт 4 плана.

**Источник:** `06-llm/01-foundations/Учебные конспекты/Г — QKV, multi-head attention и точный backward.md:5`

**Основа:** S1, страницы 9–13; S2, строки 1817–3404. В чатах спрашивалось, что именно представляют Q/K/V и как проходит градиент. **Исправление C02:** производная softmax связывает все координаты строки; запись только $s(1-s)$ не даёт полного backward. **Углубление:** ниже приведён согласованный числовой пример, который можно пересчитать вручную.

**Источник:** `06-llm/01-foundations/Учебные конспекты/Г — QKV, multi-head attention и точный backward.md:9`

**Уровень:** общее ядро — формы и forward; подробный backward — углубление. **Предпосылки:** A1, A3, матрицы и производные. **Используют:** объяснение attention, BERT и inference.

**Источник:** `06-llm/01-foundations/Учебные конспекты/Д — Hidden states, LM head и границы интерпретации.md:1`

# Д. Hidden states и logits: что именно выходит из слоёв

**Источник:** `06-llm/01-foundations/Учебные конспекты/Д — Hidden states, LM head и границы интерпретации.md:3`

[Оглавление](../05-language-models/01-foundations/README.md). Пункт 5 плана.

**Источник:** `06-llm/01-foundations/Учебные конспекты/Д — Hidden states, LM head и границы интерпретации.md:5`

**Основа:** S1, страницы 13–15; S2, строки 1817–3404. Сохранён вопрос обсуждения: что находится между token embedding и выбранным следующим токеном? **Углубление:** числовой LM head, weight tying, связь контекстного hidden state с retrieval embedding и границы интерпретаций слоёв. Уточнения соответствуют C04/C14 из реестра исправлений.

**Источник:** `06-llm/01-foundations/Учебные конспекты/Д — Hidden states, LM head и границы интерпретации.md:7`

**Уровень:** общее ядро — интерфейсы; анализ слоёв — углубление. **Предпосылки:** A1, A3–A4. **Используют:** adaptation и представления encoder.

**Источник:** `06-llm/02-adaptation/Учебные конспекты/А — Выбор адаптации, PEFT и память обучения.md:1`

# А. Зачем адаптировать модель и откуда берётся расход памяти

**Источник:** `06-llm/02-adaptation/Учебные конспекты/А — Выбор адаптации, PEFT и память обучения.md:3`

[Оглавление](../05-language-models/03-adaptation/README.md). Пункты 1 и 14A плана.

**Источник:** `06-llm/02-adaptation/Учебные конспекты/А — Выбор адаптации, PEFT и память обучения.md:5`

**Основа:** S2, строки 4–165; S3, строки 428–505. Сохранён ход чата: full fine-tuning → память и забывание → frozen features → PEFT. **Углубление:** выбор вмешательства по типу ошибки и раздельный memory accounting. Исправлены универсальные «14 bytes/parameter», обещания отсутствия forgetting и оценки адаптеров фиксированного размера.

**Источник:** `06-llm/02-adaptation/Учебные конспекты/А — Выбор адаптации, PEFT и память обучения.md:7`

**Уровень:** Общее ядро ветки. **Предпосылки:** A1/A3/A5 and baseline evaluation. **Используют:** C2–C10.

**Источник:** `06-llm/02-adaptation/Учебные конспекты/Б — LoRA, target modules и два шага обучения.md:1`

# Б. LoRA: куда добавляются A/B и как они обучаются

**Источник:** `06-llm/02-adaptation/Учебные конспекты/Б — LoRA, target modules и два шага обучения.md:3`

[Оглавление](../05-language-models/03-adaptation/README.md). Пункт 2 и протокол 14C плана; варианты семейства — в главе В.

**Источник:** `06-llm/02-adaptation/Учебные конспекты/Б — LoRA, target modules и два шага обучения.md:5`

**Основа:** S2, строки 169–500, 1817–3404 и 3413–3487; S1, страницы 2–3, 8–9 и 15–17. Ключевой запрос пользователя: «я так и не понял, куда мы вставляем эти матрицы». **Исправления:** единая ориентация матриц, r не число параметров, точный loss первого шага, target modules не отдельные головы. **Углубление:** второй шаг SGD и прямое сравнение с merged weight.

**Источник:** `06-llm/02-adaptation/Учебные конспекты/Б — LoRA, target modules и два шага обучения.md:7`

**Уровень:** Общее ядро ветки. **Предпосылки:** C1, A3–A5. **Используют:** C3/C5/C7/C10.

**Источник:** `06-llm/02-adaptation/Учебные конспекты/В — DoRA, AdaLoRA, LoRA+ и QLoRA.md:1`

# В. Семейство LoRA: какие ограничения меняют DoRA, AdaLoRA и QLoRA

**Источник:** `06-llm/02-adaptation/Учебные конспекты/В — DoRA, AdaLoRA, LoRA+ и QLoRA.md:3`

[Оглавление](../05-language-models/03-adaptation/README.md). Пункт 2, граница quantization из пункта 7 и протокол 14C.

**Источник:** `06-llm/02-adaptation/Учебные конспекты/В — DoRA, AdaLoRA, LoRA+ и QLoRA.md:5`

**Основа:** S2, строки 3413–4587 (DoRA), 4591–5453 (AdaLoRA), 5457–5549 (QLoRA); LoRA+ кратко упоминалась в 4–165. Сохранены вопросы «m — просто число?», «а AdaDoRA делали?» и замечание, что QLoRA была названа, но ещё не объяснена. **Углубление:** broadcasting записан явно, приведены независимые примеры magnitude/direction, бюджет рангов и стоимость quantization metadata.

**Источник:** `06-llm/02-adaptation/Учебные конспекты/В — DoRA, AdaLoRA, LoRA+ и QLoRA.md:7`

**Уровень:** Основа PEFT-ветки. **Предпосылки:** C2; quantization in D3. **Используют:** C5/C8/C10 and deployment.

**Источник:** `06-llm/02-adaptation/Учебные конспекты/Г — Adapters, prefix tuning, soft prompts и BitFit.md:1`

# Г. PEFT за пределами LoRA: что именно меняется

**Источник:** `06-llm/02-adaptation/Учебные конспекты/Г — Adapters, prefix tuning, soft prompts и BitFit.md:3`

[Оглавление](../05-language-models/03-adaptation/README.md). Пункт 3 плана.

**Источник:** `06-llm/02-adaptation/Учебные конспекты/Г — Adapters, prefix tuning, soft prompts и BitFit.md:5`

**Основа:** S3, строки 49–224. Сохранён вопрос: «лучше раскрыть суть Prefix Tuning и BitFit: как и почему работают, что и куда добавляем». **Исправление:** soft prompt на входе и layerwise prefix K/V разделены; исходное описание P-Tuning v2 «только первый слой» не переносится. Универсальные проценты качества/latency удалены. **Углубление:** формы и один attention-пример с prefix.

**Источник:** `06-llm/02-adaptation/Учебные конспекты/Г — Adapters, prefix tuning, soft prompts и BitFit.md:7`

**Уровень:** Основа PEFT-ветки. **Предпосылки:** C1, A3–A5. **Используют:** C8/C10 and adapter serving.

**Источник:** `06-llm/02-adaptation/Учебные конспекты/Д — SFT, данные из книг, masks и training objective.md:1`

# Д. SFT: чему именно учат демонстрации

**Источник:** `06-llm/02-adaptation/Учебные конспекты/Д — SFT, данные из книг, masks и training objective.md:3`

[Оглавление](../05-language-models/03-adaptation/README.md). Пункт 4 и протокол 14B; связи с 14A.

**Источник:** `06-llm/02-adaptation/Учебные конспекты/Д — SFT, данные из книг, masks и training objective.md:5`

**Основа:** S3, строки 228–339 и 428–505. Сохранён вопрос пользователя: «можно использовать соответствующие книги, тексты из них для SFT? Там иногда есть диалоги». Категоричный ответ исходного чата заменён разбором цели и objective. **Углубление:** пример labels, нормировка token loss, truncation/packing и воспроизводимый первый эксперимент.

**Источник:** `06-llm/02-adaptation/Учебные конспекты/Д — SFT, данные из книг, masks и training objective.md:7`

**Уровень:** Общее ядро адаптации. **Предпосылки:** A1/A3/A5, D8, C1. **Используют:** C6/C9/C10; F3/F4.

**Источник:** `06-llm/02-adaptation/Учебные конспекты/Е — RLHF, PPO, DPO, IPO, KTO и ORPO.md:1`

# Е. От предпочтений к обновлению policy

**Источник:** `06-llm/02-adaptation/Учебные конспекты/Е — RLHF, PPO, DPO, IPO, KTO и ORPO.md:3`

[Оглавление](../05-language-models/03-adaptation/README.md). Объединены пункты 5–6 и протокол 14D: RLHF/PPO и direct preference methods решают связанную задачу, но используют разные objectives и данные.

**Источник:** `06-llm/02-adaptation/Учебные конспекты/Е — RLHF, PPO, DPO, IPO, KTO и ORPO.md:5`

**Основа:** S3, строки 343–423 и 509–673. Сохранена последовательность reward model → policy optimization → DPO → альтернативы. **Исправления:** роли не равны четырём обязательным копиям модели; KTO не требует пар; reference point KTO не «среднее качество»; никакой метод не гарантирует отсутствие переобучения. **Углубление:** вывод DPO-параметризации, ручной loss и градиент margin.

**Источник:** `06-llm/02-adaptation/Учебные конспекты/Е — RLHF, PPO, DPO, IPO, KTO и ORPO.md:7`

**Уровень:** DPO — основа preference route; PPO — углубление. **Предпосылки:** C5, A5. **Используют:** C8/C9 and F6/F8.

**Источник:** `06-llm/02-adaptation/Учебные конспекты/Ж — Model soup, TIES, DARE, merge и switching.md:1`

# Ж. Объединение весов: что сохраняется, что может сломаться

**Источник:** `06-llm/02-adaptation/Учебные конспекты/Ж — Model soup, TIES, DARE, merge и switching.md:3`

[Оглавление](../05-language-models/03-adaptation/README.md). Пункт 8 и часть 14E.

**Источник:** `06-llm/02-adaptation/Учебные конспекты/Ж — Model soup, TIES, DARE, merge и switching.md:5`

**Основа:** S3, строки 719–816. Сохранены две задачи чата: выбор между несколькими хорошими checkpoints и обслуживание нескольких задач с одной базой. **Исправления C08:** soup не гарантирует улучшение; TIES не всегда majority vote; усреднение LoRA factors и full deltas различается; switching — отдельная операция. **Углубление:** контрпример factors, значений sign election и ожидание DARE.

**Источник:** `06-llm/02-adaptation/Учебные конспекты/Ж — Model soup, TIES, DARE, merge и switching.md:7`

**Уровень:** Углубление. **Предпосылки:** C2/C3/C5. **Используют:** C8 and multi-adapter serving.

**Источник:** `06-llm/02-adaptation/Учебные конспекты/З — Evaluation, perplexity, judge и экспорт.md:1`

# З. Как установить, что адаптация помогла

**Источник:** `06-llm/02-adaptation/Учебные конспекты/З — Evaluation, perplexity, judge и экспорт.md:3`

[Оглавление](../05-language-models/03-adaptation/README.md). Пункт 9 и протокол 14E; общая проверка 14A–D.

**Источник:** `06-llm/02-adaptation/Учебные конспекты/З — Evaluation, perplexity, judge и экспорт.md:5`

**Основа:** S3, строки 820–918. В исходнике evaluation завершает старый девятипунктный маршрут. Его вводные «мы обучили, сделали DPO, квантовали» заменены условной схемой: пользовательский запуск не подтверждён. **Исправления C09:** PPL не объявляется бесполезной, human preference не единственная ценность, contamination не выводится из высокой оценки. **Углубление:** paired comparison, ties, uncertainty и проверка экспортированного artifact.

**Источник:** `06-llm/02-adaptation/Учебные конспекты/З — Evaluation, perplexity, judge и экспорт.md:7`

**Уровень:** Общее ядро до deployment. **Предпосылки:** C1 and chosen C5/C6 method. **Используют:** C9/C10, D2–D10, E5.

**Источник:** `06-llm/02-adaptation/Учебные конспекты/И — AI feedback, несколько целей и continual learning.md:1`

# И. AI feedback, конфликтующие цели и последовательное обучение

**Источник:** `06-llm/02-adaptation/Учебные конспекты/И — AI feedback, несколько целей и continual learning.md:3`

[Оглавление](../05-language-models/03-adaptation/README.md). Объединены пункты 10, 11 и 13: источники feedback, постановка нескольких целей и сохранение прежнего поведения.

**Источник:** `06-llm/02-adaptation/Учебные конспекты/И — AI feedback, несколько целей и continual learning.md:5`

**Происхождение:** S4, строки 3914–4767, содержит вопросы о teacher, синтетике и возможных схемах обучения нового поколения моделей; S2, строки 4–72, обсуждает forgetting. Но systematic RLAIF/Constitutional AI, multi-objective и continual learning в чатах не пройдены. **Эта глава — явно обозначенное дополнение к плану**, связанное с теми вопросами. Не следует считать её восстановлением уже состоявшегося подробного занятия или описанием закрытых training recipes компаний.

**Источник:** `06-llm/02-adaptation/Учебные конспекты/И — AI feedback, несколько целей и continual learning.md:7`

**Уровень:** Углубление; основа if task requires it. **Предпосылки:** C6/C8. **Используют:** F6/F8 and continued adaptation.

**Источник:** `06-llm/02-adaptation/Учебные конспекты/К — Дистилляция, teacher annealing и обучение на своих траекториях.md:1`

# К. Дистилляция: чему student учится у teacher

**Источник:** `06-llm/02-adaptation/Учебные конспекты/К — Дистилляция, teacher annealing и обучение на своих траекториях.md:3`

[Оглавление](../05-language-models/03-adaptation/README.md). Пункт 12 плана; перенесённая тема старого inference A.5.

**Источник:** `06-llm/02-adaptation/Учебные конспекты/К — Дистилляция, teacher annealing и обучение на своих траекториях.md:5`

**Основа:** S4, строки 3445–4767. Сохранены вопросы: обязательно ли student меньше teacher; можно ли в начале обучения дать мягкий сигнал учителя и постепенно его ослабить; что можно сказать о training новых поколений моделей. **Исправления C13:** гипотезы о закрытых компаниях не факты, произвольные schedules/температуры не общие нормы, экономия total compute не следует из меньшего числа student steps. **Углубление:** KL, T², tokenizer compatibility, on-policy shift и честное сравнение стоимости.

**Источник:** `06-llm/02-adaptation/Учебные конспекты/К — Дистилляция, teacher annealing и обучение на своих траекториях.md:7`

**Уровень:** Углубление. **Предпосылки:** C5/C8, A5. **Используют:** Compact deployment; F3.

**Источник:** `06-llm/03-inference/Учебные конспекты/А — Квантизация, активации, PTQ, QAT, GPTQ и AWQ.md:1`

# А. Квантизация: что хранится, что вычисляется и где появляется ошибка

**Источник:** `06-llm/03-inference/Учебные конспекты/А — Квантизация, активации, PTQ, QAT, GPTQ и AWQ.md:3`

[Оглавление](../05-language-models/04-inference/README.md). Объединены inference A.1–A.3 и практический протокол 2.

**Источник:** `06-llm/03-inference/Учебные конспекты/А — Квантизация, активации, PTQ, QAT, GPTQ и AWQ.md:5`

**Основа:** S4, строки 4–2649; S3, 674–715. Сохранены вопросы про activations, обозначения GGUF в локальном интерфейсе, обучение сразу в INT4, Hessian/Jacobian и «online quantization». **Исправления C05/C10/C16:** алгоритм, representation, container и runtime разделены; нет правила «INT4 обязательно ломает модель», «1% весов делает 99% работы» или универсального dtype для JSON. **Углубление второй редакции:** путь activations через Transformer, ручные quantizers, metadata и память, calibration, LLM.int8()/SmoothQuant, QAT/STE, вывод одного шага GPTQ, численный AWQ и ограничения online requantization.

**Источник:** `06-llm/03-inference/Учебные конспекты/А — Квантизация, активации, PTQ, QAT, GPTQ и AWQ.md:7`

**Уровень:** Основа compression-ветки. **Предпосылки:** D1; numeric formats/linear algebra. **Используют:** C3/D4 and cost-limited serving.

**Источник:** `06-llm/03-inference/Учебные конспекты/Б — Pruning и граница с дистилляцией.md:1`

# Б. Pruning: удалить веса не всегда означает ускорить модель

**Источник:** `06-llm/03-inference/Учебные конспекты/Б — Pruning и граница с дистилляцией.md:3`

[Оглавление](../05-language-models/04-inference/README.md). Пункты A.4–A.5. Дистилляция подробно раскрыта в [адаптации](<../05-language-models/03-adaptation/adapt-distillation.md>).

**Источник:** `06-llm/03-inference/Учебные конспекты/Б — Pruning и граница с дистилляцией.md:5`

**Основа:** S4, строки 2653–3441 и 3445–4767. Сохранены вопросы «почему сначала создают избыточные головы, а потом удаляют?» и предположение о деградации закрытых сервисов. **Исправления C10/C13/C17:** удаление параметров не гарантирует runtime speedup; жалоба на ухудшение не доказывает pruning. **Углубление:** параметры, FLOPs, sparse storage и причинная проверка.

**Источник:** `06-llm/03-inference/Учебные конспекты/Б — Pruning и граница с дистилляцией.md:7`

**Уровень:** Углубление. **Предпосылки:** D3. **Используют:** Optimized serving; training → C10.

**Источник:** `06-llm/03-inference/Учебные конспекты/В — KV-cache, prefill, decode, MQA и GQA.md:1`

# В. KV-cache: что переиспользуется и как растёт память

**Источник:** `06-llm/03-inference/Учебные конспекты/В — KV-cache, prefill, decode, MQA и GQA.md:3`

[Оглавление](../05-language-models/04-inference/README.md). Пункт A.6; связи с C.4 и serving B.

**Источник:** `06-llm/03-inference/Учебные конспекты/В — KV-cache, prefill, decode, MQA и GQA.md:5`

**Основа:** S4, строки 4771–4922. **Исправления C10/C18:** cache не устраняет attention к истории, decode не всегда ограничен только памятью, GQA меняет число KV heads. **Углубление:** forward двух шагов, точный budget учебной модели и границы equivalence.

**Источник:** `06-llm/03-inference/Учебные конспекты/В — KV-cache, prefill, decode, MQA и GQA.md:7`

**Уровень:** Общее ядро ветки. **Предпосылки:** A1/A3/A5. **Используют:** D2–D10 and applications.

**Источник:** `06-llm/03-inference/Учебные конспекты/Г — Serving, continuous batching и PagedAttention.md:1`

# Г. Serving: как исполнять много запросов

**Источник:** `06-llm/03-inference/Учебные конспекты/Г — Serving, continuous batching и PagedAttention.md:3`

[Оглавление](../05-language-models/04-inference/README.md). Блок B.1–B.4 и практические протоколы 1/5.

**Источник:** `06-llm/03-inference/Учебные конспекты/Г — Serving, continuous batching и PagedAttention.md:5`

**Основа:** S4, строки 4926–6435. Сохранены static/continuous batching, paged allocation, serving engines и контраст с CV/encoder inference. **Исправления C19:** streaming короткого ответа возможен и без continuous batching; paging не меняет математическую модель; конкретные frameworks не получают вечного ranking. **Углубление:** occupancy example, измерения и состав runtime contract.

**Источник:** `06-llm/03-inference/Учебные конспекты/Г — Serving, continuous batching и PagedAttention.md:7`

**Уровень:** Основа serving-ветки. **Предпосылки:** D1. **Используют:** D9/D10 and production systems.

**Источник:** `06-llm/03-inference/Учебные конспекты/Д — Speculative decoding, rejection sampling и beam search.md:1`

# Д. Speculative decoding: угадать несколько токенов и проверить

**Источник:** `06-llm/03-inference/Учебные конспекты/Д — Speculative decoding, rejection sampling и beam search.md:3`

[Оглавление](../05-language-models/04-inference/README.md). C.1; углубления и case study из чата.

**Источник:** `06-llm/03-inference/Учебные конспекты/Д — Speculative decoding, rejection sampling и beam search.md:5`

**Основа:** S4, строки 6439–8230. Сохранены вопросы «станет ли большая модель быстрее от прогресса маленьких?», «что с beam search?» и последующее обсуждение DSpark. **Исправления C12/C13/C20:** сохраняется target distribution, а не обязательно тот же текст для seed; acceptance без residual недостаточен; stronger draft не автоматически быстрее. **Углубление:** числовое доказательство sampling correction и явная speed модель.

**Источник:** `06-llm/03-inference/Учебные конспекты/Д — Speculative decoding, rejection sampling и beam search.md:7`

**Уровень:** Углубление. **Предпосылки:** D1 and probability distributions. **Используют:** High-throughput decode/agent serving.

**Источник:** `06-llm/03-inference/Учебные конспекты/Е — Greedy, temperature, top-p, beam и constrained decoding.md:1`

# Е. Как выбирается следующий токен и ограничивается формат

**Источник:** `06-llm/03-inference/Учебные конспекты/Е — Greedy, temperature, top-p, beam и constrained decoding.md:3`

[Оглавление](../05-language-models/04-inference/README.md). Объединены C.2–C.3; семантика приложения раскрывается в его отдельном разделе.

**Источник:** `06-llm/03-inference/Учебные конспекты/Е — Greedy, temperature, top-p, beam и constrained decoding.md:5`

**Основа:** S5, строки 120–413; S4, 6439–7697 для beam context. Сохранён ваш вопрос: «greedy + high temperature — разве это не бессмыслица?» **Исправления C12/C21:** при T>0 простое деление logits сохраняет argmax в точной арифметике, даже при близких значениях. **Углубление:** ручной top-k/top-p, order of processors, состояние грамматики и validation layers.

**Источник:** `06-llm/03-inference/Учебные конспекты/Е — Greedy, temperature, top-p, beam и constrained decoding.md:7`

**Уровень:** Sampling — ядро; beam/grammar — углубление. **Предпосылки:** A5/D1. **Используют:** E1/E2 and applications.

**Источник:** `06-llm/03-inference/Учебные конспекты/Ж — Длинный контекст, sliding window и prompt caching.md:1`

# Ж. Длинный контекст и повторное использование prefix

**Источник:** `06-llm/03-inference/Учебные конспекты/Ж — Длинный контекст, sliding window и prompt caching.md:3`

[Оглавление](../05-language-models/04-inference/README.md). Пункт C.4; KV memory из A.6 и отличие от E.3.

**Источник:** `06-llm/03-inference/Учебные конспекты/Ж — Длинный контекст, sliding window и prompt caching.md:5`

**Основа:** S5, строки 417–1034. Сохранён вопрос о prompt/context caching «на примере стандартного проекта и нескольких запросов». **Исправления C14/C22:** возможности/TTL/тарифы провайдера не вечные факты, cached prefix не готовый answer, последующий query всё равно взаимодействует с prefix. **Углубление:** exact prefix examples, cache boundaries и оценка long-context quality.

**Источник:** `06-llm/03-inference/Учебные конспекты/Ж — Длинный контекст, sliding window и prompt caching.md:7`

**Уровень:** Основа long-context branch. **Предпосылки:** D1. **Используют:** RAG and production apps.

**Источник:** `06-llm/03-inference/Учебные конспекты/З — Orchestration, tools, RAG и границы runtime.md:1`

# З. От model forward к приложению: state, tools и retrieval

**Источник:** `06-llm/03-inference/Учебные конспекты/З — Orchestration, tools, RAG и границы runtime.md:3`

[Оглавление](../05-language-models/04-inference/README.md). Блок D.1–D.6; общая механика без привязки к одному framework API.

**Источник:** `06-llm/03-inference/Учебные конспекты/З — Orchestration, tools, RAG и границы runtime.md:5`

**Основа:** S5, строки 1038–3041; S6, 4–2197. В S6:179–181 вы прямо уточнили: сейчас изучается материал, а не создаётся агент. Поэтому ниже учебная система, без утверждений о вашем deployment. **Углубление:** явный state machine и границы ответственности.

**Источник:** `06-llm/03-inference/Учебные конспекты/З — Orchestration, tools, RAG и границы runtime.md:7`

**Уровень:** Основа application branch. **Предпосылки:** D1/D6; adaptation is not required. **Используют:** E1–E5; F1/F4.

**Источник:** `06-llm/03-inference/Учебные конспекты/И — Deployment, cold start, очереди и масштабирование.md:1`

# И. Эксплуатация inference-сервиса: запуск и рост нагрузки

**Источник:** `06-llm/03-inference/Учебные конспекты/И — Deployment, cold start, очереди и масштабирование.md:3`

[Оглавление](../05-language-models/04-inference/README.md). E.1–E.2 и практический протокол 5.

**Источник:** `06-llm/03-inference/Учебные конспекты/И — Deployment, cold start, очереди и масштабирование.md:5`

**Основа:** S6, строки 2201–3904. Сохранены container layers/model artifacts, cold/warm start, queue/latency и autoscaling. **Исправления C29:** image не обязан включать weights, одинаковые model replicas могут быть взаимозаменяемыми при external state, utilisation не бесполезна, но недостаточна одна. Цены и «всегда 2–5 минут» из чата не переносятся как нормы.

**Источник:** `06-llm/03-inference/Учебные конспекты/И — Deployment, cold start, очереди и масштабирование.md:7`

**Уровень:** Основа by deployment need. **Предпосылки:** D2 and workload metrics. **Используют:** D10 and rollout.

**Источник:** `06-llm/03-inference/Учебные конспекты/К — Кэш ответов, semantic cache, мониторинг и качество.md:1`

# К. Application cache и мониторинг качества

**Источник:** `06-llm/03-inference/Учебные конспекты/К — Кэш ответов, semantic cache, мониторинг и качество.md:3`

[Оглавление](../05-language-models/04-inference/README.md). Объединены E.3–E.4, практические протоколы 5–6.

**Источник:** `06-llm/03-inference/Учебные конспекты/К — Кэш ответов, semantic cache, мониторинг и качество.md:5`

**Происхождение:** prompt caching подробно обсуждался в S5:417–1034, deployment/scaling — S6:2201–3904. Отдельные E.3/E.4 являются следующими темами после конца исходной цепочки. **Эта глава — дополнение/продолжение по плану**, а не выдуманное прошлое занятие. Различие cache layers опирается на вопросы чата; keys/invalidation и monitoring углублены редакционно.

**Источник:** `06-llm/03-inference/Учебные конспекты/К — Кэш ответов, semantic cache, мониторинг и качество.md:7`

**Уровень:** Evaluation — ядро; advanced cache — углубление. **Предпосылки:** D1/D6 and service metrics. **Используют:** E5 and F3.

**Источник:** `06-llm/04-applications/Учебные конспекты/А — Structured output, schema и смысл результата.md:1`

# А. Structured output: формат, значения и проверяемый контракт

**Источник:** `06-llm/04-applications/Учебные конспекты/А — Structured output, schema и смысл результата.md:3`

[Оглавление](../05-language-models/05-applications/README.md). Пункт 1 плана.

**Источник:** `06-llm/04-applications/Учебные конспекты/А — Structured output, schema и смысл результата.md:5`

**Основа:** S5:120–242, S6:4–178. Сохранено сопоставление parser и constrained generation; исправлено смешение tool-tuned model с grammar constraint. **Углубление:** end-to-end data contract, ограничения схемы и четыре валидных, но неправильных результата.

**Источник:** `06-llm/04-applications/Учебные конспекты/А — Structured output, schema и смысл результата.md:7`

**Уровень:** Общее ядро приложения. **Предпосылки:** A1/A3/A5 and D6. **Используют:** E2/E3/E5.

**Источник:** `06-llm/04-applications/Учебные конспекты/Б — Tool calling, execution, ошибки и idempotency.md:1`

# Б. Tool calling: от предложенного вызова к реальному observation

**Источник:** `06-llm/04-applications/Учебные конспекты/Б — Tool calling, execution, ошибки и idempotency.md:3`

[Оглавление](../05-language-models/05-applications/README.md). Пункт 2 плана.

**Источник:** `06-llm/04-applications/Учебные конспекты/Б — Tool calling, execution, ошибки и idempotency.md:5`

**Основа:** S5:2826–3041 и продолжение S6:4–178. В S6:179–181 вы уточнили теоретический режим — здесь примеры учебные. **Исправления:** model proposal не actual execution; observation не произвольная system instruction; native format, schema enforcement и capability разные. **Углубление:** lifecycle call ID, retry ambiguity и idempotency.

**Источник:** `06-llm/04-applications/Учебные конспекты/Б — Tool calling, execution, ошибки и idempotency.md:7`

**Уровень:** Основа tool-use branch. **Предпосылки:** E1. **Используют:** E3/E5, F1/F4.

**Источник:** `06-llm/04-applications/Учебные конспекты/В — State, orchestration, графы и участники агента.md:1`

# В. Orchestration: состояние, переходы и воспроизводимый loop

**Источник:** `06-llm/04-applications/Учебные конспекты/В — State, orchestration, графы и участники агента.md:3`

[Оглавление](../05-language-models/05-applications/README.md). Пункт 3 плана.

**Источник:** `06-llm/04-applications/Учебные конспекты/В — State, orchestration, графы и участники агента.md:5`

**Основа:** S5:1038–2822, S6:1377–2197. Сохранены вопросы о локальных Ollama/LM Studio models и о том, реализован ли каждый AutoGen participant через LangGraph. **Исправления:** одно framework name не определяет internals; loop может быть graph, messages или обычным code; orchestration не равно agent training. **Углубление:** reducers/checkpoints и recovery side effects.

**Источник:** `06-llm/04-applications/Учебные конспекты/В — State, orchestration, графы и участники агента.md:7`

**Уровень:** Основа application workflow. **Предпосылки:** E2. **Используют:** E4/E5 and F1/F4.

**Источник:** `06-llm/04-applications/Учебные конспекты/Г — RAG, BM25, dense, hybrid и multi-vector retrieval.md:1`

# Г. RAG: от вопроса и корпуса к проверяемому evidence

**Источник:** `06-llm/04-applications/Учебные конспекты/Г — RAG, BM25, dense, hybrid и multi-vector retrieval.md:3`

[Оглавление](../05-language-models/05-applications/README.md). Пункт 4 плана.

**Источник:** `06-llm/04-applications/Учебные конспекты/Г — RAG, BM25, dense, hybrid и multi-vector retrieval.md:5`

**Основа:** S6:182–1373; дополнительно S5:1038–1817 о pipeline. Сохранены ваши вопросы: «почему LLM поймёт наш embedding», «откуда BM25 берёт keywords» и «насколько вырастет база при нескольких vectors». **Исправления:** ordinary RAG передаёт generator текст, BM25 не требует отдельного LLM keyword extractor, memory/latency не растут по одному универсальному проценту. **Углубление:** полный ручной BM25, ANN vs relevance и storage accounting.

**Источник:** `06-llm/04-applications/Учебные конспекты/Г — RAG, BM25, dense, hybrid и multi-vector retrieval.md:7`

**Уровень:** Основа knowledge-application branch. **Предпосылки:** E1; local retrieval/embedding definition. **Используют:** E5; no encoder-training course required.

**Источник:** `06-llm/04-applications/Учебные конспекты/Д — Evaluation retrieval, ответов, citations и task success.md:1`

# Д. Оценка приложения: где именно пропадает правильный ответ

**Источник:** `06-llm/04-applications/Учебные конспекты/Д — Evaluation retrieval, ответов, citations и task success.md:3`

[Оглавление](../05-language-models/05-applications/README.md). Пункт 5 и итоговый checkpoint плана.

**Источник:** `06-llm/04-applications/Учебные конспекты/Д — Evaluation retrieval, ответов, citations и task success.md:5`

**Основа:** S6:182–1373 о RAG, S5/S6 о tools и S3:820–918 об evaluation. **Углубление:** ручные ranking metrics, oracle context, различие correctness/groundedness/citation и стоимость успешного решения. Никакие числа ниже не объявляются результатом вашего запуска.

**Источник:** `06-llm/04-applications/Учебные конспекты/Д — Evaluation retrieval, ответов, citations и task success.md:7`

**Уровень:** Общее ядро для применения. **Предпосылки:** E1–E4. **Используют:** C8, D10, F1/F3/F10.

**Источник:** `06-llm/05-lifecycle/Учебные конспекты/А — Lifecycle и выбор следующего вмешательства.md:1`

# А. Lifecycle: как связать данные, модель, приложение и обучение

**Источник:** `06-llm/05-lifecycle/Учебные конспекты/А — Lifecycle и выбор следующего вмешательства.md:3`

[Оглавление](../05-language-models/06-agent-learning/README.md). Пункт 1 плана.

**Источник:** `06-llm/05-lifecycle/Учебные конспекты/А — Lifecycle и выбор следующего вмешательства.md:5`

**Связь с чатами:** S3:228–918 обсуждает SFT/preferences/evaluation; S5/S6 — runtime, RAG и deployment. Эта глава связывает уже обсуждённые механизмы и дополняет их диагностикой agentic tasks. Самостоятельное обучение агента в источниках не подтверждено.

**Источник:** `06-llm/05-lifecycle/Учебные конспекты/А — Lifecycle и выбор следующего вмешательства.md:7`

**Уровень:** Общее ядро. **Предпосылки:** A1/A3/A5, E5. **Используют:** F2–F10.

**Источник:** `06-llm/05-lifecycle/Учебные конспекты/Б — Среда, observations, actions, reward и horizon.md:1`

# Б. Формальная постановка agentic-задачи

**Источник:** `06-llm/05-lifecycle/Учебные конспекты/Б — Среда, observations, actions, reward и horizon.md:3`

[Оглавление](../05-language-models/06-agent-learning/README.md). Пункт 2 плана.

**Источник:** `06-llm/05-lifecycle/Учебные конспекты/Б — Среда, observations, actions, reward и horizon.md:5`

**Основа и дополнение:** S6:4–178 и 1377–2197 дают tool loop и coordination. Формальная MDP/POMDP постановка, credit assignment и toy environment ниже — новое углубление по плану, а не восстановленное занятие о training.

**Источник:** `06-llm/05-lifecycle/Учебные конспекты/Б — Среда, observations, actions, reward и horizon.md:7`

**Уровень:** Общее ядро agent-training ветки. **Предпосылки:** F1, E2/E3. **Используют:** F3–F10.

**Источник:** `06-llm/05-lifecycle/Учебные конспекты/В — Behavior cloning и SFT на tool trajectories.md:1`

# В. SFT на действиях: как учить выбор tool и реакцию на observation

**Источник:** `06-llm/05-lifecycle/Учебные конспекты/В — Behavior cloning и SFT на tool trajectories.md:3`

[Оглавление](../05-language-models/06-agent-learning/README.md). Пункт 3 плана.

**Источник:** `06-llm/05-lifecycle/Учебные конспекты/В — Behavior cloning и SFT на tool trajectories.md:5`

**Связь с источниками:** S3:228–339 объясняет обычный SFT, S6:4–178 — tool execution. Их соединение в trajectory behavior cloning — дополнительный учебный материал. Источники не подтверждают ваш training run агента.

**Источник:** `06-llm/05-lifecycle/Учебные конспекты/В — Behavior cloning и SFT на tool trajectories.md:7`

**Уровень:** Основа demonstration learning. **Предпосылки:** F2/F10; C5 необязателен. **Используют:** F4–F9.

**Источник:** `06-llm/05-lifecycle/Учебные конспекты/Г — Distribution shift, recovery и DAgger.md:1`

# Г. Почему имитация ломается после первой ошибки: distribution shift и DAgger

**Источник:** `06-llm/05-lifecycle/Учебные конспекты/Г — Distribution shift, recovery и DAgger.md:3`

[Оглавление](../05-language-models/06-agent-learning/README.md). Пункт 4 плана. **Происхождение:** новое углубление по плану; в чатах S5–S6 обсуждалось исполнение инструментов, но итеративное обучение на посещённых состояниях систематически не разбиралось. Базовый метод: [DAgger](https://arxiv.org/abs/1011.0686).

**Источник:** `06-llm/05-lifecycle/Учебные конспекты/Г — Distribution shift, recovery и DAgger.md:5`

**Уровень:** Углубление recovery branch. **Предпосылки:** F2/F3, expert or validated feedback. **Используют:** F6/F7.

**Источник:** `06-llm/05-lifecycle/Учебные конспекты/Д — Reasoning, верификация и test-time compute.md:1`

# Д. Рассуждение, проверка и бюджет попыток

**Источник:** `06-llm/05-lifecycle/Учебные конспекты/Д — Reasoning, верификация и test-time compute.md:3`

[Оглавление](../05-language-models/06-agent-learning/README.md). Пункт 5 плана. **Происхождение:** новое углубление. Чаты обсуждали generation и orchestration, но не содержали полного курса обучения reasoning. Здесь разделены обновление весов и использование дополнительных вычислений готовой моделью.

**Источник:** `06-llm/05-lifecycle/Учебные конспекты/Д — Reasoning, верификация и test-time compute.md:5`

**Уровень:** Углубление. **Предпосылки:** F1/F3/F10 and a verifier/outcome metric. **Используют:** F6/F10.

**Источник:** `06-llm/05-lifecycle/Учебные конспекты/Е — RL, credit assignment, PPO и GRPO.md:1`

# Е. Обучение через взаимодействие: от награды к изменению политики

**Источник:** `06-llm/05-lifecycle/Учебные конспекты/Е — RL, credit assignment, PPO и GRPO.md:3`

[Оглавление](../05-language-models/06-agent-learning/README.md). Пункт 6 плана. **Основа:** S3:343–423 вводит RLHF/PPO; общая адаптация раскрыта в соседнем разделе. Agentic rollouts, delayed reward и GRPO — новое углубление по плану, а не уже проведённое вами обучение. Первичные источники: [PPO](https://arxiv.org/abs/1707.06347), [DeepSeekMath, GRPO](https://arxiv.org/abs/2402.03300).

**Источник:** `06-llm/05-lifecycle/Учебные конспекты/Е — RL, credit assignment, PPO и GRPO.md:5`

**Уровень:** Углубление; основа при проверяемой среде. **Предпосылки:** F1–F3/F10, reward definition and budget. **Используют:** F7–F10.

**Источник:** `06-llm/05-lifecycle/Учебные конспекты/Ж — Память, summary и длинный горизонт.md:1`

# Ж. Как сохранить нужное состояние длинной задачи

**Источник:** `06-llm/05-lifecycle/Учебные конспекты/Ж — Память, summary и длинный горизонт.md:3`

[Оглавление](../05-language-models/06-agent-learning/README.md). Пункт 7 плана. **Связь с чатами:** S5:417–1034 объясняет длинный контекст и prefix cache; S5:1821–2822 — state/checkpoint в orchestration. Здесь это углублено до проектирования памяти policy. Обучение такой памяти в чатах не подтверждено.

**Источник:** `06-llm/05-lifecycle/Учебные конспекты/Ж — Память, summary и длинный горизонт.md:5`

**Уровень:** Основа memory branch by task. **Предпосылки:** F2/F3/F10. **Используют:** F8/F9.

**Источник:** `06-llm/05-lifecycle/Учебные конспекты/З — Предпочтения между полными траекториями.md:1`

# З. Как сравнивать сессии, а не только красивые ответы

**Источник:** `06-llm/05-lifecycle/Учебные конспекты/З — Предпочтения между полными траекториями.md:3`

[Оглавление](../05-language-models/06-agent-learning/README.md). Пункт 8 плана. **Основа:** S3:509–673 — DPO и родственные методы; критерии AI feedback раскрыты в адаптации. Перенос единицы сравнения на полную agentic-сессию — дополнительный материал.

**Источник:** `06-llm/05-lifecycle/Учебные конспекты/З — Предпочтения между полными траекториями.md:5`

**Уровень:** Углубление. **Предпосылки:** F3/F10, preference rubric and judged data. **Используют:** F6/F9.

**Источник:** `06-llm/05-lifecycle/Учебные конспекты/И — Multi-agent, роли и обучение координации.md:1`

# И. Несколько агентов: когда координация приносит пользу

**Источник:** `06-llm/05-lifecycle/Учебные конспекты/И — Multi-agent, роли и обучение координации.md:3`

[Оглавление](../05-language-models/06-agent-learning/README.md). Пункт 9 плана. **Основа:** S6:1377–2197 — frameworks, роли и вопрос о внутреннем устройстве участников; это материал об orchestration. Role-specific adaptation, self-play и одновременное обучение политик ниже — новое обзорное углубление, не выполненные эксперименты.

**Источник:** `06-llm/05-lifecycle/Учебные конспекты/И — Multi-agent, роли и обучение координации.md:5`

**Уровень:** Обзор; углубление by need. **Предпосылки:** F1/F2/F10 and equal total-budget protocol. **Используют:** Terminal branch; its result uses F10's evaluation protocol.

**Источник:** `06-llm/05-lifecycle/Учебные конспекты/К — Оценка траекторий, стоимость и воспроизводимость.md:1`

# К. Как доказать, что агент стал лучше

**Источник:** `06-llm/05-lifecycle/Учебные конспекты/К — Оценка траекторий, стоимость и воспроизводимость.md:3`

[Оглавление](../05-language-models/06-agent-learning/README.md). Пункт 10 плана.

**Источник:** `06-llm/05-lifecycle/Учебные конспекты/К — Оценка траекторий, стоимость и воспроизводимость.md:5`

**Происхождение:** S3:820–918 обсуждает оценку LLM, S6:3013–3904 — стоимость и масштабирование serving. Полный протокол оценки траекторий — дополнительный материал по плану. Числа ниже заданы для учебных примеров; настоящие training runs и production benchmark не проводились.

**Источник:** `06-llm/05-lifecycle/Учебные конспекты/К — Оценка траекторий, стоимость и воспроизводимость.md:11`

<!-- source-depth-lifecycle-К -->

**Источник:** `06-llm/05-lifecycle/Учебные конспекты/К — Оценка траекторий, стоимость и воспроизводимость.md:13`

**Уровень:** Общее ядро до обучения. **Предпосылки:** F1/F2; task-success/cost definition. **Используют:** F3–F9; C8/D10/E5.

**Источник:** `07-bert/Учебные конспекты/А — Encoder, tokenizer, attention и pooling.md:1`

# А. От текста до контекстных состояний BERT

**Источник:** `07-bert/Учебные конспекты/А — Encoder, tokenizer, attention и pooling.md:3`

[Оглавление](../05-language-models/02-bert-encoders/README.md). Модули **1–5** плана. **Происхождение:** BERT — новая ветка по плану. Предпосылка из чатов S2:1817–3404 — токены/Transformer; подробный QKV/backward уже есть в [основах LLM](../05-language-models/01-foundations/README.md). Здесь объясняются именно bidirectional encoding и interfaces задач.

**Источник:** `07-bert/Учебные конспекты/А — Encoder, tokenizer, attention и pooling.md:5`

**Уровень:** Общее ядро ветки. **Предпосылки:** A1, A3–A5; matrices and tensor shapes. **Используют:** B2–B9.

**Источник:** `07-bert/Учебные конспекты/Б — MLM, objectives и эволюция encoder-моделей.md:1`

# Б. Чему учится encoder до появления прикладной головы

**Источник:** `07-bert/Учебные конспекты/Б — MLM, objectives и эволюция encoder-моделей.md:3`

[Оглавление](../05-language-models/02-bert-encoders/README.md). Модули **6–9**. **Происхождение:** полный BERT-pretraining — дополнение по плану. Чаты про next-token LLM дают полезный контраст, но не заменяют MLM. Каталог ниже — объяснение mechanisms и критериев, а не рейтинг checkpoint для вашей задачи.

**Источник:** `07-bert/Учебные конспекты/Б — MLM, objectives и эволюция encoder-моделей.md:5`

**Уровень:** Общее ядро для pretraining route. **Предпосылки:** B1. **Используют:** B3/B4/B8; BERT model choice.

**Источник:** `07-bert/Учебные конспекты/В — Данные, головы, fine-tuning и DAPT.md:1`

# В. Как обучить encoder под задачу и понять причину ошибки

**Источник:** `07-bert/Учебные конспекты/В — Данные, головы, fine-tuning и DAPT.md:3`

[Оглавление](../05-language-models/02-bert-encoders/README.md). Модули **10–14**. **Основа:** S3 объясняет SFT/CPT, S1–S3 — PEFT; применение к encoder и task heads — дополнение. Общие методики подробно находятся в [данных](../01-data-features/README.md) и [адаптации](../05-language-models/03-adaptation/README.md).

**Источник:** `07-bert/Учебные конспекты/В — Данные, головы, fine-tuning и DAPT.md:5`

**Уровень:** Общее ядро для supervised route. **Предпосылки:** B1–B2; D2 for split. **Используют:** B4/B8/B9.

**Источник:** `07-bert/Учебные конспекты/Г — Classification, NER, spans, пары и extractive QA.md:1`

# Г. Прикладные головы encoder: семь разных постановок

**Источник:** `07-bert/Учебные конспекты/Г — Classification, NER, spans, пары и extractive QA.md:3`

[Оглавление](../05-language-models/02-bert-encoders/README.md). Модули **15–21**. **Происхождение:** новое раскрытие BERT-плана; общие structured outputs из чатов не означают обучение этих heads. Один backbone может обслуживать разные задачи, но labels, objective и decoding должны совпадать с постановкой.

**Источник:** `07-bert/Учебные конспекты/Г — Classification, NER, spans, пары и extractive QA.md:9`

**Уровень:** Основа branch by task. **Предпосылки:** B1–B3. **Используют:** B8/B9; selected app integration.

**Источник:** `07-bert/Учебные конспекты/Д — Sentence embeddings, contrastive loss и геометрия.md:1`

# Д. Как научить vectors выражать нужную похожесть

**Источник:** `07-bert/Учебные конспекты/Д — Sentence embeddings, contrastive loss и геометрия.md:3`

[Оглавление](../05-language-models/02-bert-encoders/README.md). Группа E, модули **22–25**. **Основа:** S6:182–1373 задаёт вопросы internal/retrieval embeddings; sentence representation learning здесь расширено по BERT-плану.

**Источник:** `07-bert/Учебные конспекты/Д — Sentence embeddings, contrastive loss и геометрия.md:5`

**Уровень:** Основа representation branch. **Предпосылки:** B1–B3; probability/geometry. **Используют:** B6/B7/B8/B9; E4 local retrieval.

**Источник:** `07-bert/Учебные конспекты/Е — Dense, cross-encoder, SPLADE и ColBERT retrieval.md:1`

# Е. Четыре способа оценить релевантность query–document

**Источник:** `07-bert/Учебные конспекты/Е — Dense, cross-encoder, SPLADE и ColBERT retrieval.md:3`

[Оглавление](../05-language-models/02-bert-encoders/README.md). Группа F, модули **26–29**. **Основа:** S6:182–1373 подробно обсуждает RAG/BM25/multi-vector storage; здесь добавлены encoder training, sparse expansion и точный MaxSim. Общий RAG pipeline находится в [приложениях LLM](../05-language-models/05-applications/README.md).

**Источник:** `07-bert/Учебные конспекты/Е — Dense, cross-encoder, SPLADE и ColBERT retrieval.md:5`

**Уровень:** Основа retrieval branch. **Предпосылки:** B1/B5; D2 data contract. **Используют:** B8/B9; E4 can use trained/frozen retriever.

**Источник:** `07-bert/Учебные конспекты/Ж — Языки, длинные документы, домены и layout.md:1`

# Ж. Когда task меняется вместе с языком, длиной и модальностью

**Источник:** `07-bert/Учебные конспекты/Ж — Языки, длинные документы, домены и layout.md:3`

[Оглавление](../05-language-models/02-bert-encoders/README.md). Группа G, модули **30–34**. **Происхождение:** углубление нового BERT-плана; общие long-context вопросы из S5 здесь применены к encoder tasks, остальные специализированные ветки — дополнение.

**Источник:** `07-bert/Учебные конспекты/Ж — Языки, длинные документы, домены и layout.md:5`

**Уровень:** Основа only for selected data domain. **Предпосылки:** B1/B3; B5/B6 only for their retrieval cases. **Используют:** B8/B9 and domain applications.

**Источник:** `07-bert/Учебные конспекты/З — Метрики, калибровка, OOD и интерпретация.md:1`

# З. Как оценить encoder без переоценки его возможностей

**Источник:** `07-bert/Учебные конспекты/З — Метрики, калибровка, OOD и интерпретация.md:3`

[Оглавление](../05-language-models/02-bert-encoders/README.md). Группа H, модули **35–37**. **Основа:** evaluation из S3 и retrieval из S6; task-specific statistics и interpretation — расширение по плану. Полные общие разделы находятся в [данных](../../01-data/Учебные%20 конспекты/README.md).

**Источник:** `07-bert/Учебные конспекты/З — Метрики, калибровка, OOD и интерпретация.md:5`

**Уровень:** Общее ядро до утверждений о качестве. **Предпосылки:** B3 and chosen B4/B5/B6 task. **Используют:** B9; E5 application evaluation.

**Источник:** `07-bert/Учебные конспекты/И — Compression, encoder service и жизненный цикл индекса.md:1`

# И. Как сохранить качество после сжатия, экспорта и обновления

**Источник:** `07-bert/Учебные конспекты/И — Compression, encoder service и жизненный цикл индекса.md:3`

[Оглавление](../05-language-models/02-bert-encoders/README.md). Группа I, модули **38–40**. **Основа:** S4 обсуждает compression/serving, S6 — deployment/scaling. Здесь раскрыты encoder-specific tasks, fixed-vector index и monitoring; общая механика не дублируется вместо объяснения отличий.

**Источник:** `07-bert/Учебные конспекты/И — Compression, encoder service и жизненный цикл индекса.md:5`

**Уровень:** Углубление; service contract is core for deployment. **Предпосылки:** B3/B8 and selected B4/B5/B6. **Используют:** E4/E5 and production systems.
