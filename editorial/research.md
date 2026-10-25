# Источники и границы современных дополнений

## Пакет 02 — 25 октября 2026

[Отдельный реестр первичных проверок](<batch02-research.md>) охватывает 11 глав адаптации и девять глав инференса. Расчёты и авторские схемы проверены отдельно от опубликованных результатов моделей; запуск обучения и GPU-benchmark не заявлены.

## Повторная сверка при редактуре v3 — 21 октября 2026

- BERT: [статья](https://aclanthology.org/N19-1423/) и [официальный modeling.py](https://github.com/google-research/bert/blob/master/modeling.py). Сверены входная сумма, Post-LN, GELU/FFN, dropout, pooler, MLM и формы голов. Рисунок 1 включён с атрибуцией по [политике ACL](https://aclanthology.org/faq/copyright/); параметры извлечения и лицензия находятся в `illustrations.json`.
- Transformer: [исходная статья](https://arxiv.org/abs/1706.03762), [RMSNorm](https://arxiv.org/abs/1910.07467), [положение LayerNorm](https://arxiv.org/abs/2002.04745), [RoFormer](https://arxiv.org/abs/2104.09864). Использованы для механизма и ограничений, без переноса измеренных преимуществ на произвольную модель.
- [FlashAttention](https://arxiv.org/abs/2205.14135): различие полной математической операции и хранения промежуточных матриц. Ускорение конкретной конфигурации не измерялось.
- [K-Means++](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf), [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html), [HDBSCAN API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html), [описание HDBSCAN авторами реализации](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html). API-страницы показывали scikit-learn 1.9.1. Сверены квадратичная память конкретной реализации DBSCAN, учёт самой точки, разные роли двух параметров HDBSCAN; библиотеки в вычислительных примерах не запускались.
- [Связывание входной и выходной таблиц](https://arxiv.org/abs/1608.05859) и [AdamW](https://arxiv.org/abs/1711.05101): механизм, пути градиента и границы экономии параметров. Источники не используются как гарантия качества текущего учебного примера.
- [Tokenizers Encoding](https://huggingface.co/docs/tokenizers/api/encoding): текущая страница указывает неполные rc0 bindings и отсутствие части полей. В `bert-interface` явно требуется проверять доступность смещений и word IDs в выбранной версии, без обещания одинакового API во всех реализациях.
- [DSpark](https://arxiv.org/abs/2607.05147): проверено наличие первичной работы и границы авторских результатов. Заявленные ускорения относятся к описанным авторами конфигурациям, не являются независимым измерением курса.

### Цели предобучения и варианты кодировщиков

Повторная сверка 21 октября 2026 для `bert-objectives`: вопрос — какие входы, цели, параметры и пути градиента меняются; производительность не измерялась.

- [BERT: подготовка данных](https://github.com/google-research/bert/blob/master/create_pretraining_data.py) и [расчёт предобучения](https://github.com/google-research/bert/blob/master/run_pretraining.py), архивный официальный репозиторий: исходные ID, выбранные позиции, правила замен, сбор состояний, связанная таблица и нормировка голов. Используется для исторической реализации, не обещает одинаковый API современных библиотек.
- Lan et al., [ALBERT v6](https://arxiv.org/html/1909.11942v6), 9 февраля 2020: сверены факторизация таблицы, общие параметры блоков и SOP. Liu et al., [RoBERTa](https://arxiv.org/abs/1907.11692), 2019: изменения процедуры обучения и ограничения вывода об исключении NSP. Решение: сравнительная ветка с выходом к адаптации, без гарантии качества метода на произвольной задаче.
- Joshi et al., [SpanBERT v3](https://arxiv.org/html/1907.10529v3), 18 января 2020: внешние состояния границ и относительная позиция предсказывают токен фрагмента отдельно от обычной MLM-головы. Clark et al., [ELECTRA](https://arxiv.org/html/2003.10555v1), 23 марта 2020: генератор обучается по MLM, различитель — по фактическому совпадению ID; правильный предложенный токен остаётся настоящим. Решение: отдельные авторские схемы целей, без смешения с прикладной проверкой ответов.
- Sanh et al., [DistilBERT v4](https://arxiv.org/abs/1910.01108v4), 1 марта 2020: три обучающих сигнала и меньший ученик, не простое удаление слоёв. He et al., [DeBERTa-v3 v4](https://arxiv.org/html/2111.09543v4), 24 марта 2023: RTD и GDES, формула `E_D=sg(E_G)+E_Δ`, остановка RTD-градиента к таблице генератора. Conneau et al., [XLM-R v2](https://arxiv.org/abs/1911.02116v2), 8 апреля 2020: общая многоязычная модель и конкуренция за ёмкость. Решение: сравнение изменяемого механизма; данные из аннотаций о превосходстве не используются как критерий приёмки курса.
- Warner et al., [ModernBERT v2](https://arxiv.org/html/2412.13663v2), 19 декабря 2024: проверены Pre-LN, GeGLU, RoPE, локальное/глобальное внимание и исходная длина 8192. Le Breton et al., [NeoBERT v2](https://arxiv.org/html/2502.19587v2), 6 июня 2025: Pre-RMSNorm, SwiGLU, RoPE, около 250 млн параметров и длина 4096. Решение: необязательные примеры выбора архитектуры, без переноса авторских benchmark-результатов на все языки и задачи.
- Marone et al., [mmBERT](https://proceedings.mlr.press/v306/marone26a.html), ICML/PMLR 306, 2026: проверено включение дополнительных малоресурсных языков в фазе снижения скорости обучения. Weller et al., [Ettin v2](https://arxiv.org/abs/2507.11412v2), 12 марта 2026: отдельные парные кодировщики и декодеры, согласованные данные и бюджет. Решение: примеры расписания и контролируемого сравнения, не соединённая encoder–decoder модель.

Рисунки — собственные русскоязычные SVG. Обозначения и числовые примеры MLM/факторизации проверяются отдельно от запуска предобучения. Полный разбор других задач этих моделей не требуется для перехода к адаптации.

### Кластеризация: геометрия, совместное обучение и приложения

Повторно проверено 21 октября 2026. Цель — объяснить механизм и ограничения конкретных учебных конвейеров; новых сравнительных экспериментов с моделями не проводилось.

- Van der Maaten, Hinton, [t-SNE](https://jmlr.org/papers/v9/vandermaaten08a.html), JMLR 9, 2008; прочитан полный PDF для условных гауссовских сходств, симметризации и перплексии. [Документация UMAP по кластеризации](https://umap-learn.readthedocs.io/en/latest/clustering.html) и [FAQ](https://umap-learn.readthedocs.io/en/latest/faq.html), страница версии 0.5.8: ограничения сохранения плотности, разрывы и контроль случайности. Решение: основа, без обещания сохранения истинных групп.
- Tibshirani, Walther, Hastie, [Gap statistic](https://statistics.stanford.edu/technical-reports/estimating-number-clusters-dataset-gap-statistic), 2001: доступна карточка первичной работы; полный авторский PDF при проверке вернул 403. Формула и различия правил дополнительно сверены с [официальным R `cluster::clusGap`](https://stat.ethz.ch/R-manual/R-devel/library/cluster/html/clusGap.html). Степени расстояния, `firstSEmax` и `Tibs2001SEmax` различаются; учебное соглашение `sd` задано явно. Решение: основа с оговорками реализации, без заявления о прочтении недоступного PDF.
- Xie, Girshick, Farhadi, [DEC](https://proceedings.mlr.press/v48/xieb16.html), ICML/PMLR 48, 2016; Guo et al., [IDEC](https://www.ijcai.org/Proceedings/2017/243), IJCAI 2017: прочитаны PDF для предварительного автокодировщика, инициализации K-Means, назначения Стьюдента, фиксированной вспомогательной цели и возвращения восстановления в IDEC. Решение: углубление. Улучшение внутренней ошибки не переносится на смысловое качество без внешнего контроля.
- Achanta et al., [SLIC](https://www.epfl.ch/labs/ivrl/research/slic-superpixels/), TPAMI 2012 и авторская страница: пятикоординатное пространство цвета и положения, компактность и отдельный SLICO. Chen et al., [SimCLR](https://proceedings.mlr.press/v119/chen20j.html), ICML 2020; He et al., [MoCo v3 текста статьи](https://arxiv.org/abs/1911.05722v3), CVPR 2020; Caron et al., [DINO v2 текста статьи](https://arxiv.org/abs/2104.14294v2), ICCV 2021. Для последних двух CVF вернул 403, использованы первичные arXiv-страницы. Различены контрастивные пары, очередь с усредняемым кодировщиком и самодистилляция. Решение: именованные примеры, без пересказа benchmark-преимуществ как гарантии кластеризации.
- Blei, Ng, Jordan, [LDA](https://www.jmlr.org/papers/v3/blei03a.html), JMLR 3, 2003; Lee, Seung, [алгоритмы NMF](https://proceedings.neurips.cc/paper/2000/hash/f9d1152547c0bde01830b7e8bd60024c-Abstract.html), NeurIPS 2000; Angelov, [Top2Vec](https://arxiv.org/abs/2008.09470), авторский препринт 2020. Использованы для различения смеси вероятностей, неотрицательного произведения и группировки совместных представлений; заявления превосходства из аннотаций не перенесены в курс. Решение: основа текстовой ветки.
- [FAISS: индексы](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes), официальная wiki, отображаемая редакция 28 июля 2025. Проверены списки IVF, число просмотренных ячеек, приближение поиска и отдельный вклад сжатия. Решение: системная ветка; оценка числа кандидатов не выдаётся за измеренную задержку.

Новые рисунки этих глав — авторские SVG; источники поддерживают механизм, не являются лицензией на копирование чужого рисунка. Числа в схемах — учебные вычисления, проверяемые отдельно.

## Пакет 01: токены, состояния, позиции и прикладная ветка BERT

Повторная проверка первичных источников: **21 октября 2026**. Названия версий ниже относятся к открытым страницам, не к установленным пакетам. Обучение моделей и измерение их производительности не выполнялись.

- [SentencePiece](https://arxiv.org/abs/1808.06226), 2018, и [токенизация исходного BERT](https://github.com/google-research/bert/blob/master/tokenization.py): инструментарий отделён от алгоритма сегментации; нормализация и словарь входят в контракт. [RoFormer](https://arxiv.org/abs/2104.09864), открытая v5 от 8 ноября 2023: поворот Q/K и относительный угол; не обещание неограниченной длины.
- [DAPT/TAPT](https://aclanthology.org/2020.acl-main.740/), ACL 2020: предметный корпус и распределение задачи различены. [LoRA](https://arxiv.org/abs/2106.09685), v2 от 16 октября 2021, оставлена необязательным углублением. [Autograd PyTorch 2.14](https://docs.pytorch.org/docs/2.14/notes/autograd.html) поддерживает различение замороженных параметров и производной по входу; это не тест установленного PyTorch.
- [CRF в официальном учебнике PyTorch](https://docs.pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html), отображённая документация 2.14.0+cu130: logsumexp при обучении и max при Витерби. Руководства [token classification](https://huggingface.co/docs/transformers/main/tasks/token_classification) и [QA](https://huggingface.co/docs/transformers/main/tasks/question_answering) открывались в ветке main; на страницах показывались стабильные v5.20.2 и v5.20.1. Поля offsets и word IDs зависят от интерфейса; страницы не заменяют фиксацию установленной версии. [SetFit](https://arxiv.org/abs/2209.11055), 2022: пары и последующая голова, без превращения пар в независимые источники.
- [Sentence-BERT](https://aclanthology.org/D19-1410/), 2019; [SimCSE](https://arxiv.org/abs/2104.08821), v4 от 18 мая 2022; [Matryoshka](https://arxiv.org/abs/2205.13147), v4 от 8 февраля 2024: независимое кодирование, случайные виды и обучение префиксов объяснены раздельно. [Sentence Transformers losses](https://sbert.net/docs/package_reference/sentence_transformer/losses.html): общий пул сравнений не отождествлён с накоплением градиента.
- [Исходный RAG, генератор](https://arxiv.org/html/2005.11401v4#S2.SS3), открытая v4: DPR выбирает фрагменты, собственный encoder BART обрабатывает их текст. Это конкретный интерфейс, а не запрет на любые обученные интерфейсы векторов.
- [Lucene BM25 10.3.1](https://lucene.apache.org/core/10_3_1/core/org/apache/lucene/search/similarities/BM25Similarity.html): IDF, насыщение и параметры; независимый расчёт исправляет раннее округление 11.73 на 11.710677. [SPLADE v2](https://arxiv.org/abs/2109.10086) и [код агрегации авторов](https://github.com/naver/splade/blob/main/splade/models/transformer_rep.py) различают sum и max. [ColBERTv2](https://arxiv.org/abs/2112.01488), v3 от 10 июля 2022: позднее взаимодействие не равно среднему вектору. [BGE-M3](https://arxiv.org/html/2402.03216v5), v5 от 12 декабря 2025: три способа представления без универсальной гарантии качества. [Карточка E5](https://huggingface.co/intfloat/multilingual-e5-base) поддерживает конкретные префиксы query/passage, а не правило для всех encoder-моделей.
- [Longformer](https://arxiv.org/abs/2004.05150), v2 от 2 декабря 2020: локальные и глобальные пути; два слоя для удалённой передачи через global. [SciBERT](https://aclanthology.org/D19-1371/), 2019; [CodeBERT](https://arxiv.org/abs/2002.08155), v4 от 18 сентября 2020: корпус и цели не подменяют прикладную адаптацию. [ruBert-base](https://huggingface.co/ai-forever/ruBert-base) рассматривается как конкретная основа, а не готовая голова NER/поиска.
- [LayoutLM](https://arxiv.org/abs/1912.13318), v5 от 16 июня 2020: текст, пространственные координаты и визуальные признаки. Собственная схема обозначает визуальный путь с оговоркой о варианте; не утверждает, что исходная модель вовсе не использует изображение.
- [Калибровка](https://arxiv.org/abs/1706.04599), v2 от 3 августа 2017; [BEIR](https://arxiv.org/abs/2104.08663), v4 от 21 октября 2021; [MTEB](https://arxiv.org/abs/2210.07316), v3 от 19 марта 2023; [MMTEB](https://arxiv.org/abs/2502.13595), v4 от 13 ноября 2025: протоколы и составы задач, без переноса места рейтинга на произвольный корпус. [Integrated Gradients](https://arxiv.org/abs/1703.01365), v2 от 13 июня 2017: полнота относительно baseline/пути не объявлена доказательством человеческой причинности.
- [Дистилляция Hinton et al.](https://arxiv.org/abs/1503.02531), 2015; [квантизация ONNX Runtime](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html); [Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference): сохранение поведения, ограничения форматов и реализации. Поддержка произвольной головы и ускорение конкретного оборудования не утверждаются без запуска.

Рисунки пакета созданы локально в SVG; чужие иллюстрации не копировались. Их первичные источники механизма и авторский генератор перечислены в [реестре иллюстраций](<illustrations.json>). Числовые модели очереди, перекрытия и памяти — собственные учебные примеры.

Далее сохранён исторический реестр перестройки v2. Его даты и результаты не заменяют текущую проверку изменяемого материала.

Проверка при перестройке: **20 октября 2026**. Дата относится к редакционному реестру. В учебных объяснениях остаются первичные ссылки и ограничения утверждений; рейтинг по дате выпуска не используется.

## Включённые примеры

- **BERT:** Devlin et al., [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805), arXiv v2, 24.05.2019. Используется для определения bidirectional encoder и MLM recipe. Исторические benchmark results не переносятся на современный произвольный dataset. Место: `lm-bert`, `bert-objectives`.
- **ModernBERT:** Warner et al., [Smarter, Better, Faster, Longer](https://arxiv.org/abs/2412.13663), v2, 19.12.2024. Место: `bert-objectives`. Сопоставляются context, training recipe и runtime; результаты относятся к evaluation авторов.
- **NeoBERT:** Le Breton et al., [NeoBERT](https://arxiv.org/abs/2502.19587), v2, 06.06.2025. Место: `bert-objectives`. Проверена авторская страница; универсальное превосходство над другими encoder не утверждается.
- **Ettin:** Weller et al., [Seq vs Seq: An Open Suite of Paired Encoders and Decoders](https://arxiv.org/abs/2507.11412), v2, 12.03.2026. Место: `bert-objectives`. Используется как пример контролируемого сравнения encoder/decoder. **Исправление реестра:** прежний общий план ссылался на `2507.15864`, другую работу о NER; неверная ссылка сохранена только в архивном срезе.
- **TabPFN-3.5:** Prior Labs, [technical report](https://priorlabs.ai/technical-reports/tabpfn-3-5), 15.09.2026. Место: `df-task`, сравнение baseline. Источник — отчёт разработчика; TabArena/BeyondArena ranks ограничены его протоколами. Variant, license, API/local режим, стоимость и hardware являются условиями воспроизводимого применения.
- **DINOv3:** Meta, [официальный проект](https://ai.meta.com/research/dinov3/). Место: `cv-pretraining`, frozen-feature case. Backbone, readout и downstream evaluation разделены. Dense features не означают готовые boxes или masks без task head.
- **SAM 3:** Meta, [Segment Anything with Concepts](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/), 19.11.2025; [репозиторий авторов](https://github.com/facebookresearch/sam3). Места: `cv-instance`, `cv-open-vocabulary`. Интерфейс concept prompt и SA-Co отделены от closed-set COCO mask AP.
- **SAM 3.1:** Meta, [официальное обновление](https://ai.meta.com/blog/segment-anything-model-3/), 27.03.2026. Место: `cv-tracking`. Object multiplexing и приведённые H100 FPS — author-reported implementation results для объявленной нагрузки. Установка checkpoints и model execution этой редакцией не выполняются.
- **YOLO26 task heads:** [официальная документация Ultralytics](https://docs.ultralytics.com/models/yolo26/), динамическая документация, срез проверки указан выше. Таблица перечисляет отдельные semantic/depth/pose/OBB variants. Наличие detector checkpoint не означает наличие всех голов в одном forward. Места: `cv-detection`, `cv-semantic`, `cv-depth`.
- **AgentGym-RL:** Xi et al., [авторская статья](https://arxiv.org/abs/2509.08755), 10.09.2025. Место: `agent-rl`. Проверены interactive training, модульная среда и staged interaction horizon. OpenReview PDF при повторном открытии потребовал browser verification; доступная первичная arXiv-ссылка подтверждает механизм, но не используется для вывода о статусе площадки.
- **TRAJECT-Bench:** [репозиторий авторов](https://github.com/PengfeiHePower/TRAJECT-Bench), начальный release 21.08.2025. Место: `agent-evaluation`. Различаются tool inclusion, parameterization, order/dependency, execution accuracy и judge-based trajectory scores. OpenReview PDF при повторном открытии потребовал verification; код является доступным первичным подтверждением metric definitions.
- **ACEBench:** Chen et al., [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.697/), Findings of EMNLP 2025. Место: `agent-evaluation`. Benchmark scenarios и tool contracts ограничивают область вывода.

## Исследовательские кандидаты

Поведение нативных категорий дополнительно сверено с [XGBoost 3.4.2 categorical tutorial](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html), [LightGBM Advanced Topics](https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html) (ветка документации 4.7.0.99) и [CatBoost categorical features](https://catboost.ai/docs/en/features/categorical-features). Проверялись partitioning/vector leaves, recoding/serialization и missing/ordered-category conventions. Эти версии относятся к страницам документации, а не к установленным и запущенным библиотекам.

- Weng et al., [Clustering Properties of Self-Supervised Learning](https://proceedings.mlr.press/v267/weng25a.html), ICML/PMLR 267, 2025. Проверены название, авторы и площадка. Кандидат расширения `cl-frozen`/`cl-deep`; не объявлен обязательной заменой базовых алгоритмов.
- Ghriss, Monteleoni, [Deep Clustering via Probabilistic Ratio-Cut Optimization](https://proceedings.mlr.press/v258/ghriss25a.html), AISTATS/PMLR 258, 2025. Проверена первичная страница. Graph objective требует отдельного сценария; название не включено в ядро вместо объяснения.
- Uziel et al., [Clustering via Self-Supervised Diffusion](https://proceedings.mlr.press/v267/uziel25a.html), ICML/PMLR 267, 2025. Проверена первичная страница; оставлен исследовательским направлением.
- [SAMAR](https://arxiv.org/abs/2412.16854), arXiv, 22.12.2024. Проверено соответствие названия адаптивной регуляризации SAM. Статус за пределами проверенной страницы и воспроизводимость не утверждаются. Обязательная глава подробно объясняет исходный SAM.

Прочие прежние research-кандидаты сохраняются в [исходном пакете планов](<../archive/2026-10-20-before-rebuild/root/План подготовки курса.md>). Они не объявлены проверенными рекомендациями новой редакции. Включение требует механизма, условий сравнения, первичного источника и содержательной проверки.


## Расширение CV до 60 глав — 2026-10-25

Карта новых материалов, prerequisites, вычислительных схем и источников: [cv-expansion.json](<cv-expansion.json>). Новые авторские объяснения отделены от архивных единиц coverage; прежние вопросы и примеры не удалены. Проверки источников касаются механизмов, а не собственного воспроизведения benchmark.

- [PyTorch Conv2d reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html); проверено 25.10.2026; первичный механизм / implementation contract.
- [SIFT — Lowe, IJCV 2004](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf); проверено 25.10.2026; первичный механизм / implementation contract.
- [HOG — Dalal, Triggs, CVPR 2005](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf); проверено 25.10.2026; первичный механизм / implementation contract.
- [ResNet — He et al., CVPR 2016](https://arxiv.org/abs/1512.03385); проверено 25.10.2026; первичный механизм / implementation contract.
- [MobileNetV2 — Sandler et al., CVPR 2018](https://arxiv.org/abs/1801.04381); проверено 25.10.2026; первичный механизм / implementation contract.
- [EfficientNet — Tan, Le, ICML 2019](https://arxiv.org/abs/1905.11946); проверено 25.10.2026; первичный механизм / implementation contract.
- [ConvNeXt — Liu et al., CVPR 2022](https://arxiv.org/abs/2201.03545); проверено 25.10.2026; первичный механизм / implementation contract.
- [ViT — Dosovitskiy et al., ICLR 2021](https://arxiv.org/abs/2010.11929); проверено 25.10.2026; первичный механизм / implementation contract.
- [Swin — Liu et al., ICCV 2021](https://arxiv.org/abs/2103.14030); проверено 25.10.2026; первичный механизм / implementation contract.
- [PyTorch transfer learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html); проверено 25.10.2026; первичный механизм / implementation contract.
- [Focal loss — Lin et al., ICCV 2017](https://arxiv.org/abs/1708.02002); проверено 25.10.2026; первичный механизм / implementation contract.
- [Calibration — Guo et al., ICML 2017](https://arxiv.org/abs/1706.04599); проверено 25.10.2026; первичный механизм / implementation contract.
- [FPN — Lin et al., CVPR 2017](https://arxiv.org/abs/1612.03144); проверено 25.10.2026; первичный механизм / implementation contract.
- [PANet — Liu et al., CVPR 2018](https://arxiv.org/abs/1803.01534); проверено 25.10.2026; первичный механизм / implementation contract.
- [SimCLR — Chen et al., ICML 2020](https://arxiv.org/abs/2002.05709); проверено 25.10.2026; первичный механизм / implementation contract.
- [MoCo — He et al., CVPR 2020](https://arxiv.org/abs/1911.05722); проверено 25.10.2026; первичный механизм / implementation contract.
- [MAE — He et al., CVPR 2022](https://arxiv.org/abs/2111.06377); проверено 25.10.2026; первичный механизм / implementation contract.
- [DeiT — Touvron et al., ICML 2021](https://arxiv.org/abs/2012.12877); проверено 25.10.2026; первичный механизм / implementation contract.
- [DINO — Caron et al., ICCV 2021](https://arxiv.org/abs/2104.14294); проверено 25.10.2026; первичный механизм / implementation contract.
- [DINOv2 — Oquab et al., 2023/2024](https://arxiv.org/abs/2304.07193); проверено 25.10.2026; первичный механизм / implementation contract.
- [DINOv3 — Meta, official project](https://ai.meta.com/research/dinov3/); проверено 25.10.2026; первичный механизм / implementation contract.
- [FaceNet — Schroff et al., CVPR 2015](https://arxiv.org/abs/1503.03832); проверено 25.10.2026; первичный механизм / implementation contract.
- [COCO evaluation implementation](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py); проверено 25.10.2026; первичный механизм / implementation contract.
- [Faster R-CNN — Ren et al., NeurIPS 2015](https://arxiv.org/abs/1506.01497); проверено 25.10.2026; первичный механизм / implementation contract.
- [Mask R-CNN / RoIAlign — He et al., ICCV 2017](https://arxiv.org/abs/1703.06870); проверено 25.10.2026; первичный механизм / implementation contract.
- [FCOS — Tian et al., ICCV 2019](https://arxiv.org/abs/1904.01355); проверено 25.10.2026; первичный механизм / implementation contract.
- [YOLOv3 — Redmon, Farhadi, 2018](https://arxiv.org/abs/1804.02767); проверено 25.10.2026; первичный механизм / implementation contract.
- [YOLOv8 — Ultralytics documentation](https://docs.ultralytics.com/models/yolov8/); проверено 25.10.2026; первичный механизм / implementation contract.
- [Ultralytics v8.3.0 — loss implementation](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/loss.py); проверено 25.10.2026; первичный механизм / implementation contract.
- [Ultralytics v8.3.0 — task aligned assignment](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/tal.py); проверено 25.10.2026; первичный механизм / implementation contract.
- [DETR — Carion et al., ECCV 2020](https://arxiv.org/abs/2005.12872); проверено 25.10.2026; первичный механизм / implementation contract.
- [Deformable DETR — Zhu et al., ICLR 2021](https://arxiv.org/abs/2010.04159); проверено 25.10.2026; первичный механизм / implementation contract.
- [RT-DETR — Zhao et al., CVPR 2024](https://arxiv.org/abs/2304.08069); проверено 25.10.2026; первичный механизм / implementation contract.
- [DOTA — authors benchmark](https://captain-whu.github.io/DOTA/); проверено 25.10.2026; первичный механизм / implementation contract.
- [DeepLabv3+ — Chen et al., ECCV 2018](https://arxiv.org/abs/1802.02611); проверено 25.10.2026; первичный механизм / implementation contract.
- [U-Net — Ronneberger et al., MICCAI 2015](https://arxiv.org/abs/1505.04597); проверено 25.10.2026; первичный механизм / implementation contract.
- [nnU-Net — Isensee et al., 2018](https://arxiv.org/abs/1809.10486); проверено 25.10.2026; первичный механизм / implementation contract.
- [nnU-Net official repository](https://github.com/MIC-DKFZ/nnUNet); проверено 25.10.2026; первичный механизм / implementation contract.
- [SegFormer — Xie et al., NeurIPS 2021](https://arxiv.org/abs/2105.15203); проверено 25.10.2026; первичный механизм / implementation contract.
- [YOLACT — Bolya et al., ICCV 2019](https://arxiv.org/abs/1904.02689); проверено 25.10.2026; первичный механизм / implementation contract.
- [Mask2Former — Cheng et al., CVPR 2022](https://arxiv.org/abs/2112.01527); проверено 25.10.2026; первичный механизм / implementation contract.
- [Panoptic segmentation — Kirillov et al., CVPR 2019](https://arxiv.org/abs/1801.00868); проверено 25.10.2026; первичный механизм / implementation contract.
- [CLIP — Radford et al., ICML 2021](https://arxiv.org/abs/2103.00020); проверено 25.10.2026; первичный механизм / implementation contract.
- [Grounding DINO — Liu et al., ECCV 2024](https://arxiv.org/abs/2303.05499); проверено 25.10.2026; первичный механизм / implementation contract.
- [SAM — Kirillov et al., ICCV 2023](https://arxiv.org/abs/2304.02643); проверено 25.10.2026; первичный механизм / implementation contract.
- [SAM 3 — official repository](https://github.com/facebookresearch/sam3); проверено 25.10.2026; первичный механизм / implementation contract.
- [HRNet — Sun et al., CVPR 2019](https://arxiv.org/abs/1902.09212); проверено 25.10.2026; первичный механизм / implementation contract.
- [ViTPose — Xu et al., NeurIPS 2022](https://arxiv.org/abs/2204.12484); проверено 25.10.2026; первичный механизм / implementation contract.
- [RTMPose — Jiang et al., 2023](https://arxiv.org/abs/2303.07399); проверено 25.10.2026; первичный механизм / implementation contract.
- [SORT — Bewley et al., ICIP 2016](https://arxiv.org/abs/1602.00763); проверено 25.10.2026; первичный механизм / implementation contract.
- [DeepSORT — Wojke et al., ICIP 2017](https://arxiv.org/abs/1703.07402); проверено 25.10.2026; первичный механизм / implementation contract.
- [ByteTrack — Zhang et al., ECCV 2022](https://arxiv.org/abs/2110.06864); проверено 25.10.2026; первичный механизм / implementation contract.
- [BoT-SORT — Aharon et al., 2022](https://arxiv.org/abs/2206.14651); проверено 25.10.2026; первичный механизм / implementation contract.
- [RAFT — Teed, Deng, ECCV 2020](https://arxiv.org/abs/2003.12039); проверено 25.10.2026; первичный механизм / implementation contract.
- [SlowFast — Feichtenhofer et al., ICCV 2019](https://arxiv.org/abs/1812.03982); проверено 25.10.2026; первичный механизм / implementation contract.
- [TimeSformer — Bertasius et al., ICML 2021](https://arxiv.org/abs/2102.05095); проверено 25.10.2026; первичный механизм / implementation contract.
- [VideoMAE — Tong et al., NeurIPS 2022](https://arxiv.org/abs/2203.12602); проверено 25.10.2026; первичный механизм / implementation contract.
- [OpenCV camera calibration reference](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html); проверено 25.10.2026; первичный механизм / implementation contract.
- [PSMNet — Chang, Chen, CVPR 2018](https://arxiv.org/abs/1803.08669); проверено 25.10.2026; первичный механизм / implementation contract.
- [DPT — Ranftl et al., ICCV 2021](https://arxiv.org/abs/2103.13413); проверено 25.10.2026; первичный механизм / implementation contract.
- [Depth Anything V2 — Yang et al., NeurIPS 2024](https://arxiv.org/abs/2406.09414); проверено 25.10.2026; первичный механизм / implementation contract.
- [COLMAP tutorial — authors documentation](https://colmap.github.io/tutorial.html); проверено 25.10.2026; первичный механизм / implementation contract.
- [PointNet — Qi et al., CVPR 2017](https://arxiv.org/abs/1612.00593); проверено 25.10.2026; первичный механизм / implementation contract.
- [PointNet++ — Qi et al., NeurIPS 2017](https://arxiv.org/abs/1706.02413); проверено 25.10.2026; первичный механизм / implementation contract.
- [PointPillars — Lang et al., CVPR 2019](https://arxiv.org/abs/1812.05784); проверено 25.10.2026; первичный механизм / implementation contract.
- [NeRF — Mildenhall et al., ECCV 2020](https://arxiv.org/abs/2003.08934); проверено 25.10.2026; первичный механизм / implementation contract.
- [3DGS — Kerbl et al., SIGGRAPH 2023](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/); проверено 25.10.2026; первичный механизм / implementation contract.
- [CRAFT — Baek et al., CVPR 2019](https://arxiv.org/abs/1904.01941); проверено 25.10.2026; первичный механизм / implementation contract.
- [DBNet — Liao et al., AAAI 2020](https://arxiv.org/abs/1911.08947); проверено 25.10.2026; первичный механизм / implementation contract.
- [CRNN — Shi et al., TPAMI 2017](https://arxiv.org/abs/1507.05717); проверено 25.10.2026; первичный механизм / implementation contract.
- [TrOCR — Li et al., AAAI 2023](https://arxiv.org/abs/2109.10282); проверено 25.10.2026; первичный механизм / implementation contract.
- [LayoutLMv3 — Huang et al., ACM MM 2022](https://arxiv.org/abs/2204.08387); проверено 25.10.2026; первичный механизм / implementation contract.
- [PubTables-1M — Smock et al., CVPR 2022](https://arxiv.org/abs/2110.00061); проверено 25.10.2026; первичный механизм / implementation contract.
- [PaDiM — Defard et al., ICPR workshop 2020](https://arxiv.org/abs/2011.08785); проверено 25.10.2026; первичный механизм / implementation contract.
- [PatchCore — Roth et al., CVPR 2022](https://arxiv.org/abs/2106.08265); проверено 25.10.2026; первичный механизм / implementation contract.
- [SRCNN — Dong et al., TPAMI 2016](https://arxiv.org/abs/1501.00092); проверено 25.10.2026; первичный механизм / implementation contract.
- [VAE — Kingma, Welling, ICLR 2014](https://arxiv.org/abs/1312.6114); проверено 25.10.2026; первичный механизм / implementation contract.
- [GAN — Goodfellow et al., NeurIPS 2014](https://arxiv.org/abs/1406.2661); проверено 25.10.2026; первичный механизм / implementation contract.
- [DDPM — Ho et al., NeurIPS 2020](https://arxiv.org/abs/2006.11239); проверено 25.10.2026; первичный механизм / implementation contract.
- [DDIM — Song et al., ICLR 2021](https://arxiv.org/abs/2010.02502); проверено 25.10.2026; первичный механизм / implementation contract.
- [Classifier-free guidance — Ho, Salimans, 2022](https://arxiv.org/abs/2207.12598); проверено 25.10.2026; первичный механизм / implementation contract.
- [Latent diffusion — Rombach et al., CVPR 2022](https://arxiv.org/abs/2112.10752); проверено 25.10.2026; первичный механизм / implementation contract.
- [DiT — Peebles, Xie, ICCV 2023](https://arxiv.org/abs/2212.09748); проверено 25.10.2026; первичный механизм / implementation contract.
- [Flow Matching — Lipman et al., ICLR 2023](https://arxiv.org/abs/2210.02747); проверено 25.10.2026; первичный механизм / implementation contract.
- [ControlNet — Zhang et al., ICCV 2023](https://arxiv.org/abs/2302.05543); проверено 25.10.2026; первичный механизм / implementation contract.
- [LoRA — Hu et al., ICLR 2022](https://arxiv.org/abs/2106.09685); проверено 25.10.2026; первичный механизм / implementation contract.
- [FID — Heusel et al., NeurIPS 2017](https://arxiv.org/abs/1706.08500); проверено 25.10.2026; первичный механизм / implementation contract.
- [TIDE — Bolya et al., ECCV 2020](https://arxiv.org/abs/2008.08115); проверено 25.10.2026; первичный механизм / implementation contract.
- [ImageNet-C — Hendrycks, Dietterich, ICLR 2019](https://arxiv.org/abs/1903.12261); проверено 25.10.2026; первичный механизм / implementation contract.
- [Selective classification — Geifman, El-Yaniv, NeurIPS 2017](https://arxiv.org/abs/1705.08500); проверено 25.10.2026; первичный механизм / implementation contract.
- [ONNX Runtime quantization documentation](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html); проверено 25.10.2026; первичный механизм / implementation contract.

YOLOv8: implementation-срез Ultralytics v8.3.0. Текущие страницы YOLO26, DINOv3, SAM 3 и обновление SAM 3.1 повторно открыты 25.10.2026; author-reported скорости не превращены в собственные измерения. Рисунки статей сохранены локально; номера, страницы PDF, crop и хэши указаны в реестре иллюстраций.
