# Sentence embeddings и contrastive learning

**Уровень:** основа ветки.

**Предпосылки:** [Encoder, tokenizer, attention и pooling](<bert-interface.md>), [Данные, task heads, fine-tuning и DAPT](<bert-adaptation.md>)

**Используют:** [Dense retrieval, cross-encoder, SPLADE и ColBERT](<bert-retrieval.md>)

**Учебный вопрос:** Как обучить пространство, где близость полезна для конкретной задачи?





## Карта вычисления и границы этапов

![Sentence embeddings и contrastive learning: входы, преобразования и выходы](<../../оформление/course/bert-embeddings.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Сравните формы и смысл каждого перехода с формулами главы. Выход преобразования и доказательство его качества — разные объекты: последняя проверка использует отдельный протокол оценки.

<!-- unit:a12fec4d293f02f1c3 -->
<a id="u-a12fec4d293f02f1c3"></a>

## Независимое encoding и learned geometry

<!-- /unit:a12fec4d293f02f1c3 -->

<!-- unit:e305777b0555b88e63 -->
<a id="u-e305777b0555b88e63"></a>

Bi-encoder вычисляет vectors независимо: $z_x=\operatorname{pool}(f_\theta(x))$, $z_y=\operatorname{pool}(f_\theta(y))$, затем score. Shared weights образуют siamese architecture; separate query/document weights задают asymmetric model. Document vectors можно precompute, если representation не зависит от query.

<!-- /unit:e305777b0555b88e63 -->

<!-- unit:a8b9d65e7e2c9e67b6 -->
<a id="u-a8b9d65e7e2c9e67b6"></a>

[Sentence-BERT](https://aclanthology.org/D19-1410/) адаптирует encoder под sentence comparisons. Это не просто mean pooling любого MLM backbone: objective влияет на geometry. Pooling/projection/prefix/normalization сохраняются вместе с checkpoint. Cross-encoder обрабатывает pair jointly и может выучить более детальные interactions, но обычно не даёт reusable одного document vector, заменяющего всю pair computation.

<!-- /unit:a8b9d65e7e2c9e67b6 -->

<!-- unit:dcb20b11bf74a57ea7 -->
<a id="u-dcb20b11bf74a57ea7"></a>

Dot product учитывает norm и direction, cosine — только direction. Query $(1,0)$, documents A=(2,0), B=(0.9,0.1): dot A=2,B=0.9, cosine A=1,B около 0.9939. Если у другого document huge norm при менее точном direction, dot может предпочесть его. После L2 normalization cosine/dot совпадают, а squared Euclidean равен $2-2z_x^\top z_y$. До normalization rankings могут различаться.

<!-- /unit:dcb20b11bf74a57ea7 -->

<!-- unit:35825484ca873551d9 -->
<a id="u-35825484ca873551d9"></a>

<!-- source-depth-bert-Д -->

<!-- /unit:35825484ca873551d9 -->

<!-- unit:8398473f2ec4deb2a7 -->
<a id="u-8398473f2ec4deb2a7"></a>

### Вопрос S6: «Почему мы уверены, что модель поймёт наш embedding?»

<!-- /unit:8398473f2ec4deb2a7 -->

<!-- unit:f23192ba5060c3de9a -->
<a id="u-f23192ba5060c3de9a"></a>

В разбираемом примере, строки 685–845, вы заметили, что embedding — внутреннее представление модели, и спросили, почему другая модель должна его понимать. Для обычного retrieve-then-read pipeline ответ состоит из двух разных операций. Retriever преобразует query и документы в совместимое пространство и выбирает document IDs. По IDs извлекаются **исходные тексты**. Generator получает эти тексты в своём входном формате и заново вычисляет собственные token embeddings и hidden states. Числа retrieval-вектора ему напрямую читать не требуется.

<!-- /unit:f23192ba5060c3de9a -->

<!-- unit:61af50c60dca2bde57 -->
<a id="u-61af50c60dca2bde57"></a>

Поисковый вектор отвечает на вопрос «какие документы выбрать», а encoder states генератора — «как обработать переданный вход». Их нельзя отождествлять только потому, что оба являются массивами float. Embedding-модель может сама быть Transformer encoder или иной моделью с pooling/projection; слово embedding не задаёт отдельный универсальный тип архитектуры. Размерность также не определяет назначение: нет общего правила, что retrieval embeddings всегда лежат только в диапазоне 384–1024, а hidden states — только 4096–12288.

<!-- /unit:61af50c60dca2bde57 -->

<!-- unit:b9d7f2707a19d4fb33 -->
<a id="u-b9d7f2707a19d4fb33"></a>

Фраза исходного ответа «никогда не подаются в LLM» относится лишь к разобранному простому pipeline и слишком категорична как универсальное утверждение. Существуют обученные интерфейсы для внешних представлений. Но одинаковая длина вектора не делает произвольную инъекцию корректной. Нужны согласованное пространство, обученное преобразование или архитектура, которая специально умеет его читать.

<!-- /unit:b9d7f2707a19d4fb33 -->

<!-- unit:6614f14ff74d3b856c -->
<a id="u-6614f14ff74d3b856c"></a>

Отдельное исправление: original RAG не нужно описывать как прямую передачу DPR search embeddings в cross-attention генератора. Retriever выбирает passages; BART получает query и passage как входной текст, а decoder читает состояния своего encoder. Эти состояния отличаются от retrieval-векторов. Разделение видно в [описании RAG generator](https://arxiv.org/html/2005.11401v4#S2.SS3). CLIP также не является сам по себе генеративной LLM, читающей произвольные embeddings. Эти оговорки сохраняют смысл вашего вопроса без смешения интерфейсов.

<!-- /unit:6614f14ff74d3b856c -->

<!-- unit:3574b6a2252f7818e2 -->
<a id="u-3574b6a2252f7818e2"></a>

### Почему одинаковая размерность не означает совместимое пространство

<!-- /unit:3574b6a2252f7818e2 -->

<!-- unit:d58cb7a9a48f3324bf -->
<a id="u-d58cb7a9a48f3324bf"></a>

Пусть в пространстве A query=(1,0), правильный документ d+=(1,0), неправильный d−=(0,1). Dot product выбирает правильный. Теперь новая модель B повернула **все** свои векторы на 90°: query B=(0,1), d+ B=(0,1), d− B=(−1,0). Внутри B правильный документ по-прежнему получает score 1, неправильный 0. Геометрия задачи сохранена.

<!-- /unit:d58cb7a9a48f3324bf -->

<!-- unit:14db7cc6a9d04d8731 -->
<a id="u-14db7cc6a9d04d8731"></a>

Но если новый query B сравнить со старым индексом A, scores будут 0 для правильного и 1 для неправильного. Совпали размерность, формат float и смысл обучения — а смешанный index сломался из-за несогласованных координат. Это точный маленький пример необходимости переиндексации или обоснованного alignment при смене encoder.

<!-- /unit:14db7cc6a9d04d8731 -->

<!-- unit:84698a66572f3b2be7 -->
<a id="u-84698a66572f3b2be7"></a>

Ортогональная матрица R сохраняет внутренние dot products: $(Rq)^\top(Rd)=q^\top d$. Но смешанный score $(Rq)^\top d$ в общем случае другой. Поэтому coordinate MSE между independently trained models может быть большим при одинаковых внутримодельных similarities. И наоборот, небольшой MSE на ограниченной выборке не гарантирует совместимость каждого нового документа.

<!-- /unit:84698a66572f3b2be7 -->

<!-- unit:cb3fed4b1d908026db -->
<a id="u-cb3fed4b1d908026db"></a>

### Как norm меняет ранжирование

<!-- /unit:cb3fed4b1d908026db -->

<!-- unit:280f1680c976123daf -->
<a id="u-280f1680c976123daf"></a>

В приведённом выше примере A=(2,0) остаётся первым и по dot, и по cosine, так что он иллюстрирует формулы, но не перестановку ranking. Для явного расхождения возьмём q=(1,0), A=(1,0), B=(2,2). Dot предпочитает B: 2 против 1; cosine предпочитает A: 1 против $1/\sqrt2$. Большая norm B компенсировала менее точное направление.

<!-- /unit:280f1680c976123daf -->

<!-- unit:7b6be1824b9e81a043 -->
<a id="u-7b6be1824b9e81a043"></a>

Если нормировать все ненулевые векторы до единичной длины, dot и cosine совпадут. Кроме того, $\|q-d\|^2=2-2q^\top d$, поэтому максимальный dot соответствует минимальному squared L2. Без нормировки или при нулевом векторе эти выводы нельзя применять автоматически. Выбранная метрика и normalization должны совпадать при обучении, индексировании и query encoding.

<!-- /unit:7b6be1824b9e81a043 -->

<!-- unit:a81ef6875eb14957a4 -->
<a id="u-a81ef6875eb14957a4"></a>

![Диагональные positives — лишь один способ организации batch. Если вне диагонали есть другой правильный ответ, его нельзя бездумно отталкивать как negative.](../../archive/2026-10-20-before-rebuild/оформление/transformer/contrastive-positive-matrix.svg)

<!-- /unit:a81ef6875eb14957a4 -->

<!-- unit:070487da95393fcdbd -->
<a id="u-070487da95393fcdbd"></a>

Диагональные positives — лишь один способ организации batch. Если вне диагонали есть другой правильный ответ, его нельзя бездумно отталкивать как negative.

<!-- /unit:070487da95393fcdbd -->

<!-- unit:55b6b9a9e34c3792ac -->
<a id="u-55b6b9a9e34c3792ac"></a>

## Objectives и supervision

<!-- /unit:55b6b9a9e34c3792ac -->

<!-- unit:b8e3feb238813cd21d -->
<a id="u-b8e3feb238813cd21d"></a>

Cosine regression подгоняет similarity к заданному continuous label; scale label должен соответствовать score. Triplet loss требует anchor, positive и negative:

<!-- /unit:b8e3feb238813cd21d -->

<!-- unit:dd4a651bcebb27fed3 -->
<a id="u-dd4a651bcebb27fed3"></a>

$$
L_{\mathrm{triplet}}=\max\{0,m+d(a,p)-d(a,n)\}.
$$

<!-- /unit:dd4a651bcebb27fed3 -->

<!-- unit:cdc0c0e4e2c82d8ec5 -->
<a id="u-cdc0c0e4e2c82d8ec5"></a>

При margin 0.2, positive distance 0.3, negative 0.4 loss равна 0.1. После достаточного separation gradient этой triplet становится нулевым; mining меняет долю информативных cases. Это другой objective, чем CE по in-batch candidates.

<!-- /unit:cdc0c0e4e2c82d8ec5 -->

<!-- unit:379f48629f9b838ef1 -->
<a id="u-379f48629f9b838ef1"></a>

Для $N$ query-positive pairs InfoNCE/multiple-negatives form:

<!-- /unit:379f48629f9b838ef1 -->

<!-- unit:a4bb484fed6f55c28a -->
<a id="u-a4bb484fed6f55c28a"></a>

$$
S_{ij}=\frac{\operatorname{sim}(q_i,d_j)}\tau,
\qquad
L_i=-\log\frac{e^{S_{ii}}}{\sum_j e^{S_{ij}}},
\qquad L=\frac1N\sum_iL_i.
$$

<!-- /unit:a4bb484fed6f55c28a -->

<!-- unit:18fe6669b6b26a0164 -->
<a id="u-18fe6669b6b26a0164"></a>

First row logits $(2,1,0)$ даёт probabilities примерно $(0.665241,0.244728,0.090031)$ и loss $0.407606$. Gradient строки по logits — $(-0.334759,0.244728,0.090031)$; при общем mean по N он делится на N. По unscaled similarity добавляется factor $1/\tau$. Маленькая temperature делает softmax sharper и меняет gradient scale; это не inference sampling temperature causal LM.

<!-- /unit:18fe6669b6b26a0164 -->

<!-- unit:be7c65a4804c64f754 -->
<a id="u-be7c65a4804c64f754"></a>

Symmetric loss добавляет обратное direction document→query по выбранному protocol. Оно уместно не для любой asymmetric relevance task. Несколько positives требуют mask/multi-positive objective: второй релевантный document не должен автоматически считаться negative.

<!-- /unit:be7c65a4804c64f754 -->

<!-- unit:e519d729f50c19506e -->
<a id="u-e519d729f50c19506e"></a>

### Как InfoNCE учит геометрию, а не выдаёт координаты вручную

<!-- /unit:e519d729f50c19506e -->

<!-- unit:9893fa14e1d23d8bd6 -->
<a id="u-9893fa14e1d23d8bd6"></a>

В строке i query сравнивается с несколькими candidate documents. Loss требует выделить positive среди этих candidates. Для dot similarity без дополнительной нормировки производная по query имеет вид

<!-- /unit:9893fa14e1d23d8bd6 -->

<!-- unit:a9e8cd14323aa9c40a -->
<a id="u-a9e8cd14323aa9c40a"></a>

$$
\frac{\partial L_i}{\partial q_i}=\frac1\tau\left(\sum_jp_{ij}d_j-d_i\right).
$$

<!-- /unit:a9e8cd14323aa9c40a -->

<!-- unit:5629d1c1b516287cb3 -->
<a id="u-5629d1c1b516287cb3"></a>

Gradient descent сдвигает q в сторону positive относительно текущей вероятностной смеси всех candidates. Это точнее, чем правило из разбираемого примера «похожие всегда cosine≈1, разные≈0». Objective зависит от сравнительных scores и состава negatives; он не обязывает каждому unrelated тексту иметь ровно нулевой cosine. При cosine similarity дополнительный backward проходит через normalization vectors, поэтому формула непосредственно относится к dot по тем переменным, которые подаются в score.

<!-- /unit:5629d1c1b516287cb3 -->

<!-- unit:3aa011a2c6ae8d8138 -->
<a id="u-3aa011a2c6ae8d8138"></a>

Для q=(1,0), positive d1=(1,0), negative d2=(0,1), τ=1 logits равны (1,0). Вероятности примерно (0.731059,0.268941), loss около 0.313262, gradient query=(−0.268941,0.268941). Малый шаг gradient descent усиливает первый компонент и ослабляет второй. Document gradients одновременно меняют сами candidate representations; при shared encoder все эти пути складываются в одни weights.

<!-- /unit:3aa011a2c6ae8d8138 -->

<!-- unit:4f90ccfe5fe2fcabf7 -->
<a id="u-4f90ccfe5fe2fcabf7"></a>

Полный batch loss добавляет contributions нескольких queries. Один document может быть positive собственной query и negative других. Поэтому batching и наличие альтернативных релевантных документов влияют на то, чему учится геометрия. Если d2 тоже правильный ответ q1, объявление его negative создаёт конфликт, а не полезную трудность.

<!-- /unit:4f90ccfe5fe2fcabf7 -->

<!-- unit:99b96bf392ec22e219 -->
<a id="u-99b96bf392ec22e219"></a>

### Несколько positives задают несколько разных objectives

<!-- /unit:99b96bf392ec22e219 -->

<!-- unit:d74c3cf73722732d58 -->
<a id="u-d74c3cf73722732d58"></a>

Для query с множеством positives P можно использовать loss $-\log\sum_{j\in P}p_j$: достаточно высокой суммарной массы positive set. Другой вариант — усреднить $-\log p_j$ по positives; он требует распределять вероятность между ними по другой цели. Это не тождественные функции.

<!-- /unit:d74c3cf73722732d58 -->

<!-- unit:c32aa28491a319a7cd -->
<a id="u-c32aa28491a319a7cd"></a>

Например, $p=(0.5,0.3,0.2)$, первые два candidates релевантны. Set-mass loss равна $-\log0.8\approx0.223144$. Среднее двух positive CE равно $-(\log0.5+\log0.3)/2\approx0.948560$. Они по-разному оценивают ситуацию, когда почти вся positive mass сосредоточена на одном из двух документов. Нужно выбрать правило под supervision и метрику, а не просто пометить дополнительный positive и оставить прежний denominator/labels без проверки.

<!-- /unit:c32aa28491a319a7cd -->

<!-- unit:ef6604f4dc0392598f -->
<a id="u-ef6604f4dc0392598f"></a>

### Temperature меняет распределение ошибки

<!-- /unit:ef6604f4dc0392598f -->

<!-- unit:bf20daec21522ea34c -->
<a id="u-bf20daec21522ea34c"></a>

Для logits (1,0) при τ=1 positive probability около 0.731059; при τ=0.5 scores становятся (2,0), probability около 0.880797. При этом градиент по **исходным similarities** дополнительно делится на τ. Поэтому меньшая temperature не означает, что каждый gradient всегда меньше: его величина зависит и от трудности примера, и от насыщения softmax.

<!-- /unit:bf20daec21522ea34c -->

<!-- unit:73b9e3c4feb2b1fa7f -->
<a id="u-73b9e3c4feb2b1fa7f"></a>

Это training scale задачи сопоставления candidates. Он отличается по назначению от temperature при sampling следующего токена. Совпадение математической операции «делить logits» не делает два pipeline одинаковыми. Для нормированных vectors τ особенно заметно меняет мягкость выбора, поскольку cosine ограничен диапазоном [−1,1].

<!-- /unit:73b9e3c4feb2b1fa7f -->

<!-- unit:e875c317eb96357437 -->
<a id="u-e875c317eb96357437"></a>

## Hard negatives, batch и SimCSE

<!-- /unit:e875c317eb96357437 -->

<!-- unit:df06f8e74d2485a861 -->
<a id="u-df06f8e74d2485a861"></a>

Random negatives часто просты; mined lexical/semantic negatives учат различать близкие irrelevant candidates. Но near-duplicate/alternative relevant passages становятся false negatives, и off-diagonal gradient тогда отталкивает полезный document. Teacher ranking не безошибочен; judgement provenance и review важны.

<!-- /unit:df06f8e74d2485a861 -->

<!-- unit:41cca634ca5ff8f6a2 -->
<a id="u-41cca634ca5ff8f6a2"></a>

В [SimCSE](https://arxiv.org/abs/2104.08821) unsupervised positive views получают разные dropout masks одного text; supervised variant использует NLI information. Два deterministic identical passes задают другой training signal. Это не утверждение, что любая dropout perturbation сохраняет всё нужное meaning.

<!-- /unit:41cca634ca5ff8f6a2 -->

<!-- unit:278f842df6a545ed72 -->
<a id="u-278f842df6a545ed72"></a>

Contrastive negative pool отличается от optimization batch. Accumulation нескольких independent losses не создаёт cross-microbatch terms. Distributed all-gather/cached losses могут расширять pool, но имеют requirements gradient correctness, stale representations и memory. [Loss implementations и data formats](https://sbert.net/docs/package_reference/sentence_transformer/losses.html) сверяются с installed version.

<!-- /unit:278f842df6a545ed72 -->

<!-- unit:e214ad70f780c72395 -->
<a id="u-e214ad70f780c72395"></a>

Mining проводится без final held-out labels; cross-validation включает encoder training. Иначе supervision test попадёт в representation до оценки downstream classifier/clustering.

<!-- /unit:e214ad70f780c72395 -->

<!-- unit:92ddf8c3150217e229 -->
<a id="u-92ddf8c3150217e229"></a>

### Отрицательный пример — суждение о задаче, а не просто другой текст

<!-- /unit:92ddf8c3150217e229 -->

<!-- unit:f547b5013f522fb4cf -->
<a id="u-f547b5013f522fb4cf"></a>

Для запроса «как сбросить пароль» документы «смена пароля в профиле» и «восстановление доступа без старого пароля» могут оба быть полезными, но отвечать разным сценариям. Документ о смене имени пользователя лексически похож и может быть трудным negative. Разметка должна учитывать вопрос, а не только совпадение ключевого слова «профиль».

<!-- /unit:f547b5013f522fb4cf -->

<!-- unit:cca5fbcb951ba048a4 -->
<a id="u-cca5fbcb951ba048a4"></a>

В in-batch loss чужой positive часто автоматически становится negative текущей query. Если две query описывают одну задачу или документы дублируются, такая автоматизация ошибочна. Deduplication, positive masks и review mined negatives исправляют supervision, а не просто очищают косметический шум. Очень трудный false negative способен давать большой градиент именно потому, что модель правильно считает его релевантным.

<!-- /unit:cca5fbcb951ba048a4 -->

<!-- unit:17fc23e5fde1f06e32 -->
<a id="u-17fc23e5fde1f06e32"></a>

Hard-negative mining должен использовать разрешённые train данные. Если взять ошибки на финальном test и дообучить на них encoder, независимость оценки потеряется. При обновлении encoder полезно пересматривать mined candidates: трудность зависит от текущего пространства, а старые negatives могут стать слишком простыми. Это дополнительная стоимость training pipeline.

<!-- /unit:17fc23e5fde1f06e32 -->

<!-- unit:4a497edd0e1146fbe3 -->
<a id="u-4a497edd0e1146fbe3"></a>

### SimCSE и два разных смысла batch

<!-- /unit:4a497edd0e1146fbe3 -->

<!-- unit:4ca81e96c54b436472 -->
<a id="u-4ca81e96c54b436472"></a>

Два stochastic forward одного текста с разными dropout masks дают две близкие по входу views. Contrastive objective учит сохранять нужное представление при этих возмущениях. Это использует randomness модели, а не создаёт новый независимый документ. Одинаковые deterministic views дают другое распределение обучения; точные рецепты unsupervised и supervised вариантов различаются в [SimCSE](https://arxiv.org/abs/2104.08821).

<!-- /unit:4ca81e96c54b436472 -->

<!-- unit:1745502fdce5390865 -->
<a id="u-1745502fdce5390865"></a>

Optimization batch определяет, после скольких contributions обновятся weights. Negative pool определяет, какие candidates присутствуют в denominator каждого query. Два microbatches по две пары при независимых losses дают каждому query два candidates; объединённый batch из четырёх пар — четыре. Gradient accumulation сам по себе не создаёт отсутствующие similarity terms между microbatches.

<!-- /unit:1745502fdce5390865 -->

<!-- unit:34445f368862395d1e -->
<a id="u-34445f368862395d1e"></a>

Если negatives всего одна пара и denominator содержит только собственный positive, softmax probability равна 1 и эта loss равна 0 независимо от качества embedding. Добавить accumulation по нескольким таким одиночным losses недостаточно: сумма нулей не создаст контрастивного различения. Нужен реальный negative pool или другой training objective.

<!-- /unit:34445f368862395d1e -->

<!-- unit:f9070bd3b12f3f634e -->
<a id="u-f9070bd3b12f3f634e"></a>

## Distillation и размерность

<!-- /unit:f9070bd3b12f3f634e -->

<!-- unit:b9f2b99b4cfbeb4253 -->
<a id="u-b9f2b99b4cfbeb4253"></a>

Для teacher column $t\in\mathbb R^{768}$ и student $s\in\mathbb R^{256}$ coordinate matching требует общего space, например fixed/обоснованного projection $P\in\mathbb R^{256\times768}$:

<!-- /unit:b9f2b99b4cfbeb4253 -->

<!-- unit:07cf9294125f7869e9 -->
<a id="u-07cf9294125f7869e9"></a>

$$
L_{\mathrm{coord}}=\|s-Pt\|_2^2.
$$

<!-- /unit:07cf9294125f7869e9 -->

<!-- unit:70f8f19c8c76780e24 -->
<a id="u-70f8f19c8c76780e24"></a>

Если обе стороны/проекция свободно обучаются без других constraints, collapse может сделать loss малым. Поэтому target, detach и normalization задаются явно. Alternative — pairwise score MSE либо KL между teacher/student candidate distributions. Score matching сохраняет выбранные relationships без требования одинаковых coordinates, но ограниченный набор pairs не определяет всю geometry.

<!-- /unit:70f8f19c8c76780e24 -->

<!-- unit:ee707fc4c2d8ae55e3 -->
<a id="u-ee707fc4c2d8ae55e3"></a>

Teacher language coverage, data и capacity student влияют независимо от loss. Прямое copy coordinates разных independently trained spaces не имеет смысла без alignment: rotations могут сохранить similarities и изменить coordinates. Teacher encoding cost включается в training budget.

<!-- /unit:ee707fc4c2d8ae55e3 -->

<!-- unit:c7e6d81161b29619dc -->
<a id="u-c7e6d81161b29619dc"></a>

### Почему loss только на координатах может схлопнуть обе стороны

<!-- /unit:c7e6d81161b29619dc -->

<!-- unit:8497b31bc1babb4241 -->
<a id="u-8497b31bc1babb4241"></a>

В формуле $\|s-Pt\|^2$ фиксированный teacher задаёт цель, а обучаемые части student должны её воспроизвести. Если позволить и s, и t, и P свободно двигаться только ради этой ошибки, решение s=0 и Pt=0 даст loss 0 без полезного различения текстов. Даже при фиксированном t свободный P=0 и s=0 создаёт такую проблему, если нет ограничений и других задач. Поэтому trainable set и дополнительные objectives нужно объявить явно.

<!-- /unit:8497b31bc1babb4241 -->

<!-- unit:9ed2fa27794b1e59e6 -->
<a id="u-9ed2fa27794b1e59e6"></a>

Можно сопоставлять pairwise scores вместо coordinates. Если teacher даёт распределение (0.7,0.2,0.1) по трём candidates, student обучают похожему распределению через выбранную KL или CE. Это передаёт отношения внутри candidate set, не требуя одинаковой размерности vectors. Но по трём candidates нельзя доказать сохранение ranking по всему corpus: нужны покрытие training pairs и отдельная evaluation.

<!-- /unit:9ed2fa27794b1e59e6 -->

<!-- unit:00f7d2ff5dcf08bff4 -->
<a id="u-00f7d2ff5dcf08bff4"></a>

Размерность — лишь одна ресурсная ось. Для миллиона vectors float32 ширины 768 сырой массив занимает 3 072 000 000 bytes; ширины 256 — 1 024 000 000 bytes, в три раза меньше. Индекс, IDs, metadata и тексты добавляют собственные затраты. Сжатие координат само по себе не обещает трёхкратного уменьшения всей БД или latency.

<!-- /unit:00f7d2ff5dcf08bff4 -->

<!-- unit:d8330afc915d40ee27 -->
<a id="u-d8330afc915d40ee27"></a>

## Matryoshka и geometry diagnostics

<!-- /unit:d8330afc915d40ee27 -->

<!-- unit:38ff1149e88a9eae86 -->
<a id="u-38ff1149e88a9eae86"></a>

Matryoshka Representation Learning обучает несколько prefixes одного vector:

<!-- /unit:38ff1149e88a9eae86 -->

<!-- unit:108bc8703d034931ea -->
<a id="u-108bc8703d034931ea"></a>

$$
L=\sum_{d\in\mathcal D}\alpha_d\,
L_d\!\left(\operatorname{normalize}(z_{1:d})\right).
$$

<!-- /unit:108bc8703d034931ea -->

<!-- unit:6e1be7a87a4a11dc63 -->
<a id="u-6e1be7a87a4a11dc63"></a>

Конкретные task losses/weights зависят от recipe. Prefix ordinary embedding без такого обучения не имеет гарантированной полезности. После truncation нужна согласованная renormalization/similarity и новый index contract. [Matryoshka](https://arxiv.org/abs/2205.13147).

<!-- /unit:6e1be7a87a4a11dc63 -->

<!-- unit:e3df710f32b383402f -->
<a id="u-e3df710f32b383402f"></a>

Anisotropy — concentration vectors в ограниченных направлениях; hubness — некоторые points часто становятся neighbours многим queries. Norm drift влияет на dot-product ranking. Collapse делает vectors почти одинаковыми и уничтожает discriminative information. Проверять нужно variance, norms, neighbour quality и target task, а не только projection plot. Whitening/centering fit на разрешённой training части и сами меняют geometry.

<!-- /unit:e3df710f32b383402f -->

<!-- unit:4296eae502a7a5ba4f -->
<a id="u-4296eae502a7a5ba4f"></a>

### Prefix truncation меняет направление и требует своего обучения

<!-- /unit:4296eae502a7a5ba4f -->

<!-- unit:0037b73f2b6e57afb9 -->
<a id="u-0037b73f2b6e57afb9"></a>

Возьмём q=(1,1), d1=(1,0), d2=(0,1). В полном двумерном cosine оба документа одинаково близки q. Если оставить только первую координату, d2 станет нулевым вектором, для которого обычный cosine не определён. Техническое обрезание возможно, но полезность нового пространства из этого не следует.

<!-- /unit:0037b73f2b6e57afb9 -->

<!-- unit:5dd6a365d136951b09 -->
<a id="u-5dd6a365d136951b09"></a>

Matryoshka training включает losses для нескольких первых d координат, чтобы они сами содержали полезный сигнал. После выбора d нужна соответствующая normalization; нормировать полный вектор и затем просто отрезать хвост — не то же самое, что нормировать выбранный prefix. Метрика и index строятся для фактически используемого представления.

<!-- /unit:5dd6a365d136951b09 -->

<!-- unit:988827976742bfdd12 -->
<a id="u-988827976742bfdd12"></a>

### Что отличает anisotropy, hubness и collapse

<!-- /unit:988827976742bfdd12 -->

<!-- unit:91f0fc6fd82e597a40 -->
<a id="u-91f0fc6fd82e597a40"></a>

Если все vectors имеют большой общий компонент (10,ε), разные тексты могут показывать высокий cosine даже при слабом различении во второй координате. Это иллюстрирует концентрацию направлений. Если один документ часто входит в top-k несвязанных запросов, наблюдается hubness; причина может быть в геометрии, norm или общих формулировках. Если все vectors почти одинаковы, ranking теряет полезную информацию — collapse.

<!-- /unit:91f0fc6fd82e597a40 -->

<!-- unit:794de0f2cbc38ba634 -->
<a id="u-794de0f2cbc38ba634"></a>

Эти симптомы не определяются одной красивой двумерной проекцией. Смотрят распределения norms/similarities, variance, частоту соседства и примеры ошибок на целевой задаче. Centering или whitening может менять ranking; параметры преобразования оценивают на допустимой train части и сохраняют в inference pipeline. Улучшение геометрического proxy не гарантирует улучшение retrieval или классификации.

<!-- /unit:794de0f2cbc38ba634 -->

<!-- unit:0b30a50c2eaa7e2321 -->
<a id="u-0b30a50c2eaa7e2321"></a>

## STS, duplicates, clustering, anomalies и рекомендации

<!-- /unit:0b30a50c2eaa7e2321 -->

<!-- unit:13501639589ee8df11 -->
<a id="u-13501639589ee8df11"></a>

STS оценивает graded similarity, duplicate detection — identity по policy, retrieval — полезность ответа query. A≈B и B≈C не доказывают, что A duplicate C: transitive closure threshold graph может слить разные records цепочкой. Оценивать следует pair precision и ошибки resulting groups.

<!-- /unit:13501639589ee8df11 -->

<!-- unit:ad1cdfd75283f4ae0b -->
<a id="u-ad1cdfd75283f4ae0b"></a>

Clustering требует stability/semantics из отдельного курса. Необычный text не автоматически harmful/anomalous outcome. Item-text vectors могут помочь cold-start recommendations, но textual closeness не равна user preference; нужны behavioural labels с учётом exposure/popularity и temporal split.

<!-- /unit:ad1cdfd75283f4ae0b -->

<!-- unit:83ba68cae2dd16c392 -->
<a id="u-83ba68cae2dd16c392"></a>

Одна STS correlation не доказывает полезность всех приложений. Сравнивайте lexical baseline, frozen sentence-trained checkpoint и task adaptation на независимых cases с числами/отрицаниями/rare terms. Reindexing cost и compatibility версий учитываются при смене representation.

<!-- /unit:83ba68cae2dd16c392 -->

<!-- unit:9f9e83242928499a0f -->
<a id="u-9f9e83242928499a0f"></a>

### Почему один threshold может слить разные сущности цепочкой

<!-- /unit:9f9e83242928499a0f -->

<!-- unit:dcb90efabf90a12547 -->
<a id="u-dcb90efabf90a12547"></a>

Расположим три unit vectors под углами 0°,40°,80°. Соседние cosine равны примерно 0.766044, а крайние — 0.173648. При пороге 0.7 пары A–B и B–C проходят, A–C — нет. Если взять connected components threshold graph, все три попадут в одну группу. Значит, pair threshold и политика построения групп — разные части duplicate pipeline.

<!-- /unit:dcb90efabf90a12547 -->

<!-- unit:d574f66b98cb744416 -->
<a id="u-d574f66b98cb744416"></a>

Для deduplication ошибка объединения двух разных записей может быть гораздо важнее пропущенной похожей пары; это определяется задачей. Для semantic clustering, напротив, цепочка тематических связей может быть допустима. Нельзя переносить duplicate threshold на clustering только потому, что vectors те же.

<!-- /unit:d574f66b98cb744416 -->

<!-- unit:9691a137514b1f05cf -->
<a id="u-9691a137514b1f05cf"></a>

Item-text embeddings помогают сравнить описание нового объекта с существующими без истории взаимодействий. Но сходство текстов не равно предпочтению конкретного пользователя: поведенческие labels, показанные candidates и время наблюдения задают другую цель. Для anomaly detection необычное направление векторов тоже не является автоматически нежелательным событием. Каждое приложение должно назвать свой target и независимый тест.

<!-- /unit:9691a137514b1f05cf -->

<!-- unit:d7767b4aacf126e829 -->
<a id="u-d7767b4aacf126e829"></a>

## Контроль и практика

<!-- /unit:d7767b4aacf126e829 -->

<!-- unit:c0d1d0d3da10a5a1d0 -->
<a id="u-c0d1d0d3da10a5a1d0"></a>

1. Что делает false negative в InfoNCE? **Ответ:** off-diagonal pressure уменьшает similarity релевантного candidate, противореча задаче.
2. Гарантирует ли coordinate MSE одинаковый ranking всюду? **Ответ:** нет; зависит от coverage/normalization/projection и fit error.
3. Можно ли обрезать любой vector как Matryoshka? **Ответ:** технически да, но полезность prefixes без обучения/validation не гарантируется.
4. Равна ли document similarity предпочтению пользователя? **Ответ:** нет; recommendation supervision — отдельный target.

<!-- /unit:c0d1d0d3da10a5a1d0 -->

<!-- unit:b79a2b2c1e607c4828 -->
<a id="u-b79a2b2c1e607c4828"></a>

5. Должен ли generator понимать retrieval-вектор в обычном retrieve-then-read RAG? **Ответ:** нет. Retriever выбирает IDs, затем generator получает тексты и строит собственные представления.
6. Совместимы ли два embedding spaces одинаковой ширины? **Ответ:** это не гарантировано. Даже общий поворот сохраняет внутримодельную геометрию, но ломает смешанное query/index сравнение.
7. Почему InfoNCE с единственным собственным positive не различает документы? **Ответ:** denominator содержит только numerator, probability всегда 1, loss и соответствующий gradient равны нулю.
8. Делает ли threshold similarity отношение duplicate транзитивным? **Ответ:** нет. Цепочка A–B–C может объединить A и C, хотя их pair score ниже порога.

<!-- /unit:b79a2b2c1e607c4828 -->

<!-- unit:0c6834c43016a923ac -->
<a id="u-0c6834c43016a923ac"></a>

**Практика:** вычислите строку InfoNCE и gradient, затем объявите второй document тоже positive и объясните конфликт. Сравните dot/cosine разных norms. Спроектируйте teacher/student dimension mapping и две задачи с разным понятием similarity; укажите отдельные metrics и source splits.

<!-- /unit:0c6834c43016a923ac -->

## Результат главы и переход дальше

Ответьте на вопрос: **Как обучить пространство, где близость полезна для конкретной задачи?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

Следующие темы: [Dense retrieval, cross-encoder, SPLADE и ColBERT](<bert-retrieval.md>).

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
