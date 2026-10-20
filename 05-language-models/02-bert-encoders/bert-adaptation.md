# Данные, task heads, fine-tuning и DAPT

**Уровень:** основа ветки.

**Предпосылки:** [Encoder, tokenizer, attention и pooling](<bert-interface.md>), [MLM, альтернативные objectives и семейства encoder](<bert-objectives.md>)

**Используют:** [Classification, NER, spans и extractive QA](<bert-tasks.md>), [Sentence embeddings и contrastive learning](<bert-embeddings.md>), [Языки, длинные документы, домены и layout](<bert-domains.md>), [Метрики, калибровка, OOD и интерпретация](<bert-evaluation.md>)

**Учебный вопрос:** Как согласовать разметку, форму выхода и objective encoder?





## Карта вычисления и границы этапов

![Данные, task heads, fine-tuning и DAPT: входы, преобразования и выходы](<../../оформление/course/bert-adaptation.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Сравните формы и смысл каждого перехода с формулами главы. Выход преобразования и доказательство его качества — разные объекты: последняя проверка использует отдельный протокол оценки.

<!-- unit:83d2636463f1a435aa -->
<a id="u-83d2636463f1a435aa"></a>

## Annotation unit и split

<!-- /unit:83d2636463f1a435aa -->

<!-- unit:d38070ebf6830f4877 -->
<a id="u-d38070ebf6830f4877"></a>

Для classification размечается текст/документ, для NER — span/type, для QA — question/context/answers, для retrieval — query/corpus/qrels. Отсутствие judgement документа не доказывает нерелевантность. Несколько сообщений одной email thread, overlapping chunks и translation-equivalent pairs связаны; random rows split завышает перенос на новые sources.

<!-- /unit:d38070ebf6830f4877 -->

<!-- unit:7a89cd5c7a8c5fa23f -->
<a id="u-7a89cd5c7a8c5fa23f"></a>

Label ontology определяет ambiguity, unknown, nesting и правила boundaries. Пример «не жалуется на боль» может иметь medical entity «боль», но не positive symptom assertion. Обучение LR не исправит противоречивую annotation policy. Сначала небольшой guideline с пограничными cases и audit, затем model comparison.

<!-- /unit:7a89cd5c7a8c5fa23f -->

<!-- unit:a59e144ba5ae6deec7 -->
<a id="u-a59e144ba5ae6deec7"></a>

Manifest сохраняет source IDs/version, split membership, hashes, label map и preprocessing. Synthetic supervision требует teacher provenance и independent human evaluation. Unlabeled test texts, использованные для DAPT, тоже меняют evaluation protocol: отсутствие labels не делает adaptation незаметной.

<!-- /unit:a59e144ba5ae6deec7 -->

<!-- unit:a161d5577a67067012 -->
<a id="u-a161d5577a67067012"></a>

**Схема курса:** versioned texts + group/time split → tokenizer and aligned labels → encoder → task head/loss → training or DAPT → held-out task metrics and export

<!-- /unit:a161d5577a67067012 -->

<!-- unit:984d66747ac15db053 -->
<a id="u-984d66747ac15db053"></a>

<!-- source-depth-bert-В -->

<!-- /unit:984d66747ac15db053 -->

<!-- unit:b5788e55928ea0b795 -->
<a id="u-b5788e55928ea0b795"></a>

### Сначала определяем, что означает одна метка

<!-- /unit:b5788e55928ea0b795 -->

<!-- unit:1ac2e2d1f93d65f8a1 -->
<a id="u-1ac2e2d1f93d65f8a1"></a>

Для обращений в поддержку можно выбрать один основной intent: «оплата», «доставка», «возврат». Тогда каждое письмо получает один ID, а модель сравнивает три альтернативы. Можно выбрать независимые признаки: «нужен возврат», «упомянута задержка», «нужен ответ оператора». Они способны встречаться одновременно и требуют другого target representation. Текст и число выходных logits могут быть одинаковыми, но смысл обучения различается.

<!-- /unit:1ac2e2d1f93d65f8a1 -->

<!-- unit:5c60290490ebe0039b -->
<a id="u-5c60290490ebe0039b"></a>

Нужны правила для неоднозначного письма, отсутствующей разметки и новой категории. Если разметчик проверял только признак «возврат», отсутствие остальных labels не означает отрицательные ответы по ним. Нельзя заполнить все пропуски нулями и считать получившиеся ошибки только проблемой encoder. Модель оптимизирует то, что записано в targets; она не знает, какие нули означали «нет», а какие — «не проверено».

<!-- /unit:5c60290490ebe0039b -->

<!-- unit:a822fe5332928fde6e -->
<a id="u-a822fe5332928fde6e"></a>

Для NER отдельно фиксируют границы: включается ли дефис, размечается ли вложенная сущность, как обрабатывается сокращение. Если два эксперта систематически используют разные boundaries, высокий token accuracy способен сосуществовать с низким exact-span F1. Поэтому полезно сначала разобрать несколько пограничных примеров и устранить противоречия в инструкции, а уже затем сравнивать архитектуры.

<!-- /unit:a822fe5332928fde6e -->

<!-- unit:d810a639b881f1e811 -->
<a id="u-d810a639b881f1e811"></a>

### Почему строки датасета не всегда независимы

<!-- /unit:d810a639b881f1e811 -->

<!-- unit:c90c7dca40fed7604b -->
<a id="u-c90c7dca40fed7604b"></a>

Представим четыре исходных документа, каждый разрезан на пять перекрывающихся chunks. Получились 20 строк, но не 20 независимых источников. Если распределить строки случайно, соседние chunks одного документа могут оказаться в train и test. Тогда тест частично измерит узнавание уже виденного текста. Разбиение по document ID сохраняет группы целиком.

<!-- /unit:c90c7dca40fed7604b -->

<!-- unit:9ec76090ee68d61a21 -->
<a id="u-9ec76090ee68d61a21"></a>

Аналогично связаны сообщения одной переписки, версии одного шаблона, переводы одного текста и снимки одного объекта в соседних кадрах. Выбор группы зависит от желаемого переноса: на новые сообщения известных клиентов или на новых клиентов; на новые документы или на новые источники. Один универсальный random split не отвечает на все эти вопросы.

<!-- /unit:9ec76090ee68d61a21 -->

<!-- unit:307e7d085d4466e163 -->
<a id="u-307e7d085d4466e163"></a>

Стратификация по классу не отменяет группировку. Если редкий класс встречается только у одного источника, нельзя честно получить одновременно независимые train/test источники и хорошее покрытие этого класса простым перемешиванием. Это ограничение данных, которое нужно увидеть и описать. Validation выбирает решения, test оценивает уже выбранный протокол; многократная настройка по test превращает его в ещё одну validation выборку.

<!-- /unit:307e7d085d4466e163 -->

<!-- unit:495b5789ebcc718296 -->
<a id="u-495b5789ebcc718296"></a>

## Head задаёт output и objective

<!-- /unit:495b5789ebcc718296 -->

<!-- unit:3f883511ac2516c101 -->
<a id="u-3f883511ac2516c101"></a>

Backbone outputs $H\in\mathbb R^{B\times L\times D}$. Sequence head проецирует pooled vector: $Z=HW_c+b$ после pooling, shape $B\times C$. Token head применяет shared projection к каждой позиции, shape $B\times L\times C$. QA — two scalar projections per token. Embedding head — pooling/projection в chosen vector space.

<!-- /unit:3f883511ac2516c101 -->

<!-- unit:47aebc90e9378110a6 -->
<a id="u-47aebc90e9378110a6"></a>

Single-label CE при одном correct class:

<!-- /unit:47aebc90e9378110a6 -->

<!-- unit:38cf3d8f2a65961d41 -->
<a id="u-38cf3d8f2a65961d41"></a>

$$
L_{\mathrm{CE}}=-\log\frac{e^{z_y}}{\sum_c e^{z_c}}.
$$

<!-- /unit:38cf3d8f2a65961d41 -->

<!-- unit:48cea9fc8d86937815 -->
<a id="u-48cea9fc8d86937815"></a>

Для independent multi-label targets используем BCE по classes:

<!-- /unit:48cea9fc8d86937815 -->

<!-- unit:80856c395130729782 -->
<a id="u-80856c395130729782"></a>

$$
L_{\mathrm{BCE}}=\frac1C\sum_c
\left[\operatorname{softplus}(z_c)-y_cz_c\right].
$$

<!-- /unit:80856c395130729782 -->

<!-- unit:252114387ec669ae95 -->
<a id="u-252114387ec669ae95"></a>

Это numerically stable form без явного logarithm sigmoid. Missing labels требуют mask; незаполненный label не автоматически 0. Regression может использовать MSE либо Huber:

<!-- /unit:252114387ec669ae95 -->

<!-- unit:353ef102ff531253a5 -->
<a id="u-353ef102ff531253a5"></a>

$$
\ell_\delta(r)=\begin{cases}\tfrac12r^2,&|r|\le\delta,\\
\delta(|r|-\tfrac12\delta),&|r|>\delta.
\end{cases}
$$

<!-- /unit:353ef102ff531253a5 -->

<!-- unit:572c11f53c0e04639c -->
<a id="u-572c11f53c0e04639c"></a>

Последняя меньше усиливает большие residuals, но выбор соответствует предположениям о шуме и цене ошибок. Contrastive objectives рассмотрены в главе [«Sentence embeddings и contrastive learning»](<bert-embeddings.md>), а их применение к retrieval — в Е; они не заменяются произвольным CE по координатам embedding.

<!-- /unit:572c11f53c0e04639c -->

<!-- unit:7c992e05fd04133102 -->
<a id="u-7c992e05fd04133102"></a>

На logits $(0,0,0)$ single-label CE равна $\log3\approx1.098612$. BCE для targets $(1,0,1)$ при тех же logits равна $\log2\approx0.693147$ при mean по classes. Одинаковая shape logits не означает одинаковые target/loss contracts.

<!-- /unit:7c992e05fd04133102 -->

<!-- unit:0efff2ec67b4f7b24f -->
<a id="u-0efff2ec67b4f7b24f"></a>

### Одинаковые logits — разные производные

<!-- /unit:0efff2ec67b4f7b24f -->

<!-- unit:23c6a9331525e2d900 -->
<a id="u-23c6a9331525e2d900"></a>

Для трёх нулевых logits softmax равен (1/3,1/3,1/3). Если первый класс единственно правильный, CE gradient равен (−2/3,1/3,1/3): увеличение одного класса конкурирует с другими через общую нормировку. При multi-label targets (1,0,1) три sigmoid равны 1/2, а mean BCE gradient равен (−1/6,1/6,−1/6). Первый и третий классы должны расти одновременно. Softmax заставил бы их конкурировать там, где задача допускает оба.

<!-- /unit:23c6a9331525e2d900 -->

<!-- unit:90391a6713b292c41b -->
<a id="u-90391a6713b292c41b"></a>

Если часть class labels неизвестна, вводится отдельная маска наблюдаемых решений. Знаменатель mean BCE должен соответствовать выбранному правилу: например, числу известных class targets. Образец с одной размеченной категорией и образец с десятью категориями получают разные веса при среднем по всем известным решениям и при среднем по отдельно усреднённым примерам. Как и в MLM, нормировка — часть objective.

<!-- /unit:90391a6713b292c41b -->

<!-- unit:a672c3bb1a358c9dd3 -->
<a id="u-a672c3bb1a358c9dd3"></a>

В regression важно согласовать единицы: ошибка в секундах и миллисекундах отличается масштабом в тысячу раз, а квадратичная ошибка — в миллион. Для Huber с δ=1 и residual r=5 loss равна 4.5, производная по residual — 1; для половины квадрата loss равна 12.5, производная — 5. Это объясняет меньшую чувствительность Huber к крупным residuals, но не делает её автоматически правильной для любой задачи.

<!-- /unit:a672c3bb1a358c9dd3 -->

<!-- unit:c19acf37725b93b719 -->
<a id="u-c19acf37725b93b719"></a>

### Как task loss доходит до encoder

<!-- /unit:c19acf37725b93b719 -->

<!-- unit:d4fa4bfa075480ae71 -->
<a id="u-d4fa4bfa075480ae71"></a>

Пусть pooled states образуют $U\in\mathbb R^{B\times D}$, head — $Z=UW+b$, $W\in\mathbb R^{D\times C}$, а градиент logits — $G\in\mathbb R^{B\times C}$. Тогда $G_W=U^\top G$, $\nabla_b\mathcal L=\sum_{i=1}^B G_{i,:}$, $G_U=GW^\top$. Bias имеет C координат, поэтому его производная суммирует вклады всех примеров; градиент по U сохраняет отдельную строку для каждого примера. Последнее выражение — входной сигнал для pooling и Transformer.

<!-- /unit:d4fa4bfa075480ae71 -->

<!-- unit:c7faaff5babf0794fe -->
<a id="u-c7faaff5babf0794fe"></a>

При замороженном encoder обновляются только разрешённые параметры head. При full fine-tuning этот же сигнал проходит дальше через encoder и меняет его представления. Если случайно вызвать detach между encoder и головой, head продолжит обучаться, но encoder не получит gradient даже при выставленных trainable flags. Снижение train loss само по себе не доказывает, что выбранные части модели действительно обновлялись.

<!-- /unit:c7faaff5babf0794fe -->

<!-- unit:6834b8e521b89d9598 -->
<a id="u-6834b8e521b89d9598"></a>

## Frozen, full и PEFT

<!-- /unit:6834b8e521b89d9598 -->

<!-- unit:942c5b54294dea8b1a -->
<a id="u-942c5b54294dea8b1a"></a>

Frozen encoder плюс head/probe учит только task mapping. Full tuning меняет весь backbone; head-only warmup сначала стабилизирует новую голову. Gradual unfreezing и layer-wise LR decay задают разные strengths обновлений по depth, например $\eta_l=\eta_{\mathrm{top}}\gamma^{L-l}$ при $0<\gamma\le1$ и явно выбранном indexing.

<!-- /unit:942c5b54294dea8b1a -->

<!-- unit:17eff9390a0352d5b6 -->
<a id="u-17eff9390a0352d5b6"></a>

LoRA/adapters выбираются по actual module names/shapes. Head должен быть trainable и сохранён вместе с adapter; иначе reload восстановит случайную новую голову. Для linear classification при $D=768,C=3$ head содержит $768\cdot3+3=2307$ parameters. Rank-8 LoRA на Q и V двенадцати square $768\times768$ layers добавляет $12\cdot2\cdot8\cdot(768+768)=294\,912$, без biases/других modules. Это trainable count, а не полная memory footprint.

<!-- /unit:17eff9390a0352d5b6 -->

<!-- unit:74250db30d15971a62 -->
<a id="u-74250db30d15971a62"></a>

Frozen base weights остаются в memory. При gradient flow через frozen blocks к earlier adapters нужны соответствующие backward activations; `requires_grad=False` base parameters не делает весь forward автоматически inference-only. Optimizer states относятся к trainable parameters, activations зависят от графа и lengths.

<!-- /unit:74250db30d15971a62 -->

<!-- unit:6c709813f5a1ae4fb1 -->
<a id="u-6c709813f5a1ae4fb1"></a>

### Три режима обучения на одном вычислительном графе

<!-- /unit:6c709813f5a1ae4fb1 -->

<!-- unit:21d31ffd26471a262e -->
<a id="u-21d31ffd26471a262e"></a>

В frozen режиме полезно сначала проверить, насколько хороши уже готовые признаки. Можно один раз вычислить embeddings и обучать простую голову, если входной preprocessing и поведение encoder фиксированы. Если при каждом предъявлении меняются аугментации или активен dropout, кешированное представление уже не воспроизводит такой же stochastic forward. Выбор train/eval режима для frozen encoder нужно объявить отдельно от запрета обновления его весов.

<!-- /unit:21d31ffd26471a262e -->

<!-- unit:1d6be4853decce6635 -->
<a id="u-1d6be4853decce6635"></a>

При full tuning новая голова и все encoder layers могут меняться совместно. У этого режима больше гибкости, но больше optimizer states и риск подстроиться под малый набор данных. Head-only warmup и меньший learning rate для нижних слоёв — способы управлять обновлениями, а не обязательные свойства BERT. Их полезность проверяют на отложенной задаче.

<!-- /unit:1d6be4853decce6635 -->

<!-- unit:d249f47061bd9aa040 -->
<a id="u-d249f47061bd9aa040"></a>

При PEFT базовые матрицы фиксированы, но адаптер внутри блока влияет на состояния последующих блоков. Чтобы обучить ранний adapter, backward должен пройти через поздние frozen операции. Оборачивать весь encoder в `no_grad` в этом случае нельзя: это отсоединит и нужный путь к adapter. Заморозка параметра означает «не вычислять/не накапливать его параметрический градиент», а не «всем операциям с этим параметром запрещён backward по входу».

<!-- /unit:d249f47061bd9aa040 -->

<!-- unit:d4e6e1d51fea424dbb -->
<a id="u-d4e6e1d51fea424dbb"></a>

Если же заморожен весь префикс до первого trainable узла и перед ним нет нужных обучаемых параметров, его forward может не требовать сохранения графа. Граница зависит от расположения trainable modules, а не только от общего названия frozen/LoRA. Поэтому сначала рисуют путь градиента, затем выбирают способ экономии памяти.

<!-- /unit:d4e6e1d51fea424dbb -->

<!-- unit:c79f8a09bf809c2eab -->
<a id="u-c79f8a09bf809c2eab"></a>

### Что именно считаем в числе trainable parameters

<!-- /unit:c79f8a09bf809c2eab -->

<!-- unit:92d8828d5875778b6f -->
<a id="u-92d8828d5875778b6f"></a>

В приведённом примере 294 912 LoRA weights плюс 2307 head weights/bias дают 297 219 обучаемых параметров. Базовые embeddings, attention и FFN остаются частью модели и занимают память, даже если optimizer их не обновляет. Увеличение числа выходных labels меняет head count, а применение LoRA к другим linear modules меняет adapter count. Число query heads напрямую не задаёт ранг LoRA.

<!-- /unit:92d8828d5875778b6f -->

<!-- unit:5a6572d497884cde45 -->
<a id="u-5a6572d497884cde45"></a>

Сохранить только A/B недостаточно, если прикладная голова тоже обучалась: после reload новая случайная head уничтожит нужное отображение в классы. Аналогично label map «0=возврат» должна совпадать при inference. При той же форме весов перестановка label IDs может молча поменять смысл всех ответов. Поэтому сохраняют и проверяют весь inference contract, включая tokenizer и configuration.

<!-- /unit:5a6572d497884cde45 -->

<!-- unit:4b6295507c1b6eb434 -->
<a id="u-4b6295507c1b6eb434"></a>

## Training loop и diagnosis

<!-- /unit:4b6295507c1b6eb434 -->

<!-- unit:54010a8c72a4d37288 -->
<a id="u-54010a8c72a4d37288"></a>

Порядок: очистить gradients по выбранному accumulation protocol → forward → masked/normalized loss → backward → при границе accumulation unscale/clipping по необходимости → optimizer step → scheduler step в объявленных units. Mixed precision, clipping и checkpointing меняют resource/stability, но не отменяют проверку objective.

<!-- /unit:54010a8c72a4d37288 -->

<!-- unit:ec56d6fac71968019e -->
<a id="u-ec56d6fac71968019e"></a>

При $N=1024$, per-device batch 8, world size 2 и accumulation 4 имеем 64 synchronized microsteps и 16 optimizer updates за epoch, effective полный batch 64. При $N=1000$ и equal distributed partition без dropping получается 63 microsteps; если последний неполный accumulation flush выполняется, updates 16. Реальный sampler padding/drop_last и normalization последней группы уточняются в implementation. Scheduler по updates нельзя шагать как будто каждый microbatch — один update.

<!-- /unit:ec56d6fac71968019e -->

<!-- unit:391ff89a51d15f31c3 -->
<a id="u-391ff89a51d15f31c3"></a>

Gradient accumulation складывает gradients, но при InfoNCE отдельные microbatches не видят negatives друг друга автоматически. Это не тот же negative pool, что у concatenated batch. Для equivalence нужны специальные cached/cross-device методы с корректным gradient protocol.

<!-- /unit:391ff89a51d15f31c3 -->

<!-- unit:b1fdaa8020c210a509 -->
<a id="u-b1fdaa8020c210a509"></a>

Tiny clean batch overfit проверяет learning loop: label-map, gradient к head/backbone, mode dropout, masks и shapes. Затем несколько seeds и полноценная validation. Exploding loss может быть LR/precision проблемой; no gradient — detach/freeze/mask; хороший train и плохой validation — overfit/shift/leakage protocol. Эти причины требуют разных вмешательств.

<!-- /unit:b1fdaa8020c210a509 -->

<!-- unit:ede8990a9b1b2e4f60 -->
<a id="u-ede8990a9b1b2e4f60"></a>

AdamW decay groups, warmup/decay, early stopping и dropout выбираются по validation. Универсального LR «для BERT вообще» нет. Сохраняются best checkpoint, optimizer/scheduler при resume, config и processed target units, не только epoch number.

<!-- /unit:ede8990a9b1b2e4f60 -->

<!-- unit:2ff99a07f17bbff4f5 -->
<a id="u-2ff99a07f17bbff4f5"></a>

### Последняя неполная accumulation group требует своего знаменателя

<!-- /unit:2ff99a07f17bbff4f5 -->

<!-- unit:4fb08ea49eed7969e9 -->
<a id="u-4fb08ea49eed7969e9"></a>

Для N=1000, двух устройств, per-device batch 8 и accumulation 4 на каждом устройстве по 500 примеров: 62 полных microbatches по 8 и последний из 4. Всего 63 synchronized microsteps. Первые 60 образуют 15 полных accumulation groups, каждая охватывает глобально 64 примера. Последняя группа состоит из трёх microsteps: глобальные размеры 16,16,8, всего 40. Получаем $15\cdot64+40=1000$ примеров и 16 optimizer updates.

<!-- /unit:4fb08ea49eed7969e9 -->

<!-- unit:dee49f0ecdab793f78 -->
<a id="u-dee49f0ecdab793f78"></a>

Если нужны одинаковые веса всех examples внутри последней группы, её loss равна сумме 40 individual losses, делённой на 40. Три global microbatch means должны получить веса 0.4,0.4,0.2, а не автоматически по 1/3 или по 1/4. При token-level задачах аналогично учитывают число допустимых targets, а не только примеров. Конкретное распределённое суммирование должно согласовать усреднение gradients между устройствами.

<!-- /unit:dee49f0ecdab793f78 -->

<!-- unit:7bef2f4b07149db8db -->
<a id="u-7bef2f4b07149db8db"></a>

Accumulation может воспроизвести градиент суммы независимых losses при фиксированных параметрах и согласованных вычислениях. Но один большой batch не всегда тождествен нескольким microbatches: могут различаться dropout samples, batch-dependent операции и состав contrastive negatives. У InfoNCE embedding другого microbatch не становится отрицательным примером автоматически от того, что optimizer step отложен.

<!-- /unit:7bef2f4b07149db8db -->

<!-- unit:1888713a0a6496433f -->
<a id="u-1888713a0a6496433f"></a>

### Что различает optimizer step и scheduler step

<!-- /unit:1888713a0a6496433f -->

<!-- unit:ea773c8102ce444f68 -->
<a id="u-ea773c8102ce444f68"></a>

Backward накапливает gradients; optimizer step изменяет параметры и свои состояния; scheduler меняет learning rate по объявленному счётчику. Если scheduler рассчитан на 16 updates за эпоху, вызов на каждом из 63 microsteps ускорит его почти в четыре раза относительно ожидаемого обучения. Номер epoch не заменяет число обработанных examples, targets и updates.

<!-- /unit:ea773c8102ce444f68 -->

<!-- unit:67632f77a87afeaa93 -->
<a id="u-67632f77a87afeaa93"></a>

Перед clipping при scaled mixed-precision training нужно работать с корректным масштабом градиента, а обновление optimizer не должно незаметно выполняться несколько раз на одной группе. Gradient checkpointing экономит сохранённые активации ценой повторного forward в backward; оно не уменьшает число весов и не заменяет gradient accumulation. Эти механизмы решают разные ресурсные задачи.

<!-- /unit:67632f77a87afeaa93 -->

<!-- unit:5867fc2defed209dba -->
<a id="u-5867fc2defed209dba"></a>

### Диагностика по наблюдаемым симптомам

<!-- /unit:5867fc2defed209dba -->

<!-- unit:51440ba7253ec05db6 -->
<a id="u-51440ba7253ec05db6"></a>

Если маленький чистый batch не удаётся переобучить, сначала проверяют labels, loss, masks, выбранные trainable modules и наличие gradients. Если training loss снижается, а validation ухудшается, это уже другой симптом: возможны overfitting, сдвиг источников или неверная постановка evaluation. Поднимать LR в обоих случаях одинаково — не диагностика.

<!-- /unit:51440ba7253ec05db6 -->

<!-- unit:d24fbc5c3cdbd8ace4 -->
<a id="u-d24fbc5c3cdbd8ace4"></a>

Если accuracy высока, но редкий класс почти всегда пропускается, надо смотреть per-class recall и confusion matrix, а не объявлять обучение успешным по одному среднему. Если после reload качество исчезло, проверяют head, tokenizer, label map, eval mode и preprocessing, прежде чем повторять тренировку. Если train/validation почти идеальны с первых шагов, ищут leakage и лёгкие артефакты разметки: это может быть хорошая модель, но необычный результат требует объяснения данными.

<!-- /unit:d24fbc5c3cdbd8ace4 -->

<!-- unit:d97a0bdae404f2442c -->
<a id="u-d97a0bdae404f2442c"></a>

Все числа выше — проверяемая учебная арифметика, не отчёт о ваших training runs. Для реального запуска сохраняют несколько seeds или оценивают вариативность, фиксируют выбранный checkpoint и сравнивают по одному заранее объявленному протоколу.

<!-- /unit:d97a0bdae404f2442c -->

<!-- unit:aa4f577cc00bf5080d -->
<a id="u-aa4f577cc00bf5080d"></a>

## DAPT, TAPT и forgetting

<!-- /unit:aa4f577cc00bf5080d -->

<!-- unit:7d9c6b646bf293edef -->
<a id="u-7d9c6b646bf293edef"></a>

DAPT продолжает MLM на domain corpus, TAPT — на texts целевой task distribution; затем supervised tuning. [Don’t Stop Pretraining](https://aclanthology.org/2020.acl-main.740/) исследует такой перенос. Улучшение MLM loss — proxy соответствия objective, а не гарантированное улучшение classification/NER/retrieval.

<!-- /unit:7d9c6b646bf293edef -->

<!-- unit:4183de570c0c953ba9 -->
<a id="u-4183de570c0c953ba9"></a>

Сравнивайте исходный encoder→task tuning с DAPT→task tuning при понятном total compute. Mixing general/domain data, replay и teacher regularization могут ограничивать forgetting, но требуют validation general-domain regressions. Малый noisy corpus может не оправдать continued pretraining.

<!-- /unit:4183de570c0c953ba9 -->

<!-- unit:be706ec68921680621 -->
<a id="u-be706ec68921680621"></a>

Vocabulary mismatch иногда мотивирует новые tokens. Тогда нужны новые trained rows, согласованный tokenizer и проверка sequence lengths; простая замена словаря invalidates pretrained lookup. Domain-trained checkpoint, SFT/head tuning и DAPT — разные candidate interventions. Выбор определяется diagnosis: missing domain facts, плохая fragmentation или недостаточная task supervision.

<!-- /unit:be706ec68921680621 -->

<!-- unit:335b143b8159a7b9bc -->
<a id="u-335b143b8159a7b9bc"></a>

### Продолжение MLM меняет представление, но не выдаёт прикладные labels

<!-- /unit:335b143b8159a7b9bc -->

<!-- unit:057dcb6d118381566c -->
<a id="u-057dcb6d118381566c"></a>

DAPT на корпусе обращений поддержки может адаптировать encoder к названиям продуктов, сокращениям и стилю сообщений. TAPT использует тексты, близкие распределению конкретной задачи. Обе процедуры могут идти без ручных task labels, потому что MLM targets строятся автоматически. Затем задача классификации всё равно требует своей головы и supervision. Исследование подхода — [Don’t Stop Pretraining](https://aclanthology.org/2020.acl-main.740/).

<!-- /unit:057dcb6d118381566c -->

<!-- unit:90486222b1353d3b27 -->
<a id="u-90486222b1353d3b27"></a>

Падение MLM loss показывает, что модель лучше решает выбранную задачу восстановления на выбранных текстах. Оно не гарантирует улучшения intent classification: модель может выучить частые шаблоны, забыть полезные общие зависимости или получить плохую прикладную голову. Поэтому оценивают итоговую цепочку encoder → task fine-tuning → held-out task metrics, а также нужные general-domain проверки.

<!-- /unit:90486222b1353d3b27 -->

<!-- unit:434c18ad58a067f53c -->
<a id="u-434c18ad58a067f53c"></a>

Для понимания причины полезны отдельные сравнения: исходный encoder с task tuning; общий continued pretraining с тем же дополнительным бюджетом и затем task tuning; domain pretraining и затем тот же task tuning. Второе помогает отделить эффект дополнительного compute от специфики домена. Нельзя одновременно менять corpus, tokenizer, длину и supervised budget и приписывать весь результат одному DAPT.

<!-- /unit:434c18ad58a067f53c -->

<!-- unit:81993bec62bdf1d783 -->
<a id="u-81993bec62bdf1d783"></a>

Если test texts использованы даже без labels, модель уже видела их распределение или конкретные строки. Для некоторых протоколов transductive adaptation допустима, но должна называться явно. Это другая постановка, чем оценка на полностью новых текстах. Отсутствие ручных labels не доказывает отсутствие влияния test на обучение.

<!-- /unit:81993bec62bdf1d783 -->

<!-- unit:2736f6071cec67666b -->
<a id="u-2736f6071cec67666b"></a>

### Новые tokens требуют обучения новых строк, а не только редактирования словаря

<!-- /unit:2736f6071cec67666b -->

<!-- unit:eb5fad71581ae471c6 -->
<a id="u-eb5fad71581ae471c6"></a>

Пусть частый код товара `XQ-17` раньше разбивался на несколько subwords. Добавление цельного token может сократить последовательность, но новый ID должен получить embedding. Случайная или усреднённая инициализация старых subwords — только начальное приближение: новый вектор ещё нужно обучить. Изменяется и токенизация окружения, поэтому проверяют реальные lengths и task quality.

<!-- /unit:eb5fad71581ae471c6 -->

<!-- unit:acaab00fe04867444c -->
<a id="u-acaab00fe04867444c"></a>

Если добавить 100 токенов при D=768, входная таблица увеличится на 76 800 весов. При tied MLM output projection те же строки используются на выходе; у отдельной выходной матрицы добавится свой набор параметров, а vocabulary bias может вырасти на 100 чисел. Нужно согласовать новый размер с config, head, optimizer states и сохранением checkpoint. Старые IDs нельзя произвольно перенумеровывать: pretrained lookup привязан к их строкам.

<!-- /unit:acaab00fe04867444c -->

<!-- unit:58e1ac01ae9ad27ce9 -->
<a id="u-58e1ac01ae9ad27ce9"></a>

Vocabulary expansion — лишь один возможный ответ на сильную fragmentation. Иногда достаточно исходных subwords и domain adaptation, иногда лучше другой уже обученный tokenizer/encoder. Сначала измеряют проблему и сравнивают кандидатов. Уменьшение tokens-per-word само по себе не доказывает улучшения качества, как и увеличение словаря не создаёт знания о новых названиях автоматически.

<!-- /unit:58e1ac01ae9ad27ce9 -->

<!-- unit:7bfa56e733791555ea -->
<a id="u-7bfa56e733791555ea"></a>

## Контроль и практика

<!-- /unit:7bfa56e733791555ea -->

<!-- unit:810d6745cb50f761a1 -->
<a id="u-810d6745cb50f761a1"></a>

1. Почему head `num_labels=3` не определяет CE/BCE автоматически по смыслу задачи? **Ответ:** single-label и multi-label имеют одинаковое число logits, но разные labels/objectives.
2. Что кроме adapter нужно сохранить? **Ответ:** task head, tokenizer/config/label map и остальные изменённые modules.
3. Равен ли accumulation увеличению contrastive negatives? **Ответ:** нет, если embeddings разных microbatches не участвуют в одном objective.
4. Доказывает ли меньший MLM loss полезность DAPT? **Ответ:** нет; нужен downstream и forgetting comparison.

<!-- /unit:810d6745cb50f761a1 -->

<!-- unit:235cd802628d2c5d33 -->
<a id="u-235cd802628d2c5d33"></a>

5. Почему нельзя обернуть весь encoder с trainable LoRA в no_grad? **Ответ:** будет отсоединён нужный путь backward к адаптерам, хотя базовые матрицы и так заморожены.
6. Как усреднить последнюю группу из 16,16,8 примеров? **Ответ:** по 40 индивидуальным losses; веса microbatch means — 0.4,0.4,0.2.
7. Достаточно ли меньшего tokens-per-word после расширения словаря? **Ответ:** нет. Новые строки embeddings должны обучиться, а польза проверяется на целевой задаче и согласованном preprocessing.
8. Доказывает ли отсутствие labels, что DAPT на test texts не влияет на оценку? **Ответ:** нет. Модель видела сами тексты или их распределение; transductive протокол должен быть объявлен отдельно.

<!-- /unit:235cd802628d2c5d33 -->

<!-- unit:93cc9fc905633136a9 -->
<a id="u-93cc9fc905633136a9"></a>

**Практика:** составьте manifest и guideline десяти примеров; посчитайте CE/BCE, head/LoRA counts и optimizer updates. Нарисуйте full/frozen/PEFT gradient paths. Для маленького domain corpus выберите одно вмешательство и baseline, заранее задав fair budget и independent test.

<!-- /unit:93cc9fc905633136a9 -->

## Результат главы и переход дальше

Ответьте на вопрос: **Как согласовать разметку, форму выхода и objective encoder?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

Следующие темы: [Classification, NER, spans и extractive QA](<bert-tasks.md>), [Sentence embeddings и contrastive learning](<bert-embeddings.md>), [Языки, длинные документы, домены и layout](<bert-domains.md>), [Метрики, калибровка, OOD и интерпретация](<bert-evaluation.md>).

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
