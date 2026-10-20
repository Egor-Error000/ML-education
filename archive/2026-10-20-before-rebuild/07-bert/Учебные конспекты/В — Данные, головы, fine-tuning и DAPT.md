# В. Как обучить encoder под задачу и понять причину ошибки

[Оглавление](README.md). Модули **10–14**. **Основа:** S3 объясняет SFT/CPT, S1–S3 — PEFT; применение к encoder и task heads — дополнение. Общие методики подробно находятся в [данных](../../01-data/Учебные%20конспекты/README.md) и [адаптации](../../06-llm/02-adaptation/Учебные%20конспекты/README.md).

**Уровень:** Общее ядро для supervised route. **Предпосылки:** B1–B2; D2 for split. **Используют:** B4/B8/B9.

## 1. Annotation unit и split

Для classification размечается текст/документ, для NER — span/type, для QA — question/context/answers, для retrieval — query/corpus/qrels. Отсутствие judgement документа не доказывает нерелевантность. Несколько сообщений одной email thread, overlapping chunks и translation-equivalent pairs связаны; random rows split завышает перенос на новые sources.

Label ontology определяет ambiguity, unknown, nesting и правила boundaries. Пример «не жалуется на боль» может иметь medical entity «боль», но не positive symptom assertion. Обучение LR не исправит противоречивую annotation policy. Сначала небольшой guideline с пограничными cases и audit, затем model comparison.

Manifest сохраняет source IDs/version, split membership, hashes, label map и preprocessing. Synthetic supervision требует teacher provenance и independent human evaluation. Unlabeled test texts, использованные для DAPT, тоже меняют evaluation protocol: отсутствие labels не делает adaptation незаметной.




**Схема курса:** versioned texts + group/time split → tokenizer and aligned labels → encoder → task head/loss → training or DAPT → held-out task metrics and export

<!-- source-depth-bert-В -->
### 1.1. Сначала определяем, что означает одна метка

Для обращений в поддержку можно выбрать один основной intent: «оплата», «доставка», «возврат». Тогда каждое письмо получает один ID, а модель сравнивает три альтернативы. Можно выбрать независимые признаки: «нужен возврат», «упомянута задержка», «нужен ответ оператора». Они способны встречаться одновременно и требуют другого target representation. Текст и число выходных logits могут быть одинаковыми, но смысл обучения различается.

Нужны правила для неоднозначного письма, отсутствующей разметки и новой категории. Если разметчик проверял только признак «возврат», отсутствие остальных labels не означает отрицательные ответы по ним. Нельзя заполнить все пропуски нулями и считать получившиеся ошибки только проблемой encoder. Модель оптимизирует то, что записано в targets; она не знает, какие нули означали «нет», а какие — «не проверено».

Для NER отдельно фиксируют границы: включается ли дефис, размечается ли вложенная сущность, как обрабатывается сокращение. Если два эксперта систематически используют разные boundaries, высокий token accuracy способен сосуществовать с низким exact-span F1. Поэтому полезно сначала разобрать несколько пограничных примеров и устранить противоречия в инструкции, а уже затем сравнивать архитектуры.

### 1.2. Почему строки датасета не всегда независимы

Представим четыре исходных документа, каждый разрезан на пять перекрывающихся chunks. Получились 20 строк, но не 20 независимых источников. Если распределить строки случайно, соседние chunks одного документа могут оказаться в train и test. Тогда тест частично измерит узнавание уже виденного текста. Разбиение по document ID сохраняет группы целиком.

Аналогично связаны сообщения одной переписки, версии одного шаблона, переводы одного текста и снимки одного объекта в соседних кадрах. Выбор группы зависит от желаемого переноса: на новые сообщения известных клиентов или на новых клиентов; на новые документы или на новые источники. Один универсальный random split не отвечает на все эти вопросы.

Стратификация по классу не отменяет группировку. Если редкий класс встречается только у одного источника, нельзя честно получить одновременно независимые train/test источники и хорошее покрытие этого класса простым перемешиванием. Это ограничение данных, которое нужно увидеть и описать. Validation выбирает решения, test оценивает уже выбранный протокол; многократная настройка по test превращает его в ещё одну validation выборку.


## 2. Head задаёт output и objective

Backbone outputs $H\in\mathbb R^{B\times L\times D}$. Sequence head проецирует pooled vector: $Z=HW_c+b$ после pooling, shape $B\times C$. Token head применяет shared projection к каждой позиции, shape $B\times L\times C$. QA — two scalar projections per token. Embedding head — pooling/projection в chosen vector space.

Single-label CE при одном correct class:

$$
L_{\mathrm{CE}}=-\log\frac{e^{z_y}}{\sum_c e^{z_c}}.
$$

Для independent multi-label targets используем BCE по classes:

$$
L_{\mathrm{BCE}}=\frac1C\sum_c
\left[\operatorname{softplus}(z_c)-y_cz_c\right].
$$

Это numerically stable form без явного logarithm sigmoid. Missing labels требуют mask; незаполненный label не автоматически 0. Regression может использовать MSE либо Huber:

$$
\ell_\delta(r)=\begin{cases}\tfrac12r^2,&|r|\le\delta,\\
\delta(|r|-\tfrac12\delta),&|r|>\delta.
\end{cases}
$$

Последняя меньше усиливает большие residuals, но выбор соответствует предположениям о шуме и цене ошибок. Contrastive objectives рассмотрены в главе Д, а их применение к retrieval — в Е; они не заменяются произвольным CE по координатам embedding.

На logits $(0,0,0)$ single-label CE равна $\log3\approx1.098612$. BCE для targets $(1,0,1)$ при тех же logits равна $\log2\approx0.693147$ при mean по classes. Одинаковая shape logits не означает одинаковые target/loss contracts.


### 2.1. Одинаковые logits — разные производные

Для трёх нулевых logits softmax равен (1/3,1/3,1/3). Если первый класс единственно правильный, CE gradient равен (−2/3,1/3,1/3): увеличение одного класса конкурирует с другими через общую нормировку. При multi-label targets (1,0,1) три sigmoid равны 1/2, а mean BCE gradient равен (−1/6,1/6,−1/6). Первый и третий классы должны расти одновременно. Softmax заставил бы их конкурировать там, где задача допускает оба.

Если часть class labels неизвестна, вводится отдельная маска наблюдаемых решений. Знаменатель mean BCE должен соответствовать выбранному правилу: например, числу известных class targets. Образец с одной размеченной категорией и образец с десятью категориями получают разные веса при среднем по всем известным решениям и при среднем по отдельно усреднённым примерам. Как и в MLM, нормировка — часть objective.

В regression важно согласовать единицы: ошибка в секундах и миллисекундах отличается масштабом в тысячу раз, а квадратичная ошибка — в миллион. Для Huber с δ=1 и residual r=5 loss равна 4.5, производная по residual — 1; для половины квадрата loss равна 12.5, производная — 5. Это объясняет меньшую чувствительность Huber к крупным residuals, но не делает её автоматически правильной для любой задачи.

### 2.2. Как task loss доходит до encoder

Пусть pooled states образуют $U\in\mathbb R^{B\times D}$, head — $Z=UW+b$, $W\in\mathbb R^{D\times C}$, а градиент logits — $G\in\mathbb R^{B\times C}$. Тогда $G_W=U^\top G$, $\nabla_b\mathcal L=\sum_{i=1}^B G_{i,:}$, $G_U=GW^\top$. Bias имеет C координат, поэтому его производная суммирует вклады всех примеров; градиент по U сохраняет отдельную строку для каждого примера. Последнее выражение — входной сигнал для pooling и Transformer.

При замороженном encoder обновляются только разрешённые параметры head. При full fine-tuning этот же сигнал проходит дальше через encoder и меняет его представления. Если случайно вызвать detach между encoder и головой, head продолжит обучаться, но encoder не получит gradient даже при выставленных trainable flags. Снижение train loss само по себе не доказывает, что выбранные части модели действительно обновлялись.


## 3. Frozen, full и PEFT

Frozen encoder плюс head/probe учит только task mapping. Full tuning меняет весь backbone; head-only warmup сначала стабилизирует новую голову. Gradual unfreezing и layer-wise LR decay задают разные strengths обновлений по depth, например $\eta_l=\eta_{\mathrm{top}}\gamma^{L-l}$ при $0<\gamma\le1$ и явно выбранном indexing.

LoRA/adapters выбираются по actual module names/shapes. Head должен быть trainable и сохранён вместе с adapter; иначе reload восстановит случайную новую голову. Для linear classification при $D=768,C=3$ head содержит $768\cdot3+3=2307$ parameters. Rank-8 LoRA на Q и V двенадцати square $768\times768$ layers добавляет $12\cdot2\cdot8\cdot(768+768)=294\,912$, без biases/других modules. Это trainable count, а не полная memory footprint.

Frozen base weights остаются в memory. При gradient flow через frozen blocks к earlier adapters нужны соответствующие backward activations; `requires_grad=False` base parameters не делает весь forward автоматически inference-only. Optimizer states относятся к trainable parameters, activations зависят от графа и lengths.


### 3.1. Три режима обучения на одном вычислительном графе

В frozen режиме полезно сначала проверить, насколько хороши уже готовые признаки. Можно один раз вычислить embeddings и обучать простую голову, если входной preprocessing и поведение encoder фиксированы. Если при каждом предъявлении меняются аугментации или активен dropout, кешированное представление уже не воспроизводит такой же stochastic forward. Выбор train/eval режима для frozen encoder нужно объявить отдельно от запрета обновления его весов.

При full tuning новая голова и все encoder layers могут меняться совместно. У этого режима больше гибкости, но больше optimizer states и риск подстроиться под малый набор данных. Head-only warmup и меньший learning rate для нижних слоёв — способы управлять обновлениями, а не обязательные свойства BERT. Их полезность проверяют на отложенной задаче.

При PEFT базовые матрицы фиксированы, но адаптер внутри блока влияет на состояния последующих блоков. Чтобы обучить ранний adapter, backward должен пройти через поздние frozen операции. Оборачивать весь encoder в `no_grad` в этом случае нельзя: это отсоединит и нужный путь к adapter. Заморозка параметра означает «не вычислять/не накапливать его параметрический градиент», а не «всем операциям с этим параметром запрещён backward по входу».

Если же заморожен весь префикс до первого trainable узла и перед ним нет нужных обучаемых параметров, его forward может не требовать сохранения графа. Граница зависит от расположения trainable modules, а не только от общего названия frozen/LoRA. Поэтому сначала рисуют путь градиента, затем выбирают способ экономии памяти.

### 3.2. Что именно считаем в числе trainable parameters

В приведённом примере 294 912 LoRA weights плюс 2307 head weights/bias дают 297 219 обучаемых параметров. Базовые embeddings, attention и FFN остаются частью модели и занимают память, даже если optimizer их не обновляет. Увеличение числа выходных labels меняет head count, а применение LoRA к другим linear modules меняет adapter count. Число query heads напрямую не задаёт ранг LoRA.

Сохранить только A/B недостаточно, если прикладная голова тоже обучалась: после reload новая случайная head уничтожит нужное отображение в классы. Аналогично label map «0=возврат» должна совпадать при inference. При той же форме весов перестановка label IDs может молча поменять смысл всех ответов. Поэтому сохраняют и проверяют весь inference contract, включая tokenizer и configuration.


## 4. Training loop и diagnosis

Порядок: очистить gradients по выбранному accumulation protocol → forward → masked/normalized loss → backward → при границе accumulation unscale/clipping по необходимости → optimizer step → scheduler step в объявленных units. Mixed precision, clipping и checkpointing меняют resource/stability, но не отменяют проверку objective.

При $N=1024$, per-device batch 8, world size 2 и accumulation 4 имеем 64 synchronized microsteps и 16 optimizer updates за epoch, effective полный batch 64. При $N=1000$ и equal distributed partition без dropping получается 63 microsteps; если последний неполный accumulation flush выполняется, updates 16. Реальный sampler padding/drop_last и normalization последней группы уточняются в implementation. Scheduler по updates нельзя шагать как будто каждый microbatch — один update.

Gradient accumulation складывает gradients, но при InfoNCE отдельные microbatches не видят negatives друг друга автоматически. Это не тот же negative pool, что у concatenated batch. Для equivalence нужны специальные cached/cross-device методы с корректным gradient protocol.

Tiny clean batch overfit проверяет learning loop: label-map, gradient к head/backbone, mode dropout, masks и shapes. Затем несколько seeds и полноценная validation. Exploding loss может быть LR/precision проблемой; no gradient — detach/freeze/mask; хороший train и плохой validation — overfit/shift/leakage protocol. Эти причины требуют разных вмешательств.

AdamW decay groups, warmup/decay, early stopping и dropout выбираются по validation. Универсального LR «для BERT вообще» нет. Сохраняются best checkpoint, optimizer/scheduler при resume, config и processed target units, не только epoch number.


### 4.1. Последняя неполная accumulation group требует своего знаменателя

Для N=1000, двух устройств, per-device batch 8 и accumulation 4 на каждом устройстве по 500 примеров: 62 полных microbatches по 8 и последний из 4. Всего 63 synchronized microsteps. Первые 60 образуют 15 полных accumulation groups, каждая охватывает глобально 64 примера. Последняя группа состоит из трёх microsteps: глобальные размеры 16,16,8, всего 40. Получаем $15\cdot64+40=1000$ примеров и 16 optimizer updates.

Если нужны одинаковые веса всех examples внутри последней группы, её loss равна сумме 40 individual losses, делённой на 40. Три global microbatch means должны получить веса 0.4,0.4,0.2, а не автоматически по 1/3 или по 1/4. При token-level задачах аналогично учитывают число допустимых targets, а не только примеров. Конкретное распределённое суммирование должно согласовать усреднение gradients между устройствами.

Accumulation может воспроизвести градиент суммы независимых losses при фиксированных параметрах и согласованных вычислениях. Но один большой batch не всегда тождествен нескольким microbatches: могут различаться dropout samples, batch-dependent операции и состав contrastive negatives. У InfoNCE embedding другого microbatch не становится отрицательным примером автоматически от того, что optimizer step отложен.

### 4.2. Что различает optimizer step и scheduler step

Backward накапливает gradients; optimizer step изменяет параметры и свои состояния; scheduler меняет learning rate по объявленному счётчику. Если scheduler рассчитан на 16 updates за эпоху, вызов на каждом из 63 microsteps ускорит его почти в четыре раза относительно ожидаемого обучения. Номер epoch не заменяет число обработанных examples, targets и updates.

Перед clipping при scaled mixed-precision training нужно работать с корректным масштабом градиента, а обновление optimizer не должно незаметно выполняться несколько раз на одной группе. Gradient checkpointing экономит сохранённые активации ценой повторного forward в backward; оно не уменьшает число весов и не заменяет gradient accumulation. Эти механизмы решают разные ресурсные задачи.

### 4.3. Диагностика по наблюдаемым симптомам

Если маленький чистый batch не удаётся переобучить, сначала проверяют labels, loss, masks, выбранные trainable modules и наличие gradients. Если training loss снижается, а validation ухудшается, это уже другой симптом: возможны overfitting, сдвиг источников или неверная постановка evaluation. Поднимать LR в обоих случаях одинаково — не диагностика.

Если accuracy высока, но редкий класс почти всегда пропускается, надо смотреть per-class recall и confusion matrix, а не объявлять обучение успешным по одному среднему. Если после reload качество исчезло, проверяют head, tokenizer, label map, eval mode и preprocessing, прежде чем повторять тренировку. Если train/validation почти идеальны с первых шагов, ищут leakage и лёгкие артефакты разметки: это может быть хорошая модель, но необычный результат требует объяснения данными.

Все числа выше — проверяемая учебная арифметика, не отчёт о ваших training runs. Для реального запуска сохраняют несколько seeds или оценивают вариативность, фиксируют выбранный checkpoint и сравнивают по одному заранее объявленному протоколу.


## 5. DAPT, TAPT и forgetting

DAPT продолжает MLM на domain corpus, TAPT — на texts целевой task distribution; затем supervised tuning. [Don’t Stop Pretraining](https://aclanthology.org/2020.acl-main.740/) исследует такой перенос. Улучшение MLM loss — proxy соответствия objective, а не гарантированное улучшение classification/NER/retrieval.

Сравнивайте исходный encoder→task tuning с DAPT→task tuning при понятном total compute. Mixing general/domain data, replay и teacher regularization могут ограничивать forgetting, но требуют validation general-domain regressions. Малый noisy corpus может не оправдать continued pretraining.

Vocabulary mismatch иногда мотивирует новые tokens. Тогда нужны новые trained rows, согласованный tokenizer и проверка sequence lengths; простая замена словаря invalidates pretrained lookup. Domain-trained checkpoint, SFT/head tuning и DAPT — разные candidate interventions. Выбор определяется diagnosis: missing domain facts, плохая fragmentation или недостаточная task supervision.


### 5.1. Продолжение MLM меняет представление, но не выдаёт прикладные labels

DAPT на корпусе обращений поддержки может адаптировать encoder к названиям продуктов, сокращениям и стилю сообщений. TAPT использует тексты, близкие распределению конкретной задачи. Обе процедуры могут идти без ручных task labels, потому что MLM targets строятся автоматически. Затем задача классификации всё равно требует своей головы и supervision. Исследование подхода — [Don’t Stop Pretraining](https://aclanthology.org/2020.acl-main.740/).

Падение MLM loss показывает, что модель лучше решает выбранную задачу восстановления на выбранных текстах. Оно не гарантирует улучшения intent classification: модель может выучить частые шаблоны, забыть полезные общие зависимости или получить плохую прикладную голову. Поэтому оценивают итоговую цепочку encoder → task fine-tuning → held-out task metrics, а также нужные general-domain проверки.

Для понимания причины полезны отдельные сравнения: исходный encoder с task tuning; общий continued pretraining с тем же дополнительным бюджетом и затем task tuning; domain pretraining и затем тот же task tuning. Второе помогает отделить эффект дополнительного compute от специфики домена. Нельзя одновременно менять corpus, tokenizer, длину и supervised budget и приписывать весь результат одному DAPT.

Если test texts использованы даже без labels, модель уже видела их распределение или конкретные строки. Для некоторых протоколов transductive adaptation допустима, но должна называться явно. Это другая постановка, чем оценка на полностью новых текстах. Отсутствие ручных labels не доказывает отсутствие влияния test на обучение.

### 5.2. Новые tokens требуют обучения новых строк, а не только редактирования словаря

Пусть частый код товара `XQ-17` раньше разбивался на несколько subwords. Добавление цельного token может сократить последовательность, но новый ID должен получить embedding. Случайная или усреднённая инициализация старых subwords — только начальное приближение: новый вектор ещё нужно обучить. Изменяется и токенизация окружения, поэтому проверяют реальные lengths и task quality.

Если добавить 100 токенов при D=768, входная таблица увеличится на 76 800 весов. При tied MLM output projection те же строки используются на выходе; у отдельной выходной матрицы добавится свой набор параметров, а vocabulary bias может вырасти на 100 чисел. Нужно согласовать новый размер с config, head, optimizer states и сохранением checkpoint. Старые IDs нельзя произвольно перенумеровывать: pretrained lookup привязан к их строкам.

Vocabulary expansion — лишь один возможный ответ на сильную fragmentation. Иногда достаточно исходных subwords и domain adaptation, иногда лучше другой уже обученный tokenizer/encoder. Сначала измеряют проблему и сравнивают кандидатов. Уменьшение tokens-per-word само по себе не доказывает улучшения качества, как и увеличение словаря не создаёт знания о новых названиях автоматически.


## 6. Контроль и практика модулей 10–14

1. Почему head `num_labels=3` не определяет CE/BCE автоматически по смыслу задачи? **Ответ:** single-label и multi-label имеют одинаковое число logits, но разные labels/objectives.
2. Что кроме adapter нужно сохранить? **Ответ:** task head, tokenizer/config/label map и остальные изменённые modules.
3. Равен ли accumulation увеличению contrastive negatives? **Ответ:** нет, если embeddings разных microbatches не участвуют в одном objective.
4. Доказывает ли меньший MLM loss полезность DAPT? **Ответ:** нет; нужен downstream и forgetting comparison.

5. Почему нельзя обернуть весь encoder с trainable LoRA в no_grad? **Ответ:** будет отсоединён нужный путь backward к адаптерам, хотя базовые матрицы и так заморожены.
6. Как усреднить последнюю группу из 16,16,8 примеров? **Ответ:** по 40 индивидуальным losses; веса microbatch means — 0.4,0.4,0.2.
7. Достаточно ли меньшего tokens-per-word после расширения словаря? **Ответ:** нет. Новые строки embeddings должны обучиться, а польза проверяется на целевой задаче и согласованном preprocessing.
8. Доказывает ли отсутствие labels, что DAPT на test texts не влияет на оценку? **Ответ:** нет. Модель видела сами тексты или их распределение; transductive протокол должен быть объявлен отдельно.

**Практика:** составьте manifest и guideline десяти примеров; посчитайте CE/BCE, head/LoRA counts и optimizer updates. Нарисуйте full/frozen/PEFT gradient paths. Для маленького domain corpus выберите одно вмешательство и baseline, заранее задав fair budget и independent test.
