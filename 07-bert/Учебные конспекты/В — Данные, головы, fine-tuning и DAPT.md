# В. Как обучить encoder под задачу и понять причину ошибки

[Оглавление](README.md). Модули **10–14**. **Основа:** S3 объясняет SFT/CPT, S1–S3 — PEFT; применение к encoder и task heads — дополнение. Общие методики подробно находятся в [данных](../../01-data/Учебные%20конспекты/README.md) и [адаптации](../../06-llm/02-adaptation/Учебные%20конспекты/README.md).

## 1. Annotation unit и split

Для classification размечается текст/документ, для NER — span/type, для QA — question/context/answers, для retrieval — query/corpus/qrels. Отсутствие judgement документа не доказывает нерелевантность. Несколько сообщений одной email thread, overlapping chunks и translation-equivalent pairs связаны; random rows split завышает перенос на новые sources.

Label ontology определяет ambiguity, unknown, nesting и правила boundaries. Пример «не жалуется на боль» может иметь medical entity «боль», но не positive symptom assertion. Обучение LR не исправит противоречивую annotation policy. Сначала небольшой guideline с пограничными cases и audit, затем model comparison.

Manifest сохраняет source IDs/version, split membership, hashes, label map и preprocessing. Synthetic supervision требует teacher provenance и independent human evaluation. Unlabeled test texts, использованные для DAPT, тоже меняют evaluation protocol: отсутствие labels не делает adaptation незаметной.

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

Последняя меньше усиливает большие residuals, но choice соответствует noise/cost assumptions. Contrastive objective рассмотрен в Е и не заменяется случайным CE по embedding coordinates.

На logits $(0,0,0)$ single-label CE равна $\log3\approx1.098612$. BCE для targets $(1,0,1)$ при тех же logits равна $\log2\approx0.693147$ при mean по classes. Одинаковая shape logits не означает одинаковые target/loss contracts.

## 3. Frozen, full и PEFT

Frozen encoder плюс head/probe учит только task mapping. Full tuning меняет весь backbone; head-only warmup сначала стабилизирует новую голову. Gradual unfreezing и layer-wise LR decay задают разные strengths обновлений по depth, например $\eta_l=\eta_{\mathrm{top}}\gamma^{L-l}$ при $0<\gamma\le1$ и явно выбранном indexing.

LoRA/adapters выбираются по actual module names/shapes. Head должен быть trainable и сохранён вместе с adapter; иначе reload восстановит случайную новую голову. Для linear classification при $D=768,C=3$ head содержит $768\cdot3+3=2307$ parameters. Rank-8 LoRA на Q и V двенадцати square $768\times768$ layers добавляет $12\cdot2\cdot8\cdot(768+768)=294\,912$, без biases/других modules. Это trainable count, а не полная memory footprint.

Frozen base weights остаются в memory. При gradient flow через frozen blocks к earlier adapters нужны соответствующие backward activations; `requires_grad=False` base parameters не делает весь forward автоматически inference-only. Optimizer states относятся к trainable parameters, activations зависят от графа и lengths.

## 4. Training loop и diagnosis

Порядок: очистить gradients по выбранному accumulation protocol → forward → masked/normalized loss → backward → при границе accumulation unscale/clipping по необходимости → optimizer step → scheduler step в объявленных units. Mixed precision, clipping и checkpointing меняют resource/stability, но не отменяют проверку objective.

При $N=1024$, per-device batch 8, world size 2 и accumulation 4 имеем 64 synchronized microsteps и 16 optimizer updates за epoch, effective полный batch 64. При $N=1000$ и equal distributed partition без dropping получается 63 microsteps; если последний неполный accumulation flush выполняется, updates 16. Реальный sampler padding/drop_last и normalization последней группы уточняются в implementation. Scheduler по updates нельзя шагать как будто каждый microbatch — один update.

Gradient accumulation складывает gradients, но при InfoNCE отдельные microbatches не видят negatives друг друга автоматически. Это не тот же negative pool, что у concatenated batch. Для equivalence нужны специальные cached/cross-device методы с корректным gradient protocol.

Tiny clean batch overfit проверяет learning loop: label-map, gradient к head/backbone, mode dropout, masks и shapes. Затем несколько seeds и полноценная validation. Exploding loss может быть LR/precision проблемой; no gradient — detach/freeze/mask; хороший train и плохой validation — overfit/shift/leakage protocol. Эти причины требуют разных вмешательств.

AdamW decay groups, warmup/decay, early stopping и dropout выбираются по validation. Универсального LR «для BERT вообще» нет. Сохраняются best checkpoint, optimizer/scheduler при resume, config и processed target units, не только epoch number.

## 5. DAPT, TAPT и forgetting

DAPT продолжает MLM на domain corpus, TAPT — на texts целевой task distribution; затем supervised tuning. [Don’t Stop Pretraining](https://aclanthology.org/2020.acl-main.740/) исследует такой перенос. Улучшение MLM loss — proxy соответствия objective, а не гарантированное улучшение classification/NER/retrieval.

Сравнивайте исходный encoder→task tuning с DAPT→task tuning при понятном total compute. Mixing general/domain data, replay и teacher regularization могут ограничивать forgetting, но требуют validation general-domain regressions. Малый noisy corpus может не оправдать continued pretraining.

Vocabulary mismatch иногда мотивирует новые tokens. Тогда нужны новые trained rows, согласованный tokenizer и проверка sequence lengths; простая замена словаря invalidates pretrained lookup. Domain-trained checkpoint, SFT/head tuning и DAPT — разные candidate interventions. Выбор определяется diagnosis: missing domain facts, плохая fragmentation или недостаточная task supervision.

## 6. Контроль и практика модулей 10–14

1. Почему head `num_labels=3` не определяет CE/BCE автоматически по смыслу задачи? **Ответ:** single-label и multi-label имеют одинаковое число logits, но разные labels/objectives.
2. Что кроме adapter нужно сохранить? **Ответ:** task head, tokenizer/config/label map и остальные изменённые modules.
3. Равен ли accumulation увеличению contrastive negatives? **Ответ:** нет, если embeddings разных microbatches не участвуют в одном objective.
4. Доказывает ли меньший MLM loss полезность DAPT? **Ответ:** нет; нужен downstream и forgetting comparison.

**Практика:** составьте manifest и guideline десяти примеров; посчитайте CE/BCE, head/LoRA counts и optimizer updates. Нарисуйте full/frozen/PEFT gradient paths. Для маленького domain corpus выберите одно вмешательство и baseline, заранее задав fair budget и independent test.
