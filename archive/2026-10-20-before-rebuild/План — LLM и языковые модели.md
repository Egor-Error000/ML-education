# План модуля: LLM, языковые модели и BERT

**Срез источников: 20 октября 2026.** План подготовки, не учебный текст. Универсальная форма глав и правила покрытия заданы в [плане подготовки курса](<План подготовки курса.md>) и `.agents/skills/ml-course-planning`, `.agents/skills/ml-current-research`, `.agents/skills/ml-chapter-writing`.

## Маршруты и уровни

У каждого обучающегося общий вход — основы языковой модели и Transformer. После него доступны независимые ветки: encoder/BERT; адаптация генеративных моделей; инференс; приложение; обучение агентов. Не требуется изучать LoRA для вызова инструмента или RAG. Подробное обучение retrieval не является предпосылкой чтения инференса. Внутри главы BERT даёт достаточное локальное введение retrieval/embeddings перед приложением.

Уровни у главы указываются отдельно: **общее ядро**, **основа ветки**, **углубление**. Предпосылки — понимание, а не факт наличия раньше написанного конспекта. Повторное использование понятия ссылается на его первое полное определение.

| Маршрут | Обязательная последовательность | Можно пропустить |
|---|---|---|
| Основа генерации | Foundations A1 → A3 → A4 forward → A5 | A2 RNN и подробный backward — углубление |
| BERT / encoder | Foundations A1/A3/A4 forward/A5 → BERT А–Б → выбранная ветка В–И | полный вывод MLM, старые objectives, необязательные архитектуры; RNN не требуется |
| Адаптация | Foundations A1/A3/A5 → выбор intervention → SFT → LoRA/PEFT → оценка | merging, continual learning и расширенный RLHF |
| Инференс | Foundations A1/A3/A5 → вводный next-token цикл (Е) → prefill/decode/KV (В) → serving (Г) → вернуться к decoding policies в Е → оптимизация по bottleneck | speculative decoding, compression и deployment — по задаче |
| Приложения | Foundations A1/A3/A5 → structured output → tools → state/orchestration → RAG → оценка | обучение собственных retrievers, если достаточно готовой модели |
| Обучение агента | приложения и baseline → lifecycle А → Б → К → демонстрации/SFT → recovery → RL при наличии среды | multi-agent, сложный reasoning и память без задачи, которая их требует |

## A. Основы языковых моделей

**Критерий каждой главы:** читатель без подсказки воспроизводит её числовой пример или objective, проходит по подписанной схеме, объясняет предпосылку и ограничение. Для глав основы ветки читатель также указывает необходимые предпосылки и темы, которые дальше используют материал.

Источник сценариев: [план основ LLM](<06-llm/01-foundations/План — основы LLM.md>) и [оглавление глав](<06-llm/01-foundations/Учебные конспекты/README.md>). Полный маршрут обязателен до генеративных веток; ветка BERT использует только нужные главы.

### A1. Токены, embeddings, позиции и обучение следующему токену

**Уровень:** общее ядро. **Предпосылки:** векторы, матричное умножение, условная вероятность. **Используют:** A2–A5, все ветки ниже.

**Сценарий:** начать с текстовой последовательности и токенизатора; развести строку, token ID, embedding и hidden state; показать lookup-таблицу, позиционную информацию, сдвинутые входы/цели, logits, softmax и cross-entropy next-token loss. Разобрать короткую последовательность с 3–4 токенами и вручную показать один целевой индекс. Схема: текст → tokenizer → IDs → embedding+position → модель → logits → loss/следующий токен. Практика: найти target shift и маску ответа в одном batch. Проверка: объяснить, что предсказывает loss и почему embedding не является вероятностью.

### A2. RNN и ограничения рекуррентного состояния

**Уровень:** углубление; можно пропустить. **Предпосылки:** A1. **Используют:** историческое сравнение с Transformer.

**Сценарий:** определить hidden state и рекуррентный переход; вывести вычисление по шагам и путь градиента через время; на маленьком примере показать последовательное состояние, затухание/взрыв градиента и зависимость вычислений по шагам. Схема: токен_t + state_{t-1} → cell → state_t/output. Практика: сравнить длину цепочки вычислений с параллельной обработкой входных позиций. Ограничение: RNN не сводится к «не умеет дальний контекст»; обозначить gating, размер состояния и вычислительную цену.

### A3. Transformer, causal mask и семейства архитектур

**Уровень:** общее ядро. **Предпосылки:** A1; A2 не обязателен. **Используют:** A4–A5 и все генеративные ветки; BERT использует encoder часть.

**Сценарий:** от self-attention перейти к residual/normalization/MLP и повторяемому блоку; задать causal mask как ограничение доступной информации; сопоставить encoder, decoder-only и encoder-decoder. Ввести матричную формулу attention и размерности прежде схемы полного блока. Числовой пример: маска длины 4 и один attention row. Схема подписывает embeddings, attention, residual, normalization, MLP, output и направление информации. Практика: определить, какие позиции доступны при causal и bidirectional mask. Ограничение: одинаковое слово Transformer не означает одинаковую цель обучения или интерфейс.

### A4. Q/K/V, multi-head attention и backward

**Уровень:** общее ядро — формы и forward; подробный backward — углубление. **Предпосылки:** A1, A3, матрицы и производные. **Используют:** объяснение attention, BERT и inference.

**Сценарий:** определить Query/Key/Value как обучаемые проекции; вывести score, scaling, mask, softmax и weighted sum; затем объединить heads и применить output projection. На матрицах с D=4 вручную пройти forward и один gradient path до Q/K/V; backward полного блока вынести во вложенное углубление. Отдельная схема показывает [B,L,D] → [B,H,L,d_h] и attention weights. Практика: проверить shapes при padding mask. Ограничение: attention weights — веса вычисления, а не автоматически причинное объяснение.

### A5. Hidden states, LM head и границы интерпретации

**Уровень:** общее ядро — интерфейсы; анализ слоёв — углубление. **Предпосылки:** A1, A3–A4. **Используют:** adaptation и представления encoder.

**Сценарий:** развести hidden state, pooling, LM head, logits и sampling; показать tied/untied embeddings и позицию головы в графе. Разобрать миниатюрную матрицу logits и softmax. Схема: последнее скрытое состояние → LM head → словарные logits → decoding policy. Практика: проверить, какая позиция даёт next-token prediction. Ограничение: смысл слоя нельзя выводить только из cosine similarity или probe без контроля.

## B. BERT и encoder-модели

Уровень ветки: **основа ветки** после A1/A3/A4/A5. Сценарии подробно разработаны в [плане BERT](<07-bert/План — BERT и encoder-модели для NLP.md>); текущая карта групп включает все 40 подпунктов в девяти самостоятельных главах.

1. **А — Encoder, tokenizer, attention и pooling (подпункты 1–5).** Определение задачи и выходов → WordPiece/offsets → входные embeddings и shapes → bidirectional attention/padding/cost → CLS/pooling. Пример: один текст с subword offsets и attention mask. Схема показывает tokens, encoder layers и разные pooling heads. Практика: сравнить CLS, mean pooling и masked mean; проверки — shape и отсутствие padding в pooling.
2. **Б — MLM, objectives и эволюция encoder (6–9).** MLM corruption/targets/CE/backward → NSP/SOP/span/RTD → какие ограничения меняет семейство BERT → выбор современных encoder checkpoints. Пример MLM с выбранной маской и unchanged token. Схема потока corruption → encoder → masked-position head. Практика: составить shortlist под длину, язык, memory и лицензию; не использовать дату выпуска как критерий качества.
3. **В — Данные, головы и fine-tuning (10–14).** Annotation/split → head/loss → frozen/full/PEFT → training steps/diagnostics → DAPT/TAPT. Числовой пример CE/BCE и подсчёт шагов. Схема data contract → tokenizer → encoder/head → loss/optimizer. Практика: обнаружить leakage и сравнить режим обучения на одном split.
4. **Г — Classification, extraction и QA (15–21).** Single/multilabel → hierarchy/selective decision → subword NER → nested spans/relations → paired NLI → extractive QA и null answer → few-shot alternatives. Пример: alignment character spans к subwords и восстановление offset. Схема отдельно рисует sequence head, span head и pair head. Практика: strict-span scoring и отказ при непокрытом ответе.
5. **Д — Sentence embeddings и геометрия (22–25).** Pooling/SBERT → contrastive objectives и negatives → distillation/Matryoshka → задачи и пределы embedding space. Вывести маленький InfoNCE batch с false negative. Схема двух views → shared encoder → normalized vectors → similarity/loss. Практика: сравнить retrieval, STS и clustering метрики; cosine не трактуется как абсолютная семантическая вероятность.
6. **Е — Dense, cross-encoder, SPLADE и ColBERT (26–29).** Dual encoder/index → pointwise/pairwise/listwise reranking → learned sparse/hybrid → late interaction/MaxSim. Пример вручную считает один MaxSim score и размер multi-vector index. Схема разделяет offline document encoding/index и online query/reranking. Практика: BM25 vs dense vs hybrid по фиксированным qrels и latency.
7. **Ж — Языки, документы и структуры (30–34).** Русский/multilingual transfer → hierarchical long context → domain-specific corpora → structured extraction → OCR/layout/table boundary. Пример: одинаковые OCR-токены с разной геометрией. Схема отмечает text-only и layout/vision input. Практика: оценить язык, длину, domain shift и источники OCR errors.
8. **З — Метрики и надёжность (35–37).** Под задачу выбрать метрику и статистику → калибровка/OOD/selective risk → attribution/probe/privacy limits. Вручную посчитать MRR/nDCG и coverage-risk. Практика: paired comparison, group bootstrap и counterfactual slice. Схема оценки разделяет retrieval, prediction, calibration и service cost.
9. **И — Compression и deployment (38–40).** Teacher signals/distillation/quantization → encoder batching/export → monitoring, feedback и index migration. Рассчитать memory компонентов по отдельности. Схема: versioned model bundle + tokenizer/pooling + index + policy → canary/rollback. Практика: post-export parity и совместимость embedding версии с индексом.

**Зависимости BERT:** encoder retrieval объясняется локально в главах Д–Е и не зависит от прохождения курса обучения поиска. RAG в приложениях ниже требует только выбранный готовый retriever и локальное определение evidence retrieval. Генеративное alignment не является предпосылкой BERT.

## C. Адаптация генеративных моделей

Уровень ветки: **основа ветки** после A1/A3/A5 и baseline evaluation. Подробный план и сценарии: [адаптация LLM](<06-llm/02-adaptation/План — адаптация LLM.md>). Обязательная последовательность: определить, нужен ли новый вес; поставить SFT и проверить baseline; объяснить LoRA после понятия trainable parameters; оценить перед preference optimization.

1. **А — Выбор адаптации, PEFT и память обучения.** Сформулировать failure mode → сравнить prompting/RAG/SFT/PEFT → считать trainable parameters, optimizer, gradients and activations. Пример бюджетирует полный и низкоранговый update. Схема показывает точку вмешательства. Проверка: предложить подход под конкретную ошибку и измеримую цель.
2. **Б — LoRA, target modules и два шага.** От полного update ΔW к BA → rank/alpha/target modules → shapes → первый и второй gradient updates → merge. На матрице 2×2 показать, почему первый grad A может быть нулевым, а следующий — уже нет. Схема: замороженная W0 + низкоранговая ветвь. Практика: посчитать trainable parameters; ограничения target-module/architecture.
3. **В — DoRA, AdaLoRA, LoRA+ и QLoRA.** Сопоставить факторизацию и parameter allocation → DoRA magnitude/direction → rank adaptation → differing learning rates → quantized base + LoRA. Числовой пример проверяет broadcast/magnitude и четыре-bit quantization boundary. Схема каждого метода показывает изменяемые состояния. Ограничения: memory savings зависят от optimizer/activations/runtime; method names do not establish quality.
4. **Г — Adapters, prefix, soft prompts и BitFit.** Определить, где именно вводятся trainable parameters → compare parameter count, sequence cost and serving path. Схема блока показывает inserted adapter, prefix K/V, input soft prompt and bias-only parameters. Практика: классифицировать checkpoint artifacts and compatibility. Ограничения: soft prompt и prefix K/V — разные объекты.
5. **Д — SFT, данные, masks и objective.** Instruction-response dataset → roles/template → labels/loss masks → token-level CE → packing/gradient accumulation. Разобрать batch с prompt labels=-100, target shift and loss normalization. Схема tokens+mask → causal LM → loss only on target tokens. Практика: проверить один transformed batch and source provenance.
6. **Е — RLHF, PPO и preference objectives.** Развести reward model/policy optimization/direct preference → вывести clipped PPO ratio and one DPO log-ratio term → compare IPO/KTO/ORPO by objective and data assumptions. Схема различает three-stage RLHF and preference-only training. Пример разбирает KL/reference policy/beta. Ограничения: reward hacking, judge quality and off-policy preference data.
7. **Ж — Model soup, TIES/DARE, merging и switching.** Weight/delta arithmetic → merge scales and sign conflicts → task vectors → switching/adapters. Ручной пример покажет cancellation при усреднении opposing updates. Схема base weights + updates + deployment choice. Практика: parity and task regression test. Ограничения: matching architecture/tokenizer/config and merge validation.
8. **З — Evaluation, perplexity, judge и экспорт.** Baseline/held-out data → automatic metrics/human rubric → judge calibration → regression/latency → save/reload. Пример: perplexity changes without improved task success. Схема раздельно показывает train/validation/test and deployment artifact. Практика: one post-export check and error slices.
9. **И — AI feedback, multiple objectives и continual learning.** Define feedback source → agreement and judge bias → multi-objective constraints/Pareto/scalarization → data update/rehearsal/forgetting. Toy example trades off two preferences and old/new tasks. Схема labels→critique→policy→re-evaluation. Ограничения: correlated feedback, proxy targets and forgetting; this chapter is an explicit addition to thin source coverage.
10. **К — Distillation, teacher annealing и собственные траектории.** Teacher logits/trajectories → temperature/KL direction → student objective/capacity → anneal teacher signals → data provenance. Рассчитать one soft-target loss. Схема teacher/student training stages. Практика: compare task metrics/compute and error transfer. Ограничения: teacher errors and data rights; this is adaptation training, not inference quantization.

## D. Инференс и исполнение

Уровень ветки: **основа ветки** после A1/A3/A5; приложению для работы достаточно базовой генерации и A6. Детальный plan ledger: [инференс LLM](<06-llm/03-inference/План — инференс LLM.md>).

**Вводное начало сценария:** прежде чем переходить к оптимизации, определить обычный следующий токен и один greedy/sampling step; это вводная часть D1, отдельную учебную главу не создаёт.

1. **А — Generation, prefill/decode и KV cache.** Из logits выбрать следующий token → разделить prompt prefill и autoregressive decode → вывести KV memory по batch/layers/heads/sequence/dtype. Схема request→prefill→cache→decode→stop. Рассчитать один token distribution и память cache. Сравнить MHA/MQA/GQA. Практика: measurement baseline and length scaling. Ограничение: cache экономит recomputation, но занимает память.
2. **Б — Serving, batching и PagedAttention.** Continuous batching, queueing, paged allocation/fairness. Пример fragmentation, padding and active sequence changes. Схема scheduler + KV blocks. Практика: throughput and p95 latency under two workloads. Engine version and model support are implementation-specific.
3. **В — Quantization: activations, PTQ/QAT, GPTQ/AWQ.** Define representation/error → affine example → calibration data → PTQ/QAT gradient distinction → modern weight quantizers. Hand-quantize vector and calculate output difference. Diagram weights/activations/runtime path. Practice: post-export quality/memory/latency. Limits: benchmark hardware/kernel and calibration distribution matter.
4. **Г — Pruning и граница с distillation.** Unstructured/structured pruning → sparsity pattern and hardware kernel → where distillation is retraining. Calculate parameter/activation counts. Diagram compares masking, compact weights, student training. Practice: measure only supported runtime. Sparse weights alone do not imply speedup; full distillation training is in C10.
5. **Д — Speculative decoding.** Draft sequence → verifier acceptance → residual distribution; manually compute accept/reject for a short case. Diagram preserves target distribution under rejection sampling. Practice: estimate throughput by workload and acceptance rate. Enrichment: multi-token heads/agent batching with explicit evidence status.
6. **Е — Greedy, sampling, beam and constrained decoding.** Temperature/top-k/top-p → beam limitations → grammar/structured decoding → parser and semantic validation. Example: syntactically valid JSON with wrong meaning. Diagram separates model token selection, constraints and application validator. Practice: errors, valid strings and end conditions.
7. **Ж — Long context, sliding window and prompt cache.** Position effects → window policies → retrieval vs long prompt → cache reuse/identity. Example compares evidence at two context positions and budget. Practice: position-controlled evaluation. Advertised context length is not uniform task quality.
8. **З — Orchestration, tools, RAG and runtime.** Local prompt→state→route→tool/error→observation→stop; RAG retrieval→evidence→answer. Diagram is a state machine with error edges. Practice: idempotency, timeout, missing argument and evidence provenance. Do not assume a universal learned tool head.
9. **И — Deployment, cold start, queues and scaling.** Model/artifact startup → probes → queue/admission → batching and autoscaling → rollout/rollback. Example distinguishes average from tail latency and queue wait. Diagram request lifecycle and scaling controller. Practice: compute cost per success.
10. **К — Cache, monitoring and application evaluation.** Exact vs semantic cache identity/invalidation → cohort/drift monitoring → judge/human evaluation → rollout. Example demonstrates stale/unsafe cache and single-flight. Diagram versioned service→telemetry→decision. Practice: total cost/quality and failure by slice. Framework names/versions are rechecked at chapter writing.

## E. Приложения на основе LLM

Уровень ветки: **основа ветки** после A1/A3/A5 и D1. Курс адаптации не требуется. Общие serving-затраты — в D; retriever обучения — в BERT Е лишь по необходимости.

1. **А — Structured output.** Определить контракт/schema → constrained decoding и parser → семантическая проверка. Пример: schema-valid, но противоречивый объект. Схема показывает границу model/parser/business validator. Практика: malformed, missing и conflicting values.
2. **Б — Tool calling и исполнение.** Определить tool как внешний эффект; name/arguments/authorization → execute → return observation → failure. Числовой пример idempotency/retry. Схема request/response cycle с ошибками и rollback. Проверка: назвать, где верифицируются права.
3. **В — State и orchestration.** State contract, reducer/checkpoint, graph routing, stop/retry/interruption. Разобрать одну state transition. Диаграмма направленного графа с состояниями и контрольными переходами. Практика: recovery после дублированного tool response. Ограничение: workflow не равен обученной модели.
4. **Г — RAG.** Локально определить retrieval и embedding → ingestion/chunk/evidence IDs → BM25/dense/hybrid → reranking → grounded answer. Пример вычисляет sparse/dense ranking и RRF. Диаграмма разделяет offline index и online query. Практика: gold-context/oracle сравнение, recall@k и groundedness. Ограничение: найденный источник не гарантирует правильный ответ.
5. **Д — Evaluation приложения.** Раздельно оценивать retrieval, evidence coverage, answer correctness, citation support, task success, safety/cost. Вручную разобрать false citation и failure attribution. Схема оценивает component и end-to-end. Практика: paired test set и group/bootstrap uncertainty.

## F. Обучение и улучшение агентов

Уровень: основа ветки — постановка задачи, траекторий и evaluation; методы обучения — **основа ветки** или **углубление** при наличии среды и хорошего outcome signal. Зависимость на инструменты проходит через E1–E3; LoRA — необязательная реализация SFT.

1. **А — Lifecycle и выбор вмешательства.** Failure analysis → prompting/retrieval/SFT/RL → cost/reproducibility. Пример одного defect, исправляемого каждым вмешательством. Схема помечает метрику и artifact каждого этапа.
2. **Б — Среда, observations, actions, reward, horizon.** Формально задать state, observation, transition, action, terminal, reward и partial observability; на toy task посчитать return и stop. Схема environment-agent loop.
3. **В — Behavior cloning и SFT tool trajectories.** Траектории с observations/actions → token serialization/masks → supervised objective. Разобрать, почему target не включает чужую observation. Схема trajectory → training example. Практика: валидировать rollout data.
4. **Г — Distribution shift, recovery и DAgger.** Ошибка политики → новые состояния → recovery demonstrations → агрегирование dataset. Пример на краткой цепи переходов. Схема expert/data aggregation loop. Ограничение: label budget и экспертная стоимость.
5. **Д — Reasoning и test-time compute.** Разделить learned reasoning и inference-time search → verifier/ORM/PRM → best-of-N/rejection/self-critique. Числовой пример success probability vs sample budget. Схема candidate→verifier→selection. Ограничения: compute, judge reliability и hidden evaluation leakage.
6. **Е — RL, PPO и GRPO.** Delayed reward/credit assignment → policy gradient/advantage → clipping/reference/KL → group-relative update → reward hacking. Вывести toy advantage. Схема rollout/train update/evaluation. Практика: trace reward exploit.
7. **Ж — Память и long horizon.** State vs transcript → summary/evidence/checkpoint → hierarchy. Пример с потерей provenance при summary. Схема persistent store, retrieval, summary and control. Практика: ablation memory components.
8. **З — Предпочтения траекторий.** Rubric/constraints → pairwise preference and policy objective → judge bias/Pareto → transfer limits from response-level DPO. Пример двух траекторий с разными cost/success. Схема trajectory comparison to training signal.
9. **И — Multi-agent.** Roles vs learned coordination → shared/separate memory → communication/debate/self-play → correlated failures. Сравнить одинаковый общий token/tool budget. Схема coordinator and agents with message paths. Ограничение: больше агентов не доказывает quality gain.
10. **К — Оценка траекторий.** Outcome oracle, recovery, loops, first-attempt vs budgeted success, cost over all attempts, splits/seeds. Рассчитать один metric report. Схема evaluator tracks complete trajectory and costs. Практика: contamination and reproducibility audit. Это предпосылка любого обучения агента.

## Реестр уровней и зависимостей глав

Эти записи дополняют предметные сценарии выше: уровень, предпосылки и темы-потребители заданы для каждой главы отдельно. Идентификаторы B1–B9 относятся к девяти BERT-главам, C1–C10 — к адаптации, D1–D10 — к инференсу, E1–E5 — к приложениям, F1–F10 — к lifecycle.

### BERT / encoder

| Глава | Уровень | Точные предпосылки | Какие темы используют её |
|---|---|---|---|
| B1 А — Encoder/tokenizer/attention/pooling | Общее ядро ветки | A1, A3–A5; matrices and tensor shapes | B2–B9 |
| B2 Б — MLM/objectives/model families | Общее ядро для pretraining route | B1 | B3/B4/B8; BERT model choice |
| B3 В — Data, head, loss, fine-tuning | Общее ядро для supervised route | B1–B2; D2 for split | B4/B8/B9 |
| B4 Г — Classification/extraction/QA | Основа branch by task | B1–B3 | B8/B9; selected app integration |
| B5 Д — Sentence embeddings/contrastive geometry | Основа representation branch | B1–B3; probability/geometry | B6/B7/B8/B9; E4 local retrieval |
| B6 Е — Retrieval/reranking | Основа retrieval branch | B1/B5; D2 data contract | B8/B9; E4 can use trained/frozen retriever |
| B7 Ж — Languages/domains/layout | Основа only for selected data domain | B1/B3; B5/B6 only for their retrieval cases | B8/B9 and domain applications |
| B8 З — Metrics/calibration/OOD/interpretation | Общее ядро до утверждений о качестве | B3 and chosen B4/B5/B6 task | B9; E5 application evaluation |
| B9 И — Compression/service/index lifecycle | Углубление; service contract is core for deployment | B3/B8 and selected B4/B5/B6 | E4/E5 and production systems |

### Адаптация

| Глава | Уровень | Точные предпосылки | Какие темы используют её |
|---|---|---|---|
| C1 А — Выбор вмешательства/память | Общее ядро ветки | A1/A3/A5 and baseline evaluation | C2–C10 |
| C2 Б — LoRA | Общее ядро ветки | C1, A3–A5 | C3/C5/C7/C10 |
| C3 В — DoRA/AdaLoRA/LoRA+/QLoRA | Основа PEFT-ветки | C2; quantization in D3 | C5/C8/C10 and deployment |
| C4 Г — Adapters/prefix/soft prompt/BitFit | Основа PEFT-ветки | C1, A3–A5 | C8/C10 and adapter serving |
| C5 Д — SFT/data/masks/objective | Общее ядро адаптации | A1/A3/A5, D8, C1 | C6/C9/C10; F3/F4 |
| C6 Е — RLHF/PPO/preference objectives | DPO — основа preference route; PPO — углубление | C5, A5 | C8/C9 and F6/F8 |
| C7 Ж — Merging | Углубление | C2/C3/C5 | C8 and multi-adapter serving |
| C8 З — Evaluation/export | Общее ядро до deployment | C1 and chosen C5/C6 method | C9/C10, D2–D10, E5 |
| C9 И — AI feedback/multiple objectives/continual learning | Углубление; основа if task requires it | C6/C8 | F6/F8 and continued adaptation |
| C10 К — Distillation | Углубление | C5/C8, A5 | Compact deployment; F3 |

LoRA is optional for SFT; structured output, tools and RAG do not depend on this route.

### Инференс

| Глава | Уровень | Точные предпосылки | Какие темы используют её |
|---|---|---|---|
| D1 В — Generation/prefill/decode/KV cache | Общее ядро ветки | A1/A3/A5 | D2–D10 and applications |
| D2 Г — Serving/batching/PagedAttention | Основа serving-ветки | D1 | D9/D10 and production systems |
| D3 А — Quantization | Основа compression-ветки | D1; numeric formats/linear algebra | C3/D4 and cost-limited serving |
| D4 Б — Pruning/distillation boundary | Углубление | D3 | Optimized serving; training → C10 |
| D5 Д — Speculative decoding | Углубление | D1 and probability distributions | High-throughput decode/agent serving |
| D6 Е — Decoding/constrained output | Sampling — ядро; beam/grammar — углубление | A5/D1 | E1/E2 and applications |
| D7 Ж — Long context/prompt cache | Основа long-context branch | D1 | RAG and production apps |
| D8 З — Orchestration/tools/RAG | Основа application branch | D1/D6; adaptation is not required | E1–E5; F1/F4 |
| D9 И — Deployment/scaling | Основа by deployment need | D2 and workload metrics | D10 and rollout |
| D10 К — Cache/monitoring/evaluation | Evaluation — ядро; advanced cache — углубление | D1/D6 and service metrics | E5 and F3 |

### Приложения

| Глава | Уровень | Точные предпосылки | Какие темы используют её |
|---|---|---|---|
| E1 А — Structured output | Общее ядро приложения | A1/A3/A5 and D6 | E2/E3/E5 |
| E2 Б — Tool execution | Основа tool-use branch | E1 | E3/E5, F1/F4 |
| E3 В — State/orchestration | Основа application workflow | E2 | E4/E5 and F1/F4 |
| E4 Г — RAG | Основа knowledge-application branch | E1; local retrieval/embedding definition | E5; no encoder-training course required |
| E5 Д — Application evaluation | Общее ядро для применения | E1–E4 | C8, D10, F1/F3/F10 |

### Lifecycle и обучение агентов

Порядок для основной ветки: **F1 постановка → F2 среда → F10 оценка траекторий → F3 обучение на демонстрациях → F4 восстановление/DAgger → F6 RL при наличии среды → F7–F9 по задаче.** F5 reasoning/test-time compute — отдельное углубление после F3/F10. Это делает оценку предпосылкой обучения, даже если её source chapter в старом плане обозначен последней буквой.

| Глава | Уровень | Точные предпосылки | Какие темы используют её |
|---|---|---|---|
| F1 А — Lifecycle/intervention | Общее ядро | A1/A3/A5, E5 | F2–F10 |
| F2 Б — Environment/reward/horizon | Общее ядро agent-training ветки | F1, E2/E3 | F3–F10 |
| F3 В — Behavior cloning/SFT trajectories | Основа demonstration learning | F2/F10; C5 необязателен | F4–F9 |
| F4 Г — Distribution shift/DAgger | Углубление recovery branch | F2/F3, expert or validated feedback | F6/F7 |
| F5 Д — Reasoning/test-time compute | Углубление | F1/F3/F10 and a verifier/outcome metric | F6/F10 |
| F6 Е — RL/PPO/GRPO | Углубление; основа при проверяемой среде | F1–F3/F10, reward definition and budget | F7–F10 |
| F7 Ж — Memory/long horizon | Основа memory branch by task | F2/F3/F10 | F8/F9 |
| F8 З — Trajectory preferences | Углубление | F3/F10, preference rubric and judged data | F6/F9 |
| F9 И — Multi-agent | Обзор; углубление by need | F1/F2/F10 and equal total-budget protocol | Terminal branch; its result uses F10's evaluation protocol |
| F10 К — Trajectory evaluation | Общее ядро до обучения | F1/F2; task-success/cost definition | F3–F9; C8/D10/E5 |

## Актуальные исследования: редакционный срез

Проверено 20 октября 2026. Это опорные современные точки сравнения, не готовый список «лучших моделей». Перед написанием главы сверить paper, code/version и текущий benchmark. Использовать по тематической главе и отмечать область доказательств.

| Тема | Источник и статус | Как применить и ограничить вывод |
|---|---|---|
| Encoder retrieval для нескольких языков | [Bekko Embedding (2026 preprint)](https://arxiv.org/abs/2607.25180) | Сопоставить active parameters, multilingual benchmark и hard-negative training; считать утверждения авторским результатом до независимой репликации. Подпункты BERT Д–Е/З–И. |
| Decoder-only/encoder architecture tradeoffs | [ModernBERT](https://arxiv.org/abs/2412.13663), [NeoBERT](https://arxiv.org/abs/2502.19587), [Ettin](https://arxiv.org/abs/2507.15864) в плане BERT | Сравнивать длину, pretraining tokens, task mix, languages, license and inference; не ранжировать модели по дате или единственному leaderboard. BERT Б. |
| Long-context speculative inference | [AgentSpec, EMNLP 2026](https://www.microsoft.com/en-us/research/publication/agentspec-speculative-decoding-for-batch-inference-of-llm-agents/) | Отдельно проверить workloads с повторяющимися agent prompts, acceptance/throughput and implementation availability; conference result не переносить на все generation traffic. D5. |
| Trajectory-level evaluation | [TRAJECT-Bench, ICLR 2026](https://openreview.net/pdf?id=TZWnWvsQ0X); [ACEBench, Findings of EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.697/) | F10: measure final task result plus tool choice, arguments, order/dependency and recovery where benchmark supports it. Results remain benchmark-scoped; simulated/executable tools and rubric define what they prove. |
| Long-horizon multi-turn RL | [AgentGym-RL, ICLR 2026](https://openreview.net/pdf/d8e15fc2cabf98ead8ffbe0d8e6ad1b2764a3d28.pdf) | F6: compare unified interactive environments and multi-turn updates; report environment suite, initialization, reward/verifier, rollout budget and compute. Framework availability is not evidence of general task transfer. |
| Interactive user feedback during agent training | [MUA-RL (2025 preprint)](https://arxiv.org/abs/2508.18669) | F2/F6: candidate enrichment for dynamic user preferences and clarification loops. Preprint and author-reported benchmark results; keep simulated-user behavior and domain-specific claims explicit. |

Для адаптации и serving основные сравнения и гипотезы уже разложены по главам исходных планов. Этот ledger намеренно фиксирует recheck перед написанием; версии движков, hardware и заявленные acceleration ratios быстро меняются. **Источник недостающий или не проверяемый первично — дополнение не включать до аудита.**

## Связанные планы и источники

- Foundations, adaptation, inference, applications, lifecycle: подпланы и главы в `06-llm/`.
- BERT/encoder: [40 подпунктов и девять глав](<07-bert/План — BERT и encoder-модели для NLP.md>).
- История источников S1–S6, исправленные утверждения и продолжение: [`06-llm/migration/`](<06-llm/migration/MIGRATION_PLAN.md>); главы имеют отдельные source links/ranges.
- Данные SFT/chunking и leakage: [данные и признаки](<План — данные и признаки.md>).
- Карта исходников и семантических единиц: [редакционная карта](<Редакционная карта курса.md>).
