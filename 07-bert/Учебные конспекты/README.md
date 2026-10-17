# BERT и encoder-модели: подробные учебные материалы

[Версия для чтения с набранными формулами](<../../Читать конспекты.html#course-11-0>) · работает без интернета.

[План на 40 модулей](<../План — BERT и encoder-модели для NLP.md>) · [Общее оглавление](<../../Учебные конспекты.md>) · [Подборка первоисточников](<../notes/Источники и современные encoder-модели.md>)

40 модулей объединены в **девять глав по группам A–I исходного плана**. Внутри каждой главы темы выделены подразделами; это несколько занятий, а не один обязательный длинный урок. BERT — новая ветка: чаты Transformer/PEFT/RAG дают предпосылки, но систематическое изучение BERT и ваши training runs ими не подтверждены. Новые explanations явно отмечены как дополнение.

## Соответствие всем модулям

1. [А — Encoder, tokenizer, attention и pooling](<А — Encoder, tokenizer, attention и pooling.md>). Группа A, **1–5**: постановки и outputs; WordPiece/offsets; embeddings/post-LN/shapes/counts; padding/cost; CLS/pooler/masked pooling.
2. [Б — Pretraining и эволюция](<Б — MLM, objectives и эволюция encoder-моделей.md>). Группа B, **6–9**: MLM/CE/backward; NSP/SOP/span/RTD; BERT-family mechanisms; ModernBERT/NeoBERT/mmBERT/Ettin и выбор checkpoint.
3. [В — Fine-tuning и data contract](<В — Данные, головы, fine-tuning и DAPT.md>). Группа C, **10–14**: annotation/splits; heads/losses; frozen/full/PEFT; training loop/steps; DAPT/TAPT/forgetting.
4. [Г — Прикладные головы](<Г — Classification, NER, spans, пары и extractive QA.md>). Группа D, **15–21**: classification; multi-label/hierarchy/selective; NER/POS/slots/CRF; nested spans/RE/events/coreference; NLI/STS/pairs; QA/windows/null; NLI-zero-shot/verbalizers/SetFit.
5. [Д — Representation learning](<Д — Sentence embeddings, contrastive loss и геометрия.md>). Группа E, **22–25**: SBERT/pooling; InfoNCE/triplet/SimCSE/negatives; distillation/Matryoshka/geometry; STS/dedup/clustering/anomalies/recsys.
6. [Е — Retrieval и reranking](<Е — Dense, cross-encoder, SPLADE и ColBERT retrieval.md>). Группа F, **26–29**: dense/index/prefix; pointwise/pairwise/listwise cross-encoder; sparse/SPLADE/hybrid; ColBERT/MaxSim/storage/M3.
7. [Ж — Специализированные ветки](<Ж — Языки, длинные документы, домены и layout.md>). Группа G, **30–34**: русский/multilingual; long/hierarchical; science/code/domain; aspect/linking/structured fields; OCR/layout/tables.
8. [З — Оценка и надёжность](<З — Метрики, калибровка, OOD и интерпретация.md>). Группа H, **35–37**: task metrics/benchmarks/statistics; calibration/OOD/coverage-risk; attribution/probes/shortcuts/data access.
9. [И — Compression и deployment](<И — Compression, encoder service и жизненный цикл индекса.md>). Группа I, **38–40**: teacher signals/quantization/memory; encoder batching/export; monitoring/active learning/index migration/rollout.

Все математические expressions — LaTeX. Числовые примеры имеют явные assumptions и не являются benchmark results ваших models. В каждой главе есть механика, ограничения, контрольные вопросы с ответами и практика.

## Маршруты

**База:** А → MLM и выбор семейства из Б → В → classification из Г → З. **NER/extraction:** база → NER/spans из Г → language/domain/linking из Ж. **Search:** база → Д → Е → З/И. **QA:** pair/QA из Г → retrieval из Е → long documents из Ж. **Малый budget:** В → few-shot из Г → Д/Ж → З. Не нужно сначала обучать encoder с нуля.

## Шесть сквозных проектов: спецификации и критерии

### P1. Классификация обращений

Russian intent/topic texts с guideline и group/time split. Сравнить TF-IDF+linear, frozen task-appropriate representations и tuned encoder. Выбрать одну интервенцию, одинаковый tuning budget и закрытый test. Artifact: manifest, config, script/notebook, per-class errors, calibration/abstention и cost. Готовность определяется проверенным task outcome, а не training loss. Главы А/В/Г/Ж/З.

### P2. NER и structured extraction

Char spans/types → subword alignment → token/span head → original offsets. Проверить фамилии/дефисы, padding, rare types и boundaries. Сначала strict entity metrics; RE добавлять после сравнения gold/predicted entities. Artifact: guideline, alignment examples, label map, errors и reproducible pipeline. Главы А/Г/Ж/З.

### P3. Поиск по документам

Версионированный corpus, query split и qrels. BM25 baseline → dense → reranker, меняя один stage. Pooling/prefixes фиксируются по checkpoint. Report Recall/MRR/nDCG, ANN approximation отдельно, latency/storage, disagreements и candidate ceiling. Artifact: index snapshot/IDs/config и held-out report. Главы Д/Е/З/И.

### P4. QA по русским документам

К P3 добавить extractive head с offsets/no-answer. Сравнить gold-context и retrieved-context, evidence positions и window boundaries. Report answer correctness, retrieval failures, null calibration и source consistency. Generative RAG — отдельный baseline с собственным budget. Главы Г/Е/Ж/З.

### P5. Мало labels и domain shift

Несколько independent few-shot samples/seeds → frozen/SetFit/task tuning. Optional DAPT сравнивается с учетом total compute. Построить learning curve human-label count, rare/new label errors и target-domain test. Pair generation не увеличивает число independent исходных texts. Artifact: sample IDs, uncertainty и следующий annotation batch. Главы Б/В/Г/Ж/З.

### P6. Компактный encoder service

Начать с проверенного P1/P3; сравнить compact/distilled/quantized option, затем export/reload и load test. Artifact: versioned bundle, post-export task metrics, p50/p95, throughput/memory, rollback и index compatibility. Teacher/offline encoding costs включены. Главы И и task-specific части Е/З.

Теоретический режим: достаточно specification, ручных вычислений и анализа failures. Численные результаты запуска и практическое освоение фиксируются отдельно после выполнения, не начисляются за наличие текста.

## Контрольные задачи большого плана: где разобраны

1. Shapes/masks — А. 2. MLM CE/gradient/unchanged — Б. 3. CE/BCE heads — В/Г. 4. BIO/subword/offsets — А/Г. 5. QA constrained span/null — Г. 6. Contrastive row/false negative — Д. 7. Mean/MaxSim/cross-encoder — Д/Е. 8. MRR/nDCG/ANN recall — Е/З. 9. Russian/long/multilingual shortlist — Б/Ж. 10. Reindex/retraining/rollback — И.

Для каждого модуля сохраняйте обсуждение, самостоятельный ответ и практический artifact раздельно. Источники подтверждают mechanisms/существование моделей; они не являются результатом эксперимента на вашем corpus.
