# И. Как сохранить качество после сжатия, экспорта и обновления

[Оглавление](README.md). Группа I, модули **38–40**. **Основа:** S4 обсуждает compression/serving, S6 — deployment/scaling. Здесь раскрыты encoder-specific tasks, fixed-vector index и monitoring; общая механика не дублируется вместо объяснения отличий.

## 1. Distillation classifier и retriever

Classifier student может получать hard labels и teacher soft distribution. Для temperature $T$, $p_T=\operatorname{softmax}(z_T/T)$, $p_S=\operatorname{softmax}(z_S/T)$:

$$
L=(1-\alpha)L_{\mathrm{hard}}+
\alpha T^2D_{\mathrm{KL}}(p_T\Vert p_S).
$$

Teacher detached; KL direction важна. Factor $T^2$ согласован с распространённой convention компенсации temperature scaling gradients, но normalization/classes и coefficient definition меняют effective strength. Его нельзя копировать без проверки полного objective.

Retriever student может match embeddings с projection, pairwise scores или candidate ranking distribution. Требуется сохранить target geometry/pooling/prefixes, а не только classifier accuracy. Intermediate states разных depths/dimensions сравнивают через explicit alignment; «удалили layers» без дополнительного обучения — другой baseline.

[DistilBERT](https://arxiv.org/abs/1910.01108) — ранний пример encoder distillation. Его mechanism не означает, что любой student сохраняет все teacher skills. Evaluate rare/long/OOD cases и calibration; teacher labels могут переносить mistakes/contamination. Teacher inference/training data cost входит в full budget.

## 2. Quantization и pruning

Weight quantization снижает representation precision, activation quantization — intermediate tensors, pruning удаляет selected weights/structures. Affine toy quantization:

$$
q=\operatorname{clip}\left(\operatorname{round}(w/s)+z,q_{\min},q_{\max}\right),
\qquad \widehat w=s(q-z).
$$

Scale, zero point, granularity и calibration data определяют error. Tiny perturbations logits могут менять rank/threshold near boundary, даже если average vector MSE мала. Поэтому post-export evaluation включает actual task metric, не только cosine agreement vectors.

Unstructured zeros не ускоряют dense kernels автоматически. Structured pruning меняет shapes/heads и требует supported runtime. Quantized weights не означают, что activation memory и transfer costs уменьшились тем же factor. Supported architectures/heads/device formats проверяются у выбранного runtime version.

## 3. Memory accounting

Для inference fp32 weights одного encoder с $P$ parameters занимают примерно $4P$ bytes плюс activations/workspace/runtime. На training добавляются gradients и optimizer states. FP32 Adam с weights/gradients/two moments даёт порядка $4P+4P+8P=16P$ bytes для trainable parameters, без activation memory и framework overhead. Mixed precision/master copies меняют components; frozen base и trainable head считаются раздельно.

Для $P=100$ million weights fp32 — 400 MB decimal; rough trainable parameter tensors FP32 Adam — 1.6 GB до activations. Это расчёт representations, не обещание actual peak VRAM. Length/batch/attention backend и checkpointing могут сделать activations dominant.

При retrieval total memory включает corpus vectors/index/metadata, часто больше encoder weights. Уменьшение student encoder не уменьшает старый 768-dimensional index, если vector dimension не поменялась. Смена dimension/objective требует re-encoding либо специально проверенного compatibility protocol.

## 4. Encoder inference отличается от causal decode

Обычный bidirectional encoder обрабатывает whole input; нет sequential next-token decode с тем же KV-cache contract, что causal LLM. Добавление нового token может изменить states прежних tokens из-за bidirectional attention, поэтому нельзя просто reuse causal cache semantics.

Workload — texts, pairs или chunks. Tokenization, padding, transfer, forward и pooling измеряются отдельно и end-to-end. Для dense retrieval offline document encoding и online query path имеют разные throughput/latency goals. Cross-encoder обслуживает pairs и candidate count, а не один document embedding.

Dynamic batching увеличивает utilisation ценой queue wait; length bucketing снижает padding, но меняет ожидание отдельных requests. Throughput и p95 latency измеряются на representative length/arrival distribution, warm/cold runs и same task. Один long pair может раздуть batch work всех коротких.

## 5. Export и versioned bundle

Eager baseline → один export/runtime change → reload → output и task metric comparison. ONNX/другой backend может не поддержать custom architecture/masks или dynamic shapes так, как ожидается. Empty, Unicode, long/truncated и padding cases проверяются отдельно. Небольшая numeric difference допустима только в объявленных task tolerances.

Bundle содержит tokenizer/revision, weights/head, pooling/projection, normalization/prefixes, label map, thresholds, max-length/truncation policy, runtime/config и data/index compatibility. Сохранить только tensor weights недостаточно. [Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference) — пример runtime, чью поддержку конкретного checkpoint/head проверяют по версии.

## 6. Monitoring и active learning

Отслеживают input/source/language/length, class frequencies, missing/unknown rates, embedding norms/distributions, latency/errors и delayed task labels. Vector drift — proxy изменений, а не доказанное падение relevance. Calibration/threshold change меняет decisions даже без нового backbone и версионируется отдельно.

Active learning выбирает uncertain и diverse cases для human annotation. Но выбор только uncertain examples смещает labelled sample; такой sample нельзя напрямую считать unbiased production test. Нужен independent representative audit и provenance feedback. Diversity не гарантирует coverage critical rare slices, поэтому sampling strategy проверяют.

После исправления labels обучают candidate и сравнивают на held-out tasks. Feedback пользователя может быть noisy или отражать exposure предыдущего ranking; automatic ingestion без review усиливает bias. Annotation policy и retention/source contract сохраняются.

## 7. Обновление index и rollout

При новом tokenizer/encoder/pooling/normalization query и document vectors обычно должны перейти в согласованный space. Равная dimension не доказывает compatibility. Полный reindex делается в новой version с проверенными counts, IDs, ACL и удалёнными устаревшими documents. Ingestion новых docs и migration старых — разные операции.

Порядок: offline regression/cost → shadow на реальном distribution без изменения ответа → ограниченный canary по заранее заданным metrics → rollout либо rollback. Rollback восстанавливает coherent model+index+threshold bundle, а не только old weights. Одновременный поиск новых queries по старым docs vectors требует специально verified mapping/dual-index protocol.

Итоговый service report включает post-export quality, p50/p95, throughput, memory, cold start и limits. Настоящий benchmark записывается после запуска; формулы памяти и pipeline design являются учебным расчётом.

## 8. Контроль и практика модулей 38–40

1. Почему small vector MSE не гарантирует unchanged retrieval ranking? **Ответ:** near-tie candidates могут поменяться местами, а metric зависит от порядка и relevance.
2. Почему sparse weights не обязательно быстрее? **Ответ:** runtime может всё ещё исполнять dense kernels.
3. Можно ли кэшировать BERT как causal decode? **Ответ:** ordinary bidirectional states зависят от обеих сторон, поэтому semantics различается.
4. Что rollback должен восстановить для search? **Ответ:** согласованные model/tokenizer/pooling/index/decision-policy versions.

**Практика:** выберите teacher signals для classifier и retriever, посчитайте rough memory components. Спроектируйте eager/export comparison и нагрузочный test. Напишите migration нового encoder с dual snapshots, checks IDs/ACL и criterion rollback; это основа проекта P6.
