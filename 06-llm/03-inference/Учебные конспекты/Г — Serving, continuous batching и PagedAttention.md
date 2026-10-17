# Г. Serving: как исполнять много запросов

[Оглавление](README.md). Блок B.1–B.4 и практические протоколы 1/5.

**Основа:** S4, строки 4926–6435. Сохранены static/continuous batching, paged allocation, serving engines и контраст с CV/encoder inference. **Исправления:** streaming короткого ответа возможен и без continuous batching; paging не меняет математическую модель; конкретные frameworks не получают вечного ranking. **Углубление:** occupancy example, измерения и состав runtime contract.

## 1. Зачем нужен scheduler

Авторегрессивные запросы имеют разные prompt/output lengths. В fixed batch sequences могут закончиться в разное время; простой scheduler не подставляет новые запросы до конца группы. Если lengths outputs $\left(2,\,5,\,1,\,4\right)$, за5 iterations имеется20 sequence slots, из них useful12. Упрощённая occupancy12/20=60%; это показатель slots, а не точный GPU utilisation.

В исходной аналогии короткий пользователь «обязан ждать до конца самого длинного». Это верно для некоторых nonstreaming wrappers, но не обязательно для всех static batches: engine может stream/вернуть готовый output раньше, даже если не умеет refill свободного slot. Поэтому latency пользователя и batch scheduling — связанные, но разные свойства.

Continuous/in-flight batching допускает обновление active sequences на boundaries iterations: завершённые освобождают slots/cache, новые проходят prefill и входят в decode. Scheduler должен согласовать разные phases, memory budgets и fairness. Новый длинный prefill может задержать уже декодирующиеся interactive requests, поэтому применяются token budgets/chunked prefill и другие policies.

## 2. PagedAttention: allocation blocks вместо монолитного запаса

Если каждой sequence заранее резервировать maximum context, много памяти пропадает. Variable-sized contiguous allocations могут фрагментироваться. Paged KV scheme хранит logical sequence blocks в доступных physical blocks с mapping metadata. Это помогает динамическим lengths, sharing и освобождению без обязательной contiguous области.

Учебный пример block16tokens: sequence длины17 использует2blocks, последний заполнен1/16. В пределах block остаётся internal fragmentation15tokens, но не резервируется сразу весь maximum4096. При sharing prefix несколько sequences могут ссылаться на общие blocks с refcounts/copy-on-write policy, если implementation это поддерживает.

PagedAttention сочетает memory layout/management с kernels чтения такого cache. Она не меняет softmax objective и не позволяет игнорировать past keys без специальной attention architecture. [PagedAttention](https://arxiv.org/abs/2309.06180).

Continuous batching и paging решают разные задачи: первое scheduling, второе memory allocation/access. Prefix caching переиспользует вычисленный prefix; speculative decoding уменьшает число expensive target iterations. Наличие одного механизма не доказывает наличие остальных.

## 3. Как сравнивать engines

vLLM, TensorRT-LLM, TGI, llama.cpp/Ollama и другие systems различаются supported architectures, kernels, parallelism, scheduler/cache и interfaces. Это examples разных implementation choices; API/status/support конкретного проекта могут меняться, поэтому перед запуском нужна текущая документация и version pin.

Локальный GGUF runtime может использовать CPU, GPU и offload. Ollama/LM Studio предоставляют удобный доступ к models/API, но приложение с tools/loop обычно требует внешнего orchestrator или встроенной конкретной feature. OpenAI-compatible endpoint означает совместимость части request/response contract, а не полную идентичность structured output, tool schemas, cache или sampling semantics.

Выбирать engine по actual model, hardware, task constraints и workload. Benchmark только одного короткого prompt не определяет multi-user serving. Если модель не поддерживается backend или нужный quantization path работает через fallback, опубликованные speed numbers другого path не применимы.

## 4. Throughput, latency и очередь

Throughput: tokens/s или requests/s с объявленным definition. Для output tokens/s нельзя незаметно добавить input tokens и получить несравнимое число. Per-user tokens/s, aggregate output throughput и task-success throughput отвечают на разные вопросы.

TTFT, inter-token latency и end-to-end p50/p95 измерять отдельно. Увеличение batch часто повышает aggregate efficiency, но добавляет очередь/более длинный iteration. Если arrival rate превышает service capacity, queue растёт: больше batch не бесконечный источник capacity.

Для stable system Little's law связывает mean in-flight count N, arrival rate λ и mean total time W: $N\approx\lambda W$ при соответствующих assumptions. При λ=4requests/s и W=2s в среднем8requests in system. Это не формула p95 и не обещание stability при bursty overload.

Тестировать матрицу short/long input × short/long output × concurrency. Записывать error/OOM/cancellation, warm/cold, peak memory и quality. GPU timings требуют корректной synchronization/measurement; wall-clock around async launch может измерить только enqueue.

## 5. CV/encoder inference для контраста

Classifier/encoder обычно обрабатывает полный input одним forward; нет такого же autoregressive decode loop с накоплением KV по ответу. Dynamic batching всё равно полезен, а lengths/padding важны для text encoders. Video/recurrent models могут быть stateful, поэтому «все CV stateless» — тоже упрощение.

ONNX — representation/export ecosystem, ONNX Runtime — execution runtime. `torch.compile` — compilation path внутри PyTorch ecosystem; export требует supported graph/shapes/ops. TorchScript встречается в existing projects, но не следует назначать его универсальным актуальным default без проверки версии/статуса.

Export success не подтверждает task equivalence. Сравнить outputs/metrics после reload, representative sizes, preprocessing/postprocessing и latency. Input resize/normalization/tokenizer — часть bundle, даже если не лежит внутри graph.

## 6. Контрольные вопросы

1. Почему occupancy60% не равно GPU60%? **Ответ:** slot costs неодинаковы, есть phases, kernel efficiency и other overhead.
2. Что paging не устраняет? **Ответ:** логическую потребность хранить/read доступные K/V и compute новой позиции.
3. Достаточно ли одного aggregate tokens/s? **Ответ:** нет, нужны per-user latency, quality и errors под заданным workload.

**Практика:** нарисовать schedule lengths2,5,1,4 и refill slots новыми requests. Посчитать allocated blocks для lengths1,16,17,31 при block16. Затем составить benchmark config без фиктивных результатов: model/runtime/revision, lengths, arrival policy, concurrency, metrics и quality gate.
