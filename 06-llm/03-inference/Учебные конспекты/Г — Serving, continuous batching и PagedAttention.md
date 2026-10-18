# Г. Serving: как исполнять много запросов

[Оглавление](README.md). Блок B.1–B.4 и практические протоколы 1/5.

**Основа:** S4, строки 4926–6435. Сохранены static/continuous batching, paged allocation, serving engines и контраст с CV/encoder inference. **Исправления C19:** streaming короткого ответа возможен и без continuous batching; paging не меняет математическую модель; конкретные frameworks не получают вечного ranking. **Углубление:** occupancy example, измерения и состав runtime contract.

## 1. Зачем нужен scheduler

Авторегрессивные запросы имеют разные prompt/output lengths. В fixed batch sequences могут закончиться в разное время; простой scheduler не подставляет новые запросы до конца группы. Если lengths outputs $\left(2,\,5,\,1,\,4\right)$, за 5 iterations имеется 20 sequence slots, из них useful12. Упрощённая occupancy12/20=60%; это показатель slots, а не точный GPU utilisation.

В исходной аналогии короткий пользователь «обязан ждать до конца самого длинного». Это верно для некоторых nonstreaming wrappers, но не обязательно для всех static batches: engine может stream/вернуть готовый output раньше, даже если не умеет refill свободного slot. Поэтому latency пользователя и batch scheduling — связанные, но разные свойства.

Continuous/in-flight batching допускает обновление active sequences на boundaries iterations: завершённые освобождают slots/cache, новые проходят prefill и входят в decode. Scheduler должен согласовать разные phases, memory budgets и fairness. Новый длинный prefill может задержать уже декодирующиеся interactive requests, поэтому применяются token budgets/chunked prefill и другие policies.

### Восстанавливаем исходный пример с четырьмя пользователями

В S4 outputs A,B,C,D имели длины 2,300,150,1tokens. Наивный fixed batch делает 300iterations с четырьмя позициями в каждой:1200slot-iterations. Полезных output tokens453, то есть упрощённая доля 37.75%. После 150-го step остаётся один незавершённый запрос, но из этого не следует автоматически, что ровно 75% GPU простаивает. Один большой матричный kernel может занимать устройство, а эффективная работа на token зависит от phase, batch и history length.

Если wrapper возвращает массив ответов только после всего `generate`, короткий D действительно ждёт B. Но это свойство **выдачи результата**, а не математическая необходимость static batch. Можно stream tokens или вернуть finished sequence раньше, сохраняя fixed состав группы. Можно также compact active batch, не принимая новые requests. Continuous batching добавляет возможность принять следующий запрос на iteration boundary; streaming и compaction —отдельные решения.

Для короткого примера(2,5,1,4) обозначим новые задачи E длины 3,F длины 2,G длины 1. Предположим, prefill уже готов или его стоимость в этой схеме отдельно исключена. Возможен schedule:

```text
iteration 1: A B C D   # C заканчивается
iteration 2: A B E D   # A заканчивается
iteration 3: F B E D
iteration 4: F B E D   # F,E,D заканчиваются
iteration 5: - B G -   # B,G заканчиваются
```

Получилось 18useful slots из 20 вместо 12. Последний batch всё равно неполон: новых задач больше нет. Если бы E требовал длинного prefill, его нельзя было бы бесплатно поставить на место C. Так аналогия S4 с поездом сохраняет смысл входа/выхода на остановках, но не обещает постоянно полный поезд и мгновенный первый token. Iteration-level scheduling подробно исследуется в [Orca](https://www.usenix.org/conference/osdi22/presentation/yu).

### У scheduler два бюджета: работа и resident state

Ограничение числа sequences не равно ограничению processed tokens. Один decode request обычно приносит в iteration одну новую position, а новый prompt —сотни или тысячи. Если token budget512 и зарезервировано 20decode positions, на prefill в этом условном iteration остаётся 492positions. Prompt1024 можно обработать chunks492+492+40 при сохранении такого резерва. Реальная policy также учитывает, что состав active requests меняется.

Chunked prefill ограничивает монополию длинного prompt, но может увеличить его собственный TTFT. При decode-first policy длинные новые prompts могут ждать; при prefill-first уже отвечающие users получают длинные inter-token gaps. Приоритеты, aging, deadlines и admission выбирают компромисс. Continuous batching не доказывает минимальный latency для каждого клиента одновременно.

Второй бюджет —cache blocks и workspaces. Новый запрос может помещаться по token budget текущего iteration, но не помещаться по прогнозируемому resident KV. Нужно определить, можно ли принять его, ждать, preempt/recompute другой request или отказать. Отмена клиента должна остановить лишнюю работу и освободить только принадлежащие ему references после завершения использующих их операций. Это часть корректности scheduler, а не только способ улучшить benchmark.

При packing разных requests causal mask ещё и изолирует sequences. Token нового User E не должен читать историю User B, хотя tensors исполняются одной GPU batch. Возможны отдельные lengths/block tables и специализированные kernels вместо огромной explicit mask; важна эквивалентность разрешённых attention edges. Перепутанные metadata дадут неправильную функцию даже при perfectly valid matmul.

## 2. PagedAttention: allocation blocks вместо монолитного запаса

Если каждой sequence заранее резервировать maximum context, много памяти пропадает. Variable-sized contiguous allocations могут фрагментироваться. Paged KV scheme хранит logical sequence blocks в доступных physical blocks с mapping metadata. Это помогает динамическим lengths, sharing и освобождению без обязательной contiguous области.

Учебный пример block16tokens: sequence длины 17 использует 2blocks, последний заполнен 1/16. В пределах block остаётся internal fragmentation15tokens, но не резервируется сразу весь maximum4096. При sharing prefix несколько sequences могут ссылаться на общие blocks с refcounts/copy-on-write policy, если implementation это поддерживает.

PagedAttention сочетает memory layout/management с kernels чтения такого cache. Она не меняет softmax objective и не позволяет игнорировать past keys без специальной attention architecture. [PagedAttention](https://arxiv.org/abs/2309.06180).

Continuous batching и paging решают разные задачи: первое scheduling, второе memory allocation/access. Prefix caching переиспользует вычисленный prefix; speculative decoding уменьшает число expensive target iterations. Наличие одного механизма не доказывает наличие остальных.

### Два вида потерянной памяти и исправление исходной схемы

Резервировать maximum context для каждого короткого request —это unused reservation. Разбить свободную память на части, из которых нельзя получить нужный contiguous allocation, —external fragmentation. Это разные причины. В S4 после завершения A показывалась одна свободная contiguous область 2GB, затем утверждалось, что новый D размером 2GB туда не помещается из-за fragmentation. Показанная схема такого вывода не доказывает. Корректный простой пример: есть две раздельные free области 30MB и 30MB, требуется один contiguous50MB buffer; суммарно 60MB достаточно, любой отдельный hole мал.

Paging выдаёт logical blocks одинаковой вместимости из общего pool. Для token position i с zero-based indexing logical block=floor(i/b), offset=i mod b. При b=16 и i=34 это block2,offset2. Если block table=[B1,B3,B7], нужные K/V лежат в physicalB7 по offset2. Logical последовательность по-прежнему упорядочена, даже если physical addresses разбросаны. Kernel должен читать её по mapping, а не случайному порядку allocator.

Для lengths(1,16,17,31) при b=16 нужно(1,1,2,2)blocks:6blocks вместимостью 96slots на 65useful tokens. Waste31slots, occupancy≈67.71%. Это internal fragmentation в последних blocks, которая не исчезла. При 100prompt tokens выделяют 7blocks на 112slots; append token101 остаётся в последнем block, а position113 требует восьмой block. Эти числа не зависят от названия модели; bytes/block затем определяются KV layout предыдущей главы.

Block table тоже не бесплатна. При 4096tokens,b=16 получаем 256entries. Если условно index занимает 8bytes, это 2048bytes=2KiB **одной такой таблицы**, до refcounts, allocator и остальных metadata. Это не доказательство универсального overhead<1%: память и kernel cost —разные измерения. Слишком маленькие blocks увеличивают mapping work, слишком крупные —waste последнего block. [PagedAttention](https://arxiv.org/abs/2309.06180) связывает placement и эффективное чтение такого cache.

### Softmax должен нормироваться по всей logical history

В псевдокоде исходника сначала собирались оценки attention из всех блоков, затем вычислялся общий softmax. Это важное условие: независимо нормировать каждый блок и просто сложить результаты неверно. Для двух однопозиционных блоков с оценками 0 и ln3 и значениями 0 и 4 общие веса равны (1/4,3/4), а выход равен 3. Локальный softmax каждого одиночного блока равен 1; сумма выходов дала бы 4, простое среднее — 2. Ни одно не равно 3.

Можно избежать материализации всего вектора оценок, сохраняя статистики каждого блока: максимум m_j, знаменатель l_j и взвешенную сумму значений u_j, рассчитанные после вычитания m_j. Для объединения:

$$
\begin{gathered}
m=\max_jm_j,\qquad
l=\sum_j e^{m_j-m}l_j,\\
u=\sum_j e^{m_j-m}u_j,\qquad o=u/l.
\end{gathered}
$$

Это алгебраический способ сохранить общий softmax при разбиении вычисления; конкретное ядро может организовать редукции иначе. Последний блок читается только в пределах фактической длины последовательности: пустые позиции не становятся дополнительными токенами с нулевыми значениями в знаменателе. В S4 `attention_weights[i]` внутри цикла по блокам также нужно понимать как **срез весов данного блока**, а не один скалярный вес токена с номером i.

### Prefix sharing, copy-on-write и branching

Общий префикс означает одинаковые вычисленные состояния для одних и тех же токенов при одинаковых модели, адаптере и позициях. Общие блоки можно хранить один раз, а таблицы последовательностей содержат ссылки на них. Счётчик ссылок не даёт освободить блок, пока его использует хотя бы один запрос. Если ветви по-разному продолжают частично заполненный общий блок, перед записью нужен отдельный блок с copy-on-write либо другой поддерживаемый способ хранить общую часть неизменяемой.

В примере S4 четыре branches имеют по 3logical blocks, первые 2 общие. Без sharing это 12physical blocks, со sharing —2common+4private=6. Экономия 50%, а не 66% из чата. Доля зависит от common prefix и divergent suffix; с ростом suffix относительная выгода уменьшается. Parallel sampling и beam search —разные способы выбирать branches, хотя sharing history может помогать обоим.

Изменение раннего system prompt не является простой записью новых bytes в cache block. В deeper layers последующие K/V уже зависели от старого prefix, поэтому нужно пересчитать затронутый suffix. CoW защищает других readers от mutation, но не делает старые downstream representations корректными для нового prompt. Одинаковый human-readable system message также недостаточен, если различаются template, special tokens или preceding context.

Prefix caching экономит повторный **prefill вычислений**, block sharing —ещё и duplication хранения, когда implementation допускает совместное использование. Это связанные, но разные выигрыши. История может быть evicted, частично совпадать либо не давать hit после смены revision. Ни 10–100×, ни всегда 99%utilization не следуют из самого названия. Paging не удаляет логическую history и не уменьшает число weights модели.

## 3. Как сравнивать engines

vLLM, TensorRT-LLM, TGI, llama.cpp/Ollama и другие systems различаются supported architectures, kernels, parallelism, scheduler/cache и interfaces. Это examples разных implementation choices; API/status/support конкретного проекта могут меняться, поэтому перед запуском нужна текущая документация и version pin.

Локальный GGUF runtime может использовать CPU, GPU и offload. Ollama/LM Studio предоставляют удобный доступ к models/API, но приложение с tools/loop обычно требует внешнего orchestrator или встроенной конкретной feature. OpenAI-compatible endpoint означает совместимость части request/response contract, а не полную идентичность structured output, tool schemas, cache или sampling semantics.

Выбирать engine по actual model, hardware, task constraints и workload. Benchmark только одного короткого prompt не определяет multi-user serving. Если модель не поддерживается backend или нужный quantization path работает через fallback, опубликованные speed numbers другого path не применимы.

### Из чего состоит runtime contract

Сначала проверяют supported model architecture и exact checkpoint revision. Затем tokenizer/chat template, dtype/quantization layout, attention/cache implementation, parallelism и нужные generation features. Наконец —API: token limits, stop handling, cancellation, streaming и error responses. Один backend может загрузить weights и отвечать на обычный prompt, но ещё не воспроизводить нужную tool schema или multimodal preprocessing. Поддержка OpenAI-compatible формы JSON не устраняет эти различия.

Из S4 сохраняем смысл разных systems без автомобильного рейтинга. vLLM сочетает kernels, memory management и scheduling; описание «просто интерпретирует PyTorch на лету» упускает compilation/CUDA-graph paths. TensorRT-LLM предлагает NVIDIA-oriented оптимизации и разные execution paths; не все сценарии сводятся к обязательному неизменному `.engine` для одного batch size. Build profiles, supported shapes и version задают ограничения конкретного path. TGI рассматривается как serving system с router/model backend и telemetry, а не как гарантия, что любое изменение потребует одного лишь Rust. llama.cpp и приложения поверх local runtimes оценивают по поддерживаемому backend и workload, а не по лозунгу «только laptop».

Не превращаем S4 в обещания «TRT всегда на 10–30%быстрее», «llama.cpp всегда проигрывает 2–5×» или «один framework первым поддержит любую архитектуру». Даже на одной GPU разный prompt length, quantization, batch и decoding могут менять ranking. SGLang/RadixAttention из чата добавляют интерес к prefix reuse и structured workloads; одно имя не доказывает ускорения вашего набора. Нужны controlled comparisons и current project docs.

### Читаем параметры запуска из чата по назначению

В [vLLM serve](https://docs.vllm.ai/en/stable/cli/serve/) context limit относится к prompt+output; число sequences и token budget iteration —разные ограничения. `gpu_memory_utilization` задаёт budget executor, а не процент занятости GPU вычислениями. `dtype=auto` не является обещанием автоматически превратить любое float checkpoint в FP8. Cache dtype и weight quantization —отдельные настройки. Остаток памяти нужен не только одному виду «CUDA overhead», поэтому 0.9 из S4 —пример настройки, не гарантия отсутствия OOM.

Учебная последовательность запуска остаётся конкретной: установить совместимую version runtime; выбрать supported artifact и зафиксировать его revision; проверить CLI help; запустить server с declared context/memory limits; отправить короткий обычный запрос; затем отдельно проверить streaming, stop и необходимые schema/tools. Offline `generate` со списком prompts проверяет другой interface, чем HTTP serving. Успех offline batch не доказывает latency под сетевой нагрузкой.

Текст S4 с `pip install vllm`, module entrypoint и self-hosted client является иллюстрацией, а не подтверждённым пользовательским запуском. Здесь не требуется скачивать LLM ради переписывания конспекта. В reviewable manifest сохраняют command/config, model/tokenizer revisions, runtime version, hardware, quantization/cache dtype и sampling. Именно этот bundle делает будущий experiment воспроизводимым, в отличие от одного названия engine.

## 4. Throughput, latency и очередь

Throughput: tokens/s или requests/s с объявленным definition. Для output tokens/s нельзя незаметно добавить input tokens и получить несравнимое число. Per-user tokens/s, aggregate output throughput и task-success throughput отвечают на разные вопросы.

TTFT, inter-token latency и end-to-end p50/p95 измерять отдельно. Увеличение batch часто повышает aggregate efficiency, но добавляет очередь/более длинный iteration. Если arrival rate превышает service capacity, queue растёт: больше batch не бесконечный источник capacity.

Для stable system Little's law связывает mean in-flight count N, arrival rate λ и mean total time W: $N\approx\lambda W$ при соответствующих assumptions. При λ=4requests/s и W=2s в среднем 8requests in system. Это не формула p95 и не обещание stability при bursty overload.

Тестировать матрицу short/long input × short/long output × concurrency. Записывать error/OOM/cancellation, warm/cold, peak memory и quality. GPU timings требуют корректной synchronization/measurement; wall-clock around async launch может измерить только enqueue.

### Разделяем clocks и denominators

Время arrival→first delivered token —client-observed TTFT. Оно включает transport/queue и серверные phases. Для L output tokens интервалы между первым и последним дают среднее TPOT=(t_last−t_first)/(L−1), если L>1. Для одного token denominator0: такую метрику не вычисляют. Некоторые reports усредняют иначе, поэтому definition сохраняют вместе с числом. Итоговый end-to-end latency дополнительно включает завершение/serialization response.

Если first token пришёл на 0.2s, last на 1.0s и всего 5tokens, TPOT=0.2s. Весь response даёт 5/1=5output tokens/s по arrival→last clock, а reciprocal TPOT=5tokens/s здесь совпал случайно из-за выбранных чисел. При first0.6,last1.0 число tokens прежнее: средний interval0.1s, reciprocal10tokens/s, но end-to-end output rate по-прежнему 5tokens/s. TTFT и decode rate нельзя заменять друг другом.

Aggregate output throughput равен сумме завершённых/доставленных output tokens за оговорённый measurement interval. Input+output tokens/s имеет другой numerator. Ещё одна полезная величина —goodput: завершённые задачи, удовлетворяющие quality и latency constraints. Можно повысить aggregate token rate, разрешив очень длинные нерелевантные ответы, но не сделать сервис полезнее.

### Почему benchmark S4 не считаем измерением

Цифры 100/2000/6000/8000tokens/s и TTFT50/80/150/250ms для разных batch sizes не сопровождались dataset, prompt/output lengths, runtime commit, sampling и процедурой. Поэтому они не подтверждают реальное испытание A100/Llama. Нельзя также сравнить sequential pipeline без batching с tuned multi-request engine и весь 60×effect назвать PagedAttention: одновременно изменено несколько факторов.

Closed-loop нагрузка держит заданное число outstanding requests: новый приходит после завершения старого. При замедлении система получает меньше новых arrivals, что может скрывать очередь. Open-loop задаёт arrivals независимо от completion и лучше показывает overload при выбранной rate. В обоих случаях нужны distribution lengths, warmup, длительность окна и доля errors/timeouts. Bursts нельзя описать только mean RPS.

Чтобы оценить batching отдельно, сначала сравнивают matching workload/config и меняют scheduler policy; для layout —matching cache semantics и attention math. Затем оценивают весь итоговый engine как систему. Не каждую ablation можно сделать доступным flag, но тогда нужно честно ограничить причинный вывод. GPU busy time, memory allocated, useful slots и delivered requests —четыре разных наблюдения.

При устойчивом arrival rate4requests/s и mean time2s Little's law даёт 8requests в системе, включая queued и обслуживаемые. Это не означает 8requests непосредственно в GPU batch. Даже при mean service capacity выше arrival rate variability может создавать длинные tails. При overload очередь растёт, а latency SLO нарушается; нужны admission limits или дополнительная capacity, а не бесконечное увеличение batch.

## 5. CV/encoder inference для контраста

Classifier/encoder обычно обрабатывает полный input одним forward; нет такого же autoregressive decode loop с накоплением KV по ответу. Dynamic batching всё равно полезен, а lengths/padding важны для text encoders. Video/recurrent models могут быть stateful, поэтому «все CV stateless» — тоже упрощение.

ONNX — representation/export ecosystem, ONNX Runtime — execution runtime. `torch.compile` — compilation path внутри PyTorch ecosystem; export требует supported graph/shapes/ops. TorchScript встречается в existing projects, но не следует назначать его универсальным актуальным default без проверки версии/статуса.

Export success не подтверждает task equivalence. Сравнить outputs/metrics после reload, representative sizes, preprocessing/postprocessing и latency. Input resize/normalization/tokenizer — часть bundle, даже если не лежит внутри graph.

### Почему contrast с CV не является запретом использовать общие engines

Обычный classifier, object detector или text embedder часто делает один model forward на input, но workload остаётся pipeline: decode bytes, resize/tokenize, normalize, transfers, model, pooling/NMS, serialization. Detection outputs бывают variable-size, input resolutions и text lengths различаются, а video trackers и recurrent networks хранят state. Поэтому S4 «все CV входы/выходы фиксированы, stateless и compute-bound» —слишком широкая формулировка.

Для encoder lengths(16,32,512) padding всех трёх до 512 даёт 1536token slots при 560useful tokens, occupancy≈36.46%. Если attention materializes dense length×length работы, relevant cost не просто число tokens. Bucketing по lengths может уменьшить waste, но ожидание подходящего bucket добавляет queue delay. Fixed embedding dimension на выходе не означает fixed compute входного текста.

LLM-serving framework с поддерживаемыми pooling/encoder paths может обслуживать embeddings, а общий inference server может размещать LLM backend. Смысл сравнения в том, какие computation/state нужно исполнять, а не в вечных запретах по бренду. Для RAG отделяют model inference от vector indexing/search: Faiss в S4 не заменяет encoder, а принимает уже подготовленные vectors.

### ONNX export и execution provider — разные проверки

ONNX описывает graph operations, tensors/initializers, shapes и opset. ONNX Runtime исполняет graph через выбранные [execution providers](https://onnxruntime.ai/docs/execution-providers/); неподдержанная операция может вызвать отказ или partition/fallback в зависимости от конфигурации. Переходы между provider partitions и CPU/GPU copies влияют на latency. Экспорт в ONNX не гарантирует автоматической совместимости с CoreML/TFLite, любым device и custom op.

Исходный export example задавал batch axis dynamic при dummy input(1,3,224,224). Это не делает H/W автоматически dynamic: такие dimensions и shape constraints задают отдельно поддерживаемым exporter. Trace-based capture может пропустить data-dependent ветку, не исполненную на example; export path и control-flow support зависят от версии. Перед переносом старого snippet смотрят актуальную документацию, а не заменяют dtype/opset вслепую.

Проверка artifact: reload в target runtime; сверка names, dtypes, layout NCHW/NHWC и preprocessing; сравнение outputs float baseline при нескольких representative shapes; затем task metrics и timings. Random tensor полезен для shape smoke test, но не подтверждает точность классификации или качество embeddings. Pre/postprocessing, labels и pooling входят в bundle даже тогда, когда graph их не содержит.

### Fusion можно проверить алгебраически

Для inference BatchNorm использует fixed running mean μ и variance v. Пусть conv output z=Wx+b, BN output γ(z−μ)/√(v+ε)+β. Для каждого output channel можно определить a=γ/√(v+ε),W'=aW,b'=a(b−μ)+β и получить тот же affine output без отдельного BN operation. Это объясняет [graph fusion](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html), а не просто утверждает «компилятор ускорит».

Например,scalar W=2,b=1,μ=3,γ=4,β=−1,√(v+ε)=2 дают W'=4,b'=−5. При x=2 исходные conv→BN и fused affine оба дают 3. Training BN с текущей batch statistics так с постоянными weights в общем случае не складывается. ReLU затем может быть fused implementation detail; из algebraic folding BN не следует, что каждая platform обязательно исполняет всё одним kernel.

`torch.compile` оптимизирует исполняемые graph regions, shapes/guards и backend определяют возможности. Dynamic shapes не обещают отсутствия recompilation. В [документации](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html) `fullgraph=True` требует capture всей функции: graph break приводит к ошибке. Поэтому сочетание S4 «fullgraph=True и всегда fallback unsupported ops» неверно. Compilation/warmup timing отделяют от steady-state. TorchScript trace/script и compiled execution также не являются одинаковыми portable artifacts; для нового export выбирают документированный под задачу путь.

### Сохраняем смысл Triton configuration из чата

Model repository, names/input contract, backend, max batch и instances —разные части server configuration. [Triton dynamic batcher](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html) группирует ожидающие compatible requests. `max_batch_size=64` не означает, что сервер всегда ждёт 64items, а `preferred_batch_size=[8,16,32]` не является универсальной оптимальной настройкой. Delay100microseconds в S4 —допустимый учебный пример budget ожидания, не measured optimum и не общий end-to-end SLO.

Две model instances на одной GPU могут конкурировать за compute и создавать дополнительные weights/workspaces. Их польза определяется overlap и workload. Архив TorchServe, Triton repository и minimal HTTP wrapper решают packaging/serving differently; historical snippets и container tags из S4 не объявляются актуальным готовым deployment. Общее требование —контроль model version, preprocessing, resource limits и output contract.

### Разбор ошибок minimal FastAPI/ONNX и batcher

В image example надо определить handling RGB/grayscale/RGBA и training-matching resize/crop/normalization. Numpy operations с Python lists могут повысить dtype сfloat32 доfloat64, поэтому итоговый dtype проверяют явно. Raw classifier logit не является confidence probability; softmax тоже не гарантирует calibration. Numpy scalar class index приводят к сериализуемому int. Четыре HTTP workers могут создать четыре runtime sessions/копии модели, а synchronous `session.run` внутри async handler может заблокировать event loop: async keyword не делает вычисление неблокирующим.

В source DynamicBatcher был отдельный sleep task на каждый запрос, а `_process_batch` забирал всю queue без cap `max_batch_size`. Также preprocess возвращал shape(1,C,H,W), и `np.stack` добавлял ещё axis: получался(B,1,C,H,W), тогда как model ожидала(B,C,H,W). Для input с собственным batch axis применяют согласованное concatenate либо возвращают unbatched CHW до stack. Это реальные ошибки contract, а не только недостаток оптимизации.

Корректная учебная схема использует bounded queue пар(input,future), один worker/таймер от oldest pending request, сбор не более max items и проверку compatible shapes. После inference outputs возвращают соответствующим futures; exceptions и cancellation тоже завершают ожидание, а не оставляют его навсегда. Разрешённый parallel inference задают отдельно, чтобы не создать неконтролируемую конкуренцию GPU sessions. Этот механизм можно описать и проверить без копирования неполного source class как production-ready кода.

Для embeddings сохраняют tokenizer, truncation, pooling, normalization, output dimension и model revision вместе с index. Если documents index созданы одной representation, а query encoder другой, одинаковая vector width ещё не гарантирует совместимость пространства. Batch encode нескольких texts ускоряет model work при suitable shape, но не заменяет правильный retrieval evaluation. Числа 2–3×/10–20× из S4 остаются гипотезами для измерения, не встроенными свойствами ONNX или batching.

## 6. Контрольные вопросы

1. Почему occupancy60% не равно GPU60%? **Ответ:** slot costs неодинаковы, есть phases, kernel efficiency и other overhead.
2. Что paging не устраняет? **Ответ:** логическую потребность хранить/read доступные K/V и compute новой позиции.
3. Достаточно ли одного aggregate tokens/s? **Ответ:** нет, нужны per-user latency, quality и errors под заданным workload.

**Практика:** нарисовать schedule lengths2,5,1,4 и refill slots новыми requests. Посчитать allocated blocks для lengths1,16,17,31 при block16. Затем составить benchmark config без фиктивных результатов: model/runtime/revision, lengths, arrival policy, concurrency, metrics и quality gate.

<!-- source-depth-inference-Г -->
