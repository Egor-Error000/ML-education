# Deployment, cold start, очереди и масштабирование

**Уровень:** основа ветки.

**Предпосылки:** [От prompt к обычному циклу генерации](<infer-generation.md>), [Prefill, decode, KV-cache, MQA и GQA](<infer-kv.md>), [Serving, batching и PagedAttention](<infer-serving.md>)

**Используют:** Выбранная прикладная задача и итоговая проверка результата.

**Учебный вопрос:** Как согласовать admission, startup и воспроизводимый rollout?





## Карта вычисления и границы этапов

![Deployment, cold start, очереди и масштабирование: входы, преобразования и выходы](<../../оформление/course/infer-deployment.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Параметры модели при обычном выполнении фиксированы. Состояние запроса, cache и состояние внешнего инструмента имеют разные жизненные циклы; их нельзя смешивать с optimizer states.

<!-- unit:7fad0635d39178eaa9 -->
<a id="u-7fad0635d39178eaa9"></a>

## Что составляет воспроизводимый deployment

<!-- /unit:7fad0635d39178eaa9 -->

<!-- unit:edfd7d4d8c0fc38ed9 -->
<a id="u-edfd7d4d8c0fc38ed9"></a>

Bundle включает model/adapter revision, tokenizer/template, runtime/dependencies, dtype/quantization, generation defaults и preprocessing/postprocessing. Container фиксирует часть environment, но не автоматически весь semantic contract. Если weights по mutable URL изменились, одинаковый Docker image ещё не означает одинаковую модель.

<!-- /unit:edfd7d4d8c0fc38ed9 -->

<!-- unit:5e415e0fba6615f9c3 -->
<a id="u-5e415e0fba6615f9c3"></a>

Weights можно включить в image или доставлять отдельным versioned artifact/volume/cache. Первый путь упрощает immutability bundle, но увеличивает image transfer/storage; второй позволяет reuse и отдельные releases, но требует проверки version/hash и доступности. Ни один путь не запрещён сам по себе.

<!-- /unit:5e415e0fba6615f9c3 -->

<!-- unit:6205cc9fcb7418e04e -->
<a id="u-6205cc9fcb7418e04e"></a>

В layer caching сначала обычно устанавливают редко меняющиеся pinned dependencies, затем копируют часто меняющийся code. Multi-stage build может убрать build dependencies, но сам по себе не делает любое изображение маленьким. В исходном примере `COPY .` до dependency installation действительно создавал нежелательную cache invalidation при code changes; это конкретная причина, а не «каждая пересборка всегда заново загружает все слои».

<!-- /unit:6205cc9fcb7418e04e -->

<!-- unit:3efd964b43ded4382f -->
<a id="u-3efd964b43ded4382f"></a>

**Схема курса:** artifact load → warmup/readiness → request queue/admission → scheduler and batch → inference → telemetry → scale, rollback or reject overload

<!-- /unit:3efd964b43ded4382f -->

<!-- unit:531718ad7efc693fab -->
<a id="u-531718ad7efc693fab"></a>

<!-- source-depth-inference-И -->

<!-- /unit:531718ad7efc693fab -->

<!-- unit:e4194e21ed14bf2f6f -->
<a id="u-e4194e21ed14bf2f6f"></a>

### Воспроизводимость: образ, артефакт и функция ответа

<!-- /unit:e4194e21ed14bf2f6f -->

<!-- unit:8cc1bc38a1408a2532 -->
<a id="u-8cc1bc38a1408a2532"></a>

В S6 рассматривается container с model weights, затем предлагается вынести weights в volume. Оба варианта допустимы. Различие не в том, «настоящий» ли это inference, а в том, какие компоненты обновляются вместе. Image может содержать runtime и код, model artifact — weights/tokenizer, а deployment manifest — выбранные revisions и параметры запуска. Воспроизводимым должен быть их согласованный набор.

<!-- /unit:8cc1bc38a1408a2532 -->

<!-- unit:8b0183aceb1025276b -->
<a id="u-8b0183aceb1025276b"></a>

Например, одинаковый image запускается с model v1 и v2 на внешнем volume. Это два разных поведения, хотя container digest совпал. Обратный случай: weights прежние, но изменился tokenizer или chat template — модель получает другие IDs или структуру сообщений. Поэтому «мы не меняли веса» не доказывает неизменность прикладного ответа.

<!-- /unit:8b0183aceb1025276b -->

<!-- unit:886e90fa2929e1d61c -->
<a id="u-886e90fa2929e1d61c"></a>

Артефакт фиксируют по immutable revision/hash, а не только по mutable `latest`. Во время загрузки не следует представлять наполовину скачанный набор shards как готовую модель: сначала проверяют полноту и публикуют завершённую версию. Это особенно важно, если несколько реплик одновременно используют общий cache. Сам volume не задаёт атомарность загрузки или совместимость файлов.

<!-- /unit:886e90fa2929e1d61c -->

<!-- unit:85ae838d71231371e4 -->
<a id="u-85ae838d71231371e4"></a>

### Почему порядок Docker layers важнее числа FROM

<!-- /unit:85ae838d71231371e4 -->

<!-- unit:d4f72cea52be7c8909 -->
<a id="u-d4f72cea52be7c8909"></a>

В исходном Dockerfile сначала выполняется `COPY .`, затем установка Python dependencies. Изменение application code меняет вход этого слоя и последующих: дорогая установка может повториться. Если сначала скопировать dependency manifest, установить зависимости, а затем добавить code, изменение только code не обязано инвалидировать dependency layer. Это конкретный механизм reuse, а не правило «Docker всегда пересобирает всё».

<!-- /unit:d4f72cea52be7c8909 -->

<!-- unit:aa444ed984025168cc -->
<a id="u-aa444ed984025168cc"></a>

Несколько `FROM` действительно создают stages. Но если финальный stage начинается `FROM dependencies`, он наследует содержимое dependencies. Чтобы оставить build tools вне финального образа, обычно выбирают отдельную runtime основу и копируют нужные результаты из builder. Само переименование трёх последовательно наследуемых stages не уменьшает их содержимое. Механизм selective copying объяснён в [Docker multi-stage builds](https://docs.docker.com/build/building/multi-stage/).

<!-- /unit:aa444ed984025168cc -->

<!-- unit:f5d67986cfef1c83dc -->
<a id="u-f5d67986cfef1c83dc"></a>

Точный CUDA/Python/runtime набор сверяют для выбранного engine. Показанные в разбираемом примере version tags — исторические примеры, а не готовый рецепт для любой машины. Если `HEALTHCHECK` вызывает curl, утилита должна присутствовать в образе. Если snippet объявлен как полный Deployment, нужны также обязательные selectors/labels и согласованные volume definitions: фрагменты S6 опускали часть этих полей и не должны выдаваться за проверенный полный manifest.

<!-- /unit:f5d67986cfef1c83dc -->

<!-- unit:71fcc545e3b0af7e19 -->
<a id="u-71fcc545e3b0af7e19"></a>

### Host port и container port: ошибка исходной Compose-схемы

<!-- /unit:71fcc545e3b0af7e19 -->

<!-- unit:84cd78b412abbaf21f -->
<a id="u-84cd78b412abbaf21f"></a>

В S6 указано `8001:8000` для inference server и `6333:8000` для vector service. Это отображения host→container. Внутри общей service network другой container обычно обращается к service name и **container port**, а не к опубликованному host port:

<!-- /unit:84cd78b412abbaf21f -->

<!-- unit:fb1ff6cd9d1b03dbe4 -->
<a id="u-fb1ff6cd9d1b03dbe4"></a>

```text
host → localhost:8001 → llm-server:8000
agent в service network → llm-server:8000
agent в той же сети → vector-db:8000
```

<!-- /unit:fb1ff6cd9d1b03dbe4 -->

<!-- unit:58bb5b0c79ed4ffa59 -->
<a id="u-58bb5b0c79ed4ffa59"></a>

Поэтому адреса `llm-server:8001` и `vector-db:6333` в исходном agent environment не следуют из показанных port mappings. Monitoring внутри этой сети также должен использовать доступный container endpoint. Назначение host port 6333 не превращает Chroma API в Qdrant API: protocol определяется самим сервисом.

<!-- /unit:58bb5b0c79ed4ffa59 -->

<!-- unit:c54359a4e8c910faaf -->
<a id="u-c54359a4e8c910faaf"></a>

Переменная `MODEL_PATH` влияет на engine только если launcher действительно её читает. Mount готовых weights не задаёт model автоматически. Аналогично порядок запуска зависимостей не равнозначен их readiness: process может существовать, пока модель ещё загружается. Эти связи нужно прослеживать явно, а не считать property любого Compose файла.

<!-- /unit:c54359a4e8c910faaf -->

<!-- unit:ab93176fce46464e92 -->
<a id="u-ab93176fce46464e92"></a>

## Cold start по стадиям

<!-- /unit:ab93176fce46464e92 -->

<!-- unit:8c9f1612d70b0e0df1 -->
<a id="u-8c9f1612d70b0e0df1"></a>

Возможные stages: scheduler получает hardware → image/artifact fetch → process imports → weights load/partition → device transfer → kernel compilation/init → warmup → ready. Local disk weights экономят download, но не обязательно GPU load и warmup.

<!-- /unit:8c9f1612d70b0e0df1 -->

<!-- unit:45f864b5b2188d35c2 -->
<a id="u-45f864b5b2188d35c2"></a>

**Startup check** учитывает долгую initialisation; **readiness** означает способность обслуживать новые requests; **liveness** — необходимость restart при неисправности. Если использовать слишком агрессивный liveness во время normal long startup, можно получить restart loop. Если объявить ready до loaded weights, traffic столкнётся с errors/latency.

<!-- /unit:45f864b5b2188d35c2 -->

<!-- unit:1fdb95e7f038d886ec -->
<a id="u-1fdb95e7f038d886ec"></a>

Warmup должен отражать supported shapes/path: один short prompt не гарантирует прогрев всех kernels для long context/batch или разных adapters. Измерять cold и warm отдельно, не прятать first requests из отчёта без указания.

<!-- /unit:1fdb95e7f038d886ec -->

<!-- unit:eb3929e9d75e2c7210 -->
<a id="u-eb3929e9d75e2c7210"></a>

Для same-version replicas weights могут быть одинаковыми; routing между ними допустим, если application state хранится совместимо. KV-cache locality делает выгодным sticky/prefix-aware routing, но потеря replica не должна означать потерю единственной копии пользовательского workflow, если требуется recovery.

<!-- /unit:eb3929e9d75e2c7210 -->

<!-- unit:1fc3b64b8f6fb11e60 -->
<a id="u-1fc3b64b8f6fb11e60"></a>

### Где именно экономится cold start

<!-- /unit:1fc3b64b8f6fb11e60 -->

<!-- unit:7414d94b7deeade4fa -->
<a id="u-7414d94b7deeade4fa"></a>

Разложим учебный запуск на последовательные стадии: process/imports 4s, download 12s, CPU load 18s, device transfer 10s, compilation/init 6s, warmup 8s. Сумма 58s. Если weights уже в artifact cache и download полностью пропущен, остаётся 46s. Это специально выбранные числа для понимания timeline, не измерение vLLM или определённого GPU.

<!-- /unit:7414d94b7deeade4fa -->

<!-- unit:4ea56dcb934b92f06b -->
<a id="u-4ea56dcb934b92f06b"></a>

В реальной системе стадии могут перекрываться, поэтому нельзя всегда просто сложить все отдельные maxima. Но их timestamps показывают, что именно ускорилось. Local disk/PVC устраняет или уменьшает доставку файлов; он не отменяет копирование в device memory и инициализацию процесса. Общий network volume также может стать bottleneck при одновременном чтении многими репликами.

<!-- /unit:4ea56dcb934b92f06b -->

<!-- unit:ddd77d15eeaa6bb6f6 -->
<a id="u-ddd77d15eeaa6bb6f6"></a>

Init container может заранее скачать и проверить model artifact. Main container стартует после него и видит файлы, но ещё должен создать engine, загрузить weights и выполнить нужный warmup. Время init входит в общее ожидание готовой реплики. Перенос download в другой container не делает эту работу мгновенной и не сохраняет CUDA state основного процесса между рестартами.

<!-- /unit:ddd77d15eeaa6bb6f6 -->

<!-- unit:2d50f64f6eb0d9a97a -->
<a id="u-2d50f64f6eb0d9a97a"></a>

### Почему save/load weights не означает «сохранили hot GPU»

<!-- /unit:2d50f64f6eb0d9a97a -->

<!-- unit:0764647c8a5b737cfa -->
<a id="u-0764647c8a5b737cfa"></a>

В S6 предложен `torch.save(llm.state_dict())` как сохранение загруженной модели между рестартами, а затем сказано, что CUDA tensors нельзя просто сериализовать. Для обычного PyTorch module tensors можно сохранять и загружать, в том числе с переносом устройства; это описано в [PyTorch save/load across devices](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-across-devices). Но сохранение данных не сохраняет живой process, allocator, kernel state или готовый engine.

<!-- /unit:0764647c8a5b737cfa -->

<!-- unit:700dadf72f0138a3ee -->
<a id="u-700dadf72f0138a3ee"></a>

Кроме того, high-level serving engine не обязан иметь тот же `state_dict/load_state_dict` interface, что `nn.Module`. Его artifact loading следует проверять по documented API выбранной версии. В исходной ветви load объект `llm` ещё и не создавался до обращения к нему. Этот snippet нельзя считать работающим механизмом warm restart.

<!-- /unit:700dadf72f0138a3ee -->

<!-- unit:bbf76ba4e5670f7fcf -->
<a id="u-bbf76ba4e5670f7fcf"></a>

У artifact cache, compilation cache и KV-cache разные lifetime и contents. Artifact cache хранит файлы модели; compilation cache — результаты подготовки kernels при поддержке backend; KV-cache — состояния конкретных вычислений и префиксов. Наличие одного не доказывает наличие двух других. После потери процесса поведение restore определяется каждым уровнем отдельно.

<!-- /unit:bbf76ba4e5670f7fcf -->

<!-- unit:7e0d62d1a81e750cca -->
<a id="u-7e0d62d1a81e750cca"></a>

### Warmup, readiness и liveness решают разные вопросы

<!-- /unit:7e0d62d1a81e750cca -->

<!-- unit:7c52f999b34881045b -->
<a id="u-7c52f999b34881045b"></a>

Warmup выполняется до объявления готовности для нужных execution paths. Три вызова короткого `Hello` могут прогреть один путь и не затронуть long prefill, большой batch, другие adapters или shapes. Это не обязательно плохо: важно объявить coverage warmup и затем измерить первые запросы остальных типов. В исходном class `sampling_params`, созданный локально в `_warmup`, недоступен в другом method без передачи или сохранения — ещё одна причина считать код иллюстрацией, а не готовым server.

<!-- /unit:7c52f999b34881045b -->

<!-- unit:cac89e1fc382592abc -->
<a id="u-cac89e1fc382592abc"></a>

Startup probe защищает длительную инициализацию; до её успеха обычные liveness/readiness проверки не начинаются по соответствующему Kubernetes contract. Readiness failure убирает готовность к traffic, liveness/startup failure может вызвать restart после заданных условий. Эти механизмы отдельно описаны в [Kubernetes probes](https://kubernetes.io/docs/concepts/workloads/pods/probes/). Dockerfile `HEALTHCHECK` сам по себе не является определением трёх probes в Pod spec.

<!-- /unit:cac89e1fc382592abc -->

<!-- unit:6b14f047277ea46a79 -->
<a id="u-6b14f047277ea46a79"></a>

У проверки «free GPU memory≥2GB» из S6 нет универсального смысла health. Engine мог заранее зарезервировать cache, оставаясь здоровым. Недостаток ресурса для **нового** запроса относится к admission/readiness policy, а не автоматически к неисправности и restart. Точно так же `hasattr(llm,'model')` не является универсальным контрактом готовности любой engine wrapper.

<!-- /unit:6b14f047277ea46a79 -->

<!-- unit:f9df4aafa35c02f027 -->
<a id="u-f9df4aafa35c02f027"></a>

Тестовый inference на каждую частую probe добавляет нагрузку и может ждать в той же очереди. Если overload вызывает failing probes, а restart уменьшает capacity, возникает feedback loop. Поэтому состояние процесса, готовность model path и периодическая функциональная проверка должны иметь явные значения и частоту. Один endpoint, всегда отвечающий OK, не доказывает исправность GPU, но тяжёлая генерация тоже не обязана быть правильной liveness проверкой.

<!-- /unit:f9df4aafa35c02f027 -->

<!-- unit:c0fe746bb0e730b2a8 -->
<a id="u-c0fe746bb0e730b2a8"></a>

### Rolling update не обязательно равен N×cold_start

<!-- /unit:c0fe746bb0e730b2a8 -->

<!-- unit:8ae1bf5a994f5e71ed -->
<a id="u-8ae1bf5a994f5e71ed"></a>

В разбираемом примере 10 реплик с запуском по 2–5 минут превращались в 20–50 минут обновления. Это верно только для конкретного последовательного расписания без перекрытия. Если в учебном примере 10 реплик заменяются волнами по 2, а каждая волна занимает 120s до readiness, пять волн занимают 600s; по одной —1200s. Дополнительно влияют drain, quotas, доступные GPU и стратегия rollout.

<!-- /unit:8ae1bf5a994f5e71ed -->

<!-- unit:eb0017f16f8443c1aa -->
<a id="u-eb0017f16f8443c1aa"></a>

Если для дополнительной warm replica нет ресурса, capacity может временно уменьшиться при остановке старой. Если новая версия использует другой объём памяти, старые assumptions о совместном размещении могут перестать работать. Поэтому cold-start timeline включают в план обновления, а не только в отдельный startup benchmark.

<!-- /unit:eb0017f16f8443c1aa -->

<!-- unit:b55349e84cac33376c -->
<a id="u-b55349e84cac33376c"></a>

## Admission control и queue

<!-- /unit:b55349e84cac33376c -->

<!-- unit:81105de7d045f7b5ce -->
<a id="u-81105de7d045f7b5ce"></a>

Каждый request занимает разные memory/compute: prompt100/output50 и prompt20k/output2k не равны. Admission policy учитывает current tokens/cache, max output budget, concurrency и queue deadline. Нельзя бесконечно принимать работу, когда memory/capacity уже исчерпаны.

<!-- /unit:81105de7d045f7b5ce -->

<!-- unit:7a5a602733dd4b6bd9 -->
<a id="u-7a5a602733dd4b6bd9"></a>

Backpressure означает bounded queue/ограничение новых requests или явный overload response. Cancellation должна по возможности освобождать engine state и downstream work; закрытое клиентское соединение само по себе не доказывает остановку GPU generation.

<!-- /unit:7a5a602733dd4b6bd9 -->

<!-- unit:1896d4bc5ba0be0c41 -->
<a id="u-1896d4bc5ba0be0c41"></a>

Timeout разбивается на phases: waiting, model, tool, total deadline. Если каждый retry получает полный новый timeout без общего deadline, пользователь может ждать гораздо дольше заявленного budget. Для retries нужны caps, backoff/jitter по задаче и distinction transient/permanent errors.

<!-- /unit:1896d4bc5ba0be0c41 -->

<!-- unit:c48bb21db0d4501de1 -->
<a id="u-c48bb21db0d4501de1"></a>

Пример: service устойчиво завершает 2requests/s данного workload, приходит 3requests/s. Без отказов/новой capacity queue растёт примерно 1request/s после transient effects. Через минуту добавилось около 60. Никакой красивый average GPU percentage не отменяет conservation of arrivals/completions.

<!-- /unit:c48bb21db0d4501de1 -->

<!-- unit:47b60095f3bc1e8002 -->
<a id="u-47b60095f3bc1e8002"></a>

### Считать не только requests, но и резервируемую работу

<!-- /unit:47b60095f3bc1e8002 -->

<!-- unit:662e0e31f327f67c51 -->
<a id="u-662e0e31f327f67c51"></a>

У decoder request с prompt1000 и max_new_tokens512 может потребоваться резерв под 1512 позиций; точная cache boundary зависит от этапа generation, а такой резерв консервативен. При blocks16 это 95 blocks, вместимость 1520 tokens. Если одна cache position занимает 128KiB, резерв равен 190MiB. При 8GiB свободного cache budget теоретически помещаются 43 таких requests, без учёта других ограничений и изменений нагрузки.

<!-- /unit:662e0e31f327f67c51 -->

<!-- unit:4bbefb1a07f85b061b -->
<a id="u-4bbefb1a07f85b061b"></a>

Эта арифметика не предсказывает latency. Один и тот же cache footprint может соответствовать разным prefill/decode стадиям и числу вычислений. Admission должен учитывать и memory, и execution budgets. `max_num_seqs` задаёт ограничение concurrency в выбранном engine, но фактическая batch composition зависит также от token budget, lengths и scheduler. Максимально разрешённый batch не означает, что каждый шаг действительно исполняется с таким batch.

<!-- /unit:4bbefb1a07f85b061b -->

<!-- unit:016ceb0227039b3bdd -->
<a id="u-016ceb0227039b3bdd"></a>

Присваивать всем «ещё 512 output tokens» не всегда обязательно: engine может использовать другую reservation/preemption policy. Но тогда нужно понимать, что случится, если активные requests действительно вырастут: wait, recompute, explicit failure или иной документированный путь. Нельзя обещать отсутствие OOM только из среднего output length.

<!-- /unit:016ceb0227039b3bdd -->

<!-- unit:e3ac2c11969d45a228 -->
<a id="u-e3ac2c11969d45a228"></a>

### Очередь, running state и terminal status

<!-- /unit:e3ac2c11969d45a228 -->

<!-- unit:3031a6743620609f11 -->
<a id="u-3031a6743620609f11"></a>

Для числа принятых незавершённых requests N, включая waiting и running, действует bookkeeping:

<!-- /unit:3031a6743620609f11 -->

<!-- unit:36540cabbfa1337fcf -->
<a id="u-36540cabbfa1337fcf"></a>

$$
N_{t+1}=N_t+A_t-D_t-X_t,
$$

<!-- /unit:36540cabbfa1337fcf -->

<!-- unit:4ab968dd4156caa8f1 -->
<a id="u-4ab968dd4156caa8f1"></a>

где A — новые admissions, D — завершения, X — отмены/другие terminal exits, не включённые в D. Эти множества должны быть определены без двойного учёта. Waiting queue Q имеет другие departures: request покидает её уже при запуске, до завершения generation. Метрика queue не должна молча означать waiting+running.

<!-- /unit:4ab968dd4156caa8f1 -->

<!-- unit:2a6fb8c4091d1ac54a -->
<a id="u-2a6fb8c4091d1ac54a"></a>

В S6 endpoint сначала ставит request в queue, затем напрямую вызывает processing, не показывая consumer, который извлекает запись. Так queue entries остаются, хотя реальное выполнение уже идёт по другому пути. Учебный producer–worker contract выглядит иначе:

<!-- /unit:2a6fb8c4091d1ac54a -->

<!-- unit:f4e236a2ba808cd5d1 -->
<a id="u-f4e236a2ba808cd5d1"></a>

```text
handler: проверить admission и общий deadline
handler: создать request ID + future результата, передать задание queue
handler: ждать future или cancellation/deadline
worker: извлечь допустимую группу заданий
worker/engine: выполнить forward/generation, вернуть результаты по IDs
terminal transition: освободить reservations ровно один раз
```

<!-- /unit:f4e236a2ba808cd5d1 -->

<!-- unit:f0807ce9a65d46a9ca -->
<a id="u-f0807ce9a65d46a9ca"></a>

При error/cancellation также нужен terminal transition. Gauge, увеличенный до успешного blocking put, может включить ещё не помещённые задания; если put отменился, значение требует корректного восстановления. Это не проблема самого Counter/Gauge, а несогласованного определения жизненного цикла.

<!-- /unit:f0807ce9a65d46a9ca -->

<!-- unit:2b47dc98b21cacfc82 -->
<a id="u-2b47dc98b21cacfc82"></a>

Закрытие client connection не доказывает, что engine остановил generation. Пока отмена не обработана, request может занимать GPU/cache. Если response уже частично отправлен, завершение должно отличать successful answer, truncated result и failure. Для mutable tool effects дополнительно нужны idempotency и сверка — это уровень orchestration, а не автоматическое свойство inference queue.

<!-- /unit:2b47dc98b21cacfc82 -->

<!-- unit:a2c8eb0007ee16f6b6 -->
<a id="u-a2c8eb0007ee16f6b6"></a>

### Общий deadline ограничивает retries

<!-- /unit:a2c8eb0007ee16f6b6 -->

<!-- unit:7537843c74a2778e59 -->
<a id="u-7537843c74a2778e59"></a>

Пусть end-to-end budget10s. Queue wait занял 4s, подготовка/transport ещё 1s; на дальнейшее выполнение осталось 5s. Выдавать engine новый полный timeout10s означало бы нарушить исходный budget. Retry также тратит remaining budget и может увеличить нагрузку, особенно если прежняя попытка всё ещё вычисляется.

<!-- /unit:7537843c74a2778e59 -->

<!-- unit:4e476a29b3b2dd9b8d -->
<a id="u-4e476a29b3b2dd9b8d"></a>

Отказ admission при исчерпанной capacity делает нагрузку явной. Бесконечное накопление waiting requests не создаёт throughput, а переносит проблему в latency и memory. Policy rejection, retry-after и fallback задаются вместе с задачей; ограничение очереди без понимания отмен и повторов не образует полный overload protocol.

<!-- /unit:4e476a29b3b2dd9b8d -->

<!-- unit:81fdaa0c5835ff6c00 -->
<a id="u-81fdaa0c5835ff6c00"></a>

### Почему throughput не позволяет вычислить mean latency

<!-- /unit:81fdaa0c5835ff6c00 -->

<!-- unit:d3288bef9336c20f02 -->
<a id="u-d3288bef9336c20f02"></a>

Для 100 requests, пришедших одновременно,10 batches по 10 с длительностью 0.2s дают completions0.2,0.4,…,2.0s, по 10 requests на каждую отметку. Throughput100/2=50req/s. Mean request latency=1.1s, поскольку каждый ждёт свою волну. Формула S6 `total_time/N*batch_size` даёт 0.2s — здесь это длительность batch, а не средняя задержка.

<!-- /unit:d3288bef9336c20f02 -->

<!-- unit:b1070fc8a8ca950f2e -->
<a id="u-b1070fc8a8ca950f2e"></a>

Внутри benchmark сохраняют arrival, enqueue/start и completion каждого request. TTFT включает путь до первого output token; inter-token intervals описывают streaming после него; end-to-end — завершение. Их нельзя заменить единственным model wall time. Wrapper `async def` не делает blocking `generate()` неблокирующим, если вызов не передан в подходящий execution interface.

<!-- /unit:b1070fc8a8ca950f2e -->

<!-- unit:efdfa11eb73b9dd5c5 -->
<a id="u-efdfa11eb73b9dd5c5"></a>

В исходном benchmark model создаётся заново для каждого выбранного batch limit, но timer запускается после создания; часть cold start исключается, часть первого warm execution может остаться. Сто одинаковых prompts также могут иначе взаимодействовать с cache, чем реальная смесь. Для честного сравнения отдельно фиксируют warmup, arrivals, lengths, sampling/output budget, cache state и engine configuration. Числа таблицы «7B/A100» в разбираемом примере не являются проведённым benchmark.

<!-- /unit:efdfa11eb73b9dd5c5 -->

<!-- unit:0fcb51dc8582e74c59 -->
<a id="u-0fcb51dc8582e74c59"></a>

## Autoscaling с учётом startup delay

<!-- /unit:0fcb51dc8582e74c59 -->

<!-- unit:c4a3e9ab5ae31c973f -->
<a id="u-c4a3e9ab5ae31c973f"></a>

Signals: queue age/length, active sequences/tokens, TTFT/inter-token/end-to-end percentiles, throughput, memory headroom, errors и utilisation. CPU/GPU metrics полезны для diagnosis, но не являются достаточным выражением user SLO. Высокий utilisation при пустой queue не обязательно требует ещё replica, а низкий utilisation не доказывает отсутствие другого bottleneck.

<!-- /unit:c4a3e9ab5ae31c973f -->

<!-- unit:e61a21b3b56729b451 -->
<a id="u-e61a21b3b56729b451"></a>

Scaling реагирует с задержкой: пока новая replica загружается, очередь может вырасти. Нужны запас capacity/min ready replicas по expected bursts, warming policy и stabilisation/hysteresis, чтобы избежать oscillation. Одинаковый threshold для любой модели/instance не обоснован.

<!-- /unit:e61a21b3b56729b451 -->

<!-- unit:6a7f999db49e53f320 -->
<a id="u-6a7f999db49e53f320"></a>

Scale-in должен drain existing work и корректно завершать stream/cache state. Просто убить pod с длинными запросами создаёт errors/retries и может усилить нагрузку. Max replicas/cost limits и overload policy должны согласоваться: если больше hardware недоступно, система должна ограничить admission.

<!-- /unit:6a7f999db49e53f320 -->

<!-- unit:0f5815a1160f904eb0 -->
<a id="u-0f5815a1160f904eb0"></a>

### Рост очереди во время подготовки новой capacity

<!-- /unit:0f5815a1160f904eb0 -->

<!-- unit:3caad39c429407ca4e -->
<a id="u-3caad39c429407ca4e"></a>

Продолжим source пример: приходит 3 requests/s, одна готовая replica завершает 2/s на данном однородном workload. В fluid approximation backlog растёт на 1/s. Если новая независимая replica такой же capacity готовится 40s, к её готовности накопится около 40 requests. После запуска общий service rate4/s, при arrivals3/s backlog убывает на 1/s и требует ещё около 40s. Реакция «создали pod» не равна немедленному исчезновению очереди.

<!-- /unit:3caad39c429407ca4e -->

<!-- unit:8229eefb2f4b3c81f7 -->
<a id="u-8229eefb2f4b3c81f7"></a>

Это пример с постоянными rates, отсутствием отказов и независимой добавленной capacity. Если replicas делят один занятый GPU, суммарная производительность может не удвоиться. Если arrival burst закончился раньше readiness, добавленная capacity может уже не понадобиться. Поэтому нужны временная структура нагрузки, startup distribution и ограничения ресурсов.

<!-- /unit:8229eefb2f4b3c81f7 -->

<!-- unit:407318077131790154 -->
<a id="u-407318077131790154"></a>

Номинальное 50requests/s при 10/s на replica даёт пять реплик только без запаса относительно указанной capacity. Для учебного target utilizationρ=0.75 и arrivals30/s при 10/s на replica получаем $\lceil30/(10\cdot0.75)\rceil=4$. Такая арифметика зависит от одинакового workload и не заменяет tail-latency test при случайных arrivals и lengths.

<!-- /unit:407318077131790154 -->

<!-- unit:7e99d7941fe8b2e969 -->
<a id="u-7e99d7941fe8b2e969"></a>

### Что HPA делает с metric, а что остаётся вне формулы

<!-- /unit:7e99d7941fe8b2e969 -->

<!-- unit:8a888e3da6afe49724 -->
<a id="u-8a888e3da6afe49724"></a>

В базовом приближении HPA использует отношение текущей средней metric к target: desired replicas≈ceil(current replicas×current metric/target). Например, две replicas со средней queue12 при target5 дают ceil(2×12/5)=5. Реальный controller также учитывает доступность metrics, readiness, policies и stabilization; это описано в [HPA algorithm](https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/#algorithm-details).

<!-- /unit:8a888e3da6afe49724 -->

<!-- unit:4d015e66900189722e -->
<a id="u-4d015e66900189722e"></a>

Custom queue/GPU signals нужно доставить в соответствующий metrics API; одно произвольное имя в YAML не создаёт exporter/adapter. Выделенный GPU resource и GPU utilisation — разные данные. `nvidia.com/gpu:1` обычно выражает allocation устройства в ресурсном contract, а не процент его загрузки. CPU/memory metrics и custom metrics также имеют разные источники и определения.

<!-- /unit:4d015e66900189722e -->

<!-- unit:414fbe5271db80c1a4 -->
<a id="u-414fbe5271db80c1a4"></a>

Stabilization window — обработка истории рекомендаций по policy, а не универсальная команда «после превышения threshold спать ровно 60 секунд». Rate limits ограничивают изменение replicas за интервал. Pending replicas ещё не дают service capacity. Чтобы понять результат, надо проследить все стадии controller→scheduler→artifact load→readiness→traffic.

<!-- /unit:414fbe5271db80c1a4 -->

<!-- unit:162d455e7babd7ae16 -->
<a id="u-162d455e7babd7ae16"></a>

### Почему utilisation полезна, но не является SLO

<!-- /unit:162d455e7babd7ae16 -->

<!-- unit:d28ac1228daa4b9df5 -->
<a id="u-d28ac1228daa4b9df5"></a>

Высокий GPU utilisation с пустой waiting queue может означать эффективную полезную работу. Низкий GPU utilisation при большой queue может быть связан с CPU tokenization, serialization, transport, locks или неэффективным admission. Источник утверждал, что CPU почти не используется и scaling по utilisation «не работает» вообще; это слишком категорично. Метрика помогает искать bottleneck, но её одной недостаточно для user latency и решения о новой replica.

<!-- /unit:d28ac1228daa4b9df5 -->

<!-- unit:ccec1e90253490acb4 -->
<a id="u-ccec1e90253490acb4"></a>

Queue length тоже не независима от работы: десять requests по 32 tokens и десять по 1024 имеют разный compute. Для dense attention квадратичная часть отличается в 1024 раза при прочих фиксированных размерах, линейные части — в 32. Поэтому вместе с queue count измеряют age, lengths/tokens и active work. Cache-block utilisation имеет свой denominator и не равна автоматически всей GPU VRAM usage.

<!-- /unit:ccec1e90253490acb4 -->

<!-- unit:cb27c25ea78cd29669 -->
<a id="u-cb27c25ea78cd29669"></a>

Уже зарезервированная cache memory может оставаться почти постоянной при меняющемся числе requests. Добавление replicas по высокой memory percentage не гарантирует исправления OOM активного запроса. Сначала должны работать корректные per-request limits и admission, затем controller добавляет capacity по объявленной policy.

<!-- /unit:cb27c25ea78cd29669 -->

<!-- unit:3068c4a1b1319a18fc -->
<a id="u-3068c4a1b1319a18fc"></a>

### Percentile нельзя усреднять как latency samples

<!-- /unit:3068c4a1b1319a18fc -->

<!-- unit:d757971cce59519b07 -->
<a id="u-d757971cce59519b07"></a>

В source списке [100,100,100,100,5000]ms среднее 1080ms, но медленный request составляет 20%, а не 5%. Nearest-rank p95 равен 5000ms; линейная interpolation quantile даёт 4020ms. Для пяти samples определение percentile заметно влияет на число. Само имя p95 не означает, что ровно 5% пользователей в этой маленькой выборке ждали 5s.

<!-- /unit:d757971cce59519b07 -->

<!-- unit:77bd15c5441f4bdecf -->
<a id="u-77bd15c5441f4bdecf"></a>

У pod A сто requests по 100ms, у pod B один по 5000ms. Среднее двух pod p95 равно 2550ms, а nearest-rank p95 объединённых 101 requests —100ms. Поэтому average pod quantiles не восстанавливает global quantile. Для общего tail analysis объединяют корректные request samples или согласованные histogram counts, понимая точность bins.

<!-- /unit:77bd15c5441f4bdecf -->

<!-- unit:65c4fcfa6a6ace04d6 -->
<a id="u-65c4fcfa6a6ace04d6"></a>

Если HPA использует average отдельной per-pod metric, это может быть осознанным control signal, но не тем же самым, что global p95 SLO. Единицы тоже должны совпадать: metric в seconds и target «3000» не означают 3 секунды без объявленного пересчёта из milliseconds. Название metric не задаёт её unit автоматически.

<!-- /unit:65c4fcfa6a6ace04d6 -->

<!-- unit:97a2319b588ca2ee92 -->
<a id="u-97a2319b588ca2ee92"></a>

## Sharing и parallelism

<!-- /unit:97a2319b588ca2ee92 -->

<!-- unit:7334709a165bb6092f -->
<a id="u-7334709a165bb6092f"></a>

Batching нескольких запросов одной model, несколько разных models на GPU, tensor parallel и data parallel решают разные задачи. Tensor parallel распределяет один model computation между devices и добавляет communication. Data parallel replicas увеличивают independent request capacity, если каждая помещается в hardware. Pipeline parallel добавляет stages и scheduling trade-offs.

<!-- /unit:7334709a165bb6092f -->

<!-- unit:86a1c1a24d5aef82c0 -->
<a id="u-86a1c1a24d5aef82c0"></a>

Несколько tenants/adapters одной base могут экономить weights, но требуют isolation, routing и compatible batching. GPU sharing разных processes может конкурировать за memory/bandwidth; «два процесса на одном GPU» не гарантирует doubled capacity. Benchmark всего workload обязательнее отдельных microbenchmarks.

<!-- /unit:86a1c1a24d5aef82c0 -->

<!-- unit:eefea3eb65abe2ba84 -->
<a id="u-eefea3eb65abe2ba84"></a>

### Weights, cache и runtime budget занимают разные ресурсы

<!-- /unit:eefea3eb65abe2ba84 -->

<!-- unit:c17721bc2e7fe328b9 -->
<a id="u-c17721bc2e7fe328b9"></a>

Семь миллиардов параметров по 2 bytes —14 billion bytes raw weights, около 13.04GiB. Семьдесят миллиардов —140 billion bytes, около 130.39GiB. Это арифметика условных counts/dtype, не точные размеры каждого checkpoint. Свободное место после weight load ещё расходуется на KV-cache, activations, temporary buffers и runtime. Из «weights7B≈14GB» не следует, что вся оставшаяся память и compute бесполезно простаивают.

<!-- /unit:c17721bc2e7fe328b9 -->

<!-- unit:1f0bb3b85c64075e46 -->
<a id="u-1f0bb3b85c64075e46"></a>

Обычный GPU KV-cache не обязан находиться в host RAM, как можно понять из формулировки S6; memory placement зависит от engine/offload policy. CPU RAM, GPU VRAM и disk cache считают раздельно. Container `memory` resource в Kubernetes не является универсальным лимитом GPU VRAM. Резерв device resources и host memory описывают разные ограничения.

<!-- /unit:1f0bb3b85c64075e46 -->

<!-- unit:1442272c197d1c33c6 -->
<a id="u-1442272c197d1c33c6"></a>

### MIG, MPS и time-slicing не означают одно «деление GPU»

<!-- /unit:1442272c197d1c33c6 -->

<!-- unit:9a7b53978f1c8fba7e -->
<a id="u-9a7b53978f1c8fba7e"></a>

MIG на поддержанном hardware выделяет определённые GPU resource partitions и memory paths; доступные profiles зависят от модели. Детали — в [NVIDIA MIG guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/introduction.html). Source команда с profile IDs и подписью `1g.10gb` противоречила результату «три по 20GB»: ни эта арифметика, ни IDs не являются универсальным рецептом. Деление не обязано быть произвольным 80/3.

<!-- /unit:9a7b53978f1c8fba7e -->

<!-- unit:d2320da96136138b7d -->
<a id="u-d2320da96136138b7d"></a>

MPS организует совместное выполнение CUDA workloads нескольких processes; это не просто другое название time-slicing. Современные address-space/provisioning возможности зависят от поколения и configuration, но не тождественны hardware MIG partition. Общую архитектуру описывает [NVIDIA MPS](https://docs.nvidia.com/deploy/mps/latest/architecture.html). Утверждения «на любом GPU» и «вообще никакой изоляции» из разбираемого примера нельзя принимать без этих условий.

<!-- /unit:d2320da96136138b7d -->

<!-- unit:739cd6939349615c06 -->
<a id="u-739cd6939349615c06"></a>

Time-slicing/oversubscription в device plugin может предоставить больше logical allocations, чем физических GPU. Четыре allocations не означают вчетверо больше memory или throughput. Суммарные resident weights/cache всё ещё конкурируют за исходные ресурсы. Выбор механизма проверяют по требованиям изоляции, совместимости и latency конкретного workload, а не по числу видимых logical devices.

<!-- /unit:739cd6939349615c06 -->

<!-- unit:4b82287998fcdd07e2 -->
<a id="u-4b82287998fcdd07e2"></a>

### Один Python process ещё не один общий inference scheduler

<!-- /unit:4b82287998fcdd07e2 -->

<!-- unit:ae5652af3f82e67d9f -->
<a id="u-ae5652af3f82e67d9f"></a>

В S6 `self.models={name:LLM(...),...}` объявлен как multi-model batching с автоматической выгрузкой unused models. Сам показанный dictionary только хранит отдельные объекты и выбирает один для вызова. В нём нет общего memory admission, batching разных weights или eviction manager. Такие capabilities должен реализовать и документировать runtime; их нельзя вывести из одной оболочки Python.

<!-- /unit:ae5652af3f82e67d9f -->

<!-- unit:2220aa69366bc41654 -->
<a id="u-2220aa69366bc41654"></a>

Requests одной base model можно объединять в подходящий batch. Разные LoRA adapters поверх общей базы тоже могут поддерживаться специализированным execution path. Полностью разные models имеют разные weights и обычно свои execution groups; простое смешивание requests не делает их одним умножением на общую матрицу. Sharing, batching, tensor parallel и independent replicas решают разные задачи.

<!-- /unit:2220aa69366bc41654 -->

<!-- unit:634f4b4e8f07184cea -->
<a id="u-634f4b4e8f07184cea"></a>

Tensor parallel распределяет операции одной модели между devices и требует communication. Data-parallel serving replicas исполняют независимые requests с копиями model bundle. Pipeline parallel распределяет stages по devices и вводит своё расписание. Реальное ускорение зависит от communication, batch, memory и kernels; число GPU не является само по себе multiplier скорости.

<!-- /unit:634f4b4e8f07184cea -->

<!-- unit:a786eb38867b6e849e -->
<a id="u-a786eb38867b6e849e"></a>

## Rollout и проверка

<!-- /unit:a786eb38867b6e849e -->

<!-- unit:7f25c04168706779e1 -->
<a id="u-7f25c04168706779e1"></a>

Новая version: offline regression evaluation → artifact reload validation → shadow/canary по declared scope → comparison quality/latency/errors → expand или rollback. Shadow responses не выдаются пользователю, но могут стоить full compute; mutating tools в shadow требуют отдельной policy, чтобы не повторить real actions.

<!-- /unit:7f25c04168706779e1 -->

<!-- unit:f51d9c329c0305da85 -->
<a id="u-f51d9c329c0305da85"></a>

Rollback version включает model, template, retrieval/index/decision policy совместимость. Откат только weights при новом incompatible index может не восстановить прежний quality. Данные observability должны позволять связать request с exact versions без утечки лишнего content в logs.

<!-- /unit:f51d9c329c0305da85 -->

<!-- unit:30452a1aa9176d60e2 -->
<a id="u-30452a1aa9176d60e2"></a>

### Routing учитывает работу и cache locality

<!-- /unit:30452a1aa9176d60e2 -->

<!-- unit:822feab06a43de7a33 -->
<a id="u-822feab06a43de7a33"></a>

Round robin не учитывает, что один pod уже занят длинным request. Least-active-request/shortest-queue policy учитывает counts, но два requests могут иметь очень разные lengths. Prefix-aware routing может сэкономить prefill, но cache hit на перегруженном pod не всегда лучше свободной replica. Поэтому routing cost должен отражать chosen trade-off между wait, compute и locality.

<!-- /unit:822feab06a43de7a33 -->

<!-- unit:ef595f2f3dff1c097a -->
<a id="u-ef595f2f3dff1c097a"></a>

Source load balancer синхронно опрашивает `/metrics` каждого pod на каждый request. При многих pods это добавляет network calls, sequential timeouts и решения по уже устаревшим данным. Если все наблюдатели одновременно выбирают одну «самую пустую» replica, появляется herd effect. Нужны объявленная периодичность observation, допустимая stale metric и fallback, а не предположение о мгновенной полной картине.

<!-- /unit:ef595f2f3dff1c097a -->

<!-- unit:febf88034e6f8e9baa -->
<a id="u-febf88034e6f8e9baa"></a>

Приоритет коротких requests может снижать их ожидание, но бесконечный приток short jobs способен оставить long jobs без service. Ограничения долей, aging или отдельные queues задают fairness. Это выбор product/workload policy, а не теорема «короткие первыми всегда повышают общую удовлетворённость». Оценивать нужно и tail коротких, и starvation длинных.

<!-- /unit:febf88034e6f8e9baa -->

<!-- unit:1055ff8235b65b3b6f -->
<a id="u-1055ff8235b65b3b6f"></a>

### Graceful degradation меняет допустимое поведение

<!-- /unit:1055ff8235b65b3b6f -->

<!-- unit:5b5d9af5114c589ef7 -->
<a id="u-5b5d9af5114c589ef7"></a>

Переключение 70B→7B — смена модели, а не прозрачная оптимизация: могут измениться context, tool/schema support, качество и defaults. Маленькая fallback model должна быть доступна и проверена; её cold load во время overload может ухудшить ситуацию. Нельзя просто сменить строку имени и считать предыдущие guarantees сохранёнными.

<!-- /unit:5b5d9af5114c589ef7 -->

<!-- unit:f4b9c76697590d379c -->
<a id="u-f4b9c76697590d379c"></a>

Удалить KV-state старого активного request и продолжить как ни в чём не бывало нельзя. Возможны documented recompute/preemption, pause или explicit termination; они имеют цену и protocol. Circuit breaker ограничивает обращения к проблемной dependency и периодически проверяет восстановление по заданным правилам; это не замена admission и не средство «после падения всё стало безопасно».

<!-- /unit:f4b9c76697590d379c -->

<!-- unit:41c6096d9d59a1759b -->
<a id="u-41c6096d9d59a1759b"></a>

### Учебный rollout и завершение работы

<!-- /unit:41c6096d9d59a1759b -->

<!-- unit:c73ab93813546584cd -->
<a id="u-c73ab93813546584cd"></a>

До замены версий фиксируют workload, quality gates, model/template/index compatibility и cold/warm metrics. Shadow может удвоить часть compute и не должен автоматически повторять side effects. Canary оценивает заранее выбранные cohorts, затем расширяется или возвращается назад. Необычный результат одного короткого запроса ещё не является основанием для общего вывода о capacity или regression.

<!-- /unit:c73ab93813546584cd -->

<!-- unit:b5a82967b6837663f8 -->
<a id="u-b5a82967b6837663f8"></a>

Scale-in сначала прекращает новые admissions на выбранную replica, затем drain завершает или явно отменяет existing work. Длинные streams требуют согласованного timeout и понятного partial result. Rollback восстанавливает согласованные версии, а не только старый weight file. Source рекомендации «ваш LangGraph deployment» здесь остаются учебной архитектурой: вы уточняли, что изучаете тему, а не создаёте такого агента.

<!-- /unit:b5a82967b6837663f8 -->

<!-- unit:cdeedfee012144a67c -->
<a id="u-cdeedfee012144a67c"></a>

## Самопроверка

<!-- /unit:cdeedfee012144a67c -->

<!-- unit:f165c9b29c75e28184 -->
<a id="u-f165c9b29c75e28184"></a>

1. Почему weights вне image не означают warm model? **Ответ:** остаются load/transfer/init/cache stages.
2. Почему utilisation недостаточна для scaling? **Ответ:** не выражает очередь, deadline, mixed token cost и user latency полностью.
3. Чем readiness отличается от liveness? **Ответ:** готовность принимать работу против сигнала неисправности/restart.

<!-- /unit:f165c9b29c75e28184 -->

<!-- unit:c0c6daeb6be098252a -->
<a id="u-c0c6daeb6be098252a"></a>

4. Сохраняет ли init container готовый GPU engine основного процесса? **Ответ:** нет. Он может подготовить files; создание engine, device load и warmup main process остаются отдельными стадиями.
5. Совпадают ли average pod p95 и global p95? **Ответ:** нет. В примере 100 быстрых requests одного pod и одного медленного другого среднее quantiles2550ms, global nearest-rank p95=100ms.
6. Означает ли высокая GPU memory percentage неисправность? **Ответ:** не обязательно: engine мог заранее зарезервировать cache. Admission capacity и liveness — разные вопросы.
7. Делает ли dict из двух LLM общий scheduler? **Ответ:** нет. Dictionary выбирает объект; memory management, совместимый batching и eviction должны иметь реальную реализацию.
8. Почему короткие requests первыми не являются универсальной policy? **Ответ:** могут ухудшить ожидание/доступность service длинным jobs. Нужны выбранные fairness и tail constraints.
9. Равен ли расчёт 43 requests обещанию throughput? **Ответ:** нет. Это идеальная граница одного cache budget при заданном reservation, без других memory/compute ограничений.
10. Почему CPU/GPU utilisation всё же полезна? **Ответ:** помогает диагностировать bottlenecks, но не выражает целиком queue, deadlines и user SLO.
11. Где читать порт сервиса внутри Compose network? **Ответ:** из container endpoint. При mapping8001:8000 обычный service-to-service адрес использует 8000.
12. Переносит ли fallback на другие weights прежний KV-cache? **Ответ:** произвольная замена модели не сохраняет смысл вычисленных states; нужен отдельный корректный execution protocol.

<!-- /unit:c0c6daeb6be098252a -->

<!-- unit:50da4b0bbfe5a31692 -->
<a id="u-50da4b0bbfe5a31692"></a>

**Практика:** спроектировать cold-start timeline и load-test matrix без неподтверждённых чисел. Ввести bounded queue, total deadline/cancellation и drain. Измерить p95/TTFT/error rates при плавном росте и burst отдельно; сравнить с target SLO, а не только tokens/s.

<!-- /unit:50da4b0bbfe5a31692 -->

## Результат главы и переход дальше

Ответьте на вопрос: **Как согласовать admission, startup и воспроизводимый rollout?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
