# И. Эксплуатация inference-сервиса: запуск и рост нагрузки

[Оглавление](README.md). E.1–E.2 и практический протокол5.

**Основа:** S6, строки 2201–3904. Сохранены container layers/model artifacts, cold/warm start, queue/latency и autoscaling. **Исправления:** image не обязан включать weights, одинаковые model replicas могут быть взаимозаменяемыми при external state, utilisation не бесполезна, но недостаточна одна. Цены и «всегда2–5 минут» из чата не переносятся как нормы.

## 1. Что составляет воспроизводимый deployment

Bundle включает model/adapter revision, tokenizer/template, runtime/dependencies, dtype/quantization, generation defaults и preprocessing/postprocessing. Container фиксирует часть environment, но не автоматически весь semantic contract. Если weights по mutable URL изменились, одинаковый Docker image ещё не означает одинаковую модель.

Weights можно включить в image или доставлять отдельным versioned artifact/volume/cache. Первый путь упрощает immutability bundle, но увеличивает image transfer/storage; второй позволяет reuse и отдельные releases, но требует проверки version/hash и доступности. Ни один путь не запрещён сам по себе.

В layer caching сначала обычно устанавливают редко меняющиеся pinned dependencies, затем копируют часто меняющийся code. Multi-stage build может убрать build dependencies, но сам по себе не делает любое изображение маленьким. В исходном примере `COPY .` до dependency installation действительно создавал нежелательную cache invalidation при code changes; это конкретная причина, а не «каждая пересборка всегда заново загружает все слои».

## 2. Cold start по стадиям

Возможные stages: scheduler получает hardware → image/artifact fetch → process imports → weights load/partition → device transfer → kernel compilation/init → warmup → ready. Local disk weights экономят download, но не обязательно GPU load и warmup.

**Startup check** учитывает долгую initialisation; **readiness** означает способность обслуживать новые requests; **liveness** — необходимость restart при неисправности. Если использовать слишком агрессивный liveness во время normal long startup, можно получить restart loop. Если объявить ready до loaded weights, traffic столкнётся с errors/latency.

Warmup должен отражать supported shapes/path: один short prompt не гарантирует прогрев всех kernels для long context/batch или разных adapters. Измерять cold и warm отдельно, не прятать first requests из отчёта без указания.

Для same-version replicas weights могут быть одинаковыми; routing между ними допустим, если application state хранится совместимо. KV-cache locality делает выгодным sticky/prefix-aware routing, но потеря replica не должна означать потерю единственной копии пользовательского workflow, если требуется recovery.

## 3. Admission control и queue

Каждый request занимает разные memory/compute: prompt100/output50 и prompt20k/output2k не равны. Admission policy учитывает current tokens/cache, max output budget, concurrency и queue deadline. Нельзя бесконечно принимать работу, когда memory/capacity уже исчерпаны.

Backpressure означает bounded queue/ограничение новых requests или явный overload response. Cancellation должна по возможности освобождать engine state и downstream work; закрытое клиентское соединение само по себе не доказывает остановку GPU generation.

Timeout разбивается на phases: waiting, model, tool, total deadline. Если каждый retry получает полный новый timeout без общего deadline, пользователь может ждать гораздо дольше заявленного budget. Для retries нужны caps, backoff/jitter по задаче и distinction transient/permanent errors.

Пример: service устойчиво завершает2requests/s данного workload, приходит3requests/s. Без отказов/новой capacity queue растёт примерно1request/s после transient effects. Через минуту добавилось около60. Никакой красивый average GPU percentage не отменяет conservation of arrivals/completions.

## 4. Autoscaling с учётом startup delay

Signals: queue age/length, active sequences/tokens, TTFT/inter-token/end-to-end percentiles, throughput, memory headroom, errors и utilisation. CPU/GPU metrics полезны для diagnosis, но не являются достаточным выражением user SLO. Высокий utilisation при пустой queue не обязательно требует ещё replica, а низкий utilisation не доказывает отсутствие другого bottleneck.

Scaling реагирует с задержкой: пока новая replica загружается, очередь может вырасти. Нужны запас capacity/min ready replicas по expected bursts, warming policy и stabilisation/hysteresis, чтобы избежать oscillation. Одинаковый threshold для любой модели/instance не обоснован.

Scale-in должен drain existing work и корректно завершать stream/cache state. Просто убить pod с длинными запросами создаёт errors/retries и может усилить нагрузку. Max replicas/cost limits и overload policy должны согласоваться: если больше hardware недоступно, система должна ограничить admission.

## 5. Sharing и parallelism

Batching нескольких запросов одной model, несколько разных models на GPU, tensor parallel и data parallel решают разные задачи. Tensor parallel распределяет один model computation между devices и добавляет communication. Data parallel replicas увеличивают independent request capacity, если каждая помещается в hardware. Pipeline parallel добавляет stages и scheduling trade-offs.

Несколько tenants/adapters одной base могут экономить weights, но требуют isolation, routing и compatible batching. GPU sharing разных processes может конкурировать за memory/bandwidth; «два процесса на одном GPU» не гарантирует doubled capacity. Benchmark всего workload обязательнее отдельных microbenchmarks.

## 6. Rollout и проверка

Новая version: offline regression evaluation → artifact reload validation → shadow/canary по declared scope → comparison quality/latency/errors → expand или rollback. Shadow responses не выдаются пользователю, но могут стоить full compute; mutating tools в shadow требуют отдельной policy, чтобы не повторить real actions.

Rollback version включает model, template, retrieval/index/decision policy совместимость. Откат только weights при новом incompatible index может не восстановить прежний quality. Данные observability должны позволять связать request с exact versions без утечки лишнего content в logs.

1. Почему weights вне image не означают warm model? **Ответ:** остаются load/transfer/init/cache stages.
2. Почему utilisation недостаточна для scaling? **Ответ:** не выражает очередь, deadline, mixed token cost и user latency полностью.
3. Чем readiness отличается от liveness? **Ответ:** готовность принимать работу против сигнала неисправности/restart.

**Практика:** спроектировать cold-start timeline и load-test matrix без неподтверждённых чисел. Ввести bounded queue, total deadline/cancellation и drain. Измерить p95/TTFT/error rates при плавном росте и burst отдельно; сравнить с target SLO, а не только tokens/s.
