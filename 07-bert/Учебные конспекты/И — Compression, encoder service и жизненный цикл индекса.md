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


<!-- source-depth-bert-И -->
### 1.1. Сначала выбирают, какое поведение student должен сохранить

У classifier teacher даёт распределение по labels; student может обучаться одновременно на истинных labels и soft targets. У retriever важно ранжирование query–documents, а не только классификация одного текста. Сохранение teacher accuracy на topic dataset не доказывает сохранение multilingual retrieval. Для каждого target задаются свои pairs/candidates, loss и evaluation.

Teacher target фиксируется для backward. Если сопоставляются hidden states, требуется явное соответствие слоёв, ширин и normalization. У student в шесть слоёв нет автоматического «того же» состояния для каждого из двенадцати teacher layers. Можно выбрать mapping или projection, но это часть процедуры. Прямое удаление слоёв без обучения сохраняет другую функцию и служит отдельным baseline.

Distillation не создаёт новый независимый источник истины. Teacher errors и особенности train corpus могут перейти student. Поэтому сохраняют independent labels для final evaluation и считают стоимость teacher forward/target generation. В небольшом эксперименте можно изучить механизм на toy data, не называя это воспроизведением результатов большого checkpoint.


## 2. Quantization и pruning

Weight quantization снижает representation precision, activation quantization — intermediate tensors, pruning удаляет selected weights/structures. Affine toy quantization:

$$
q=\operatorname{clip}\left(\operatorname{round}(w/s)+z,q_{\min},q_{\max}\right),
\qquad \widehat w=s(q-z).
$$

Scale, zero point, granularity и calibration data определяют error. Tiny perturbations logits могут менять rank/threshold near boundary, даже если average vector MSE мала. Поэтому post-export evaluation включает actual task metric, не только cosine agreement vectors.

Unstructured zeros не ускоряют dense kernels автоматически. Structured pruning меняет shapes/heads и требует supported runtime. Quantized weights не означают, что activation memory и transfer costs уменьшились тем же factor. Supported architectures/heads/device formats проверяются у выбранного runtime version.


### 2.1. Маленькая ошибка чисел способна изменить дискретное решение

Для dot retrieval возьмём q=(1,0), d1=(1,0), d2=(0.9999,0). До сжатия d1 выше. Если первая координата d1 стала 0.9998, ranking перевернулся. Mean squared error двух координат d1 всего $2\cdot10^{-8}$. Малый vector MSE не гарантирует сохранения near-tie ranking; значения здесь специально выбраны близкими.

Аналогично logits (1,0.9999) меняют argmax, если первый logit уменьшился на 0.0002. Важны margins и task decisions, а не только средняя numeric tolerance. Для retrieval проверяют ranking/qrels, для classification — метрики и thresholds, для QA — spans/null decisions. Где score далеко от boundary, та же perturbation может не менять ответ.

Calibration quantization использует representative activations, в том числе нужные lengths, языки и источники. Случайные короткие English inputs не обязательно описывают распределение длинных Russian документов. Granularity, clipping и поддержка kernels меняют trade-off. Алгебра q→dequant сама по себе не доказывает runtime speedup.

### 2.2. Что на самом деле занимает память после low-bit conversion

Сто миллионов fp32 weights —400 MB decimal; int8 payload тех же weights —100 MB. Если отдельно хранить по одному fp32 scale на 128 weights, scales добавят 3.125 MB. Zero points и другие metadata могут добавить ещё память по выбранному формату. Активации, buffers и runtime не обязаны стать int8 только из-за int8 weights.

Поэтому уменьшение weight payload в четыре раза не равно четырёхкратному уменьшению peak VRAM. Для длинного batch активации могут доминировать; для search corpus index может быть намного больше модели. Сравнение должно измерять полный выбранный workload после export, а не только размер файла weights.


## 3. Memory accounting

Для inference fp32 weights одного encoder с $P$ parameters занимают примерно $4P$ bytes плюс activations/workspace/runtime. На training добавляются gradients и optimizer states. FP32 Adam с weights/gradients/two moments даёт порядка $4P+4P+8P=16P$ bytes для trainable parameters, без activation memory и framework overhead. Mixed precision/master copies меняют components; frozen base и trainable head считаются раздельно.

Для $P=100$ million weights fp32 — 400 MB decimal; rough trainable parameter tensors FP32 Adam — 1.6 GB до activations. Это расчёт representations, не обещание actual peak VRAM. Length/batch/attention backend и checkpointing могут сделать activations dominant.

При retrieval total memory включает corpus vectors/index/metadata, часто больше encoder weights. Уменьшение student encoder не уменьшает старый 768-dimensional index, если vector dimension не поменялась. Смена dimension/objective требует re-encoding либо специально проверенного compatibility protocol.


### 3.1. Параметры модели и индекс — разные бюджеты

В FP32 Adam примере 16P bytes относятся к weights, gradients и двум moment tensors при объявленных dtype. Не включены saved activations, temporary workspaces, allocator reserve, runtime и inputs. У frozen base отсутствуют его optimizer states и параметрические gradients, но forward weights остаются, а backward по нужным входам может требовать активации. Эти компоненты считают раздельно.

Если encoder уменьшился со 100M до 50M параметров, fp32 weights сократились на 200 MB. Но миллион corpus vectors ширины 768 float32 по-прежнему занимает 3.072 GB raw. Чтобы уменьшить этот массив до 256 coordinates, нужен соответствующий embedding output и переиндексация либо обоснованная compression mapping. Меньший backbone не сокращает сохранённые document vectors автоматически.

У процесса может быть и CPU, и GPU memory, и disk index. Peak при загрузке способен временно включать две версии весов или staging buffers. Steady-state usage после прогрева и peak rollout/reindex — разные измерения. Одна цифра размера checkpoint не описывает ни одно из них полностью.


## 4. Encoder inference отличается от causal decode

Обычный bidirectional encoder обрабатывает whole input; нет sequential next-token decode с тем же KV-cache contract, что causal LLM. Добавление нового token может изменить states прежних tokens из-за bidirectional attention, поэтому нельзя просто reuse causal cache semantics.

Workload — texts, pairs или chunks. Tokenization, padding, transfer, forward и pooling измеряются отдельно и end-to-end. Для dense retrieval offline document encoding и online query path имеют разные throughput/latency goals. Cross-encoder обслуживает pairs и candidate count, а не один document embedding.

Dynamic batching увеличивает utilisation ценой queue wait; length bucketing снижает padding, но меняет ожидание отдельных requests. Throughput и p95 latency измеряются на representative length/arrival distribution, warm/cold runs и same task. Один long pair может раздуть batch work всех коротких.


### 4.1. Почему causal append-cache не переносится на обычный BERT

Возьмём attention без mask, WQ=WK=0 и WV=1 на одномерных values. Для единственного значения 2 output равен 2. После добавления значения 6 softmax равномерный, и output **старой первой позиции** становится 4. Это маленькое attention-ядро для доказательства зависимости; полный Transformer ещё добавляет norm, residual и другие слои.

В causal схеме первая позиция не получила бы новый источник и сохранила output. У bidirectional encoder старые contextual states могут измениться, поэтому K/V следующих слоёв нельзя в общем случае считать прежними. На первом слое некоторые проекции старых неизменённых embeddings можно переиспользовать, но это не доказывает корректность обычного многослойного causal KV-cache протокола. Для точного forward нужно сохранить семантику всей сети.

Кеш готового результата **полностью одинакового входа** — другая операция. Он допустим при согласованных weights, tokenizer, mode, pooling и relevant policy/version. Смена projection, normalization или max-length меняет функцию encoding и требует другого cache identity. Возможность кешировать функцию не равнозначна возможности дописывать новые tokens без пересчёта предыдущих states.

### 4.2. Вопрос S6 о latency и throughput: считать timestamps каждого запроса

В S6:3434–3561 приведены illustrative batches и функция `total_time/num_requests*batch_size`, названная average latency. Эта формула в общем случае её не измеряет. Возьмём учебное расписание:100 запросов пришли в момент 0, обслуживаются 10 batches по 10, каждый batch длится 0.2s. Completion times равны 0.2,0.4,…,2.0s, каждое время относится к 10 запросам.

Throughput за интервал —100/2=50 req/s. Средняя пользовательская latency —среднее completion−arrival, то есть 1.1s. Формула из чата даёт 2/100×10=0.2s, длительность одного batch в этом расписании. Даже 2/100=0.02s —только амортизированное время на request при полной параллельной обработке, а не его waiting time.

В реальном load test записывают arrival/enqueue, начало обработки и completion каждого request. Из них получают queue wait, service interval и end-to-end latency с оговорёнными границами измерения. p50/p95 считаются по request durations, а не выводятся из общего throughput. Cold start и warm steady state измеряются отдельно; разные arrivals и lengths меняют результат.

### 4.3. Queue length и одинаковые weights не определяют весь сервис

Десять коротких запросов и десять очень длинных имеют одинаковую queue length, но разный объём работы. У dense encoder attention component для длины 1024 против 32 отличается в 1024 раза при фиксированных остальных размерах; линейные components растут в 32 раза. Поэтому queue count полезен, но не является полной workload metric. Нужны length/token budgets и измеренная service time distribution.

Скелет queue handler в S6 сначала вызывает enqueue, затем напрямую process_request; без показанного dequeue/worker задания остаются в очереди, хотя запрос уже обрабатывается другим путём. В полноценной схеме worker извлекает задания и возвращает результаты ожидающим handlers, а ошибки и отмены имеют определённое поведение. Gauge должен отражать объявленную величину: ожидающие, принятые или выполняемые запросы — это разные множества. Увеличение счётчика до блокирующего put может включить ещё не помещённые в очередь задания.

Таблица S6 «7B/A100, batch1…128» не является воспроизводимым benchmark без фактического запуска, lengths и конфигурации. Plateau throughput может быть связан не только с заполненной памятью, но и с compute/bandwidth/kernels/очередью. Приведённые размеры batches не становятся универсальными для encoder или causal LLM.

Ещё одно уточнение S6: наличие отдельной копии weights в каждой реплике не делает запросы автоматически невзаимозаменяемыми. Для одинакового encoder bundle и одинакового детерминированного входа функция та же. Локальный cache влияет на повторное вычисление и latency; session state, разные revisions или несогласованный index — отдельные причины, почему маршрутизацию нужно организовать аккуратно.


## 5. Export и versioned bundle

Eager baseline → один export/runtime change → reload → output и task metric comparison. ONNX/другой backend может не поддержать custom architecture/masks или dynamic shapes так, как ожидается. Empty, Unicode, long/truncated и padding cases проверяются отдельно. Небольшая numeric difference допустима только в объявленных task tolerances.

Bundle содержит tokenizer/revision, weights/head, pooling/projection, normalization/prefixes, label map, thresholds, max-length/truncation policy, runtime/config и data/index compatibility. Сохранить только tensor weights недостаточно. [Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference) — пример runtime, чью поддержку конкретного checkpoint/head проверяют по версии.


### 5.1. Что значит «сравнить после export»

Сначала фиксируют eager outputs и task metrics на разрешённом validation наборе. Затем изменяют один runtime/export шаг, загружают сохранённый bundle заново и повторяют тот же набор. Shape equality проверяет интерфейс, numeric tolerances — близость вычисления, task metrics — сохранение нужных решений. Эти три проверки не заменяют друг друга.

Особенно полезны near-tie examples, разные padding lengths, пустой текст, Unicode, truncation и batch из неодинаковых длин. Ошибка dynamic shape может проявиться только при другой длине, а неверная pooling mask —только при padding. Проверка одного короткого предложения не подтверждает весь serving contract.

Bundle включает и невесовые компоненты. Если поменять order labels, тот же tensor logits станет означать другие классы. Если поменять query prefix, прежний document index может оказаться несогласованным. Thresholds, preprocessing и index version версионируются вместе с weights; «модель та же» не означает прежнее прикладное поведение.


## 6. Monitoring и active learning

Отслеживают input/source/language/length, class frequencies, missing/unknown rates, embedding norms/distributions, latency/errors и delayed task labels. Vector drift — proxy изменений, а не доказанное падение relevance. Calibration/threshold change меняет decisions даже без нового backbone и версионируется отдельно.

Active learning выбирает uncertain и diverse cases для human annotation. Но выбор только uncertain examples смещает labelled sample; такой sample нельзя напрямую считать unbiased production test. Нужен independent representative audit и provenance feedback. Diversity не гарантирует coverage critical rare slices, поэтому sampling strategy проверяют.

После исправления labels обучают candidate и сравнивают на held-out tasks. Feedback пользователя может быть noisy или отражать exposure предыдущего ranking; automatic ingestion без review усиливает bias. Annotation policy и retention/source contract сохраняются.


### 6.1. Мониторинг без новых labels даёт proxies

Распределение языков, lengths, norms и predicted classes помогает заметить изменение входов. Но embedding mean может сдвинуться из-за общего поворота пространства, сохранив внутримодельные similarities при согласованном reindex. И наоборот, падение relevance способно начаться без большого mean drift. Для качества нужны delayed labels, representative audit и разбор retrieval outcomes.

Active learning выбирает полезные для разметки случаи, например uncertain и diverse inputs. Если из 1000 production texts размечены только 10 самых uncertain и в них 5 ошибок,50% —ошибка выбранной трудной подвыборки, а не unbiased оценка всего production. Для независимого monitoring держат representative sample с понятным sampling protocol.

Пользовательский feedback тоже зависит от exposure: пользователь видел только выданные candidates. Невыданный документ не становится irrelevant из-за отсутствия click. Обновление labels, training и выбор новой модели должны учитывать эту зависимость. Иначе система усиливает решения предыдущего ranking вместо исправления ошибок.


## 7. Обновление index и rollout

При новом tokenizer/encoder/pooling/normalization query и document vectors обычно должны перейти в согласованный space. Равная dimension не доказывает compatibility. Полный reindex делается в новой version с проверенными counts, IDs, ACL и удалёнными устаревшими documents. Ingestion новых docs и migration старых — разные операции.

Порядок: offline regression/cost → shadow на реальном distribution без изменения ответа → ограниченный canary по заранее заданным metrics → rollout либо rollback. Rollback восстанавливает coherent model+index+threshold bundle, а не только old weights. Одновременный поиск новых queries по старым docs vectors требует специально verified mapping/dual-index protocol.

Итоговый service report включает post-export quality, p50/p95, throughput, memory, cold start и limits. Настоящий benchmark записывается после запуска; формулы памяти и pipeline design являются учебным расчётом.


### 7.1. Согласованность обновления важнее совпадения размерности

Новый query encoder v2 и старые document vectors v1 могут быть несовместимы при одинаковой ширине. Это уже показано поворотным примером в главе Д. При обычной миграции создают отдельный v2 index и проверяют corpus snapshot, counts, IDs, удаления и policy. Новые документы, появившиеся во время reindex, требуют согласованного catch-up; удалённые не должны незаметно воскреснуть из старого snapshot.

Shadow сравнивает candidate behavior на том же workload, сохраняя выбранный operational ответ прежней системы. Canary меняет ответ ограниченной части запросов по заранее заданному protocol. Это разные этапы проверки, а не две названия одного запуска. В учебном проекте сначала описывают критерии качества, latency и допустимых regressions, затем выполняют запуск, если он действительно входит в задачу.

Rollback возвращает согласованный bundle: encoder/tokenizer/pooling, document index, label map и thresholds по нужной задаче. Возврат только старых weights при оставшемся новом index не является полным восстановлением. Сохранение dual versions на время сравнения требует дополнительной памяти и operational стоимости, которые включают в budget.


## 8. Контроль и практика модулей 38–40

1. Почему small vector MSE не гарантирует unchanged retrieval ranking? **Ответ:** near-tie candidates могут поменяться местами, а metric зависит от порядка и relevance.
2. Почему sparse weights не обязательно быстрее? **Ответ:** runtime может всё ещё исполнять dense kernels.
3. Можно ли кэшировать BERT как causal decode? **Ответ:** ordinary bidirectional states зависят от обеих сторон, поэтому semantics различается.
4. Что rollback должен восстановить для search? **Ответ:** согласованные model/tokenizer/pooling/index/decision-policy versions.

5. Почему total_time/requests×batch_size не является средней latency? **Ответ:** не учитывает arrival/completion каждого request и его queue wait. В учебном расписании получается 0.2s вместо actual mean1.1s.
6. Делает ли отдельная копия weights реплику обязательно session-stateful? **Ответ:** нет. Одинаковый детерминированный encoder bundle вычисляет ту же функцию; cache/session/index state рассматриваются отдельно.
7. Почему маленький vector MSE может поменять retrieval? **Ответ:** near-tie margins могут быть меньше numeric perturbation; ranking — дискретное решение.
8. Можно ли оценить общий error rate по самым uncertain размеченным случаям? **Ответ:** без учёта selection protocol это смещённая подвыборка; нужен representative audit.

**Практика:** выберите teacher signals для classifier и retriever, посчитайте rough memory components. Спроектируйте eager/export comparison и нагрузочный test. Напишите migration нового encoder с dual snapshots, checks IDs/ACL и criterion rollback; это основа проекта P6.
