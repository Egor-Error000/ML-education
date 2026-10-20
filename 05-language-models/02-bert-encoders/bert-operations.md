# Сжатие, сервис encoder и жизненный цикл индекса

**Уровень:** углубление.

**Предпосылки:** [Encoder, tokenizer, attention и pooling](<bert-interface.md>), [Метрики, калибровка, OOD и интерпретация](<bert-evaluation.md>)

**Используют:** Выбранная прикладная задача и итоговая проверка результата.

**Учебный вопрос:** Как обновить encoder и индекс, сохраняя согласованность представлений?





## Карта вычисления и границы этапов

![Сжатие, сервис encoder и жизненный цикл индекса: входы, преобразования и выходы](<../../оформление/course/bert-operations.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Сравните формы и смысл каждого перехода с формулами главы. Выход преобразования и доказательство его качества — разные объекты: последняя проверка использует отдельный протокол оценки.

<!-- unit:b102cc90d9e5b25f12 -->
<a id="u-b102cc90d9e5b25f12"></a>

## Distillation classifier и retriever

<!-- /unit:b102cc90d9e5b25f12 -->

<!-- unit:fdece3210eceab70b8 -->
<a id="u-fdece3210eceab70b8"></a>

Classifier student может получать hard labels и teacher soft distribution. Для temperature $T$, $p_T=\operatorname{softmax}(z_T/T)$, $p_S=\operatorname{softmax}(z_S/T)$:

<!-- /unit:fdece3210eceab70b8 -->

<!-- unit:fdf350dda6a6a76ae7 -->
<a id="u-fdf350dda6a6a76ae7"></a>

$$
L=(1-\alpha)L_{\mathrm{hard}}+
\alpha T^2D_{\mathrm{KL}}(p_T\Vert p_S).
$$

<!-- /unit:fdf350dda6a6a76ae7 -->

<!-- unit:397008f7ec5264ee23 -->
<a id="u-397008f7ec5264ee23"></a>

Teacher detached; KL direction важна. Factor $T^2$ согласован с распространённой convention компенсации temperature scaling gradients, но normalization/classes и coefficient definition меняют effective strength. Его нельзя копировать без проверки полного objective.

<!-- /unit:397008f7ec5264ee23 -->

<!-- unit:8f657bf18c56197038 -->
<a id="u-8f657bf18c56197038"></a>

Retriever student может match embeddings с projection, pairwise scores или candidate ranking distribution. Требуется сохранить target geometry/pooling/prefixes, а не только classifier accuracy. Intermediate states разных depths/dimensions сравнивают через explicit alignment; «удалили layers» без дополнительного обучения — другой baseline.

<!-- /unit:8f657bf18c56197038 -->

<!-- unit:5e28f7c977154a8e48 -->
<a id="u-5e28f7c977154a8e48"></a>

[DistilBERT](https://arxiv.org/abs/1910.01108) — ранний пример encoder distillation. Его mechanism не означает, что любой student сохраняет все teacher skills. Evaluate rare/long/OOD cases и calibration; teacher labels могут переносить mistakes/contamination. Teacher inference/training data cost входит в full budget.

<!-- /unit:5e28f7c977154a8e48 -->

<!-- unit:92a79fa3500d0f4eec -->
<a id="u-92a79fa3500d0f4eec"></a>

**Схема курса:** encoder + tokenizer/pooling → optional quantization/export → embedding parity → versioned index rebuild → canary traffic → monitor and rollback

<!-- /unit:92a79fa3500d0f4eec -->

<!-- unit:187dbe2c2eb29e41bf -->
<a id="u-187dbe2c2eb29e41bf"></a>

<!-- source-depth-bert-И -->

<!-- /unit:187dbe2c2eb29e41bf -->

<!-- unit:90cc20b493e476e032 -->
<a id="u-90cc20b493e476e032"></a>

### Сначала выбирают, какое поведение student должен сохранить

<!-- /unit:90cc20b493e476e032 -->

<!-- unit:86a7944fdad88c280b -->
<a id="u-86a7944fdad88c280b"></a>

У classifier teacher даёт распределение по labels; student может обучаться одновременно на истинных labels и soft targets. У retriever важно ранжирование query–documents, а не только классификация одного текста. Сохранение teacher accuracy на topic dataset не доказывает сохранение multilingual retrieval. Для каждого target задаются свои pairs/candidates, loss и evaluation.

<!-- /unit:86a7944fdad88c280b -->

<!-- unit:66c041a8e1ab89e7a4 -->
<a id="u-66c041a8e1ab89e7a4"></a>

Teacher target фиксируется для backward. Если сопоставляются hidden states, требуется явное соответствие слоёв, ширин и normalization. У student в шесть слоёв нет автоматического «того же» состояния для каждого из двенадцати teacher layers. Можно выбрать mapping или projection, но это часть процедуры. Прямое удаление слоёв без обучения сохраняет другую функцию и служит отдельным baseline.

<!-- /unit:66c041a8e1ab89e7a4 -->

<!-- unit:96ef3e5d2199686b29 -->
<a id="u-96ef3e5d2199686b29"></a>

Distillation не создаёт новый независимый источник истины. Teacher errors и особенности train corpus могут перейти student. Поэтому сохраняют independent labels для final evaluation и считают стоимость teacher forward/target generation. В небольшом эксперименте можно изучить механизм на toy data, не называя это воспроизведением результатов большого checkpoint.

<!-- /unit:96ef3e5d2199686b29 -->

<!-- unit:e631ee938033437b94 -->
<a id="u-e631ee938033437b94"></a>

## Quantization и pruning

<!-- /unit:e631ee938033437b94 -->

<!-- unit:35ab7a7d76ee4a394f -->
<a id="u-35ab7a7d76ee4a394f"></a>

Weight quantization снижает representation precision, activation quantization — intermediate tensors, pruning удаляет selected weights/structures. Affine toy quantization:

<!-- /unit:35ab7a7d76ee4a394f -->

<!-- unit:e59cb07552c8eecebd -->
<a id="u-e59cb07552c8eecebd"></a>

$$
q=\operatorname{clip}\left(\operatorname{round}(w/s)+z,q_{\min},q_{\max}\right),
\qquad \widehat w=s(q-z).
$$

<!-- /unit:e59cb07552c8eecebd -->

<!-- unit:128e1b340544e9938a -->
<a id="u-128e1b340544e9938a"></a>

Scale, zero point, granularity и calibration data определяют error. Tiny perturbations logits могут менять rank/threshold near boundary, даже если average vector MSE мала. Поэтому post-export evaluation включает actual task metric, не только cosine agreement vectors.

<!-- /unit:128e1b340544e9938a -->

<!-- unit:6927e913fe85af16a7 -->
<a id="u-6927e913fe85af16a7"></a>

Unstructured zeros не ускоряют dense kernels автоматически. Structured pruning меняет shapes/heads и требует supported runtime. Quantized weights не означают, что activation memory и transfer costs уменьшились тем же factor. Supported architectures/heads/device formats проверяются у выбранного runtime version.

<!-- /unit:6927e913fe85af16a7 -->

<!-- unit:c4e0aa9582dcc100b0 -->
<a id="u-c4e0aa9582dcc100b0"></a>

### Маленькая ошибка чисел способна изменить дискретное решение

<!-- /unit:c4e0aa9582dcc100b0 -->

<!-- unit:59633408f16553f191 -->
<a id="u-59633408f16553f191"></a>

Для dot retrieval возьмём q=(1,0), d1=(1,0), d2=(0.9999,0). До сжатия d1 выше. Если первая координата d1 стала 0.9998, ranking перевернулся. Mean squared error двух координат d1 всего $2\cdot10^{-8}$. Малый vector MSE не гарантирует сохранения near-tie ranking; значения здесь специально выбраны близкими.

<!-- /unit:59633408f16553f191 -->

<!-- unit:da2174d50259338bed -->
<a id="u-da2174d50259338bed"></a>

Аналогично logits (1,0.9999) меняют argmax, если первый logit уменьшился на 0.0002. Важны margins и task decisions, а не только средняя numeric tolerance. Для retrieval проверяют ranking/qrels, для classification — метрики и thresholds, для QA — spans/null decisions. Где score далеко от boundary, та же perturbation может не менять ответ.

<!-- /unit:da2174d50259338bed -->

<!-- unit:f6a4a858314c9ddb2b -->
<a id="u-f6a4a858314c9ddb2b"></a>

Calibration quantization использует representative activations, в том числе нужные lengths, языки и источники. Случайные короткие English inputs не обязательно описывают распределение длинных Russian документов. Granularity, clipping и поддержка kernels меняют trade-off. Алгебра q→dequant сама по себе не доказывает runtime speedup.

<!-- /unit:f6a4a858314c9ddb2b -->

<!-- unit:02559a3b9e76429e0c -->
<a id="u-02559a3b9e76429e0c"></a>

### Что на самом деле занимает память после low-bit conversion

<!-- /unit:02559a3b9e76429e0c -->

<!-- unit:1c6e7c28589682049d -->
<a id="u-1c6e7c28589682049d"></a>

Сто миллионов fp32 weights —400 MB decimal; int8 payload тех же weights —100 MB. Если отдельно хранить по одному fp32 scale на 128 weights, scales добавят 3.125 MB. Zero points и другие metadata могут добавить ещё память по выбранному формату. Активации, buffers и runtime не обязаны стать int8 только из-за int8 weights.

<!-- /unit:1c6e7c28589682049d -->

<!-- unit:8c76554639c3c39173 -->
<a id="u-8c76554639c3c39173"></a>

Поэтому уменьшение weight payload в четыре раза не равно четырёхкратному уменьшению peak VRAM. Для длинного batch активации могут доминировать; для search corpus index может быть намного больше модели. Сравнение должно измерять полный выбранный workload после export, а не только размер файла weights.

<!-- /unit:8c76554639c3c39173 -->

<!-- unit:f392e7a300ee93a1e9 -->
<a id="u-f392e7a300ee93a1e9"></a>

## Memory accounting

<!-- /unit:f392e7a300ee93a1e9 -->

<!-- unit:9b48e39722977789bf -->
<a id="u-9b48e39722977789bf"></a>

Для inference fp32 weights одного encoder с $P$ parameters занимают примерно $4P$ bytes плюс activations/workspace/runtime. На training добавляются gradients и optimizer states. FP32 Adam с weights/gradients/two moments даёт порядка $4P+4P+8P=16P$ bytes для trainable parameters, без activation memory и framework overhead. Mixed precision/master copies меняют components; frozen base и trainable head считаются раздельно.

<!-- /unit:9b48e39722977789bf -->

<!-- unit:c2b4056ae9c3ae24a8 -->
<a id="u-c2b4056ae9c3ae24a8"></a>

Для $P=100$ million weights fp32 — 400 MB decimal; rough trainable parameter tensors FP32 Adam — 1.6 GB до activations. Это расчёт representations, не обещание actual peak VRAM. Length/batch/attention backend и checkpointing могут сделать activations dominant.

<!-- /unit:c2b4056ae9c3ae24a8 -->

<!-- unit:654d45441f465c4f05 -->
<a id="u-654d45441f465c4f05"></a>

При retrieval total memory включает corpus vectors/index/metadata, часто больше encoder weights. Уменьшение student encoder не уменьшает старый 768-dimensional index, если vector dimension не поменялась. Смена dimension/objective требует re-encoding либо специально проверенного compatibility protocol.

<!-- /unit:654d45441f465c4f05 -->

<!-- unit:0b79dc35200c480129 -->
<a id="u-0b79dc35200c480129"></a>

### Параметры модели и индекс — разные бюджеты

<!-- /unit:0b79dc35200c480129 -->

<!-- unit:d9680f5d14453153d8 -->
<a id="u-d9680f5d14453153d8"></a>

В FP32 Adam примере 16P bytes относятся к weights, gradients и двум moment tensors при объявленных dtype. Не включены saved activations, temporary workspaces, allocator reserve, runtime и inputs. У frozen base отсутствуют его optimizer states и параметрические gradients, но forward weights остаются, а backward по нужным входам может требовать активации. Эти компоненты считают раздельно.

<!-- /unit:d9680f5d14453153d8 -->

<!-- unit:39f60a2cd3c10d19c6 -->
<a id="u-39f60a2cd3c10d19c6"></a>

Если encoder уменьшился со 100M до 50M параметров, fp32 weights сократились на 200 MB. Но миллион corpus vectors ширины 768 float32 по-прежнему занимает 3.072 GB raw. Чтобы уменьшить этот массив до 256 coordinates, нужен соответствующий embedding output и переиндексация либо обоснованная compression mapping. Меньший backbone не сокращает сохранённые document vectors автоматически.

<!-- /unit:39f60a2cd3c10d19c6 -->

<!-- unit:5e60afb2eb304c290d -->
<a id="u-5e60afb2eb304c290d"></a>

У процесса может быть и CPU, и GPU memory, и disk index. Peak при загрузке способен временно включать две версии весов или staging buffers. Steady-state usage после прогрева и peak rollout/reindex — разные измерения. Одна цифра размера checkpoint не описывает ни одно из них полностью.

<!-- /unit:5e60afb2eb304c290d -->

<!-- unit:b2ecc154f09b3d906f -->
<a id="u-b2ecc154f09b3d906f"></a>

## Encoder inference отличается от causal decode

<!-- /unit:b2ecc154f09b3d906f -->

<!-- unit:a3477590934f0fc590 -->
<a id="u-a3477590934f0fc590"></a>

Обычный bidirectional encoder обрабатывает whole input; нет sequential next-token decode с тем же KV-cache contract, что causal LLM. Добавление нового token может изменить states прежних tokens из-за bidirectional attention, поэтому нельзя просто reuse causal cache semantics.

<!-- /unit:a3477590934f0fc590 -->

<!-- unit:769d641ce16eb1acaa -->
<a id="u-769d641ce16eb1acaa"></a>

Workload — texts, pairs или chunks. Tokenization, padding, transfer, forward и pooling измеряются отдельно и end-to-end. Для dense retrieval offline document encoding и online query path имеют разные throughput/latency goals. Cross-encoder обслуживает pairs и candidate count, а не один document embedding.

<!-- /unit:769d641ce16eb1acaa -->

<!-- unit:3593305fcf362cd60e -->
<a id="u-3593305fcf362cd60e"></a>

Dynamic batching увеличивает utilisation ценой queue wait; length bucketing снижает padding, но меняет ожидание отдельных requests. Throughput и p95 latency измеряются на representative length/arrival distribution, warm/cold runs и same task. Один long pair может раздуть batch work всех коротких.

<!-- /unit:3593305fcf362cd60e -->

<!-- unit:5ad31dd58033186646 -->
<a id="u-5ad31dd58033186646"></a>

### Почему causal append-cache не переносится на обычный BERT

<!-- /unit:5ad31dd58033186646 -->

<!-- unit:c229de0005dac81ed0 -->
<a id="u-c229de0005dac81ed0"></a>

Возьмём attention без mask, WQ=WK=0 и WV=1 на одномерных values. Для единственного значения 2 output равен 2. После добавления значения 6 softmax равномерный, и output **старой первой позиции** становится 4. Это маленькое attention-ядро для доказательства зависимости; полный Transformer ещё добавляет norm, residual и другие слои.

<!-- /unit:c229de0005dac81ed0 -->

<!-- unit:967b7dca49ab176440 -->
<a id="u-967b7dca49ab176440"></a>

В causal схеме первая позиция не получила бы новый источник и сохранила output. У bidirectional encoder старые contextual states могут измениться, поэтому K/V следующих слоёв нельзя в общем случае считать прежними. На первом слое некоторые проекции старых неизменённых embeddings можно переиспользовать, но это не доказывает корректность обычного многослойного causal KV-cache протокола. Для точного forward нужно сохранить семантику всей сети.

<!-- /unit:967b7dca49ab176440 -->

<!-- unit:1b6c3c7132331289ad -->
<a id="u-1b6c3c7132331289ad"></a>

Кеш готового результата **полностью одинакового входа** — другая операция. Он допустим при согласованных weights, tokenizer, mode, pooling и relevant policy/version. Смена projection, normalization или max-length меняет функцию encoding и требует другого cache identity. Возможность кешировать функцию не равнозначна возможности дописывать новые tokens без пересчёта предыдущих states.

<!-- /unit:1b6c3c7132331289ad -->

<!-- unit:08deb51f5798369282 -->
<a id="u-08deb51f5798369282"></a>

### Вопрос S6 о latency и throughput: считать timestamps каждого запроса

<!-- /unit:08deb51f5798369282 -->

<!-- unit:e3408a2137b2459e0f -->
<a id="u-e3408a2137b2459e0f"></a>

В S6:3434–3561 приведены illustrative batches и функция `total_time/num_requests*batch_size`, названная average latency. Эта формула в общем случае её не измеряет. Возьмём учебное расписание:100 запросов пришли в момент 0, обслуживаются 10 batches по 10, каждый batch длится 0.2s. Completion times равны 0.2,0.4,…,2.0s, каждое время относится к 10 запросам.

<!-- /unit:e3408a2137b2459e0f -->

<!-- unit:74f1cd822d5debb3f1 -->
<a id="u-74f1cd822d5debb3f1"></a>

Throughput за интервал —100/2=50 req/s. Средняя пользовательская latency —среднее completion−arrival, то есть 1.1s. Формула из разбираемого примера даёт 2/100×10=0.2s, длительность одного batch в этом расписании. Даже 2/100=0.02s —только амортизированное время на request при полной параллельной обработке, а не его waiting time.

<!-- /unit:74f1cd822d5debb3f1 -->

<!-- unit:382b3dae6c7802407d -->
<a id="u-382b3dae6c7802407d"></a>

В реальном load test записывают arrival/enqueue, начало обработки и completion каждого request. Из них получают queue wait, service interval и end-to-end latency с оговорёнными границами измерения. p50/p95 считаются по request durations, а не выводятся из общего throughput. Cold start и warm steady state измеряются отдельно; разные arrivals и lengths меняют результат.

<!-- /unit:382b3dae6c7802407d -->

<!-- unit:757eb8e15a951d180f -->
<a id="u-757eb8e15a951d180f"></a>

### Queue length и одинаковые weights не определяют весь сервис

<!-- /unit:757eb8e15a951d180f -->

<!-- unit:59f16f4df20187e82e -->
<a id="u-59f16f4df20187e82e"></a>

Десять коротких запросов и десять очень длинных имеют одинаковую queue length, но разный объём работы. У dense encoder attention component для длины 1024 против 32 отличается в 1024 раза при фиксированных остальных размерах; линейные components растут в 32 раза. Поэтому queue count полезен, но не является полной workload metric. Нужны length/token budgets и измеренная service time distribution.

<!-- /unit:59f16f4df20187e82e -->

<!-- unit:9f5377a840be26efb6 -->
<a id="u-9f5377a840be26efb6"></a>

Скелет queue handler в S6 сначала вызывает enqueue, затем напрямую process_request; без показанного dequeue/worker задания остаются в очереди, хотя запрос уже обрабатывается другим путём. В полноценной схеме worker извлекает задания и возвращает результаты ожидающим handlers, а ошибки и отмены имеют определённое поведение. Gauge должен отражать объявленную величину: ожидающие, принятые или выполняемые запросы — это разные множества. Увеличение счётчика до блокирующего put может включить ещё не помещённые в очередь задания.

<!-- /unit:9f5377a840be26efb6 -->

<!-- unit:e4d7057803092fe656 -->
<a id="u-e4d7057803092fe656"></a>

Таблица S6 «7B/A100, batch1…128» не является воспроизводимым benchmark без фактического запуска, lengths и конфигурации. Plateau throughput может быть связан не только с заполненной памятью, но и с compute/bandwidth/kernels/очередью. Приведённые размеры batches не становятся универсальными для encoder или causal LLM.

<!-- /unit:e4d7057803092fe656 -->

<!-- unit:77bcacd4839a32faa2 -->
<a id="u-77bcacd4839a32faa2"></a>

Ещё одно уточнение S6: наличие отдельной копии weights в каждой реплике не делает запросы автоматически невзаимозаменяемыми. Для одинакового encoder bundle и одинакового детерминированного входа функция та же. Локальный cache влияет на повторное вычисление и latency; session state, разные revisions или несогласованный index — отдельные причины, почему маршрутизацию нужно организовать аккуратно.

<!-- /unit:77bcacd4839a32faa2 -->

<!-- unit:263b99a3773d56e073 -->
<a id="u-263b99a3773d56e073"></a>

## Export и versioned bundle

<!-- /unit:263b99a3773d56e073 -->

<!-- unit:5efa7a4714be363d51 -->
<a id="u-5efa7a4714be363d51"></a>

Eager baseline → один export/runtime change → reload → output и task metric comparison. ONNX/другой backend может не поддержать custom architecture/masks или dynamic shapes так, как ожидается. Empty, Unicode, long/truncated и padding cases проверяются отдельно. Небольшая numeric difference допустима только в объявленных task tolerances.

<!-- /unit:5efa7a4714be363d51 -->

<!-- unit:ce4bf98b71e8d3ab8c -->
<a id="u-ce4bf98b71e8d3ab8c"></a>

Bundle содержит tokenizer/revision, weights/head, pooling/projection, normalization/prefixes, label map, thresholds, max-length/truncation policy, runtime/config и data/index compatibility. Сохранить только tensor weights недостаточно. [Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference) — пример runtime, чью поддержку конкретного checkpoint/head проверяют по версии.

<!-- /unit:ce4bf98b71e8d3ab8c -->

<!-- unit:2c12727b20e03f5e8c -->
<a id="u-2c12727b20e03f5e8c"></a>

### Что значит «сравнить после export»

<!-- /unit:2c12727b20e03f5e8c -->

<!-- unit:7455efb978d29db42c -->
<a id="u-7455efb978d29db42c"></a>

Сначала фиксируют eager outputs и task metrics на разрешённом validation наборе. Затем изменяют один runtime/export шаг, загружают сохранённый bundle заново и повторяют тот же набор. Shape equality проверяет интерфейс, numeric tolerances — близость вычисления, task metrics — сохранение нужных решений. Эти три проверки не заменяют друг друга.

<!-- /unit:7455efb978d29db42c -->

<!-- unit:c771fb5d1ffc68c4f4 -->
<a id="u-c771fb5d1ffc68c4f4"></a>

Особенно полезны near-tie examples, разные padding lengths, пустой текст, Unicode, truncation и batch из неодинаковых длин. Ошибка dynamic shape может проявиться только при другой длине, а неверная pooling mask —только при padding. Проверка одного короткого предложения не подтверждает весь serving contract.

<!-- /unit:c771fb5d1ffc68c4f4 -->

<!-- unit:07a0dbd990ed94b727 -->
<a id="u-07a0dbd990ed94b727"></a>

Bundle включает и невесовые компоненты. Если поменять order labels, тот же tensor logits станет означать другие классы. Если поменять query prefix, прежний document index может оказаться несогласованным. Thresholds, preprocessing и index version версионируются вместе с weights; «модель та же» не означает прежнее прикладное поведение.

<!-- /unit:07a0dbd990ed94b727 -->

<!-- unit:55a3b13cd255071100 -->
<a id="u-55a3b13cd255071100"></a>

## Monitoring и active learning

<!-- /unit:55a3b13cd255071100 -->

<!-- unit:66fb54412df6e532a2 -->
<a id="u-66fb54412df6e532a2"></a>

Отслеживают input/source/language/length, class frequencies, missing/unknown rates, embedding norms/distributions, latency/errors и delayed task labels. Vector drift — proxy изменений, а не доказанное падение relevance. Calibration/threshold change меняет decisions даже без нового backbone и версионируется отдельно.

<!-- /unit:66fb54412df6e532a2 -->

<!-- unit:e7aa7ec92f1a97dd2a -->
<a id="u-e7aa7ec92f1a97dd2a"></a>

Active learning выбирает uncertain и diverse cases для human annotation. Но выбор только uncertain examples смещает labelled sample; такой sample нельзя напрямую считать unbiased production test. Нужен independent representative audit и provenance feedback. Diversity не гарантирует coverage critical rare slices, поэтому sampling strategy проверяют.

<!-- /unit:e7aa7ec92f1a97dd2a -->

<!-- unit:903760cd54aa4aabf9 -->
<a id="u-903760cd54aa4aabf9"></a>

После исправления labels обучают candidate и сравнивают на held-out tasks. Feedback пользователя может быть noisy или отражать exposure предыдущего ranking; automatic ingestion без review усиливает bias. Annotation policy и retention/source contract сохраняются.

<!-- /unit:903760cd54aa4aabf9 -->

<!-- unit:6b05182f28c274f54f -->
<a id="u-6b05182f28c274f54f"></a>

### Мониторинг без новых labels даёт proxies

<!-- /unit:6b05182f28c274f54f -->

<!-- unit:75acbca776fd86499b -->
<a id="u-75acbca776fd86499b"></a>

Распределение языков, lengths, norms и predicted classes помогает заметить изменение входов. Но embedding mean может сдвинуться из-за общего поворота пространства, сохранив внутримодельные similarities при согласованном reindex. И наоборот, падение relevance способно начаться без большого mean drift. Для качества нужны delayed labels, representative audit и разбор retrieval outcomes.

<!-- /unit:75acbca776fd86499b -->

<!-- unit:4fb79f16c61d7a7a24 -->
<a id="u-4fb79f16c61d7a7a24"></a>

Active learning выбирает полезные для разметки случаи, например uncertain и diverse inputs. Если из 1000 production texts размечены только 10 самых uncertain и в них 5 ошибок,50% —ошибка выбранной трудной подвыборки, а не unbiased оценка всего production. Для независимого monitoring держат representative sample с понятным sampling protocol.

<!-- /unit:4fb79f16c61d7a7a24 -->

<!-- unit:5fdd3465892174f63b -->
<a id="u-5fdd3465892174f63b"></a>

Пользовательский feedback тоже зависит от exposure: пользователь видел только выданные candidates. Невыданный документ не становится irrelevant из-за отсутствия click. Обновление labels, training и выбор новой модели должны учитывать эту зависимость. Иначе система усиливает решения предыдущего ranking вместо исправления ошибок.

<!-- /unit:5fdd3465892174f63b -->

<!-- unit:269181ca38844a3767 -->
<a id="u-269181ca38844a3767"></a>

## Обновление index и rollout

<!-- /unit:269181ca38844a3767 -->

<!-- unit:830220c9ec6d4ef225 -->
<a id="u-830220c9ec6d4ef225"></a>

При новом tokenizer/encoder/pooling/normalization query и document vectors обычно должны перейти в согласованный space. Равная dimension не доказывает compatibility. Полный reindex делается в новой version с проверенными counts, IDs, ACL и удалёнными устаревшими documents. Ingestion новых docs и migration старых — разные операции.

<!-- /unit:830220c9ec6d4ef225 -->

<!-- unit:479dfb23787e968378 -->
<a id="u-479dfb23787e968378"></a>

Порядок: offline regression/cost → shadow на реальном distribution без изменения ответа → ограниченный canary по заранее заданным metrics → rollout либо rollback. Rollback восстанавливает coherent model+index+threshold bundle, а не только old weights. Одновременный поиск новых queries по старым docs vectors требует специально verified mapping/dual-index protocol.

<!-- /unit:479dfb23787e968378 -->

<!-- unit:3cf0f2a15c335f594b -->
<a id="u-3cf0f2a15c335f594b"></a>

Итоговый service report включает post-export quality, p50/p95, throughput, memory, cold start и limits. Настоящий benchmark записывается после запуска; формулы памяти и pipeline design являются учебным расчётом.

<!-- /unit:3cf0f2a15c335f594b -->

<!-- unit:f5731568495057c1a2 -->
<a id="u-f5731568495057c1a2"></a>

### Согласованность обновления важнее совпадения размерности

<!-- /unit:f5731568495057c1a2 -->

<!-- unit:323d26370870dafffc -->
<a id="u-323d26370870dafffc"></a>

Новый query encoder v2 и старые document vectors v1 могут быть несовместимы при одинаковой ширине. Это уже показано поворотным примером в главе [«Sentence embeddings и contrastive learning»](<bert-embeddings.md>). При обычной миграции создают отдельный v2 index и проверяют corpus snapshot, counts, IDs, удаления и policy. Новые документы, появившиеся во время reindex, требуют согласованного catch-up; удалённые не должны незаметно воскреснуть из старого snapshot.

<!-- /unit:323d26370870dafffc -->

<!-- unit:0d581a7dccc76cba63 -->
<a id="u-0d581a7dccc76cba63"></a>

Shadow сравнивает candidate behavior на том же workload, сохраняя выбранный operational ответ прежней системы. Canary меняет ответ ограниченной части запросов по заранее заданному protocol. Это разные этапы проверки, а не две названия одного запуска. В учебном проекте сначала описывают критерии качества, latency и допустимых regressions, затем выполняют запуск, если он действительно входит в задачу.

<!-- /unit:0d581a7dccc76cba63 -->

<!-- unit:8ffd9da6ac036b5005 -->
<a id="u-8ffd9da6ac036b5005"></a>

Rollback возвращает согласованный bundle: encoder/tokenizer/pooling, document index, label map и thresholds по нужной задаче. Возврат только старых weights при оставшемся новом index не является полным восстановлением. Сохранение dual versions на время сравнения требует дополнительной памяти и operational стоимости, которые включают в budget.

<!-- /unit:8ffd9da6ac036b5005 -->

<!-- unit:ca9eaa775682f90726 -->
<a id="u-ca9eaa775682f90726"></a>

## Контроль и практика

<!-- /unit:ca9eaa775682f90726 -->

<!-- unit:2e780e0846d46716b4 -->
<a id="u-2e780e0846d46716b4"></a>

1. Почему small vector MSE не гарантирует unchanged retrieval ranking? **Ответ:** near-tie candidates могут поменяться местами, а metric зависит от порядка и relevance.
2. Почему sparse weights не обязательно быстрее? **Ответ:** runtime может всё ещё исполнять dense kernels.
3. Можно ли кэшировать BERT как causal decode? **Ответ:** ordinary bidirectional states зависят от обеих сторон, поэтому semantics различается.
4. Что rollback должен восстановить для search? **Ответ:** согласованные model/tokenizer/pooling/index/decision-policy versions.

<!-- /unit:2e780e0846d46716b4 -->

<!-- unit:ea4809dd99421680a1 -->
<a id="u-ea4809dd99421680a1"></a>

5. Почему total_time/requests×batch_size не является средней latency? **Ответ:** не учитывает arrival/completion каждого request и его queue wait. В учебном расписании получается 0.2s вместо actual mean1.1s.
6. Делает ли отдельная копия weights реплику обязательно session-stateful? **Ответ:** нет. Одинаковый детерминированный encoder bundle вычисляет ту же функцию; cache/session/index state рассматриваются отдельно.
7. Почему маленький vector MSE может поменять retrieval? **Ответ:** near-tie margins могут быть меньше numeric perturbation; ranking — дискретное решение.
8. Можно ли оценить общий error rate по самым uncertain размеченным случаям? **Ответ:** без учёта selection protocol это смещённая подвыборка; нужен representative audit.

<!-- /unit:ea4809dd99421680a1 -->

<!-- unit:d1d659fe9c350fd667 -->
<a id="u-d1d659fe9c350fd667"></a>

**Практика:** выберите teacher signals для classifier и retriever, посчитайте rough memory components. Спроектируйте eager/export comparison и нагрузочный test. Напишите migration нового encoder с dual snapshots, checks IDs/ACL и criterion rollback; это основа проекта P6.

<!-- /unit:d1d659fe9c350fd667 -->

## Проектная практика ветки

<!-- unit:dcf77e87db978eb8ef -->
<a id="u-dcf77e87db978eb8ef"></a>

## Сквозные проекты и артефакты

<!-- /unit:dcf77e87db978eb8ef -->

<!-- unit:10c22e85c88da64062 -->
<a id="u-10c22e85c88da64062"></a>

### P6. Компактный encoder service — системная ветка

<!-- /unit:10c22e85c88da64062 -->

<!-- unit:43345093aecb36f921 -->
<a id="u-43345093aecb36f921"></a>

Выбрать успешно проверенный P1/P3; compact/distilled/quantized alternative → export → нагрузочный test. Итог: versioned bundle, post-export metrics, p50/p95 latency, throughput, memory, rollback и index compatibility.

<!-- /unit:43345093aecb36f921 -->

<!-- unit:3e841c6351394b6c6f -->
<a id="u-3e841c6351394b6c6f"></a>

В теоретическом режиме для каждого проекта сначала достаточно спецификации входов/выходов, ручного примера, выбранных losses/метрик и разбора возможных ошибок. Это подтверждает теоретическую работу; запуск и численные результаты засчитываются после пользовательского выполнения.

<!-- /unit:3e841c6351394b6c6f -->

## Результат главы и переход дальше

Ответьте на вопрос: **Как обновить encoder и индекс, сохраняя согласованность представлений?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
