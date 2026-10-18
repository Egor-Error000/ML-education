# Поправки для обновлённых конспектов

12 октября 2026. Источники S1–S6 определены в [плане миграции](MIGRATION_PLAN.md); строки относятся к исходным TXT, страницы — к исходному PDF. Это готовые редакционные указания для последующей миграции заметок. В действующих планах общие ошибки уже исправлены; ниже конкретизируются места в истории.

## C01. Единая нотация и арифметика LoRA — приоритет 1

**Где:** S1, стр. 1–3, 8–9, 15–17; S2, разбор анатомии LoRA и вопросы target_modules/DoRA, строки 169–500 и 3405–4587.

В PDF одновременно записаны y=W0*x+B*A*x и A размера d_in*r, B размера r*d_out. При column-vector x это не согласовано. Использовать одну конвенцию:

```text
x: d_in × 1
W0: d_out × d_in
A: r × d_in
B: d_out × r
y = W0 x + s B A x, s = alpha/r
```

Для batch row-vectors: Y=X W0^T + s X A^T B^T. MLP может иметь d_in != d_out; GQA меняет размер k/v projections. Shapes брать из конфигурации/weight.shape конкретной модели, не смешивать Llama-7B MHA и произвольный GQA пример.

`r` — ранг/внутренняя размерность, а не количество параметров. Их число для одной проекции: r(d_in+d_out). При обычной LoRA модули выбираются по соответствию names/конфигурации; это не означает отдельную пару A/B для каждой attention-head.

Числовой шаг S1, стр. 15–17, полезен и переносится. При x=(1,2), W0=[[1,2],[3,4]], A=(0.1,0.2), B=(0,0)^T, target=(6,10), s=1 и L=0.5*||y-target||² получаем:

```text
y=(5,11), L=1
grad_B=(-0.5,0.5)^T, grad_A=(0,0)
eta=0.1 -> B_new=(0.05,-0.05)^T
y_new=(5.025,10.975)
L_new=0.950625
```

В PDF итог округлён к 0.9503; заменить на 0.950625 (примерно 0.9506). Если s не равен 1, он участвует и в forward, и в gradients. Следующий самостоятельный checkpoint — второй шаг, где gradient A уже может стать ненулевым.

Для DoRA magnitude — вектор по выходным каналам в этой конвенции; запись W_i = m_i*(V_i+DeltaV_i)/||V_i+DeltaV_i|| пояснять построчно. При транспонированной конвенции ось нормы меняется. Не называть m одним скаляром.

## C02. Attention backprop и трактовка Q/K/V — приоритет 1

**Где:** S1, стр. 10–13; особенно запись `⊙ Softmax'(A)` на стр. 12.

У softmax связанные координаты. Поэлементное умножение на производную отдельной координаты не описывает полный backward. Для rowwise attention без дополнительных узлов:

```text
Z = Q K^T / sqrt(d_k) + mask
S = softmax_rows(Z)
O = S V
G_O = dL/dO
G_V = S^T G_O
G_S = G_O V^T
G_Z = S ⊙ (G_S - row_sum(S ⊙ G_S))
G_Q = G_Z K / sqrt(d_k)
G_K = G_Z^T Q / sqrt(d_k)
```

`row_sum` имеет shape N×1 и broadcast по столбцам. При Q=XW_Q получаем G_WQ=X^T G_Q. Маска исключает запрещённые связи; для реального блока добавить output projection, residual, normalization и chain rule. Не путать attention scores Z с LoRA A. При extraction PDF отдельно проверить сохранность sqrt и индексов, а не автоматически верстать extracted text.

Q/K/V имеют функциональные роли matching/content, но координаты не обязаны означать конкретные грамматические категории. Softmax нормируется по строке: даже симметричные raw scores не гарантируют симметричные normalized weights. Градиенты не гарантируют «поворот к нулевому углу», ортогональность heads или обнуление ложной связи.

Самостоятельного attention-loss в обычном Transformer нет; gradient идёт от общей задачи. Auxiliary losses возможны в специальных моделях. Основа формулы: [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

## C03. Инициализация и merge LoRA — приоритет 1

**Где:** S1, стр. 1–3, 9; S2, early LoRA overview; S3, стартовая сводка и сравнение PEFT.

Общий принцип: начать с нулевой DeltaW, например A random/B zero. «Всегда Gaussian A» неверно для всех реализаций: в PEFT default — Kaiming-uniform A и zero B, существуют другие init modes. При этом при B=0 на первом шаге gradient A=0. [Документация PEFT](https://huggingface.co/docs/peft/v0.20.0/en/package_reference/lora).

Merge устраняет отдельный low-rank путь, если конфигурация допускает слияние. Обещание «ровно 0 ms/идентичная скорость» заменить отсутствием дополнительных adapter matmuls и замером runtime. Quantized base может требовать dequantization/requantization; совместимость и качество после merge проверять. [PEFT quantization](https://github.com/huggingface/peft/blob/main/docs/source/developer_guides/quantization.md).

## C04. RNN, Transformer и hidden states — приоритет 1

**Где:** S1, стр. 4–14.

«Attention полностью решил forgetting RNN» заменить смягчением фиксированного context bottleneck. RNN нельзя распараллелить по зависимым временным шагам так же, как Transformer, но можно по batch/другим осям; распределённое обучение RNN возможно.

Transformer параллелит обработку позиций при training/prefill, а autoregressive decode сохраняет зависимость от ранее выбранных токенов. Causal mask действует и на позиции prompt. Token-ID остаётся индексом; embeddings/hidden states — непрерывные representations на позициях, logits — оценки vocabulary. Последний hidden state не является гарантированным без потерь «сжатым смыслом всей истории».

Не утверждать фиксированную лестницу «первый слой синтаксис, последний абстракция» как обязательное свойство. Разделять точную механику и иллюстративную интерпретацию.

## C05. Память, dtype и размер модели — приоритет 1

**Где:** S2, строки 4–72; S3, 1–48; S1, стр. 2; S4, quantization/memory разделы.

14 bytes/parameter — один способ accounting, не универсальная константа AdamW. Указывать веса, gradients, master weights если есть, optimizer dtype, activations, временные buffers и sharding/offload. 7B*14 bytes=98 GB в десятичной системе; больше 100 GB получается только после дополнительных расходов/другой конфигурации.

7B весов в FP16/BF16 — примерно 14 GB без metadata, а не обязательные 26 GB. Расход adapter зависит от rank, modules и dtype, не всегда 50–100 MB.

NF4 — storage format; compute dtype отдельный. QLoRA не означает обучение всех состояний/activations в 4-bit. GGUF — container format, GGML — tensor-библиотека/историческая экосистема; CPU-only ограничение неверно.

## C06. CPT, SFT, данные и masking — приоритет 2

**Где:** S3, 228–339 и 428–505.

Сырой доменный корпус подходит для continued pretraining; подготовленные instruction-response или иной выбранный objective — для SFT. Книжные диалоги не автоматически полезны/бесполезны: всё зависит от цели, формата и качества. Формулировку «SFT не может исправлять галлюцинации, только DPO» убрать: разные интервенции могут влиять на factuality, эффект проверяется.

Assistant-only mask — часто полезная конфигурация, не единственный корректный SFT objective. Проверять labels, shift, EOS, padding и packing boundaries. Датасеты «1000 SFT/500 DPO» — примеры budgets, не минимум/гарантия результата. Learning curves и coverage важнее фиксированных норм.

## C07. RLHF, DPO и KTO — приоритет 2

**Где:** S3, 343–423 и 509–673.

Policy/reference/reward/value — логические роли PPO-RLHF, не обязательные четыре независимые полные копии. DPO использует pairwise preferences; KTO использует отдельные desirable/undesirable примеры. Финальное обобщение «DPO/IPO/KTO обучаются на парах» исправить.

KTO reference point связан с оценкой KL относительно reference policy, не «средним качеством ответов» в обычном смысле. Асимметрия задаётся objective/weights; «плохое всегда штрафуется сильнее» не универсально, как и обещания экономии 3–10×. [KTO objective и weights](https://github.com/huggingface/trl/blob/main/docs/source/kto_trainer.md).

Не превращать «IPO ограничивает margin» в гарантию отсутствия переобучения; сравнить objective, reference и данные. DPO не отменяет необходимость оценивать качество задачи и регрессии.

## C08. Model soup, TIES, DARE и switching — приоритет 2

**Где:** S3, 719–816.

Model soup может улучшать compatible checkpoints, но не «почти всегда побеждает». Общая инициализация/совместимость и evaluation обязательны. Усреднение full deltas BA и отдельное усреднение факторов A/B не эквивалентны; не переносить совет «усредните LoRA-веса» без выбранного алгоритма.

Для стандартного TIES sign election опирается на aggregate значений после trim, а не всегда на число голосов «три модели против одной». DARE — drop/rescale task deltas; это не универсальный conflict-resolution алгоритм. [Статья TIES](https://arxiv.org/abs/2306.01708), [PEFT merging](https://huggingface.co/docs/peft/developer_guides/model_merging).

Hot-swapping — inference routing, merge adapter в base — другая операция, merging нескольких task models — третья. `ADAPTER` в runtime config сам по себе не доказывает поддержку мгновенного динамического switching в каждом backend.

## C09. Evaluation — приоритет 1

**Где:** S3, 820–918, особенно заключение строки 909.

«Perplexity мертва» заменить: PPL полезна для выбранного language-model objective и corpus, но не покрывает instruction following, factuality и успешность действий. Human preference также не единственная бизнес-метрика и не безошибочный ground truth.

Judge: rubric, blind comparison, order swap, ties, человеческая подвыборка. Отдельно correctness, formatting, retrieval и task success. JSON validity не равна correctness. Contamination risk не доказывает, что вся высокая benchmark оценка — запоминание. Alignment tax измеряется; универсального падения/порога «10–15% слишком много» нет.

«Мы собрали данные/запустили SFT/DPO» в резюме заменить на «разобрали теоретическую схему», пока нет пользовательского результата.

## C10. Quantization/KV-cache и bottlenecks — приоритет 2

**Где:** S4, A.1–A.3 и 4771–4922; S3, 674–715.

Различать algorithm (GPTQ/AWQ), numeric representation (INT4/NF4), format (GGUF) и engine. AWQ использует activation-aware scaling; «важные веса обязательно хранятся в более высоком dtype» не является полным описанием.

Weight-only ≠ dynamic activation quantization ≠ KV-cache quantization. Они не поддерживаются «везде» одинаково. Pruning/compression не гарантируют speedup без подходящих kernels/shapes.

Cache сохраняет K/V прошлых позиций, но каждый новый query всё ещё взаимодействует с разрешённой историей. Prefill/decode могут иметь разные bottlenecks. Для обычного cache: 2*layers*batch*tokens*kv_heads*head_dim*bytes_per_element, плюс overhead.

Правило «для JSON минимум Q5/Q6» и числа 75%/95% убрать как недоказанные общие рекомендации. Сравнивать конкретные model/dtype/runtime/prompt/decoding и отдельно schema/семантику.

## C11. Structured output и tool calling — приоритет 1

**Где:** S5, 120–242; S6, 4–178, особенно строка 132.

Constrained decoding ограничивает допустимые продолжения в поддерживаемой грамматике/schema. Оно не гарантирует смысл аргументов, правильный инструмент и завершение при timeout/token limit. Backend может поддерживать subset JSON Schema; validation/error handling остаются нужны. [Пример реализации vLLM](https://docs.vllm.ai/en/stable/examples/features/structured_outputs/).

Output parser проверяет/разбирает уже полученный текст; его наличие само не доказывает logit masking. Проверка формы, semantic validation, permissions и execution — отдельные шаги. Декларация tools в API не обучает веса модели при каждом запросе.

## C12. Greedy, temperature, speculative sampling и beam search — приоритет 1

**Где:** S5, 246–413; S4, 6439–7697.

При T>0 деление logits на T сохраняет argmax в точной арифметике. «Почти одинаковые вероятности могут поменять greedy из-за temperature» не является свойством алгоритма. Numeric rounding/ties — отдельные implementation cases. T=0 — специальный greedy режим API, не арифметическое деление на ноль.

Корректная speculative sampling схема сохраняет distribution target, а не обязательно тот же текст для одного seed. Acceptance=min(1,p/q), на rejection требуется нормированное positive residual (p-q)+. Beam search оптимизирует другую процедуру отбора гипотез; доказательство sampling нельзя автоматически переносить на beam. [Speculative sampling](https://arxiv.org/abs/2302.01318).

Более сильный draft не обязательно ускоряет: важны близость distributions, цена draft и verifier, workload/конкурентность. Не считать искусственно приведённые benchmarks Qwen пользовательским измерением.

## C13. Research и неподтверждённые индустриальные цифры — приоритет 2

**Где:** S4, 3361–3441, 3917–4767, 6922–7318, 7701–8230; S6, multi-vector storage/ROI 1146–1373.

Выделить три категории: опубликованный первоисточник, иллюстративный пример, гипотеза о непубличной системе. Числа latency/cost/quality без experiment settings и источника не переносить как benchmark evidence; рецепты закрытых компаний не выводить из пользовательских жалоб.

**DSpark подтверждён первоисточником:** статья от 6 июля 2026 описывает semi-autoregressive drafting и confidence-scheduled verification, включая применение в DeepSeek-V4. Заявленный прирост 60–85% относится к сравнению с MTP-1 при согласованном throughput в условиях статьи, а не ко всем runtime/hardware. [DSpark](https://arxiv.org/abs/2607.05147).

Значит, DSpark сохраняется как реальный case study; детальные значения из чата сверять с таблицами статьи, не удалять тему целиком и не переносить обещания без условий. MTP, tree speculation и beam search различать.

## C14. API/runtime, caching и deployment — приоритет 2

**Где:** S5, 417–1034, 1460–2822; S6, 1377–3904.

Код LangGraph/AutoGen/CrewAI/Ollama/LM Studio и runtime configuration — version-sensitive examples, не подтверждённо рабочий проект. OpenAI-compatible interface не гарантирует поддержку tools/streaming/schema/checkpoint semantics всеми серверами.

Prompt/KV cache не равно cache готовых ответов. TTL, минимальный prefix и цена API зависят от провайдера/версии/модели; перед практикой сверять документацию, учебный конспект хранит механизм. Cache key/invalidation должны учитывать данные, модель, права и контекст.

E.2 scaling не сводится к одному «правильному» сигналу или порогу. Учитывать очередь, rate/workload, latency, capacity, cold start и hysteresis. Health checks отличать по назначению; показанный YAML не считается проверенным deployment.

Обсуждение внутреннего устройства IDE-агентов без исходников/официального описания помечать как концептуальную гипотезу, а не установленную реализацию конкретного продукта.


## C15. Углубление адаптации по S2–S4

Во второй редакции развёрнуты все десять глав адаптации. Сохранены вопросы о работе Prefix Tuning/BitFit и подготовке данных из книг, различия preference objectives, merge/switching и оценка итогового artifact. Для темы дистилляции восстановлен вопрос пользователя из S4 о раннем teacher, который может быть меньше student. Разделы RLAIF/multiobjective/continual learning явно дополняют план: полноценного разбора этих методов в исходных чатах не было.

- S3, PEFT: замороженные weights не означают отсутствие производных через них; soft prompt на входе и layerwise prefix K/V имеют разные shapes и budgets. Память optimizer не равна полной памяти training. Ускорение и мгновенный hot-swap не следуют только из выбора метода.
- S3, SFT: книги можно использовать как continued pretraining или для основанных на evidence QA; SFT не ограничивается форматом и может менять фактическое поведение. Mask labels не обрывает все gradient paths через prompt. Token- и sequence-weighted objectives различаются.
- S3, preference learning: IPO quadratic loss не прекращает обучение после порога, а возвращает margin к target. KTO reference point — величина, связанная с reference policy/KL, а не средняя «оценка качества». ORPO odds не равны вероятности события, универсальная экономия compute не заявляется.
- S3, merging: averaging weights не равно averaging outputs. Exact sum LoRA deltas представима конкатенацией factors с ростом rank. TIES sum-based sign election не равно большинству знаков. DARE сохраняет expectation delta, но не nonlinear output.
- S3, evaluation: PPL не бесполезна, human preference не исчерпывает ценность, judge не заменяет объективные проверки. Размер выборки, допустимый alignment tax и latency switching не имеют универсальных порогов из чата.
- S4, дистилляция: исправлены probabilities для logits(10,1,0.5) при T=1/5; DistilBERT сопоставляется с BERT-base, а не с BERT-large. Student не обязан быть меньше; sequence answer API не предоставляет полного vocab distribution. Annealing alpha и temperature различаются; расписания и closed-company recipes из чата не считаются установленными фактами. Учтены teacher compute/cache, движение targets и distribution student prefixes.

Новые числовые примеры проверяются отдельно в `tmp/verify_depth_revision.py`. Расширение и успешный рендер не заменяют финальный содержательный аудит уникальных фрагментов источников.

## C16. Квантизация: уточнения при подробной сверке S4

**Где:** S4, 4–2649; S3, 674–715. Расширена глава инференса А. Возвращены пояснения активаций, memory examples, calibration criteria, mixed-precision decomposition, SmoothQuant, QAT и исходные вопросы о GGUF, INT4, производных и online quantization.

- Attention scores для32heads,L=2048,2bytes занимают256MiB при полной материализации;128MB в S4 неверно. Указаны lifetime/fusion и различия training/inference. Каждый layer использует свой cache.
- Cache80layers,L=8192,Hkv=8,dh=128,B=1,2bytes занимает2.5GiB. Подстановка hidden width вместо KV width игнорирует GQA. Исходное20GiB также не больше70GB INT8 weights70B.
- Для absmax100 symmetric INT8 step=100/127;0.1 и0.5 округляются в разные codes. Для демонстрации схлопывания взяты0.1 и0.3. Обратный scale из S4 приведён к единой convention.
-16 равномерных уровней с обоими endpoints[−1,1] имеют15 интервалов. INT4 code7 не означает maximum real value7 без scale. Optimizer second moment не центрированная variance; FP32 states не являются математически обязательными для всех schemes.
- Q4_K block имеет4.5bits/value с metadata; это не неизменный file-average Q4_K_M. Weight bitwidth и precision scales различаются. GGUF не самозапускающийся executable и не CPU-only. Утверждения оK-means, вечном превосходстве качества и dtype-порогеJSON не переносятся.
- LLM.int8() outlier decomposition не требует одного статического списка channels для всех inputs. SmoothQuant имеет и dynamic, и static variants; W8A8 не требует online scales во всех случаях. Gaussian distributions и универсальные activation ranges/порог6.7B не постулируются.
- GPTQ layer reconstruction Gram/Hessian отличается от полного LM-loss Hessian; compensation зависит от correlations. ПримерH=[[2,1],[1,2]] проверен через finite differences и constrained solve. Hessian исходного polynomial indefinite, поэтому curvature не всегда означает рост loss.
- BitNet b1.58 сохраняет SwiGLU; отсутствие нелинейностей и адаптивная bitwidth из S4 не подтверждаются. Atom serving не доказывает INT4 pretraining. QLoRA65B/48GB не заменяется обещанием65B/24GB.
- Online activation scaling, cache requantization, eviction, offline learned calibration и routing между artifacts разделены. OmniQuant не является доказательством пересмотра weights на каждом decode step. Dequantization единственного INT4 artifact не восстанавливает исходные FP16 weights.

Исправления подтверждают механизмы и арифметику, а не заявляют проведённый пользователем benchmark. Источники методов и runtime structures указаны непосредственно в главе.


## C17. Pruning и исходный вопрос об избыточности

**Где:** S4,2653–3441. В главу инференса Б возвращены виды pruning, критерии importance, schedules, названные методы, вопрос «почему не сразу оптимальная структура» и гипотеза о деградации закрытых сервисов.

В матрице исходника10nonzeros; удаление двух даёт50%final sparsity, а не удаление половины nonzeros. CSR пример показывает, что sparse storage может быть больше dense. Structured pruning gated MLP согласует up/gate/down. При fixed head dimension удаление половины MHA heads сокращает Q/K/V/O projections вдвое; четверть attention parameters из S4 не следует. MLP pruning не уменьшает cache автоматически.

Малый gradient не означает безопасное удаление, OBS quadratic optimum не является global certificate. Wanda использует input L2 norm и сравнение внутри output row; SparseGPT использует reconstruction/error compensation. SliceGPT работает с principal components signals при согласованной transformation, не просто PCA weights. ShortGPT2024/BI не получает универсальных98%quality. Сохранён Lottery Ticket мотив с ограничениями initialization/training. Семь удалений10%remaining дают52.17%sparsity.

Утверждения о secret production recipes компаний, «десятках триллионов», гарантированном падении от compression и synonym MoE=pruning не перенесены как факты. Возможные причины поведения разобраны отдельно от доказанного механизма. Импорт, quality и latency проверяются для итогового artifact; structured shape change не обещает совместимости со всеми runtimes.

## C18. KV-cache: последовательность forward и границы экономии

**Где:** S4,4771–4922; связь с activation/cache примерами4–2649. Глава инференса В расширена scalar forward двух шагов, memory/admission расчётами и cache correctness протоколом.

Prefill logits уже предсказывают первый output token; его K/V появляются при последующем forward на его ID. Каждый layer хранит свои K/V. Cached attention линейно по history на step, но суммарно не линейно по всему output length. Исправлена явная1/sqrt(dk) нормировка score.

ГруппыKV и query heads разделены. ЧислоHkv берут из config; исходное имя«Llama-2/Llama-3 8B» не определяет32KVheads. Примеры mixed lengths и available-cache budget не объявляются throughput. MQA quality и распространённость GQA не имеют универсальных оценок из чата.

ОшибкаK меняет attention weights, ошибкаV —содержимое; KIVI axes не универсальная характеристика каждого token. Sliding window не означает буквального полного забывания всех старых сведений, attention sinks не сохраняют удалённый текст. H2O eviction, quantization и paged placement разделены;99%utilization не гарантируется. Cached/uncached logits сравнивают с одинаковыми IDs, positions/masks и declared numerical tolerance.

Дополнительно к C16 разъяснены imatrix и EXL2: calibration importance не равно semantic map; EXL2 связан с GPTQ optimization, но не является тождественным именем GPTQ file format.

## C19. Serving, PagedAttention и перенос CV/encoder примеров

**Где:** S4,4926–6435. Глава инференса Г расширена по всем четырём пунктамB.1–B.4: schedule/refill, paging/CoW, engines/configuration, metrics/queues, ONNX/compile, CV batching и embeddings.

- Static batch может отдавать finished output/stream раньше конца группы; refill, compaction и streaming разделены. Original lengths2/300/150/1 дают37.75%useful slots в условной схеме, а не измеренныйGPUutilization. Continuous batching не гарантирует всегда полный batch, минимальный latency или отсутствие overload.
- В исходной fragmentation diagram есть подходящая свободная2GB contiguous area для2GBrequest. Исправлено на две раздельные30MBholes и требование50MB. Paging сохраняет internal tail waste; для lengths1/16/17/31 block16 требует96slots при65useful.
- Четыре branches с2common+1private block каждая используют6physical blocks вместо12, экономия50%, не66%. CoW не позволяет без пересчёта downstream KV менять ранний system prompt. Block softmax должен сохранять global normalization; дан counterexample и stable reduction formula.
- CLI limits относятся к разным ресурсам: context prompt+output, sequences, processed tokens и executor-memory budget. Dtype auto не обещаетFP8 quantization. Проценты выигрыша engines и A100benchmark numbers не подтверждены условиями/запуском и не перенесены как measurements. Противопоставление vLLM «интерпретатор» и TensorRT-LLM «всегда одинengine» снято.
- TTFT/TPOT/end-to-end rate и aggregate goodput различаются; TPOT для1token не определяется черезL−1. Open/closed-loop нагрузки и queue включены в протокол. Нельзя весь эффект sequential-vs-batched сравнения приписывать paging.
- ONNX и ONNX Runtime разделены; export не гарантирует поддержку всех ops/platforms. Fixed-width embedding output не означает одинаковую token длину. BN folding выведен для fixed inference statistics. Fullgraph=True при graph break требует ошибки, не гарантированного fallback.
- Source CV-batcher забирает всю очередь безmax_batch_size, создаёт таймер на каждый request и stack из batched tensors даёт лишнюю ось. Async handler с sync inference может блокировать event loop. Float dtype/RGB/preprocessing, NumPy serialization, logits-vs-confidence и несколько model sessions workers требуют проверки. Исправленная схема описана как bounded worker queue с корректным shape/error/cancellation contract; исходный class не объявлен готовым production кодом.

Источники механизмов и mutable API links указаны в самой главе; численные примеры проверяются в `tmp/verify_depth_revision.py`. Обсуждение и примеры не означают проведённый пользователем deploy или benchmark.

## C20. Speculation, прогресс draft и корректность beam

**Где:** S4,6439–8230. Глава инференса Д расширена по исходным вопросам о скорости маленьких/больших моделей, beam search и DeepSeek/DSpark. Добавлены causal verification, цикл commit/rollback, residual proof, variants, throughput/cost расчёты и границы публикаций.

- Original accept/reject pseudocode с fresh sample изp при rejection не сохраняет target law. Нужен normalized positive residual. Scalar probabilities предложенных tokens не задают полный correction distribution. В примереParis точный первый ratio18/19; после rejection второго token остальные scores не продолжают history. Заявленное2× не учитывало draft time.
- Acceptance равна1−TV(p,q); это совпадение probabilities, а не correctness external answer. Более высокий benchmark score или одинаковая model family/architecture не гарантируют большей acceptance. Source Qwen tables и future predictions не подтверждены как measurements.
- Expected committed prefix определяется survival probabilities, а неγ·α. Приγ5,conditionalα0.8 total progress3.68928, условный speedup2.45952 для5+10mscycle и10msbaseline. Higher temperature не обязана уменьшать overlap; дан counterexample.
- Выбор наиболее уверенного draft после просмотра candidates меняет proposal law. Дваuniformdraws с preferenceA при его наличии дают(.75,.25), не(.5,.5). Adaptive admission требует согласованности с используемой correction, чтобы избежать selection bias.
- Honest target scoring только предложенных branches недостаточен для exact beam equivalence: пропущенный candidate может войти в baseline top-B. Даны massesAX/AY/BZ/BW.4×4×5=80, не100. Fixed-width beam и полное exponential tree различаются; beam не гарантирует global sequence optimum.
- Ngram lookup не ограничен одним кодом, early exit требует подготовленного path, Medusa/EAGLE не сводятся к независимым Linear heads. Quantized copy не гарантирует identical cache с float target. Модель/weights/flags из source snippets не считаются подтверждённым запуском.
- MTP auxiliary work не бесплатен; один proposal/round не тождествен одному output token или обязательным двум full target passes. DSpark deployment claim сохранён с primary-paper baseline/metric conditions. Tree-adaptation paper отдельно от DSpark не доказывает всеобщего production deployment каждого варианта. Compressed-state scratch/commit объяснены без переноса смешанных cache ratios из чата.
- В hypothetical cloud arithmetic32USD/hour не заявляется актуальным price.100Mtokens/day требуют1157.4tokens/s в среднем; один540tokens/sserver этого не обеспечивает. При speed+60% reciprocal decode time уменьшается37.5%, не60%.

Числа проверены event enumeration и независимой арифметикой; это учебные examples, не benchmark выполненного пользователем inference.

## C21. Sampling и structured output после уточнения пользователя

**Где:** S5,120–413; связь с beam изS4. Глава инференса Е расширена по вашему возражению greedy+high temperature, sampling pipeline и formal grammar.

При T>0 scaling сохраняет ranking даже для близких logits; penalty order может изменить argmax по другой причине. Top-p зависит от изменённых probabilities и не коммутирует с temperature или некоторыми top-k combinations. Typical sampling использует self-information и entropy, а не«энтропию отдельного token». Формулы penalties объявлены как примеры, не всеобщие API contracts.

Grammar гарантирует prefix viability; complete-valid output дополнительно требует accepting state и успешного завершения. DFA/PDA/tokenizer-aware transitions разделены. Finite negative mask не является математическим−∞. После nonempty grammar mask и корректной нормировки низкая исходная allowed mass не делает top-p пустым: пример(.97,.02,.01)→(0,2/3,1/3). Ошибка возникает при ином порядке или несовместимых ограничениях.

JSON Schema выражает правилоstatus=error→required error_message черезif/then. Вставлен пример2020-12 сadditionalProperties/minLength; JSON syntax проверяется скриптом, semantics примеров сверены со standard keyword rules. Полный запуск внешнего JSON Schema validator не заявляется. Возможности стандарта и subset generation backend различаются. Pydantic/custom validators и application wrappers не объявлены автоматически одним constrained decoding mechanism.

Отдельный toy example показывает, что local token mask не обязательно воспроизводит global conditioning исходной LLM на валидность всей строки. Sampling/schema/tool execution проверяются по разным invariants. Retry estimates требуют assumption independence; source validity/overhead percentages не перенесены как benchmarks.

## C22. Long context и caching на проекте с несколькими вопросами

**Где:** S5,417–1034. Глава инференса Ж расширена по исходному TechCorp примеру:8Krules+15Kdocs, триquestions, изменениеprefix, TTL/counters/cost и theoretical long-context mechanisms.

Исправлена arithmeticKV:80layers,8KVheads,128head_dim,2bytes дают320KiB/token,2.5GiB/8K и40GiB/128K, не640KiB/5GB/82GB. Bytes и GiB не смешиваются. Prefill quadratic attention, decode reading history и linear KV storage разделены. Sliding receptive field1+L(W−1) не является guaranteed retrieval. Дано RoPE rotation identity иPI example; extension factors изчата не универсальны. FlashAttention memory/compute иRing per-device/aggregate scaling не смешиваются. Jamba приписываетсяAI21, неMistral; архитектурные альтернативы не представлены как runtime flag.

Восстановлены вопросы оcircuit breaker,service ports иsecrets. Client/session reuse не тождествен providerKVcache. Stable rules могут давать partial hit даже при dynamic RAG; early personalization дробитprefix, но смысл/доступы нельзя менять радиhit. Built-in Pythonhash concatenated strings не является устойчивым полным computation identity. Artificial padding доthreshold не объявляется полезной оптимизацией.

Числа8.7/2.1/1.9s иvendor rates/TTL изчата не считаются актуальными contracts/измерениями. Output lengths340/280 различались; они не выделяют prefill savings. Counters требуют понимания знаменателя иучётаcached вtotalinput.90%full hits из1000 означают100misses/900reads;3misses означают99.7%hits. Даны согласованный token budget иtoy break-even formula без заявлений текущих API цен. Input cache, answer cache, retrieval иsummary остаются разными mechanisms.


## C23. Позиционные механизмы NLP и подробное обучение CV Transformer

**Где:** S2, строки 1330–1381; S7, раздел ViT и перечисление Swin/MAE. Углублены foundations В и CV А по дополнительному приоритету пользователя.

RoPE не прибавляет готовый позиционный вектор к token embedding: стандартная схема вращает пары координат Q/K. Выведена зависимость dot product от относительного угла, показаны сохранение нормы, локальный backward и пример периодичности. Causal mask уже нарушает полную симметрию перестановки; формулировка «совсем не знает порядок» без этой оговорки слишком сильная. Семантические роли голов не заданы заранее.

Оконный attention, смена границ и patch merging Swin объяснены раздельно. Линейная по числу токенов цена оконного attention предполагает фиксированный размер окна и ширину; отношение числа scores не равно ускорению полного forward. Схема сдвинутых окон иллюстрирует геометрию без циклической упаковки; для реального roll дополнительно нужна mask, запрещающая искусственные связи краёв.

MAE отличается от BERT MLM: encoder получает только видимые patches, decoder возвращает пространственные места с mask tokens, цель — пиксельные компоненты. Восстановление индексов, средний masked MSE и его градиенты разобраны на собственных числах. 75% маскирования и отношение attention pairs не объявлены универсальным оптимумом или измеренным ускорением.

Добавлены четыре SVG-схемы; вместе с предыдущими их семь. Проверены границы текста SVG, загрузка офлайн и мобильный layout, вращение RoPE, обратная перестановка MAE, причинность двух слоёв с bidirectional отрицательным контролем и соединение окон. Эти проверки подтверждают отдельные механизмы и отображение, но не завершают содержательную сверку всех 95 глав.


## C24. Multi-head, cross-attention и полный цикл BERT MLM

**Где:** foundations Г и BERT Б. Разбор нескольких голов продолжает вопросы S2 о Q/K/V и головах; полный MLM — дополнение по действующему BERT-плану, не выдаваемое за содержание отсутствующего подробного чата.

Числовой пример разделяет concat и усреднение attention weights: [3,1] после WO даёт [2,7], средняя heatmap при тех же V — [0,12]. При фиксированной D количество обычных Q/K/V/O весов не растёт пропорционально числу heads. Cross-attention имеет T×S scores, Q от decoder, K/V от encoder; padding source требует маски, а не нулевого V. Добавлен вывод градиента к encoder по двум путям K и V и сверка конечными разностями.

В BERT разобраны исходные IDs, отдельно выбранные targets, MASK/random/unchanged corruption, неизменённые цели, отсутствие next-token shift, контекстные градиенты и tied embedding paths. Исторические проценты имеют разные знаменатели; маленький пример специально не имитирует 15%. Случайная замена может совпасть с исходником. Нормировка по targets и по examples даёт разные objectives; static/dynamic masks и границы train/validation разъяснены.

Три новые SVG-схемы доводят число иллюстраций Transformer до десяти. Проверены числовой forward, padding, backward encoder, MLM averaging и masking coverage; это не проверка качества обученного checkpoint и не доказательство завершения всех 95 глав.


## C25. BERT encoder interfaces и fine-tuning без смешения масок и единиц

**Где:** BERT А и В, модули 1–5 и 10–14. Углубление по плану с использованием Transformer/PEFT предпосылок S1–S3; новые BERT-примеры не приписываются чатам.

Token IDs, строковые subwords и offsets исходного текста различаются; пример фамилии восстанавливается по полуоткрытым интервалам [8,23). SEP/type IDs не закрывают cross-segment attention, batch items остаются независимыми. Post-LN и pre-LN не переставляются без изменения функции. Embeddings складываются, состояния PAD могут оставаться ненулевыми. Приведены mask-aware pooling и его градиент; padding pairs 80000→40400 — расчёт, не benchmark времени.

CE/BCE при одинаковых logits имеют разные targets и градиенты. Замороженные параметры не означают no_grad всего encoder с trainable adapters. Head/label map/tokenizer сохраняются вместе с PEFT. Для N=1000, world size2, batch8, accumulation4 последняя группа содержит глобально16+16+8=40 examples и требует весов0.4/0.4/0.2 при среднем по примерам, а не произвольного деления всех microbatch means на4. Scheduler units и objective normalization разделены.

DAPT/TAPT не заменяют task labels и не гарантируют downstream улучшение по одному MLM loss. Test-text adaptation обозначается отдельно. Vocabulary expansion требует новых обученных строк, согласованных IDs/config/head/optimizer. Все расчёты и конечные разности в tmp/verify_bert_interfaces.py проверяют объявленные учебные механизмы; качество реальной модели ими не заявляется.


## C26. Прикладные головы и геометрия embeddings с восстановлением вопроса S6

**Где:** BERT Г–Д; S6:685–845 прочитан двумя короткими диапазонами. Сохранён вопрос пользователя о том, почему LLM должна понимать внешний embedding. Дополнительно просмотрены S6:182–291 (частично обрезанный вывод),1091–1190 и1281–1373; это не означает завершения всей главы RAG по S6.

Исправлены обобщения исходного ответа: embedding-модель не обязана принадлежать отдельному от Transformer классу; ширина не определяет назначение; «никогда не подаётся в LLM» не является общим правилом. Обычный retrieve-then-read возвращает тексты. Original RAG BART читает encoded query/passages, а не непосредственно DPR search vectors; CLIP сам по себе не генеративная LLM. Различие проверено по primary RAG paper, без переноса неподтверждённых архитектурных аналогий из чата.

Разобраны независимые и structured головы, CRF enumeration/forward/Viterbi/backward, width pruning ceiling, directed NLI, QA constraints/window/null scores и зависимость few-shot pairs. Смена общей константы QA logits сохраняет внутривоконные distributions, но меняет raw cross-window scores; один threshold не гарантирует калибровку.

Геометрия: одинаковая размерность не доказывает совместимость пространств; ортогональный поворот сохраняет внутримодельные scores и может перевернуть смешанное ранжирование. Contrastive loss не требует всем positives cosine≈1 и negatives≈0. Выведен query gradient, multi-positive alternatives, singleton negative-pool degeneracy, coordinate-collapse и nontransitive duplicate chain. Векторы памяти посчитаны отдельно от индекса/текстов. Две новые SVG-схемы; всего12. Числа и gradients проверены в tmp/verify_bert_tasks_geometry.py; эти проверки не доказывают качества обученных моделей или полноты всех95 глав.


## C27. DeiT training и retrieval по вопросам S6/S7

**Где:** CV А и BERT Е. S7:790–806 кратко называет DeiT; полный teacher/student механизм добавлен по первичным paper/losses.py/models.py. S6 BM25:860–1091 прочитан диапазонами860–949,950–1045,1046–1091; multi-vector1191–1280 дополнен ранее прочитанными соседними частями.

DeiT distilled и обычный variant разделены. У DIST своя head, hard target teacher argmax, soft KD teacher→student KL. Teacher не обновляется; отключение autograd не заменяет eval. В опубликованном soft loss знаменатель B×C, а eval forward усредняет logits, не probabilities. Ручной пример и gradients проверены; дополнительная token/phead арифметика не выдаётся за полный training overhead. Сравнения ViT/CNN из S7 не перенесены как универсальные законы.

Восстановлен вопрос о keywords и полный BM25 примерN1000/avgdl200/length300/f4,2/df30,5. Score11.7106766 отличается от11.73 в чате из-за раннего округления. ANN recall отделён от semantic recall. Cross-encoder candidate ceiling, pairwise margin gradient и scale/threshold ограничения раскрыты отдельно.

Max aggregation — variant SPLADEv2, не исходный sum SPLADE. Разобраны derivative log/ReLU/max, sparsity proxy, postings и RRF. ColBERT MaxSim не взаимно однозначный matching; gradients по выбранным pairs и indirect attention paths разделены. Четыре document views S6 не отождествляются с token-level vectors. Raw bytes посчитаны точно; source latency multipliers, HNSW bytes и fixed corpus thresholds не считаются benchmarks. Parent retrieval не требует индексировать оба размера chunks.

Добавлены две SVG-схемы, всего14. Вычисления и конечные разности проверяет tmp/verify_deit_retrieval.py; это не измерение качества обученного checkpoint и не подтверждение завершения всех95 глав.


## C28. Длинный encoder, evaluation и operations по BERT Ж–И

**Где:** BERT Ж–И, модули30–40; перенос long context/serving предпосылок S4–S6. Дополнительно прочитаны S6:2201–2289 (вывод частично обрезался),3116–3180 и3434–3561; полный deployment источник ещё требует отдельной сверки в инфраструктурных главах.

Выведено условие overlap O≥A−1 для покрытия spans, отдельно от multi-hop evidence и QA correctness. Local receptive field и двухслойный путь через global token показаны явно. Token inflation не равна quality loss; language average зависит от весов evaluation. Layout boxes, page coordinates и OCR errors не смешиваются с task head accuracy.

Calibration сохраняет class argmax одного примера, но multiclass confidence order между примерами может измениться. Binary NLL/Brier, majority macro-F1, nDCG и IG interaction example пересчитаны. Endpoint saliency, probe decodability и causal use разделены. Paired/group evaluation не заменяется независимым resampling связанных token rows.

Сжатие: tiny vector MSE может перевернуть near tie; int8 payload не равен всему VRAM; уменьшение encoder не уменьшает corpus vectors. BERT bidirectional attention не поддерживает обычный causal append-cache contract, хотя некоторые отдельные проекции/identical-input cache переиспользуемы. В S6 формула total_time/N×B не измеряет mean request latency: для100 requests в10 batches по0.2s правильное среднее1.1s, формула даёт0.2s. Source benchmark batch/A100 не принят за измерение. Queue count зависит от длины workload; enqueue плюс direct processing без dequeue не является полноценным worker queue. Наличие weights не делает одинаковые encoder replicas автоматически session-stateful.

Две новые SVG-схемы (local/global и layout), всего16. Числовые проверки и finite examples находятся в tmp/verify_bert_special_eval_ops.py. Расширение всех девяти BERT-глав не завершает общую переработку95 глав или финальную сверку источников.


## C29. Deployment: E.1–E.2 S6 восстановлены с разбором ошибок

**Где:** inference И. S6:2206–3904 прочитан тематическими последовательными блоками; место с обрезанным combined output3710–3758 повторно прочитано коротко. Возвращены container/model artifact, cold-start alternatives, MIG/MPS/time-slicing, health checks, HPA signals, latency/throughput benchmark, routing, prioritization и graceful degradation. Первичные Docker/Kubernetes/PyTorch/NVIDIA docs использованы для проверки mechanisms; код чата не объявлен работающим deployment.

Image/volume/registry — варианты artifact lifecycle, не запрет weights в image. Наследование stages сохраняет предыдущие layers; selective builder/runtime separation объяснена отдельно. Port mappings8001:8000/6333:8000 не задают те же host ports для внутренней сети. MODEL_PATH требует launcher support; fragments безselectors/labels/volumes не являются полным K8s manifest. Mutable files/revisions и неполные downloads не дают воспроизводимости.

Init/preloading не создаёт hot GPU engine main process. PyTorch позволяет сохранять CUDA tensors и загружать с map_location; checkpoint не сохраняет живой process/allocator/KV. state_dict API high-level engine не приравнен nn.Module. У warmup snippet отмечена область видимости sampling_params. Cold start2–5min и N×startup не объявлены нормой: разобраны stage sum и parallel rollout waves. Probes разделены; free GPU threshold/hasattr не универсальны, heavy probe может усилить overload.

Queue/reservation/deadline и waiting/running accounting выведены явно. Source enqueue+direct processing не имеет полноценного consumer path. Формула total_time/N×B не mean request latency. Пример5 observations имеет20%, не5% медленных requests; nearest-rank p95 и linear quantile различаются. Average pod p95 не global p95. CPU/GPU metrics не бесполезны; extended GPU allocation не utilization metric, custom HPA signals требуют metrics API. Stabilization — policy истории, не просто sleep. Pending replicas не дают мгновенной capacity.

MIG profile IDs и source1g.10gb→«20GB» не перенесены как рецепт. MPS отличается от time-slicing; hardware/version/isolation conditions сохранены. Dict нескольких LLM не создаёт общий multi-model scheduler/eviction. Raw weight bytes, KV, host RAM и GPU VRAM разделены. Priority может создавать starvation; polling всех pods на каждый request даёт overhead/stale/herd. Fallback models, cache eviction и circuit breaker меняют contract и не создают прежние guarantees автоматически.

Числа и расписания проверяет tmp/verify_deployment_depth.py (15 checks). Настоящий GPU load test, Docker build и K8s deployment не выполнялись и не заявляются. Полная содержательная сверка всех95 глав остаётся обязательной.


## C30. Application cache и generative monitoring — новое продолжение E.3/E.4

**Где:** inference К. Сохраняется происхождение по плану: отдельного занятия E.3/E.4 в исходной цепочке нет. S5:417–1034 даёт prefix-cache contrast; фрагмент736–807 повторно прочитан для вопроса о портах/других questions общего context. Primary Redis lock ownership и LLM-as-judge paper используются только для соответствующих mechanisms, без заявленного production запуска.

Добавлены различия computation reuse/answer reuse, joint distribution stochastic samples и correlation повторяемых ошибок. Key fields, framed serialization и hashing не смешиваются с semantic equivalence; canonicalization зависит от application contract. TTL/eviction/invalidation различаются; source-version publication и поздний writer показывают stale resurrection. Lease ownership защищает release, но не любой publication write.

Candidate rate, accepted rate, accepted-hit precision и recall допустимых opportunities имеют разные denominators. Semantic threshold не транзитивен; правильный query match не исправляет неверный stored answer. Cost formula выведена с break-even; rejected validations, token-weighted saving, p95 и coalesced waiters учтены отдельно. Payload521.6MB — объявленная арифметика без index/allocator, не service benchmark.

У generative задач возможны deterministic свойства и несколько верных формулировок: отсутствие одного reference string не означает отсутствие correctness oracle. Разобраны trace/version cohorts, confounded hit/miss comparison, delayed labels и repeated-key dependence. Judge rubric, order mapping, position/verbosity/shared blind spots, confusion outcomes и tie conventions отделены от одного agreement/win-rate числа.

Учебные расчёты и counterexamples проверяет tmp/verify_application_cache_depth.py (19 checks). Cache backend, provider billing и production rollout не запускались; фактическая практика пользователя не переопределяется.


## C31. Tool calling: protocol, training и runtime не тождественны

**Где:** inference З. Восстановлены вопросы S5:2344 о скачанных локальных моделях/agent pipeline и S6:1846 о researcher, достаточности поиска, AutoGen/LangGraph и делегировании. S5:2829–2937 и S6:4–66 повторно прочитаны для проверки объяснений обучения. Весь D1–D6 не объявляется окончательно сверенным: подробные chapters applications ещё впереди.

Исправлено ложное универсальное утверждение S5 об отдельном tool head и общих для всех моделей special tokens. Объяснена обычная autoregressive сериализация calls через vocabulary projection; отдельный router возможен, но не обязателен. По закрытому API архитектура не выводится. Prompt/template, SFT и constrained decoding совместимы и дают разные виды контроля. SFT не гарантирует строго валидный JSON, существующее имя, правильные аргументы и разрешённость операции. Утверждение S6 «LLM не умеет считать» заменено отсутствием общей гарантии точности.

Не переносится абсолютное «локальные приложения никогда не имеют tool loops»: проверены current primary Qwen protocol/templates, Ollama multi-turn example, LM Studio endpoint tools и SDK .act(), LangGraph workflow/agent boundary. Соответствующие ссылки стоят в тексте. Исторические версии/endpoints из чата не объявлены текущим обязательным API.

Уточнено, что не каждый participant/node во всех frameworks является LLM+tools+loop. Разделены policy участника, coordinator, deterministic executor и graph structure. Упоминание subagents в исходном вопросе не используется как разрешение делегировать текущую работу, и не выдаётся за доказательство внутреннего устройства конкретного редактора.

Добавлены state transitions, separate model/tool budgets, finish proposal/acceptance, call-ID correlation, streaming completion, scope enforcement, checkpoints и ambiguity внешнего effect, idempotency, parallel merge и budget allocation. RAG прослежен через text/vector/IDs/text, filtered retrieval, versions/citations/context budget и oracle ablations. Учебные числа и counterexamples проверены tmp/verify_orchestration_depth.py; реальный framework pipeline/deployment не запускался.


## C32. Structured contracts и полный tool lifecycle по S5/S6

**Где:** applications А–Б. Повторно прочитаны S5:120–242 (C.2),1463–1559 (templates/parser/composition),2938–3041 (get_weather/analyze_data schemas; окончание экспорта обрывается, неизвестное продолжение не восстановлено выдумкой); S6:4–181 (tools, book_table, advanced patterns и уточнение теоретического режима). Возвращены уникальные входные schemas и их разбор, а не только названия patterns.

В А исправлено смешение JsonOutputParser с decoder masking. Parser, format instructions, явная schema/Pydantic validation и generation constraints разделены; проверена текущая primary реализация LangChain json.py. Комментарий о проверке template variables при создании заменён более точным моментом format/invoke. Instructor не объявляется всегда одним constrained-decoding backend: primary docs различают modes и retries.

Уточнены properties/required/null/additionalProperties/default и выразимость if/then/else. Date format assertion требует поддержки/конфигурации; pattern и grammar не доказывают календарный или бизнес-смысл. Prefix-valid не равен completed object; backend schema subset не равен всему стандарту. Unsupported исходные 90–95% adherence/overhead не выданы за measured rates. Mask/top-p порядок показан численно без тезиса, что малая valid mass математически запрещает sampling.

Добавлена полноценная внешняя Draft 2020-12 schema found/not_found/conflict, source/unit/cross-field semantics, repair/re-ask/versioning и business-calendar example. Для проверки установлена изолированная jsonschema dependency в tmp/schema-validation-deps, основной runtime не изменялся. Schema validation выполнена, но поддержка этой schema каким-либо model backend не заявлена.

В Б возвращены погода, бронирование и analyze_data nested config, значение optional/required fields, columns/dataset lookup, tool descriptions и output contract. Сохраняется correction необоснованного guests=2 и system-channel для внешнего tool output. Пример «100+ tools всегда не помещаются» заменён token budget calculation, shortlist recall/selection/args/execution разделены. Idempotency включает стабильный intent, concurrent race, TTL и различие local transaction/external effect. Backoff/deadline, partial observations, parallel IDs/dependencies и terminal outcome разобраны явно. Теоретическое изучение не превращено в production activity пользователя.

tmp/verify_application_contract_tools_depth.py проверяет новые schemas, реальные loose constraints исходного analyze_data и арифметику; не заявляет model accuracy, API execution или deployment.


## C33. State, graph, memory и frameworks: механизмы и ошибки source snippets

**Где:** applications В, дополнение applications А. Повторно изучены S5 D1–D3 тематическими диапазонами1035–1202,1274–1457,1560–1719,1719–1969,1970–2200,2210–2344; local section2344–2710 и2711–2823. Большие outputs местами обрезались; ключевые graph/interrupt/artifact/completion участки повторно прочитаны короткими блоками. S6:1374–2197 прочитан тематическими диапазонами framework/coordination/termination; ключевой вопрос1846 повторно восстановлен. Полный финальный аудит source coverage всё ещё необходим; templates/parser часть ранее раскрыта в А, calls/execution —в Б.

Stateless вызов не объявляется детерминированной чистой функцией любого API. In-context history, durable state, source snapshot и weights различаются. Интеграл sin решаем аналитически, tool не является обязательным для всякой математики; проверен247×891. Source pipeline с for не называется «не умеющим циклы». TypedDict не даёт автоматической runtime validation, typed state не исключает mutable/serialization errors.

Исправлены source graph counterexamples: unregistered end node, только tool_calls[0], overwrite results по tool name, неявный state внутри visualization, keyword completion и нерасходуемые на всех back edges counters. Reducer delta/append/ID-update, dict union conflicts и usage-event dedupe показаны отдельно. Framework-independent routing snippet проверен; реальный SDK graph run не заявлен.

Checkpoint не делает external effect атомарным. In-memory persistence не переживает процесс; сохранённое «Привет» не даёт отсутствующего имени. thread_id не заменяет ACL/user scope. Dynamic interrupt и static breakpoint не смешиваются; primary LangGraph docs подтверждают node restart и повтор кода до interrupt. У subgraph отдельный contract/budget; streaming events и terminal success различаются.

Memory buffer/window/summary/hybrid сохранены с token example и оговоркой потерь/стоимости. Нет неподтверждённого «большинство production делает X». Current primary LangChain overview подтверждает agents on LangGraph, поэтому старое «LangChain только linear, циклы невозможны» не используется как текущий факт. Graph/message/role/data акценты frameworks не являются жёсткими запретами; AutoGen имеет программные termination conditions. Закрытые internals редакторов не выводятся из интерфейса. Source scores production-ready/best не приняты как результаты измерений.

В local artifact example восстановлены load_embeddings/describe_clusters/get_ablation_row, различие shape и silhouette, row identity/revisions и count vs sum по size. Не обещается перенос любого endpoint одной строкой, универсальная скорость/VRAM/минимум параметров. Исходное напоминание S6:179–181 о теоретическом режиме соблюдено.

Исправлен совет перебирать статистические тесты до significance: это не обычная self-correction, показан риск20 независимых tests. Finding count не заменяет уникальное evidence/покрытие. В А добавлены контрпримеры GBNF source2755–2786: разрешены только age без name, повтор keys и leading-zero007; это не полный нужный JSON contract. Recognizer проверяет объявленное ASCII подмножество правил, конкретный compiler/backend не запускался.

Числовые примеры, source counterexamples и routing проверяет tmp/verify_application_state_depth.py; полный final source-content audit остаётся в плане.


## C34. RAG, BM25, multi-vector: восстановленные вопросы и cost/quality boundaries

**Где:** applications Г. S6:182–1374 повторно прочитан последовательными диапазонами182–351,351–520,521–685,685–834,834–1018,1018–1188,1188–1374. Сохранены вопросы об интерпретации embeddings, происхождении BM25 keywords и росте БД. Исходные неподтверждённые enterprise benchmarks и рекомендации по universal thresholds не приняты как measurements.

Возвращены четыре chunking strategies, extraction tables/captions/units, explicit overlap/call/token accounting, publication revisions и source grouping. Semantic sentence count не равен числу serial model forwards. Размеры chunks и generic embedding context limits не выдаются как вечные constants.

Embedding table/token states/retrieval vector различаются; query/document encoders могут быть разными, но совместимыми. Corrected original RAG/REALM text retrieval и multimodal interface, без тезисов «retriever никогда LLM» или «все positives cosine1, все negatives0». Primary RAG/REALM/HyDE papers и FAISS docs проверены; direct embedding injection не приписана ordinary RAG.

BM25 source toy corpus просчитан с объявленной positive-IDF convention, не выдан за точный rank_bm25 output. TF/DF, saturation, length change, analyzer compatibility, morphology/IDs/negation/phrase boundaries раскрыты. Source округлённый score11.73 исправлен на11.710677; len600 даёт8.968247. Sparse-neural choices не представлены универсальными winners.

MetricType FAISS squared L2, cosine/IP distinction, zero norm и encoder-space migration проверены. FAISS умеет read/write index; vector library не приравнена DB, порог<1M не использован как закон. ANN neighbour recall, relevance recall, candidate cutoff и final rerank recall разделены. Source «rerank без потери recall» уточнён. RRF list.index missing-ID bug исправлен в учебной функции.

Восстановлены expansion/HyDE/step-back и parent retrieval. Generated hypothetical document не source truth. Parent vector не обязателен; child vector mapping к stored parent text не гарантирует2x payload. Multi-view document representations не смешиваются с token-level ColBERT. Один vector_type не делает несовместимые spaces совместимыми.

Multi-vector coordinates, text, auxiliary representations, metadata, index/replicas считаются отдельно. Source100k scenario пересчитан с объявленными2KiB и искусственно заданным overhead:708.4MB→2069.2MB≈2.92095; это не HNSW measurement. Количество embeddings, API requests, tokens, cost и wall-time имеют разные units. Неподтверждённые1.5–2x slowdown,+4%recall,+300%cost,+180%latency,80/20 и50k/500k thresholds не используются как evidence. Проверка пользы —ablation на frozen protocol и actual costs, без fake user runs.

Уточнены context packing, freshness/live API, cache embeddings vs answers и RAG applicability к code/math. Малый corpus не означает обязательный fine-tuning; context/policy/data-store задачи различаются. Учебные calculations и RRF проверяет tmp/verify_application_retrieval_depth.py. Реальный encoder/index/paid API не запускался.


## C35. Evaluation приложения: units, oracle и границы judge/benchmark evidence

**Где:** applications Д. Повторно прочитаны S3:820–918 короткими диапазонами (после первого обрезанного вывода) и S6 retrieval metrics/ground truth/generation sections. Подробная source RAG секция182–1374 прочитана при Г. Финальное source утверждение «всё освоено/проект готов» не принимается за подтверждение пользовательской практики.

Добавлены task/query/chunk/claim/call units, qrels granularity, pooling/unjudged, answerability vs failed execution, clicks/annotation bias, document/group/time split и dev/holdout separation. Source inverted benchmark не объявлен надёжным detector memorization; reverse task может быть недоопределена.

Исправлено source определение MRR как средней позиции: это среднее reciprocal rank. Binary hit и sufficient evidence set, macro/micro, AP/MAP, nDCG gain/IDCG/cutoff/incomplete qrels разобраны отдельно. Дубли vector hits не считаются независимыми sources. При R=0 и меньшем returned count нужны объявленные conventions.

Groundedness/correctness/claim-citation support/completeness имеют разные denominators. Даны source-version counterexamples, citation presence vs entailment, answerable/unanswerable confusion и coverage. Always-abstain не получает искусственную perfect accuracy. Oracle intervention и matching cohort разобраны; net20points может скрывать30 changed outcomes, не все изменения объявляются исключительно retrieval errors.

Добавлены nested evidence funnel с conditional rates, task-level environment oracle, duplicate effects, failures в denominator, first-attempt vs budget success, finite-pool pass@k, общая cost-per-success и nearest-rank latency convention с timeouts. Не заявляется empirical superiority какого-либо encoder/judge/runtime.

Source «PPL мертва/не видит ошибок/никакой корреляции» заменён различием target prediction и task utility. Human preference не единственная ценность. Alignment tax не объявлен неизбежным, source universal10–15% rule не принят; points и relative percentage различаются. Judge rubric, ties/missing labels, blind spots и paired/group resampling объяснены с независимыми toy checks. Reference-free не означает отсутствие data/assumptions. Primary Ragas/ARES/MT-Bench papers и IR textbook проверены; library metric APIs не выдаются как неизменный standard. Source50 prompts+json.loads —smoke checks формы, не доказательство tool-task reliability.

tmp/verify_application_evaluation_depth.py проверяет арифметику и counterexamples; real model benchmark, human labeling campaign и deployment не выполнялись. Все пять application глав углублены, но итоговая содержательная сверка всего курса остаётся необходимой.


## C36. Lifecycle overview — синтез чатов, а не доказанный training project

**Где:** lifecycle А, пункт1 действующего плана. Сохраняется честное происхождение: S3 обсуждает adaptation/evaluation, S5/S6 runtime и deployment; отдельный systematic agent-training course в источниках не подтверждён. Новое diagnostic объяснение явно помечено как дополнение по плану. Пользователю не приписаны training runs, production artifacts или labels из несуществующего эксперимента.

Разделены weights/prompt/data/retrieval/environment/control/memory/budget и уровни intervention. LoRA —parameterization, не objective; prompting/memory/graph не являются gradient update автоматически. Для пяти source-inspired failures показаны causal hypothesis, intervention и опровергающее наблюдение. Условный format-only ceiling68→74 не выдан за universal SFT limit; primary-cause и overlapping labels различаются.

Constrained candidate selection отделено от одного maximum success; training/amortization600/(.10−.08)=30000 —заданный toy scenario, не pricing/recommendation/measurement. Baseline compute, training loss/reward и independent task metrics не смешиваются. Artifacts/version bundle связывают data, computation, environment, evaluation и deployment; rollback weights не равен восстановлению всей системы. Offline/integration/load/rollout/monitoring и feedback validation разобраны как разные этапы.

tmp/verify_lifecycle_overview_depth.py проверяет declared arithmetic и constraint counterexample; full-scale model training, load test, paid compute и rollout не выполнялись. Остальные9 lifecycle глав требуют углубления.


## C37. Среда и behavior cloning: новые formal/training дополнения по плану

**Где:** lifecycle Б–В, пункты2–3 плана. S3:228–339 повторно изучен тематическими кусками SFT/masking и вопроса о книгах после большого обрезанного вывода; S6 tools/orchestration служат уже разобранной основой, а formal MDP/POMDP/BC материал честно обозначен как новое углубление. Не объявлено существующее agent training занятие/пользовательский training run.

В Б разделены environment state, observation, belief, Transformer hidden states, action/token/model-call units и read-only knowledge/budget change. Добавлена Bayes-update illustration, полностью specified corpus R1/R2/D/X и source-backed success, stale failure и timeout recovery. Toy reward/discount, delayed credit, step-cost scale и reward-hacking counterexamples проверены.

Исправлено возможное обобщение о horizon: intrinsic finite task horizon может быть termination failure, external collector cap —truncation. Current primary Gymnasium docs подтверждают различие и соответствующий bootstrap target; remaining time нужен для Markov state finite-horizon задачи. No-results/timeout/denied, multiple budgets, clarification и reset/seed compatibility не смешиваются.

В В восстановлен source вопрос о книгах и уточнено, что raw dialogue не равен environment-action demonstration, а технические тексты могут быть преобразованы в проверенные примеры. Behavior cloning objective, teacher forcing, temporal leakage, token-vs-trajectory weighting и допустимые alternative actions разобраны отдельно. Positive SFT по failed actions не становится автоматически anti-imitation; valid prefixes/corrections требуют явных labels/masks.

Добавлены полный protocol trace, block token counts, labels/causal shift, gradient через masked context, pad=EOS trap, packing/truncation и stable ID boundaries. Primary HF chat templates/TRL assistant-only configuration проверены; само название trainer/role не гарантирует desired mask. SFT/LoRA/runtime invariants и ReAct prompting не объявлены одним механизмом. Sanity overfit —только diagnostic procedure, не held-out score.

tmp/verify_lifecycle_environment_cloning_depth.py исполняет маленький declared toy evaluator, проверяет returns/probability/masks и conditional-gradient illustration. Настоящие LLM/Gymnasium/TRL training/integration не выполнялись, external API/production corpus не использовались. Следующие7 lifecycle глав и окончательный source-content audit остаются в работе.


## C38. DAgger и reasoning — systematic дополнение, не восстановленные training runs

**Где:** lifecycle Г–Д, пункты4–5 плана. Чаты не содержали отдельных systematic DAgger/STaR/PRM занятий; это явно указанное новое углубление, связанное с S3 SFT и S5/S6 action/state/recovery материалом. Проверены primary DAgger, STaR, Let’s Verify Step by Step и Reflexion papers; их отдельные benchmark цифры не переносятся на пользователя/любую LLM.

В Г раскрыты induced histories, iid batch vs sequential rollout, conditional mixture metrics и ограниченные error-propagation illustrations. Union bound/квадратичная coupling-интуиция не выданы за универсальную точную оценку. Executed action, expert label и intervention разделены; actual observations не заменяются teacher counterfactuals. Step-wiseβ и episode-wise mixture различаются, приβ.1,H20 expected interventions2, но хотя бы одна assistance≈.878423. Teacher label queries могут быть чаще executed interventions.

Уточнена teacher privileged information: expert может знать hidden state по setup, но student targets должны учитывать information limit. Две неразличимые equiprobable histories с разными terminal answers не становятся разрешимыми из-за aggregation. Dataset union/multiset/weights, old protocol labels, recovery budget, irreversibility и equal annotation/training budget разобраны. Source-independent illustrative retail policy14days прямо названа вымышленной; единая report toy среда30/90days продолжается без legal claim.

В Д разделены prompting, inference candidate selection, filtered-data collection и gradient training. STaR answer-conditioned rationalization не становится test input; correct final filtering не доказывает всех steps. ORM/PRM label semantics и independent oracle проверены. Product scores не объявляется joint probability: chain-rule conditionals могут перемножаться без independence, arbitrary marginal/heuristic scores —не автоматически.

Noisy-verifier example полностью просчитан для stop-first-accepted max3: correct.689904,wrong.114984,reject.195112,expected attempts1.9164,cost2.29968 при заданных rates/cost. Это toy assumptions, не реальная judge accuracy. Oracle pass@N и selected success различаются; identical samples не дают independent coverage, repeated false accepts создают selection pressure. Batch/parallel work, beam/tree branches, PRM length effects и test-time budgets разделены.

Reflexion/textual feedback не объявлены optimizer update; memory может сохранять ложную critique. Self-critique repair/regression считаются вместе. Все experiments описаны как процедуры/учебные расчёты. tmp/verify_lifecycle_dagger_reasoning_depth.py проверяет arithmetic и exhaustive toy selector, не обучение/benchmark настоящей модели. Остальные5 lifecycle глав и итоговая source-content сверка ещё необходимы.


## C39. RL и память: формулы, incentives и source-backed state

**Где:** lifecycle Е–Ж, пункты6–7 плана. Повторно изучены S3 RLHF/reward-hacking/четыре роли343–423; большой первый output местами обрезался, финальная часть387–423 взята отдельно. S5/S6 memory/state/cache уже подробно восстановлены в inference/applications; formal RL и long-horizon synthesis явно указаны как дополнения, не пользовательский training проект. Проверены primary PPO/DeepSeekMath и current TRL GRPO docs; текущие library variants не выдаются одной вечной default formula.

Исправлена формула policy gradient первой редакции: при J=E discountedG0 и обычном summation по trajectory positions нужен gamma^t перед return-to-go. Альтернативная gamma1/discounted occupancy convention должна быть объявлена. Численный two-step counterexample проверяет различие. Добавлены score-function и baseline derivation, TD/GAE residuals, terminal bootstrap, causal credit vs shared terminal reward.

PPO clip разобран для четырёх сочетаний ratio/A sign; clipping не проецирует все ratios в интервал. Old/current/reference роли, stored old probabilities, token masks и whole-sequence ratio разделены. GRPO group difficulty, self-including mean, population/sample std, all-equal signal, batching/rollout units и length weighting разобраны явно. Отсутствие critic не делает любой estimator unbiased и не снимает стоимость rollouts.

KL exact/current sampling distribution и sampled log-ratio различаются; nonnegative x−lnx−1 expectation показана с условиями support/distribution. Source «всегда4models на одномGPU» исправлено на четыре conceptual roles классической схемы с вариантами executable reward/shared heads/placement. Reward hacking —риск proxy optimization, не универсальное обещание найти любой exploit.

В Ж сохранены различия environment/history/belief и model weights, memory store/summary/state/checkpoint/KV. Добавлен information-loss counterexample, claim status/provenance/revision, event identity и temporal clocks. Полный12-event read-only toy case показывает stale30days, timeoutc3, observedcurrent90daysc4, pendinganswer иremainingactions2; нет подмены planned→done илиtimeout→no-results. Это вымышленные числа, не реальные правила организации.

Memory cost105000/95000 зависит от момента добавления блока; summary maintenance input/output иrefetch считаются отдельно. Context packing8192→3992→2592 разобран. Version freshness/as-of, replay/external effects, hierarchy иmemory ablations включены. tmp/verify_lifecycle_rl_memory_depth.py проверяет numerical examples иtoy records; LLM training, paid compute, external rollouts иdeployment не выполнялись. Следующие3lifecycle главы иполная source-content сверка остаются необходимыми.


## C40. Trajectory preferences и обучение команды — новые boundaries по плану

**Где:** lifecycle З–И, пункты8–9. S3:509–673 даёт DPO-family основу, S6:1377–2197 — роли/coordination и вопрос об AutoGen/LangGraph. Эти механизмы уже восстановлены в adaptation/applications; первые source paragraphs509–538 повторно прочитаны. Full trajectory learning и MARL здесь явное дополнение, не подтверждённые пользовательские experiments. Проверены primary DPO, Multi-Agent Actor-Critic и MAPPO papers; game benchmark результаты не объявляются качеством LLM teams.

В З сравнение final prose отделено от actual actions/evidence. Разобраны common initial task/scope/snapshot, stochastic outcome luck, admissibility vs preference, resource units, Pareto/tie/incomparable/unknown и Condorcet cycle. Unknown external effect не разметили успехом. Fixed-path environment factors могут сокращаться в policy/reference ratio при одинаковом kernel/support, но не превращают любой desired path distribution в реализуемую causal policy. DPO-like margin посчитан относительно reference, sigmoid не выдан за calibrated task success.

Policy token likelihood исключает environment observation targets, но сохраняет их conditioning для каждого действительного path. Offline data не объявлены автоматически недопустимыми; нужны support/information/protocol assumptions и actual rollout validation. Source universal одно-GPU promise не перенесён. Judge agreement, style, constraints и outcome проверяются раздельно.

В И разделены role/runtime/parameter learning, shared role-conditioned theta, adapters и отдельные policies. Factorization условно independent sampling не означает independent errors. Добавлен common-error mixture с marginal.7 и majority.765625 вместо independent.784; diversity/critic benefit требуют измерений. Cooperative/competitive objectives, self-play vs debate, partner-induced non-stationarity и centralized critic/local execution explained без конкретного framework deployment.

High same-partner success не доказывает transfer: toy diagonal.91,cross.45. Budget/communication context traffic/reservations и team version bundle разобраны. Пользовательский source вопрос о subagents не использован как разрешение их создать: никаких subagents/joint-training runs здесь не выполнялось.

tmp/verify_lifecycle_preferences_team_depth.py проверяет declared math/ranking counterexamples. Последняя lifecycle К и остальные курсы требуют углубления; наличие57markers не доказывает финальную полноту всех95глав.


## C41. Завершение углубления lifecycle и начало курса данных

**Где:** lifecycle К и data А. Итоговая evaluation связывает ранее восстановленные S3:820–918 и S6:3013–3904 с новым протоколом полных trajectories. Data А — дополнение по действующему плану: отдельного полного чата по этому курсу нет. Количество расширенных глав не объявляется доказательством завершения общей задачи.

В К сохранены outcome oracle, recovery/loops/termination, полная стоимость, split/seeds и исходные примеры. Добавлены task/run/attempt/action/candidate units, deadline/unknown/partial и independent exclusion policy. Полный журнал десяти sessions даёт 35 actions, 27 tool calls, один invalid, пять successes, cost 16. Успешные sessions стоят 8: учитывать только их цену нельзя. Validity 26/27 не равна task success. Recovery 2/3 имеет заранее заданный recoverable denominator; остальные failures остаются в полном отчёте.

Уточнены reset/candidate budgets, conditional completed latency, grouped seeds, paired comparisons и test selection. Ноль failures не доказывает q=0: при IID односторонняя 95% верхняя граница равна 1−0.05^(1/n). Для n=10 она около 0.258866, для n=100 — 0.029513. Primary NIST interval source проверен; формула не устраняет dataset/label bias и не является реальным измерением модели.

В data А раскрыты prediction unit, группы/источники/время, доступность feature и lineage. Return за 30 дней — учебный target window, а не правовая норма. Пример 1200 orders содержит 105 known positives, 900 negatives и 195 unknown; naive prevalence 8.75% не подтверждает окончательное распределение outcomes. Отбор early positives также может создавать selection bias.

Различены classification/regression/ranking/retrieval и known-corpus query evaluation против переноса на новые documents. Добавлены mean/median baselines, MAE/RMSE, passport/manifest/hash/schema, sampling/support и association против intervention. При fixed slice accuracies 0.95/0.60 среднее меняется 0.915→0.67 только из-за mixture weights. Datasheets for Datasets проверен как primary motivation документации, а не обязательный пользовательский стандарт.

tmp/verify_session_evaluation_data_task_depth.py проверяет worksheet, арифметику и availability counterexamples. Все десять lifecycle глав углублены; настоящие training runs не проводились. Остальные data/regularization/categorical/clustering главы и итоговая содержательная сверка остаются в работе.


## C42. Data Б–В: split и очистка как проверяемые процедуры

**Где:** data Б–В, пункты2–3 действующего плана. Это подробное дополнение: отдельного полного data чата среди архивов нет. Связи с CV/LLM и исходные источники сохранены; настоящая очистка пользовательского dataset и model benchmark не заявлены. Проверены primary sklearn CV/common pitfalls/imputation docs и Confident Learning paper.

В Б объяснены intended transfer, group/time/source boundaries и три split одного12-row набора:8/4,6/6,4/2 с6 неиспользованными rows. Stratification не решает grouping; sparse rare groups могут делать требования к folds несовместимыми. Frozen independent pretraining и явно transductive protocol не смешаны с незаявленной inductive leakage.

Pipeline защищает только включённые fit stages. OOF, target-encoding cross-fit, final refit, threshold/calibration и nested selection разделены. Strict label readiness t+7+2<20 допускает t10, но исключает t11/t12. Overlap законных past features не объявлен автоматически утечкой; gap samples не равен days. Grid fit accounting27/36 и ограничения fold-SE показаны явно.

В В MAR/MCAR/MNAR — assumptions механизма, а не диагноз по missing rate. Одинаковые observed cells могут иметь разные completions. Показаны global-mean bias, conditional group support, variance shrink, indicators, structural missingness и all-missing schema. Imputation не объявлена oracle.

Units/schema предшествуют outlier score; IQR fences[-1,7] при объявленной linear convention дают candidate, не доказанную ошибку. Clipping/log/deletion различаются; test repair требует независимого правила. Label taxonomy/version/context, raw agreement vs truth и kappa, out-of-sample model probabilities, Confident Learning candidates и targeted-audit selection разобраны с числами. Пример100flagged/20errors плюс900/9 даёт overall2.9%, а не20%.

tmp/verify_data_split_cleaning_depth.py проверяет worksheet и arithmetic counterexamples. Реальные sklearn training, automatic relabel и dataset cleaning не выполнялись. Data Г–И, остальные ML курсы и финальная содержательная сверка ещё необходимы.


## C43. Data Г–Д: дисбаланс, аугментации и зависимости данных

**Где:** пункты 4–5 data-плана. Это углубление по плану со связями с метриками и transforms S7, teacher/distillation S4. Полного самостоятельного data-чата среди архивов нет. Сверены первичные sklearn calibration/threshold/metrics docs, Focal Loss и MMDS Finding Similar Items. Числа учебные; запуск моделей, генерация реального корпуса и benchmark не заявлены.

В Г разделены ranking, probability, решение и нагрузка. Полностью рассчитаны binary confusion и 3-class macro/micro/weighted примеры. AP=5/6 отличается от trapezoidal PR area=19/24 при объявленной последовательности точек; определения не смешаны с детекционной COCO mAP. Для batch из 32 при prevalence 1% вероятность нуля positives около 0.725.

Выведены weighted log-loss optimum и inverse; разобраны sampler vs weights, масштаб нормировки и двойная балансировка. Focal loss включает дифференцируемый множитель; ошибочные трудные метки не объявлены полезными автоматически. Порог зависит от costs/capacity; его подбор на test является использованием test для выбора. Уточнено: ranking сохраняет строго возрастающая калибровка; isotonic может создавать ties. ECE двух bins 0.325 и одного 0.025 иллюстрирует отмену ошибок при агрегации; Brier не объявлен чистой мерой calibration. Prior correction требует label shift.

В Д выведен augmentation risk, различены invariance/equivariance, проверены bbox reflection/crop/resize и continuous bounds vs pixel centers. Дискретные masks не интерполируются как непрерывный цвет. Synthetic provenance, parent split, зависимость вариантов и compute-matched сравнения раскрыты отдельно; n_eff≈198 при 100×100 и rho=0.5 является условной design-effect иллюстрацией, не измерением числа знаний.

MinHash collision=Jaccard объяснено через минимум объединения; проверено полным перебором 120 перестановок, совпадений 72. LSH banding задаёт вероятностный candidate filter, а не точный threshold. Компонента A–B–C не доказывает близость A–C. Byte/canonical/semantic identity и grouping vs удаления различены.

tmp/verify_data_balance_augmentation_depth.py проверяет 19 арифметических примеров и механизмов. Четыре data-главы и остальные общие ML курсы ещё требуют углубления; затем нужна содержательная сверка всех 95 глав.


## C44. Data Е–Ж: загрузка данных и контракты CV-разметки

**Где:** пункты 6–7 data-плана. Е — дополнительное углубление по плану; Ж сохраняет data-механизмы S7 detection/segmentation/pose/OBB/tracking. Сверены PyTorch DataLoader/reproducibility, Torchvision transforms и COCO mask/evaluator source. Архивы не изменены. Числа учебные: loader profiling, обучение и конвертация реального корпуса не заявлены.

Е: разобраны map/iterable Dataset, sampler/batch sampler/collate и переменная структура targets. Padding для шести длин даёт 204 vs 136 позиций при 133 действительных; условный dense attention workload 7444 vs 4316 не объявлен измеренным speedup. Token/sequence means 0.5 vs 1 и неравные microbatch denominators показывают изменение objective. Attention/causal/loss masks разделены.

Проверены rank-worker partition, distributed padding до 12 индексов при N=10/R=3 и повтор IDs 0/1. Локальный iterable drop_last может оставить 4 объекта вместо 8 при глобальной группировке. Prefetch cursor 12 при consumed 8 требует учёта четырёх ещё неиспользованных batches. Оценка image buffers 36.75 MiB/batch и 294 MiB для восьми batches не включает весь RAM. Разделены GPU async timing, seed streams, persistent worker state и model/optimizer/data restart.

Ж: добавлены непрерывные и inclusive координаты, arbitrary category ID mapping, RGB/dtype/EXIF contract, letterbox vs stretch и inverse. Mask ignore меняет denominator; допустимый class ID после bilinear ещё не доказывает правильность mask. Двумерный COCO column-major RLE пример 3×4 даёт runs[3,2,5,2], area4. Сохранено исправление S7 одномерного counts[2,3,2,2]; overlap не объявлен единственным основанием выбора polygon/RLE.

Уточнена ошибка S7: num_keypoints и OKS v>0 включают размеченные закрытые v=1, а не только непосредственно видимые v=2. Неравномерное растяжение OBB может дать параллелограмм; dot product сторон −1.5 в примере объясняет, почему сырые width/height/angle нельзя преобразовывать без геометрии. Crowd/empty/incomplete, sequence namespaces и finite shuffle buffer раскрыты отдельно.

tmp/verify_data_loader_cv_depth.py проверяет 22 учебных механизма и расчёта. Data З–И, остальные общие ML курсы и финальная содержательная сверка 95 глав остаются в работе.


## C45. Data З–И: текстовый pipeline и статистический вывод

**Где:** пункты8–9 data-плана. З продолжает S2/S3/S6, включая сохранённые вопросы о SFT на материалах и chunking; И является подробным дополнением по плану. Сверены Unicode normalization, SentencePiece, HF chat templates/WordPiece, BPE paper, NIST proportions и SciPy bootstrap. Неизменные чаты не переинтерпретированы как реальные benchmark.

В З добавлены source/extraction/derived layers, Unicode code points/bytes и NFC/NFKC, связь tokenizer IDs с embeddings. BPE pair counts 3/2/1 и greedy WordPiece counterexample объясняют механизм. SentencePiece не назван одним алгоритмом; исходная специфика trainer не выдумана.

ChatML из S3 не объявлен общим обязательным стандартом. На символических токенах проверены targets YES/END и shift positions4/5. Нулевой prompt-loss не обнуляет косвенный градиент через контекст. Truncation и packed causal visibility 15 vs9 раскрыты отдельно; EOS не заменяет attention isolation. Для L1000/c300/o50 получены4окна и1150позиций; остановка после достижения конца отличается от всех start<L. В S6 recursive pseudocode не принят за полную реализацию, semantic embeddings можно batch-ить; старые context limits не объявлены универсальными. Бюджеты embedding/generator и task-token mix показаны явно.

И: риск0.16→0.34 может измениться из-за смеси при прежних условных ошибках; concept drift при прежнем P(X) не виден input-only detector. Importance weights0.25/4 восстанавливают новый риск; weight n_eff≈30.77 является диагностикой концентрации, не универсальной variance theorem. Support и clipping bias объяснены.

Разделены test sampling, train/inference seeds и измерение labels. Wilson80/100≈[.7112,.8666], zero/10≈[0,.2775]; one-sided exact zero-event upper≈.2589 относится к другой sidedness. Paired SE≈.02454 отличается от independent≈.05546 для объявленных100tasks; не заявлен окончательный significance test.

Для5videos×20frames exact cluster-bootstrap enumeration3125replicates даёт условнуюSD≈.1789 противrow plug-in.04 и percentile[.4,1] при объявленном CDF-quantile. Это не доказательство достаточности5groups. Group/object estimands.5/.2 разделены. Block assumptions, selection/maxseed, 50independent tests false-alarm probability≈.9231, sequential peeking и delayed/selective labels дополнены. Реальные модельные confidence intervals и monitoring не выполнялись.

tmp/verify_data_text_statistics_depth.py проверяет23учебных механизма и расчёта. Все9dataглав расширены;28глав остальных общих ML курсов и финальная содержательная сверка95глав остаются в работе.


## C46. Regularization А–Б: фундамент и параметры штрафов

**Где:** пункты1–2 и практики penalties/AdamW regularization-плана. Это дополнение: отдельного полного чата по регуляризации нет. Связи с S2/S3 PEFT и Transformer сохранены, но полная теория не приписана чатам. Проверены primary AdamW paper, sklearn linear-model docs и PAC-Bayes tutorial.

А: train/validation числа сопоставляются по режиму, data loss, штрафам, transforms и checkpoint. Вывод bias–variance показывает исчезновение cross terms; пример с independent noise даёт риск6=1+4+1. Shrinkage полезно при theta.5 (MSE.3125), но при theta3 тот же factor даёт2.5 вместо1. Variance не объявлена измеримой по одному checkpoint.

MAP coefficient при average squared loss равен sigma²/(n tau²), при sum — sigma²/tau²; это не универсальный закон validation tuning по объёму данных. ReLU rescaling сохраняет функцию, меняя squared norm2→100.01. Finite-class Hoeffding bound использует |H|, не parameter count. VC threshold example и Rademacher toy раскрывают смысл complexity. PAC-Bayes относится к Gibbs risks и требует prior independence, bounded loss и support; KL penalty RL не объявлен автоматически сертификатом. Нулевой empirical risk пример инвертирует binary kl, а не обещает реальный user risk.

Б: выведены normal equations и положительная определённость ridge. Diagonal X example даёт w(.9,.5), data.17+penalty.53=.70, df1.4. Singular shrinkage и variance отделены от общего parameter count. L1 subgradient/soft threshold, lambda_max и отличие proximal L2 от explicit gradient раскрыты пошагово. Полностью дублированные признаки дают множество lasso решений с sum1.5; ridge/Elastic Net выбирают симметричные solutions при объявленных penalties.

Group threshold сохраняет направление либо выключает группу; overlapping groups требуют иной процедуры. Units/scaling и one-hot coding меняют norm penalty. Штраф вокруг pretrained W0 отличается от shrink к нулю; factor penalties LoRA не равны norm update.

Coupled preconditioner shrink(1.98,1.998) отличается от decoupled(1.98,1.98); Adam moments получают разные gradients. First scalar steps1.99/1.988 проверены при нулевом epsilon. Cumulative decay.999^100≈.9048 показывает зависимость от LR/числа фактических updates. Parameter-group policy и отсутствие gradient не подменены универсальным правилом.

tmp/verify_regularization_foundations_penalties_depth.py проверяет24учебных расчёта и механизма. Реальное model training и benchmark не выполнялись. Регуляризация В–К, категории/кластеризация и финальная содержательная сверка95глав остаются в работе.


## C47. Regularization В–Г: траектория и стохастические операторы

**Где:** пункты3–4 и практики early stopping/dropout. Дополнение по плану; S2 lora_dropout и S7 classifier/CV noise сохранены как контекст, не объявлены полным исходным курсом. Сверены SWA, Dropout, DropBlock, Stochastic Depth, Shake-Shake/ShakeDrop papers и PyTorch averaging/dropout docs.

В: spectral recurrence выведена с w0=0, ненулевыми singular values и условиями сходимости/неосциллирующего filter. GD steps.8/1.28/1.568 и filters.784/.011952 показывают разные скорости; matching ridge lambda≈1.102/3.307 не одна общая lambda. Patience/min-delta/checkpoint policies различены; повторная selection не названа независимой validation. Schedule units учитывают microbatches/accumulation и фактические optimizer updates.

SWA online mean3 и EMA3.5/3.375 с разными initializations вычислены явно. Усреднение nonlinear predictions и переставленных hidden parameters не тождественно weight average; toy permutation снижает функцию x до.5x. BN buffers и optimizer state требуют отдельного решения, test не используется для fit. Уточнены чрезмерные обещания SWA в плане и granularity-порядок dropout family; stochastic depth не ограничен одним ResNet.

Г: dropout moments выведены через Bernoulli second moment, four-mask outputs0/−4/2/−2 дают mean−1,var5 и ReLU expectation.5 противeval0. Linear squared expected loss3=.5+2.5 показывает data-dependent quadratic term, не универсальный AdamW. Sharing masks меняет covariance1 vs0 при одинаковых marginals.

DropBlock gamma≈.02178 даёт желаемую долю только приближённо; overlap/edges и нормировка фактического числа survivors не объявлены покоординатно unbiased. Stochastic depth сохраняет conditional mean блока, но не гарантирует FLOPs savings. Shake-Shake backward coefficient может дать1.25 вместообычной derivative−.25. ShakeDrop forward/backward scales и q vs inverted1 eval scaling явно различены.

Transformer attention dropout послеsoftmax имеет row sums0/.5/1.5/2 в toy masks, хотя expectation исходная; causal/padding mask не является dropout. LoRA input dropout не выключает base branch; merge equivalence относится к совместимому eval. MC-dropout не подменяет полную Bayesian uncertainty и требует отдельного управления BN mode.

tmp/verify_regularization_trajectory_dropout_depth.py проверяет25учебных механизмов и расчётов. Реальные model training, speed и uncertainty benchmarks не выполнялись. Регуляризация Д–К, категории/кластеризация и финальная содержательная сверка95глав остаются в работе.


## C48. Regularization Д–Е: нормализации и устойчивость

**Где:** пункты5–6 regularization-плана. Д связывает S2 RMSNorm/Transformer и S7 CV с систематическим новым разбором; Е является дополнением по плану, не объявленным пройденным исходным чатом. Сверены PyTorch BN/LN/GN, Batch Renorm/LN/RMSNorm/WS, FGSM/PGD/VAT/Mean Teacher и spectral normalization primary sources.

Д: shape2×4×2×2 разобран по reduction sets и affine sharing; одинаковые sets GN/LN/IN не объявлены полным тождеством. Epsilon даёт variance v/(v+epsilon), gradient coupling проверен finite differences. BN forward variance5 отличается от corrected20/3; running .4 и1.5667 при momentum.1 не population truth. Accumulation не объединяет microbatch forwards; frozen parameters не фиксируют buffers. Fixed-state folding scalar example даёт2.5 обоими путями.

Batch Renorm r/d с clipping и stop-gradient не подменён прямой running normalization во всех derivatives. LN и RMSNorm различаются shift invariance; ускорение RMSNorm из S2 не объявлено универсальным. Pre-/Post-Norm Jacobians раскрывают путь gradients; normalization по T может передать future information до causal attention. Weight Norm/Standardization не являются спектральным сертификатом. Уточнены чрезмерные формулировки плана о GN/RMSNorm/SN.

Е: raw/normalized perturbation units разделены; фиксированы inner theta и outer update. Dual norms дают linear maxima.3/.2236/.2 для L-infinity/L2/L1. Constrained PGD пример с pixels.95/.2 даёт(1,.12)→(1,.1), прирост.2 из-за pixel boundary. Failed attack не является robustness certificate; budget и clean/attacked качество учитываются отдельно.

VAT self-KL имеет нулевой gradient в нуле, поэтому объяснён Hessian/power-like поиск. Diagonal9/1 example улучшает quadratic.025→.04451 при максимуме.045. KL direction.09152/.10465 и MSE.08 различаются. Π-model не подменён обязательно frozen teacher; Mean Teacher EMA1.2 отличается от uniform2. Dense flip alignment устраняет ложный MSE.49.

Finite power estimate≈3.784 недооценивает true norm4 и не даёт строгой верхней границы. Convolution kernel(1,1) имеет flattened norm sqrt2, полный valid operator norm sqrt3; division на sqrt2 оставляет norm sqrt1.5>1. Composition/residual bounds требуют всей architecture, nonlinearities и margin для сертификата.

tmp/verify_regularization_norm_robust_depth.py проверяет26учебных расчётов и механизмов. Реальные model attacks/training/speed benchmarks не выполнялись. Регуляризация Ж–К, категории/кластеризация и финальная содержательная сверка95глав остаются в работе.


## C49. Regularization Ж–З: soft targets и NLP/RL distributions

**Где:** пункты7–10 regularization-плана. Ж связывает S7 аугментации с отдельным углублением по плану; З использует S2 representations и S3 SFT/RLHF как переход, не объявляя весь новый материал исходным чатом. Проверены первичные label smoothing/Inception, MixUp/Manifold/CutMix, AutoAugment/RandAugment, tying, recurrent dropout, R-Drop и PPO sources.

Ж: soft CE gradient p−q выведен с суммой target1; smoothing меняет направление true-logit gradient и конечный margin log28 vs log18. Положительный минимальный loss равен entropy target. Uniform smoothing соответствует KL(u||p), не KL(p||u), а идеальная probability цель(.9→.86 в binary toy) не обязана быть calibrated к исходной population.

MixUp CE.76009 вычисляется на одном mixed prediction, не автоматически на двух исходных forwards. Beta moments, fixed uniform smoothing commutation и frozen hidden encoder ограничения раскрыты. CutMix clipped region25×30 даёт area750 и weight.925 вместо nominal.84. Spatial targets detection/segmentation не заменены global label mixture. Policy search и банк magnitude требуют development/budget, TV2 vs sqrt2 зависит от boundary convention. Уточнены слишком сильные рекомендации Manifold Mixup в плане.

З: tying означает shared parameter, не две одинаковые copies; input/output gradients суммируются. Shapes/counts и3-token projection example вычислены. Locked masks имеют другую временную зависимость, Gaussian noise E norm²=d sigma² не назван средним norm. Token replacement/MLM/decoder corruption разделены.

Attention entropy зависит от допустимых keys; sign и logit derivative проверены finite differences. R-Drop mean CE≈.36698, symmetric KL≈.09808, total≈.56315 приalpha2; detach меняет gradient, identical outputs даютKL0. Teacher-forced positions сопоставлены; независимые generated histories не объявлены тем же objective. Sequence targets и beam length score разделены.

Entropy к uniform и KL к неравномерной reference решают разные задачи. Bandit optimum rho exp(R/lambda) выводит reward/reference compromise. PPO four cases2.4/1/−1.6/−3 не являются hard policy projection или гарантией KL. Old rollout/reference roles, token reductions и bootstrap target network4.6 vs19 различены. Реальный RL/SFT run не выполнялся.

tmp/verify_regularization_data_nlp_rl_depth.py проверяет24учебных механизма и расчёта. Регуляризация И–К, категории/кластеризация и финальная содержательная сверка95глав остаются в работе.


## C50. Regularization И–К: структурные constraints и выбор алгоритма

**Где:** пункты11–14 regularization-плана. И связывает S1–S3 PEFT с дополнениями temporal/graph/physics; К — самостоятельное углубление по плану. Сверены Manifold Regularization, PINNs authors page, LoRA, SAM, Lookahead и reparametrization/flatness papers. Персональный опыт FRESNEL/NICA не принят за установленный факт или эксперимент; соответствующая формулировка плана уточнена.

И: трёхточечное smoothing даёт(.375,.75,1.875) и не объявлено causal forecast. Seasonal period2 penalty0 при local energy20 демонстрирует разные priors; нерегулярное время меняет discrete derivative weights. Laplacian identity/PSD/gradient раскрыты; harmonic interpolation не задаёт правильные labels без anchors. Normalized null vector sqrt(degree) отличается от constant vector. Transductive graph и test-label leakage разделены.

Decay residual mean≈.68394 при корректном initial condition показывает разные loss terms; нулевое решение удовлетворяет equation, но нарушает initial value. Неизвестный k не идентифицируется одним initial condition. Units и collocation counterexample u=c+sin²(pi t) показывают ограничения finite residual. LoRA rank1 norm100 сохраняет full rank итогового W и может сильно изменить output; factor initialization gradients и residual bottleneck rank разъяснены.

К: SAM linearized inner perturbation(.06,.08), second gradient(3.06,4.32), outer result(2.694,.568). Scalar1.79 против ошибочных1.89/1.59 проверяет restore и clearing gradients. Stateful layers, accumulation и gradient-evaluation budget являются частью метода. Bilinear одинаковая функция имеет Hessian eigenvalues2/100.01 при rescaling, поэтому flatness не объявлена absolute quality.

Lookahead cycles4→fast1→slow2.5 и следующийslow1.5625 отличаются от posthoc SWA/EMA. Quadratic multiplier.625 может стать1.625 при чрезмерном inner LR; стабильность не обещана автоматически. Resume требует slow/fast/moments/cycle state. Factorial toy interaction.08 объясняет, почему два полезных по отдельности метода могут мешать вместе. Sweep/selection/test и actual budget уточнены без реального model benchmark.

tmp/verify_regularization_structure_selection_depth.py проверяет25учебных механизмов и расчётов. Все10regularization-глав расширены;18глав категорий/кластеризации и финальная содержательная сверка95глав остаются в работе.


## C51. Категории А–Б: смысл кодирования, память и hashing

**Основание:** самостоятельное углубление по девяти пунктам плана категорий; отдельного полного чата по ним нет. Связь embeddings с LLM сохраняется. Сверены первичные документация OneHotEncoder/OrdinalEncoder/FeatureHasher и работа Feature Hashing for Large Scale Multitask Learning.

А: nominal IDs не задают измеримое расстояние. OLS для IDs 1/2/3 и targets 10/13/8 даёт MSE 32/9; full one-hot выражает три значения точно. Cumulative ordinal indicators допускают неравные increments. Full/dummy distances, влияние масштабирования редких indicators, rank design и intercept раскрыты раздельно. Penalty 13 против 29 при разных baselines и минимум 38/3 для full one-hot относятся к фиксированным predictions, а не обещают их сохранение после penalized fitting. Odds ratio 4 не объявлен изменением вероятности в четыре раза. Unknown/missing, literal marker collision, all-zero поведение и контракт columns/serialization объяснены.

Б: dense 4 GB и CSR 1,200,004 bytes относятся к заданным dtype и одному активному indicator на строку; vocabulary, optimizer и временные allocations не включены. Frequency не является target probability; сглаживание распределения задано на конечном словаре с единым UNK. Future counts могут нарушать доступность даже без target. Namespaces и стабильная сериализация предшествуют hashing. Signed collision связывает effective coefficients; cancellation в bag не переносится на одно активное значение поля. Unbiased inner product — ожидание по семейству hashes, а не гарантия одного fixed hash. Для K=3,m=2 ожидаемые pairs=1.5, occupied=1.75, surplus=1.25 различаются. Для K=100,m=1000 вероятность хотя бы одной collision около 99.4%. Независимые k blocks уменьшают full-signature collisions, но при фиксированной общей ширине идеализированная pair variance остаётся 1/D. Bloom membership не подменяет classifier representation.

Независимые проверки учебных расчётов: tmp/verify_categorical_basis_hash_depth.py. Реального обучения и сравнительного benchmark здесь не заявлено. После этих двух глав расширены 79 из 95; остальные 16 и итоговая содержательная сверка остаются в работе.
