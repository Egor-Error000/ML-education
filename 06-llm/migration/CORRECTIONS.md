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
