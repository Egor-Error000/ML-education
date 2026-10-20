# В. KV-cache: что переиспользуется и как растёт память

[Оглавление](README.md). Пункт A.6; связи с C.4 и serving B.

**Основа:** S4, строки 4771–4922. **Исправления C10/C18:** cache не устраняет attention к истории, decode не всегда ограничен только памятью, GQA меняет число KV heads. **Углубление:** forward двух шагов, точный budget учебной модели и границы equivalence.

**Уровень:** Общее ядро ветки. **Предпосылки:** A1/A3/A5. **Используют:** D2–D10 and applications.

## 1. Почему прошлые K/V можно сохранить

В causal decoder hidden state прошлой позиции зависит только от разрешённого prefix. При добавлении нового токена его старый causal контекст не изменился. Поэтому K/V прошлых позиций каждого слоя можно повторно использовать, если параметры, positions и computation semantics прежние.

На **prefill** модель обрабатывает input prompt и создаёт cache. На **decode** вычисляется representation следующей новой позиции, её Q/K/V, K/V добавляются в cache, а Q сопоставляется с доступными past/current keys. Cache содержит arrays по слоям, а не один «смысл всего диалога».

Для одной головы на новом шаге:

$$
\begin{gathered}q_t\in\mathbb{R}^{1\times d_k},\quad K_{1:t}\in\mathbb{R}^{t\times d_k},\quad V_{1:t}\in\mathbb{R}^{t\times d_v},\\z_t=\frac{q_tK_{1:t}^\top}{\sqrt{d_k}}\in\mathbb{R}^{1\times t},\qquad o_t=\operatorname{softmax}(z_t)V_{1:t}\in\mathbb{R}^{1\times d_v}.\end{gathered}
$$

Q прошлых позиций обычному next-token decode уже не нужны как queries: их outputs были вычислены. K/V нужны, потому что будущий query будет читать их content. Внутренние operations последующих слоёв используют соответствующий cache своего слоя.

Пример: prompt3 positions создаёт K/V длины 3. Следующая входная позиция добавляет четвёртую строку, и query4 взаимодействует с 4 keys. Без cache пришлось бы заново прогнать prefix, создавая старые states. С cache старый forward не повторяется, но чтение разрешённых K/V для нового query остаётся.

### Prefill предсказывает первый output token, а не только «читает текст»

Для prompt длины P prefill вычисляет representations всех prompt positions и logits после последней из них. Из этих logits выбирается первый output token. Он ещё не обязан иметь строку в KV-cache: чтобы получить K/V **этого** token, его ID подают на следующий forward. Новый decode output предсказывает уже следующий token. Поэтому счётчик generated tokens и cache length могут различаться на 1 на конкретной границе API call. Сначала определяют момент измерения, затем считают slots.

На обычном cached decode в каждом layer новый hidden vector проходит Q/K/V projections. Новые K/V вместе с прошлым cache участвуют в attention текущей позиции по causal mask. Output этого layer передаётся следующему, который использует свой собственный history. Нельзя один раз сохранить K/V первого слоя и тем самым заменить весь Transformer stack: последующие representations и projections различаются.

### Полный маленький forward с cache

Упростим до одной scalar head, d_k=d_v=1, без normalization/residual/MLP. Пусть для input x projections равны q=x,k=x,v=2x. Для prefix(1,2) последний query2 даёт scores(2,4), attention≈(0.119203,0.880797), output≈3.761594. В cache сохраняются K=(1,2),V=(2,4). Это история **данного слоя**, а не готовый ответ на произвольный будущий вопрос.

Теперь приходит новая input position x=3. Вычисляем только q3=3,k3=3,v3=6; после append K=(1,2,3),V=(2,4,6). Scores=(3,6,9), probabilities≈(0.002356,0.047314,0.950330), output≈5.895949. Полный recomputation causal prefix(1,2,3) даёт тот же последний результат: прошлые K/V не зависели от позиции 3. Кэшированные values **прочитаны снова**; избежали их вычисления, но не использования.

Для реального многослойного decoder аргумент повторяется индуктивно: если старые hidden states предыдущего слоя не изменились, старые K/V следующего слоя тоже прежние. Causal mask и неизменная positional/computation semantics —условия доказательства. В bidirectional encoder добавление нового token может изменить representations прошлых positions, поэтому этот cache argument нельзя перенести туда автоматически.

### Почему O(N) на шаг не означает O(N) на весь ответ

Для одного нового query attention по t прошлым/current positions требует порядка t·d operations на head и чтения соответствующей истории. Cached attention одного step линейно по accessible length, но N sequential steps суммарно дают сумму 1+…+N≈N²/2 при росте истории с нуля. Если на каждом step без cache пересчитывать full attention всех prefix positions, сумма квадратов lengths имеет порядокN³. Это сравнение attention-компоненты наивных реализаций; projections/MLP имеют свои costs и hardware effects.

Prompt длины P добавляет общий initial prefill и сдвигает lengths следующих steps до P+t. Cache не делает первый длинный prefill бесплатным и не убирает serial sampling dependency между output tokens. Числа «в сотни раз быстрее» из S4 могут возникать в некоторых условиях, но не следуют как постоянный множитель из asymptotic notation. Сравнивать нужно actual prompt/output lengths, batch и kernels.

## 2. Память по компонентам

Для обычного full-attention KV layout без архитектурных special cases:

$$
M_{\mathrm{KV}}\approx2N_{\mathrm{layers}}\,B\,T\,H_{\mathrm{kv}}\,d_{\mathrm{head}}\,b_{\mathrm{element}}.
$$

Двойка — K иV, B — simultaneous sequences, T — cached tokens каждой при равных длинах. При разных длинах вместо B·T нужна сумма lengths. Добавляются allocator/page overhead, scale factors при quantized cache, padding и alignment; distributed placement меняет память отдельного устройства.

Учебная модель:32 layers,8 KV heads,head_dim128,$B=1$,$T=4096$,BF16=2bytes. Тогда:

$$
2\cdot32\cdot1\cdot4096\cdot8\cdot128\cdot2=536\,870\,912\;\text{байт}=512\;\mathrm{MiB}.
$$

Для $B=16$ —8 GiB до overhead. При MHA с 32 KV heads тот же пример потребует 2 GiB на sequence, или 32 GiB на 16 sequences. Эти числа не относятся автоматически к каждой модели 7B: нужны actual config и cache dtype.

Weight quantization не сжимает KV автоматически. Можно иметь INT4 weights и BF16 cache, который становится основным memory consumer при high concurrency/long context. KV quantization вводит отдельные errors/metadata, требуя quality tests.

### От bytes/token к допустимой конкуренции

Для учебного 32-layer,Hkv8,dh128,BF16 cache один token во всех layers требует 131072bytes=128KiB. При смешанных cached lengths(512,1024,4096) сумма 5632token slots даёт 704MiB, а не 3×4096 slots. Allocation с padding до maximum length может физически потребовать больше; packed/paged layout экономит этот wasted capacity при своей metadata.

Если **после остальных расходов** доступно 16GiB только под такой cache, идеальная вместимость 131072token slots. При одинаковых 4096tokens это 32sequences. MHA сHkv32 при прочих равных даёт 8sequences; MQA сHkv1 —256. Это memory upper bounds без fragmentation/workspaces, а не обещание throughput или число пользователей, которым сервис способен выдержать latency SLO. Не каждый зарегистрированный пользователь одновременно держит cache в GPU; admission/scheduling определяет resident workload.

В исходном S4 пример назывался сразу «Llama-2/Llama-3 8B», но использовал 32KVheads. Имя семейства не заменяет actual config. Следует извлечь num layers, num KV heads, head dim, cache dtype и attention type конкретной revision. Для распределённого запуска cache может sharded или replicated по-разному, поэтому деление whole-model bytes на число GPU без проверки layout не всегда верно.

## 3. MHA, MQA и GQA

В обычном MHA query heads имеют отдельные K/V heads. MQA делит одну KV head между query heads, GQA — несколько KV groups. Query capacity и shapes Q могут оставаться шире K/V. Для H_q32,H_kv8 каждая KV group обслуживает 4 query heads по архитектурному mapping.

Memory reduction пропорционален H_kv при прочих равных, но качество не является свободным параметром serving: архитектура/weights должны соответствовать group design. Нельзя просто выкинуть три из четырёх K heads и считать это эквивалентным GQA checkpoint. Возможные conversion/uptraining methods — отдельная процедура. [GQA](https://arxiv.org/abs/2305.13245).

### Shared K/V не делают все query heads одинаковыми

При Hq32,Hkv8 group h использует свои shared K/V и четыре разных query projections. Эти queries вычисляют разные score vectors и могут выбирать разные mixtures одних values. MQA сохраняет много query heads, хотя K/V head одна; это ограничение representational design, а не сведение attention к одному одинаковому output для всех heads. MHA/MQA/GQA нужно сравнивать как обученные architectures.

Для фиксированных layers,length,dh,dtype ratios memory равны 32:8:1 в таком примере. Но speed ratios не обязаны повторять их: query projections и MLP сохранились, а reuse K/V между query heads зависит от kernels. Наивно развернуть shared K/V физическим копированием доHq heads —потерять часть storage advantage. Конкретная реализация может использовать views или fused group-aware attention, поэтому сравнивают allocated memory и фактические reads.

Работа [GQA](https://arxiv.org/abs/2305.13245) рассматривает и conversion/uptraining из MHA checkpoints. Это отдельная подготовка, а не runtime flag, который меняет количество trained KV projections без последствий. Утверждения S4 «MQA обязательно плохая, GQA стоит в 99%новых моделей» не нужны для понимания механизма и не считаются фактом про все models. Другие attention/cache architectures требуют своей формулы памяти.

## 4. Что cache не ускоряет автоматически

Attention нового query по истории всё ещё требует чтения/operations, растущих с accessible length. Long prompt ещё нужно prefill, если prefix не cached. Cache не устраняет MLP/linear layers для новой позиции и сам занимает память/bandwidth.

Prefill часто имеет высокий arithmetic intensity, decode малым batch часто ограничен bandwidth/serial dependencies. При больших batches, длинной истории и другом hardware bottleneck меняется. Поэтому исходные универсальные проценты «GPU работает только 5–10%» не являются общим свойством LLM.

TTFT включает queue, tokenization/prefill и подготовку первого token. Inter-token latency относится к intervals дальнейших tokens. Уменьшение decode time не обязательно улучшает TTFT длинного uncached prompt; ускорение prefill не обязано решить p95 под перегруженной очередью.

### K error и V error действуют через разные места attention

Ошибка V меняет прочитанное содержимое при заданных attention weights. Ошибка K меняет scores, затем softmax weights всех выбранных positions. В приближённой записи с A'=A+ΔA,V'=V+ΔV итоговое изменение равно AΔV+ΔA·V+ΔA·ΔV. Поэтому одинаковый scalar MSE K и V ещё не означает одинаковый output effect. Для K важны также query magnitude и направления errors.

Пример: q=1,K=(0,0),V=(0,10) дают output5. Если округление изменило только второй key на 0.1, output становится 10·sigmoid(0.1)≈5.249792. Если keys прежние, но обе values выросли на 0.1, output=5.1. Это illustration разных paths ошибки; в реальных layers применяется scaling1/√d_k и errors распространяются дальше. Более низкая bitwidth cache не обязана приводить к конкретной «галлюцинации начала текста», как утверждал S4, но требует task/length tests.

Наблюдения KIVI мотивируют per-channel keys и per-token values в его scheme; это не универсальный закон, что каждый punctuation token всегда огромный V outlier. Keys/values quantizer нужно читать целиком: groups, scale overhead, residual higher-precision часть, dequantization и kernels. INT4 weights не определяют cache dtype, а hardware FP8 matmul support не равна автоматически поддержке FP8 KV layout выбранным runtime.

### Memory policies: удалить, перекодировать, разместить иначе

Sliding-window attention ограничивает direct edges к старой истории согласно architecture. Если у каждого слоя окно W, информация может пройти через несколько successive layers, так что «полностью забывает всё за W» слишком буквально. Но это не гарантирует возможность точно восстановить произвольную старую цитату. Принудительная обрезка full-attention cache меняет разрешённые edges и не является тождественным ускорением исходного decoder.

[StreamingLLM](https://arxiv.org/abs/2309.17453) сохраняет attention sinks и recent window для устойчивого streaming. Сохранённые первые tokens не являются сжатой копией удалённой середины и не обеспечивают бесконечную semantic memory. [H₂O](https://arxiv.org/abs/2306.14048) использует heavy-hitter идею выбора history entries по observed attention. Будущий важный token не обязан иметь большой прошлый score, поэтому eviction policy тоже оценивается на целевой distribution.

PagedAttention меняет physical placement/cache allocation при сохранении logical attention history. Это другая цель, чем eviction. Page size, partially filled last blocks, metadata и scheduler делают utilization конечной и workload-dependent;99% из S4 не гарантия. Подробнее allocation и batching разобраны в следующей главе: перенос страниц не означает, что можно удалить любые tokens без изменения функции.

### Cached/uncached equality: что проверяем

Фиксируют exact input IDs, attention mask, positions/RoPE configuration, model/adapter revision, dtype и eval mode. Сравнивают last-position logits полного forward и результата prefix+decode, затем несколько steps на заранее заданном continuation. Так sampling noise не смешивается с cache correctness. Дополнительно проверяют left padding, batch с разными lengths, empty/new cache и границы prefix reuse.

Математическая эквивалентность не обещает bitwise equality разных kernels: порядок floating-point reductions может отличаться. Объявляют tolerance и проверяют масштаб discrepancies. Если два top logits почти равны, малая численная разница может сменить greedy token и весь дальнейший path. Это не доказывает ошибку cache само по себе, но влияет на воспроизводимость generation. При смене system prompt в начале или adapter прежние KV нельзя сохранять только потому, что тема диалога осталась той же.

## 5. Корректность повторного использования

Cache связан с exact tokens prefix, position scheme, model/adapters/revision и computation mode. Одинаковый смысл другой строкой не даёт identical KV. Смена системного prompt в начале, template или active adapter меняет states downstream. Если менять веса во время диалога, прежний cache нельзя автоматически считать корректным.

Sliding-window cache удаляет часть истории только при поддерживаемой attention semantics. Обычный full-attention checkpoint с принудительно отрезанным cache получает другую функцию. Cache eviction — не безвредное сжатие текста; качество зависит от того, какие dependencies потеряны.

В training KV-cache обычно не используется как в inference: gradients, dropout и memory trade-offs другие. Наличие cache API не доказывает пригодность такого path для обычного full-sequence training.

## 6. Вопросы и практика

1. Почему не хранят все прошлые Q для decode? **Ответ:** будущие queries читают past K/V; outputs прежних queries уже получены.
2. При doubling T что происходит с memory? **Ответ:** в обычной формуле растёт вдвое, до особенностей allocation/window.
3. Почему INT4 weights не означают INT4 cache? **Ответ:** разные tensors и runtime options.
4. Что проверить при несовпадении cached/uncached outputs? **Ответ:** positions, masks, dtype, dropout/eval, exact prefix, active adapter и cache boundaries.

**Практика:** пересчитать budget для H_kv1,8,32; затем для mixed lengths $\left(512,\,1024,\,4096\right)$ заменить B·T их суммой. Сравнивать cached и uncached logits в eval mode с объявленным tolerance; speed измерять отдельно от численной корректности.



**Схема курса:** prompt → prefill computes K/V for all prompt positions → KV-cache → decode one token from cached history → append new K/V → stop token or next decode

<!-- latex-display-v1 -->

<!-- source-depth-inference-В -->
