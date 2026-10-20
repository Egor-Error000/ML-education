# Prefill, decode, KV-cache, MQA и GQA

**Уровень:** общее ядро.

**Предпосылки:** [От prompt к обычному циклу генерации](<infer-generation.md>)

**Используют:** [Контекст, sliding window и prefix caching](<infer-context.md>), [Greedy, temperature, top-p, beam и ограничения](<infer-decoding.md>), [Serving, batching и PagedAttention](<infer-serving.md>), [Speculative decoding и rejection sampling](<infer-speculative.md>), [Квантизация: PTQ, QAT, GPTQ и AWQ](<infer-quantization.md>), [Pruning и границы сжатия](<infer-pruning.md>), [Deployment, cold start, очереди и масштабирование](<infer-deployment.md>), [Кэш ответов, monitoring и качество](<infer-monitoring.md>)

**Учебный вопрос:** Какие вычисления и тензоры можно повторно использовать?

Обычный цикл генерации уже задан без оптимизаций. Теперь выделим повторяющиеся вычисления и состояние запроса; сначала проверьте эквивалентность logits, затем измеряйте ускорение.



## Карта вычисления и границы этапов

![Prefill, decode, KV-cache, MQA и GQA: входы, преобразования и выходы](<../../оформление/course/infer-kv.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Параметры модели при обычном выполнении фиксированы. Состояние запроса, cache и состояние внешнего инструмента имеют разные жизненные циклы; их нельзя смешивать с optimizer states.

<!-- unit:fb42f96864fb89de48 -->
<a id="u-fb42f96864fb89de48"></a>

### Почему последние logits не заменяют K/V прошлых позиций

<!-- /unit:fb42f96864fb89de48 -->

<!-- unit:93c41f6478850cd318 -->
<a id="u-93c41f6478850cd318"></a>

Рассмотрим prompt `[кот, спит]`. Для генерации третьего токена нужен state второй позиции и его LM head. После выбора, например, токена «дома» нужно обработать новую третью позицию во всех блоках. Её query в каждом attention-слое обращается к K/V, ранее рассчитанным для «кот» и «спит» именно на этом слое. Последний hidden state второго токена не является набором всех этих ключей и значений.

<!-- /unit:93c41f6478850cd318 -->

<!-- unit:532fb94b22060a3ac1 -->
<a id="u-532fb94b22060a3ac1"></a>

Если отбросить cache, точный вариант — пересчитать весь prefix и снова получить прошлые states. Если оставить cache, можно не пересчитывать неизменившиеся предыдущие позиции. Для causal модели добавление будущего токена не меняет их разрешённое прошлое, что и делает повторное использование корректным при согласованных позициях и параметрах. Если поменять прежние токены или веса, прежний cache в общем случае станет недействительным.

<!-- /unit:532fb94b22060a3ac1 -->

<!-- unit:9b7669f3dcb71cece4 -->
<a id="u-9b7669f3dcb71cece4"></a>

В прямоугольном batch последняя колонка может быть PAD. Для длины 3 в массиве длины 5 continuation относится к последнему настоящему токену, а не автоматически к колонке 5. Реальный генерационный код может выбирать определённую сторону padding и реализовывать согласованный порядок операций. Общий принцип один: позиция, по logits которой выбирается продолжение, должна представлять реальный конец prompt.

<!-- /unit:9b7669f3dcb71cece4 -->

<!-- unit:28fbfce21defce3cb5 -->
<a id="u-28fbfce21defce3cb5"></a>

## Почему прошлые K/V можно сохранить

<!-- /unit:28fbfce21defce3cb5 -->

<!-- unit:fdc33b68b2a38dd8e8 -->
<a id="u-fdc33b68b2a38dd8e8"></a>

В causal decoder hidden state прошлой позиции зависит только от разрешённого prefix. При добавлении нового токена его старый causal контекст не изменился. Поэтому K/V прошлых позиций каждого слоя можно повторно использовать, если параметры, positions и computation semantics прежние.

<!-- /unit:fdc33b68b2a38dd8e8 -->

<!-- unit:02188c6eda8a7eca0d -->
<a id="u-02188c6eda8a7eca0d"></a>

На **prefill** модель обрабатывает input prompt и создаёт cache. На **decode** вычисляется representation следующей новой позиции, её Q/K/V, K/V добавляются в cache, а Q сопоставляется с доступными past/current keys. Cache содержит arrays по слоям, а не один «смысл всего диалога».

<!-- /unit:02188c6eda8a7eca0d -->

<!-- unit:5effe5f19333639d0e -->
<a id="u-5effe5f19333639d0e"></a>

Для одной головы на новом шаге:

<!-- /unit:5effe5f19333639d0e -->

<!-- unit:60c256809d7162b718 -->
<a id="u-60c256809d7162b718"></a>

$$
\begin{gathered}q_t\in\mathbb{R}^{1\times d_k},\quad K_{1:t}\in\mathbb{R}^{t\times d_k},\quad V_{1:t}\in\mathbb{R}^{t\times d_v},\\z_t=\frac{q_tK_{1:t}^\top}{\sqrt{d_k}}\in\mathbb{R}^{1\times t},\qquad o_t=\operatorname{softmax}(z_t)V_{1:t}\in\mathbb{R}^{1\times d_v}.\end{gathered}
$$

<!-- /unit:60c256809d7162b718 -->

<!-- unit:b9abf61b58d4283954 -->
<a id="u-b9abf61b58d4283954"></a>

Q прошлых позиций обычному next-token decode уже не нужны как queries: их outputs были вычислены. K/V нужны, потому что будущий query будет читать их content. Внутренние operations последующих слоёв используют соответствующий cache своего слоя.

<!-- /unit:b9abf61b58d4283954 -->

<!-- unit:c5d5ad0cb9dc639178 -->
<a id="u-c5d5ad0cb9dc639178"></a>

Пример: prompt3 positions создаёт K/V длины 3. Следующая входная позиция добавляет четвёртую строку, и query4 взаимодействует с 4 keys. Без cache пришлось бы заново прогнать prefix, создавая старые states. С cache старый forward не повторяется, но чтение разрешённых K/V для нового query остаётся.

<!-- /unit:c5d5ad0cb9dc639178 -->

<!-- unit:56c3a874725e7edc51 -->
<a id="u-56c3a874725e7edc51"></a>

### Prefill предсказывает первый output token, а не только «читает текст»

<!-- /unit:56c3a874725e7edc51 -->

<!-- unit:0ee9bd058786cb01eb -->
<a id="u-0ee9bd058786cb01eb"></a>

Для prompt длины P prefill вычисляет representations всех prompt positions и logits после последней из них. Из этих logits выбирается первый output token. Он ещё не обязан иметь строку в KV-cache: чтобы получить K/V **этого** token, его ID подают на следующий forward. Новый decode output предсказывает уже следующий token. Поэтому счётчик generated tokens и cache length могут различаться на 1 на конкретной границе API call. Сначала определяют момент измерения, затем считают slots.

<!-- /unit:0ee9bd058786cb01eb -->

<!-- unit:0f3527dc41facbaf67 -->
<a id="u-0f3527dc41facbaf67"></a>

На обычном cached decode в каждом layer новый hidden vector проходит Q/K/V projections. Новые K/V вместе с прошлым cache участвуют в attention текущей позиции по causal mask. Output этого layer передаётся следующему, который использует свой собственный history. Нельзя один раз сохранить K/V первого слоя и тем самым заменить весь Transformer stack: последующие representations и projections различаются.

<!-- /unit:0f3527dc41facbaf67 -->

<!-- unit:73f5ac2134943a8d64 -->
<a id="u-73f5ac2134943a8d64"></a>

### Полный маленький forward с cache

<!-- /unit:73f5ac2134943a8d64 -->

<!-- unit:a5f1797a1b2a9f5f15 -->
<a id="u-a5f1797a1b2a9f5f15"></a>

Упростим до одной scalar head, d_k=d_v=1, без normalization/residual/MLP. Пусть для input x projections равны q=x,k=x,v=2x. Для prefix(1,2) последний query2 даёт scores(2,4), attention≈(0.119203,0.880797), output≈3.761594. В cache сохраняются K=(1,2),V=(2,4). Это история **данного слоя**, а не готовый ответ на произвольный будущий вопрос.

<!-- /unit:a5f1797a1b2a9f5f15 -->

<!-- unit:fbee6a94cbeef6e603 -->
<a id="u-fbee6a94cbeef6e603"></a>

Теперь приходит новая input position x=3. Вычисляем только q3=3,k3=3,v3=6; после append K=(1,2,3),V=(2,4,6). Scores=(3,6,9), probabilities≈(0.002356,0.047314,0.950330), output≈5.895949. Полный recomputation causal prefix(1,2,3) даёт тот же последний результат: прошлые K/V не зависели от позиции 3. Кэшированные values **прочитаны снова**; избежали их вычисления, но не использования.

<!-- /unit:fbee6a94cbeef6e603 -->

<!-- unit:9640862465d3bc0c8c -->
<a id="u-9640862465d3bc0c8c"></a>

Для реального многослойного decoder аргумент повторяется индуктивно: если старые hidden states предыдущего слоя не изменились, старые K/V следующего слоя тоже прежние. Causal mask и неизменная positional/computation semantics —условия доказательства. В bidirectional encoder добавление нового token может изменить representations прошлых positions, поэтому этот cache argument нельзя перенести туда автоматически.

<!-- /unit:9640862465d3bc0c8c -->

<!-- unit:873a2ad1d414c43e00 -->
<a id="u-873a2ad1d414c43e00"></a>

### Почему O(N) на шаг не означает O(N) на весь ответ

<!-- /unit:873a2ad1d414c43e00 -->

<!-- unit:61a9b96934485cfc7b -->
<a id="u-61a9b96934485cfc7b"></a>

Для одного нового query attention по t прошлым/current positions требует порядка t·d operations на head и чтения соответствующей истории. Cached attention одного step линейно по accessible length, но N sequential steps суммарно дают сумму 1+…+N≈N²/2 при росте истории с нуля. Если на каждом step без cache пересчитывать full attention всех prefix positions, сумма квадратов lengths имеет порядокN³. Это сравнение attention-компоненты наивных реализаций; projections/MLP имеют свои costs и hardware effects.

<!-- /unit:61a9b96934485cfc7b -->

<!-- unit:3a26165d84071f42e7 -->
<a id="u-3a26165d84071f42e7"></a>

Prompt длины P добавляет общий initial prefill и сдвигает lengths следующих steps до P+t. Cache не делает первый длинный prefill бесплатным и не убирает serial sampling dependency между output tokens. Числа «в сотни раз быстрее» из S4 могут возникать в некоторых условиях, но не следуют как постоянный множитель из asymptotic notation. Сравнивать нужно actual prompt/output lengths, batch и kernels.

<!-- /unit:3a26165d84071f42e7 -->

<!-- unit:c0364aca8ce4e62961 -->
<a id="u-c0364aca8ce4e62961"></a>

## Память по компонентам

<!-- /unit:c0364aca8ce4e62961 -->

<!-- unit:2e4424ebc1c953914d -->
<a id="u-2e4424ebc1c953914d"></a>

Для обычного full-attention KV layout без архитектурных special cases:

<!-- /unit:2e4424ebc1c953914d -->

<!-- unit:fe8ba1f5575f14e3dc -->
<a id="u-fe8ba1f5575f14e3dc"></a>

$$
M_{\mathrm{KV}}\approx2N_{\mathrm{layers}}\,B\,T\,H_{\mathrm{kv}}\,d_{\mathrm{head}}\,b_{\mathrm{element}}.
$$

<!-- /unit:fe8ba1f5575f14e3dc -->

<!-- unit:8619fa3a13fcf03f4e -->
<a id="u-8619fa3a13fcf03f4e"></a>

Двойка — K иV, B — simultaneous sequences, T — cached tokens каждой при равных длинах. При разных длинах вместо B·T нужна сумма lengths. Добавляются allocator/page overhead, scale factors при quantized cache, padding и alignment; distributed placement меняет память отдельного устройства.

<!-- /unit:8619fa3a13fcf03f4e -->

<!-- unit:af96806e8eabea9cf7 -->
<a id="u-af96806e8eabea9cf7"></a>

Учебная модель:32 layers,8 KV heads,head_dim128,$B=1$,$T=4096$,BF16=2bytes. Тогда:

<!-- /unit:af96806e8eabea9cf7 -->

<!-- unit:71bea08948673c41cf -->
<a id="u-71bea08948673c41cf"></a>

$$
2\cdot32\cdot1\cdot4096\cdot8\cdot128\cdot2=536\,870\,912\;\text{байт}=512\;\mathrm{MiB}.
$$

<!-- /unit:71bea08948673c41cf -->

<!-- unit:2fc92b2249f9371569 -->
<a id="u-2fc92b2249f9371569"></a>

Для $B=16$ —8 GiB до overhead. При MHA с 32 KV heads тот же пример потребует 2 GiB на sequence, или 32 GiB на 16 sequences. Эти числа не относятся автоматически к каждой модели 7B: нужны actual config и cache dtype.

<!-- /unit:2fc92b2249f9371569 -->

<!-- unit:903294b9740beb9b9f -->
<a id="u-903294b9740beb9b9f"></a>

Weight quantization не сжимает KV автоматически. Можно иметь INT4 weights и BF16 cache, который становится основным memory consumer при high concurrency/long context. KV quantization вводит отдельные errors/metadata, требуя quality tests.

<!-- /unit:903294b9740beb9b9f -->

<!-- unit:d12e273892f02e6db4 -->
<a id="u-d12e273892f02e6db4"></a>

### От bytes/token к допустимой конкуренции

<!-- /unit:d12e273892f02e6db4 -->

<!-- unit:7595f712da10ee0f11 -->
<a id="u-7595f712da10ee0f11"></a>

Для учебного 32-layer,Hkv8,dh128,BF16 cache один token во всех layers требует 131072bytes=128KiB. При смешанных cached lengths(512,1024,4096) сумма 5632token slots даёт 704MiB, а не 3×4096 slots. Allocation с padding до maximum length может физически потребовать больше; packed/paged layout экономит этот wasted capacity при своей metadata.

<!-- /unit:7595f712da10ee0f11 -->

<!-- unit:d119e2f9f7453daeca -->
<a id="u-d119e2f9f7453daeca"></a>

Если **после остальных расходов** доступно 16GiB только под такой cache, идеальная вместимость 131072token slots. При одинаковых 4096tokens это 32sequences. MHA сHkv32 при прочих равных даёт 8sequences; MQA сHkv1 —256. Это memory upper bounds без fragmentation/workspaces, а не обещание throughput или число пользователей, которым сервис способен выдержать latency SLO. Не каждый зарегистрированный пользователь одновременно держит cache в GPU; admission/scheduling определяет resident workload.

<!-- /unit:d119e2f9f7453daeca -->

<!-- unit:4da27cd2dcc59c5692 -->
<a id="u-4da27cd2dcc59c5692"></a>

В исходном S4 пример назывался сразу «Llama-2/Llama-3 8B», но использовал 32KVheads. Имя семейства не заменяет actual config. Следует извлечь num layers, num KV heads, head dim, cache dtype и attention type конкретной revision. Для распределённого запуска cache может sharded или replicated по-разному, поэтому деление whole-model bytes на число GPU без проверки layout не всегда верно.

<!-- /unit:4da27cd2dcc59c5692 -->

<!-- unit:8894f807d1e075d824 -->
<a id="u-8894f807d1e075d824"></a>

## MHA, MQA и GQA

<!-- /unit:8894f807d1e075d824 -->

<!-- unit:26843bc13114248c5e -->
<a id="u-26843bc13114248c5e"></a>

В обычном MHA query heads имеют отдельные K/V heads. MQA делит одну KV head между query heads, GQA — несколько KV groups. Query capacity и shapes Q могут оставаться шире K/V. Для H_q32,H_kv8 каждая KV group обслуживает 4 query heads по архитектурному mapping.

<!-- /unit:26843bc13114248c5e -->

<!-- unit:289fb73ac61378d35a -->
<a id="u-289fb73ac61378d35a"></a>

Memory reduction пропорционален H_kv при прочих равных, но качество не является свободным параметром serving: архитектура/weights должны соответствовать group design. Нельзя просто выкинуть три из четырёх K heads и считать это эквивалентным GQA checkpoint. Возможные conversion/uptraining methods — отдельная процедура. [GQA](https://arxiv.org/abs/2305.13245).

<!-- /unit:289fb73ac61378d35a -->

<!-- unit:f90e54c6da3785e8a4 -->
<a id="u-f90e54c6da3785e8a4"></a>

### Shared K/V не делают все query heads одинаковыми

<!-- /unit:f90e54c6da3785e8a4 -->

<!-- unit:31d1bce57cb1ced924 -->
<a id="u-31d1bce57cb1ced924"></a>

При Hq32,Hkv8 group h использует свои shared K/V и четыре разных query projections. Эти queries вычисляют разные score vectors и могут выбирать разные mixtures одних values. MQA сохраняет много query heads, хотя K/V head одна; это ограничение representational design, а не сведение attention к одному одинаковому output для всех heads. MHA/MQA/GQA нужно сравнивать как обученные architectures.

<!-- /unit:31d1bce57cb1ced924 -->

<!-- unit:67e84fc65e98b5afdc -->
<a id="u-67e84fc65e98b5afdc"></a>

Для фиксированных layers,length,dh,dtype ratios memory равны 32:8:1 в таком примере. Но speed ratios не обязаны повторять их: query projections и MLP сохранились, а reuse K/V между query heads зависит от kernels. Наивно развернуть shared K/V физическим копированием доHq heads —потерять часть storage advantage. Конкретная реализация может использовать views или fused group-aware attention, поэтому сравнивают allocated memory и фактические reads.

<!-- /unit:67e84fc65e98b5afdc -->

<!-- unit:1dd09d1070a90ce1ca -->
<a id="u-1dd09d1070a90ce1ca"></a>

Работа [GQA](https://arxiv.org/abs/2305.13245) рассматривает и conversion/uptraining из MHA checkpoints. Это отдельная подготовка, а не runtime flag, который меняет количество trained KV projections без последствий. Утверждения S4 «MQA обязательно плохая, GQA стоит в 99%новых моделей» не нужны для понимания механизма и не считаются фактом про все models. Другие attention/cache architectures требуют своей формулы памяти.

<!-- /unit:1dd09d1070a90ce1ca -->

<!-- unit:e6fc5923d066cdb59a -->
<a id="u-e6fc5923d066cdb59a"></a>

## Что cache не ускоряет автоматически

<!-- /unit:e6fc5923d066cdb59a -->

<!-- unit:84235516a69a5ea13d -->
<a id="u-84235516a69a5ea13d"></a>

Attention нового query по истории всё ещё требует чтения/operations, растущих с accessible length. Long prompt ещё нужно prefill, если prefix не cached. Cache не устраняет MLP/linear layers для новой позиции и сам занимает память/bandwidth.

<!-- /unit:84235516a69a5ea13d -->

<!-- unit:918a6fa444ca2fb11b -->
<a id="u-918a6fa444ca2fb11b"></a>

Prefill часто имеет высокий arithmetic intensity, decode малым batch часто ограничен bandwidth/serial dependencies. При больших batches, длинной истории и другом hardware bottleneck меняется. Поэтому исходные универсальные проценты «GPU работает только 5–10%» не являются общим свойством LLM.

<!-- /unit:918a6fa444ca2fb11b -->

<!-- unit:62838fca2831d0573a -->
<a id="u-62838fca2831d0573a"></a>

TTFT включает queue, tokenization/prefill и подготовку первого token. Inter-token latency относится к intervals дальнейших tokens. Уменьшение decode time не обязательно улучшает TTFT длинного uncached prompt; ускорение prefill не обязано решить p95 под перегруженной очередью.

<!-- /unit:62838fca2831d0573a -->

<!-- unit:3d6bf3a4a4b9a5d998 -->
<a id="u-3d6bf3a4a4b9a5d998"></a>

### K error и V error действуют через разные места attention

<!-- /unit:3d6bf3a4a4b9a5d998 -->

<!-- unit:a685281b4cc46b8a51 -->
<a id="u-a685281b4cc46b8a51"></a>

Ошибка V меняет прочитанное содержимое при заданных attention weights. Ошибка K меняет scores, затем softmax weights всех выбранных positions. В приближённой записи с A'=A+ΔA,V'=V+ΔV итоговое изменение равно AΔV+ΔA·V+ΔA·ΔV. Поэтому одинаковый scalar MSE K и V ещё не означает одинаковый output effect. Для K важны также query magnitude и направления errors.

<!-- /unit:a685281b4cc46b8a51 -->

<!-- unit:1539f977a9054c6b9a -->
<a id="u-1539f977a9054c6b9a"></a>

Пример: q=1,K=(0,0),V=(0,10) дают output5. Если округление изменило только второй key на 0.1, output становится 10·sigmoid(0.1)≈5.249792. Если keys прежние, но обе values выросли на 0.1, output=5.1. Это illustration разных paths ошибки; в реальных layers применяется scaling1/√d_k и errors распространяются дальше. Более низкая bitwidth cache не обязана приводить к конкретной «галлюцинации начала текста», как утверждал S4, но требует task/length tests.

<!-- /unit:1539f977a9054c6b9a -->

<!-- unit:cbbdfeecf5dd8449ff -->
<a id="u-cbbdfeecf5dd8449ff"></a>

Наблюдения KIVI мотивируют per-channel keys и per-token values в его scheme; это не универсальный закон, что каждый punctuation token всегда огромный V outlier. Keys/values quantizer нужно читать целиком: groups, scale overhead, residual higher-precision часть, dequantization и kernels. INT4 weights не определяют cache dtype, а hardware FP8 matmul support не равна автоматически поддержке FP8 KV layout выбранным runtime.

<!-- /unit:cbbdfeecf5dd8449ff -->

<!-- unit:7cad7f2ce825ce072c -->
<a id="u-7cad7f2ce825ce072c"></a>

### Memory policies: удалить, перекодировать, разместить иначе

<!-- /unit:7cad7f2ce825ce072c -->

<!-- unit:e9a06b0afedc6ad136 -->
<a id="u-e9a06b0afedc6ad136"></a>

Sliding-window attention ограничивает direct edges к старой истории согласно architecture. Если у каждого слоя окно W, информация может пройти через несколько successive layers, так что «полностью забывает всё за W» слишком буквально. Но это не гарантирует возможность точно восстановить произвольную старую цитату. Принудительная обрезка full-attention cache меняет разрешённые edges и не является тождественным ускорением исходного decoder.

<!-- /unit:e9a06b0afedc6ad136 -->

<!-- unit:85894e8c3dee4fbb5f -->
<a id="u-85894e8c3dee4fbb5f"></a>

[StreamingLLM](https://arxiv.org/abs/2309.17453) сохраняет attention sinks и recent window для устойчивого streaming. Сохранённые первые tokens не являются сжатой копией удалённой середины и не обеспечивают бесконечную semantic memory. [H₂O](https://arxiv.org/abs/2306.14048) использует heavy-hitter идею выбора history entries по observed attention. Будущий важный token не обязан иметь большой прошлый score, поэтому eviction policy тоже оценивается на целевой distribution.

<!-- /unit:85894e8c3dee4fbb5f -->

<!-- unit:06916f7b63cc2302f8 -->
<a id="u-06916f7b63cc2302f8"></a>

PagedAttention меняет physical placement/cache allocation при сохранении logical attention history. Это другая цель, чем eviction. Page size, partially filled last blocks, metadata и scheduler делают utilization конечной и workload-dependent;99% из S4 не гарантия. Подробнее allocation и batching разобраны в следующей главе: перенос страниц не означает, что можно удалить любые tokens без изменения функции.

<!-- /unit:06916f7b63cc2302f8 -->

<!-- unit:b6cd0c3955db9b36d6 -->
<a id="u-b6cd0c3955db9b36d6"></a>

### Cached/uncached equality: что проверяем

<!-- /unit:b6cd0c3955db9b36d6 -->

<!-- unit:af34d65d3db4608ab4 -->
<a id="u-af34d65d3db4608ab4"></a>

Фиксируют exact input IDs, attention mask, positions/RoPE configuration, model/adapter revision, dtype и eval mode. Сравнивают last-position logits полного forward и результата prefix+decode, затем несколько steps на заранее заданном continuation. Так sampling noise не смешивается с cache correctness. Дополнительно проверяют left padding, batch с разными lengths, empty/new cache и границы prefix reuse.

<!-- /unit:af34d65d3db4608ab4 -->

<!-- unit:e97e25084cf4b4472f -->
<a id="u-e97e25084cf4b4472f"></a>

Математическая эквивалентность не обещает bitwise equality разных kernels: порядок floating-point reductions может отличаться. Объявляют tolerance и проверяют масштаб discrepancies. Если два top logits почти равны, малая численная разница может сменить greedy token и весь дальнейший path. Это не доказывает ошибку cache само по себе, но влияет на воспроизводимость generation. При смене system prompt в начале или adapter прежние KV нельзя сохранять только потому, что тема диалога осталась той же.

<!-- /unit:e97e25084cf4b4472f -->

<!-- unit:d0ed9fb59cfdac671a -->
<a id="u-d0ed9fb59cfdac671a"></a>

## Корректность повторного использования

<!-- /unit:d0ed9fb59cfdac671a -->

<!-- unit:9d0f42fa0363cc4d3d -->
<a id="u-9d0f42fa0363cc4d3d"></a>

Cache связан с exact tokens prefix, position scheme, model/adapters/revision и computation mode. Одинаковый смысл другой строкой не даёт identical KV. Смена системного prompt в начале, template или active adapter меняет states downstream. Если менять веса во время диалога, прежний cache нельзя автоматически считать корректным.

<!-- /unit:9d0f42fa0363cc4d3d -->

<!-- unit:7e1b52b79a90fa2e16 -->
<a id="u-7e1b52b79a90fa2e16"></a>

Sliding-window cache удаляет часть истории только при поддерживаемой attention semantics. Обычный full-attention checkpoint с принудительно отрезанным cache получает другую функцию. Cache eviction — не безвредное сжатие текста; качество зависит от того, какие dependencies потеряны.

<!-- /unit:7e1b52b79a90fa2e16 -->

<!-- unit:48c1aa937198e4a077 -->
<a id="u-48c1aa937198e4a077"></a>

В training KV-cache обычно не используется как в inference: gradients, dropout и memory trade-offs другие. Наличие cache API не доказывает пригодность такого path для обычного full-sequence training.

<!-- /unit:48c1aa937198e4a077 -->

<!-- unit:f0b9c19083f6bc97ca -->
<a id="u-f0b9c19083f6bc97ca"></a>

## Вопросы и практика

<!-- /unit:f0b9c19083f6bc97ca -->

<!-- unit:b7ea4b5de73ac8936a -->
<a id="u-b7ea4b5de73ac8936a"></a>

1. Почему не хранят все прошлые Q для decode? **Ответ:** будущие queries читают past K/V; outputs прежних queries уже получены.
2. При doubling T что происходит с memory? **Ответ:** в обычной формуле растёт вдвое, до особенностей allocation/window.
3. Почему INT4 weights не означают INT4 cache? **Ответ:** разные tensors и runtime options.
4. Что проверить при несовпадении cached/uncached outputs? **Ответ:** positions, masks, dtype, dropout/eval, exact prefix, active adapter и cache boundaries.

<!-- /unit:b7ea4b5de73ac8936a -->

<!-- unit:54e4a9dd5110731a93 -->
<a id="u-54e4a9dd5110731a93"></a>

**Практика:** пересчитать budget для H_kv1,8,32; затем для mixed lengths $\left(512,\,1024,\,4096\right)$ заменить B·T их суммой. Сравнивать cached и uncached logits в eval mode с объявленным tolerance; speed измерять отдельно от численной корректности.

<!-- /unit:54e4a9dd5110731a93 -->

<!-- unit:05d1536064e0bce51d -->
<a id="u-05d1536064e0bce51d"></a>

**Схема курса:** prompt → prefill computes K/V for all prompt positions → KV-cache → decode one token from cached history → append new K/V → stop token or next decode

<!-- /unit:05d1536064e0bce51d -->

<!-- unit:96f0951a52ba87299e -->
<a id="u-96f0951a52ba87299e"></a>

<!-- latex-display-v1 -->

<!-- /unit:96f0951a52ba87299e -->

<!-- unit:a106a8658a585b1a6a -->
<a id="u-a106a8658a585b1a6a"></a>

<!-- source-depth-inference-В -->

<!-- /unit:a106a8658a585b1a6a -->

## Результат главы и переход дальше

Ответьте на вопрос: **Какие вычисления и тензоры можно повторно использовать?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

Следующие темы: [Контекст, sliding window и prefix caching](<infer-context.md>), [Greedy, temperature, top-p, beam и ограничения](<infer-decoding.md>), [Serving, batching и PagedAttention](<infer-serving.md>), [Speculative decoding и rejection sampling](<infer-speculative.md>), [Квантизация: PTQ, QAT, GPTQ и AWQ](<infer-quantization.md>), [Pruning и границы сжатия](<infer-pruning.md>), [Deployment, cold start, очереди и масштабирование](<infer-deployment.md>), [Кэш ответов, monitoring и качество](<infer-monitoring.md>).

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
