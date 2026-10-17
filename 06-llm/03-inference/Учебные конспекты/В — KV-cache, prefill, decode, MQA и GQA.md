# В. KV-cache: что переиспользуется и как растёт память

[Оглавление](README.md). Пункт A.6; связи с C.4 и serving B.

**Основа:** S4, строки 4771–4922. **Исправления C10:** cache не устраняет attention к истории, decode не всегда ограничен только памятью, GQA меняет число KV heads. **Углубление:** forward двух шагов, точный budget учебной модели и границы equivalence.

## 1. Почему прошлые K/V можно сохранить

В causal decoder hidden state прошлой позиции зависит только от разрешённого prefix. При добавлении нового токена его старый causal контекст не изменился. Поэтому K/V прошлых позиций каждого слоя можно повторно использовать, если параметры, positions и computation semantics прежние.

На **prefill** модель обрабатывает input prompt и создаёт cache. На **decode** вычисляется representation следующей новой позиции, её Q/K/V, K/V добавляются в cache, а Q сопоставляется с доступными past/current keys. Cache содержит arrays по слоям, а не один «смысл всего диалога».

Для одной головы на новом шаге:

$$
\begin{gathered}q_t\in\mathbb{R}^{1\times d_k},\quad K_{1:t}\in\mathbb{R}^{t\times d_k},\quad V_{1:t}\in\mathbb{R}^{t\times d_v},\\z_t=q_tK_{1:t}^\top\in\mathbb{R}^{1\times t},\qquad o_t=\operatorname{softmax}(z_t)V_{1:t}\in\mathbb{R}^{1\times d_v}.\end{gathered}
$$

Q прошлых позиций обычному next-token decode уже не нужны как queries: их outputs были вычислены. K/V нужны, потому что будущий query будет читать их content. Внутренние operations последующих слоёв используют соответствующий cache своего слоя.

Пример: prompt3 positions создаёт K/V длины3. Следующая входная позиция добавляет четвёртую строку, и query4 взаимодействует с4 keys. Без cache пришлось бы заново прогнать prefix, создавая старые states. С cache старый forward не повторяется, но чтение разрешённых K/V для нового query остаётся.

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

Для $B=16$ —8 GiB до overhead. При MHA с32 KV heads тот же пример потребует2 GiB на sequence, или32 GiB на16 sequences. Эти числа не относятся автоматически к каждой модели7B: нужны actual config и cache dtype.

Weight quantization не сжимает KV автоматически. Можно иметь INT4 weights и BF16 cache, который становится основным memory consumer при high concurrency/long context. KV quantization вводит отдельные errors/metadata, требуя quality tests.

## 3. MHA, MQA и GQA

В обычном MHA query heads имеют отдельные K/V heads. MQA делит одну KV head между query heads, GQA — несколько KV groups. Query capacity и shapes Q могут оставаться шире K/V. Для H_q32,H_kv8 каждая KV group обслуживает4 query heads по архитектурному mapping.

Memory reduction пропорционален H_kv при прочих равных, но качество не является свободным параметром serving: архитектура/weights должны соответствовать group design. Нельзя просто выкинуть три из четырёх K heads и считать это эквивалентным GQA checkpoint. Возможные conversion/uptraining methods — отдельная процедура. [GQA](https://arxiv.org/abs/2305.13245).

## 4. Что cache не ускоряет автоматически

Attention нового query по истории всё ещё требует чтения/operations, растущих с accessible length. Long prompt ещё нужно prefill, если prefix не cached. Cache не устраняет MLP/linear layers для новой позиции и сам занимает память/bandwidth.

Prefill часто имеет высокий arithmetic intensity, decode малым batch часто ограничен bandwidth/serial dependencies. При больших batches, длинной истории и другом hardware bottleneck меняется. Поэтому исходные универсальные проценты «GPU работает только5–10%» не являются общим свойством LLM.

TTFT включает queue, tokenization/prefill и подготовку первого token. Inter-token latency относится к intervals дальнейших tokens. Уменьшение decode time не обязательно улучшает TTFT длинного uncached prompt; ускорение prefill не обязано решить p95 под перегруженной очередью.

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

<!-- latex-display-v1 -->
