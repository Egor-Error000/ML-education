# Greedy, temperature, top-p, beam и ограничения

**Уровень:** общее ядро.

**Предпосылки:** [От prompt к обычному циклу генерации](<infer-generation.md>), [Prefill, decode, KV-cache, MQA и GQA](<infer-kv.md>)

**Используют:** [Speculative decoding и rejection sampling](<infer-speculative.md>)

**Учебный вопрос:** Как decoding policy выбирает токен из logits?





## Карта вычисления и границы этапов

![Greedy, temperature, top-p, beam и ограничения: входы, преобразования и выходы](<../../оформление/course/infer-decoding.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Параметры модели при обычном выполнении фиксированы. Состояние запроса, cache и состояние внешнего инструмента имеют разные жизненные циклы; их нельзя смешивать с optimizer states.

<!-- unit:f8297808f200e647ed -->
<a id="u-f8297808f200e647ed"></a>

## Logits, temperature и greedy

<!-- /unit:f8297808f200e647ed -->

<!-- unit:f1d729bec3218dcb00 -->
<a id="u-f1d729bec3218dcb00"></a>

Пусть logits z. При T>0 $p_i=\frac{e^{z_i/T}}{\sum_je^{z_j/T}}$. Temperature меняет относительную резкость распределения. Greedy выбирает argmax, sampling — случайный token с заданными probabilities. При положительном T порядок z/T тот же, следовательно greedy choice от одного temperature scaling не меняется.

<!-- /unit:f1d729bec3218dcb00 -->

<!-- unit:847a887e44af1737ee -->
<a id="u-847a887e44af1737ee"></a>

Исходный ответ после вашего уточнения всё ещё допускал «обычно не меняется, кроме близких probabilities». Математически это не исключение: близость не меняет монотонность положительного scaling. Floating-point rounding/ties и дополнительные processors — implementation effects, их надо рассматривать отдельно.

<!-- /unit:847a887e44af1737ee -->

<!-- unit:2d49c0603715c8fb99 -->
<a id="u-2d49c0603715c8fb99"></a>

$T=0$ в некоторых API обозначает отдельный greedy режим; это не вычисление z/0. При T→0 probability концентрируется на максимальных logits, при нескольких равных maxima предел распределяет массу между ними; tie-breaking greedy может быть иной. Полная воспроизводимость зависит также от kernels/runtime, а не только выбора greedy.

<!-- /unit:2d49c0603715c8fb99 -->

<!-- unit:133b446b054e8bf876 -->
<a id="u-133b446b054e8bf876"></a>

Пример logits $\left(2,\,1,\,0\right)$:

<!-- /unit:133b446b054e8bf876 -->

<!-- unit:fbd8642b95e7090f16 -->
<a id="u-fbd8642b95e7090f16"></a>

$$
\begin{aligned}T=1:&\quad p\approx(0.66524,0.24473,0.09003),\\T=2:&\quad p\approx(0.50648,0.30720,0.18632).\end{aligned}
$$

<!-- /unit:fbd8642b95e7090f16 -->

<!-- unit:63fb0fe6f87fca09b6 -->
<a id="u-63fb0fe6f87fca09b6"></a>

Greedy в обоих случаях первый. Sampling чаще выбирает alternatives при T2, но high temperature не добавляет знания и не гарантирует creativity/quality. Для factual task лучше measuring correctness, чем универсальный «рекомендованный creative диапазон».

<!-- /unit:63fb0fe6f87fca09b6 -->

<!-- unit:603919c33521b67deb -->
<a id="u-603919c33521b67deb"></a>

### Ответ на ваше уточнение о greedy и высокой temperature

<!-- /unit:603919c33521b67deb -->

<!-- unit:cc6032b11252ba1c69 -->
<a id="u-cc6032b11252ba1c69"></a>

В S5 вы заметили противоречие: если выбирается лучший token, зачем temperature? Для фиксированных logits и T>0 справедливо z_i>z_j тогда и только тогда, когда z_i/T>z_j/T. Softmax также сохраняет этот порядок. Даже разница 0.01 не создаёт математического исключения. При равных maxima деление на положительное T не нарушает равенства; выбор между ними задаётся tie-breaking.

<!-- /unit:cc6032b11252ba1c69 -->

<!-- unit:045feb3b34d7c6d466 -->
<a id="u-045feb3b34d7c6d466"></a>

При sampling важен не только порядок. Отношение вероятностей $p_i/p_j=\exp((z_i-z_j)/T)$ показывает механизм: для разницы logits1 odds при T1 равныe, при T2 —√e. Лучший token остался тем же, но его преимущество уменьшилось. Поэтому temperature имеет прямой смысл для случайного выбора из распределения. Она не изменяет weights и не является новой оценкой истинности ответа или «понимания» модели.

<!-- /unit:045feb3b34d7c6d466 -->

<!-- unit:077bcf9a7ae61555c6 -->
<a id="u-077bcf9a7ae61555c6"></a>

Для численно устойчивого softmax вычитают maximum из scaled logits до exponentiation: exp((z_i−max(z))/T). Это сохраняет probabilities и уменьшает риск overflow. Поддерживаемый API режимT=0 трактуют отдельно. Предел T→∞ равномерный по оставшемуся конечному support, если уже были hard masks; запрещённые tokens с−∞ не становятся разрешёнными только от большой temperature.

<!-- /unit:077bcf9a7ae61555c6 -->

<!-- unit:b3a8e8ced4c4350a47 -->
<a id="u-b3a8e8ced4c4350a47"></a>

### Откуда всё же может измениться greedy в pipeline

<!-- /unit:b3a8e8ced4c4350a47 -->

<!-- unit:f826ee98b84b02ff57 -->
<a id="u-f826ee98b84b02ff57"></a>

Если runtime игнорирует temperature при `do_sample=False`, она не участвует даже в вычислении. Если же дополнительно действуют penalties или другие processors, нужно знать их порядок. Пусть z=(2,1.5), первый token повторялся, его additive penalty0.4. Сначала penalty, потом division даёт(1.6,1.5)/T и первый argmax при любом T>0. Если division сначала, затем **фиксированный** penalty, при T2 получаем(0.6,0.75) и второй argmax.

<!-- /unit:f826ee98b84b02ff57 -->

<!-- unit:93477a2d853950b127 -->
<a id="u-93477a2d853950b127"></a>

Причина здесь —разный относительный масштаб penalty, а не перестановка от одного temperature scaling. Аналогично finite-precision tie/rounding требует конкретного воспроизводимого случая, а не универсальной оговорки «близкие probabilities могут поменяться». Фиксированный seed тоже не устраняет различия processors, versions, kernels и порядка случайных draws в server batch.

<!-- /unit:93477a2d853950b127 -->

<!-- unit:8cbf1c6f09f0267d5b -->
<a id="u-8cbf1c6f09f0267d5b"></a>

Для controlled experiment меняют один параметр при фиксированном model/template, token budget и наборе prompts. Отдельно сохраняют mode: greedy или sampling. Сравнение двух случайных ответов не доказывает эффект temperature на качество; нужна серия наблюдений по correctness, completion, diversity и типам ошибок.

<!-- /unit:8cbf1c6f09f0267d5b -->

<!-- unit:cad80f17e7e99314fe -->
<a id="u-cad80f17e7e99314fe"></a>

## Top-k и nucleus

<!-- /unit:cad80f17e7e99314fe -->

<!-- unit:3edc8042546fb7631f -->
<a id="u-3edc8042546fb7631f"></a>

Top-k оставляет k highest-score tokens, затем нормирует их. Для p=(0.5,0.3,0.15,0.05),k2 получим $\left(0.625,\,0.375,\,0,\,0\right)$. Nucleus/top-p берёт минимальный ranked prefix с cumulative mass не меньше threshold. При top-p0.8 остаются первые два; при 0.9 — первые три, normalized $\left(0.52632,\,0.31579,\,0.15789,\,0\right)$.

<!-- /unit:3edc8042546fb7631f -->

<!-- unit:99a103f2a3c05529b4 -->
<a id="u-99a103f2a3c05529b4"></a>

Равенство cumulative threshold, ties и inclusive convention определяются implementation. Top-p после temperature может выбрать другое количество tokens. Если одновременно включены top-k/top-p/penalties, нужно знать порядок processors: операции в общем случае не коммутируют.

<!-- /unit:99a103f2a3c05529b4 -->

<!-- unit:e8666dab959e53d8d5 -->
<a id="u-e8666dab959e53d8d5"></a>

Repetition/frequency/presence penalties меняют logits по прошлым tokens. Нельзя объявлять одну formula для всех runtimes с одинаковым названием option. Token-level penalty также может наказать нужные повторяющиеся JSON keys или технические symbols. Проверять по task, а не ставить penalty «для качества вообще».

<!-- /unit:e8666dab959e53d8d5 -->

<!-- unit:75d665887c0a625132 -->
<a id="u-75d665887c0a625132"></a>

Min-p/typical sampling и другие variants меняют criterion отсечения. Их полезно сравнивать при конкретной проблеме, но каталог options не заменяет fixed evaluation dataset. Избыток одновременно настроенных filters затрудняет diagnosis.

<!-- /unit:75d665887c0a625132 -->

<!-- unit:9419a09762840cc3f6 -->
<a id="u-9419a09762840cc3f6"></a>

### Шаги отбора nucleus без пропущенной нормировки

<!-- /unit:9419a09762840cc3f6 -->

<!-- unit:c7491cb3d144a74e10 -->
<a id="u-c7491cb3d144a74e10"></a>

Берём ranked probabilities, суммируем слева направо и включаем token, на котором cumulative mass достигает threshold. Затем делим retained probabilities на их сумму. Для исходных(0.5,0.3,0.15,0.05) при threshold0.8 сумма первых двух ровно 0.8, поэтому третий не нужен в оговорённой inclusive convention. При 0.9 первые два ещё недостаточны, первые три дают 0.95. Это не отбор каждого token по правилу «его probability выше 0.9».

<!-- /unit:c7491cb3d144a74e10 -->

<!-- unit:705b8d5e11a28b69a3 -->
<a id="u-705b8d5e11a28b69a3"></a>

При очень остром распределении nucleus может содержать один token, при плоском —много. Это адаптация **количества кандидатов**, а не гарантия осмысленности каждого. Top-k тоже не означает, что все k candidates хороши. [Исходная работа о nucleus sampling](https://arxiv.org/abs/1904.09751) мотивирует sampling strategy, но её результаты не устанавливают вечные идеальные параметры для каждого chatbot или tool model.

<!-- /unit:705b8d5e11a28b69a3 -->

<!-- unit:7f3a0d56fe4b8b836f -->
<a id="u-7f3a0d56fe4b8b836f"></a>

### Порядок processors меняет итоговый закон выбора

<!-- /unit:7f3a0d56fe4b8b836f -->

<!-- unit:91d4b4556c14363e38 -->
<a id="u-91d4b4556c14363e38"></a>

Temperature сохраняет ranking, поэтому сам набор top-k при фиксированных ties остаётся тем же. Но nucleus зависит от probabilities. Пусть исходно p=(0.6,0.25,0.15),threshold0.7. При T1 сохраняются первые два. При T0.5 probabilities пропорциональны квадратам исходных и равны примерно(0.80899,0.14045,0.05056); nucleus теперь состоит только из первого token. Если сначала отобрать первые два при T1, а затем уменьшить T, второй останется в усечённом support. Результат другого порядка отличается.

<!-- /unit:91d4b4556c14363e38 -->

<!-- unit:fe37db011d6ef04400 -->
<a id="u-fe37db011d6ef04400"></a>

Даже два отсечения могут не коммутировать. Для p=(0.4,0.3,0.2,0.1),k2,nucleus0.55: top-k сначала даёт(4/7,3/7), после чего nucleus оставляет только первый. Nucleus сначала сохраняет первые два с общей mass0.7; последующий top-k2 оставляет их обоих. Поэтому набор чисел в generation config без processor order не полностью описывает sampling law.

<!-- /unit:fe37db011d6ef04400 -->

<!-- unit:4ff4414bf3fd679fc7 -->
<a id="u-4ff4414bf3fd679fc7"></a>

### Min-p, typical и penalties: сохраняем различия из разбираемого примера

<!-- /unit:4ff4414bf3fd679fc7 -->

<!-- unit:95e7bf464aa77c5408 -->
<a id="u-95e7bf464aa77c5408"></a>

В min-p с параметромρ оставляют p_i≥ρ·p_max; это относительный порог. Epsilon-like cutoff p_i≥ε —абсолютный порог. При смене резкости p_max меняется, поэтому эти правила не тождественны. Имена options и крайние значения нужно сверять у конкретного runtime, не переносить значения одной implementation в другую без проверки.

<!-- /unit:95e7bf464aa77c5408 -->

<!-- unit:e462e55738339e8303 -->
<a id="u-e462e55738339e8303"></a>

В typical sampling сравнивают self-information token I_i=−lnp_i с entropy H(p)=Σp_iI_i. У отдельного token нет собственной «энтропии распределения»: у него есть information content данного события. Для p=(0.4,0.3,0.2,0.1) H≈1.27985nats. Token с probability0.3 имеет I≈1.20397, ближе к H, чем самый вероятный 0.4 с I≈0.91629. Поэтому typical ranking не просто повторяет sort по вероятности. Конкретный mass cutoff и алгоритм остаются частью метода.

<!-- /unit:e462e55738339e8303 -->

<!-- unit:0f23e0f2114489e278 -->
<a id="u-0f23e0f2114489e278"></a>

Пример additive penalty: $z'_i=z_i-\lambda_f n_i-\lambda_p\mathbf1[n_i>0]$, где n_i —число прошлых появлений. Frequency term растёт с каждым повтором, presence term применяется однажды. При n_i3 и λ_f0.2 frequency уменьшает logit на 0.6; presence с λ_p0.2 —на 0.2. Это объявленная учебная formula, не обязательный contract всех options с похожими названиями.

<!-- /unit:0f23e0f2114489e278 -->

<!-- unit:8147c4ec7fd424484e -->
<a id="u-8147c4ec7fd424484e"></a>

Multiplicative repetition penalty тоже нельзя всегда понимать как «вычесть константу». Sign-aware варианты могут делить положительный logit наr>1, а отрицательный умножать наr, чтобы оба стали менее предпочтительны. Без учёта знака отрицательный score после простого division может, наоборот, увеличиться. Penalizing token IDs влияет и на обязательные повторяющиеся JSON keys, кавычки и code syntax. Наличие repetitions само по себе не ошибка: схема и задача могут их требовать.

<!-- /unit:8147c4ec7fd424484e -->

<!-- unit:65c54d1852790b77de -->
<a id="u-65c54d1852790b77de"></a>

## Beam search и sequence objective

<!-- /unit:65c54d1852790b77de -->

<!-- unit:f3cce5727a3f0bdf1e -->
<a id="u-f3cce5727a3f0bdf1e"></a>

Beam хранит несколько prefixes, расширяет их и оставляет лучшие по score, например sum log-probs с length policy. Это approximation поиска sequence с высоким model score, а не unbiased sampling из исходного p.

<!-- /unit:f3cce5727a3f0bdf1e -->

<!-- unit:b62deffdaafe57eb53 -->
<a id="u-b62deffdaafe57eb53"></a>

Greedy может пропустить лучшую sequence: token A probability0.6, token B0.4. Лучшее продолжение после A имеет conditional0.5 → joint0.3; после B0.99 → joint0.396. Beam ширины 2 способен сохранить B и найти этот путь, greedy выберет A. При этом больший model likelihood не обязательно означает более полезный ответ.

<!-- /unit:b62deffdaafe57eb53 -->

<!-- unit:5e0eb8abafd95fb6f8 -->
<a id="u-5e0eb8abafd95fb6f8"></a>

Length penalty меняет criterion ранжирования и EOS preference. Beam width увеличивает state/compute, но операции могут batch-иться: утверждение «ровно B отдельных serial forward» неточно. Branching KV/cache sharing и early stopping policy входят в runtime.

<!-- /unit:5e0eb8abafd95fb6f8 -->

<!-- unit:bfb6e77c7eb1aa65e7 -->
<a id="u-bfb6e77c7eb1aa65e7"></a>

### Почему beam score не равен полезности ответа

<!-- /unit:bfb6e77c7eb1aa65e7 -->

<!-- unit:62e2d93e0390bc1333 -->
<a id="u-62e2d93e0390bc1333"></a>

Умножение probabilities заменяют суммой logs ради численной устойчивости; это одна и та же likelihood цель для фиксированной длины. При добавлении token его log probability≤0, поэтому raw sequence score не увеличивается. Но короткий ответ не обязательно допустим: EOS имеет собственную probability, а stopping/length policy определяет, когда сравниваются завершённые hypotheses.

<!-- /unit:62e2d93e0390bc1333 -->

<!-- unit:79547cf1430bcfe51d -->
<a id="u-79547cf1430bcfe51d"></a>

Два ответа с raw scores−2 при length2 и−3 при length6 ранжируются как первый лучше. При division наL получаются−1 и−0.5, и лучше уже второй. Это изменение objective, а не исправление вычисления одного и того же score. Формулы length penalty различаются; переносить exponent или normalization между engines без уточнения нельзя. Fixed-width beam также может раньше удалить путь к более вероятной полной sequence.

<!-- /unit:79547cf1430bcfe51d -->

<!-- unit:26adf8c58b64f93e66 -->
<a id="u-26adf8c58b64f93e66"></a>

Высокая likelihood может соответствовать общему короткому шаблону, а не выполнению конкретной инструкции. Greedy/beam/sampling оцениваются на задаче; формат сам по себе не требует одной из этих стратегий. После grammar mask можно и greedy, и sample среди допустимых tokens. Связь с exact speculative beam и полнотой candidate coverage подробно разобрана в предыдущей главе, поэтому один token acceptance theorem здесь не подменяет beam objective.

<!-- /unit:26adf8c58b64f93e66 -->

<!-- unit:baca7e32ebf3a49eba -->
<a id="u-baca7e32ebf3a49eba"></a>

## Constrained decoding до выбора token

<!-- /unit:baca7e32ebf3a49eba -->

<!-- unit:30775efb4151ed2982 -->
<a id="u-30775efb4151ed2982"></a>

Grammar/schema задаёт допустимые продолжения текущего prefix. Для allowed set A(s): запрещённым logits ставится -∞, затем применяется supported sampling pipeline. После выбора token grammar state обновляется. Ограничение действует **во время** generation, а не только после неё.

<!-- /unit:30775efb4151ed2982 -->

<!-- unit:08f7b98d4f60d65bf7 -->
<a id="u-08f7b98d4f60d65bf7"></a>

Для enum `status∈{"ok","unknown"}` после открытия строкового значения допустимы paths этих слов, а не произвольный текст. Но tokenizer tokens могут содержать несколько characters, escaped strings и boundaries; engine должен проверять token continuation against grammar, а не только следующую букву. JSON syntax, regex и context-free grammar имеют разные expressive scopes.

<!-- /unit:08f7b98d4f60d65bf7 -->

<!-- unit:0cd075aee3b39e0fb6 -->
<a id="u-0cd075aee3b39e0fb6"></a>

Если schema поддерживает `amount:number`, syntactically valid1.0 может быть неправильной суммой. Чтобы гарантировать 0≤amount≤budget, нужен supported constraint либо отдельная semantic validation, а истинный budget всё равно берётся из trusted state. JSON type не устанавливает factual correctness.

<!-- /unit:0cd075aee3b39e0fb6 -->

<!-- unit:858023301436280809 -->
<a id="u-858023301436280809"></a>

Grammar может закончиться error, если нет allowed continuation, лимит tokens/timeout остановил output или unsupported schema feature была отвергнута/приближена. Поэтому требуется handling finish reason и final validation. [Пример реализации structured output в vLLM](https://docs.vllm.ai/en/stable/features/structured_outputs/).

<!-- /unit:858023301436280809 -->

<!-- unit:2900cbdf1f9f2d5e0f -->
<a id="u-2900cbdf1f9f2d5e0f"></a>

Parser, получив готовую строку, не ограничивал logits. Retry после parser error — другой механизм: он тратит дополнительный call и может повторить semantic error. Иногда сочетание generation constraints+parser+business checks даёт нужную систему, но каждый слой отвечает за свой invariant.

<!-- /unit:2900cbdf1f9f2d5e0f -->

<!-- unit:f3662b359eb5ffa915 -->
<a id="u-f3662b359eb5ffa915"></a>

### Что именно гарантирует грамматика префиксов

<!-- /unit:f3662b359eb5ffa915 -->

<!-- unit:b3ad314c69568ee234 -->
<a id="u-b3ad314c69568ee234"></a>

Пусть L —множество допустимых **полных** строк. После prefix s разрешён token v, если decoded(s+v) остаётся prefix хотя бы одной строки L. Это можно записать $\operatorname{decode}(s+v)\in\operatorname{Pref}(L)$. EOS разрешается только в состоянии завершённой строки, если таков contract. Невозможно корректно остановиться после `{"age":` только потому, что этот prefix пока не нарушил grammar.

<!-- /unit:b3ad314c69568ee234 -->

<!-- unit:b558c94c4b129323a3 -->
<a id="u-b558c94c4b129323a3"></a>

Поэтому утверждение S5 «любой путь 100%даёт валидный JSON» требует условий: grammar соответствует обещанной схеме, tokenizer handling корректен, output дошёл до accepting state и не был оборван внешним лимитом/timeout. Язык может допускать неограниченные whitespace или длинные strings; prefix-valid generation сама не доказывает termination за max_tokens. Конечный ответ и finish reason всё равно проверяют.

<!-- /unit:b558c94c4b129323a3 -->

<!-- unit:4e16e333309314ffb4 -->
<a id="u-4e16e333309314ffb4"></a>

Регулярные ограничения могут использовать DFA, а произвольная вложенность JSON требует помнить незакрытые structures, например посредством стека/parser state. Классическое regular expression не задаёт unbounded balanced nesting; regex engine с recursion уже выходит за этот класс. Ограничив maximum depth, можно получить конечное пространство состояний, но его размер и стоимость compilation остаются вопросом. CFG также не является «самой выразительной формой любых constraints»: cross-field и внешние business rules имеют свою сложность.

<!-- /unit:4e16e333309314ffb4 -->

<!-- unit:1d6a58cacb82e09acb -->
<a id="u-1d6a58cacb82e09acb"></a>

Token не равен одному character. После prefix `{"x":` один token может содержать сразу `12}` и корректно завершить object. В другом состоянии та же последовательность bytes будет недопустима. В enum("ok","unknown") после открывающей кавычки token может покрывать несколько букв и закрывающую кавычку; matcher должен проверить весь переход. Whitespace, escaping, special tokens и byte-level representation входят в tokenizer-aware проверку. Маска только по первой букве недостаточна.

<!-- /unit:1d6a58cacb82e09acb -->

<!-- unit:3bd9f6a6d4fdecff16 -->
<a id="u-3bd9f6a6d4fdecff16"></a>

### Почему малая исходная вероятность не создаёт пустой nucleus

<!-- /unit:3bd9f6a6d4fdecff16 -->

<!-- unit:a22142962b1acc8747 -->
<a id="u-a22142962b1acc8747"></a>

Для allowed set A маска и нормировка дают:

<!-- /unit:a22142962b1acc8747 -->

<!-- unit:673ab1185dcc4fa7e7 -->
<a id="u-673ab1185dcc4fa7e7"></a>

$$
p_A(i)=\frac{p_i\mathbf1[i\in A]}{\sum_{j\in A}p_j},\qquad \sum_{j\in A}p_j>0.
$$

<!-- /unit:673ab1185dcc4fa7e7 -->

<!-- unit:bdc8490d700c28c9d4 -->
<a id="u-bdc8490d700c28c9d4"></a>

Возьмём p(A,B,C)=(0.97,0.02,0.01), grammar разрешаетB,C. После mask distribution=(0,2/3,1/3). Nucleus0.9 сохраняет оба допустимых tokens; низкая исходная mass0.03 не делает его пустым. Если сначала top-k1 оставил только запрещённый A, а затем grammar удалила его, conflict действительно возникает. Это разный порядок операций, а не неизбежная проблема top-p после корректной нормировки.

<!-- /unit:bdc8490d700c28c9d4 -->

<!-- unit:9edb429322f2152677 -->
<a id="u-9edb429322f2152677"></a>

Если allowed set пуст, либо все retained scores фактически−∞, softmax не определён. Fallback «взять самый вероятный запрещённый token» сломал бы заявленное ограничение. Нужны явная ошибка/поддерживаемый recovery и разбор причины: противоречивая grammar, unsupported feature, неверное состояние или numerical problem. Детали поддерживаемого pipeline описывает [structured-output contract runtime](https://docs.vllm.ai/en/stable/features/structured_outputs/).

<!-- /unit:9edb429322f2152677 -->

<!-- unit:dc52d0fab7214ca944 -->
<a id="u-dc52d0fab7214ca944"></a>

Очень большое конечное отрицательное число вместо−∞ —численное приближение hard mask, а не математический ноль вероятности. Корректность запрета должна обеспечиваться определённым runtime правилом, включая случай полностью пустой маски. Для вычисления self-information в typical sampling нулевые probabilities исключают из support, а не подставляют в обычный ln0.

<!-- /unit:dc52d0fab7214ca944 -->

<!-- unit:cd732cddb559804814 -->
<a id="u-cd732cddb559804814"></a>

### Условие из разбираемого примера выражается в JSON Schema

<!-- /unit:cd732cddb559804814 -->

<!-- unit:a83c439a867d53aae5 -->
<a id="u-a83c439a867d53aae5"></a>

Правило «если status=error, обязательно error_message» можно записать через стандартные [if/then](https://json-schema.org/understanding-json-schema/reference/conditionals). Например:

<!-- /unit:a83c439a867d53aae5 -->

<!-- unit:1c20178b3650889bcc -->
<a id="u-1c20178b3650889bcc"></a>

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "status": {"enum": ["ok", "error"]},
    "error_message": {"type": "string", "minLength": 1}
  },
  "required": ["status"],
  "additionalProperties": false,
  "if": {
    "properties": {"status": {"const": "error"}},
    "required": ["status"]
  },
  "then": {"required": ["error_message"]}
}
```

<!-- /unit:1c20178b3650889bcc -->

<!-- unit:523d68a48ce4db516f -->
<a id="u-523d68a48ce4db516f"></a>

`{"status":"ok"}` проходит; `{"status":"error"}` —нет; `{"status":"error","error_message":"timeout"}` проходит. Empty message не проходит minLength. Неизвестное поле запрещено additionalProperties:false. Это schema validation, но **поддержка этой schema конкретным generation backend** —отдельный факт: subset JSON Schema, handling conditional keywords и отказ на unsupported features нужно проверить. Невыразимость в одном backend нельзя выдавать за невозможность в стандарте.

<!-- /unit:523d68a48ce4db516f -->

<!-- unit:65646ccb0fa3d78c70 -->
<a id="u-65646ccb0fa3d78c70"></a>

Pydantic model может предоставить JSON Schema, однако произвольный Python validator с обращением в database не превращается автоматически в token mask. Outlines, Guidance, GBNF и XGrammar дают разные interfaces/implementations ограничений; application wrapper может также выполнять parsing/retries. Одно название Instructor или field `response_model` не доказывает, что в каждом mode/server внутри работает одинаковый constrained decoder. [XGrammar](https://github.com/mlc-ai/xgrammar), [GBNF](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md).

<!-- /unit:65646ccb0fa3d78c70 -->

<!-- unit:57e518d95af0e98cc2 -->
<a id="u-57e518d95af0e98cc2"></a>

### Углубление: локальная маска и глобальное conditioning

<!-- /unit:57e518d95af0e98cc2 -->

<!-- unit:94c71467ecf42ba698 -->
<a id="u-94c71467ecf42ba698"></a>

Grammar mask гарантирует допустимость локальных продолжений, но не обязательно sample из исходной LLM, **условленной на валидность всей будущей строки**. Пусть допустимы только двухшаговые stringsAa иBb. Исходная модель даёт p(A)=0.9,p(B)=0.1,p(a|A)=0.01,p(b|B)=1. Вероятности полных допустимых strings равны 0.009 и 0.1, а после глобального conditioning наL —примерно 0.08257 и 0.91743.

<!-- /unit:94c71467ecf42ba698 -->

<!-- unit:31005abd16283d7970 -->
<a id="u-31005abd16283d7970"></a>

Локальная grammar сначала разрешает иA, иB, поэтому оставляет 0.9/0.1; затем принудительно выбираетa послеA илиb послеB. Получаем 0.9/0.1 по полным strings —другое распределение, хотя обе строки валидны. Чтобы воспроизвести global conditioning, потребовалась бы вероятность **всего допустимого будущего** после каждого выбора. Это объясняет, почему формальная валидность, высокая model likelihood и точный conditional sampling —разные требования.

<!-- /unit:31005abd16283d7970 -->

<!-- unit:4aa047830837fdbd39 -->
<a id="u-4aa047830837fdbd39"></a>

## Контракт в приложении

<!-- /unit:4aa047830837fdbd39 -->

<!-- unit:3cfcdddf45a02810f9 -->
<a id="u-3cfcdddf45a02810f9"></a>

Последовательность: model output → parse → schema validation → semantic checks → permission checks → execution/result. Схема защищает только заявленный структурный слой. Для tools дополнительно нужны known tool name, существующий object ID, допустимые actions и ограничения пользователя.

<!-- /unit:3cfcdddf45a02810f9 -->

<!-- unit:50c9031d395fdba27f -->
<a id="u-50c9031d395fdba27f"></a>

Low temperature не заменяет schema constraints, а constrained generation не требует по определению greedy: можно sample среди допустимых tokens. Два валидных outputs могут иметь разный смысл. Выбор decoding оценивается по correctness, completion rate и cost, а не только JSON validity.

<!-- /unit:50c9031d395fdba27f -->

<!-- unit:a75456861e56205790 -->
<a id="u-a75456861e56205790"></a>

### Форма tool call и результат действия

<!-- /unit:a75456861e56205790 -->

<!-- unit:eaa2353eb6ed0ea0ec -->
<a id="u-eaa2353eb6ed0ea0ec"></a>

Вернёмся к примеру S5 `search(query,limit)`. Schema может обязать query быть string,limit —integer в допустимом range и запретить неизвестный filter. Но корректно оформленный query всё ещё может искать не ту сущность. Существующий object ID может относиться к другой версии документа, а allowed tool name не доказывает, что action разрешено для текущего пользователя. Эти сведения проверяются по trusted application state после parsing.

<!-- /unit:eaa2353eb6ed0ea0ec -->

<!-- unit:fd08db3c5c9430bb69 -->
<a id="u-fd08db3c5c9430bb69"></a>

Последовательность validation имеет разные причины отказа: JSON не закончен; JSON закончился, но schema нарушена; schema выполнена, но значения неверны; запрос к tool недопустим; tool выполнился, но цель не достигнута. Если всё свести к boolean«JSONvalid», diagnosis и training data смешают разные ошибки. Constrained decoding улучшает определённые уровни, но не делает весь tool loop правильным по определению.

<!-- /unit:fd08db3c5c9430bb69 -->

<!-- unit:86a8f6f2e0ed38d59a -->
<a id="u-86a8f6f2e0ed38d59a"></a>

### Цена маски и почему нужны compilation/cache

<!-- /unit:86a8f6f2e0ed38d59a -->

<!-- unit:1fa971034d4bfedcf3 -->
<a id="u-1fa971034d4bfedcf3"></a>

Проверять весь vocabulary заново на каждом шаге дорого. Trie объединяет общие token/byte prefixes; cached parser transitions или bitmasks переиспользуют совпадающие состояния; compilation подготавливает часть работы заранее. При batching у каждого request остаются собственные parser/stack states, даже если schema общая. Нельзя одним mutable grammar state обслужить одновременно два разных output prefixes.

<!-- /unit:1fa971034d4bfedcf3 -->

<!-- unit:99a65165922441a8e2 -->
<a id="u-99a65165922441a8e2"></a>

Compile latency, steady-state mask cost, memory и cache hit rate измеряются отдельно. S5 приводил 10–30% и 1–5%overhead без условий; эти интервалы не считаем универсальными benchmarks. Сложный regex, vocabulary, nested schema и backend version меняют расходы. Большой выигрыш в доле валидных outputs может окупать overhead, но это результат end-to-end измерения, а не доказательство бесплатности constraints.

<!-- /unit:99a65165922441a8e2 -->

<!-- unit:0033ca13250d51c8a7 -->
<a id="u-0033ca13250d51c8a7"></a>

### Retry не заменяет измерение качества

<!-- /unit:0033ca13250d51c8a7 -->

<!-- unit:974005b70077f98723 -->
<a id="u-974005b70077f98723"></a>

Оценки prompting90–95%validity из разбираемого примера не сопровождаются dataset. Если **предположить** failure ratef=0.1 и независимые одинаковые attempts, максимум 3attempts дают final failuref³=0.001, а expected calls1+f+f²=1.11. На практике повтор может воспроизвести ту же ambiguity/schema issue, и independence не выполнится. Нужно записать actual attempts и конечный completion, а не считать каждый successful retry отдельным удачным prompt.

<!-- /unit:974005b70077f98723 -->

<!-- unit:cf01b9a43336721a3f -->
<a id="u-cf01b9a43336721a3f"></a>

В regression набор входят: malformed/incomplete output, unknown enum/property, wrong type, conditional fields, пустые evidence/source IDs, отсутствующий факт и несовместимые tool parameters. Изменение schema отдельно от sampling позволяет понять, что именно улучшило результат. Для speculation сохраняют свой grammar state на ветку и проверяют target p после constraints; одинаковые seed/temperature у draft и target сами по себе не доказывают recovery правильного distribution, как уже показано в предыдущей главе.

<!-- /unit:cf01b9a43336721a3f -->

<!-- unit:b2d7439ce4a909d9d1 -->
<a id="u-b2d7439ce4a909d9d1"></a>

## Контроль и практика

<!-- /unit:b2d7439ce4a909d9d1 -->

<!-- unit:511d8b516cdb04f43e -->
<a id="u-511d8b516cdb04f43e"></a>

1. Почему greedy unaffected T>0? **Ответ:** положительный scale сохраняет порядок logits.
2. Как top-p0.9 меняет p=(0.5,0.3,0.15,0.05)? **Ответ:** сохраняет первые три и делит на 0.95.
3. Гарантирует ли parser constrained generation? **Ответ:** нет, он работает после получения output.
4. Что не гарантирует валидная schema? **Ответ:** факты, правильный tool/действие, права и успешное завершение.

<!-- /unit:511d8b516cdb04f43e -->

<!-- unit:b9c509bf0da7d3cb1d -->
<a id="u-b9c509bf0da7d3cb1d"></a>

**Практика:** пересчитать temperature/top-p examples. Спроектировать schema «answer:string, source_ids:array» и назвать три валидных, но semantically incorrect outputs. Проверить их отдельными rules. На фиксированных prompts сравнить одну decoding change, сохранив model/template и budgets.

<!-- /unit:b9c509bf0da7d3cb1d -->

<!-- unit:3571932822721bdb7b -->
<a id="u-3571932822721bdb7b"></a>

<!-- latex-display-v1 -->

<!-- /unit:3571932822721bdb7b -->

<!-- unit:da49fcc5b2c0a8f67a -->
<a id="u-da49fcc5b2c0a8f67a"></a>

<!-- source-depth-inference-Е -->

<!-- /unit:da49fcc5b2c0a8f67a -->

## Результат главы и переход дальше

Ответьте на вопрос: **Как decoding policy выбирает токен из logits?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

Следующие темы: [Speculative decoding и rejection sampling](<infer-speculative.md>).

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
