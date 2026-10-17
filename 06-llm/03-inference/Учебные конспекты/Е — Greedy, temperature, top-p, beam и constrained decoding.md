# Е. Как выбирается следующий токен и ограничивается формат

[Оглавление](README.md). Объединены C.2–C.3; семантика приложения раскрывается в его отдельном разделе.

**Основа:** S5, строки 120–413; S4, 6439–7697 для beam context. Сохранён ваш вопрос: «greedy + high temperature — разве это не бессмыслица?» **Исправление C12:** при T>0 простое деление logits сохраняет argmax в точной арифметике, даже при близких значениях. **Углубление:** ручной top-k/top-p, order of processors, finite grammar state и validation layers.

## 1. Logits, temperature и greedy

Пусть logits z. При T>0 $p_i=\frac{e^{z_i/T}}{\sum_je^{z_j/T}}$. Temperature меняет относительную резкость распределения. Greedy выбирает argmax, sampling — случайный token с заданными probabilities. При положительном T порядок z/T тот же, следовательно greedy choice от одного temperature scaling не меняется.

Исходный ответ после вашего уточнения всё ещё допускал «обычно не меняется, кроме близких probabilities». Математически это не исключение: близость не меняет монотонность положительного scaling. Floating-point rounding/ties и дополнительные processors — implementation effects, их надо рассматривать отдельно.

$T=0$ в некоторых API обозначает отдельный greedy режим; это не вычисление z/0. При T→0 probability концентрируется на максимальных logits, при нескольких равных maxima предел распределяет массу между ними; tie-breaking greedy может быть иной. Полная воспроизводимость зависит также от kernels/runtime, а не только выбора greedy.

Пример logits $\left(2,\,1,\,0\right)$:

$$
\begin{aligned}T=1:&\quad p\approx(0.66524,0.24473,0.09003),\\T=2:&\quad p\approx(0.50648,0.30720,0.18632).\end{aligned}
$$

Greedy в обоих случаях первый. Sampling чаще выбирает alternatives при T2, но high temperature не добавляет знания и не гарантирует creativity/quality. Для factual task лучше measuring correctness, чем универсальный «рекомендованный creative диапазон».

## 2. Top-k и nucleus

Top-k оставляет k highest-score tokens, затем нормирует их. Для p=(0.5,0.3,0.15,0.05),k2 получим $\left(0.625,\,0.375,\,0,\,0\right)$. Nucleus/top-p берёт минимальный ranked prefix с cumulative mass не меньше threshold. При top-p0.8 остаются первые два; при0.9 — первые три, normalized $\left(0.52632,\,0.31579,\,0.15789,\,0\right)$.

Равенство cumulative threshold, ties и inclusive convention определяются implementation. Top-p после temperature может выбрать другое количество tokens. Если одновременно включены top-k/top-p/penalties, нужно знать порядок processors: операции в общем случае не коммутируют.

Repetition/frequency/presence penalties меняют logits по прошлым tokens. Нельзя объявлять одну formula для всех runtimes с одинаковым названием option. Token-level penalty также может наказать нужные повторяющиеся JSON keys или технические symbols. Проверять по task, а не ставить penalty «для качества вообще».

Min-p/typical sampling и другие variants меняют criterion отсечения. Их полезно сравнивать при конкретной проблеме, но каталог options не заменяет fixed evaluation dataset. Избыток одновременно настроенных filters затрудняет diagnosis.

## 3. Beam search и sequence objective

Beam хранит несколько prefixes, расширяет их и оставляет лучшие по score, например sum log-probs с length policy. Это approximation поиска sequence с высоким model score, а не unbiased sampling из исходного p.

Greedy может пропустить лучшую sequence: token A probability0.6, token B0.4. Лучшее продолжение после A имеет conditional0.5 → joint0.3; после B0.99 → joint0.396. Beam ширины2 способен сохранить B и найти этот путь, greedy выберет A. При этом больший model likelihood не обязательно означает более полезный ответ.

Length penalty меняет criterion ранжирования и EOS preference. Beam width увеличивает state/compute, но операции могут batch-иться: утверждение «ровно B отдельных serial forward» неточно. Branching KV/cache sharing и early stopping policy входят в runtime.

## 4. Constrained decoding до выбора token

Grammar/schema задаёт допустимые продолжения текущего prefix. Для allowed set A(s): запрещённым logits ставится -∞, затем применяется supported sampling pipeline. После выбора token grammar state обновляется. Ограничение действует **во время** generation, а не только после неё.

Для enum `status∈{"ok","unknown"}` после открытия строкового значения допустимы paths этих слов, а не произвольный текст. Но tokenizer tokens могут содержать несколько characters, escaped strings и boundaries; engine должен проверять token continuation against grammar, а не только следующую букву. JSON syntax, regex и context-free grammar имеют разные expressive scopes.

Если schema поддерживает `amount:number`, syntactically valid1.0 может быть неправильной суммой. Чтобы гарантировать0≤amount≤budget, нужен supported constraint либо отдельная semantic validation, а истинный budget всё равно берётся из trusted state. JSON type не устанавливает factual correctness.

Grammar может закончиться error, если нет allowed continuation, лимит tokens/timeout остановил output или unsupported schema feature была отвергнута/приближена. Поэтому требуется handling finish reason и final validation. [Пример реализации structured output в vLLM](https://docs.vllm.ai/en/stable/features/structured_outputs/).

Parser, получив готовую строку, не ограничивал logits. Retry после parser error — другой механизм: он тратит дополнительный call и может повторить semantic error. Иногда сочетание generation constraints+parser+business checks даёт нужную систему, но каждый слой отвечает за свой invariant.

## 5. Контракт в приложении

Последовательность: model output → parse → schema validation → semantic checks → permission checks → execution/result. Схема защищает только заявленный структурный слой. Для tools дополнительно нужны known tool name, существующий object ID, допустимые actions и ограничения пользователя.

Low temperature не заменяет schema constraints, а constrained generation не требует по определению greedy: можно sample среди допустимых tokens. Два валидных outputs могут иметь разный смысл. Выбор decoding оценивается по correctness, completion rate и cost, а не только JSON validity.

## 6. Контроль и практика

1. Почему greedy unaffected T>0? **Ответ:** положительный scale сохраняет порядок logits.
2. Как top-p0.9 меняет p=(0.5,0.3,0.15,0.05)? **Ответ:** сохраняет первые три и делит на0.95.
3. Гарантирует ли parser constrained generation? **Ответ:** нет, он работает после получения output.
4. Что не гарантирует валидная schema? **Ответ:** факты, правильный tool/действие, права и успешное завершение.

**Практика:** пересчитать temperature/top-p examples. Спроектировать schema «answer:string, source_ids:array» и назвать три валидных, но semantically incorrect outputs. Проверить их отдельными rules. На фиксированных prompts сравнить одну decoding change, сохранив model/template и budgets.

<!-- latex-display-v1 -->
