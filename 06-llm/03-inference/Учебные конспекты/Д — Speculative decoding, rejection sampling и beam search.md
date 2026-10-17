# Д. Speculative decoding: угадать несколько токенов и проверить

[Оглавление](README.md). C.1; углубления и case study из чата.

**Основа:** S4, строки 6439–8230. Сохранены вопросы «станет ли большая модель быстрее от прогресса маленьких?», «что с beam search?» и последующее обсуждение DSpark. **Исправления C12/C13:** сохраняется target distribution, а не обязательно тот же текст для seed; acceptance без residual недостаточен; stronger draft не автоматически быстрее. **Углубление:** числовое доказательство sampling correction и явная speed модель.

## 1. Откуда берётся выигрыш

Draft policy q дешево предлагает block из γ tokens. Target p вычисляет probabilities для предложенных prefixes совместно, пользуясь parallel processing positions. Это выгодно, если verification block дешевле γ независимых target decode steps и достаточно proposals принимается.

Зависимость tokens не исчезает: draft создаёт их последовательно или специальным parallel/semi-autoregressive методом, а target оценивает каждый при соответствующем prefix. После первого rejection неподтверждённый suffix нельзя просто оставить: его conditioning содержит отклонённый token.

В one-target-iteration схеме при принятии всех γ proposals часто можно дополнительно sample ещё один target token из уже вычисленных logits. Cache commit/rollback должен сохранять ровно принятый prefix; иначе outputs могут выглядеть правдоподобно, но algorithm перестанет соответствовать заявленному distribution.

## 2. Acceptance и почему нужен residual

Для одного proposed x~q при q(x)>0 принять его с вероятностью $a(x)=\min\!\left(1,\frac{p(x)}{q(x)}\right)$. Accepted probability mass token x равна $q(x)a(x)=\min\{p(x),q(x)\}$. Остальная target mass равна positive residual:

$$
r(x)=\frac{[p(x)-q(x)]_+}{\sum_v[p(v)-q(v)]_+},\qquad[a]_+=\max(a,0).
$$

При rejection sample из r. Если p=q, acceptance1 и residual не нужен; его denominator0, поэтому вычислять его как обычное распределение в этом случае не надо. Нулевые q для непредлагаемых tokens не требуют деления p/q на таких x.

Учебный пример vocabulary A,B,C:

$$
\begin{gathered}p=(0.5,0.3,0.2),\quad q=(0.6,0.1,0.3),\\a=\left(\frac56,1,\frac23\right),\qquad q\odot a=(0.5,0.1,0.2),\qquad\sum_xq(x)a(x)=0.8.\end{gathered}
$$

Rejection probability0.2; positive residual $\left(0,\,0.2,\,0\right)$ нормируется в $\left(0,\,1,\,0\right)$. Итоговые массы $\left(0.5,\,0.1+0.2,\,0.2\right)=p$. Если при rejection просто sample ещё раз из p, итог получится $\left(0.6,\,0.16,\,0.24\right)$, а не target. Это и есть причина специальной correction.

На каждом последовательном accepted prefix условные p/q пересчитываются для правильного context. При корректном применении argument продолжается на sequence distribution. Конкретные top-k/top-p/constraints transformations должны быть включены в определение целевого p: нельзя доказать одно distribution, а исполнять другое. [Speculative decoding](https://arxiv.org/abs/2211.17192).

## 3. Distribution preservation и identical text — разные требования

Два алгоритма могут sample одно и то же distribution, потребляя random numbers в разном порядке. Поэтому одинаковый seed не обязан дать одинаковую строку. Для greedy-equivalent variants проверяется сохранение argmax choices согласно отдельной процедуре, а не этот же rejection sampling theorem.

Floating point, approximate logits, tokenizer mismatch и filters могут менять probabilities. Чтобы заявлять exact target sampling, надо определить assumptions и numerical scope. Если draft/target имеют разные tokenizers, alignment сложнее обычного matching token ID; готовая схема должна поддерживать это явно.

## 4. Когда более сильный draft полезен

При условно постоянной acceptance α и независимой approximation survival ожидаемое число accepted draft tokens $\sum_{i=1}^{\gamma}\alpha^i$. С дополнительным target token на verification step ожидаемое total около $1+\sum_{i=1}^{\gamma}\alpha^i$. Реальные acceptances зависят от positions/tasks, поэтому это объясняющая модель, а не exact universal predictor.

Для γ4,α0.8: expected accepted $0.8+0.64+0.512+0.4096=2.3616$, total≈3.3616. Пусть target baseline step10ms, draft token1ms, verify block12ms. Стоимость цикла16ms, baseline для3.3616 tokens≈33.616ms, идеализированный speedup≈2.10.

Если сильный draft стоит3ms/token и повышает α до0.9, total≈$1+0.9+0.81+0.729+0.6561=4.0951$; cycle24ms; speedup≈1.71. Acceptance лучше, но итог медленнее предыдущего draft. Больше γ также может ухудшить batching/verification efficiency и memory.

Следовательно, прогресс small models **может** ускорить big target, если улучшает trade-off proposal cost/acceptance на нужных данных. Нужно измерять draft time, verifier time, accepted length, concurrency и per-user/aggregate показатели. Full serving cost не выводится из standalone accuracy draft.

## 5. Почему beam search отдельная задача

Beam search хранит несколько prefixes и отбирает их по sequence score/length policy. Это не sample из p каждого следующего token. Его pruning path зависит от scores множества hypotheses. Стандартное speculative sampling proof не доказывает exact equivalence beam result.

Можно разрабатывать specialized speculative/tree/beam procedures, но нужно определить, сохраняется ли тот же beam objective, top hypotheses или лишь approximation. Отдельно учитывать memory на branching KV, shared prefixes и beam reordering. «Просто проверить один draft block» не означает проверку всех ветвей beam.

Self-speculative methods используют слои/heads той же модели или другие proposals вместо отдельного малого checkpoint. Это экономит некоторые расходы, но имеет свои training/runtime requirements. MTP, tree speculation и beam — разные идеи, хотя все могут создавать несколько candidates.

## 6. DSpark: опубликованный case study

В продолжении чата обсуждалась DSpark. Работа описывает semi-autoregressive drafting с моделированием зависимостей внутри block и confidence-scheduled verification, учитывающей estimated survival и serving throughput. Она иллюстрирует, почему длина проверки должна учитывать нагрузку, а не только maximal draft length. Авторы связывают deployment с DeepSeek-V4; конкретные результаты относятся к указанным в статье conditions/baseline, а не любому target/runtime. [DSpark](https://arxiv.org/abs/2607.05147).

Старые проценты/догадки из чата не превращаются в пользовательский benchmark. Отдельная [историческая research note](../notes/llm-inference-research.md) сохраняет provenance; при воспроизведении нужны workload, concurrency, hardware и точная метрика.

## 7. Самопроверка

1. Почему не достаточно accept/reject без correction? **Ответ:** accepted mass=min(p,q), нужно восполнить именно p-min(p,q).
2. Обязана ли строка совпасть при одном seed? **Ответ:** нет, distribution equivalence слабее одинакового random path.
3. Что делает expensive draft нежелательным? **Ответ:** его cost может перекрыть выигрыш accepted length.

**Практика:** пересчитать p/q example и speed examples. Затем положить p=q и объяснить acceptance/residual. Для эксперимента сравнить target baseline и speculation на одинаковом decoding с quality/distribution checks и измерениями accepted tokens, verifier/draft time, TTFT и decode latency.

<!-- latex-display-v1 -->
