# Speculative decoding и rejection sampling

**Уровень:** углубление.

**Предпосылки:** [От prompt к обычному циклу генерации](<infer-generation.md>), [Prefill, decode, KV-cache, MQA и GQA](<infer-kv.md>), [Greedy, temperature, top-p, beam и ограничения](<infer-decoding.md>)

**Используют:** Выбранная прикладная задача и итоговая проверка результата.

**Учебный вопрос:** Почему проверка draft сохраняет заданное распределение?





## Карта вычисления и границы этапов

![Speculative decoding и rejection sampling: входы, преобразования и выходы](<../../оформление/course/infer-speculative.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Параметры модели при обычном выполнении фиксированы. Состояние запроса, cache и состояние внешнего инструмента имеют разные жизненные циклы; их нельзя смешивать с optimizer states.

<!-- unit:28927faa6477c4400b -->
<a id="u-28927faa6477c4400b"></a>

## Откуда берётся выигрыш

<!-- /unit:28927faa6477c4400b -->

<!-- unit:0b2adcecb42013ff53 -->
<a id="u-0b2adcecb42013ff53"></a>

Draft policy q дешево предлагает block из γ tokens. Target p вычисляет probabilities для предложенных prefixes совместно, пользуясь parallel processing positions. Это выгодно, если verification block дешевле γ независимых target decode steps и достаточно proposals принимается.

<!-- /unit:0b2adcecb42013ff53 -->

<!-- unit:62b713659aa1faa188 -->
<a id="u-62b713659aa1faa188"></a>

Зависимость tokens не исчезает: draft создаёт их последовательно или специальным parallel/semi-autoregressive методом, а target оценивает каждый при соответствующем prefix. После первого rejection неподтверждённый suffix нельзя просто оставить: его conditioning содержит отклонённый token.

<!-- /unit:62b713659aa1faa188 -->

<!-- unit:ace2658e62df24f12e -->
<a id="u-ace2658e62df24f12e"></a>

В one-target-iteration схеме при принятии всех γ proposals часто можно дополнительно sample ещё один target token из уже вычисленных logits. Cache commit/rollback должен сохранять ровно принятый prefix; иначе outputs могут выглядеть правдоподобно, но algorithm перестанет соответствовать заявленному distribution.

<!-- /unit:ace2658e62df24f12e -->

<!-- unit:75204c00ae2acf8fc6 -->
<a id="u-75204c00ae2acf8fc6"></a>

### Почему проверка нескольких позиций не нарушает авторегрессию

<!-- /unit:75204c00ae2acf8fc6 -->

<!-- unit:c9a9cd48542d65c7aa -->
<a id="u-c9a9cd48542d65c7aa"></a>

В разбираемом примере forward большой модели сравнивался с дорогим походом за весами, который хочется оплатить сразу для нескольких токенов. Это полезная интуиция для bandwidth-bound decode, но 5–10%GPU utilization не является свойством всех LLM. При достаточно большом batch или другом attention workload ограничение может перейти к compute. Тогда дополнительные проверяемые позиции конкурируют с другими запросами и уже не «достаются бесплатно».

<!-- /unit:c9a9cd48542d65c7aa -->

<!-- unit:c8f44d977e486f6375 -->
<a id="u-c8f44d977e486f6375"></a>

Target не угадывает все будущие токены независимо. Draft уже предложил x1,x2,x3, поэтому target может одновременно вычислить p(x1|h),p(x2|h,x1),p(x3|h,x1,x2) с causal mask. Это обычная оценка **известной** последовательности при заданных prefixes. Пока первая позиция не принята, следующие остаются условными результатами для черновой истории, а не окончательными ответами. Если x2 отклонён, target score для x3 был получен при неправильном prefix и не может продолжать выбранный результат.

<!-- /unit:c8f44d977e486f6375 -->

<!-- unit:81c018bdbd214959d0 -->
<a id="u-81c018bdbd214959d0"></a>

Вычисленные logits и проверяемые token IDs должны быть сдвинуты так же, как в next-token objective. Logits после h предсказывают x1, после h+x1 —x2. Сравнивать p для token с logits **после самого этого token** —ошибка на одну позицию. Выигрыш получается от организации матричных операций и уменьшения числа последовательных target iterations, а не от разрешения модели видеть своё будущее.

<!-- /unit:81c018bdbd214959d0 -->

<!-- unit:f891eb57fe0c31ae43 -->
<a id="u-f891eb57fe0c31ae43"></a>

### Полный цикл с явным commit и rollback

<!-- /unit:f891eb57fe0c31ae43 -->

<!-- unit:c9f95001cffae82c7e -->
<a id="u-c9f95001cffae82c7e"></a>

В псевдокоде S4 список `generated` сразу пополнялся всеми draft tokens, а после rejection новый token брали прямо из target. Для stochastic sampling это недостаточно: нужен residual, выведенный ниже. Полезно логически отделить committed history h от временного proposal. Тогда цикл читается так:

<!-- /unit:c9f95001cffae82c7e -->

<!-- unit:8c713cd40a1fa6dfd9 -->
<a id="u-8c713cd40a1fa6dfd9"></a>

```text
1. Сохранить границу committed history и состояние обоих cache.
2. Draft предлагает до γ tokens и нужные условные распределения q_i.
3. Target оценивает эти prefixes и соответствующие p_i.
4. Проверять предложения слева направо до первого rejection.
5. Принятый prefix оставить, неподтверждённый suffix удалить.
6. При rejection выбрать один token из residual данной позиции.
7. Если всё принято, выбрать bonus token из доступного target p.
8. Согласовать cache обоих вычислений с итоговой history.
```

<!-- /unit:8c713cd40a1fa6dfd9 -->

<!-- unit:a6bb4f8b89e6b5d6cf -->
<a id="u-a6bb4f8b89e6b5d6cf"></a>

Bonus token и token из residual ещё могут требовать обработки на следующем forward, чтобы их K/V появились в cache. Нельзя считать строку вывода и физический cache одинаковыми на любой промежуточной границе. Draft также должен продолжать **исправленную** history, а не хранить незаметно своё отклонённое продолжение. При EOS или достижении token budget нельзя дописать лишний bonus только ради формулы ускорения.

<!-- /unit:a6bb4f8b89e6b5d6cf -->

<!-- unit:2b708edd2192f92ce0 -->
<a id="u-2b708edd2192f92ce0"></a>

## Acceptance и почему нужен residual

<!-- /unit:2b708edd2192f92ce0 -->

<!-- unit:6ac6c0439c1e47f6db -->
<a id="u-6ac6c0439c1e47f6db"></a>

Для одного proposed x~q при q(x)>0 принять его с вероятностью $a(x)=\min\!\left(1,\frac{p(x)}{q(x)}\right)$. Accepted probability mass token x равна $q(x)a(x)=\min\{p(x),q(x)\}$. Остальная target mass равна positive residual:

<!-- /unit:6ac6c0439c1e47f6db -->

<!-- unit:4baccb84616bbc9feb -->
<a id="u-4baccb84616bbc9feb"></a>

$$
r(x)=\frac{[p(x)-q(x)]_+}{\sum_v[p(v)-q(v)]_+},\qquad[a]_+=\max(a,0).
$$

<!-- /unit:4baccb84616bbc9feb -->

<!-- unit:1ef65d12ae2b43adf8 -->
<a id="u-1ef65d12ae2b43adf8"></a>

При rejection sample из r. Если p=q, acceptance1 и residual не нужен; его denominator0, поэтому вычислять его как обычное распределение в этом случае не надо. Нулевые q для непредлагаемых tokens не требуют деления p/q на таких x.

<!-- /unit:1ef65d12ae2b43adf8 -->

<!-- unit:ece9414acb8c01e734 -->
<a id="u-ece9414acb8c01e734"></a>

Учебный пример vocabulary A,B,C:

<!-- /unit:ece9414acb8c01e734 -->

<!-- unit:e4464206ee296ce348 -->
<a id="u-e4464206ee296ce348"></a>

$$
\begin{gathered}p=(0.5,0.3,0.2),\quad q=(0.6,0.1,0.3),\\a=\left(\frac56,1,\frac23\right),\qquad q\odot a=(0.5,0.1,0.2),\qquad\sum_xq(x)a(x)=0.8.\end{gathered}
$$

<!-- /unit:e4464206ee296ce348 -->

<!-- unit:47c092d754f4ddc707 -->
<a id="u-47c092d754f4ddc707"></a>

Rejection probability0.2; positive residual $\left(0,\,0.2,\,0\right)$ нормируется в $\left(0,\,1,\,0\right)$. Итоговые массы $\left(0.5,\,0.1+0.2,\,0.2\right)=p$. Если при rejection просто sample ещё раз из p, итог получится $\left(0.6,\,0.16,\,0.24\right)$, а не target. Это и есть причина специальной correction.

<!-- /unit:47c092d754f4ddc707 -->

<!-- unit:0e6e1afbf27a3565ee -->
<a id="u-0e6e1afbf27a3565ee"></a>

На каждом последовательном accepted prefix условные p/q пересчитываются для правильного context. При корректном применении argument продолжается на sequence distribution. Конкретные top-k/top-p/constraints transformations должны быть включены в определение целевого p: нельзя доказать одно distribution, а исполнять другое. [Speculative decoding](https://arxiv.org/abs/2211.17192).

<!-- /unit:0e6e1afbf27a3565ee -->

<!-- unit:2d79cd6c7cade71627 -->
<a id="u-2d79cd6c7cade71627"></a>

### От accepted mass к точному распределению

<!-- /unit:2d79cd6c7cade71627 -->

<!-- unit:1436ce6e7e520cdc6c -->
<a id="u-1436ce6e7e520cdc6c"></a>

Обозначим A=Σmin(p,q). Это полная вероятность принять предложенный token. Тогда вероятность rejection1−A равна Σ[p−q]₊, поскольку обе исходные probability vectors суммируются в 1. Для любого v итоговая масса:

<!-- /unit:1436ce6e7e520cdc6c -->

<!-- unit:2fd49d4d9538a9b078 -->
<a id="u-2fd49d4d9538a9b078"></a>

$$
\Pr(Y=v)=\min(p(v),q(v))+(1-A)\frac{[p(v)-q(v)]_+}{1-A}=p(v).
$$

<!-- /unit:2fd49d4d9538a9b078 -->

<!-- unit:c81977b1e82a55580b -->
<a id="u-c81977b1e82a55580b"></a>

Последняя дробь используется только при 1−A>0. Это не «принятый draft плюс ещё один независимый target sample», а разложение нужной массы на уже полученную и недостающую. Важно и содержание rejected sample: если q слишком часто предлагала A, correction должна добавить недостающую массу других tokens, а не ещё раз пропорционально всему p.

<!-- /unit:c81977b1e82a55580b -->

<!-- unit:a54621634c3d3cadc5 -->
<a id="u-a54621634c3d3cadc5"></a>

Для принятия одного выбранного x достаточно scalar p(x),q(x), но чтобы построить residual по vocabulary, нужны значения p(v),q(v) для остальных v либо другой корректный способ сэмплировать ту же correction. Поэтому массив `draft_probs` только с вероятностями выбранных tokens из S4 не задаёт весь stochastic algorithm. Сохранение top-k probabilities с потерянной tail тоже требует собственного обоснования.

<!-- /unit:a54621634c3d3cadc5 -->

<!-- unit:cf4c8b08d215e36381 -->
<a id="u-cf4c8b08d215e36381"></a>

### Исходный пример «Paris»: что из него можно вычислить

<!-- /unit:cf4c8b08d215e36381 -->

<!-- unit:e59fba61b9245b9185 -->
<a id="u-e59fba61b9245b9185"></a>

В S4 draft предложил пять токенов с вероятностями (0.95, 0.80, 0.70, 0.60, 0.50), а вероятности этих же предложений у target были (0.90, 0.30, 0.80, 0.65, 0.55). Для первой позиции вероятность принятия $18/19\approx0.94737$; случайное число 0.70 даёт принятие. Для второй вероятность принятия 0.375; случайное число 0.50 даёт отклонение. Проверка остальных трёх после этого не меняет принятый префикс: они зависели от отклонённого второго токена.

<!-- /unit:e59fba61b9245b9185 -->

<!-- unit:3ae7b1d4c28778b168 -->
<a id="u-3ae7b1d4c28778b168"></a>

Однако по этим десяти скалярным вероятностям нельзя определить, что коррекция обязательно выберет «is», как написал исходный ответ: полного остаточного распределения нет. Установлено только, что принялся один токен draft, а следующий выбирается отдельной корректирующей процедурой. И два выпущенных токена за один проход target ещё не означают двукратного ускорения. Если draft пяти токенов стоил 5 мс, проверка — 10 мс, такой конкретный цикл занял 15 мс вместо 20 мс baseline для двух токенов, то есть дал примерно 1.33× до прочих расходов.

<!-- /unit:3ae7b1d4c28778b168 -->

<!-- unit:b143b857acfbe9649b -->
<a id="u-b143b857acfbe9649b"></a>

### Вероятность принятия и связь с качеством ответов

<!-- /unit:b143b857acfbe9649b -->

<!-- unit:a0236af7267b31afec -->
<a id="u-a0236af7267b31afec"></a>

Используем $\min(a,b)=(a+b-|a-b|)/2$. Суммирование даёт:

<!-- /unit:a0236af7267b31afec -->

<!-- unit:03ddc20289d0637653 -->
<a id="u-03ddc20289d0637653"></a>

$$
A=1-\frac12\sum_v|p(v)-q(v)|=1-\operatorname{TV}(p,q).
$$

<!-- /unit:03ddc20289d0637653 -->

<!-- unit:50d0ee1bce729cdbf7 -->
<a id="u-50d0ee1bce729cdbf7"></a>

Важна близость **всех** вероятностей при том же контексте. Если истинный ответ A, но target распределяет вероятности между (A,B) как (0.1,0.9), draft $q_1=(0.1,0.9)$ имеет вероятность принятия 1, а более правдивый $q_2=(0.9,0.1)$ — только 0.2. Это контрпример отождествлению точности на внешней задаче и совпадения с фиксированным target. Чтобы улучшить сам target, нужна другая задача; точная speculation сохраняет в том числе его ошибки.

<!-- /unit:50d0ee1bce729cdbf7 -->

<!-- unit:6b4de7fdffaf4c35c5 -->
<a id="u-6b4de7fdffaf4c35c5"></a>

Если $p(v)>0$, $q(v)=0$, этот токен не предлагается draft, но может появиться через residual. Если $p(x)=0$ на реально предложенном x, он отклоняется. Та же логика применима к грамматическому запрету, если p уже определено с маской допустимых токенов и нормировкой. Доказательство [speculative sampling](https://arxiv.org/abs/2302.01318) требует именно согласованных распределений, а не независимого набора эвристических правил принятия.

<!-- /unit:6b4de7fdffaf4c35c5 -->

<!-- unit:b44c4f611eb9cd4520 -->
<a id="u-b44c4f611eb9cd4520"></a>

## Distribution preservation и identical text — разные требования

<!-- /unit:b44c4f611eb9cd4520 -->

<!-- unit:438320305f610d7f5c -->
<a id="u-438320305f610d7f5c"></a>

Два алгоритма могут sample одно и то же distribution, потребляя random numbers в разном порядке. Поэтому одинаковый seed не обязан дать одинаковую строку. Для greedy-equivalent variants проверяется сохранение argmax choices согласно отдельной процедуре, а не этот же rejection sampling theorem.

<!-- /unit:438320305f610d7f5c -->

<!-- unit:c748965528ecf5f189 -->
<a id="u-c748965528ecf5f189"></a>

Floating point, approximate logits, tokenizer mismatch и filters могут менять probabilities. Чтобы заявлять exact target sampling, надо определить assumptions и numerical scope. Если draft/target имеют разные tokenizers, alignment сложнее обычного matching token ID; готовая схема должна поддерживать это явно.

<!-- /unit:c748965528ecf5f189 -->

<!-- unit:1130e50190a6b1c81a -->
<a id="u-1130e50190a6b1c81a"></a>

### Sampling, greedy и изменение temperature

<!-- /unit:1130e50190a6b1c81a -->

<!-- unit:ff635341d9b4022a62 -->
<a id="u-ff635341d9b4022a62"></a>

Для greedy достаточно принимать совпадающий с deterministic argmax target префикс и при первом несовпадении выдавать target argmax. Нужно сохранить tie-breaking и stop policy. Это другое заявленное поведение, чем sampling из temperature/top-p distribution. При p как точечной массе та же probability-интуиция вырождается в принятие совпадающего token и отклонение остальных, но программный path может быть другим.

<!-- /unit:ff635341d9b4022a62 -->

<!-- unit:bdae291d30eaa9c195 -->
<a id="u-bdae291d30eaa9c195"></a>

В sampling temperature, top-p/top-k, repetition penalties и grammar state определяют фактические p и q. Draft должен сообщать probability той proposal policy, которая действительно выбрала x. Нельзя сэмплировать из truncated q, а в знаменателе использовать probabilities до truncation. Разные tokenizers дополнительно меняют пространство событий: одинаковый integer ID не обязан обозначать одинаковый кусок текста.

<!-- /unit:bdae291d30eaa9c195 -->

<!-- unit:6497b14ca7389d95c6 -->
<a id="u-6497b14ca7389d95c6"></a>

В S4 таблица утверждала, что высокая temperature обязательно уменьшает acceptance. Общего монотонного закона нет. Возьмём target logits(2,0),draft logits(0,2) и одинаковую T. При T=1 overlap≈0.2384; при T=2 overlap≈0.5379. С ростом T оба distributions приближаются к uniform, и overlap здесь растёт. На другом pair зависимость может быть иной. Для реального workload измеряют outcome выбранных transformed policies.

<!-- /unit:6497b14ca7389d95c6 -->

<!-- unit:d9bc9c50b882cab637 -->
<a id="u-d9bc9c50b882cab637"></a>

### Несколько draft и выбор по confidence меняют proposal law

<!-- /unit:d9bc9c50b882cab637 -->

<!-- unit:0045e92ac578e718ba -->
<a id="u-0045e92ac578e718ba"></a>

Если сначала выбираем модель с известными weights mixture, а затем sample, можно определить соответствующую mixture q. Если же генерируем несколько candidates и оставляем наиболее уверенный **после просмотра tokens**, distribution выбранного предложения меняется. Старый q отдельного draft больше не описывает этот experiment.

<!-- /unit:0045e92ac578e718ba -->

<!-- unit:0a1763c7b3e2e132b4 -->
<a id="u-0a1763c7b3e2e132b4"></a>

Простой контрпример: два независимых draws из q(A)=q(B)=0.5, после чего выбираем A, если он встретился хотя бы раз. Выбранный token имеет probabilities(0.75,0.25). Если target=(0.5,0.5), а acceptance ошибочно считает q=(0.5,0.5), он принимает всё и выдаёт biased(0.75,0.25). Поэтому исходный `max(drafts,key=confidence)` нельзя без новой correction добавлять к доказанному algorithm. Адаптивная длина также не должна незаметно отбирать current candidates по будущей информации без учёта такого отбора.

<!-- /unit:0a1763c7b3e2e132b4 -->

<!-- unit:74dea922516b245a07 -->
<a id="u-74dea922516b245a07"></a>

## Когда более сильный draft полезен

<!-- /unit:74dea922516b245a07 -->

<!-- unit:1744417c2d9758e6b4 -->
<a id="u-1744417c2d9758e6b4"></a>

При условно постоянной acceptance α и независимой approximation survival ожидаемое число accepted draft tokens $\sum_{i=1}^{\gamma}\alpha^i$. С дополнительным target token на verification step ожидаемое total около $1+\sum_{i=1}^{\gamma}\alpha^i$. Реальные acceptances зависят от positions/tasks, поэтому это объясняющая модель, а не exact universal predictor.

<!-- /unit:1744417c2d9758e6b4 -->

<!-- unit:e2966e905c868cd7f5 -->
<a id="u-e2966e905c868cd7f5"></a>

Для γ4,α0.8: expected accepted $0.8+0.64+0.512+0.4096=2.3616$, total≈3.3616. Пусть target baseline step10ms, draft token1ms, verify block12ms. Стоимость цикла 16ms, baseline для 3.3616 tokens≈33.616ms, идеализированный speedup≈2.10.

<!-- /unit:e2966e905c868cd7f5 -->

<!-- unit:597e4377a0e25d64dd -->
<a id="u-597e4377a0e25d64dd"></a>

Если сильный draft стоит 3ms/token и повышает α до 0.9, total≈$1+0.9+0.81+0.729+0.6561=4.0951$; cycle24ms; speedup≈1.71. Acceptance лучше, но итог медленнее предыдущего draft. Больше γ также может ухудшить batching/verification efficiency и memory.

<!-- /unit:597e4377a0e25d64dd -->

<!-- unit:9f0d9e6b0685c44914 -->
<a id="u-9f0d9e6b0685c44914"></a>

Следовательно, прогресс small models **может** ускорить big target, если улучшает trade-off proposal cost/acceptance на нужных данных. Нужно измерять draft time, verifier time, accepted length, concurrency и per-user/aggregate показатели. Full serving cost не выводится из standalone accuracy draft.

<!-- /unit:9f0d9e6b0685c44914 -->

<!-- unit:bd6a6397c202e7fcf7 -->
<a id="u-bd6a6397c202e7fcf7"></a>

### Почему число принятых tokens не равно γ·α

<!-- /unit:bd6a6397c202e7fcf7 -->

<!-- unit:333e94c6fc3bdaf46d -->
<a id="u-333e94c6fc3bdaf46d"></a>

Пусть K —длина непрерывного принятого prefix. Для целочисленной K≥0 математическое ожидание E[K]=ΣPr(K≥i). Обозначим s_i=Pr(первые i proposals приняты). Тогда expected emitted progress без раннего EOS равен 1+Σs_i: дополнительная 1 —коррекция после rejection или bonus после полного принятия. Если conditional probability каждого следующего acceptance при уже принятом prefix равнаα, то s_i=αⁱ. Независимость всех token events отдельно от этого условного определения не нужна.

<!-- /unit:333e94c6fc3bdaf46d -->

<!-- unit:8a885e4434897c2e84 -->
<a id="u-8a885e4434897c2e84"></a>

В S4 дляγ5,α0.8 использовалось 5·0.8=4accepted tokens. Правильная prefix model даёт 0.8+0.64+0.512+0.4096+0.32768=2.68928accepted proposals, total progress3.68928. При 5msdraft+10msverification среднее время одного output token около 4.066ms; speedup относительно 10ms baseline≈2.46. Это условный расчёт с fixed cycle costs, а не измерение pair моделей из разбираемого примера.

<!-- /unit:8a885e4434897c2e84 -->

<!-- unit:c5bfcc75de3c25da78 -->
<a id="u-c5bfcc75de3c25da78"></a>

Для разных позиций задают conditional α_i и s_i=∏α_j. Метрика «accepted/proposed» при заранее сгенерированном suffix, который потом отбросили, не обязана совпадать с таким α. Полезно логировать survival по глубине, среднюю committed length и waste verification отдельно. Максимальная theoretical длинаγ+1 не является средней. Увеличениеγ повышает ceiling, но не заставляет модель чаще принимать каждый уже существующий первый token.

<!-- /unit:c5bfcc75de3c25da78 -->

<!-- unit:4ffff010309e930f16 -->
<a id="u-4ffff010309e930f16"></a>

### Ответ на ваш вопрос о прогрессе 7B→9B

<!-- /unit:4ffff010309e930f16 -->

<!-- unit:03db0a69ccc55c5a78 -->
<a id="u-03db0a69ccc55c5a78"></a>

Более качественное понимание русского, длинного контекста или кода может приблизить q к p на соответствующих запросах. Но имя семейства, более свежая версия, GQA или высокий MMLU сами по себе этого не доказывают. В S4 были выдуманные benchmark tables и неподтверждённые характеристики конкретных Qwen revisions. Сохраняем ваш вопрос и механизм, но не выдаём эти числа за реальное сравнение моделей.

<!-- /unit:03db0a69ccc55c5a78 -->

<!-- unit:55db676cc6b5aeb7dc -->
<a id="u-55db676cc6b5aeb7dc"></a>

При одинаковом 16bit payload9B требует 18GB против 14GB для 7B; при 4bit —4.5GB против 3.5GB до metadata. Сравнение 9BINT4 с 7BFP16 показывает ещё и эффект precision, а не только прогресса model family. У draft есть также собственный cache, activations и scheduling cost. Его длинный context limit не гарантирует эффективного чтения нужной информации или одинакового acceptance на всех языках.

<!-- /unit:55db676cc6b5aeb7dc -->

<!-- unit:d9718538aec424d6de -->
<a id="u-d9718538aec424d6de"></a>

Упрощённо $S\approx t_p(1+\sum_i s_i)/(t_d+t_v+t_o)$, где t_p —baseline target time/token, t_d —полный draft round, t_v —verification, t_o —прочие расходы. Все времена относятся к одинаковым workload/concurrency. Verification не обязана стоить один baseline token: растут processed positions, outputs и cache writes. При высокой нагрузке более длинная speculation может улучшить single-user скорость и одновременно уменьшить aggregate throughput.

<!-- /unit:d9718538aec424d6de -->

<!-- unit:d3ea1dcfd8fe48c706 -->
<a id="u-d3ea1dcfd8fe48c706"></a>

Исходные 150/380/520tokens/s и Qwen acceptance68/78% не считаем измерениями без protocol. Для hypothetical стоимости cUSD/hour и routput tokens/s при полной загрузке cost/1M=c·10⁶/(3600r). С условным c=32 и r=180/540 получаем примерно 49.38/16.46USD. Это арифметика заданных чисел, не актуальный cloud price или доказанная экономия.100Mtokens/day требуют в среднем 1157.4tokens/s: одна hypothetical540tokens/s instance такого объёма за сутки не выдаст. Utilization, несколько instances, TTFT, retries и errors меняют реальный бюджет.

<!-- /unit:d3ea1dcfd8fe48c706 -->

<!-- unit:911d2fee30296aafca -->
<a id="u-911d2fee30296aafca"></a>

### Варианты proposal из разбираемого примера — разные engineering задачи

<!-- /unit:911d2fee30296aafca -->

<!-- unit:ff12b3ffc67ad4a3ee -->
<a id="u-ff12b3ffc67ad4a3ee"></a>

Separate small model добавляет weights и state; одинаковая tokenizer semantics удобна, но одной семьи недостаточно для высокой acceptance. N-gram prompt lookup предлагает повтор уже встреченного продолжения без второй neural model. Он может помогать любому тексту с полезными повторами, а не только коду. Отсутствие match приводит к короткому proposal или обычному target step; это не ошибка самого target.

<!-- /unit:ff12b3ffc67ad4a3ee -->

<!-- unit:cdad3331cf4e44ce4a -->
<a id="u-cdad3331cf4e44ce4a"></a>

Early exit использует intermediate representation. Но применить final LM head к произвольному восьмому слою 32-layer checkpoint недостаточно, чтобы гарантировать полезный draft. [LayerSkip](https://arxiv.org/abs/2404.16710) связывает early-exit quality с training recipe и reuse вычислений. Для quantized копии target сохранение architecture также не делает её cache автоматически тождественным float cache: representations различаются. Выгрыш времени зависит от kernels, а не только от bitwidth.

<!-- /unit:cdad3331cf4e44ce4a -->

<!-- unit:4dd4961a2462e9e318 -->
<a id="u-4dd4961a2462e9e318"></a>

[Medusa](https://arxiv.org/abs/2401.10774) использует дополнительные prediction heads и candidate verification. [EAGLE](https://arxiv.org/abs/2401.15077) рассматривает draft на уровне features с учётом token uncertainty. Их нельзя полностью определить одним class из нескольких независимых Linear layers. Требуются обучение, conditioning, attention masks и правило принятия конкретного метода. Сохранение качества в benchmark и точное сохранение target sampling law тоже нужно различать, если используется approximate/typical acceptance.

<!-- /unit:4dd4961a2462e9e318 -->

<!-- unit:6fbd7b2a9f4bc92ca8 -->
<a id="u-6fbd7b2a9f4bc92ca8"></a>

## Почему beam search отдельная задача

<!-- /unit:6fbd7b2a9f4bc92ca8 -->

<!-- unit:b3f5c51ac47d8def0c -->
<a id="u-b3f5c51ac47d8def0c"></a>

Beam search хранит несколько prefixes и отбирает их по sequence score/length policy. Это не sample из p каждого следующего token. Его pruning path зависит от scores множества hypotheses. Стандартное speculative sampling proof не доказывает exact equivalence beam result.

<!-- /unit:b3f5c51ac47d8def0c -->

<!-- unit:5220e619a5877ad7a0 -->
<a id="u-5220e619a5877ad7a0"></a>

Можно разрабатывать specialized speculative/tree/beam procedures, но нужно определить, сохраняется ли тот же beam objective, top hypotheses или лишь approximation. Отдельно учитывать memory на branching KV, shared prefixes и beam reordering. «Просто проверить один draft block» не означает проверку всех ветвей beam.

<!-- /unit:5220e619a5877ad7a0 -->

<!-- unit:70c12ecb038182ea36 -->
<a id="u-70c12ecb038182ea36"></a>

Self-speculative methods используют слои/heads той же модели или другие proposals вместо отдельного малого checkpoint. Это экономит некоторые расходы, но имеет свои training/runtime requirements. MTP, tree speculation и beam — разные идеи, хотя все могут создавать несколько candidates.

<!-- /unit:70c12ecb038182ea36 -->

<!-- unit:0906cbe58968be151d -->
<a id="u-0906cbe58968be151d"></a>

### Разворачиваем ваш вопрос о нарушении beam search

<!-- /unit:0906cbe58968be151d -->

<!-- unit:931c880daafbd038b7 -->
<a id="u-931c880daafbd038b7"></a>

Beam width B задаёт число сохраняемых hypotheses после каждого pruning step. Сумма log probabilities target, length normalization, EOS handling, early stopping и tie-breaking определяют конкретный baseline beam procedure. Даже обычный finite-width beam —эвристический поиск: он может потерять globally optimal последовательность раньше. Требование exact equivalence означает воспроизвести **этот** baseline, а не доказать global optimum по всем текстам.

<!-- /unit:931c880daafbd038b7 -->

<!-- unit:37cf481956fb54d7e4 -->
<a id="u-37cf481956fb54d7e4"></a>

В разбираемом примере предлагалось «draft дал много ветвей, target оценил все, выбралиtop-B». Это необходимая часть, но неполнота candidate set всё ещё может изменить результат. Допустим, текущие prefixes A иB имеют masses0.55 и 0.45. Для A target даёт X0.49,Y0.51; для B —Z0.6,W0.4. Полные joint scores: AX0.2695,AY0.2805,BZ0.27,BW0.18. При width2 baseline сохраняет AY иBZ.

<!-- /unit:37cf481956fb54d7e4 -->

<!-- unit:87d6dee898de7770f1 -->
<a id="u-87d6dee898de7770f1"></a>

Если draft предложил только AX иBZ, target честно пересчитает их probabilities, но выберет AX иBZ. Непредложенная AY была лучше AX. Значит, верный scoring **увиденных** candidates не доказывает equivalence полного beam update. Нужны достаточное candidate coverage, границы на исключённые alternatives либо fallback, который воспроизводит baseline frontier. Блочный выбор лучших terminal branches также не всегда совпадает с промежуточным pruning на каждом depth.

<!-- /unit:87d6dee898de7770f1 -->

<!-- unit:a42b2fbdb603738c83 -->
<a id="u-a42b2fbdb603738c83"></a>

Исходный счёт B4×k4×depth5 равен 80token positions до sharing, а не 100. У полного дерева branching factor k число nodes может расти экспоненциально по depth, но fixed-width beam с pruning на каждом step не обязан раскрывать всё дерево k^depth. Различать эти procedures нужно и для memory budget: общие ancestors могут sharing KV, private suffix и reordering остаются отдельными расходами.

<!-- /unit:a42b2fbdb603738c83 -->

<!-- unit:6248754fb9815d8952 -->
<a id="u-6248754fb9815d8952"></a>

В tree verification query узла видит committed prefix и ancestors **своей** ветки. Соседняя ветка в packed tensor не должна попадать в его causal context. Правильный mask и cache isolation сохраняют target conditional probabilities. Но они сами ещё не выбирают между sampling, greedy и beam: это следующий уровень algorithm. Для sampling дерево служит предложением candidates по заявленной процедуре, а не автоматически сохраняемым списком лучших полных ответов.

<!-- /unit:6248754fb9815d8952 -->

<!-- unit:adb3d0b67149c8703c -->
<a id="u-adb3d0b67149c8703c"></a>

## DSpark: опубликованный case study

<!-- /unit:adb3d0b67149c8703c -->

<!-- unit:696698a2113a1b84df -->
<a id="u-696698a2113a1b84df"></a>

В продолжении чата обсуждалась [DSpark](https://arxiv.org/abs/2607.05147). Parallel backbone подготавливает блок, а лёгкий sequential module добавляет зависимости между proposals. В Markov-варианте previous token через low-rank transition задаёт bias к logits следующей позиции; это не независимый sample каждого слова. Confidence head оценивает conditional acceptance, а scheduler учитывает prefix survival и профиль производительности engine при выборе verification length.

<!-- /unit:696698a2113a1b84df -->

<!-- unit:10e64fb50123c849f6 -->
<a id="u-10e64fb50123c849f6"></a>

Paper сообщает deployment в DeepSeek-V4 и прирост per-user generation speed60–85% относительноMTP-1 при matched throughput. Это сравнение конкретной serving system, не обещание любому engine. Прирост скорости 60% означает множитель 1.6, а при одинаковом числе tokens reciprocal decode time0.625 исходного, то есть уменьшение 37.5%; это не «latency уменьшилась на 60%». Offline accepted length, aggregate throughput и live per-user speed имеют разные denominators.

<!-- /unit:10e64fb50123c849f6 -->

<!-- unit:158efdf80e6ab72ad6 -->
<a id="u-158efdf80e6ab72ad6"></a>

Исходный чат правильно связывал три задачи: удешевить draft, сохранить зависимости внутри него и не расходовать verification capacity на маловероятный suffix. Но фраза «один parallel pass стоит ровно как один scalar draft token» не является гарантией. Подробности conditioning, training и scheduler нужны конкретной реализации. Пример независимых пар и расчёт адаптивной длины ниже поясняют мотив этих механизмов собственными учебными числами.

<!-- /unit:158efdf80e6ab72ad6 -->

<!-- unit:4b226c4dcf133ff606 -->
<a id="u-4b226c4dcf133ff606"></a>

Старые проценты/догадки из разбираемого примера не превращаются в пользовательский benchmark. Отдельная [историческая research note](../../archive/2026-10-20-before-rebuild/06-llm/03-inference/notes/llm-inference-research.md) сохраняет provenance; при воспроизведении нужны workload, concurrency, hardware и точная метрика.

<!-- /unit:4b226c4dcf133ff606 -->

<!-- unit:8daf8c181cc4bd9c8d -->
<a id="u-8daf8c181cc4bd9c8d"></a>

### MTP, DSpark и tree adaptation: сохраняем связь с вашими вопросами

<!-- /unit:8daf8c181cc4bd9c8d -->

<!-- unit:db3e9593e708d3335a -->
<a id="u-db3e9593e708d3335a"></a>

В [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437) multi-token prediction задаёт дополнительные предсказания с conditioning и training objective. Shared embeddings/head не означают нулевой compute. Один speculative proposal за round при идеальном принятии позволяет до двух output tokens за target verification, если получен bonus; это не правило «всегда два полных target passes ради трёх tokens». Startup, draft work и конкретный pipeline определяют реальное отношение. ЧислоMTP1 характеризует proposal глубину используемого baseline, а не обязательное число выдаваемых tokens всего цикла.

<!-- /unit:db3e9593e708d3335a -->

<!-- unit:adeb677f0640982d99 -->
<a id="u-adeb677f0640982d99"></a>

Вопрос «почему независимые parallel heads могут плохо склеивать текст» удобно рассмотреть без названия модели. Пусть допустимы пары «of course» и «no problem» с probabilities0.5/0.5. Их independent marginals дают каждой первой/второй части по 0.5. Независимая сборка выдаёт также «of problem» и «no course» с общей массой 0.5, хотя исходное joint распределение их не содержало. Это потеря зависимости, а не недостаток grammar parser. Лёгкое conditioning на предыдущую выбранную часть может убрать такой конфликт; более дальние зависимости потребуют более богатого state.

<!-- /unit:adeb677f0640982d99 -->

<!-- unit:1fa2bdb84121204e42 -->
<a id="u-1fa2bdb84121204e42"></a>

В собственном toy scheduler conditional survival(0.9,0.5,0.2) даёт s=(0.9,0.45,0.09). Expected progress для проверяемой длины 1/2/3 равен 1.9/2.35/2.44. Если round costs соответственно 10/12/18ms, rates≈0.190/0.196/0.136tokens/ms: выгодна длина 2. Если под другой нагрузкой costs стали 10/16/18ms, выигрывает длина 1. Это объясняет связь confidence и serving capacity без предположения, что максимальный draft всегда лучше. Оценки должны быть calibrated под используемое событие survival.

<!-- /unit:1fa2bdb84121204e42 -->

<!-- unit:f666d795977d824ba3 -->
<a id="u-f666d795977d824ba3"></a>

Планировщик также не должен подглядывать в realized future candidate, чтобы задним числом решить, допускать ли сам этот candidate к стандартной acceptance correction. Контрпример с выбором A из двух drafts выше показывает общий механизм selection bias. Confidence и load policy нужно включить в описание procedure; одной ссылки на rejection sampling для всех adaptive schemes недостаточно.

<!-- /unit:f666d795977d824ba3 -->

<!-- unit:1d3c0af36bd5d327dc -->
<a id="u-1d3c0af36bd5d327dc"></a>

Вторая работа, [tree adaptation к DeepSeek-V4](https://arxiv.org/abs/2609.24698), обсуждает специфические ограничения compressed attention state. Общий принцип можно выразить как S_{t+1}=F(S_t,x_{t+1}). Две speculative ветки A/B создают разные временные состояния из одного committed S_t. Если их updates смешались в persistent compressed state, простое удаление нескольких обычных K/V строк может не восстановить исходную историю. Поэтому нужны branch-local scratch states и перенос только корректно принятого состояния. Это развивает уже разобранный cache commit/rollback, а не отменяет его.

<!-- /unit:1d3c0af36bd5d327dc -->

<!-- unit:fc9bbac39d230a0360 -->
<a id="u-fc9bbac39d230a0360"></a>

Числа про CSA/HCA, cacheGiB и ratios в S4 смешивались из разных описаний; они не нужны как universal constants этой процедуры. Эксперимент tree adaptation и deployment DSpark —разные свидетельства. Наличие research implementation не доказывает использование каждого tree variant на всех production requests. Из публикаций также нельзя вывести глобальный запрет beam search во всех сервисах компании.

<!-- /unit:fc9bbac39d230a0360 -->

<!-- unit:ed175d7ef3e8ba3ac7 -->
<a id="u-ed175d7ef3e8ba3ac7"></a>

### Практический протокол завершает объяснение

<!-- /unit:ed175d7ef3e8ba3ac7 -->

<!-- unit:bd45e215f6aa70662f -->
<a id="u-bd45e215f6aa70662f"></a>

Сначала фиксируют target/tokenizer revision и точное p после decoding transformations. Затем определяют proposal q, method, max depth, cache policy и поддерживаемые runtime combinations. CLI snippets S4 с именами draft checkpoints и speculative flags не являются воспроизведённым запуском; наличие model name не гарантирует существование compatible auxiliary weights. Version-specific настройку проверяют по документации установленного engine.

<!-- /unit:bd45e215f6aa70662f -->

<!-- unit:ac804d10445001980b -->
<a id="u-ac804d10445001980b"></a>

На маленьком vocabulary полезно вычислить итоговую массу по всем accept/reject исходам и проверить recovery p. Для настоящей модели сравнивают baseline и speculative generation в одном режиме: greedy tokens при условиях equivalence либо statistical/distribution diagnostics для sampling, не обязательное совпадение строк по seed. Отдельно измеряют draft/verify/overhead time, committed progress, EOS/correction paths, memory и latency/throughput при нескольких concurrency/length slices. Ошибки residual, prefix alignment и rollback проверяют до объявления speedup.Числовые проверки всей книги не заменяют такой model experiment: практический запуск пользователя здесь не заявляется.

<!-- /unit:ac804d10445001980b -->

<!-- unit:62666b108e048813d4 -->
<a id="u-62666b108e048813d4"></a>

## Самопроверка

<!-- /unit:62666b108e048813d4 -->

<!-- unit:7dea720de61e6bffc5 -->
<a id="u-7dea720de61e6bffc5"></a>

1. Почему не достаточно accept/reject без correction? **Ответ:** accepted mass=min(p,q), нужно восполнить именно p-min(p,q).
2. Обязана ли строка совпасть при одном seed? **Ответ:** нет, distribution equivalence слабее одинакового random path.
3. Что делает expensive draft нежелательным? **Ответ:** его cost может перекрыть выигрыш accepted length.

<!-- /unit:7dea720de61e6bffc5 -->

<!-- unit:b3c36ce6419b76775f -->
<a id="u-b3c36ce6419b76775f"></a>

**Практика:** пересчитать p/q example и speed examples. Затем положить p=q и объяснить acceptance/residual. Для эксперимента сравнить target baseline и speculation на одинаковом decoding с quality/distribution checks и измерениями accepted tokens, verifier/draft time, TTFT и decode latency.

<!-- /unit:b3c36ce6419b76775f -->

<!-- unit:4ff19bd44091fffc9d -->
<a id="u-4ff19bd44091fffc9d"></a>

<!-- latex-display-v1 -->

<!-- /unit:4ff19bd44091fffc9d -->

<!-- unit:f6d886bd73c094ec80 -->
<a id="u-f6d886bd73c094ec80"></a>

<!-- source-depth-inference-Д -->

<!-- /unit:f6d886bd73c094ec80 -->

## Результат главы и переход дальше

Ответьте на вопрос: **Почему проверка draft сохраняет заданное распределение?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
