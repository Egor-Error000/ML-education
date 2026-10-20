# Д. Speculative decoding: угадать несколько токенов и проверить

[Оглавление](README.md). C.1; углубления и case study из чата.

**Основа:** S4, строки 6439–8230. Сохранены вопросы «станет ли большая модель быстрее от прогресса маленьких?», «что с beam search?» и последующее обсуждение DSpark. **Исправления C12/C13/C20:** сохраняется target distribution, а не обязательно тот же текст для seed; acceptance без residual недостаточен; stronger draft не автоматически быстрее. **Углубление:** числовое доказательство sampling correction и явная speed модель.

**Уровень:** Углубление. **Предпосылки:** D1 and probability distributions. **Используют:** High-throughput decode/agent serving.

## 1. Откуда берётся выигрыш

Draft policy q дешево предлагает block из γ tokens. Target p вычисляет probabilities для предложенных prefixes совместно, пользуясь parallel processing positions. Это выгодно, если verification block дешевле γ независимых target decode steps и достаточно proposals принимается.

Зависимость tokens не исчезает: draft создаёт их последовательно или специальным parallel/semi-autoregressive методом, а target оценивает каждый при соответствующем prefix. После первого rejection неподтверждённый suffix нельзя просто оставить: его conditioning содержит отклонённый token.

В one-target-iteration схеме при принятии всех γ proposals часто можно дополнительно sample ещё один target token из уже вычисленных logits. Cache commit/rollback должен сохранять ровно принятый prefix; иначе outputs могут выглядеть правдоподобно, но algorithm перестанет соответствовать заявленному distribution.

### Почему проверка нескольких позиций не нарушает авторегрессию

В чате forward большой модели сравнивался с дорогим походом за весами, который хочется оплатить сразу для нескольких токенов. Это полезная интуиция для bandwidth-bound decode, но 5–10%GPU utilization не является свойством всех LLM. При достаточно большом batch или другом attention workload ограничение может перейти к compute. Тогда дополнительные проверяемые позиции конкурируют с другими запросами и уже не «достаются бесплатно».

Target не угадывает все будущие токены независимо. Draft уже предложил x1,x2,x3, поэтому target может одновременно вычислить p(x1|h),p(x2|h,x1),p(x3|h,x1,x2) с causal mask. Это обычная оценка **известной** последовательности при заданных prefixes. Пока первая позиция не принята, следующие остаются условными результатами для черновой истории, а не окончательными ответами. Если x2 отклонён, target score для x3 был получен при неправильном prefix и не может продолжать выбранный результат.

Вычисленные logits и проверяемые token IDs должны быть сдвинуты так же, как в next-token objective. Logits после h предсказывают x1, после h+x1 —x2. Сравнивать p для token с logits **после самого этого token** —ошибка на одну позицию. Выигрыш получается от организации матричных операций и уменьшения числа последовательных target iterations, а не от разрешения модели видеть своё будущее.

### Полный цикл с явным commit и rollback

В псевдокоде S4 список `generated` сразу пополнялся всеми draft tokens, а после rejection новый token брали прямо из target. Для stochastic sampling это недостаточно: нужен residual, выведенный ниже. Полезно логически отделить committed history h от временного proposal. Тогда цикл читается так:

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

Bonus token и token из residual ещё могут требовать обработки на следующем forward, чтобы их K/V появились в cache. Нельзя считать строку вывода и физический cache одинаковыми на любой промежуточной границе. Draft также должен продолжать **исправленную** history, а не хранить незаметно своё отклонённое продолжение. При EOS или достижении token budget нельзя дописать лишний bonus только ради формулы ускорения.

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

### От accepted mass к точному распределению

Обозначим A=Σmin(p,q). Это полная вероятность принять предложенный token. Тогда вероятность rejection1−A равна Σ[p−q]₊, поскольку обе исходные probability vectors суммируются в 1. Для любого v итоговая масса:

$$
\Pr(Y=v)=\min(p(v),q(v))+(1-A)\frac{[p(v)-q(v)]_+}{1-A}=p(v).
$$

Последняя дробь используется только при 1−A>0. Это не «принятый draft плюс ещё один независимый target sample», а разложение нужной массы на уже полученную и недостающую. Важно и содержание rejected sample: если q слишком часто предлагала A, correction должна добавить недостающую массу других tokens, а не ещё раз пропорционально всему p.

Для принятия одного выбранного x достаточно scalar p(x),q(x), но чтобы построить residual по vocabulary, нужны значения p(v),q(v) для остальных v либо другой корректный способ сэмплировать ту же correction. Поэтому массив `draft_probs` только с вероятностями выбранных tokens из S4 не задаёт весь stochastic algorithm. Сохранение top-k probabilities с потерянной tail тоже требует собственного обоснования.

### Исходный пример «Paris»: что из него можно вычислить

В S4 draft предложил пять токенов с вероятностями (0.95, 0.80, 0.70, 0.60, 0.50), а вероятности этих же предложений у target были (0.90, 0.30, 0.80, 0.65, 0.55). Для первой позиции вероятность принятия $18/19\approx0.94737$; случайное число 0.70 даёт принятие. Для второй вероятность принятия 0.375; случайное число 0.50 даёт отклонение. Проверка остальных трёх после этого не меняет принятый префикс: они зависели от отклонённого второго токена.

Однако по этим десяти скалярным вероятностям нельзя определить, что коррекция обязательно выберет «is», как написал исходный ответ: полного остаточного распределения нет. Установлено только, что принялся один токен draft, а следующий выбирается отдельной корректирующей процедурой. И два выпущенных токена за один проход target ещё не означают двукратного ускорения. Если draft пяти токенов стоил 5 мс, проверка — 10 мс, такой конкретный цикл занял 15 мс вместо 20 мс baseline для двух токенов, то есть дал примерно 1.33× до прочих расходов.

### Вероятность принятия и связь с качеством ответов

Используем $\min(a,b)=(a+b-|a-b|)/2$. Суммирование даёт:

$$
A=1-\frac12\sum_v|p(v)-q(v)|=1-\operatorname{TV}(p,q).
$$

Важна близость **всех** вероятностей при том же контексте. Если истинный ответ A, но target распределяет вероятности между (A,B) как (0.1,0.9), draft $q_1=(0.1,0.9)$ имеет вероятность принятия 1, а более правдивый $q_2=(0.9,0.1)$ — только 0.2. Это контрпример отождествлению точности на внешней задаче и совпадения с фиксированным target. Чтобы улучшить сам target, нужна другая задача; точная speculation сохраняет в том числе его ошибки.

Если $p(v)>0$, $q(v)=0$, этот токен не предлагается draft, но может появиться через residual. Если $p(x)=0$ на реально предложенном x, он отклоняется. Та же логика применима к грамматическому запрету, если p уже определено с маской допустимых токенов и нормировкой. Доказательство [speculative sampling](https://arxiv.org/abs/2302.01318) требует именно согласованных распределений, а не независимого набора эвристических правил принятия.

## 3. Distribution preservation и identical text — разные требования

Два алгоритма могут sample одно и то же distribution, потребляя random numbers в разном порядке. Поэтому одинаковый seed не обязан дать одинаковую строку. Для greedy-equivalent variants проверяется сохранение argmax choices согласно отдельной процедуре, а не этот же rejection sampling theorem.

Floating point, approximate logits, tokenizer mismatch и filters могут менять probabilities. Чтобы заявлять exact target sampling, надо определить assumptions и numerical scope. Если draft/target имеют разные tokenizers, alignment сложнее обычного matching token ID; готовая схема должна поддерживать это явно.

### Sampling, greedy и изменение temperature

Для greedy достаточно принимать совпадающий с deterministic argmax target префикс и при первом несовпадении выдавать target argmax. Нужно сохранить tie-breaking и stop policy. Это другое заявленное поведение, чем sampling из temperature/top-p distribution. При p как точечной массе та же probability-интуиция вырождается в принятие совпадающего token и отклонение остальных, но программный path может быть другим.

В sampling temperature, top-p/top-k, repetition penalties и grammar state определяют фактические p и q. Draft должен сообщать probability той proposal policy, которая действительно выбрала x. Нельзя сэмплировать из truncated q, а в знаменателе использовать probabilities до truncation. Разные tokenizers дополнительно меняют пространство событий: одинаковый integer ID не обязан обозначать одинаковый кусок текста.

В S4 таблица утверждала, что высокая temperature обязательно уменьшает acceptance. Общего монотонного закона нет. Возьмём target logits(2,0),draft logits(0,2) и одинаковую T. При T=1 overlap≈0.2384; при T=2 overlap≈0.5379. С ростом T оба distributions приближаются к uniform, и overlap здесь растёт. На другом pair зависимость может быть иной. Для реального workload измеряют outcome выбранных transformed policies.

### Несколько draft и выбор по confidence меняют proposal law

Если сначала выбираем модель с известными weights mixture, а затем sample, можно определить соответствующую mixture q. Если же генерируем несколько candidates и оставляем наиболее уверенный **после просмотра tokens**, distribution выбранного предложения меняется. Старый q отдельного draft больше не описывает этот experiment.

Простой контрпример: два независимых draws из q(A)=q(B)=0.5, после чего выбираем A, если он встретился хотя бы раз. Выбранный token имеет probabilities(0.75,0.25). Если target=(0.5,0.5), а acceptance ошибочно считает q=(0.5,0.5), он принимает всё и выдаёт biased(0.75,0.25). Поэтому исходный `max(drafts,key=confidence)` нельзя без новой correction добавлять к доказанному algorithm. Адаптивная длина также не должна незаметно отбирать current candidates по будущей информации без учёта такого отбора.

## 4. Когда более сильный draft полезен

При условно постоянной acceptance α и независимой approximation survival ожидаемое число accepted draft tokens $\sum_{i=1}^{\gamma}\alpha^i$. С дополнительным target token на verification step ожидаемое total около $1+\sum_{i=1}^{\gamma}\alpha^i$. Реальные acceptances зависят от positions/tasks, поэтому это объясняющая модель, а не exact universal predictor.

Для γ4,α0.8: expected accepted $0.8+0.64+0.512+0.4096=2.3616$, total≈3.3616. Пусть target baseline step10ms, draft token1ms, verify block12ms. Стоимость цикла 16ms, baseline для 3.3616 tokens≈33.616ms, идеализированный speedup≈2.10.

Если сильный draft стоит 3ms/token и повышает α до 0.9, total≈$1+0.9+0.81+0.729+0.6561=4.0951$; cycle24ms; speedup≈1.71. Acceptance лучше, но итог медленнее предыдущего draft. Больше γ также может ухудшить batching/verification efficiency и memory.

Следовательно, прогресс small models **может** ускорить big target, если улучшает trade-off proposal cost/acceptance на нужных данных. Нужно измерять draft time, verifier time, accepted length, concurrency и per-user/aggregate показатели. Full serving cost не выводится из standalone accuracy draft.

### Почему число принятых tokens не равно γ·α

Пусть K —длина непрерывного принятого prefix. Для целочисленной K≥0 математическое ожидание E[K]=ΣPr(K≥i). Обозначим s_i=Pr(первые i proposals приняты). Тогда expected emitted progress без раннего EOS равен 1+Σs_i: дополнительная 1 —коррекция после rejection или bonus после полного принятия. Если conditional probability каждого следующего acceptance при уже принятом prefix равнаα, то s_i=αⁱ. Независимость всех token events отдельно от этого условного определения не нужна.

В S4 дляγ5,α0.8 использовалось 5·0.8=4accepted tokens. Правильная prefix model даёт 0.8+0.64+0.512+0.4096+0.32768=2.68928accepted proposals, total progress3.68928. При 5msdraft+10msverification среднее время одного output token около 4.066ms; speedup относительно 10ms baseline≈2.46. Это условный расчёт с fixed cycle costs, а не измерение pair моделей из чата.

Для разных позиций задают conditional α_i и s_i=∏α_j. Метрика «accepted/proposed» при заранее сгенерированном suffix, который потом отбросили, не обязана совпадать с таким α. Полезно логировать survival по глубине, среднюю committed length и waste verification отдельно. Максимальная theoretical длинаγ+1 не является средней. Увеличениеγ повышает ceiling, но не заставляет модель чаще принимать каждый уже существующий первый token.

### Ответ на ваш вопрос о прогрессе 7B→9B

Более качественное понимание русского, длинного контекста или кода может приблизить q к p на соответствующих запросах. Но имя семейства, более свежая версия, GQA или высокий MMLU сами по себе этого не доказывают. В S4 были выдуманные benchmark tables и неподтверждённые характеристики конкретных Qwen revisions. Сохраняем ваш вопрос и механизм, но не выдаём эти числа за реальное сравнение моделей.

При одинаковом 16bit payload9B требует 18GB против 14GB для 7B; при 4bit —4.5GB против 3.5GB до metadata. Сравнение 9BINT4 с 7BFP16 показывает ещё и эффект precision, а не только прогресса model family. У draft есть также собственный cache, activations и scheduling cost. Его длинный context limit не гарантирует эффективного чтения нужной информации или одинакового acceptance на всех языках.

Упрощённо $S\approx t_p(1+\sum_i s_i)/(t_d+t_v+t_o)$, где t_p —baseline target time/token, t_d —полный draft round, t_v —verification, t_o —прочие расходы. Все времена относятся к одинаковым workload/concurrency. Verification не обязана стоить один baseline token: растут processed positions, outputs и cache writes. При высокой нагрузке более длинная speculation может улучшить single-user скорость и одновременно уменьшить aggregate throughput.

Исходные 150/380/520tokens/s и Qwen acceptance68/78% не считаем измерениями без protocol. Для hypothetical стоимости cUSD/hour и routput tokens/s при полной загрузке cost/1M=c·10⁶/(3600r). С условным c=32 и r=180/540 получаем примерно 49.38/16.46USD. Это арифметика заданных чисел, не актуальный cloud price или доказанная экономия.100Mtokens/day требуют в среднем 1157.4tokens/s: одна hypothetical540tokens/s instance такого объёма за сутки не выдаст. Utilization, несколько instances, TTFT, retries и errors меняют реальный бюджет.

### Варианты proposal из чата — разные engineering задачи

Separate small model добавляет weights и state; одинаковая tokenizer semantics удобна, но одной семьи недостаточно для высокой acceptance. N-gram prompt lookup предлагает повтор уже встреченного продолжения без второй neural model. Он может помогать любому тексту с полезными повторами, а не только коду. Отсутствие match приводит к короткому proposal или обычному target step; это не ошибка самого target.

Early exit использует intermediate representation. Но применить final LM head к произвольному восьмому слою 32-layer checkpoint недостаточно, чтобы гарантировать полезный draft. [LayerSkip](https://arxiv.org/abs/2404.16710) связывает early-exit quality с training recipe и reuse вычислений. Для quantized копии target сохранение architecture также не делает её cache автоматически тождественным float cache: representations различаются. Выгрыш времени зависит от kernels, а не только от bitwidth.

[Medusa](https://arxiv.org/abs/2401.10774) использует дополнительные prediction heads и candidate verification. [EAGLE](https://arxiv.org/abs/2401.15077) рассматривает draft на уровне features с учётом token uncertainty. Их нельзя полностью определить одним class из нескольких независимых Linear layers. Требуются обучение, conditioning, attention masks и правило принятия конкретного метода. Сохранение качества в benchmark и точное сохранение target sampling law тоже нужно различать, если используется approximate/typical acceptance.

## 5. Почему beam search отдельная задача

Beam search хранит несколько prefixes и отбирает их по sequence score/length policy. Это не sample из p каждого следующего token. Его pruning path зависит от scores множества hypotheses. Стандартное speculative sampling proof не доказывает exact equivalence beam result.

Можно разрабатывать specialized speculative/tree/beam procedures, но нужно определить, сохраняется ли тот же beam objective, top hypotheses или лишь approximation. Отдельно учитывать memory на branching KV, shared prefixes и beam reordering. «Просто проверить один draft block» не означает проверку всех ветвей beam.

Self-speculative methods используют слои/heads той же модели или другие proposals вместо отдельного малого checkpoint. Это экономит некоторые расходы, но имеет свои training/runtime requirements. MTP, tree speculation и beam — разные идеи, хотя все могут создавать несколько candidates.

### Разворачиваем ваш вопрос о нарушении beam search

Beam width B задаёт число сохраняемых hypotheses после каждого pruning step. Сумма log probabilities target, length normalization, EOS handling, early stopping и tie-breaking определяют конкретный baseline beam procedure. Даже обычный finite-width beam —эвристический поиск: он может потерять globally optimal последовательность раньше. Требование exact equivalence означает воспроизвести **этот** baseline, а не доказать global optimum по всем текстам.

В чате предлагалось «draft дал много ветвей, target оценил все, выбралиtop-B». Это необходимая часть, но неполнота candidate set всё ещё может изменить результат. Допустим, текущие prefixes A иB имеют masses0.55 и 0.45. Для A target даёт X0.49,Y0.51; для B —Z0.6,W0.4. Полные joint scores: AX0.2695,AY0.2805,BZ0.27,BW0.18. При width2 baseline сохраняет AY иBZ.

Если draft предложил только AX иBZ, target честно пересчитает их probabilities, но выберет AX иBZ. Непредложенная AY была лучше AX. Значит, верный scoring **увиденных** candidates не доказывает equivalence полного beam update. Нужны достаточное candidate coverage, границы на исключённые alternatives либо fallback, который воспроизводит baseline frontier. Блочный выбор лучших terminal branches также не всегда совпадает с промежуточным pruning на каждом depth.

Исходный счёт B4×k4×depth5 равен 80token positions до sharing, а не 100. У полного дерева branching factor k число nodes может расти экспоненциально по depth, но fixed-width beam с pruning на каждом step не обязан раскрывать всё дерево k^depth. Различать эти procedures нужно и для memory budget: общие ancestors могут sharing KV, private suffix и reordering остаются отдельными расходами.

В tree verification query узла видит committed prefix и ancestors **своей** ветки. Соседняя ветка в packed tensor не должна попадать в его causal context. Правильный mask и cache isolation сохраняют target conditional probabilities. Но они сами ещё не выбирают между sampling, greedy и beam: это следующий уровень algorithm. Для sampling дерево служит предложением candidates по заявленной процедуре, а не автоматически сохраняемым списком лучших полных ответов.

## 6. DSpark: опубликованный case study

В продолжении чата обсуждалась [DSpark](https://arxiv.org/abs/2607.05147). Parallel backbone подготавливает блок, а лёгкий sequential module добавляет зависимости между proposals. В Markov-варианте previous token через low-rank transition задаёт bias к logits следующей позиции; это не независимый sample каждого слова. Confidence head оценивает conditional acceptance, а scheduler учитывает prefix survival и профиль производительности engine при выборе verification length.

Paper сообщает deployment в DeepSeek-V4 и прирост per-user generation speed60–85% относительноMTP-1 при matched throughput. Это сравнение конкретной serving system, не обещание любому engine. Прирост скорости 60% означает множитель 1.6, а при одинаковом числе tokens reciprocal decode time0.625 исходного, то есть уменьшение 37.5%; это не «latency уменьшилась на 60%». Offline accepted length, aggregate throughput и live per-user speed имеют разные denominators.

Исходный чат правильно связывал три задачи: удешевить draft, сохранить зависимости внутри него и не расходовать verification capacity на маловероятный suffix. Но фраза «один parallel pass стоит ровно как один scalar draft token» не является гарантией. Подробности conditioning, training и scheduler нужны конкретной реализации. Пример независимых пар и расчёт адаптивной длины ниже поясняют мотив этих механизмов собственными учебными числами.

Старые проценты/догадки из чата не превращаются в пользовательский benchmark. Отдельная [историческая research note](../notes/llm-inference-research.md) сохраняет provenance; при воспроизведении нужны workload, concurrency, hardware и точная метрика.

### MTP, DSpark и tree adaptation: сохраняем связь с вашими вопросами

В [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437) multi-token prediction задаёт дополнительные предсказания с conditioning и training objective. Shared embeddings/head не означают нулевой compute. Один speculative proposal за round при идеальном принятии позволяет до двух output tokens за target verification, если получен bonus; это не правило «всегда два полных target passes ради трёх tokens». Startup, draft work и конкретный pipeline определяют реальное отношение. ЧислоMTP1 характеризует proposal глубину используемого baseline, а не обязательное число выдаваемых tokens всего цикла.

Вопрос «почему независимые parallel heads могут плохо склеивать текст» удобно рассмотреть без названия модели. Пусть допустимы пары «of course» и «no problem» с probabilities0.5/0.5. Их independent marginals дают каждой первой/второй части по 0.5. Независимая сборка выдаёт также «of problem» и «no course» с общей массой 0.5, хотя исходное joint распределение их не содержало. Это потеря зависимости, а не недостаток grammar parser. Лёгкое conditioning на предыдущую выбранную часть может убрать такой конфликт; более дальние зависимости потребуют более богатого state.

В собственном toy scheduler conditional survival(0.9,0.5,0.2) даёт s=(0.9,0.45,0.09). Expected progress для проверяемой длины 1/2/3 равен 1.9/2.35/2.44. Если round costs соответственно 10/12/18ms, rates≈0.190/0.196/0.136tokens/ms: выгодна длина 2. Если под другой нагрузкой costs стали 10/16/18ms, выигрывает длина 1. Это объясняет связь confidence и serving capacity без предположения, что максимальный draft всегда лучше. Оценки должны быть calibrated под используемое событие survival.

Планировщик также не должен подглядывать в realized future candidate, чтобы задним числом решить, допускать ли сам этот candidate к стандартной acceptance correction. Контрпример с выбором A из двух drafts выше показывает общий механизм selection bias. Confidence и load policy нужно включить в описание procedure; одной ссылки на rejection sampling для всех adaptive schemes недостаточно.

Вторая работа, [tree adaptation к DeepSeek-V4](https://arxiv.org/abs/2609.24698), обсуждает специфические ограничения compressed attention state. Общий принцип можно выразить как S_{t+1}=F(S_t,x_{t+1}). Две speculative ветки A/B создают разные временные состояния из одного committed S_t. Если их updates смешались в persistent compressed state, простое удаление нескольких обычных K/V строк может не восстановить исходную историю. Поэтому нужны branch-local scratch states и перенос только корректно принятого состояния. Это развивает уже разобранный cache commit/rollback, а не отменяет его.

Числа про CSA/HCA, cacheGiB и ratios в S4 смешивались из разных описаний; они не нужны как universal constants этой процедуры. Эксперимент tree adaptation и deployment DSpark —разные свидетельства. Наличие research implementation не доказывает использование каждого tree variant на всех production requests. Из публикаций также нельзя вывести глобальный запрет beam search во всех сервисах компании.

### Практический протокол завершает объяснение

Сначала фиксируют target/tokenizer revision и точное p после decoding transformations. Затем определяют proposal q, method, max depth, cache policy и поддерживаемые runtime combinations. CLI snippets S4 с именами draft checkpoints и speculative flags не являются воспроизведённым запуском; наличие model name не гарантирует существование compatible auxiliary weights. Version-specific настройку проверяют по документации установленного engine.

На маленьком vocabulary полезно вычислить итоговую массу по всем accept/reject исходам и проверить recovery p. Для настоящей модели сравнивают baseline и speculative generation в одном режиме: greedy tokens при условиях equivalence либо statistical/distribution diagnostics для sampling, не обязательное совпадение строк по seed. Отдельно измеряют draft/verify/overhead time, committed progress, EOS/correction paths, memory и latency/throughput при нескольких concurrency/length slices. Ошибки residual, prefix alignment и rollback проверяют до объявления speedup.Числовые проверки всей книги не заменяют такой model experiment: практический запуск пользователя здесь не заявляется.

## 7. Самопроверка

1. Почему не достаточно accept/reject без correction? **Ответ:** accepted mass=min(p,q), нужно восполнить именно p-min(p,q).
2. Обязана ли строка совпасть при одном seed? **Ответ:** нет, distribution equivalence слабее одинакового random path.
3. Что делает expensive draft нежелательным? **Ответ:** его cost может перекрыть выигрыш accepted length.

**Практика:** пересчитать p/q example и speed examples. Затем положить p=q и объяснить acceptance/residual. Для эксперимента сравнить target baseline и speculation на одинаковом decoding с quality/distribution checks и измерениями accepted tokens, verifier/draft time, TTFT и decode latency.

<!-- latex-display-v1 -->

<!-- source-depth-inference-Д -->
