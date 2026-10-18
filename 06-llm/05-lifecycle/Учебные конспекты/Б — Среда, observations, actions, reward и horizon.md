# Б. Формальная постановка agentic-задачи

[Оглавление](README.md). Пункт 2 плана.

**Основа и дополнение:** S6:4–178 и 1377–2197 дают tool loop и coordination. Формальная MDP/POMDP постановка, credit assignment и toy environment ниже — новое углубление по плану, а не восстановленное занятие о training.

## 1. Состояние мира и доступное наблюдение

Environment state $s_t$ описывает существенное состояние задачи: corpus/version, текущие результаты, доступные объекты, время и последствия прошлых действий. Agent получает observation $o_t$, которое может показывать только часть состояния. Action $a_t$ меняет среду или запрашивает новую информацию.

Для MDP задаются states, actions, transition kernel, reward, discount и начальное распределение. В partially observable setting policy обычно опирается на историю:

$$
h_t=(o_1,a_1,o_2,a_2,\ldots,o_t),\qquad
a_t\sim\pi_\theta(\cdot\mid h_t).
$$

Model context — ограниченное представление этой истории, не обязательно полный $h_t$. Truncation/summary могут потерять information. Важно различать hidden state среды и скрытые нейросетевые activations: это разные смыслы слова state.

Transition можно записать:

$$
s_{t+1}\sim P(\cdot\mid s_t,a_t),\quad
o_{t+1}\sim O(\cdot\mid s_{t+1},a_t),\quad
r_t=R(s_t,a_t,s_{t+1}).
$$

Read-only search тоже меняет epistemic/task state: агент получил новые evidence и потратил budget, хотя corpus не изменился. Mutable API меняет внешний мир дополнительно. Для теоретического начала достаточно read-only environment с проверяемым результатом.


<!-- source-depth-lifecycle-Б -->
### 1.1. Что делает задачу последовательной

В чате tool loop описан как «модель предложила вызов → программа выполнила → observation вернулся модели». Для обучения нужно дополнительно определить, какую задачу решает вся последовательность и какие сведения доступны на каждом шаге. Без этого нельзя отличить правильный action от правдоподобного текста, который случайно совпал с ответом.

Среда содержит всё, что существенно для будущих outcomes: документы и их доступность, active revision, pending operations, clock/remaining budget. Observation —то, что реально показано policy: сообщение пользователя, tool result, видимые metadata и ошибки. State среды может быть неизвестен policy, но доступен evaluator/simulator для проверки. Это не hidden state внутри Transformer: neural activations —способ вычисления решения, environment state —объект задачи.

Action в этой постановке —целый структурированный call или terminal answer. Token-level decoder генерирует его по частям, но внешняя среда не обязана реагировать на каждый токен. Другой исследовательский setup может считать токен отдельным action. Тогда horizon, cost и transition units меняются. Нельзя переносить число 6 действий как 6 tokens или 6 model calls без определения.

### 1.2. Markov property и частичная наблюдаемость

В MDP будущее условное распределение зависит от текущего достаточного state и action, а не требует всей прежней истории сверх этого state. Это свойство постановки, не утверждение, что model input содержит идеальный state. Если remaining steps не включены, одинаковые внешне ситуации на первом и последнем шаге могут иметь разные допустимые планы.

Пример: observation «найден report-policy» одинаков у двух задач, но в одной active revision v1, в другой v2. Если metadata скрыты, policy должна получить дополнительное evidence. Никакой classifier по одной этой фразе не различит два мира с гарантией. Если revision известна, а summary её потерял, это уже искусственно созданная неполнота памяти.

В POMDP policy использует доступную history или её достаточное представление. При известной модели среды belief distribution $b_t(s)=P(s_t=s\mid h_t)$ обновляется по действию и новому observation:

$$
b_{t+1}(s')\propto O(o_{t+1}\mid s',a_t)\sum_s P(s'\mid s,a_t)b_t(s).
$$

Это концептуальная Bayes update, а не обещание, что LLM внутри явно хранит и пересчитывает такую таблицу. Нормировочная константа делает сумму beliefs равной 1. Если prior active-v2=0.6, тест говорит «v2» с likelihood0.9 при v2 и 0.2 при v1, posterior равен $0.6\cdot0.9/(0.6\cdot0.9+0.4\cdot0.2)=27/31\approx0.870968$.

History в начале главы записана без отдельных rewards. Это подходит, если reward не подаётся policy или включён в observation. Если policy действительно видит информативный feedback/reward, history следует расширить им; иначе формально выбрасывается доступная информация. Контекст LLM может быть ещё короче этой history из-за window/summarization.

### 1.3. Search получает информацию, хотя не меняет corpus

Read-only действие не обязано менять физическое содержимое источника. Оно меняет доступное agent evidence и расходует ресурсы. Для описания можно разделить внешний world state и augmented task state с бюджетом/результатами. Belief policy также обновляется после чтения. Поэтому фраза «read-only ничего не меняет» неверна для planning, но и знание агента не следует путать с перезаписью database.

Поиск может быть stochastic, exact lookup —deterministic, а network failure —случайным или injected. Если разные runs читают разные corpus revisions, outcome зависит не только от policy. Freeze snapshot/reset protocol нужен для controlled comparison. Он не делает реальный мир навсегда стационарным; это явно выбранная модель эксперимента.


## 2. Конкретная маленькая среда

Задача: найти действующий срок хранения отчёта в документах, назвать срок и source/version. Доступны три tools: search corpus, read document, compare numeric dates/версии по заданному правилу. Можно завершить answer или abstain. Corpus содержит устаревший v1 и current v2, один distractor и явные access scopes.

State включает remaining calls/time, returned snippets и выбранные IDs. Success oracle проверяет correct duration, current version, source support и отсутствие forbidden data. Horizon, например 6 действий, задаёт maximum interactions, а не длину одного answer в tokens.

Ошибка, невидимая single-turn checker: final text правильного формата, но agent прочитал v1, потом выдумал v2; repeated search зациклился; правильный read observation потерян при state merge; action cap превышен. Эти failures требуют trajectory record.

Reset должен возвращать controlled initial state. Если corpus меняется между candidates, сравнение policy становится неоднозначным. Seeds фиксируют stochastic environment choices, но не заменяют version snapshots.


### 2.1. Полностью задаём игрушечную среду

Продолжим задачу о сроке хранения отчёта. Это искусственный read-only corpus, не правила реальной организации. R1:revision v1,obsolete,retention30days. R2:revision v2,active,retention90days. D:документ про timeout, distractor. X:документ другого scope, access запрещён. Пользователь спрашивает действующий срок для проекта P и просит source/version.

В этой toy постановке search возвращает разрешённые IDs, titles и revisions, **без числа дней**. Read принимает ранее найденный разрешённый ID и возвращает текст/metadata. Compare_versions сравнивает numeric v1/v2 по объявленному учебному правилу; реальную активность задаёт metadata, а не универсальное «самая большая строка всегда действующая». Answer содержит days,source_id,revision; abstain имеет отдельный status.

Для простоты current rule здесь v2. Gold answer90days/R2/v2 известен evaluator, но не вставляется в первоначальный policy prompt. Directly угаданное 90 без доступного подтверждения не считается grounded success. Этот критерий нужен именно задаче с доказуемым source: в другой задаче правильный ответ из памяти мог бы оцениваться иначе.

Каждое реально начатое действие расходует один action slot, включая invalid call и отказ. Ошибка возвращается structured observation, без выдачи forbidden content. Denied attempt можно записать как invalid-action metric; если policy затем корректно восстановилась, final outcome определяется отдельной rubric. Здесь не вводится автоматическое правило «любая ошибка необратимо завершает episode».

### 2.2. Три траектории с разными outcomes

**Success:** search → observation с R1/R2; read(R2) → active v2/90days; answer(90,R2,v2) → oracle success. Три environment actions. ID R2 и его текст появляются в history до финального утверждения, а version не додумана после чтения R1.

**Stale failure:** search → R1/R2; read(R1) → obsolete v1/30days; answer(30,R1,v1) → failure по current-version задаче. Ответ faithful прочитанному тексту, но не выполняет запрос. Если policy вместо этого написала 90/R2/v2 без чтения R2, число могло бы совпасть, но evidence requirement всё равно не выполнен.

**Recovery:** первый search → transient timeout; второй search → R1/R2; read(R2) →90/v2; answer → success. Четыре actions. Первый timeout не означает, что документа нет. Эта trajectory полезна и как evaluation case, и как demonstration корректной реакции, если выбранная retry policy позволяет вторую попытку.

Можно добавить no-results, conflicting-current metadata и denied. Для каждого меняют observation и допустимое действие, а не только final prose. Если source реально недоступен в разрешённом scope, truthful abstention может быть успешным исходом отдельного unanswerable task. На текущем answerable task ранний отказ не получает success bonus.

### 2.3. Reset, seeds и сопоставимость episodes

Reset задаёт task, corpus snapshot, active revision, scope, initial evidence и budget. Agent memory от предыдущего episode не должна случайно подсказывать hidden gold answer новой задачи. Если изучается long-term memory, её наличие объявляют частью setup и сравнивают явно.

Один и тот же seed не всегда означает одинаковые внешние observations для разных policies: последовательности actions могут различаться и потреблять random draws в другом порядке. Для controlled fault injection удобно привязывать сценарий к task/action identity или сохранять tape observations. Но replay tape для другой policy не должен выдавать result несоответствующего tool call.

Training split и test split имеют разные task/source identities и условия, которые действительно должны переноситься. Поменять имя R2 наR200 при полностью том же шаблоне —узкий transfer test, а не проверка всех agentic задач. Теоретическая среда нужна для понимания механизмов; её простота и наблюдаемые outcomes не доказывают performance большой модели.


## 3. Return и delayed credit

Для конечной траектории horizon $H$ return от шага t:

$$
G_t=\sum_{k=t}^{H}\gamma^{k-t}r_k.
$$

Discount $\gamma\in[0,1]$ задаёт относительный вес будущего reward в выбранной постановке. Для finite tasks можно использовать $\gamma=1$ и явную step cost. Reward может приходить только в конце, когда oracle проверил result.

Пример: actions search, read, answer; cost каждого 0.05, terminal success bonus1. При $\gamma=1$ total $1-3\cdot0.05=0.85$. Неверная короткая answer из одного шага имеет $-0.05$, если success bonus0. Но если scale penalties слишком велика, policy может предпочесть ранний abstention: reward design меняет incentives.

Credit assignment спрашивает, какие decisions обеспечили outcome. Одна terminal failure не говорит, был ли плохой search query, неверный document selection или answer interpretation. Сохранять intermediate observations и reason-independent action labels важно для диагностики и построения training data.


### 3.1. Reward, outcome и стоимость —не одно поле

Oracle success проверяет условия задачи. Reward —число, которое выбирается для learning objective. Cost —ресурсы, которые можно включить штрафом либо ограничением. Если success bonus1 и каждый action costs0.05, правильный путь из трёх действий даёт rewards[-0.05,-0.05,0.95] и total0.85 при γ=1. Terminal reward0.95 уже включает последнюю action cost; дополнительно вычитать её второй раз нельзя.

Stale failure из трёх actions даёт[-0.05,-0.05,-0.05],total−0.15. Recovery из четырёх с тем же bonus —0.80. Неверный ответ за один action —−0.05. Таким образом, reward предпочитает успешный путь короткому неверному при этих scales, но правильное восстановление уступает чистому пути по стоимости.

Все три метрики всё равно сохраняют отдельно. Policy с высоким reward может просто экономить шаги или использовать слабый oracle. Evaluator должен показывать success, actions/cost и invalid outcomes независимо, чтобы рост reward не скрывал недопустимые действия или ухудшение качества.

### 3.2. Discount меняет цель и credit ранних шагов

Для rewards[-0.05,-0.05,0.95],γ=0.9 return первого шага $-0.05+0.9(-0.05)+0.9^2(0.95)=0.6745$. Return второго 0.805, последнего 0.95. Один terminal success таким образом относится к разным ранним decisions с разным discount. При γ=1 return по той же траектории 0.85,0.90,0.95.

Discount —не вероятность, что задача сама собой завершится, если это отдельно не задано model environment. Это weighting future rewards в objective. Большой horizon усложняет assignment: ошибочное чтение могло случиться задолго до final failure. И наоборот, неудачный search иногда нужен как информативный шаг, если alternatives неизвестны заранее.

При γ=0 первый шаг оценивается только immediate reward. Если informative search всегда получает cost, а success bonus приходит позже, такой objective не стимулирует долгий полезный путь. Поэтому γ выбирают вместе с task horizon/reward formulation, а не как декоративную стандартную константу.

### 3.3. Слишком большая step cost и reward hacking

Для answerable toy task сравним success за 3 actions,return1−3c, и immediate abstain за 1 action,return−c без bonus. Success выгоднее iff c<0.5. При c=0.6 правильный путь−0.8, отказ−0.6: objective предпочитает отказ, хотя task success хуже. Это не баг optimizer; цель задана с такими incentives.

Если bonus выдают за слово «v2», policy может писать v2 без чтения. Если за количество citations —дублировать ссылки. Если за «нет ошибок» —не вызывать ни одного инструмента. Reward нужно проверять adversarial counterexamples и независимым outcome oracle. Данные о том, **что** считается correct, не заменяют вопрос, **как** reward численно побуждает выбирать actions.

Intermediate shaping может давать feedback за валидный read или нужное evidence, но потенциально смещает objective к накоплению промежуточных достижений. Для нового shaping надо проверить, не выгодно ли бесконечно повторять «полезный» search. General policy gradient/credit assignment будут разобраны в главе Е; здесь формулируется сам signal.


## 4. Termination и budgets

Environment может terminate при verified completion, permanent error или недопустимом действии; timeout/horizon могут truncать episode без успеха. Эти причины различаются: horizon exhaustion не следует записывать как normal success termination только из-за завершённого текста.

Budget включает tools, tokens, wall-clock и nested work. Policy получает доступное remaining budget или application принудительно ограничивает calls. Если budget скрыт, model может планировать длинный путь, который executor всегда обрезает; train/inference mismatch ухудшает quality.

Частично наблюдаемая environment может требовать уточнения пользователя. Правильный action тогда — запрос missing information, если это предусмотрено task. Нельзя считать failure любую pause только потому, что evaluator ожидает немедленную answer; oracle должен отражать истинную задачу.


### 4.1. Естественный terminal state и внешнее обрезание

Termination не означает только success. Task-defined failure, invalid terminal outcome и исчерпание **встроенного** finite horizon тоже могут быть terminal states среды. External collector может остановить ещё незавершённую задачу раньше —это truncation относительно исходной постановки. Поэтому «любой timeout/horizon всегда truncation» было бы неверно: важно, входит ли limit в определение задачи.

Например, если задача определена как «решить за максимум 6 actions», то шестой шаг без success переводит в terminal budget failure по этой задаче. Если бесконечно допускающая поиск среда собрана только на 4 actions ради удобства rollout collection, внешняя остановка не доказывает отсутствие future value. В [Gymnasium time-limit discussion](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/) эти случаи разделены; для finite-horizon Markov state важно remaining time.

Для value target $y_t=r_t+\gamma(1-\mathrm{terminated})V(s_{t+1})$ внешняя truncation сама по себе не зануляет bootstrap в соответствующей постановке. Например,r=−0.05,γ=0.9,V(next)=0.8 даёт 0.67, а ошибочное зануление —−0.05. Это учебный target illustration, не запуск RL algorithm. Если final state/observation для bootstrap недоступен, его нельзя выдумать.

Для пользовательского task report оба события могут означать «пока не решено», но learning semantics различаются. Нельзя кодировать все ended episodes одним success flag и ожидать корректного обучения value function.

### 4.2. Три бюджета и зависимые действия

Action slots ограничивают число interactions. Token budget ограничивает input/output work. Wall-clock deadline ограничивает ожидание и execution. Три search actions могут сжечь context на больших tool outputs; один зависший read может исчерпать всё время. Cap действий не гарантирует low latency, а max-output tokens не ограничивает число будущих tool calls.

Remaining budget можно дать policy как trusted observation, одновременно enforce его в runtime. Если скрыть факт, что остался один шаг, policy может планировать read+answer, который executor всегда обрывает. Если model-generated summary самовольно увеличил remaining_calls, это не разрешает новые calls. Источник бюджетного state —программа/условия задачи.

При parallel calls нужно определить accounting: два независимых reads могут выполняться за один coordinator round, но расходовать 2 tool actions. Read ID, который должен появиться в предыдущем search, не становится независимым из-за одновременной генерации JSON. Horizon в rounds,tools иtokens —разные величины, особенно для nested agents.

### 4.3. Запрос недостающей информации как action

Если нужная version или scope не определены, среда может разрешать action request_clarification. Это переход в ожидание внешнего observation, а не обязательно failure. В других задачах user input недоступен и требуется abstain/partial. Training/eval должны разделять эти contracts.

Oracle не должен штрафовать за корректное уточнение только потому, что dataset содержит один immediate answer target. Но и бессмысленный вопрос при уже полном evidence не становится идеальным поведением. Важны необходимость, стоимость и влияющий на outcome ответ пользователя. Для текущего редактирования конспекта этот механизм объясняется теоретически, новые вопросы пользователю не требуются.


## 5. Trajectory data и split

Запись содержит initial task/context, chronological actions, exact observations, versions, errors, rewards/termination и total cost. Tool result хранится отдельно от policy-generated tokens. Пользовательский final goal нельзя заменить более простой подзадачей после неуспеха.

Разделять train/test по задачам, source documents и templates, где нужно. Random split отдельных messages одной trajectory позволяет model видеть продолжение своей будущей test history. Если изучается recovery, holdout должен включать новые tasks/conditions с ошибками, а не те же logs с другой перестановкой строк.

### 5.1. Что сохранять на границе environment и policy

Task ID, initial snapshot и versions связывают episode с постановкой. Для каждого шага сохраняют model-visible history/input, proposal, validated action, actual result/error, usage, reward и termination reason. Reward log не должен подменять actual observation; hidden evaluator labels не должны случайно попасть в policy input до принятия решения.

Off-policy recorded trace описывает конкретно выполненные actions, а не любые желаемые alternatives. Нельзя взять reward успешного R2 чтения и присвоить его action read(R1) только потому, что final prose похож. Для supervised data можно отдельно разметить экспертное исправление на данной history; это уже новый label с объявленным происхождением, а не переписывание фактического прошлого.

Различают environment simulation и recorded external execution. В маленькой toy среде результаты специально заданы, и это достаточно для ручного изучения transitions/returns. Ни один такой расчёт не объявляется доказательством обучения пользовательской LLM или успешного взаимодействия с реальной организационной базой.

### 5.2. Самопроверка

1. Почему observation не равно state? **Ответ:** agent видит только доступную часть мира и может ошибаться о остальном.
2. Что значит reward delayed? **Ответ:** quality ранних actions определяется поздним outcome, а не немедленным label каждого шага.
3. Чем truncation отличается от success? **Ответ:** ограничение остановило episode, но задача могла остаться нерешённой.

4. Hidden state среды —activations Transformer? **Ответ:** нет, это разные понятия.
5. Может ли read-only search менять planning state? **Ответ:** да, меняются evidence/belief и budget, хотя corpus остаётся прежним.
6. Всегда ли конец horizon есть truncation? **Ответ:** нет, встроенный task horizon может быть terminal condition; external collector cap —другой случай.
7. Значит ли высокий reward высокий success? **Ответ:** только при достаточно корректном signal; proxy может иметь обходы.
8. Почему step cost0.6 опасна в toy3-action task? **Ответ:** success return−0.8 хуже immediate abstain−0.6 при выбранной reward scale.
9. Что делать с result, не полученным из-за timeout? **Ответ:** сохранить неизвестность/ошибку; не заменять вымышленным evidence или no-results.
10. Гарантирует ли одинаковый seed одинаковую trace разных policies? **Ответ:** нет, policies могут запрашивать разные actions и потреблять random choices иначе.

**Практика:** записать три toy trajectories: success через current v2, failure через stale v1 и recovery после search timeout. Для каждой указать observations, действия, return, termination и какой feedback доступен для обучения. Начать с этих записей, не с большого RL запуска.
