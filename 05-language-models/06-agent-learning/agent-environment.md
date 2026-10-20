# Среда, observations, actions и horizon

**Уровень:** основа ветки.

**Предпосылки:** [Постановка обучения агента и baseline](<agent-task.md>)

**Используют:** [Оценка траекторий и воспроизводимость](<agent-evaluation.md>)

**Учебный вопрос:** Как описать доступную агенту информацию и последствия действия?





## Карта вычисления и границы этапов

![Среда, observations, actions и horizon: входы, преобразования и выходы](<../../оформление/course/agent-environment.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Reward и targets используются в объявленном режиме обучения. При оценке политика фиксирована; success измеряется по всей попытке, включая повторы и бюджет.

<!-- unit:3fa033a89cf900831b -->
<a id="u-3fa033a89cf900831b"></a>

## Состояние мира и доступное наблюдение

<!-- /unit:3fa033a89cf900831b -->

<!-- unit:622246daeec49dc50a -->
<a id="u-622246daeec49dc50a"></a>

Environment state $s_t$ описывает существенное состояние задачи: corpus/version, текущие результаты, доступные объекты, время и последствия прошлых действий. Agent получает observation $o_t$, которое может показывать только часть состояния. Action $a_t$ меняет среду или запрашивает новую информацию.

<!-- /unit:622246daeec49dc50a -->

<!-- unit:3b01dfa0a538c0255c -->
<a id="u-3b01dfa0a538c0255c"></a>

Для MDP задаются states, actions, transition kernel, reward, discount и начальное распределение. В partially observable setting policy обычно опирается на историю:

<!-- /unit:3b01dfa0a538c0255c -->

<!-- unit:1a7d2d3bb4cd7e8b0a -->
<a id="u-1a7d2d3bb4cd7e8b0a"></a>

$$
h_t=(o_1,a_1,o_2,a_2,\ldots,o_t),\qquad
a_t\sim\pi_\theta(\cdot\mid h_t).
$$

<!-- /unit:1a7d2d3bb4cd7e8b0a -->

<!-- unit:52bd8896ce2ceedec0 -->
<a id="u-52bd8896ce2ceedec0"></a>

Model context — ограниченное представление этой истории, не обязательно полный $h_t$. Truncation/summary могут потерять information. Важно различать hidden state среды и скрытые нейросетевые activations: это разные смыслы слова state.

<!-- /unit:52bd8896ce2ceedec0 -->

<!-- unit:15d1f603a5c899eec1 -->
<a id="u-15d1f603a5c899eec1"></a>

Transition можно записать:

<!-- /unit:15d1f603a5c899eec1 -->

<!-- unit:04a443d9602ac5a934 -->
<a id="u-04a443d9602ac5a934"></a>

$$
s_{t+1}\sim P(\cdot\mid s_t,a_t),\quad
o_{t+1}\sim O(\cdot\mid s_{t+1},a_t),\quad
r_t=R(s_t,a_t,s_{t+1}).
$$

<!-- /unit:04a443d9602ac5a934 -->

<!-- unit:5a4187209e2174cba8 -->
<a id="u-5a4187209e2174cba8"></a>

Read-only search тоже меняет epistemic/task state: агент получил новые evidence и потратил budget, хотя corpus не изменился. Mutable API меняет внешний мир дополнительно. Для теоретического начала достаточно read-only environment с проверяемым результатом.

<!-- /unit:5a4187209e2174cba8 -->

<!-- unit:d32824000c14383296 -->
<a id="u-d32824000c14383296"></a>

**Схема курса:** environment state → observation to policy → action/tool call → transition and reward → next observation → terminal or truncated horizon

<!-- /unit:d32824000c14383296 -->

<!-- unit:3579a59b41533bf070 -->
<a id="u-3579a59b41533bf070"></a>

<!-- source-depth-lifecycle-Б -->

<!-- /unit:3579a59b41533bf070 -->

<!-- unit:2d2fd423a95515e260 -->
<a id="u-2d2fd423a95515e260"></a>

### Что делает задачу последовательной

<!-- /unit:2d2fd423a95515e260 -->

<!-- unit:9582eff1798ca61812 -->
<a id="u-9582eff1798ca61812"></a>

В разбираемом примере tool loop описан как «модель предложила вызов → программа выполнила → observation вернулся модели». Для обучения нужно дополнительно определить, какую задачу решает вся последовательность и какие сведения доступны на каждом шаге. Без этого нельзя отличить правильный action от правдоподобного текста, который случайно совпал с ответом.

<!-- /unit:9582eff1798ca61812 -->

<!-- unit:7a6ab37dc0af219be8 -->
<a id="u-7a6ab37dc0af219be8"></a>

Среда содержит всё, что существенно для будущих outcomes: документы и их доступность, active revision, pending operations, clock/remaining budget. Observation —то, что реально показано policy: сообщение пользователя, tool result, видимые metadata и ошибки. State среды может быть неизвестен policy, но доступен evaluator/simulator для проверки. Это не hidden state внутри Transformer: neural activations —способ вычисления решения, environment state —объект задачи.

<!-- /unit:7a6ab37dc0af219be8 -->

<!-- unit:92cdfdf4f4e46fe198 -->
<a id="u-92cdfdf4f4e46fe198"></a>

Action в этой постановке —целый структурированный call или terminal answer. Token-level decoder генерирует его по частям, но внешняя среда не обязана реагировать на каждый токен. Другой исследовательский setup может считать токен отдельным action. Тогда horizon, cost и transition units меняются. Нельзя переносить число 6 действий как 6 tokens или 6 model calls без определения.

<!-- /unit:92cdfdf4f4e46fe198 -->

<!-- unit:6e7d07e78a11bf4a3b -->
<a id="u-6e7d07e78a11bf4a3b"></a>

### Markov property и частичная наблюдаемость

<!-- /unit:6e7d07e78a11bf4a3b -->

<!-- unit:02f8116214f5748ecc -->
<a id="u-02f8116214f5748ecc"></a>

В MDP будущее условное распределение зависит от текущего достаточного state и action, а не требует всей прежней истории сверх этого state. Это свойство постановки, не утверждение, что model input содержит идеальный state. Если remaining steps не включены, одинаковые внешне ситуации на первом и последнем шаге могут иметь разные допустимые планы.

<!-- /unit:02f8116214f5748ecc -->

<!-- unit:8c555618321a9b064b -->
<a id="u-8c555618321a9b064b"></a>

Пример: observation «найден report-policy» одинаков у двух задач, но в одной active revision v1, в другой v2. Если metadata скрыты, policy должна получить дополнительное evidence. Никакой classifier по одной этой фразе не различит два мира с гарантией. Если revision известна, а summary её потерял, это уже искусственно созданная неполнота памяти.

<!-- /unit:8c555618321a9b064b -->

<!-- unit:96730fe7a5fa64418e -->
<a id="u-96730fe7a5fa64418e"></a>

В POMDP policy использует доступную history или её достаточное представление. При известной модели среды belief distribution $b_t(s)=P(s_t=s\mid h_t)$ обновляется по действию и новому observation:

<!-- /unit:96730fe7a5fa64418e -->

<!-- unit:c71f5c15186f099226 -->
<a id="u-c71f5c15186f099226"></a>

$$
b_{t+1}(s')\propto O(o_{t+1}\mid s',a_t)\sum_s P(s'\mid s,a_t)b_t(s).
$$

<!-- /unit:c71f5c15186f099226 -->

<!-- unit:1d88054d3404b65ca8 -->
<a id="u-1d88054d3404b65ca8"></a>

Это концептуальная Bayes update, а не обещание, что LLM внутри явно хранит и пересчитывает такую таблицу. Нормировочная константа делает сумму beliefs равной 1. Если prior active-v2=0.6, тест говорит «v2» с likelihood0.9 при v2 и 0.2 при v1, posterior равен $0.6\cdot0.9/(0.6\cdot0.9+0.4\cdot0.2)=27/31\approx0.870968$.

<!-- /unit:1d88054d3404b65ca8 -->

<!-- unit:177101494d3a739fab -->
<a id="u-177101494d3a739fab"></a>

History в начале главы записана без отдельных rewards. Это подходит, если reward не подаётся policy или включён в observation. Если policy действительно видит информативный feedback/reward, history следует расширить им; иначе формально выбрасывается доступная информация. Контекст LLM может быть ещё короче этой history из-за window/summarization.

<!-- /unit:177101494d3a739fab -->

<!-- unit:86781584e2976474a9 -->
<a id="u-86781584e2976474a9"></a>

### Search получает информацию, хотя не меняет corpus

<!-- /unit:86781584e2976474a9 -->

<!-- unit:93baded108a0cc5242 -->
<a id="u-93baded108a0cc5242"></a>

Read-only действие не обязано менять физическое содержимое источника. Оно меняет доступное agent evidence и расходует ресурсы. Для описания можно разделить внешний world state и augmented task state с бюджетом/результатами. Belief policy также обновляется после чтения. Поэтому фраза «read-only ничего не меняет» неверна для planning, но и знание агента не следует путать с перезаписью database.

<!-- /unit:93baded108a0cc5242 -->

<!-- unit:7c84d4c0f2a1473fd1 -->
<a id="u-7c84d4c0f2a1473fd1"></a>

Поиск может быть stochastic, exact lookup —deterministic, а network failure —случайным или injected. Если разные runs читают разные corpus revisions, outcome зависит не только от policy. Freeze snapshot/reset protocol нужен для controlled comparison. Он не делает реальный мир навсегда стационарным; это явно выбранная модель эксперимента.

<!-- /unit:7c84d4c0f2a1473fd1 -->

<!-- unit:7854d2fb69195cf5de -->
<a id="u-7854d2fb69195cf5de"></a>

## Конкретная маленькая среда

<!-- /unit:7854d2fb69195cf5de -->

<!-- unit:9c499d19d9f25a0208 -->
<a id="u-9c499d19d9f25a0208"></a>

Задача: найти действующий срок хранения отчёта в документах, назвать срок и source/version. Доступны три tools: search corpus, read document, compare numeric dates/версии по заданному правилу. Можно завершить answer или abstain. Corpus содержит устаревший v1 и current v2, один distractor и явные access scopes.

<!-- /unit:9c499d19d9f25a0208 -->

<!-- unit:79d4c360370ea4296b -->
<a id="u-79d4c360370ea4296b"></a>

State включает remaining calls/time, returned snippets и выбранные IDs. Success oracle проверяет correct duration, current version, source support и отсутствие forbidden data. Horizon, например 6 действий, задаёт maximum interactions, а не длину одного answer в tokens.

<!-- /unit:79d4c360370ea4296b -->

<!-- unit:af1b49623afb6b9819 -->
<a id="u-af1b49623afb6b9819"></a>

Ошибка, невидимая single-turn checker: final text правильного формата, но agent прочитал v1, потом выдумал v2; repeated search зациклился; правильный read observation потерян при state merge; action cap превышен. Эти failures требуют trajectory record.

<!-- /unit:af1b49623afb6b9819 -->

<!-- unit:d6f8de2db1ba91d8f6 -->
<a id="u-d6f8de2db1ba91d8f6"></a>

Reset должен возвращать controlled initial state. Если corpus меняется между candidates, сравнение policy становится неоднозначным. Seeds фиксируют stochastic environment choices, но не заменяют version snapshots.

<!-- /unit:d6f8de2db1ba91d8f6 -->

<!-- unit:7d0a8bfefa8d010f4e -->
<a id="u-7d0a8bfefa8d010f4e"></a>

### Полностью задаём игрушечную среду

<!-- /unit:7d0a8bfefa8d010f4e -->

<!-- unit:70ec4f5565cda5c93e -->
<a id="u-70ec4f5565cda5c93e"></a>

Продолжим задачу о сроке хранения отчёта. Это искусственный read-only corpus, не правила реальной организации. R1:revision v1,obsolete,retention30days. R2:revision v2,active,retention90days. D:документ про timeout, distractor. X:документ другого scope, access запрещён. Пользователь спрашивает действующий срок для проекта P и просит source/version.

<!-- /unit:70ec4f5565cda5c93e -->

<!-- unit:0d90044b5c2eaf75b7 -->
<a id="u-0d90044b5c2eaf75b7"></a>

В этой toy постановке search возвращает разрешённые IDs, titles и revisions, **без числа дней**. Read принимает ранее найденный разрешённый ID и возвращает текст/metadata. Compare_versions сравнивает numeric v1/v2 по объявленному учебному правилу; реальную активность задаёт metadata, а не универсальное «самая большая строка всегда действующая». Answer содержит days,source_id,revision; abstain имеет отдельный status.

<!-- /unit:0d90044b5c2eaf75b7 -->

<!-- unit:01f0116fb89a2a6b05 -->
<a id="u-01f0116fb89a2a6b05"></a>

Для простоты current rule здесь v2. Gold answer90days/R2/v2 известен evaluator, но не вставляется в первоначальный policy prompt. Directly угаданное 90 без доступного подтверждения не считается grounded success. Этот критерий нужен именно задаче с доказуемым source: в другой задаче правильный ответ из памяти мог бы оцениваться иначе.

<!-- /unit:01f0116fb89a2a6b05 -->

<!-- unit:a5506c76d3f32e776e -->
<a id="u-a5506c76d3f32e776e"></a>

Каждое реально начатое действие расходует один action slot, включая invalid call и отказ. Ошибка возвращается structured observation, без выдачи forbidden content. Denied attempt можно записать как invalid-action metric; если policy затем корректно восстановилась, final outcome определяется отдельной rubric. Здесь не вводится автоматическое правило «любая ошибка необратимо завершает episode».

<!-- /unit:a5506c76d3f32e776e -->

<!-- unit:a30adf852c2ce1f1f4 -->
<a id="u-a30adf852c2ce1f1f4"></a>

### Три траектории с разными outcomes

<!-- /unit:a30adf852c2ce1f1f4 -->

<!-- unit:251a99b7096a297c71 -->
<a id="u-251a99b7096a297c71"></a>

**Success:** search → observation с R1/R2; read(R2) → active v2/90days; answer(90,R2,v2) → oracle success. Три environment actions. ID R2 и его текст появляются в history до финального утверждения, а version не додумана после чтения R1.

<!-- /unit:251a99b7096a297c71 -->

<!-- unit:172b1d3fc274a15bba -->
<a id="u-172b1d3fc274a15bba"></a>

**Stale failure:** search → R1/R2; read(R1) → obsolete v1/30days; answer(30,R1,v1) → failure по current-version задаче. Ответ faithful прочитанному тексту, но не выполняет запрос. Если policy вместо этого написала 90/R2/v2 без чтения R2, число могло бы совпасть, но evidence requirement всё равно не выполнен.

<!-- /unit:172b1d3fc274a15bba -->

<!-- unit:8830be5fcfadf1c0fd -->
<a id="u-8830be5fcfadf1c0fd"></a>

**Recovery:** первый search → transient timeout; второй search → R1/R2; read(R2) →90/v2; answer → success. Четыре actions. Первый timeout не означает, что документа нет. Эта trajectory полезна и как evaluation case, и как demonstration корректной реакции, если выбранная retry policy позволяет вторую попытку.

<!-- /unit:8830be5fcfadf1c0fd -->

<!-- unit:760b4997051a4fc0fb -->
<a id="u-760b4997051a4fc0fb"></a>

Можно добавить no-results, conflicting-current metadata и denied. Для каждого меняют observation и допустимое действие, а не только final prose. Если source реально недоступен в разрешённом scope, truthful abstention может быть успешным исходом отдельного unanswerable task. На текущем answerable task ранний отказ не получает success bonus.

<!-- /unit:760b4997051a4fc0fb -->

<!-- unit:3541bfb7829f20aeff -->
<a id="u-3541bfb7829f20aeff"></a>

### Reset, seeds и сопоставимость episodes

<!-- /unit:3541bfb7829f20aeff -->

<!-- unit:77911af9eafde0c90d -->
<a id="u-77911af9eafde0c90d"></a>

Reset задаёт task, corpus snapshot, active revision, scope, initial evidence и budget. Agent memory от предыдущего episode не должна случайно подсказывать hidden gold answer новой задачи. Если изучается long-term memory, её наличие объявляют частью setup и сравнивают явно.

<!-- /unit:77911af9eafde0c90d -->

<!-- unit:55e4944847ea9f618e -->
<a id="u-55e4944847ea9f618e"></a>

Один и тот же seed не всегда означает одинаковые внешние observations для разных policies: последовательности actions могут различаться и потреблять random draws в другом порядке. Для controlled fault injection удобно привязывать сценарий к task/action identity или сохранять tape observations. Но replay tape для другой policy не должен выдавать result несоответствующего tool call.

<!-- /unit:55e4944847ea9f618e -->

<!-- unit:83f44d0338f86cdfe2 -->
<a id="u-83f44d0338f86cdfe2"></a>

Training split и test split имеют разные task/source identities и условия, которые действительно должны переноситься. Поменять имя R2 наR200 при полностью том же шаблоне —узкий transfer test, а не проверка всех agentic задач. Теоретическая среда нужна для понимания механизмов; её простота и наблюдаемые outcomes не доказывают performance большой модели.

<!-- /unit:83f44d0338f86cdfe2 -->

<!-- unit:a5bd1feb8bc579f370 -->
<a id="u-a5bd1feb8bc579f370"></a>

## Return и delayed credit

<!-- /unit:a5bd1feb8bc579f370 -->

<!-- unit:608ef6956781de5886 -->
<a id="u-608ef6956781de5886"></a>

Для конечной траектории horizon $H$ return от шага t:

<!-- /unit:608ef6956781de5886 -->

<!-- unit:697fae7166014f9fff -->
<a id="u-697fae7166014f9fff"></a>

$$
G_t=\sum_{k=t}^{H}\gamma^{k-t}r_k.
$$

<!-- /unit:697fae7166014f9fff -->

<!-- unit:c2ebefb61304ec356b -->
<a id="u-c2ebefb61304ec356b"></a>

Discount $\gamma\in[0,1]$ задаёт относительный вес будущего reward в выбранной постановке. Для finite tasks можно использовать $\gamma=1$ и явную step cost. Reward может приходить только в конце, когда oracle проверил result.

<!-- /unit:c2ebefb61304ec356b -->

<!-- unit:cdfbb3c7614bae8d4e -->
<a id="u-cdfbb3c7614bae8d4e"></a>

Пример: actions search, read, answer; cost каждого 0.05, terminal success bonus1. При $\gamma=1$ total $1-3\cdot0.05=0.85$. Неверная короткая answer из одного шага имеет $-0.05$, если success bonus0. Но если scale penalties слишком велика, policy может предпочесть ранний abstention: reward design меняет incentives.

<!-- /unit:cdfbb3c7614bae8d4e -->

<!-- unit:fa9c12fe7e095cab26 -->
<a id="u-fa9c12fe7e095cab26"></a>

Credit assignment спрашивает, какие decisions обеспечили outcome. Одна terminal failure не говорит, был ли плохой search query, неверный document selection или answer interpretation. Сохранять intermediate observations и reason-independent action labels важно для диагностики и построения training data.

<!-- /unit:fa9c12fe7e095cab26 -->

<!-- unit:9d02001e2e8142afa3 -->
<a id="u-9d02001e2e8142afa3"></a>

### Reward, outcome и стоимость —не одно поле

<!-- /unit:9d02001e2e8142afa3 -->

<!-- unit:51e91b61d804f83afd -->
<a id="u-51e91b61d804f83afd"></a>

Oracle success проверяет условия задачи. Reward —число, которое выбирается для learning objective. Cost —ресурсы, которые можно включить штрафом либо ограничением. Если success bonus1 и каждый action costs0.05, правильный путь из трёх действий даёт rewards[-0.05,-0.05,0.95] и total0.85 при γ=1. Terminal reward0.95 уже включает последнюю action cost; дополнительно вычитать её второй раз нельзя.

<!-- /unit:51e91b61d804f83afd -->

<!-- unit:fe25fabea86507537f -->
<a id="u-fe25fabea86507537f"></a>

Stale failure из трёх actions даёт[-0.05,-0.05,-0.05],total−0.15. Recovery из четырёх с тем же bonus —0.80. Неверный ответ за один action —−0.05. Таким образом, reward предпочитает успешный путь короткому неверному при этих scales, но правильное восстановление уступает чистому пути по стоимости.

<!-- /unit:fe25fabea86507537f -->

<!-- unit:44939536d9f16b2046 -->
<a id="u-44939536d9f16b2046"></a>

Все три метрики всё равно сохраняют отдельно. Policy с высоким reward может просто экономить шаги или использовать слабый oracle. Evaluator должен показывать success, actions/cost и invalid outcomes независимо, чтобы рост reward не скрывал недопустимые действия или ухудшение качества.

<!-- /unit:44939536d9f16b2046 -->

<!-- unit:54ac47d6f0e518d736 -->
<a id="u-54ac47d6f0e518d736"></a>

### Discount меняет цель и credit ранних шагов

<!-- /unit:54ac47d6f0e518d736 -->

<!-- unit:1e0cf3f1c709ae9d67 -->
<a id="u-1e0cf3f1c709ae9d67"></a>

Для rewards[-0.05,-0.05,0.95],γ=0.9 return первого шага $-0.05+0.9(-0.05)+0.9^2(0.95)=0.6745$. Return второго 0.805, последнего 0.95. Один terminal success таким образом относится к разным ранним decisions с разным discount. При γ=1 return по той же траектории 0.85,0.90,0.95.

<!-- /unit:1e0cf3f1c709ae9d67 -->

<!-- unit:cfc8780026d5671784 -->
<a id="u-cfc8780026d5671784"></a>

Discount —не вероятность, что задача сама собой завершится, если это отдельно не задано model environment. Это weighting future rewards в objective. Большой horizon усложняет assignment: ошибочное чтение могло случиться задолго до final failure. И наоборот, неудачный search иногда нужен как информативный шаг, если alternatives неизвестны заранее.

<!-- /unit:cfc8780026d5671784 -->

<!-- unit:7956b92fa1cfc49fb5 -->
<a id="u-7956b92fa1cfc49fb5"></a>

При γ=0 первый шаг оценивается только immediate reward. Если informative search всегда получает cost, а success bonus приходит позже, такой objective не стимулирует долгий полезный путь. Поэтому γ выбирают вместе с task horizon/reward formulation, а не как декоративную стандартную константу.

<!-- /unit:7956b92fa1cfc49fb5 -->

<!-- unit:041aed9d909afe6e96 -->
<a id="u-041aed9d909afe6e96"></a>

### Слишком большая step cost и reward hacking

<!-- /unit:041aed9d909afe6e96 -->

<!-- unit:85c4e8a6617f424e70 -->
<a id="u-85c4e8a6617f424e70"></a>

Для answerable toy task сравним success за 3 actions,return1−3c, и immediate abstain за 1 action,return−c без bonus. Success выгоднее iff c<0.5. При c=0.6 правильный путь−0.8, отказ−0.6: objective предпочитает отказ, хотя task success хуже. Это не баг optimizer; цель задана с такими incentives.

<!-- /unit:85c4e8a6617f424e70 -->

<!-- unit:838a926c4fcfd8c564 -->
<a id="u-838a926c4fcfd8c564"></a>

Если bonus выдают за слово «v2», policy может писать v2 без чтения. Если за количество citations —дублировать ссылки. Если за «нет ошибок» —не вызывать ни одного инструмента. Reward нужно проверять adversarial counterexamples и независимым outcome oracle. Данные о том, **что** считается correct, не заменяют вопрос, **как** reward численно побуждает выбирать actions.

<!-- /unit:838a926c4fcfd8c564 -->

<!-- unit:03b1904c5950d0fe99 -->
<a id="u-03b1904c5950d0fe99"></a>

Intermediate shaping может давать feedback за валидный read или нужное evidence, но потенциально смещает objective к накоплению промежуточных достижений. Для нового shaping надо проверить, не выгодно ли бесконечно повторять «полезный» search. General policy gradient/credit assignment будут разобраны в главе [«Обучение через взаимодействие: PPO и GRPO»](<agent-rl.md>); здесь формулируется сам signal.

<!-- /unit:03b1904c5950d0fe99 -->

<!-- unit:80107b035af673081b -->
<a id="u-80107b035af673081b"></a>

## Termination и budgets

<!-- /unit:80107b035af673081b -->

<!-- unit:ab7bfe2772fa752d38 -->
<a id="u-ab7bfe2772fa752d38"></a>

Environment может terminate при verified completion, permanent error или недопустимом действии; timeout/horizon могут truncать episode без успеха. Эти причины различаются: horizon exhaustion не следует записывать как normal success termination только из-за завершённого текста.

<!-- /unit:ab7bfe2772fa752d38 -->

<!-- unit:83256a5746278abbc6 -->
<a id="u-83256a5746278abbc6"></a>

Budget включает tools, tokens, wall-clock и nested work. Policy получает доступное remaining budget или application принудительно ограничивает calls. Если budget скрыт, model может планировать длинный путь, который executor всегда обрезает; train/inference mismatch ухудшает quality.

<!-- /unit:83256a5746278abbc6 -->

<!-- unit:049689f47f928bcfa2 -->
<a id="u-049689f47f928bcfa2"></a>

Частично наблюдаемая environment может требовать уточнения пользователя. Правильный action тогда — запрос missing information, если это предусмотрено task. Нельзя считать failure любую pause только потому, что evaluator ожидает немедленную answer; oracle должен отражать истинную задачу.

<!-- /unit:049689f47f928bcfa2 -->

<!-- unit:8e1304f0b42f2314ee -->
<a id="u-8e1304f0b42f2314ee"></a>

### Естественный terminal state и внешнее обрезание

<!-- /unit:8e1304f0b42f2314ee -->

<!-- unit:661b7fe2d3757bf225 -->
<a id="u-661b7fe2d3757bf225"></a>

Termination не означает только success. Task-defined failure, invalid terminal outcome и исчерпание **встроенного** finite horizon тоже могут быть terminal states среды. External collector может остановить ещё незавершённую задачу раньше —это truncation относительно исходной постановки. Поэтому «любой timeout/horizon всегда truncation» было бы неверно: важно, входит ли limit в определение задачи.

<!-- /unit:661b7fe2d3757bf225 -->

<!-- unit:69049afa00c7c2f810 -->
<a id="u-69049afa00c7c2f810"></a>

Например, если задача определена как «решить за максимум 6 actions», то шестой шаг без success переводит в terminal budget failure по этой задаче. Если бесконечно допускающая поиск среда собрана только на 4 actions ради удобства rollout collection, внешняя остановка не доказывает отсутствие future value. В [Gymnasium time-limit discussion](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/) эти случаи разделены; для finite-horizon Markov state важно remaining time.

<!-- /unit:69049afa00c7c2f810 -->

<!-- unit:6f7f1731338c0b48e1 -->
<a id="u-6f7f1731338c0b48e1"></a>

Для value target $y_t=r_t+\gamma(1-\mathrm{terminated})V(s_{t+1})$ внешняя truncation сама по себе не зануляет bootstrap в соответствующей постановке. Например,r=−0.05,γ=0.9,V(next)=0.8 даёт 0.67, а ошибочное зануление —−0.05. Это учебный target illustration, не запуск RL algorithm. Если final state/observation для bootstrap недоступен, его нельзя выдумать.

<!-- /unit:6f7f1731338c0b48e1 -->

<!-- unit:d0860b7c0f5e3cf0c6 -->
<a id="u-d0860b7c0f5e3cf0c6"></a>

Для пользовательского task report оба события могут означать «пока не решено», но learning semantics различаются. Нельзя кодировать все ended episodes одним success flag и ожидать корректного обучения value function.

<!-- /unit:d0860b7c0f5e3cf0c6 -->

<!-- unit:f85915f20879751c94 -->
<a id="u-f85915f20879751c94"></a>

### Три бюджета и зависимые действия

<!-- /unit:f85915f20879751c94 -->

<!-- unit:6ad3144e79340d88f0 -->
<a id="u-6ad3144e79340d88f0"></a>

Action slots ограничивают число interactions. Token budget ограничивает input/output work. Wall-clock deadline ограничивает ожидание и execution. Три search actions могут сжечь context на больших tool outputs; один зависший read может исчерпать всё время. Cap действий не гарантирует low latency, а max-output tokens не ограничивает число будущих tool calls.

<!-- /unit:6ad3144e79340d88f0 -->

<!-- unit:ea39a0dd291ec32ac0 -->
<a id="u-ea39a0dd291ec32ac0"></a>

Remaining budget можно дать policy как trusted observation, одновременно enforce его в runtime. Если скрыть факт, что остался один шаг, policy может планировать read+answer, который executor всегда обрывает. Если model-generated summary самовольно увеличил remaining_calls, это не разрешает новые calls. Источник бюджетного state —программа/условия задачи.

<!-- /unit:ea39a0dd291ec32ac0 -->

<!-- unit:b9565b50971098b7dd -->
<a id="u-b9565b50971098b7dd"></a>

При parallel calls нужно определить accounting: два независимых reads могут выполняться за один coordinator round, но расходовать 2 tool actions. Read ID, который должен появиться в предыдущем search, не становится независимым из-за одновременной генерации JSON. Horizon в rounds,tools иtokens —разные величины, особенно для nested agents.

<!-- /unit:b9565b50971098b7dd -->

<!-- unit:d54639003c94a5f97a -->
<a id="u-d54639003c94a5f97a"></a>

### Запрос недостающей информации как action

<!-- /unit:d54639003c94a5f97a -->

<!-- unit:b82ccd416096e59a65 -->
<a id="u-b82ccd416096e59a65"></a>

Если нужная version или scope не определены, среда может разрешать action request_clarification. Это переход в ожидание внешнего observation, а не обязательно failure. В других задачах user input недоступен и требуется abstain/partial. Training/eval должны разделять эти contracts.

<!-- /unit:b82ccd416096e59a65 -->

<!-- unit:77a802c5b1e765e346 -->
<a id="u-77a802c5b1e765e346"></a>

Oracle не должен штрафовать за корректное уточнение только потому, что dataset содержит один immediate answer target. Но и бессмысленный вопрос при уже полном evidence не становится идеальным поведением. Важны необходимость, стоимость и влияющий на outcome ответ пользователя. Для текущего редактирования конспекта этот механизм объясняется теоретически, новые вопросы пользователю не требуются.

<!-- /unit:77a802c5b1e765e346 -->

<!-- unit:e6b1d68cccb851db4a -->
<a id="u-e6b1d68cccb851db4a"></a>

## Trajectory data и split

<!-- /unit:e6b1d68cccb851db4a -->

<!-- unit:2f4416b82d043d6220 -->
<a id="u-2f4416b82d043d6220"></a>

Запись содержит initial task/context, chronological actions, exact observations, versions, errors, rewards/termination и total cost. Tool result хранится отдельно от policy-generated tokens. Пользовательский final goal нельзя заменить более простой подзадачей после неуспеха.

<!-- /unit:2f4416b82d043d6220 -->

<!-- unit:16718ba986f5ae7da4 -->
<a id="u-16718ba986f5ae7da4"></a>

Разделять train/test по задачам, source documents и templates, где нужно. Random split отдельных messages одной trajectory позволяет model видеть продолжение своей будущей test history. Если изучается recovery, holdout должен включать новые tasks/conditions с ошибками, а не те же logs с другой перестановкой строк.

<!-- /unit:16718ba986f5ae7da4 -->

<!-- unit:411eca7132c9cf6284 -->
<a id="u-411eca7132c9cf6284"></a>

### Что сохранять на границе environment и policy

<!-- /unit:411eca7132c9cf6284 -->

<!-- unit:a37779309be52893f3 -->
<a id="u-a37779309be52893f3"></a>

Task ID, initial snapshot и versions связывают episode с постановкой. Для каждого шага сохраняют model-visible history/input, proposal, validated action, actual result/error, usage, reward и termination reason. Reward log не должен подменять actual observation; hidden evaluator labels не должны случайно попасть в policy input до принятия решения.

<!-- /unit:a37779309be52893f3 -->

<!-- unit:bda317bb163423327e -->
<a id="u-bda317bb163423327e"></a>

Off-policy recorded trace описывает конкретно выполненные actions, а не любые желаемые alternatives. Нельзя взять reward успешного R2 чтения и присвоить его action read(R1) только потому, что final prose похож. Для supervised data можно отдельно разметить экспертное исправление на данной history; это уже новый label с объявленным происхождением, а не переписывание фактического прошлого.

<!-- /unit:bda317bb163423327e -->

<!-- unit:0ebff45abb37fa84ac -->
<a id="u-0ebff45abb37fa84ac"></a>

Различают environment simulation и recorded external execution. В маленькой toy среде результаты специально заданы, и это достаточно для ручного изучения transitions/returns. Ни один такой расчёт не объявляется доказательством обучения пользовательской LLM или успешного взаимодействия с реальной организационной базой.

<!-- /unit:0ebff45abb37fa84ac -->

<!-- unit:dd107b13d012f5cd32 -->
<a id="u-dd107b13d012f5cd32"></a>

### Самопроверка

<!-- /unit:dd107b13d012f5cd32 -->

<!-- unit:c0417883057f41aeff -->
<a id="u-c0417883057f41aeff"></a>

1. Почему observation не равно state? **Ответ:** agent видит только доступную часть мира и может ошибаться о остальном.
2. Что значит reward delayed? **Ответ:** quality ранних actions определяется поздним outcome, а не немедленным label каждого шага.
3. Чем truncation отличается от success? **Ответ:** ограничение остановило episode, но задача могла остаться нерешённой.

<!-- /unit:c0417883057f41aeff -->

<!-- unit:cd3e3ce0dbd2bb7d93 -->
<a id="u-cd3e3ce0dbd2bb7d93"></a>

4. Hidden state среды —activations Transformer? **Ответ:** нет, это разные понятия.
5. Может ли read-only search менять planning state? **Ответ:** да, меняются evidence/belief и budget, хотя corpus остаётся прежним.
6. Всегда ли конец horizon есть truncation? **Ответ:** нет, встроенный task horizon может быть terminal condition; external collector cap —другой случай.
7. Значит ли высокий reward высокий success? **Ответ:** только при достаточно корректном signal; proxy может иметь обходы.
8. Почему step cost0.6 опасна в toy3-action task? **Ответ:** success return−0.8 хуже immediate abstain−0.6 при выбранной reward scale.
9. Что делать с result, не полученным из-за timeout? **Ответ:** сохранить неизвестность/ошибку; не заменять вымышленным evidence или no-results.
10. Гарантирует ли одинаковый seed одинаковую trace разных policies? **Ответ:** нет, policies могут запрашивать разные actions и потреблять random choices иначе.

<!-- /unit:cd3e3ce0dbd2bb7d93 -->

<!-- unit:10a15c4dfd09cf2a51 -->
<a id="u-10a15c4dfd09cf2a51"></a>

**Практика:** записать три toy trajectories: success через current v2, failure через stale v1 и recovery после search timeout. Для каждой указать observations, действия, return, termination и какой feedback доступен для обучения. Начать с этих записей, не с большого RL запуска.

<!-- /unit:10a15c4dfd09cf2a51 -->

## Результат главы и переход дальше

Ответьте на вопрос: **Как описать доступную агенту информацию и последствия действия?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

Следующие темы: [Оценка траекторий и воспроизводимость](<agent-evaluation.md>).

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
