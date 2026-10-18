# И. Несколько агентов: когда координация приносит пользу

[Оглавление](README.md). Пункт 9 плана. **Основа:** S6:1377–2197 — frameworks, роли и вопрос о внутреннем устройстве участников; это материал об orchestration. Role-specific adaptation, self-play и одновременное обучение политик ниже — новое обзорное углубление, не выполненные эксперименты.

## 1. Роль, модель и обучаемая политика

Planner формулирует подзадачи, executor действует, critic проверяет результат. Эти роли могут исполняться одной и той же frozen LLM с разными инструкциями. Тогда система multi-agent по организации работы, но веса не обучались совместно. Разные checkpoints или adapters создают другую конфигурацию; общая архитектура LLM не делает роли автоматически независимыми.

Framework управляет сообщениями, состоянием и переходами. Участник одного framework не обязан быть внутренним графом другого. Как и в чате, следует различать прикладной протокол взаимодействия и конкретный runtime, который его реализует. Для понимания достаточно описать вход каждой роли, возможные actions и ответственность за stop condition.

Если агенты имеют политики $\pi_{\theta_1},\ldots,\pi_{\theta_K}$, совместное поведение определяется ещё и правилами передачи сообщений. При shared backbone и role-specific adapters меняется только часть параметров; prompts, adapters и routing надо версионировать вместе. Само добавление роли critic не гарантирует независимой проверки истины.


<!-- source-depth-lifecycle-И -->
### 1.1. Три уровня, которые нельзя смешивать

Уровень роли задаёт задачу, context и tools: researcher ищет source, executor читает, critic проверяет claim. Уровень runtime определяет, кто получает следующее сообщение, как хранится state и когда задача завершается. Уровень обучения меняет параметры policy по данным/reward. Можно изменить первые два при неизменном checkpoint; это уже другая система поведения, но не joint training.

В S6 пользователь спрашивал, реализован ли каждый AutoGen participant через LangGraph и как researcher понимает, что пора искать или завершаться. Ответ относится прежде всего к локальной decision policy и coordinator. Participant не обязан быть LLM с собственным loop: это может быть deterministic tool executor. Framework name не задаёт точную внутреннюю реализацию любого участника. Подробный source-разбор уже находится в [orchestration](<../../04-applications/Учебные конспекты/В — State, orchestration, графы и участники агента.md>).

Здесь новое углубление — что произойдёт, если роли действительно обучаются. Наличие researcher/writer/critic labels в prompt не доказывает ни специализацию weights, ни independent expertise. Три instances одного frozen checkpoint могут видеть разные sources, а три разных checkpoints — повторять одну общую ошибку из одинакового неверного evidence.

### 1.2. Общая модель, adapters или отдельные policies

Shared policy можно записать как πθ(a|h,role): роль — часть conditioning, а parameters общие. Role-specific adapters меняют выбранную часть параметров под задачу. Отдельные checkpoints имеют отдельные θ. Эти варианты различаются capacity, interference, storage и training/inference cost; число role names само по себе не определяет число моделей в памяти.

Если common θ обновляется на examples всех ролей, их datasets/sampling weights влияют на общий objective. Многословные writer targets могут доминировать над короткими router/tool decisions при token-average loss. Если adapters раздельные, нужна совместимость message contracts: writer должен понимать references researcher, независимо от разных weights.

Training data каждой роли содержит доступную ей history, action и проверенный результат. Не следует обучать critic выдавать «подтверждаю», если label получен только из уверенного final текста executor. Для роль-specific SFT нужны соответствующие targets; для RL —signal и rollout protocol. Подключение adapter не превращает роль автоматически в надёжный specialist.

### 1.3. Joint action и factorization assumptions

Для одного joint decision при условно независимом sampling участников можно записать $\pi(\mathbf a\mid\mathbf h)=\prod_i\pi_i(a_i\mid h_i)$. Это assumption о policy factorization, а не утверждение, что errors независимы. Histories могут содержать общий неверный source, coordinator routing или shared randomness.

Если central planner выбирает согласованную комбинацию actions, joint distribution может иметь другую structure. Если агенты действуют по очереди, subsequent history включает прежние messages и factorization идёт по этому порядку. Нельзя перемножить marginal voting accuracies и получить quality team workflow без анализа dependencies.


## 2. Откуда возможен выигрыш

Разделение труда полезно, когда задачи естественно декомпозируются, специалисты видят разный релевантный context или проверка дешевле решения. Например, executor находит документ, а critic сверяет дату, цитату и исключения по самому документу. Critic должен иметь evidence, иначе он оценивает только убедительность сообщения executor.

Parallel работа может уменьшить wall-clock latency для независимых подзадач, но общее число tokens/calls обычно растёт. Если второй агент повторяет весь первый путь, выигрыша от декомпозиции может не быть. Когда действия зависимы, нельзя запускать downstream step до получения нужного observation.

Независимость ошибок нельзя предполагать по числу участников. При независимых бинарных решениях с одинаковой accuracy $p$ majority из трёх верна с вероятностью:

$$
P_{\mathrm{maj}}=p^3+3p^2(1-p).
$$

При $p=0.7$ это $0.784$. Но если три агента всегда совершают одну и ту же ошибку, их majority имеет ту же accuracy $0.7$. Общие training data, модель и неправильное evidence создают корреляцию. В реальном эксперименте нужно измерять совпадение ошибок, а не подставлять независимую формулу как обещание улучшения.


### 2.1. Декомпозиция должна давать проверяемые outputs

Для read-only задачи researcher возвращает source IDs/revisions, passages и unresolved questions. Critic проверяет applicability/currentness/citations по этим источникам. Writer формулирует answer из verified claims. Такой contract позволяет понять, на какой границе пропал факт. Если researcher возвращает только «всё проверено», critic вынужден судить по впечатлению.

Передача результата включает task scope, status, evidence references, remaining budget и pending work. Message от роли — данные/предложение, а не автоматическое расширение прав другой роли. Coordinator сверяет завершение общей задачи; локальное «моя часть готова» не означает, что выполнены все required aspects.

Если нужна одна exact lookup operation, дополнительные роли могут лишь повторить работу. Если независимым branches нужны разные subsets sources, параллельность может сократить critical path. Пользу decomposed workflow проверяют по evidence coverage и outcome, а не по сходству с человеческой командой.

### 2.2. Majority: один p не определяет совместные ошибки

Основная formula0.784 для p0.7 предполагает independent binary votes и определённую задачу с одним correct label. В реальном длинном answer majority отдельных слов или произвольных claims не обязана давать логически согласованный итог. Для multi-part answer может понадобиться evidence-based synthesis, а не голосование по одному победителю.

Возьмём toy mixture: с probability0.2 общий source заставляет всех ошибиться; в остальных 0.8 cases каждый voter независимо correct с probability0.875. Individual accuracy0.8·0.875=0.7, но majority accuracy $0.8[0.875^3+3(0.875)^2(0.125)]=0.765625$. При полностью совпадающих errors она 0.7. Одинаковая marginal accuracy трёх участников совместима с разными team outcomes.

Нужны measurements совпадений ошибок, diversity evidence и полезности corrections на held-out tasks. Разные random seeds или stylistic prompts не гарантируют независимость. Даже согласие разных моделей по одному устаревшему документу не превращает его в current source.

### 2.3. Проверка как action, а не статус роли

Critic полезен, когда делает проверяемую операцию: читает actual passage, запускает applicable test или сверяет final state. Название critic не даёт дополнительных фактов. При верном source и неверной интерпретации можно найти конкретное противоречие; при отсутствующем evidence следует сохранить unknown, а не придумывать опровержение.

Считать нужно repaired failures и spoiled successes, accepted/rejected objections и цену. В исходном примере actor80/100, critic исправляет 10 и портит 5:85/100. Это+5points, но single-agent с тем же compute может решить 88. В таком setup выигрыша именно от этой координации не показано, хотя critic относительно короткого baseline помог.


## 3. Debate и self-play

Debate может выявить возражения, если участников обязывают предъявлять проверяемые evidence и есть надёжное правило разрешения спора. Без этого дискуссия способна лишь усилить уверенность в общей ошибке. Согласие нескольких копий модели не превращает выдуманную ссылку в существующую.

Self-play создаёт задачи или противодействие через взаимодействие политик. В соревновательной игре reward одного участника может быть отрицанием reward другого; в совместной задаче чаще общий reward. В смешанной среде интересы различаются. Условия нужно описать явно, прежде чем выбирать алгоритм обучения.

Если одновременно меняются несколько policies, среда с точки зрения отдельного ученика становится нестационарной: partner меняет поведение между итерациями. Старые rollouts могут хуже отражать текущую команду. Нужны контролируемые checkpoints, оценка с замороженными партнёрами и проверка переноса на других участников. Emergent communication — исследовательский сюжет; его наличие не следует из обычного обмена natural-language messages.


### 3.1. Cooperative, competitive и mixed objectives

В cooperative task участники могут получать общий terminal reward за verified team success и ресурсы. В zero-sum game цели противоположны. В mixed setting интересы/constraints не сводятся к одному shared scalar. Тот факт, что два LLM participants спорят словами, не делает задачу zero-sum RL game.

Self-play означает, что data/opponents/partners создаются взаимодействующими policies по определённому protocol. Модель, генерирующая один synthetic prompt для себя, ещё не является доказанным self-play training алгоритмом. Нужны players, действия, transition/reward, update schedule и независимая evaluation transfer.

Debate — protocol обмена аргументами, который может использовать frozen models. Он способен предъявить новое evidence или усилить общую ошибку. Training на debate transcripts меняет отдельный objective/data; наличие самой дискуссии не доказывает gradient update или корректность вывода.

### 3.2. Почему меняющийся partner меняет среду ученика

Для actor A условный transition зависит от действий B. Упрощённо $P_B(s'\mid s,a_A)=\sum_{a_B}\pi_B(a_B\mid h_B)P(s'\mid s,a_A,a_B)$. Если B обновился, induced kernel для A изменился, даже при том же external world. Старые experiences A относятся к прежней partner policy, что осложняет interpretation/reuse.

Например, B раньше возвращал exact source references, а теперь summary без IDs. A, обученный слепо доверять прежнему contract, может завершаться с unsupported citations. Это и behavioral shift, и возможная interface regression; смена partner не должна молча переопределять message schema. Version bundle команды включает policies и протоколы.

В [multi-agent actor–critic work](https://arxiv.org/abs/1706.02275) рассматриваются non-stationarity и координация при нескольких policies. [MAPPO study](https://arxiv.org/abs/2103.01955) исследует PPO в cooperative settings. Эти исследования дают algorithmic контекст, но их game benchmarks не доказывают качество любой LLM-team на tools/RAG.

### 3.3. Centralized training и decentralized execution

Во время training critic может видеть global simulator state и actions нескольких участников для оценки return. Execution actors используют разрешённые local histories/messages. Так можно дать richer learning signal, не выдавая каждому actor недоступные facts на inference. Но protocol должен обеспечить, что trained actor реально исполним без privileged critic inputs.

Centralized critic — не тот же компонент, что LLM «критик» в пользовательском workflow. Первый оценивает value/advantage для optimizer; второй может выдавать текстовые objections как action. Они могут сосуществовать, но названия не делают их одинаковыми и не заменяют objective.

При common team reward credit assignment остаётся трудным: успех может усилить полезный поиск A и лишнее сообщение B. Простое раздать один scalar всем tokens не доказывает их causal contribution. Для independent learner, shared parameters и central critic нужны разные implementation/estimator choices; здесь даётся обзор границ, без обещания universal algorithm.

### 3.4. Проверка с замороженными партнёрами и cross-play

Self-play partners могут выучить общий convention, который не переносится. Условная матрица success: A1+B1=0.90,A1+B2=0.40,A2+B1=0.50,A2+B2=0.92. Среднее согласованных пар 0.91, cross pairs0.45. Это заданный toy example: высокая оценка привычных teams не доказывает robustness к другим partners.

Оценивают checkpoints с frozen partners, cross-play, новыми task families и изменёнными допустимыми conditions. Natural-language коммуникация не доказывает emergent protocol discovery; исследовательское claim требует отдельного evidence. И наоборот, одинаковые weights могут давать полезную роль-специализацию по context без какого-либо emergent learning.


## 4. Бюджет и полезность критика

Для двух ролей полная стоимость включает planner/executor/critic, передачи контекста, повторные генерации и tool calls:

$$
C_{\mathrm{team}}=\sum_{i=1}^K C_i+C_{\mathrm{communication}}+C_{\mathrm{tools}}.
$$

Сравнение должно дать одиночному агенту тот же общий budget, включая возможность самопроверки. Иначе эффект дополнительных вычислений ошибочно приписывается координации.

Разобранный пример: executor правильно решает 80 из 100 задач. Critic исправляет 10 из 20 ошибок, но портит 5 из 80 правильных решений. Итог — 85 успехов. Это полезный прирост в 5 процентных пунктов, однако нужно учесть дополнительные расходы. Если одиночный агент с тем же budget решает 88, выбранная командная схема уступает baseline.

Полезно считать отдельно accepted corrections, верные отклонённые решения, ложные возражения, циклы согласования и финальный success. Ограничение числа раундов предотвращает бесконечную дискуссию, но может обрезать полезный поиск; это часть operating point системы.


### 4.1. Общий budget и коммуникационные затраты

Если message500tokens передан четырём recipients и каждый читает его в своём call, это 2000input tokens по этим четырём inputs, а не 500 для всей системы. Как оплачивается/переиспользуется prefix compute, определяет backend; логический context traffic всё равно считается по получателям. Широкий group chat может тратить больше, чем targeted handoff нужного passage.

У shared cap6000tokens coordinator может зарезервировать 2500executor,2000critic,1500final synthesis. Nested participants не получают каждый ещё по 6000. Unused reservation и actual usage различаются; scheduler перераспределяет budget по явной policy. Считать только output последнего writer означает потерять большую часть работы.

Parallel branches требуют capacity и независимости. Если writer ждёт source researcher, начинать final answer до его результата —иной speculative protocol с возможными wasted outputs, а не бесплатное ускорение. Для common state нужен reducer/conflict policy; arrival order не должен терять evidence или произвольно выбирать текущую revision.

### 4.2. Контролируемая ablation координации

Сравнивают single-agent с достаточным budget/self-check, frozen role workflow, shared role-conditioned SFT и другие реально выполненные interventions по отдельности. Dataset/tool access, total calls/tokens и oracle согласуют. Role count не считается метрикой качества; measured improvement должен проявляться в held-out task outcomes и приемлемых затратах.

Ошибки можно разложить на локальную policy, lost handoff evidence, incorrect routing, shared misconception, wrong conflict resolution и exhausted global budget. Для training data сохраняют actual actor/version каждого action и доступный ему context. Для final report фиксируют pair/team checkpoints, чтобы old rollout не считался опытом уже другой команды.

Multi-agent learning здесь остаётся обзорным углублением по плану. Пользовательское создание subagents, совместное обучение LLM и deployment не выполнялись. Следующая глава замыкает lifecycle: как оценивать полные trajectories и воспроизводить именно заявленные результаты.


## 5. Контроль и практика

1. Обязательно ли multi-agent означает несколько разных обученных моделей? **Ответ:** нет; роли могут использовать один frozen checkpoint.
2. Почему три согласных агента могут ошибаться вместе? **Ответ:** их ошибки коррелируют через общую модель, context и evidence.
3. Что доказывает преимущество координации? **Ответ:** улучшение на независимых задачах относительно одного агента при сопоставимом общем budget, а не просто относительно одной короткой попытки.

4. Independent policy sampling означает independent errors? **Ответ:** нет, общие observations/premises могут связывать ошибки.
5. Почему central critic не равен текстовому reviewer? **Ответ:** это value/advantage компонент обучения, а reviewer действует в runtime protocol.
6. Зачем фиксировать partner version? **Ответ:** новая policy партнёра меняет induced environment и доступные messages.
7. Что показывает cross-play? **Ответ:** перенос к другим partners/protocol combinations сверх привычной совместной пары.
8. Почему shared6000budget не значит 6000 накаждого? **Ответ:** это суммарное ограничение с accounting всех ролей и synthesis.
9. Доказывает ли debate обучение координации? **Ответ:** нет, дискуссия может идти при frozen weights.
10. Можно ли по source вопросу о subagents считать их уже созданными в этой работе? **Ответ:** нет, учебный вопрос и реальные выполненные действия различаются.

**Практика:** спроектируйте executor–critic для read-only поиска правила. Опишите evidence каждого, protocol возражения и stop condition. Затем вручную разберите десять сессий: полезное исправление, вредное исправление, согласованная ошибка, лишний раунд. Предложите ablation без критика и с самопроверкой одиночного агента.
