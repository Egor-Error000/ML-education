# Углубление: роли и обучение координации

**Уровень:** углубление.

**Предпосылки:** [Оценка траекторий и воспроизводимость](<agent-evaluation.md>)

**Используют:** Выбранная прикладная задача и итоговая проверка результата.

**Учебный вопрос:** При каких условиях несколько участников дают измеримый выигрыш?





## Карта вычисления и границы этапов

![Углубление: роли и обучение координации: входы, преобразования и выходы](<../../оформление/course/agent-coordination.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Reward и targets используются в объявленном режиме обучения. При оценке политика фиксирована; success измеряется по всей попытке, включая повторы и бюджет.

<!-- unit:6b322947e7a69037cf -->
<a id="u-6b322947e7a69037cf"></a>

## Роль, модель и обучаемая политика

<!-- /unit:6b322947e7a69037cf -->

<!-- unit:4067a28bcc36d9ab4c -->
<a id="u-4067a28bcc36d9ab4c"></a>

Planner формулирует подзадачи, executor действует, critic проверяет результат. Эти роли могут исполняться одной и той же frozen LLM с разными инструкциями. Тогда система multi-agent по организации работы, но веса не обучались совместно. Разные checkpoints или adapters создают другую конфигурацию; общая архитектура LLM не делает роли автоматически независимыми.

<!-- /unit:4067a28bcc36d9ab4c -->

<!-- unit:b56ee8a7c67f63c9bf -->
<a id="u-b56ee8a7c67f63c9bf"></a>

Framework управляет сообщениями, состоянием и переходами. Участник одного framework не обязан быть внутренним графом другого. Как и в разбираемом примере, следует различать прикладной протокол взаимодействия и конкретный runtime, который его реализует. Для понимания достаточно описать вход каждой роли, возможные actions и ответственность за stop condition.

<!-- /unit:b56ee8a7c67f63c9bf -->

<!-- unit:ab9b8b54ea915b2161 -->
<a id="u-ab9b8b54ea915b2161"></a>

Если агенты имеют политики $\pi_{\theta_1},\ldots,\pi_{\theta_K}$, совместное поведение определяется ещё и правилами передачи сообщений. При shared backbone и role-specific adapters меняется только часть параметров; prompts, adapters и routing надо версионировать вместе. Само добавление роли critic не гарантирует независимой проверки истины.

<!-- /unit:ab9b8b54ea915b2161 -->

<!-- unit:58232e4528abccd980 -->
<a id="u-58232e4528abccd980"></a>

<!-- source-depth-lifecycle-И -->

<!-- /unit:58232e4528abccd980 -->

<!-- unit:85464faa73d38188a3 -->
<a id="u-85464faa73d38188a3"></a>

### Три уровня, которые нельзя смешивать

<!-- /unit:85464faa73d38188a3 -->

<!-- unit:3d991b0c1fb1b4dc0c -->
<a id="u-3d991b0c1fb1b4dc0c"></a>

Уровень роли задаёт задачу, context и tools: researcher ищет source, executor читает, critic проверяет claim. Уровень runtime определяет, кто получает следующее сообщение, как хранится state и когда задача завершается. Уровень обучения меняет параметры policy по данным/reward. Можно изменить первые два при неизменном checkpoint; это уже другая система поведения, но не joint training.

<!-- /unit:3d991b0c1fb1b4dc0c -->

<!-- unit:891ef886e5ebb2e8f2 -->
<a id="u-891ef886e5ebb2e8f2"></a>

В S6 пользователь спрашивал, реализован ли каждый AutoGen participant через LangGraph и как researcher понимает, что пора искать или завершаться. Ответ относится прежде всего к локальной decision policy и coordinator. Participant не обязан быть LLM с собственным loop: это может быть deterministic tool executor. Framework name не задаёт точную внутреннюю реализацию любого участника. Подробный source-разбор уже находится в [orchestration](<../05-applications/app-state.md>).

<!-- /unit:891ef886e5ebb2e8f2 -->

<!-- unit:21568c18d9205c3408 -->
<a id="u-21568c18d9205c3408"></a>

Здесь новое углубление — что произойдёт, если роли действительно обучаются. Наличие researcher/writer/critic labels в prompt не доказывает ни специализацию weights, ни independent expertise. Три instances одного frozen checkpoint могут видеть разные sources, а три разных checkpoints — повторять одну общую ошибку из одинакового неверного evidence.

<!-- /unit:21568c18d9205c3408 -->

<!-- unit:7f7f205c70d21e5026 -->
<a id="u-7f7f205c70d21e5026"></a>

### Общая модель, adapters или отдельные policies

<!-- /unit:7f7f205c70d21e5026 -->

<!-- unit:aed771c437ee46e244 -->
<a id="u-aed771c437ee46e244"></a>

Shared policy можно записать как πθ(a|h,role): роль — часть conditioning, а parameters общие. Role-specific adapters меняют выбранную часть параметров под задачу. Отдельные checkpoints имеют отдельные θ. Эти варианты различаются capacity, interference, storage и training/inference cost; число role names само по себе не определяет число моделей в памяти.

<!-- /unit:aed771c437ee46e244 -->

<!-- unit:b246dde73ed877ca79 -->
<a id="u-b246dde73ed877ca79"></a>

Если common θ обновляется на examples всех ролей, их datasets/sampling weights влияют на общий objective. Многословные writer targets могут доминировать над короткими router/tool decisions при token-average loss. Если adapters раздельные, нужна совместимость message contracts: writer должен понимать references researcher, независимо от разных weights.

<!-- /unit:b246dde73ed877ca79 -->

<!-- unit:b035268fab389fa598 -->
<a id="u-b035268fab389fa598"></a>

Training data каждой роли содержит доступную ей history, action и проверенный результат. Не следует обучать critic выдавать «подтверждаю», если label получен только из уверенного final текста executor. Для роль-specific SFT нужны соответствующие targets; для RL —signal и rollout protocol. Подключение adapter не превращает роль автоматически в надёжный specialist.

<!-- /unit:b035268fab389fa598 -->

<!-- unit:a6c9de4a8378a17795 -->
<a id="u-a6c9de4a8378a17795"></a>

### Joint action и factorization assumptions

<!-- /unit:a6c9de4a8378a17795 -->

<!-- unit:a34412058c2d2c7e18 -->
<a id="u-a34412058c2d2c7e18"></a>

Для одного joint decision при условно независимом sampling участников можно записать $\pi(\mathbf a\mid\mathbf h)=\prod_i\pi_i(a_i\mid h_i)$. Это assumption о policy factorization, а не утверждение, что errors независимы. Histories могут содержать общий неверный source, coordinator routing или shared randomness.

<!-- /unit:a34412058c2d2c7e18 -->

<!-- unit:cb9fbd547c1ec111b3 -->
<a id="u-cb9fbd547c1ec111b3"></a>

Если central planner выбирает согласованную комбинацию actions, joint distribution может иметь другую structure. Если агенты действуют по очереди, subsequent history включает прежние messages и factorization идёт по этому порядку. Нельзя перемножить marginal voting accuracies и получить quality team workflow без анализа dependencies.

<!-- /unit:cb9fbd547c1ec111b3 -->

<!-- unit:777b55222d9804398b -->
<a id="u-777b55222d9804398b"></a>

## Откуда возможен выигрыш

<!-- /unit:777b55222d9804398b -->

<!-- unit:30fdadc0059685af5e -->
<a id="u-30fdadc0059685af5e"></a>

Разделение труда полезно, когда задачи естественно декомпозируются, специалисты видят разный релевантный context или проверка дешевле решения. Например, executor находит документ, а critic сверяет дату, цитату и исключения по самому документу. Critic должен иметь evidence, иначе он оценивает только убедительность сообщения executor.

<!-- /unit:30fdadc0059685af5e -->

<!-- unit:18c8027f4d97492d7c -->
<a id="u-18c8027f4d97492d7c"></a>

Parallel работа может уменьшить wall-clock latency для независимых подзадач, но общее число tokens/calls обычно растёт. Если второй агент повторяет весь первый путь, выигрыша от декомпозиции может не быть. Когда действия зависимы, нельзя запускать downstream step до получения нужного observation.

<!-- /unit:18c8027f4d97492d7c -->

<!-- unit:e44976f3d3b27301b3 -->
<a id="u-e44976f3d3b27301b3"></a>

Независимость ошибок нельзя предполагать по числу участников. При независимых бинарных решениях с одинаковой accuracy $p$ majority из трёх верна с вероятностью:

<!-- /unit:e44976f3d3b27301b3 -->

<!-- unit:dcb38398d9438dbbde -->
<a id="u-dcb38398d9438dbbde"></a>

$$
P_{\mathrm{maj}}=p^3+3p^2(1-p).
$$

<!-- /unit:dcb38398d9438dbbde -->

<!-- unit:c61a5b048e4c9bd9e4 -->
<a id="u-c61a5b048e4c9bd9e4"></a>

При $p=0.7$ это $0.784$. Но если три агента всегда совершают одну и ту же ошибку, их majority имеет ту же accuracy $0.7$. Общие training data, модель и неправильное evidence создают корреляцию. В реальном эксперименте нужно измерять совпадение ошибок, а не подставлять независимую формулу как обещание улучшения.

<!-- /unit:c61a5b048e4c9bd9e4 -->

<!-- unit:94196d3dcc38ff50df -->
<a id="u-94196d3dcc38ff50df"></a>

### Декомпозиция должна давать проверяемые outputs

<!-- /unit:94196d3dcc38ff50df -->

<!-- unit:77d4931d9b188d6265 -->
<a id="u-77d4931d9b188d6265"></a>

Для read-only задачи researcher возвращает source IDs/revisions, passages и unresolved questions. Critic проверяет applicability/currentness/citations по этим источникам. Writer формулирует answer из verified claims. Такой contract позволяет понять, на какой границе пропал факт. Если researcher возвращает только «всё проверено», critic вынужден судить по впечатлению.

<!-- /unit:77d4931d9b188d6265 -->

<!-- unit:a1e18ef9276d8bd047 -->
<a id="u-a1e18ef9276d8bd047"></a>

Передача результата включает task scope, status, evidence references, remaining budget и pending work. Message от роли — данные/предложение, а не автоматическое расширение прав другой роли. Coordinator сверяет завершение общей задачи; локальное «моя часть готова» не означает, что выполнены все required aspects.

<!-- /unit:a1e18ef9276d8bd047 -->

<!-- unit:f0c773928c4faea59e -->
<a id="u-f0c773928c4faea59e"></a>

Если нужна одна exact lookup operation, дополнительные роли могут лишь повторить работу. Если независимым branches нужны разные subsets sources, параллельность может сократить critical path. Пользу decomposed workflow проверяют по evidence coverage и outcome, а не по сходству с человеческой командой.

<!-- /unit:f0c773928c4faea59e -->

<!-- unit:3f408f1724540a0a31 -->
<a id="u-3f408f1724540a0a31"></a>

### Majority: один p не определяет совместные ошибки

<!-- /unit:3f408f1724540a0a31 -->

<!-- unit:555ccf5fdbd41057ca -->
<a id="u-555ccf5fdbd41057ca"></a>

Основная formula0.784 для p0.7 предполагает independent binary votes и определённую задачу с одним correct label. В реальном длинном answer majority отдельных слов или произвольных claims не обязана давать логически согласованный итог. Для multi-part answer может понадобиться evidence-based synthesis, а не голосование по одному победителю.

<!-- /unit:555ccf5fdbd41057ca -->

<!-- unit:8036e709a2eacf5ba7 -->
<a id="u-8036e709a2eacf5ba7"></a>

Возьмём toy mixture: с probability0.2 общий source заставляет всех ошибиться; в остальных 0.8 cases каждый voter независимо correct с probability0.875. Individual accuracy0.8·0.875=0.7, но majority accuracy $0.8[0.875^3+3(0.875)^2(0.125)]=0.765625$. При полностью совпадающих errors она 0.7. Одинаковая marginal accuracy трёх участников совместима с разными team outcomes.

<!-- /unit:8036e709a2eacf5ba7 -->

<!-- unit:e14e7f65d665dc7461 -->
<a id="u-e14e7f65d665dc7461"></a>

Нужны measurements совпадений ошибок, diversity evidence и полезности corrections на held-out tasks. Разные random seeds или stylistic prompts не гарантируют независимость. Даже согласие разных моделей по одному устаревшему документу не превращает его в current source.

<!-- /unit:e14e7f65d665dc7461 -->

<!-- unit:c402e1a4d457285b3c -->
<a id="u-c402e1a4d457285b3c"></a>

### Проверка как action, а не статус роли

<!-- /unit:c402e1a4d457285b3c -->

<!-- unit:db32cc41fe1b63b2df -->
<a id="u-db32cc41fe1b63b2df"></a>

Critic полезен, когда делает проверяемую операцию: читает actual passage, запускает applicable test или сверяет final state. Название critic не даёт дополнительных фактов. При верном source и неверной интерпретации можно найти конкретное противоречие; при отсутствующем evidence следует сохранить unknown, а не придумывать опровержение.

<!-- /unit:db32cc41fe1b63b2df -->

<!-- unit:2f3d9a08b093e45608 -->
<a id="u-2f3d9a08b093e45608"></a>

Считать нужно repaired failures и spoiled successes, accepted/rejected objections и цену. В исходном примере actor80/100, critic исправляет 10 и портит 5:85/100. Это+5points, но single-agent с тем же compute может решить 88. В таком setup выигрыша именно от этой координации не показано, хотя critic относительно короткого baseline помог.

<!-- /unit:2f3d9a08b093e45608 -->

<!-- unit:75e96bb97dc31f2396 -->
<a id="u-75e96bb97dc31f2396"></a>

## Debate и self-play

<!-- /unit:75e96bb97dc31f2396 -->

<!-- unit:1dc5066a67c5b1e62a -->
<a id="u-1dc5066a67c5b1e62a"></a>

Debate может выявить возражения, если участников обязывают предъявлять проверяемые evidence и есть надёжное правило разрешения спора. Без этого дискуссия способна лишь усилить уверенность в общей ошибке. Согласие нескольких копий модели не превращает выдуманную ссылку в существующую.

<!-- /unit:1dc5066a67c5b1e62a -->

<!-- unit:ab52a54aa277cc69bd -->
<a id="u-ab52a54aa277cc69bd"></a>

Self-play создаёт задачи или противодействие через взаимодействие политик. В соревновательной игре reward одного участника может быть отрицанием reward другого; в совместной задаче чаще общий reward. В смешанной среде интересы различаются. Условия нужно описать явно, прежде чем выбирать алгоритм обучения.

<!-- /unit:ab52a54aa277cc69bd -->

<!-- unit:2a02628ecc1d22f362 -->
<a id="u-2a02628ecc1d22f362"></a>

Если одновременно меняются несколько policies, среда с точки зрения отдельного ученика становится нестационарной: partner меняет поведение между итерациями. Старые rollouts могут хуже отражать текущую команду. Нужны контролируемые checkpoints, оценка с замороженными партнёрами и проверка переноса на других участников. Emergent communication — исследовательский сюжет; его наличие не следует из обычного обмена natural-language messages.

<!-- /unit:2a02628ecc1d22f362 -->

<!-- unit:9b085b8bdecf0eea52 -->
<a id="u-9b085b8bdecf0eea52"></a>

### Cooperative, competitive и mixed objectives

<!-- /unit:9b085b8bdecf0eea52 -->

<!-- unit:9aba8ac087125f9a2d -->
<a id="u-9aba8ac087125f9a2d"></a>

В cooperative task участники могут получать общий terminal reward за verified team success и ресурсы. В zero-sum game цели противоположны. В mixed setting интересы/constraints не сводятся к одному shared scalar. Тот факт, что два LLM participants спорят словами, не делает задачу zero-sum RL game.

<!-- /unit:9aba8ac087125f9a2d -->

<!-- unit:abe36d90191b1aa600 -->
<a id="u-abe36d90191b1aa600"></a>

Self-play означает, что data/opponents/partners создаются взаимодействующими policies по определённому protocol. Модель, генерирующая один synthetic prompt для себя, ещё не является доказанным self-play training алгоритмом. Нужны players, действия, transition/reward, update schedule и независимая evaluation transfer.

<!-- /unit:abe36d90191b1aa600 -->

<!-- unit:6cc85676ec316cd826 -->
<a id="u-6cc85676ec316cd826"></a>

Debate — protocol обмена аргументами, который может использовать frozen models. Он способен предъявить новое evidence или усилить общую ошибку. Training на debate transcripts меняет отдельный objective/data; наличие самой дискуссии не доказывает gradient update или корректность вывода.

<!-- /unit:6cc85676ec316cd826 -->

<!-- unit:414918d69dcfa8784a -->
<a id="u-414918d69dcfa8784a"></a>

### Почему меняющийся partner меняет среду ученика

<!-- /unit:414918d69dcfa8784a -->

<!-- unit:320e7a86593ad49a0d -->
<a id="u-320e7a86593ad49a0d"></a>

Для actor A условный transition зависит от действий B. Упрощённо $P_B(s'\mid s,a_A)=\sum_{a_B}\pi_B(a_B\mid h_B)P(s'\mid s,a_A,a_B)$. Если B обновился, induced kernel для A изменился, даже при том же external world. Старые experiences A относятся к прежней partner policy, что осложняет interpretation/reuse.

<!-- /unit:320e7a86593ad49a0d -->

<!-- unit:db10ffb5b3439de280 -->
<a id="u-db10ffb5b3439de280"></a>

Например, B раньше возвращал exact source references, а теперь summary без IDs. A, обученный слепо доверять прежнему contract, может завершаться с unsupported citations. Это и behavioral shift, и возможная interface regression; смена partner не должна молча переопределять message schema. Version bundle команды включает policies и протоколы.

<!-- /unit:db10ffb5b3439de280 -->

<!-- unit:748b63f3d4700d8393 -->
<a id="u-748b63f3d4700d8393"></a>

В [multi-agent actor–critic work](https://arxiv.org/abs/1706.02275) рассматриваются non-stationarity и координация при нескольких policies. [MAPPO study](https://arxiv.org/abs/2103.01955) исследует PPO в cooperative settings. Эти исследования дают algorithmic контекст, но их game benchmarks не доказывают качество любой LLM-team на tools/RAG.

<!-- /unit:748b63f3d4700d8393 -->

<!-- unit:824c9333fe67951447 -->
<a id="u-824c9333fe67951447"></a>

### Centralized training и decentralized execution

<!-- /unit:824c9333fe67951447 -->

<!-- unit:792df3c5f06ab63ef9 -->
<a id="u-792df3c5f06ab63ef9"></a>

Во время training critic может видеть global simulator state и actions нескольких участников для оценки return. Execution actors используют разрешённые local histories/messages. Так можно дать richer learning signal, не выдавая каждому actor недоступные facts на inference. Но protocol должен обеспечить, что trained actor реально исполним без privileged critic inputs.

<!-- /unit:792df3c5f06ab63ef9 -->

<!-- unit:bffaafc8af52ff61e5 -->
<a id="u-bffaafc8af52ff61e5"></a>

Centralized critic — не тот же компонент, что LLM «критик» в пользовательском workflow. Первый оценивает value/advantage для optimizer; второй может выдавать текстовые objections как action. Они могут сосуществовать, но названия не делают их одинаковыми и не заменяют objective.

<!-- /unit:bffaafc8af52ff61e5 -->

<!-- unit:e7c4291e248ada2df0 -->
<a id="u-e7c4291e248ada2df0"></a>

При common team reward credit assignment остаётся трудным: успех может усилить полезный поиск A и лишнее сообщение B. Простое раздать один scalar всем tokens не доказывает их causal contribution. Для independent learner, shared parameters и central critic нужны разные implementation/estimator choices; здесь даётся обзор границ, без обещания universal algorithm.

<!-- /unit:e7c4291e248ada2df0 -->

<!-- unit:1f4f0ad84d40f50754 -->
<a id="u-1f4f0ad84d40f50754"></a>

### Проверка с замороженными партнёрами и cross-play

<!-- /unit:1f4f0ad84d40f50754 -->

<!-- unit:a9e53b9cc05e3bd354 -->
<a id="u-a9e53b9cc05e3bd354"></a>

Self-play partners могут выучить общий convention, который не переносится. Условная матрица success: A1+B1=0.90,A1+B2=0.40,A2+B1=0.50,A2+B2=0.92. Среднее согласованных пар 0.91, cross pairs0.45. Это заданный toy example: высокая оценка привычных teams не доказывает robustness к другим partners.

<!-- /unit:a9e53b9cc05e3bd354 -->

<!-- unit:600b0d15e600291580 -->
<a id="u-600b0d15e600291580"></a>

Оценивают checkpoints с frozen partners, cross-play, новыми task families и изменёнными допустимыми conditions. Natural-language коммуникация не доказывает emergent protocol discovery; исследовательское claim требует отдельного evidence. И наоборот, одинаковые weights могут давать полезную роль-специализацию по context без какого-либо emergent learning.

<!-- /unit:600b0d15e600291580 -->

<!-- unit:6776b7065279d69b95 -->
<a id="u-6776b7065279d69b95"></a>

## Бюджет и полезность критика

<!-- /unit:6776b7065279d69b95 -->

<!-- unit:649b103c1a865b7782 -->
<a id="u-649b103c1a865b7782"></a>

Для двух ролей полная стоимость включает planner/executor/critic, передачи контекста, повторные генерации и tool calls:

<!-- /unit:649b103c1a865b7782 -->

<!-- unit:d55b36b2ff0c4ff5ec -->
<a id="u-d55b36b2ff0c4ff5ec"></a>

$$
C_{\mathrm{team}}=\sum_{i=1}^K C_i+C_{\mathrm{communication}}+C_{\mathrm{tools}}.
$$

<!-- /unit:d55b36b2ff0c4ff5ec -->

<!-- unit:8026c751ace2fa25a9 -->
<a id="u-8026c751ace2fa25a9"></a>

Сравнение должно дать одиночному агенту тот же общий budget, включая возможность самопроверки. Иначе эффект дополнительных вычислений ошибочно приписывается координации.

<!-- /unit:8026c751ace2fa25a9 -->

<!-- unit:df07574963eac8af6b -->
<a id="u-df07574963eac8af6b"></a>

Разобранный пример: executor правильно решает 80 из 100 задач. Critic исправляет 10 из 20 ошибок, но портит 5 из 80 правильных решений. Итог — 85 успехов. Это полезный прирост в 5 процентных пунктов, однако нужно учесть дополнительные расходы. Если одиночный агент с тем же budget решает 88, выбранная командная схема уступает baseline.

<!-- /unit:df07574963eac8af6b -->

<!-- unit:5ed979e0e887524b34 -->
<a id="u-5ed979e0e887524b34"></a>

Полезно считать отдельно accepted corrections, верные отклонённые решения, ложные возражения, циклы согласования и финальный success. Ограничение числа раундов предотвращает бесконечную дискуссию, но может обрезать полезный поиск; это часть operating point системы.

<!-- /unit:5ed979e0e887524b34 -->

<!-- unit:b078c70cbf5c016ecd -->
<a id="u-b078c70cbf5c016ecd"></a>

### Общий budget и коммуникационные затраты

<!-- /unit:b078c70cbf5c016ecd -->

<!-- unit:457e204e8f12b24e94 -->
<a id="u-457e204e8f12b24e94"></a>

Если message500tokens передан четырём recipients и каждый читает его в своём call, это 2000input tokens по этим четырём inputs, а не 500 для всей системы. Как оплачивается/переиспользуется prefix compute, определяет backend; логический context traffic всё равно считается по получателям. Широкий group chat может тратить больше, чем targeted handoff нужного passage.

<!-- /unit:457e204e8f12b24e94 -->

<!-- unit:8931d7c9c1028e86bd -->
<a id="u-8931d7c9c1028e86bd"></a>

У shared cap6000tokens coordinator может зарезервировать 2500executor,2000critic,1500final synthesis. Nested participants не получают каждый ещё по 6000. Unused reservation и actual usage различаются; scheduler перераспределяет budget по явной policy. Считать только output последнего writer означает потерять большую часть работы.

<!-- /unit:8931d7c9c1028e86bd -->

<!-- unit:4ef150efbd00799848 -->
<a id="u-4ef150efbd00799848"></a>

Parallel branches требуют capacity и независимости. Если writer ждёт source researcher, начинать final answer до его результата —иной speculative protocol с возможными wasted outputs, а не бесплатное ускорение. Для common state нужен reducer/conflict policy; arrival order не должен терять evidence или произвольно выбирать текущую revision.

<!-- /unit:4ef150efbd00799848 -->

<!-- unit:e16e7a323eeead093a -->
<a id="u-e16e7a323eeead093a"></a>

### Контролируемая ablation координации

<!-- /unit:e16e7a323eeead093a -->

<!-- unit:53a9ba8b7f22390560 -->
<a id="u-53a9ba8b7f22390560"></a>

Сравнивают single-agent с достаточным budget/self-check, frozen role workflow, shared role-conditioned SFT и другие реально выполненные interventions по отдельности. Dataset/tool access, total calls/tokens и oracle согласуют. Role count не считается метрикой качества; measured improvement должен проявляться в held-out task outcomes и приемлемых затратах.

<!-- /unit:53a9ba8b7f22390560 -->

<!-- unit:66b3300b6fb651bfdd -->
<a id="u-66b3300b6fb651bfdd"></a>

Ошибки можно разложить на локальную policy, lost handoff evidence, incorrect routing, shared misconception, wrong conflict resolution и exhausted global budget. Для training data сохраняют actual actor/version каждого action и доступный ему context. Для final report фиксируют pair/team checkpoints, чтобы old rollout не считался опытом уже другой команды.

<!-- /unit:66b3300b6fb651bfdd -->

<!-- unit:ba8a47934d8728cd4e -->
<a id="u-ba8a47934d8728cd4e"></a>

Multi-agent learning здесь остаётся обзорным углублением по плану. Пользовательское создание subagents, совместное обучение LLM и deployment не выполнялись. Следующая глава замыкает lifecycle: как оценивать полные trajectories и воспроизводить именно заявленные результаты.

<!-- /unit:ba8a47934d8728cd4e -->

<!-- unit:635a82bf92cdd051e6 -->
<a id="u-635a82bf92cdd051e6"></a>

## Контроль и практика

<!-- /unit:635a82bf92cdd051e6 -->

<!-- unit:c81b4f0db726ae66a4 -->
<a id="u-c81b4f0db726ae66a4"></a>

1. Обязательно ли multi-agent означает несколько разных обученных моделей? **Ответ:** нет; роли могут использовать один frozen checkpoint.
2. Почему три согласных агента могут ошибаться вместе? **Ответ:** их ошибки коррелируют через общую модель, context и evidence.
3. Что доказывает преимущество координации? **Ответ:** улучшение на независимых задачах относительно одного агента при сопоставимом общем budget, а не просто относительно одной короткой попытки.

<!-- /unit:c81b4f0db726ae66a4 -->

<!-- unit:495b223ea77bc42898 -->
<a id="u-495b223ea77bc42898"></a>

4. Independent policy sampling означает independent errors? **Ответ:** нет, общие observations/premises могут связывать ошибки.
5. Почему central critic не равен текстовому reviewer? **Ответ:** это value/advantage компонент обучения, а reviewer действует в runtime protocol.
6. Зачем фиксировать partner version? **Ответ:** новая policy партнёра меняет induced environment и доступные messages.
7. Что показывает cross-play? **Ответ:** перенос к другим partners/protocol combinations сверх привычной совместной пары.
8. Почему shared6000budget не значит 6000 накаждого? **Ответ:** это суммарное ограничение с accounting всех ролей и synthesis.
9. Доказывает ли debate обучение координации? **Ответ:** нет, дискуссия может идти при frozen weights.
10. Можно ли по source вопросу о subagents считать их уже созданными в этой работе? **Ответ:** нет, учебный вопрос и реальные выполненные действия различаются.

<!-- /unit:495b223ea77bc42898 -->

<!-- unit:02d3d3e80dbe98b281 -->
<a id="u-02d3d3e80dbe98b281"></a>

**Практика:** спроектируйте executor–critic для read-only поиска правила. Опишите evidence каждого, protocol возражения и stop condition. Затем вручную разберите десять сессий: полезное исправление, вредное исправление, согласованная ошибка, лишний раунд. Предложите ablation без критика и с самопроверкой одиночного агента.

<!-- /unit:02d3d3e80dbe98b281 -->

## Результат главы и переход дальше

Ответьте на вопрос: **При каких условиях несколько участников дают измеримый выигрыш?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
