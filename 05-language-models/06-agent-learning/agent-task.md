# Постановка обучения агента и baseline

**Уровень:** основа ветки.

**Предпосылки:** [Состояние, orchestration и граф управления](<../05-applications/app-state.md>), [Оценка retrieval, ответов и task success](<../05-applications/app-evaluation.md>)

**Используют:** [Среда, observations, actions и horizon](<agent-environment.md>)

**Учебный вопрос:** Как выбрать вмешательство по измеренному типу ошибки?

До выбора метода обучения зафиксируйте задачу, среду и оценку целой попытки. Первые три главы образуют обязательную постановку эксперимента; демонстрации, recovery и RL читаются после неё.

## Карта вычисления и границы этапов

![Постановка обучения агента и baseline: входы, преобразования и выходы](<../../оформление/course/agent-task.svg>)

Задача агента определяется изменением среды и проверкой результата. Красивый финальный текст не заменяет достигнутого состояния.

## Цикл начинается с задачи и оценки

Pretrained checkpoint (сохранённое состояние модели) — начальная точка маршрута, а не конечное приложение. Требуются данные, спецификация поведения, evaluation и интерфейсы. После deployment появляются новые inputs/errors, которые могут привести к обновлению prompt (входной запрос или инструкция), corpus, runtime (среда исполнения модели) или параметров. Такое обновление снова проходит оценку.

Useful lifecycle: задача → независимые evaluation cases → baseline (базовый вариант сравнения) → diagnosis ошибок → одна интервенция → сравнение → export/integration → нагрузочная проверка → rollout/rollback → мониторинг и новые данные. Это не обязательная цепочка «каждый раз SFT→DPO→RL»: intervention выбирается по evidence.

Например, assistant неверно отвечает на новую внутреннюю инструкцию. Если инструкция отсутствует в доступных данных, RAG/обновление corpus — естественная гипотеза. Если документ найден, но его version неверна, исправляется ingestion/filtering. Если evidence правильный, но модель систематически игнорирует отрицания, рассматриваются prompt/data/model interventions. Один final failure ещё не определяет метод.

**Схема курса:** task metric + baseline failures → diagnosis → choose prompt/RAG/SFT/RL → change one intervention → held-out outcome/cost → continue or rollback



### От этапов из разбираемого примера к управляемому циклу

PEFT, SFT, обучение предпочтениям, квантование, объединение и оценка решают разные задачи. Это порядок знакомства с методами, а не обязательный рецепт «каждую модель нужно провести через всё». Контекст, инструменты, граф выполнения и обслуживание относятся к другим компонентам системы. Вместе они описывают несколько мест, где может изменяться поведение системы. Следующий метод выбирается по наблюдаемой проблеме, а не потому, что его название идёт следующим в списке.

Для дальнейших глав удобно мыслить системой из компонентов: параметры θ, prompt/template (шаблон сериализации входа) P, данные/corpus D, retriever/index R, tool (инструмент приложения)/environment (среда взаимодействия) E, control flow G, memory M и budget B. Пользователь видит результат совместного исполнения. Одинаковая θ не означает одинаковые ответы при других D/P/E, а новая θ не гарантирует, что исправлена ошибка в G. Версия системы должна фиксировать существенные компоненты, иначе сравнение неинтерпретируемо.

Ниже используются учебные постановки и траектории. Они объясняют выбор обучающего сигнала и не являются отчётом о проведённом обучении агента.

### Сначала задаём observable success

Задача «хороший ассистент по документации» слишком расплывчата. Проверяемая постановка: по доступным документам нужной revision ответить на перечисленные вопросы, сослаться на поддерживающие passages, при отсутствии данных честно обозначить пробел и уложиться в выбранный бюджет. Для анализа artifacts добавляются dataset/config identity и корректная интерпретация метрики. Для действия —проверяемое изменение состояния без лишних effects.

Сразу задают, что **не считается** успехом. JSON без нужного смысла, citation на старую версию, повторённая после timeout операция и красивый вывод по несуществующей строке ablation (сравнение с удалённой или изменённой частью метода) table могут выглядеть убедительно. Эти cases нужны в baseline evaluation до изменения weights. Если oracle замечает только parse success, он направит и training, и выбор candidates в слишком узкую цель.

Сначала выбирают unit оценки: independent task с initial state (состояние системы) и stop condition. Отдельные messages/calls становятся диагностическими units. Затем выбирают task families, source/time/group split (разбиение данных) и критерии final outcome. Нельзя обещать качество на будущих данных по одной тренировочной траектории, даже если она полностью понятна и воспроизводима.

### Failure trace важнее догадки по последней фразе

Для неверного timeout проверяют цепочку: правило есть в corpus? correct revision проиндексирована? retriever нашёл passage? filtering/packing сохранили его? generator получил условие целиком? число извлечено правильно? final citation соответствует claim? На каждом этапе можно получить похожий неправильный ответ по разной причине.

Если источник отсутствует, сильнее prompt не создаст его содержимое. Если passage найден, но вырезан packing, новый embedding (векторное представление) encoder (кодировщик входа) может не затронуть сбой. Если model output верен, а UI/cache (кеш повторно используемых вычислений) показывает старый ответ, policy training также не является прямым исправлением. Диагностика связывает intervention с конкретной causal hypothesis, а не с общей категорией «модель ошиблась».

Для каждого изменения формулируют ожидаемый признак и опровержение. Гипотеза «нужный документ не доходит до prompt» предсказывает успех при контролируемой подстановке evidence. Если и с oracle context (доступный модели контекст) ответ неверный, одной этой гипотезы недостаточно. Если correct passage уже в actual input, сначала исследуют его интерпретацию/format, не переписывая всё приложение.

## Что меняет каждое вмешательство

Prompting меняет вход и доступные instructions. Structured output (ответ заданной структуры) ограничивает форму. Retrieval (поиск кандидатов) меняет доступные evidence. Tool execution добавляет внешние operations/observations. Graph меняет ordering/state/termination. SFT/PEFT меняют параметры на demonstrations. Preferences дают сигнал сравнительного качества, RL — feedback из rollouts/reward (награда за действие или результат).

Compression/serving (обслуживание запросов к модели) меняют representation/runtime и quality-cost trade-off, но сами по себе не объясняют, как выбирать sequence действий. Continual learning обновляет поведение по новым данным, требуя проверки прежних способностей. У каждого layer собственные failure modes и evaluation artifacts.

Задачу агентного поведения нужно отличать от single-turn formatting. Если одна tool request имеет correct syntax, но после её результата policy выбирает неверный следующий шаг, schema (схема допустимых данных) уже не основной bottleneck. Если tools возвращают ошибочные данные, обучение policy может закрепить работу вокруг сломанного интерфейса вместо исправления среды.

### Prompt/schema/runtime и обучение дают разные типы изменений

Prompt меняет instructions/examples, которые модель видит в текущем call. Он может ясно определить field meanings и unknown outcome. Schema-aware generation ограничивает допустимую форму. Runtime validator проверяет структуру/права/смысл по своим данным. Эти механизмы способны исправить часть failures при неизменных θ, но не создают гарантии истинного ответа только из хорошего оформления output.

Retrieval добавляет evidence, index update меняет доступную версию знания. Tool исполняет external operation или вычисление и создаёт observation. Graph задаёт последовательность шагов, state transitions и termination. Memory определяет, какие прежние observations/constraints доступны. Наличие любого из этих компонентов не означает online gradient (градиент) learning: следующий context —это всё ещё inference (применение обученной модели) input.

SFT меняет θ по demonstration targets (целевые значения), обычно обучая вероятность нужных assistant tokens (токены). PEFT определяет, **какие параметры и как** обновляются; LoRA сама не является отдельной задачей обучения. На LoRA parameters можно оптимизировать SFT loss (функция потерь) или другой supported objective. «Используем PEFT» не сообщает, какие demonstrations, negatives и критерии качества выбраны.

Preference learning использует сравнение candidates/trajectories по rubric, а RL —return/reward от rollouts в выбранной постановке. Один objective не является автоматическим «следующим уровнем» другого. Для preference training нужны сопоставимые предпочтения, для RL —осмысленная environment/reward и способ получать достаточно информативные rollouts. Если reward отмечает fake success, optimizer (оптимизатор обновления параметров) может улучшить proxy вместо задачи.

### Пять failures: что проверить и как опровергнуть выбор

**Нет нового правила в доступном corpus.** Проверяют ingestion/source availability и freshness, затем controlled retrieval. Возможное изменение —добавить правильный versioned source или live lookup. Если источник уже есть и generator его видит, гипотеза missing knowledge не объясняет этот case полностью. SFT не рассматривают как обязательную замену обновляемого source store.

**Найдена v7 вместо v8.** Проверяют metadata, filters, index-document consistency и prompt packaging. Изменение —правильная version policy. Если v8 дошла до prompt, но модель цитирует v7, исследуют confusion между passages, а не объявляют проблему только database update. Готовая schema может потребовать version field, но не гарантирует выбор нужной версии.

**Malformed tool arguments.** Проверяют template/parser и generation constraints, затем training examples. Возможно, backend вообще неверно сериализует tool descriptions, а checkpoint обладает нужной способностью. Если после исправления protocol ошибка исчезла без изменения θ, это свидетельство проблемы integration. Если корректный interface сохраняет systematic semantic args errors, SFT/data/model intervention становится более обоснованной гипотезой.

**Повтор операции после timeout.** Проверяют intent identity, actual effect/status и retry policy. Исправление может лежать в idempotency/reconciliation исполнителя. Обучить модель «никогда не повторять» тоже недостаточно: оно способно потерять операцию, действительно не выполненную. Следующее действие выбирают по наблюдаемому состоянию и контракту среды, а не по слову timeout как доказательству отсутствия effect.

**Неверный следующий шаг при правильных observations.** Здесь появляется потребность в trajectory (траектория действий и наблюдений) data: модель увидела denied и снова зовёт тот же tool, прочитала нужный passage и продолжает бесконечно искать либо завершилась до обязательной проверки. Сначала проверяют prompt/state/budget. Затем demonstrations/recovery data могут учить правильному выбору actions; RL имеет смысл только с достаточно корректным feedback. Одна success trace не покрывает все такие states.

### Локальный ceiling вмешательства и interactions

Учебная диагностика 100 задач:68 solved,32 failed. По выбранной взаимоисключающей primary-cause разметке 10 failures связаны с missing data,8 с runtime,6 с format и 8 с action policy. Если исправление затрагивает только эти 6 format cases, не создаёт регрессий и другие причины неизменны, максимальный прирост на этом frozen наборе —6points,68→74.

Это условный ceiling для объявленной модели причин, а не доказательство, что «SFT никогда не даст больше 6points». Реальное изменение может затронуть несколько классов, а primary label (метка) скрывает interactions. Но расчёт полезен: ожидать+25points только от закрытия шести известных format failures требует дополнительных оснований.

Ошибка в нескольких stages может иметь несколько корректных labels (метки). Для controlled analysis можно хранить все наблюдаемые failures и отдельно правило primary attribution. Иначе суммы categories не совпадут с числом tasks, а сравнение «самый большой bottleneck» окажется неоднозначным.

### Когда результат требует именно последовательного поведения

Single-turn SFT может научить выдавать search call. Agentic behavior дополнительно требует прочитать result, выбрать lookup, обработать отсутствие документа и вовремя завершиться. Unit данных становится trajectory или её state-action fragment, а evaluation —outcome всего исполнения. Предсказание правильного следующего token (токен) внутри call и правильный control decision после tool error связаны, но не тождественны.

В следующих главах среда будет задавать observations/actions/transitions, demonstrations —корректные actions на history, DAgger-style сбор —состояния, куда попадает ошибающаяся policy, RL —feedback за rollouts. Здесь важно не перескочить к алгоритму до определения интерфейса и oracle. Не всякая проблема длинного loop требует нового large-model training: иногда отсутствующий stop guard виден прямо в коде.

## Сопоставимый baseline и стоимость

Сначала зафиксировать simple prompting/tool loop без нового training. Дать ему те же tasks, tools, context и budgets, которые будут у candidate. Если candidate использует в пять раз больше попыток, часть improvement может объясняться test-time compute, а не лучшей policy.

Общая полезность может быть сформулирована как constrained optimisation:

$$
\max_{\text{system}}\;\mathbb{E}[\mathrm{success}]
\quad\text{при}\quad
\mathbb{E}[C]\le B_C,\qquad P(T> B_T)\le\varepsilon_T.
$$

Здесь $C$ — cost, $T$ — latency (задержка отдельного запроса), budgets задаются приложением. Такая запись помогает не скрыть failure, купленный бесконечными retries. Она не определяет автоматически численные thresholds.

Training cost также входит в выбор. Улучшение на десяти запросах может не окупать expensive tuning, если тот же effect даёт небольшой deterministic fix. Для больших repeated workloads расходы распределяются иначе; сравнивать надо в выбранном горизонте эксплуатации.

### Сравнение вмешательств при заданных constraints

Учебные кандидаты на одном наборе задач: baseline success70%, cost2units/task; retrieval change78%, cost2.2; tuned model80%, cost3. Если допустимый mean cost cap2.5, третий candidate нарушает constraint, несмотря на максимальный success. Среди двух допустимых второй предпочтительнее по success при этих числах и при условии, что прочие требования выполнены.

Один scalar ranking не выражает все trade-offs. Могут быть constraints на forbidden actions, tail latency, memory и доступность оборудования. Снижение mean cost не гарантирует p99; рост average success не гарантирует отсутствие regression на редком важном slice. Hard constraint нельзя незаметно заменить маленьким штрафом в общей weighted sum и затем назвать нарушающую систему допустимой.

Определение budget включает tool/model calls и время. Если baseline делает один attempt, а candidate —best-of5 с oracle selection, сравнивается не только model policy. Можно построить cost-quality curve, сравнить при равном total compute или честно заявить, что улучшена вся система за большую цену. Важно не объяснять весь эффект изменением weights автоматически.

### Training cost и горизонт использования

При одинаковом условном качестве/прочих требованиях пусть подготовка+обучение+evaluation стоят F=600units, inference baseline0.10units/request, candidate0.08. Формальный break-even по этой упрощённой цене —$N_*=600/(0.10-0.08)=30000$ requests. При 15000 requests baseline1500units, candidate1800; при 100000 —10000 против 8600.

Это арифметическая учебная модель, а не рекомендация покупать compute. Реальная фиксированная цена включает сбор/проверку данных, experimentation, integration и обслуживание; качество может различаться, а прогноз нагрузки ошибаться. Если data/schema меняются раньше окупаемости, модель costs тоже пересматривают. «Fine-tuning (дообучение модели) дешевле в inference» само по себе не доказывает меньшую полную стоимость.

Ресурсы изучения тоже ограничены. Для теоретического освоения можно разобрать несколько записанных trajectories и вручную проверить masks/returns, не выполняя полноразмерное обучение. Но успех toy arithmetic не объявляется качеством обученной LLM. Отчёт отделяет объяснённый механизм, simulation и настоящий experiment.

### Веса, loss и task metrics

Training loss сообщает о выбранном objective/data. Низкий SFT loss на demonstrations не доказывает хорошего recovery в states вне этих demonstrations. Высокий training reward не доказывает успех по независимому environment oracle. Preference margin может расти на похожих training pairs и не переноситься на другие task families.

Поэтому сохраняют независимую оценку до обучения и измеряют после него те же outcomes: selection/args, task success, recovery, premature stop/loops и затраты. При необходимости добавляют новые diagnostic cases, но не подменяют прежний holdout (отложенная часть данных) только потому, что новая метрика выглядит лучше. Проверка forgetting оценивает прежние полезные способности, а не только новую специализацию.

Оптимизатор не знает реальной цели вне предоставленного сигнала. Если «успех» награждает слово «готово», он может закрепить именно это слово. Если training data включает fake tool observations, policy учится взаимодействовать с вымышленной средой. Исправление objective/annotation может быть важнее смены LoRA rank или learning rate (шаг обучения).

## Какие артефакты связывают этапы

Data manifest, split IDs и annotation guideline задают происхождение. Model/base revision, tokenizer (токенизатор)/template, adapters и configs определяют computation. Tool/environment schemas и versions определяют действия. Evaluation report содержит outputs, failures и cost. Deployment bundle/rollback связывает tested candidate с реально исполняемой системой.

Если новая model evaluation прошла в одном template, а production использует другой, lifecycle разорван. Если rollout переключает weights, но оставляет несовместимый retrieval space, нельзя считать это проверенной версией. Если в training попали evaluation trajectories, улучшение не доказывает перенос на новые задачи.

В текущем курсе подробный training objective находится в адаптации, serving — в инференсе, tool/RAG contracts — в приложениях. Следующие главы добавляют именно agentic постановку: environment, trajectory data, distribution shift, delayed reward и horizon.

### Version bundle связывает доказательство и исполняемый candidate

Data manifest указывает источник, фильтры, split/group IDs и annotation version. Training record —base checkpoint, tokenizer/chat template, masks, objective, adapter settings, seed (начальное состояние генератора случайных чисел) и существенные hyperparameters. Environment record —tool contracts/implementations, source snapshots, permissions и timeout/idempotency policy. Evaluation record —tasks, outputs, traces, labels, costs и failures.

Если checkpoint обучен на одном template, а serving использует другой, measured model и deployed computation различаются. Если merged/quantized candidate ещё не оценён, нельзя переносить результат FP16 adapter run как доказанный score (оценка) нового artifact. Compression может менять behavior; runtime constraints могут изменить invalid-call rate даже при тех же weights.

Bundle нужен и для rollback: вернуть одни weights без совместимых tokenizer/template/index может не восстановить предыдущую систему. Источники, которые обязаны оставаться актуальными по policy, нельзя произвольно откатить к запрещённой/удалённой версии ради одинакового score. Reproducibility и допустимость текущих данных —разные требования, которые фиксируют явно.

### Offline test, integration и rollout проверяют разные вещи

Offline evaluation даёт controlled outcomes на наборе tasks. Integration проверяет реальные serialization, parser, tools и outputs. Load test —capacity/queue/tail latency с выбранным traffic. Rollout —поведение candidate в настоящем input distribution по принятому протоколу. Хороший offline task score не доказывает, что memory footprint помещается в serving deployment или что retries не усиливаются под нагрузкой.

При сравнении old/new в production нужна сопоставимость traffic/cohorts и версия labels. Shadow run read-only может наблюдать inputs без исполнения effects, но для mutating workflows его semantics отличается от реального state transition; это учитывают в интерпретации. В учебном курсе такой rollout описывается концептуально, не выполняется автоматически при редактировании конспекта.

Обновление можно отвергнуть, даже если одна метрика выросла: worsened critical slice, недопустимые effects, incompatible artifact или excessive cost —отдельные основания. Причину отказа связывают с зафиксированным requirement. Это делает решение reviewable и не превращает evaluation в поиск любой зелёной цифры.

### Monitoring → новые данные → снова независимая проверка

После внедрения меняются queries, corpus и tools. Рост ошибок может быть data drift (изменение распределения или зависимости со временем), обновлением API, сменой index revision или regression модели. Monitoring по версиям/типам задач помогает отличать причины. Feedback пользователя и model-generated trajectories —кандидаты для анализа, а не автоматически правильные training targets.

Если успешная session содержит случайный лишний call, обучение всей последовательности как образца закрепит эту лишнюю работу. Если итог красивый, но effect не произошёл, positive label будет ошибочным. Новые demonstration/preference данные требуют validation (проверочная часть для выбора решения), deduplication и сохранения происхождения. Data split не должен терять связь между несколькими fragments одной session.

Fine-tuning по новым cases может ухудшить прежние. Проводят повторную evaluation старых/новых slices, оценивают forgetting и стоимость; continual update не освобождает от lifecycle. Если исправлена source data policy без training, θ остаётся прежней, но system version всё равно меняется и её нужно оценить.

### Учебный маршрут к следующим главам

Сначала описывается маленькая среда с ограниченными read-only tools и финальным oracle. Затем записываются success/error/recovery traces и объясняется SFT mask. Потом рассматриваются ошибки на visited histories, test-time verification, delayed rewards и preference learning. Оценка из главы [«Оценка траекторий и воспроизводимость»](<agent-evaluation.md>) проектируется до выбора алгоритма, даже если читается после основных механизмов.

Такой порядок позволяет проверять понимание конкретными объектами: observation, proposed action, real result, следующий action, terminal outcome. Graph runtime и policy training соединяются через trajectories, но не становятся одним механизмом. Именно эту связь дополняет lifecycle блок к уже подробным материалам adaptation/inference/applications.

## Самопроверка и упражнение

1. Почему graph improvement не обязательно training? **Ответ:** sequencing/state может меняться при тех же weights.
2. Как понять, что нужен agentic dataset? **Ответ:** failures относятся к выбору последовательности действий/восстановлению при корректных отдельных interfaces.
3. Зачем evaluation до training? **Ответ:** задаёт independent критерий, который нельзя подменить снижением выбранного train loss.

4. Обязан ли lifecycle всегда включать SFT/DPO/RL? **Ответ:** нет, intervention выбирают по failure evidence и constraints.
5. LoRA задаёт цель обучения? **Ответ:** нет, это parameterization/update method; objective и data задаются отдельно.
6. Почему форматный fix имеет условный ceiling? **Ответ:** если он меняет только ограниченный набор format failures, прочие ошибки остаются.
7. Всегда ли лучший success candidate допустим? **Ответ:** нет, он может нарушать cost/latency/operation constraints.
8. Когда окупится F600 при экономии 0.02 на request в toy model? **Ответ:** при 30000 requests без дополнительных затрат/различий качества.
9. Достаточно ли откатить только weights? **Ответ:** не всегда, совместимость template/tokenizer/index/runtime тоже важна.
10. Можно ли все успешные сессии автоматически брать в training? **Ответ:** нет, trace может содержать лишние/неверные действия или ошибочный success label.

**Практика:** взять пять условных failures и назначить layer/hypothesis: missing knowledge, wrong source version, malformed args, повтор после timeout, неверный вывод при верном evidence. Для каждого выбрать одну проверяемую intervention и результат, который опроверг бы гипотезу. Не назначать SFT/RL автоматически всем случаям.

## Результат главы и переход дальше

Следующие темы: [Среда, observations, actions и horizon](<agent-environment.md>).

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
