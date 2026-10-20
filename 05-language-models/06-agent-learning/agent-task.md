# Постановка обучения агента и baseline

**Уровень:** основа ветки.

**Предпосылки:** [Состояние, orchestration и граф управления](<../05-applications/app-state.md>), [Оценка retrieval, ответов и task success](<../05-applications/app-evaluation.md>)

**Используют:** [Среда, observations, actions и horizon](<agent-environment.md>)

**Учебный вопрос:** Как выбрать вмешательство по измеренному типу ошибки?

До выбора метода обучения зафиксируйте задачу, среду и оценку целой попытки. Первые три главы образуют обязательную постановку эксперимента; демонстрации, recovery и RL читаются после неё.



## Карта вычисления и границы этапов

![Постановка обучения агента и baseline: входы, преобразования и выходы](<../../оформление/course/agent-task.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Reward и targets используются в объявленном режиме обучения. При оценке политика фиксирована; success измеряется по всей попытке, включая повторы и бюджет.

<!-- unit:c00d7d28e68d458d0f -->
<a id="u-c00d7d28e68d458d0f"></a>

## Цикл начинается с задачи и оценки

<!-- /unit:c00d7d28e68d458d0f -->

<!-- unit:c7687c0fd499f2811e -->
<a id="u-c7687c0fd499f2811e"></a>

Pretrained checkpoint — начальная точка маршрута, а не конечное приложение. Требуются данные, спецификация поведения, evaluation и интерфейсы. После deployment появляются новые inputs/errors, которые могут привести к обновлению prompt, corpus, runtime или параметров. Такое обновление снова проходит оценку.

<!-- /unit:c7687c0fd499f2811e -->

<!-- unit:82d2ffe479b2c13237 -->
<a id="u-82d2ffe479b2c13237"></a>

Useful lifecycle: задача → независимые evaluation cases → baseline → diagnosis ошибок → одна интервенция → сравнение → export/integration → нагрузочная проверка → rollout/rollback → мониторинг и новые данные. Это не обязательная цепочка «каждый раз SFT→DPO→RL»: intervention выбирается по evidence.

<!-- /unit:82d2ffe479b2c13237 -->

<!-- unit:86457d3606091c1aa0 -->
<a id="u-86457d3606091c1aa0"></a>

Например, assistant неверно отвечает на новую внутреннюю инструкцию. Если инструкция отсутствует в доступных данных, RAG/обновление corpus — естественная гипотеза. Если документ найден, но его version неверна, исправляется ingestion/filtering. Если evidence правильный, но модель систематически игнорирует отрицания, рассматриваются prompt/data/model interventions. Один final failure ещё не определяет метод.

<!-- /unit:86457d3606091c1aa0 -->

<!-- unit:b19b493e5047303c2a -->
<a id="u-b19b493e5047303c2a"></a>

**Схема курса:** task metric + baseline failures → diagnosis → choose prompt/RAG/SFT/RL → change one intervention → held-out outcome/cost → continue or rollback

<!-- /unit:b19b493e5047303c2a -->

<!-- unit:1e8d179350b84bf7d1 -->
<a id="u-1e8d179350b84bf7d1"></a>

<!-- source-depth-lifecycle-А -->

<!-- /unit:1e8d179350b84bf7d1 -->

<!-- unit:6c94effda1bc4c587d -->
<a id="u-6c94effda1bc4c587d"></a>

### От этапов из разбираемого примера к управляемому циклу

<!-- /unit:6c94effda1bc4c587d -->

<!-- unit:09637164f8bc46f7e7 -->
<a id="u-09637164f8bc46f7e7"></a>

S3 последовательно обсуждает PEFT, SFT, preferences, quantization, merging и evaluation. Это порядок знакомства с методами, а не обязательный рецепт «каждую модель нужно провести через всё». S5/S6 добавляют context, tools, graph и serving. Вместе они описывают несколько мест, где может изменяться поведение системы. Следующий метод выбирается по наблюдаемой проблеме, а не потому, что его название идёт следующим в списке.

<!-- /unit:09637164f8bc46f7e7 -->

<!-- unit:9fff253f27daf62dd0 -->
<a id="u-9fff253f27daf62dd0"></a>

Для дальнейших глав удобно мыслить системой из компонентов: параметры θ, prompt/template P, данные/corpus D, retriever/index R, tool/environment E, control flow G, memory M и budget B. Пользователь видит результат совместного исполнения. Одинаковая θ не означает одинаковые ответы при других D/P/E, а новая θ не гарантирует, что исправлена ошибка в G. Версия системы должна фиксировать существенные компоненты, иначе сравнение неинтерпретируемо.

<!-- /unit:9fff253f27daf62dd0 -->

<!-- unit:cedc1d52166b6fb2ab -->
<a id="u-cedc1d52166b6fb2ab"></a>

В этой главе lifecycle —синтез уже обсуждённого и новое связующее объяснение по плану. Последовательное agentic training в исходных чатах не подтверждено. Поэтому дальше будут учебные постановки/траектории, а не рассказ о якобы проведённых пользователем экспериментах.

<!-- /unit:cedc1d52166b6fb2ab -->

<!-- unit:d55f49498823461085 -->
<a id="u-d55f49498823461085"></a>

### Сначала задаём observable success

<!-- /unit:d55f49498823461085 -->

<!-- unit:8441166e8b9bb6f81b -->
<a id="u-8441166e8b9bb6f81b"></a>

Задача «хороший ассистент по документации» слишком расплывчата. Проверяемая постановка: по доступным документам нужной revision ответить на перечисленные вопросы, сослаться на поддерживающие passages, при отсутствии данных честно обозначить пробел и уложиться в выбранный бюджет. Для анализа artifacts добавляются dataset/config identity и корректная интерпретация метрики. Для действия —проверяемое изменение состояния без лишних effects.

<!-- /unit:8441166e8b9bb6f81b -->

<!-- unit:147faa34c56cf4caf7 -->
<a id="u-147faa34c56cf4caf7"></a>

Сразу задают, что **не считается** успехом. JSON без нужного смысла, citation на старую версию, повторённая после timeout операция и красивый вывод по несуществующей строке ablation table могут выглядеть убедительно. Эти cases нужны в baseline evaluation до изменения weights. Если oracle замечает только parse success, он направит и training, и выбор candidates в слишком узкую цель.

<!-- /unit:147faa34c56cf4caf7 -->

<!-- unit:a3ff64fa048fbd065c -->
<a id="u-a3ff64fa048fbd065c"></a>

Сначала выбирают unit оценки: independent task с initial state и stop condition. Отдельные messages/calls становятся диагностическими units. Затем выбирают task families, source/time/group split и критерии final outcome. Нельзя обещать качество на будущих данных по одной тренировочной траектории, даже если она полностью понятна и воспроизводима.

<!-- /unit:a3ff64fa048fbd065c -->

<!-- unit:e7be9026dcff081600 -->
<a id="u-e7be9026dcff081600"></a>

### Failure trace важнее догадки по последней фразе

<!-- /unit:e7be9026dcff081600 -->

<!-- unit:6706c66721e1ea7e9f -->
<a id="u-6706c66721e1ea7e9f"></a>

Для неверного timeout проверяют цепочку: правило есть в corpus? correct revision проиндексирована? retriever нашёл passage? filtering/packing сохранили его? generator получил условие целиком? число извлечено правильно? final citation соответствует claim? На каждом этапе можно получить похожий неправильный ответ по разной причине.

<!-- /unit:6706c66721e1ea7e9f -->

<!-- unit:8c56a7ee965926d05c -->
<a id="u-8c56a7ee965926d05c"></a>

Если источник отсутствует, сильнее prompt не создаст его содержимое. Если passage найден, но вырезан packing, новый embedding encoder может не затронуть сбой. Если model output верен, а UI/cache показывает старый ответ, policy training также не является прямым исправлением. Диагностика связывает intervention с конкретной causal hypothesis, а не с общей категорией «модель ошиблась».

<!-- /unit:8c56a7ee965926d05c -->

<!-- unit:0bb27f683c487c2727 -->
<a id="u-0bb27f683c487c2727"></a>

Для каждого изменения формулируют ожидаемый признак и опровержение. Гипотеза «нужный документ не доходит до prompt» предсказывает успех при контролируемой подстановке evidence. Если и с oracle context ответ неверный, одной этой гипотезы недостаточно. Если correct passage уже в actual input, сначала исследуют его интерпретацию/format, не переписывая всё приложение.

<!-- /unit:0bb27f683c487c2727 -->

<!-- unit:eb58ef6d91d4aedc24 -->
<a id="u-eb58ef6d91d4aedc24"></a>

## Что меняет каждое вмешательство

<!-- /unit:eb58ef6d91d4aedc24 -->

<!-- unit:c160b21fdbf8003a75 -->
<a id="u-c160b21fdbf8003a75"></a>

Prompting меняет вход и доступные instructions. Structured output ограничивает форму. Retrieval меняет доступные evidence. Tool execution добавляет внешние operations/observations. Graph меняет ordering/state/termination. SFT/PEFT меняют параметры на demonstrations. Preferences дают сигнал сравнительного качества, RL — feedback из rollouts/reward.

<!-- /unit:c160b21fdbf8003a75 -->

<!-- unit:7562aabaf81d5f59f6 -->
<a id="u-7562aabaf81d5f59f6"></a>

Compression/serving меняют representation/runtime и quality-cost trade-off, но сами по себе не объясняют, как выбирать sequence действий. Continual learning обновляет поведение по новым данным, требуя проверки прежних способностей. У каждого layer собственные failure modes и evaluation artifacts.

<!-- /unit:7562aabaf81d5f59f6 -->

<!-- unit:b1e2f35f1c4a92e3d0 -->
<a id="u-b1e2f35f1c4a92e3d0"></a>

Задачу агентного поведения нужно отличать от single-turn formatting. Если одна tool request имеет correct syntax, но после её результата policy выбирает неверный следующий шаг, schema уже не основной bottleneck. Если tools возвращают ошибочные данные, обучение policy может закрепить работу вокруг сломанного интерфейса вместо исправления среды.

<!-- /unit:b1e2f35f1c4a92e3d0 -->

<!-- unit:3f5b4aba0bc3649d48 -->
<a id="u-3f5b4aba0bc3649d48"></a>

### Prompt/schema/runtime и обучение дают разные типы изменений

<!-- /unit:3f5b4aba0bc3649d48 -->

<!-- unit:7de365bd1b66179a4b -->
<a id="u-7de365bd1b66179a4b"></a>

Prompt меняет instructions/examples, которые модель видит в текущем call. Он может ясно определить field meanings и unknown outcome. Schema-aware generation ограничивает допустимую форму. Runtime validator проверяет структуру/права/смысл по своим данным. Эти механизмы способны исправить часть failures при неизменных θ, но не создают гарантии истинного ответа только из хорошего оформления output.

<!-- /unit:7de365bd1b66179a4b -->

<!-- unit:bbb798dfce98852bcb -->
<a id="u-bbb798dfce98852bcb"></a>

Retrieval добавляет evidence, index update меняет доступную версию знания. Tool исполняет external operation или вычисление и создаёт observation. Graph задаёт последовательность шагов, state transitions и termination. Memory определяет, какие прежние observations/constraints доступны. Наличие любого из этих компонентов не означает online gradient learning: следующий context —это всё ещё inference input.

<!-- /unit:bbb798dfce98852bcb -->

<!-- unit:0a7903a783d1162bec -->
<a id="u-0a7903a783d1162bec"></a>

SFT меняет θ по demonstration targets, обычно обучая вероятность нужных assistant tokens. PEFT определяет, **какие параметры и как** обновляются; LoRA сама не является отдельной задачей обучения. На LoRA parameters можно оптимизировать SFT loss или другой supported objective. «Используем PEFT» не сообщает, какие demonstrations, negatives и критерии качества выбраны.

<!-- /unit:0a7903a783d1162bec -->

<!-- unit:30065752eff655b24b -->
<a id="u-30065752eff655b24b"></a>

Preference learning использует сравнение candidates/trajectories по rubric, а RL —return/reward от rollouts в выбранной постановке. Один objective не является автоматическим «следующим уровнем» другого. Для preference training нужны сопоставимые предпочтения, для RL —осмысленная environment/reward и способ получать достаточно информативные rollouts. Если reward отмечает fake success, optimizer может улучшить proxy вместо задачи.

<!-- /unit:30065752eff655b24b -->

<!-- unit:ce7d6a77019f51fcdd -->
<a id="u-ce7d6a77019f51fcdd"></a>

### Пять failures: что проверить и как опровергнуть выбор

<!-- /unit:ce7d6a77019f51fcdd -->

<!-- unit:906670e0e5c1329205 -->
<a id="u-906670e0e5c1329205"></a>

**Нет нового правила в доступном corpus.** Проверяют ingestion/source availability и freshness, затем controlled retrieval. Возможное изменение —добавить правильный versioned source или live lookup. Если источник уже есть и generator его видит, гипотеза missing knowledge не объясняет этот case полностью. SFT не рассматривают как обязательную замену обновляемого source store.

<!-- /unit:906670e0e5c1329205 -->

<!-- unit:5f41ae24fd846c8e32 -->
<a id="u-5f41ae24fd846c8e32"></a>

**Найдена v7 вместо v8.** Проверяют metadata, filters, index-document consistency и prompt packaging. Изменение —правильная version policy. Если v8 дошла до prompt, но модель цитирует v7, исследуют confusion между passages, а не объявляют проблему только database update. Готовая schema может потребовать version field, но не гарантирует выбор нужной версии.

<!-- /unit:5f41ae24fd846c8e32 -->

<!-- unit:d12c6170e9e97b60e7 -->
<a id="u-d12c6170e9e97b60e7"></a>

**Malformed tool arguments.** Проверяют template/parser и generation constraints, затем training examples. Возможно, backend вообще неверно сериализует tool descriptions, а checkpoint обладает нужной способностью. Если после исправления protocol ошибка исчезла без изменения θ, это свидетельство проблемы integration. Если корректный interface сохраняет systematic semantic args errors, SFT/data/model intervention становится более обоснованной гипотезой.

<!-- /unit:d12c6170e9e97b60e7 -->

<!-- unit:bfabd05e73d1575c94 -->
<a id="u-bfabd05e73d1575c94"></a>

**Повтор операции после timeout.** Проверяют intent identity, actual effect/status и retry policy. Исправление может лежать в idempotency/reconciliation исполнителя. Обучить модель «никогда не повторять» тоже недостаточно: оно способно потерять операцию, действительно не выполненную. Следующее действие выбирают по наблюдаемому состоянию и контракту среды, а не по слову timeout как доказательству отсутствия effect.

<!-- /unit:bfabd05e73d1575c94 -->

<!-- unit:86b67db6a85cbe5413 -->
<a id="u-86b67db6a85cbe5413"></a>

**Неверный следующий шаг при правильных observations.** Здесь появляется потребность в trajectory data: модель увидела denied и снова зовёт тот же tool, прочитала нужный passage и продолжает бесконечно искать либо завершилась до обязательной проверки. Сначала проверяют prompt/state/budget. Затем demonstrations/recovery data могут учить правильному выбору actions; RL имеет смысл только с достаточно корректным feedback. Одна success trace не покрывает все такие states.

<!-- /unit:86b67db6a85cbe5413 -->

<!-- unit:a9f5cc7f614345e189 -->
<a id="u-a9f5cc7f614345e189"></a>

### Локальный ceiling вмешательства и interactions

<!-- /unit:a9f5cc7f614345e189 -->

<!-- unit:17dc55d17743c46716 -->
<a id="u-17dc55d17743c46716"></a>

Учебная диагностика 100 задач:68 solved,32 failed. По выбранной взаимоисключающей primary-cause разметке 10 failures связаны с missing data,8 с runtime,6 с format и 8 с action policy. Если исправление затрагивает только эти 6 format cases, не создаёт регрессий и другие причины неизменны, максимальный прирост на этом frozen наборе —6points,68→74.

<!-- /unit:17dc55d17743c46716 -->

<!-- unit:b2cafc08c45619634d -->
<a id="u-b2cafc08c45619634d"></a>

Это условный ceiling для объявленной модели причин, а не доказательство, что «SFT никогда не даст больше 6points». Реальное изменение может затронуть несколько классов, а primary label скрывает interactions. Но расчёт полезен: ожидать+25points только от закрытия шести известных format failures требует дополнительных оснований.

<!-- /unit:b2cafc08c45619634d -->

<!-- unit:f554451ce19ad4edeb -->
<a id="u-f554451ce19ad4edeb"></a>

Ошибка в нескольких stages может иметь несколько корректных labels. Для controlled analysis можно хранить все наблюдаемые failures и отдельно правило primary attribution. Иначе суммы categories не совпадут с числом tasks, а сравнение «самый большой bottleneck» окажется неоднозначным.

<!-- /unit:f554451ce19ad4edeb -->

<!-- unit:e6821f885e40b5b7ee -->
<a id="u-e6821f885e40b5b7ee"></a>

### Когда результат требует именно последовательного поведения

<!-- /unit:e6821f885e40b5b7ee -->

<!-- unit:d93d344ffc0c484b0b -->
<a id="u-d93d344ffc0c484b0b"></a>

Single-turn SFT может научить выдавать search call. Agentic behavior дополнительно требует прочитать result, выбрать lookup, обработать отсутствие документа и вовремя завершиться. Unit данных становится trajectory или её state-action fragment, а evaluation —outcome всего исполнения. Предсказание правильного следующего token внутри call и правильный control decision после tool error связаны, но не тождественны.

<!-- /unit:d93d344ffc0c484b0b -->

<!-- unit:c55387281b31cb942a -->
<a id="u-c55387281b31cb942a"></a>

В следующих главах среда будет задавать observations/actions/transitions, demonstrations —корректные actions на history, DAgger-style сбор —состояния, куда попадает ошибающаяся policy, RL —feedback за rollouts. Здесь важно не перескочить к алгоритму до определения интерфейса и oracle. Не всякая проблема длинного loop требует нового large-model training: иногда отсутствующий stop guard виден прямо в коде.

<!-- /unit:c55387281b31cb942a -->

<!-- unit:c66a704638e4908bd2 -->
<a id="u-c66a704638e4908bd2"></a>

## Сопоставимый baseline и стоимость

<!-- /unit:c66a704638e4908bd2 -->

<!-- unit:8dd561ddea7efcaea9 -->
<a id="u-8dd561ddea7efcaea9"></a>

Сначала зафиксировать simple prompting/tool loop без нового training. Дать ему те же tasks, tools, context и budgets, которые будут у candidate. Если candidate использует в пять раз больше попыток, часть improvement может объясняться test-time compute, а не лучшей policy.

<!-- /unit:8dd561ddea7efcaea9 -->

<!-- unit:93a1d821d1b691c582 -->
<a id="u-93a1d821d1b691c582"></a>

Общая полезность может быть сформулирована как constrained optimisation:

<!-- /unit:93a1d821d1b691c582 -->

<!-- unit:d07884b5206a1f4038 -->
<a id="u-d07884b5206a1f4038"></a>

$$
\max_{\text{system}}\;\mathbb{E}[\mathrm{success}]
\quad\text{при}\quad
\mathbb{E}[C]\le B_C,\qquad P(T> B_T)\le\varepsilon_T.
$$

<!-- /unit:d07884b5206a1f4038 -->

<!-- unit:9574b3813e5e39ea80 -->
<a id="u-9574b3813e5e39ea80"></a>

Здесь $C$ — cost, $T$ — latency, budgets задаются приложением. Такая запись помогает не скрыть failure, купленный бесконечными retries. Она не определяет автоматически численные thresholds.

<!-- /unit:9574b3813e5e39ea80 -->

<!-- unit:33f7cc58da0ae364c5 -->
<a id="u-33f7cc58da0ae364c5"></a>

Training cost также входит в выбор. Улучшение на десяти запросах может не окупать expensive tuning, если тот же effect даёт небольшой deterministic fix. Для больших repeated workloads расходы распределяются иначе; сравнивать надо в выбранном горизонте эксплуатации.

<!-- /unit:33f7cc58da0ae364c5 -->

<!-- unit:6086ee7523549fc78f -->
<a id="u-6086ee7523549fc78f"></a>

### Сравнение вмешательств при заданных constraints

<!-- /unit:6086ee7523549fc78f -->

<!-- unit:9683e0a41859a746b8 -->
<a id="u-9683e0a41859a746b8"></a>

Учебные кандидаты на одном наборе задач: baseline success70%, cost2units/task; retrieval change78%, cost2.2; tuned model80%, cost3. Если допустимый mean cost cap2.5, третий candidate нарушает constraint, несмотря на максимальный success. Среди двух допустимых второй предпочтительнее по success при этих числах и при условии, что прочие требования выполнены.

<!-- /unit:9683e0a41859a746b8 -->

<!-- unit:cd735f07bced09ec3d -->
<a id="u-cd735f07bced09ec3d"></a>

Один scalar ranking не выражает все trade-offs. Могут быть constraints на forbidden actions, tail latency, memory и доступность оборудования. Снижение mean cost не гарантирует p99; рост average success не гарантирует отсутствие regression на редком важном slice. Hard constraint нельзя незаметно заменить маленьким штрафом в общей weighted sum и затем назвать нарушающую систему допустимой.

<!-- /unit:cd735f07bced09ec3d -->

<!-- unit:5b58bbed2691cbfeaa -->
<a id="u-5b58bbed2691cbfeaa"></a>

Определение budget включает tool/model calls и время. Если baseline делает один attempt, а candidate —best-of5 с oracle selection, сравнивается не только model policy. Можно построить cost-quality curve, сравнить при равном total compute или честно заявить, что улучшена вся система за большую цену. Важно не объяснять весь эффект изменением weights автоматически.

<!-- /unit:5b58bbed2691cbfeaa -->

<!-- unit:09d76d05d6c1b233f7 -->
<a id="u-09d76d05d6c1b233f7"></a>

### Training cost и горизонт использования

<!-- /unit:09d76d05d6c1b233f7 -->

<!-- unit:f28a7c826151b993d0 -->
<a id="u-f28a7c826151b993d0"></a>

При одинаковом условном качестве/прочих требованиях пусть подготовка+обучение+evaluation стоят F=600units, inference baseline0.10units/request, candidate0.08. Формальный break-even по этой упрощённой цене —$N_*=600/(0.10-0.08)=30000$ requests. При 15000 requests baseline1500units, candidate1800; при 100000 —10000 против 8600.

<!-- /unit:f28a7c826151b993d0 -->

<!-- unit:798bb877d1ee287a00 -->
<a id="u-798bb877d1ee287a00"></a>

Это арифметическая учебная модель, а не рекомендация покупать compute. Реальная фиксированная цена включает сбор/проверку данных, experimentation, integration и обслуживание; качество может различаться, а прогноз нагрузки ошибаться. Если data/schema меняются раньше окупаемости, модель costs тоже пересматривают. «Fine-tuning дешевле в inference» само по себе не доказывает меньшую полную стоимость.

<!-- /unit:798bb877d1ee287a00 -->

<!-- unit:de16ee05539f9273a0 -->
<a id="u-de16ee05539f9273a0"></a>

Ресурсы изучения тоже ограничены. Для теоретического освоения можно разобрать несколько записанных trajectories и вручную проверить masks/returns, не выполняя полноразмерное обучение. Но успех toy arithmetic не объявляется качеством обученной LLM. Отчёт отделяет объяснённый механизм, simulation и настоящий experiment.

<!-- /unit:de16ee05539f9273a0 -->

<!-- unit:514c37653b29fbae70 -->
<a id="u-514c37653b29fbae70"></a>

### Веса, loss и task metrics

<!-- /unit:514c37653b29fbae70 -->

<!-- unit:949b9ebb698f5ff1c7 -->
<a id="u-949b9ebb698f5ff1c7"></a>

Training loss сообщает о выбранном objective/data. Низкий SFT loss на demonstrations не доказывает хорошего recovery в states вне этих demonstrations. Высокий training reward не доказывает успех по независимому environment oracle. Preference margin может расти на похожих training pairs и не переноситься на другие task families.

<!-- /unit:949b9ebb698f5ff1c7 -->

<!-- unit:e3d6ff206e4936488e -->
<a id="u-e3d6ff206e4936488e"></a>

Поэтому сохраняют независимую оценку до обучения и измеряют после него те же outcomes: selection/args, task success, recovery, premature stop/loops и затраты. При необходимости добавляют новые diagnostic cases, но не подменяют прежний holdout только потому, что новая метрика выглядит лучше. Проверка forgetting оценивает прежние полезные способности, а не только новую специализацию.

<!-- /unit:e3d6ff206e4936488e -->

<!-- unit:0f915ccae8f32b59fa -->
<a id="u-0f915ccae8f32b59fa"></a>

Оптимизатор не знает реальной цели вне предоставленного сигнала. Если «успех» награждает слово «готово», он может закрепить именно это слово. Если training data включает fake tool observations, policy учится взаимодействовать с вымышленной средой. Исправление objective/annotation может быть важнее смены LoRA rank или learning rate.

<!-- /unit:0f915ccae8f32b59fa -->

<!-- unit:823f060dc118782385 -->
<a id="u-823f060dc118782385"></a>

## Какие артефакты связывают этапы

<!-- /unit:823f060dc118782385 -->

<!-- unit:836f49c989a3a0e586 -->
<a id="u-836f49c989a3a0e586"></a>

Data manifest, split IDs и annotation guideline задают происхождение. Model/base revision, tokenizer/template, adapters и configs определяют computation. Tool/environment schemas и versions определяют действия. Evaluation report содержит outputs, failures и cost. Deployment bundle/rollback связывает tested candidate с реально исполняемой системой.

<!-- /unit:836f49c989a3a0e586 -->

<!-- unit:bb2ebc8a104669971e -->
<a id="u-bb2ebc8a104669971e"></a>

Если новая model evaluation прошла в одном template, а production использует другой, lifecycle разорван. Если rollout переключает weights, но оставляет несовместимый retrieval space, нельзя считать это проверенной версией. Если в training попали evaluation trajectories, улучшение не доказывает перенос на новые задачи.

<!-- /unit:bb2ebc8a104669971e -->

<!-- unit:e0a68f9ef7629443b3 -->
<a id="u-e0a68f9ef7629443b3"></a>

В текущем курсе подробный training objective находится в адаптации, serving — в инференсе, tool/RAG contracts — в приложениях. Следующие главы добавляют именно agentic постановку: environment, trajectory data, distribution shift, delayed reward и horizon.

<!-- /unit:e0a68f9ef7629443b3 -->

<!-- unit:2e2d3d92553af50b8c -->
<a id="u-2e2d3d92553af50b8c"></a>

### Version bundle связывает доказательство и исполняемый candidate

<!-- /unit:2e2d3d92553af50b8c -->

<!-- unit:f6cdaab678ef81a42f -->
<a id="u-f6cdaab678ef81a42f"></a>

Data manifest указывает источник, фильтры, split/group IDs и annotation version. Training record —base checkpoint, tokenizer/chat template, masks, objective, adapter settings, seed и существенные hyperparameters. Environment record —tool contracts/implementations, source snapshots, permissions и timeout/idempotency policy. Evaluation record —tasks, outputs, traces, labels, costs и failures.

<!-- /unit:f6cdaab678ef81a42f -->

<!-- unit:ed3c514bf0889f2b18 -->
<a id="u-ed3c514bf0889f2b18"></a>

Если checkpoint обучен на одном template, а serving использует другой, measured model и deployed computation различаются. Если merged/quantized candidate ещё не оценён, нельзя переносить результат FP16 adapter run как доказанный score нового artifact. Compression может менять behavior; runtime constraints могут изменить invalid-call rate даже при тех же weights.

<!-- /unit:ed3c514bf0889f2b18 -->

<!-- unit:9f3dfafb3dd9f04b8a -->
<a id="u-9f3dfafb3dd9f04b8a"></a>

Bundle нужен и для rollback: вернуть одни weights без совместимых tokenizer/template/index может не восстановить предыдущую систему. Источники, которые обязаны оставаться актуальными по policy, нельзя произвольно откатить к запрещённой/удалённой версии ради одинакового score. Reproducibility и допустимость текущих данных —разные требования, которые фиксируют явно.

<!-- /unit:9f3dfafb3dd9f04b8a -->

<!-- unit:580e3fc146555b898b -->
<a id="u-580e3fc146555b898b"></a>

### Offline test, integration и rollout проверяют разные вещи

<!-- /unit:580e3fc146555b898b -->

<!-- unit:d0c36522a3ec96e4d0 -->
<a id="u-d0c36522a3ec96e4d0"></a>

Offline evaluation даёт controlled outcomes на наборе tasks. Integration проверяет реальные serialization, parser, tools и outputs. Load test —capacity/queue/tail latency с выбранным traffic. Rollout —поведение candidate в настоящем input distribution по принятому протоколу. Хороший offline task score не доказывает, что memory footprint помещается в serving deployment или что retries не усиливаются под нагрузкой.

<!-- /unit:d0c36522a3ec96e4d0 -->

<!-- unit:7c6fc672725098c088 -->
<a id="u-7c6fc672725098c088"></a>

При сравнении old/new в production нужна сопоставимость traffic/cohorts и версия labels. Shadow run read-only может наблюдать inputs без исполнения effects, но для mutating workflows его semantics отличается от реального state transition; это учитывают в интерпретации. В учебном курсе такой rollout описывается концептуально, не выполняется автоматически при редактировании конспекта.

<!-- /unit:7c6fc672725098c088 -->

<!-- unit:c5572eb22ffc3299ee -->
<a id="u-c5572eb22ffc3299ee"></a>

Обновление можно отвергнуть, даже если одна метрика выросла: worsened critical slice, недопустимые effects, incompatible artifact или excessive cost —отдельные основания. Причину отказа связывают с зафиксированным requirement. Это делает решение reviewable и не превращает evaluation в поиск любой зелёной цифры.

<!-- /unit:c5572eb22ffc3299ee -->

<!-- unit:2c4ba30d2c5f33bee7 -->
<a id="u-2c4ba30d2c5f33bee7"></a>

### Monitoring → новые данные → снова независимая проверка

<!-- /unit:2c4ba30d2c5f33bee7 -->

<!-- unit:452224a9e9635b7641 -->
<a id="u-452224a9e9635b7641"></a>

После внедрения меняются queries, corpus и tools. Рост ошибок может быть data drift, обновлением API, сменой index revision или regression модели. Monitoring по версиям/типам задач помогает отличать причины. Feedback пользователя и model-generated trajectories —кандидаты для анализа, а не автоматически правильные training targets.

<!-- /unit:452224a9e9635b7641 -->

<!-- unit:7f2438d26a75d41b86 -->
<a id="u-7f2438d26a75d41b86"></a>

Если успешная session содержит случайный лишний call, обучение всей последовательности как образца закрепит эту лишнюю работу. Если итог красивый, но effect не произошёл, positive label будет ошибочным. Новые demonstration/preference данные требуют validation, deduplication и сохранения происхождения. Data split не должен терять связь между несколькими fragments одной session.

<!-- /unit:7f2438d26a75d41b86 -->

<!-- unit:904d485bb6777f60ed -->
<a id="u-904d485bb6777f60ed"></a>

Fine-tuning по новым cases может ухудшить прежние. Проводят повторную evaluation старых/новых slices, оценивают forgetting и стоимость; continual update не освобождает от lifecycle. Если исправлена source data policy без training, θ остаётся прежней, но system version всё равно меняется и её нужно оценить.

<!-- /unit:904d485bb6777f60ed -->

<!-- unit:e0917c54603561be28 -->
<a id="u-e0917c54603561be28"></a>

### Учебный маршрут к следующим главам

<!-- /unit:e0917c54603561be28 -->

<!-- unit:72635e185de05f4959 -->
<a id="u-72635e185de05f4959"></a>

Сначала описывается маленькая среда с ограниченными read-only tools и финальным oracle. Затем записываются success/error/recovery traces и объясняется SFT mask. Потом рассматриваются ошибки на visited histories, test-time verification, delayed rewards и preference learning. Оценка из главы [«Оценка траекторий и воспроизводимость»](<agent-evaluation.md>) проектируется до выбора алгоритма, даже если читается после основных механизмов.

<!-- /unit:72635e185de05f4959 -->

<!-- unit:811b2fe47f48044590 -->
<a id="u-811b2fe47f48044590"></a>

Такой порядок позволяет проверять понимание конкретными объектами: observation, proposed action, real result, следующий action, terminal outcome. Graph runtime и policy training соединяются через trajectories, но не становятся одним механизмом. Именно эту связь дополняет lifecycle блок к уже подробным материалам adaptation/inference/applications.

<!-- /unit:811b2fe47f48044590 -->

<!-- unit:0ac007f97722bd2336 -->
<a id="u-0ac007f97722bd2336"></a>

## Самопроверка и упражнение

<!-- /unit:0ac007f97722bd2336 -->

<!-- unit:e67fca85771f429161 -->
<a id="u-e67fca85771f429161"></a>

1. Почему graph improvement не обязательно training? **Ответ:** sequencing/state может меняться при тех же weights.
2. Как понять, что нужен agentic dataset? **Ответ:** failures относятся к выбору последовательности действий/восстановлению при корректных отдельных interfaces.
3. Зачем evaluation до training? **Ответ:** задаёт independent критерий, который нельзя подменить снижением выбранного train loss.

<!-- /unit:e67fca85771f429161 -->

<!-- unit:00026589da924df064 -->
<a id="u-00026589da924df064"></a>

4. Обязан ли lifecycle всегда включать SFT/DPO/RL? **Ответ:** нет, intervention выбирают по failure evidence и constraints.
5. LoRA задаёт цель обучения? **Ответ:** нет, это parameterization/update method; objective и data задаются отдельно.
6. Почему форматный fix имеет условный ceiling? **Ответ:** если он меняет только ограниченный набор format failures, прочие ошибки остаются.
7. Всегда ли лучший success candidate допустим? **Ответ:** нет, он может нарушать cost/latency/operation constraints.
8. Когда окупится F600 при экономии 0.02 на request в toy model? **Ответ:** при 30000 requests без дополнительных затрат/различий качества.
9. Достаточно ли откатить только weights? **Ответ:** не всегда, совместимость template/tokenizer/index/runtime тоже важна.
10. Можно ли все успешные сессии автоматически брать в training? **Ответ:** нет, trace может содержать лишние/неверные действия или ошибочный success label.

<!-- /unit:00026589da924df064 -->

<!-- unit:ba0b6c3c2d0ef04784 -->
<a id="u-ba0b6c3c2d0ef04784"></a>

**Практика:** взять пять условных failures и назначить layer/hypothesis: missing knowledge, wrong source version, malformed args, повтор после timeout, неверный вывод при верном evidence. Для каждого выбрать одну проверяемую intervention и результат, который опроверг бы гипотезу. Не назначать SFT/RL автоматически всем случаям.

<!-- /unit:ba0b6c3c2d0ef04784 -->

## Результат главы и переход дальше

Ответьте на вопрос: **Как выбрать вмешательство по измеренному типу ошибки?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

Следующие темы: [Среда, observations, actions и horizon](<agent-environment.md>).

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
