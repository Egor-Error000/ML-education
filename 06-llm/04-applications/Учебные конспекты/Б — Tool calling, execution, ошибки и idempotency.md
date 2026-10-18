# Б. Tool calling: от предложенного вызова к реальному observation

[Оглавление](README.md). Пункт 2 плана.

**Основа:** S5:2826–3041 и продолжение S6:4–178. В S6:179–181 вы уточнили теоретический режим — здесь примеры учебные. **Исправления:** model proposal не actual execution; observation не произвольная system instruction; native format, schema enforcement и capability разные. **Углубление:** lifecycle call ID, retry ambiguity и idempotency.

## 1. Что модель делает, а что исполняет приложение

Tools расширяют доступный способ решения: поиск получает источники, calculator — точный результат заданного вычисления, API — состояние системы. Модель создаёт structured request, executor запускает разрешённую функцию и возвращает результат. Текст «я проверил базу» без actual observation не доказательство проверки.

Пример read-only tool `lookup_document(document_id)` принимает ID и возвращает document/version либо structured error. Description сообщает модели назначение и входы; schema задаёт types. Credentials и права берутся из application context, а не из предложенных моделью полей вроде `is_admin:true`.

Если нужен запрос «на завтра», relative date разрешается по явному reference date/timezone. Если отсутствует обязательное число участников, его нельзя молча выдумывать как 2 только потому, что так было в assistant snippet исходника. Недостающие параметры либо получают documented default из trusted policy, либо уточняются.


<!-- source-depth-applications-Б -->
### 1.1. Возвращаем примеры погоды, бронирования и анализа данных

S5 вводит `get_weather(city,units)`, затем `analyze_data(dataset_path,columns,analysis_type,config)`. S6 продолжает примером `book_table(date,time,guests)`. Они показывают три разные задачи: получить текущее наблюдение; вычислить результат по выбранным данным; изменить внешнее состояние. У всех может быть похожая JSON-форма proposal, но последствия ошибки и стратегия восстановления различаются.

Для погоды недостаточно правдоподобного числа. Нужны результат реального источника, город без неразрешённой неоднозначности, единицы и время наблюдения. Для анализа недостаточно существующего CSV: нужно понять, какие columns и какой анализ запрошены. Для бронирования правильный date/time не заменяет отсутствующее число гостей. В исходном примере пользователь сказал только «завтра в 19:00», а generated call уже содержит guests=2. Это необоснованная подстановка, если нет отдельно заданного допустимого default.

Общее правило вывода аргументов: брать данные из запроса и trusted state, применять только объявленные defaults, а недостающую существенную информацию уточнять или возвращать предусмотренный missing-input status. То, что schema требует поле, не даёт права придумать его значение. Тот же принцип относится к dataset_path: имя «sales.csv» не обязательно однозначно определяет конкретный разрешённый файл.

### 1.2. Контракт имеет входную и выходную стороны

Tool name и description объясняют, когда использовать операцию. Input schema описывает форму аргументов. Реальный implementation задаёт, что операция делает, какие ошибки возвращает и какие эффекты возможны. Output contract нужен, чтобы модель и runtime различали результат, отсутствие данных, частичный результат и ошибку. Один список parameters не описывает весь инструмент.

Например, lookup принимает document ID, а возвращает `status`, `document_id`, `revision`, `content` либо typed error. Search может возвращать candidate IDs, snippets и scores; он не обязан сразу загружать полный документ. Если score —cosine, его нельзя трактовать как calibrated probability правильности. Если результат усечён, это должно быть известно следующему шагу, иначе модель примет preview за полный источник.

Descriptions полезно формулировать конкретно: что ищется, когда нужен инструмент, какие поля результата можно цитировать и какие ограничения применяются. Два tools «ищет данные» и «тоже ищет данные» затрудняют выбор. Но хорошее описание не заменяет enforcement: права и диапазоны проверяются программой даже тогда, когда в prompt написано «никогда не превышай limit».

### 1.3. Обучение protocol не выполняет функцию

В S5 ошибочно заявлен обязательный отдельный tool head. В распространённой autoregressive схеме Transformer выдаёт последовательность токенов, представляющую call, через обычную vocabulary projection. SFT на trajectories делает такие продолжения вероятнее; chat template задаёт сериализацию roles/tools/results; parser восстанавливает structured message. Их связь с обучением и next-token objective подробно разобрана в [главе orchestration](<../../03-inference/Учебные конспекты/З — Orchestration, tools, RAG и границы runtime.md>).

Назвать функцию в тексте, получить parsed call и исполнить функцию —три разных события. Даже если SDK объединяет их в одном удобном методе, работа всё равно разделена между model generation и внешним кодом. Модельный forward сам по себе не открывает файл и не отправляет запрос в ресторан. Backend/SDK может автоматически запустить зарегистрированный handler; это способность окружающей программы.


## 2. Один полный цикл

1. Получить request и known authorization scope.
2. Передать model доступные tool contracts и relevant state.
3. Принять proposal с уникальным call ID.
4. Проверить known tool, schema, semantic args, permissions и budgets.
5. Исполнить функцию с timeout/cancellation policy.
6. Сохранить observation, связанное с call ID: status,data,error,provenance.
7. Вернуть observation модели в правильном channel/template.
8. Получить следующий call или final answer, проверить completion.

Никакого automatic learning weights в этих шагах нет. Это inference execution loop; SFT/RL на trajectories — следующая отдельная тема lifecycle.

В observation должны различаться empty result и error. «Ничего не найдено» — успешный поиск без matches; «search timeout» — нет проверенного ответа на query. Если превратить timeout в пустой list, model может неправильно уверенно заявить об отсутствии документа.


### 2.1. Прослеживаем read-only задачу с двумя calls

Пользователь спрашивает правило timeout для payment-service v8. Доступные tools: search и lookup. Первый model response предлагает search с query и limit=5, runtime назначает/принимает протокольный call ID c1. После validation и исполнения observation c1 содержит ID `pay-v8#timeout`, snippet и revision. Второй model response предлагает lookup этого ID с call ID c2. Только после lookup появляется полный passage с условиями применимости, на который ссылается final answer.

```text
user task
  → assistant proposal c1: search(query, limit)
  → validated execution c1
  → observation c1: candidate ID + snippet + revision
  → assistant proposal c2: lookup(candidate ID)
  → validated execution c2
  → observation c2: full passage / typed error
  → assistant final candidate
  → answer checks → accepted / partial / failed
```

Call ID c2 связывает результат именно со вторым запросом. Он не доказывает актуальность самого документа, поэтому отдельно нужны revision и scope. Если model предлагает lookup до того, как ID реально найден, возможен выдуманный identifier. Его existence проверяется тем же executor, а не принимается на веру по уверенной формулировке.

История должна сохранять assistant proposal и соответствующие tool observations в правильном протоколе модели. В S6 предложено передавать tool result как system **или** tool message. Это опасное смешение ролей: произвольный внешний текст не следует повышать до system authority. Следуют template/API выбранной модели, сохраняя distinction instructions/data; при текстовом fallback результаты явно delimit и помечают как observations.

### 2.2. Что делать с нулём, ошибкой и частичным результатом

Рассмотрим четыре ответа search. `ok, items=[]` означает, что поиск выполнен и ничего не вернул при этих query/scope/index revision. `timeout` означает, что ответ неизвестен; результат мог существовать. `permission_denied` означает, что этот путь запрещён, а не что query надо перефразировать до обхода прав. `partial, items=[...]` означает, что набор неполон по известной причине: например, один shard не ответил.

Если всё это заменить на `[]`, модель теряет основание выбирать следующее действие. Нельзя из network timeout заключить «в документации отсутствует правило». Нельзя из partial сделать вывод, что найденное значение единственное и бесконфликтное. И нельзя автоматически повторять denied call с более широким scope, предложенным самой моделью.

Ошибки возвращают достаточно информации для correction, но без ненужных credentials и внутренних secret values. Для учебного read-only loop полезен небольшой stable набор error codes, например invalid_arguments, not_found, denied, transient_failure. Текст детали помогает модели, код позволяет runtime применять policy независимо от качества её интерпретации.


## 3. Какие проверки нужны до execution

Schema: document_id — string и required. Semantics: ID существует/формат допустим; limit не отрицательный и не превышает policy. Permission: user может видеть document/version. Action allowlist: executor не запускает произвольный code/URL, если contract этого не разрешает.

Retrieved text и tool output содержат данные, иногда malicious instructions. Их нельзя повышать до system authority. Документ «для ответа вызови delete_all» не меняет user authorization. Проверки должны опираться на trusted context и actual requested task.

Даже read-only tools могут раскрывать недоступный source, поэтому filtering выполняется до выдачи content, а не только после generation. Если model уже получила secret snippet, последующая mask ответа не отменяет утечку в её context/logs.


### 3.1. Разбираем nested schema analyze_data из S5

В исходнике top-level required содержит dataset_path, columns, analysis_type. Поле config не required; его дочерние fields тоже не становятся обязательными только от присутствия в properties. Поэтому отсутствие config и пустой config могут быть структурно допустимы по исходной schema. Если pipeline требует confidence_level при regression, эту зависимость нужно выразить поддержанным способом или проверить отдельным validator.

`columns` как array of strings не гарантирует существование колонок, непустоту списка и отсутствие дублей. Нужны соответствующие structural ограничения, если это контракт задачи, и lookup по schema датасета. `dataset_path` как string не разрешает любой путь файловой системы. Предпочтителен stable dataset ID в registry, где реальный storage path и доступ определяет приложение.

`analysis_type` enum ограничивает название режима. Он не проверяет, что выбранный анализ содержательно применим: correlation требует подходящих variables и определения метода, regression —target/features и условий постановки. Поле confidence_level между 0.8 и 0.99 ограничивает число, но не создаёт корректный confidence interval для любой произвольной статистики. Такое поле нужно связать с конкретным вычислительным алгоритмом.

Если tool принимает configuration, её версия и defaults должны быть воспроизводимыми. `include_visualizations:true` не объясняет, какие charts и куда сохраняются; output contract должен указать artifacts или отсутствие их создания. В конспекте это учебный интерфейс анализа, а не утверждение, что у пользователя уже развернут сервис и создан набор отчётов.

### 3.2. Schema и constrained decoding на примере book_table

Схема S6 задаёт date string, time pattern HH:MM и guests integer от 1 до 20. Pattern проверяет диапазон часов/минут, но не наличие свободного столика. Формат даты требует отдельной поддержки проверки, а «завтра» определяется reference date и timezone задачи. Реальное booking API может иметь другие диапазоны и ограничения; учебные 1–20 нельзя объявлять универсальным правилом ресторанов.

Constrained generation, если backend поддерживает нужные keywords, способна ограничить форму вызова. Но предложение «не нужны parsing, retry logs и проверки» из исходника неверно. Нужны завершённый message, извлечение arguments, runtime validation и обработка execution error. Структурно безупречная дата может быть в прошлом, в другом городе или не соответствовать согласованному желанию пользователя.

Нужно также различать разрешение вызвать tool и пользовательскую задачу, которую call реализует. Для текущего изучения не требуется делать реальную бронь; пример нужен, чтобы увидеть, где заканчивается inference и начинаются effects. Права выполнения задаются реальным приложением, без выдуманного общего закона «любое действие всегда требует одинакового дополнительного подтверждения».

### 3.3. Большой каталог tools: поиск кандидатов и окончательный выбор

S6 предлагает retrieval описаний, иерархию категорий и few-shot examples при большом числе инструментов. Это разные методы. Retrieval сужает список доступных описаний по задаче; иерархия делает несколько решений на разных уровнях; examples показывают корректные связи request→call. Они помогают вместить context и уменьшить неоднозначность, но создают дополнительные ошибки отбора.

Нельзя говорить, что 100 tools всегда не помещаются. В учебном расчёте 100 описаний по 120 tokens занимают 12000 до общей инструкции и истории. Для окна 8192 это уже больше окна; для 32768 сами описания помещаются, но оставшийся бюджет и качество выбора всё равно требуют оценки. Размер и содержимое definitions важнее одного порога количества.

Если правильный tool попадает в shortlist с вероятностью 0.95, выбирается из него с вероятностью 0.90, args правильны с условной вероятностью 0.98, execution успешно с условной вероятностью 0.99, цепочка даёт $0.95\cdot0.90\cdot0.98\cdot0.99=0.829521$. Это произведение условных вероятностей по объявленным этапам; независимость не нужна, если каждая вероятность действительно условная на успех предыдущих. Высокая точность выбора внутри shortlist не компенсирует пропущенный правильный инструмент.

Подменить версии или одноимённые tools нельзя незаметно. Registry связывает имя, contract и implementation revision. Во время выполнения длинной задачи изменение каталога может потребовать повторной validation pending call. Model-visible description и actual handler должны описывать одну операцию, иначе даже разумный выбор модели выполнит не то, что она ожидала.


## 4. Retry и ambiguous outcomes

Transient read timeout можно повторить ограниченно, если общий deadline позволяет. Invalid arguments требуют correction, missing permission — termination/допустимой альтернативы. Бесконечный retry не является восстановлением.

Для mutating tool timeout неоднозначен: операция могла завершиться, а response потеряться. Повтор может создать duplicate. Idempotency key позволяет серверу распознать повтор того же intent и вернуть прежний результат вместо нового действия, если API поддерживает такой contract. Key должен быть стабилен для retry одной операции и не совпадать случайно для разных intents.

Пример: `create_report(request_id=R1)` завершился, но клиент не получил response. Retry с R1 возвращает существующий report. Retry с новым R2 может создать второй. Если server API не предоставляет idempotency, нужен другой проверяемый recovery/status protocol; клиентский текст «это повтор» сам по себе не гарантирует exactly-once.


### 4.1. Failure до отправки и unknown после отправки

Если validation отклонила call до исполнения, effect ещё не было в этом пути. Если соединение оборвалось после отправки запроса, результат неоднозначен. Если сервис ответил permanent domain error, повтор того же запроса не обязательно полезен. Разбор failure stage важнее общего слова failed.

В booking-примере сервер мог создать reservation R7 и потерять response. Для клиента это timeout, для ресторана —существующая бронь. «Попробуем ещё раз с новым request ID» может создать R8. Проверка только локального state не покажет R7, если observation не был сохранён. Нужен контракт server-side idempotency или поиск фактического результата по business identity.

Стабильный idempotency key относится к одному intent: например, одной согласованной брони. Если аргументы поменялись с двух гостей на четырёх, повтор с прежним key должен обработать конфликт по объявленной policy, а не молча вернуть старую бронь как новую. Новый transport attempt и новое бизнес-намерение —разные сущности; одинаковый call ID или текст запроса не заменяет их явного определения.

### 4.2. Почему локального словаря «уже выполнено» недостаточно

Два workers могут одновременно не увидеть key, оба выполнить операцию, а затем записать result. Поэтому deduplication должна разрешать race атомарно и связывать key с параметрами/статусом/result. Если эффект и запись dedupe находятся в одной transactional системе, можно использовать её guarantees. Если внешний эффект происходит в другом сервисе, одна локальная transaction не делает оба события атомарными.

Полезная модель server operation record: pending → succeeded(result) либо failed(reason), плюс policy recovery для зависшего pending. Но сама запись pending ещё не доказательство, что внешняя операция не произошла. А истечение TTL dedupe record может снова позволить effect позднего retry. Поэтому guarantees всегда ограничены supported API, retention window и recovery protocol.

Exactly-once effect не следует путать с ровно одним network request. Идемпотентный API может получить несколько одинаковых requests и создать один effect. Retry transport с at-least-once доставкой может быть приемлем, если effect deduplicated. Эти механизмы обсуждаются как контракт исполнителя, без обещания, что произвольный wrapper автоматически обеспечивает их.

### 4.3. Deadline и backoff считаются на всю операцию

Учебная policy: максимум 4 attempts, каждый занимает до 1.2 s, delays перед retries 0.25, 0.5 и 1 s. Верхняя сумма без других затрат —$4\cdot1.2+0.25+0.5+1=6.55$ s. Если общий remaining deadline 6 s, четвёртая полная попытка уже не помещается. Перед каждым retry нужно оценить остаток, а не заново выдать tool 6 секунд.

Exponential backoff ограничивает частоту повторов; jitter может разнести синхронные clients. Ни один из них не исправляет неверные arguments и не создаёт отсутствующее разрешение. Для rate limit учитывают server policy и retry-after при наличии; для permanent error прекращают механическое повторение. Для mutation сначала разрешают неоднозначность effect, а потом выбирают retry.


## 5. Parallel calls и termination

Независимые reads разных documents можно исполнить параллельно; зависимый call «прочитать ID, найденный первым search» требует результата search. Parallel execution с общим mutable state требует merge/conflict policy. Если result arrival order меняется, финальный state не должен случайно терять observations.

Termination conditions: user task выполнена, evidence достаточно, exhausted steps/tokens/time, permanent error или нужен human input. Model saying «готово» — не единственный oracle: приложение сверяет требуемые artifacts/state. Budget учитывает model и tool time/cost, retries и nested agents.


### 5.1. Один response с несколькими calls и реальная зависимость

Два weather calls для разных городов могут исполняться независимо. Но «найди документ и прочитай найденный ID» требует результата поиска. Model response, в котором второй call уже содержит не наблюдавшийся ID, не создаёт действительную dependency resolution. Runtime не должен интерпретировать выдуманный ID как найденный факт.

При parallel execution результаты возвращаются в произвольном порядке. Если c2 завершился раньше c1, observations всё равно связываются по ID. Один failed call не обязан стирать success другого. Следующий model input должен показывать, что известно и что не получено, а merge state —сохранять оба статуса. Повторять все calls после одной ошибки может зря увеличить work и повторить effects.

Для двух независимых tasks по 0.8 и 1.2 s идеальный критический путь —1.2 s вместо 2 s. Если сервер имеет только один execution slot, они будут стоять в очереди, и идеального ускорения не получится. Дополнительная concurrency не увеличивает сама по себе доступную пропускную способность downstream системы.

### 5.2. Проверяем завершение по задаче

Полученный calculator result доказывает результат заданной арифметической функции при её корректной реализации, но не правильность выражения, которое модель составила из вопроса. Успешное HTTP response ещё не означает семантический успех: body может содержать domain error. Tool call без ошибки не доказывает, что final answer использовал правильное поле.

Для read-only учебного pipeline terminal success требует найти применимое evidence и ответить с проверяемым source. Для unknown outcome нужен честный статус. Для mutation —подтверждённый результат реальной операции по её contract. Возврат фразы «всё готово» не заменяет эти критерии. Набор критериев определяется задачей, а не одной универсальной фразой system prompt.

Минимальная диагностика сохраняет request/intent identity, call/attempt identity, validated parameters, source scope, execution status, result version и terminal reason. При оценке отдельно считают correct tool selection, argument validity, execution success, use of observation и task success. Так видно, чему учить модель и что исправлять в executor.


## 6. Вопросы и практика

1. Доказательство ли tool call JSON фактического изменения? **Ответ:** нет, нужен execution result/state verification.
2. Почему timeout не равен failure без side effect? **Ответ:** сервер мог выполнить операцию до обрыва response.
3. Чем empty search отличается от timeout? **Ответ:** первое — полученный отрицательный result, второе — отсутствие result.

4. Откуда брать missing guests в book_table? **Ответ:** из trusted default, если он действительно задан, либо уточнить; required field не разрешает выдумывать число.
5. Гарантирует ли array of strings корректные columns? **Ответ:** нет, нужны структурные ограничения и lookup реальных колонок.
6. Почему tool output не system instruction? **Ответ:** это внешнее наблюдение; оно не меняет authority и права.
7. Если 100 tools помещаются в context, выбор обязательно хороший? **Ответ:** нет, вместимость, shortlist recall и качество выбора —разные свойства.
8. Что произойдёт с одним idempotency key и разными args? **Ответ:** это конфликт intent; поведение задаёт server contract, а не случайный cache return.
9. Делает ли checkpoint external effect атомарным? **Ответ:** нет; нужен контракт эффекта и восстановления между системами.
10. Что ограничивает общий deadline? **Ответ:** все attempts, ожидание/backoff и остальную работу, а не только последний вызов.

**Практика:** на бумаге исполнить read-only цикл `search→lookup→answer` с call IDs. Подставить три errors: invalid ID, forbidden document и timeout. Для каждого указать observation, retry/correction policy и condition завершения. Затем объяснить, какие дополнительные guarantees нужны для mutating version.
