# Structured output: schema и смысл результата

**Уровень:** основа ветки.

**Предпосылки:** [Обычная генерация по одному токену](<../01-foundations/lm-generation.md>), [Постановка задачи и паспорт данных](<../../01-data-features/01-reliability/df-task.md>)

**Используют:** [Вызов инструментов и исполнение](<app-tools.md>)

**Учебный вопрос:** Что гарантирует синтаксис результата и что ещё должно проверить приложение?





## Карта вычисления и границы этапов

![Structured output: schema и смысл результата: входы, преобразования и выходы](<../../оформление/course/app-structured.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Параметры модели при обычном выполнении фиксированы. Состояние запроса, cache и состояние внешнего инструмента имеют разные жизненные циклы; их нельзя смешивать с optimizer states.

<!-- unit:86c87a290a6511b616 -->
<a id="u-86c87a290a6511b616"></a>

## Почему свободный текст неудобен приложению

<!-- /unit:86c87a290a6511b616 -->

<!-- unit:2266514d34eef8243c -->
<a id="u-2266514d34eef8243c"></a>

Пользователь может прочитать «примерно три дня», а программа ожидает integer $days=3$ или явное unknown. Structured output задаёт contract fields/types, который можно parse, validate и использовать дальше. Это полезно для extraction, classification, routing и аргументов tools.

<!-- /unit:2266514d34eef8243c -->

<!-- unit:3cf037d14cfa6caf0d -->
<a id="u-3cf037d14cfa6caf0d"></a>

Contract начинается со смысла: что означает days, calendar или business days, от какой даты отсчитывается, допускается ли unknown, есть ли source evidence. Если эти вопросы не определены, syntactically perfect JSON лишь аккуратно упакует ambiguity.

<!-- /unit:3cf037d14cfa6caf0d -->

<!-- unit:0bafc71f332324ea11 -->
<a id="u-0bafc71f332324ea11"></a>

Пример extraction из документа: `status∈{found,not_found}`, `value:number|null`, `source_ids:list`. Нужно определить invariant: при not_found value=null; при found value подтверждается указанным source. Некоторые cross-field constraints можно выразить supported schema, другие проверяет application.

<!-- /unit:0bafc71f332324ea11 -->

<!-- unit:a322bce1cce17f4e43 -->
<a id="u-a322bce1cce17f4e43"></a>

<!-- source-depth-applications-А -->

<!-- /unit:a322bce1cce17f4e43 -->

<!-- unit:05a611a1a6983ab292 -->
<a id="u-05a611a1a6983ab292"></a>

### Восстанавливаем мотивацию из S5: что именно ломается

<!-- /unit:05a611a1a6983ab292 -->

<!-- unit:47f4024f8c06683990 -->
<a id="u-47f4024f8c06683990"></a>

В исходном объяснении C.2 перечислены пять ошибок: незакрытые скобки, пояснение перед JSON, неизвестный enum, неверный type и пропущенный required field. Это не одно и то же. Первые две относятся к выбранному синтаксическому протоколу, остальные могут встретиться в совершенно корректном JSON. Строка `{"age":"18"}` разбирается как JSON, но не соответствует контракту integer age. Строка `{"age":18,"extra":1}` тоже корректна; запрещено ли extra, определяет schema.

<!-- /unit:47f4024f8c06683990 -->

<!-- unit:68d161ceff98df27f7 -->
<a id="u-68d161ceff98df27f7"></a>

Приложению нужен результат с заранее определённым значением полей. Модель не должна в одном ответе писать сумму с налогом, в другом без налога, а в третьем помещать в то же поле текст «неизвестно». Перед выбором parser определяют задачу: что извлекается, из какого источника, какая единица и какой scope утверждения. Например, not_found означает «не найдено в предоставленных документах», а не «такого факта нет нигде».

<!-- /unit:68d161ceff98df27f7 -->

<!-- unit:431cf2188c98856607 -->
<a id="u-431cf2188c98856607"></a>

Числа исходника «prompt соблюдается в 90–95% случаев» не являются универсальной статистикой всех моделей. Если в объявленной учебной нагрузке независимая вероятность format success каждого из пяти обязательных шагов равна 0.95, вероятность пройти всю цепочку без format error —$0.95^5\approx0.773781$. Из 1000 таких задач в среднем около 226 столкнутся хотя бы с одной ошибкой формата. Это уже свойство выбранной модели вероятностей, а не измерение какого-либо provider.

<!-- /unit:431cf2188c98856607 -->

<!-- unit:46440d56deef9a52bf -->
<a id="u-46440d56deef9a52bf"></a>

Независимость здесь сильное допущение. Одна и та же сложная schema может ломаться на всех шагах, а простые задачи проходить целиком; тогда произведение marginal rates недостаточно. Тем не менее пример объясняет, почему умеренная локальная надёжность может быть недостаточна для длинной pipeline. Добавление retry меняет cost/latency и не исправляет ошибку смысла, которую validator не распознаёт.

<!-- /unit:46440d56deef9a52bf -->

<!-- unit:7a796896150c5a90af -->
<a id="u-7a796896150c5a90af"></a>

### Object, отсутствующее поле и null

<!-- /unit:7a796896150c5a90af -->

<!-- unit:389da9505c4f7beaad -->
<a id="u-389da9505c4f7beaad"></a>

В JSON Schema `properties` описывает поля, если они присутствуют. Чтобы поле обязательно было в object, его указывают в `required`. Разрешение `null` —отдельный выбор типа. Поэтому «optional» и «nullable» не синонимы: поле может быть required и принимать null, либо быть необязательным, но при наличии принимать только number.

<!-- /unit:389da9505c4f7beaad -->

<!-- unit:b3fec7414082a03666 -->
<a id="u-b3fec7414082a03666"></a>

Минимальная учебная схема:

<!-- /unit:b3fec7414082a03666 -->

<!-- unit:f1e416b682ac373707 -->
<a id="u-f1e416b682ac373707"></a>

```json
{"type":"object","properties":{"value":{"type":["number","null"]}},"required":["value"],"additionalProperties":false}
```

<!-- /unit:f1e416b682ac373707 -->

<!-- unit:d06334c8cd510a32d9 -->
<a id="u-d06334c8cd510a32d9"></a>

Здесь `{}` не подходит, `{"value":null}` подходит, `{"value":"3"}` не подходит, `{"value":3,"unit":"days"}` не подходит из-за extra field. Если единица нужна задаче, её добавляют в contract; удаление неизвестного unit ради успешной validation может уничтожить смысл результата. Standard validator проверяет schema, а не создаёт недостающие данные. Эти различия определены в [описании object/required/null JSON Schema](https://json-schema.org/understanding-json-schema/reference/object).

<!-- /unit:d06334c8cd510a32d9 -->

<!-- unit:110a299aee46569342 -->
<a id="u-110a299aee46569342"></a>

Поле `default` также не следует считать командой автоматически вставить значение во всех validators. Нормализация и defaults —отдельная application policy. Если эта policy превратила отсутствующее число участников в 2, это решение приложения, которое должно быть обосновано условиями задачи. Запись default в schema сама не доказывает согласие пользователя с таким значением.

<!-- /unit:110a299aee46569342 -->

<!-- unit:3cadf4d2d614892dd0 -->
<a id="u-3cadf4d2d614892dd0"></a>

## Три способа получить структуру

<!-- /unit:3cadf4d2d614892dd0 -->

<!-- unit:89a4cc729d16dd8f7d -->
<a id="u-89a4cc729d16dd8f7d"></a>

Prompt просит нужный формат; model может выполнить, но enforcement нет. Parser после generation разбирает строку и выявляет нарушение; он не мешал model выбирать лишние tokens. Constrained decoder строит допустимые token continuations grammar/schema во время generation.

<!-- /unit:89a4cc729d16dd8f7d -->

<!-- unit:ba6fa2fe26be585fb9 -->
<a id="u-ba6fa2fe26be585fb9"></a>

Tool-tuned model обучена воспроизводить соответствующие patterns; это training свойство. Grammar masking — runtime механизм. Они могут сочетаться, но не являются одинаковыми гарантиями: tuned model может нарушить формат, constrained weak model может создать семантически плохой, но валидный output.

<!-- /unit:ba6fa2fe26be585fb9 -->

<!-- unit:62af10836dd6f46f82 -->
<a id="u-62af10836dd6f46f82"></a>

В supported constraint decoder запрещённые logits маскируются, allowed renormalized. Tokenizer pieces могут покрывать несколько symbols; validator должен учитывать весь возможный token continuation, escaping, числа и nested state. Реальный backend может поддерживать только subset JSON Schema, поэтому schema acceptance/validation проверяют заранее.

<!-- /unit:62af10836dd6f46f82 -->

<!-- unit:d432137a43423e4b82 -->
<a id="u-d432137a43423e4b82"></a>

### Prompt → parser: где проходит граница

<!-- /unit:d432137a43423e4b82 -->

<!-- unit:764832aa71da982a83 -->
<a id="u-764832aa71da982a83"></a>

Template может поместить schema и пример в model input. Модель затем свободно генерирует токены. Parser читает уже полученную строку и возвращает object либо ошибку. Это post-hoc путь. Сам факт `parser` в выражении `prompt | model | parser` не доказывает, что при model generation маскировались logits.

<!-- /unit:764832aa71da982a83 -->

<!-- unit:3bf7a2276f8fd965be -->
<a id="u-3bf7a2276f8fd965be"></a>

В S5 пример `JsonOutputParser(pydantic_object=Translation)` снабжён неверным комментарием «под капотом constrained decoding». Parser не управляет decoder автоматически. Более того, в [реализации JsonOutputParser](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/output_parsers/json.py) разбор JSON и получение format instructions нужно отличать от строгой Pydantic-validation конкретного object. У library abstractions проверяют реальный method/version, а не делают вывод по имени аргумента. В собственном учебном коде явный вызов validator после parse снимает эту неоднозначность.

<!-- /unit:3bf7a2276f8fd965be -->

<!-- unit:a766c4824b053d7cdf -->
<a id="u-a766c4824b053d7cdf"></a>

Функция выдачи instructions сама не добавляет их в prompt без соответствующей композиции. Ошибка missing template variable обычно обнаруживается при форматировании/invoke, когда известны входные значения, а не обязательно при объявлении template. Типизированный шаблон облегчает проверку и reuse; он не гарантирует качество перевода и не обучает веса модели.

<!-- /unit:a766c4824b053d7cdf -->

<!-- unit:a9f1d615ac99a751ca -->
<a id="u-a9f1d615ac99a751ca"></a>

Для JSON parser также важен режим строгости. Некоторые wrappers умеют извлекать object из Markdown, дописывать закрывающие скобки или выдавать partial objects при streaming. Это полезные функции для интерфейса, но их результат нельзя автоматически называть исходным полностью завершённым ответом модели. Нужно различать raw output, repaired/parsed candidate и final accepted record.

<!-- /unit:a9f1d615ac99a751ca -->

<!-- unit:2c16e1d9e0eebf93c7 -->
<a id="u-2c16e1d9e0eebf93c7"></a>

### Constrained decoding: множество допустимых продолжений

<!-- /unit:2c16e1d9e0eebf93c7 -->

<!-- unit:647deee655592aa418 -->
<a id="u-647deee655592aa418"></a>

Пусть $L$ —язык строк, допустимых выбранной grammar. Для текущего текстового префикса s разрешён token v, если после декодирования v префикс всё ещё можно продолжить до некоторого элемента L. Например, после `{"value":` токен, который начинает quoted string, может быть разрешён обычным JSON, но запрещён schema с number/null. Grammar state содержит незакрытые containers, ожидание ключа/значения, string escapes и другие ограничения выбранного backend.

<!-- /unit:647deee655592aa418 -->

<!-- unit:11e9ebdace05c6e759 -->
<a id="u-11e9ebdace05c6e759"></a>

Это не обязательно посимвольный выбор. Один tokenizer token способен содержать несколько characters, например конец строки и запятую. Проверяют весь token continuation в контексте tokenizer/grammar. Нельзя просто объявить «следующий токен обязан быть цифрой», как в исходном date-примере, не зная поддержки format и фактических token pieces. Даже корректный date pattern не проверяет автоматически все календарные правила.

<!-- /unit:11e9ebdace05c6e759 -->

<!-- unit:aa3faba27b86059a5d -->
<a id="u-aa3faba27b86059a5d"></a>

Для допустимого множества A распределение после маскирования:

<!-- /unit:aa3faba27b86059a5d -->

<!-- unit:57cfbfae6d696c840b -->
<a id="u-57cfbfae6d696c840b"></a>

$$
q(v\mid s)=\frac{p(v\mid s)\,\mathbf 1[v\in A(s)]}{\sum_{u\in A(s)}p(u\mid s)}.
$$

<!-- /unit:57cfbfae6d696c840b -->

<!-- unit:ef3a2205b911ba1580 -->
<a id="u-ef3a2205b911ba1580"></a>

Если до маски model probabilities трёх вариантов равны 0.60, 0.25 и 0.15, первый запрещён, а два других разрешены, после нормализации остаются 0.625 и 0.375. Маленькая суммарная исходная вероятность допустимых вариантов не делает softmax математически невозможным; проблемы возникают при пустом множестве, численных ошибках или неудачном порядке применения ограничений/sampling. Изменившееся распределение всё равно не выбирает истинное значение автоматически.

<!-- /unit:ef3a2205b911ba1580 -->

<!-- unit:9150c84a510f8aced6 -->
<a id="u-9150c84a510f8aced6"></a>

Сначала ограничить grammar, а затем делать top-p по разрешённому распределению —не то же самое, что сначала оставить top-p без учёта grammar и обнаружить, что все оставшиеся tokens запрещены. Например, unrestricted top-p=0.5 оставил бы только вариант с вероятностью 0.60; после пересечения allowed set пуст. Реальная engine policy должна согласовывать эти операции. Повторять общий тезис «top-p неизбежно ломается при low-probability valid tokens» неправильно.

<!-- /unit:9150c84a510f8aced6 -->

<!-- unit:86aa48493f8de5ab1e -->
<a id="u-86aa48493f8de5ab1e"></a>

### От валидного префикса до валидного завершения

<!-- /unit:86aa48493f8de5ab1e -->

<!-- unit:9fd8037d7e4603573d -->
<a id="u-9fd8037d7e4603573d"></a>

Префикс `{"value":` можно продолжить до правильного object. Сам префикс ещё не object. В терминах автомата это достижимое промежуточное состояние, а не принимающее конечное. Для корректного завершения EOS допускается только в нужном состоянии, а runtime должен действительно получить завершённый result. Max-output limit, отмена и transport failure могут прервать любой путь до закрывающей скобки.

<!-- /unit:9fd8037d7e4603573d -->

<!-- unit:2ce3cc050f63e1036b -->
<a id="u-2ce3cc050f63e1036b"></a>

Поэтому условная гарантия constrained decoder звучит точнее: при корректной реализации, поддержанной specification и успешном завершении output соответствует скомпилированным ограничениям. Из неё не следует, что всегда будет завершённый output за любой token budget, что поддержана вся JSON Schema или что underlying tool выполнится успешно. Парсинг и ошибки terminal status всё равно остаются частью протокола.

<!-- /unit:2ce3cc050f63e1036b -->

<!-- unit:08b1592edd7216ac62 -->
<a id="u-08b1592edd7216ac62"></a>

JSON grammar, schema-aware generation и application schema validation различаются по силе. JSON mode может требовать допустимый JSON, но не конкретные поля. Schema-aware decoder может поддерживать subset keywords. Полный внешний validator способен отклонить то, что прошло более слабое generation constraint. Для библиотеки Instructor также нельзя объявлять один вечный механизм: [её документация различает modes и validation/retry](https://python.useinstructor.com/concepts/patching/).

<!-- /unit:08b1592edd7216ac62 -->

<!-- unit:d5c6c4cd2fee06745b -->
<a id="u-d5c6c4cd2fee06745b"></a>

### Почему grammar из локального примера не требует name и age одновременно

<!-- /unit:d5c6c4cd2fee06745b -->

<!-- unit:f86e78480fa13e9655 -->
<a id="u-f86e78480fa13e9655"></a>

В S5:2755–2786 дана GBNF grammar, после которой утверждается, что output обязательно содержит name и age. Но правило object требует одну или несколько pair, а pair выбирает name **либо** age. Оно не требует обоих ключей и не запрещает повторения. Поэтому объект только с age удовлетворяет такой структуре, как и два age поля. Это пример того, почему надо проверять саму specification, а не только факт включения grammar mode.

<!-- /unit:f86e78480fa13e9655 -->

<!-- unit:38c5ebef417cb1e09d -->
<a id="u-38c5ebef417cb1e09d"></a>

Есть и другая ошибка: `number ::= [0-9]+` разрешает `007`. В обычном JSON leading zeros такого integer недопустимы. Grammar может успешно ограничивать generation своим языком и при этом описывать не тот язык, который ожидало приложение. Sample `{"age":007}` —контрпример к утверждению «эта grammar всегда даёт JSON», а `{"age":35}` —контрпример к утверждению «обязательно два поля».

<!-- /unit:38c5ebef417cb1e09d -->

<!-- unit:16e55b8b3bfb60d2a0 -->
<a id="u-16e55b8b3bfb60d2a0"></a>

При повторяющихся keys JSON consumers могут вести себя по-разному; стандартная загрузка в Python dict обычно сохранит последнее значение. Если protocol запрещает duplicates, проверять это нужно до потери информации о парах ключ/значение. После превращения `{"age":35,"age":36}` в один dict прежний duplicate уже не виден обычной schema validation. Корректный контракт определяет синтаксический приём, duplicate policy и required fields отдельно.

<!-- /unit:16e55b8b3bfb60d2a0 -->

<!-- unit:ae22a114f47d9b2c88 -->
<a id="u-ae22a114f47d9b2c88"></a>

Эти строки —контрпримеры к исходным правилам, а не результаты запуска конкретной локальной модели. Реальный compiler/backend grammar здесь не выполнялся. Для приложения schema с required полями и проверенным JSON parser удобнее, чем предположение, что произвольная короткая grammar уже эквивалентна нужному контракту.

<!-- /unit:ae22a114f47d9b2c88 -->

<!-- unit:686bacd8a284d3fdef -->
<a id="u-686bacd8a284d3fdef"></a>

## Final validation остаётся нужна

<!-- /unit:686bacd8a284d3fdef -->

<!-- unit:cc72fae067cb0f629a -->
<a id="u-cc72fae067cb0f629a"></a>

Generation может быть прервана max tokens, timeout, cancellation, tool transition или engine error. Stream prefix `{"value":` не является final valid object. Не следует выполнять действие по неполному streaming fragment. Сначала terminal state и полная validation выбранного output.

<!-- /unit:cc72fae067cb0f629a -->

<!-- unit:3f90a8c1996cec8e72 -->
<a id="u-3f90a8c1996cec8e72"></a>

Различать:

<!-- /unit:3f90a8c1996cec8e72 -->

<!-- unit:befb4730149ccb02c7 -->
<a id="u-befb4730149ccb02c7"></a>

1. **Parse:** строка разбирается как JSON.
2. **Schema:** поля/types/enums/required constraints соблюдены.
3. **Semantics:** данные существуют, units и cross-field invariants верны.
4. **Task correctness:** извлечён правильный факт/выбрано нужное действие.

<!-- /unit:befb4730149ccb02c7 -->

<!-- unit:7edf4d3fc5555bc2ed -->
<a id="u-7edf4d3fc5555bc2ed"></a>

`{"status":"found","value":10,"source_ids":["D1"]}` может быть parser/schema-valid, но D1 не существует, содержит другое число, недоступен этому user или число относится к другой currency. Это четыре разных ошибки. «JSON validity100%» не равно extraction accuracy100%.

<!-- /unit:7edf4d3fc5555bc2ed -->

<!-- unit:e6e52631d17f0d9ebc -->
<a id="u-e6e52631d17f0d9ebc"></a>

Schema должна представлять неизвестность, если task допускает отсутствие ответа. Если forced enum содержит только yes/no при недостатке evidence, модель вынуждена выбрать один неверно определённый вариант. Принудительная структура не должна принуждать придумывать факт.

<!-- /unit:e6e52631d17f0d9ebc -->

<!-- unit:2462b6a8e5f2bd23d7 -->
<a id="u-2462b6a8e5f2bd23d7"></a>

### Сквозной контракт с неизвестностью и конфликтом

<!-- /unit:2462b6a8e5f2bd23d7 -->

<!-- unit:c18e07652ccc02f194 -->
<a id="u-c18e07652ccc02f194"></a>

Разберём извлечение срока из документов. Output содержит `status`, `value`, `unit`, `source_ids`. При found нужен неотрицательный number, допустимая единица и хотя бы один source. При not_found значение и единица null, список источников пуст. При conflict scalar тоже null, но должны быть минимум два разных source IDs, между которыми обнаружен конфликт. Последнее —упрощённый учебный contract; полная система может сохранять отдельные conflicting claims.

<!-- /unit:c18e07652ccc02f194 -->

<!-- unit:e9def698ad1f0f58cf -->
<a id="u-e9def698ad1f0f58cf"></a>

Ниже schema для внешней проверки по Draft 2020-12; её поддержка конкретным generation backend **не предполагается**:

<!-- /unit:e9def698ad1f0f58cf -->

<!-- unit:e167127550c615cd7d -->
<a id="u-e167127550c615cd7d"></a>

```json
{
  "$schema":"https://json-schema.org/draft/2020-12/schema",
  "type":"object",
  "properties":{
    "status":{"enum":["found","not_found","conflict"]},
    "value":{"type":["number","null"],"minimum":0},
    "unit":{"enum":["business_days","calendar_days",null]},
    "source_ids":{"type":"array","items":{"type":"string","minLength":1},"uniqueItems":true}
  },
  "required":["status","value","unit","source_ids"],
  "additionalProperties":false,
  "allOf":[
    {"if":{"properties":{"status":{"const":"found"}},"required":["status"]},
     "then":{"properties":{"value":{"type":"number"},"unit":{"type":"string"},"source_ids":{"minItems":1}}}},
    {"if":{"properties":{"status":{"const":"not_found"}},"required":["status"]},
     "then":{"properties":{"value":{"type":"null"},"unit":{"type":"null"},"source_ids":{"maxItems":0}}}},
    {"if":{"properties":{"status":{"const":"conflict"}},"required":["status"]},
     "then":{"properties":{"value":{"type":"null"},"unit":{"type":"null"},"source_ids":{"minItems":2}}}}
  ]
}
```

<!-- /unit:e167127550c615cd7d -->

<!-- unit:78466aca1afccbf949 -->
<a id="u-78466aca1afccbf949"></a>

Root restrictions и conditional restrictions применяются одновременно. В ветке found тип string для unit не разрешает произвольную строку: root enum по-прежнему оставляет только две единицы. Минимум source IDs не проверяет, что документы действительно существуют и спорят. Нельзя ожидать от structural validator знания содержимого вашего document store.

<!-- /unit:78466aca1afccbf949 -->

<!-- unit:73c332c39144930cb4 -->
<a id="u-73c332c39144930cb4"></a>

В S5 сказано, что условие «при error требуется error_message» может не выражаться в JSON Schema. Как общее ограничение стандарта это неверно: есть conditional keywords и составные schemas. Практический generation backend может не поддержать их, поэтому следует разделять выразимость стандарта и поддержку конкретного implementation. Механика if/then/else описана в [JSON Schema conditional validation](https://json-schema.org/understanding-json-schema/reference/conditionals).

<!-- /unit:73c332c39144930cb4 -->

<!-- unit:5319c71f8df1dc344e -->
<a id="u-5319c71f8df1dc344e"></a>

### Два контура проверки на одном примере

<!-- /unit:5319c71f8df1dc344e -->

<!-- unit:4084a16769bd2be723 -->
<a id="u-4084a16769bd2be723"></a>

Документ D1 содержит «3 рабочих дня». Output `{"status":"found","value":3,"unit":"business_days","source_ids":["D1"]}` соответствует schema. Чтобы принять его, application проверяет: D1 был доступен и включён в evidence; нужный passage действительно содержит это значение; вопрос относится к тому же процессу и версии. Эти проверки могут быть deterministic частично, требовать model judge или ручной разметки частично.

<!-- /unit:4084a16769bd2be723 -->

<!-- unit:820b3394b66c2f782c -->
<a id="u-820b3394b66c2f782c"></a>

Output с `unit:"calendar_days"` тоже проходит schema, но противоречит источнику. Output со `source_ids:["D999"]` проходит schema, но lookup не подтверждает ID. Output с `value:30` проходит schema, но не подтверждается числом из документа. Output с D1 другой версии может быть factually faithful самому passage и всё равно не отвечать текущему вопросу. Разные контуры обнаруживают разные классы ошибок.

<!-- /unit:820b3394b66c2f782c -->

<!-- unit:90d8c0ffdf7e2e47c5 -->
<a id="u-90d8c0ffdf7e2e47c5"></a>

Для conflict две ссылки недостаточны: D1 мог говорить «3 рабочих дня», D2 —повторять те же условия, тогда конфликта нет. И наоборот, 3 calendar days и 3 business days не эквивалентны только из-за одинакового scalar. Сначала проверяют применимость источников, units и scope, затем сравнивают claims. Структура помогает хранить результат такого анализа, но не заменяет его.

<!-- /unit:90d8c0ffdf7e2e47c5 -->

<!-- unit:8a36eb28e0067dcfd2 -->
<a id="u-8a36eb28e0067dcfd2"></a>

Недоступный tool/search failure не следует кодировать как not_found в этом output. Not_found —результат успешно выполненной проверки evidence, error —невозможность её выполнить. Runtime envelope может содержать `execution_status` и transport error вне извлечённого content object. Это позволяет не превращать отсутствие наблюдения в отрицательный факт.

<!-- /unit:8a36eb28e0067dcfd2 -->

<!-- unit:01407bc3ae1c515c5e -->
<a id="u-01407bc3ae1c515c5e"></a>

### Строки, даты, числа и скрытая нормализация

<!-- /unit:01407bc3ae1c515c5e -->

<!-- unit:886bde67eb7eb187e5 -->
<a id="u-886bde67eb7eb187e5"></a>

В S6 date описан как string с `format:"date"`. В JSON Schema формат может быть annotation; в [python-jsonschema format checking включают явно](https://python-jsonschema.readthedocs.io/en/stable/validate/). Синтаксический pattern вида YYYY-MM-DD не проверяет существование 30 февраля. И даже существующая дата не доказывает, что «завтра» вычислено в нужной timezone/reference date.

<!-- /unit:886bde67eb7eb187e5 -->

<!-- unit:281e65059e18d1723f -->
<a id="u-281e65059e18d1723f"></a>

Для money важно объявить decimal representation. Двоичный float не представляет многие десятичные дроби точно: классический пример 0.1+0.2 отличается от точного 0.3 в binary floating arithmetic. Для integer minor units отдельно определяют валюту и допустимое число знаков; не всякий monetary contract следует упрощать до «всегда умножь на 100». Для точного decimal используют подходящий decimal/string contract и проверяемое преобразование.

<!-- /unit:281e65059e18d1723f -->

<!-- unit:3ff4bf295cac29a58c -->
<a id="u-3ff4bf295cac29a58c"></a>

Отсутствие type coercion делает некоторые ошибки заметнее: `"3"` остаётся string и отклоняется number-схемой. Coercion может быть намеренной preprocessing policy, но тогда нужно описать locale («1,234»), ranges и сохранение raw value. Пропустить непонятный символ, округлить число и затем объявить extraction безошибочным —плохая замена валидации.

<!-- /unit:3ff4bf295cac29a58c -->

<!-- unit:fec873ac1e24565ff5 -->
<a id="u-fec873ac1e24565ff5"></a>

## Исправления, retries и data provenance

<!-- /unit:fec873ac1e24565ff5 -->

<!-- unit:cd177917a1b1850c41 -->
<a id="u-cd177917a1b1850c41"></a>

При parse error можно bounded retry или запросить repair, но сохранять original output/error для diagnosis. «Починка JSON» не должна незаметно менять value. После repair заново применяются все semantic checks.

<!-- /unit:cd177917a1b1850c41 -->

<!-- unit:03bb305c103fc3221e -->
<a id="u-03bb305c103fc3221e"></a>

Для numeric extraction сохранять source span/ID и units. Для dates — declared timezone и reference date, не автоматически текущее время runtime. Для entity IDs — existence lookup; пользовательский display name не всегда уникальный ID.

<!-- /unit:03bb305c103fc3221e -->

<!-- unit:042f139dc1b201dcd8 -->
<a id="u-042f139dc1b201dcd8"></a>

При schema evolution old cached outputs могут стать несовместимыми. Версионировать schema и parsing rules, зафиксировать migration/defaults. Если добавили required field, старый output не становится корректным только от удачного JSON parse.

<!-- /unit:042f139dc1b201dcd8 -->

<!-- unit:88786c71a270f256a1 -->
<a id="u-88786c71a270f256a1"></a>

### Repair, re-ask и повторное извлечение решают разные задачи

<!-- /unit:88786c71a270f256a1 -->

<!-- unit:50a41cc263e35499d2 -->
<a id="u-50a41cc263e35499d2"></a>

Repair исправляет форму уже полученного кандидата: например, удаляет явно выделенный Markdown wrapper. Re-ask сообщает модели конкретное нарушение и запрашивает новый candidate. Повторное извлечение заново обращается к evidence, если проблема в выбранном значении. Эти действия имеют разные риски. Автоматически заменить 30 на 3 на основании «похоже на опечатку» —уже новое утверждение о содержимом, а не восстановление скобки.

<!-- /unit:50a41cc263e35499d2 -->

<!-- unit:6cde2f44ff330fa0e9 -->
<a id="u-6cde2f44ff330fa0e9"></a>

После любого изменения снова выполняются parse, schema и необходимые semantic checks. Для diagnosis сохраняют исходный candidate, classification ошибки и итоговую transformation. В UI разрешено показывать промежуточный preview, но downstream action должен получать только accepted record с понятным terminal status.

<!-- /unit:6cde2f44ff330fa0e9 -->

<!-- unit:d33a17d2909cc94797 -->
<a id="u-d33a17d2909cc94797"></a>

Если каждая независимая попытка проходит **проверяемый** контракт с вероятностью 0.9, максимум три попытки дают вероятность хотя бы одного прохождения $1-0.1^3=0.999$. Ожидаемое число попыток $1+0.1+0.01=1.11$: вторая нужна после первого fail, третья после двух fail. Это не означает 99.9% истинных ответов: validator мог пропускать semantic errors, а retries могли быть зависимыми. Систематически неверный source не исправляется независимым sampling той же hallucination.

<!-- /unit:d33a17d2909cc94797 -->

<!-- unit:212fcbfdd1fd7b7dd4 -->
<a id="u-212fcbfdd1fd7b7dd4"></a>

Latency по хвосту растёт даже при хорошей средней надёжности. Для одинаковой длительности T без overhead первые успехи занимают T, вторые 2T, третьи 3T, exhausted cases тоже 3T. Общий deadline должен ограничивать весь процесс, включая retrieval/validation и очереди, а не выдавать каждому retry новый полный deadline.

<!-- /unit:212fcbfdd1fd7b7dd4 -->

<!-- unit:f82e6a4b415e3aa0f2 -->
<a id="u-f82e6a4b415e3aa0f2"></a>

### Schema version — часть воспроизводимости

<!-- /unit:f82e6a4b415e3aa0f2 -->

<!-- unit:4707c76fd0004e29cf -->
<a id="u-4707c76fd0004e29cf"></a>

Пусть v1 хранила только `days`, а v2 требует `value`, `unit`, `source_ids`. Старая cache entry не стала новой schema после переименования поля. Если раньше unit была неизвестна, migration не вправе придумать calendar_days. Нужно либо подтверждённое преобразование с provenance, либо повторное извлечение, либо явный incompatible status.

<!-- /unit:4707c76fd0004e29cf -->

<!-- unit:b6a552b1c13009964b -->
<a id="u-b6a552b1c13009964b"></a>

Сохраняют schema revision, prompt/template revision, модель, источник/его версию и policy нормализации. Для сравнения моделей используют одинаковый output contract; иначе «качество повысилось» может означать лишь то, что новая версия перестала проверять сложные поля. Метрики считают на raw generation, parsed candidates и final accepted objects отдельно, чтобы repair/retries не исчезали из стоимости.

<!-- /unit:b6a552b1c13009964b -->

<!-- unit:a1d7abab79ffbc0ec8 -->
<a id="u-a1d7abab79ffbc0ec8"></a>

На 100 задачах возможны 98 parse successes, 94 schema successes и 86 правильных извлечений среди всех задач. Если эти множества вложены, schema success=94%, correctness=86%, correctness среди schema-valid=86/94≈91.49%. Сказать «94% точности извлечения» по числу прошедших schema нельзя. Если разрешён отказ, отдельно считают coverage и correctness среди ответов, чтобы система, всегда возвращающая unknown, не выглядела идеальным extractor.

<!-- /unit:a1d7abab79ffbc0ec8 -->

<!-- unit:ed470cf98ad2cd2ea6 -->
<a id="u-ed470cf98ad2cd2ea6"></a>

## Учебный разбор одного запроса

<!-- /unit:ed470cf98ad2cd2ea6 -->

<!-- unit:a4f88e37b521df874f -->
<a id="u-a4f88e37b521df874f"></a>

Документ D1: «Срок обработки —3 рабочих дня». Question: «Сколько календарных дней занимает обработка?» Простое extraction days3 с unit=calendar неправильно. Корректный output должен сохранить business_days и/или указать, что calendar duration зависит от calendar/start date. Формат не снимает смысловое различие.

<!-- /unit:a4f88e37b521df874f -->

<!-- unit:1042297baee122f010 -->
<a id="u-1042297baee122f010"></a>

Пусть contract требует `duration_days` без unit и unknown запрещён. Проблема в schema/task design, а не только в model quality. Добавление более сильной model не исправляет недоопределённый output specification.

<!-- /unit:1042297baee122f010 -->

<!-- unit:37ba26e77922e09926 -->
<a id="u-37ba26e77922e09926"></a>

### Почему business days нельзя механически переименовать

<!-- /unit:37ba26e77922e09926 -->

<!-- unit:79f16de29185c9f9b3 -->
<a id="u-79f16de29185c9f9b3"></a>

Предположим, старт —пятница, день старта не считается, рабочие дни только понедельник–пятница, праздников нет. Три рабочих дня —понедельник, вторник, среда; дедлайн через пять календарных дней. Если старт в понедельник, те же три рабочих дня заканчиваются в четверг, через три календарных дня. Поэтому scalar 3 сам по себе не задаёт calendar duration.

<!-- /unit:79f16de29185c9f9b3 -->

<!-- unit:5079a5a7f9e734d06f -->
<a id="u-5079a5a7f9e734d06f"></a>

Если исходник не определяет calendar, начало отсчёта и час окончания, extractor должен сохранить исходную единицу и границу знания. Это пример semantic contract, который сильнее regex/JSON shape. Не надо обучать модель всегда отвечать «5 календарных дней» по единственному примеру пятницы.

<!-- /unit:5079a5a7f9e734d06f -->

<!-- unit:5b0113208873d732c2 -->
<a id="u-5b0113208873d732c2"></a>

Проверяемый mini-experiment: сохранить original phrase, извлечь 3 business_days, затем отдельной deterministic функцией рассчитать deadline по объявленному calendar. Так видно, где language understanding, где арифметика календаря и где внешние данные о праздниках. Ни один шаг не считается уже реализованным пользовательским сервисом: это учебный разбор.

<!-- /unit:5b0113208873d732c2 -->

<!-- unit:e7a72d96229234fc3d -->
<a id="u-e7a72d96229234fc3d"></a>

## Вопросы и практика

<!-- /unit:e7a72d96229234fc3d -->

<!-- unit:ba09cc9af76e8112db -->
<a id="u-ba09cc9af76e8112db"></a>

1. Что гарантирует parser? **Ответ:** разбор структуры по syntax, не truth/authorization.
2. Устраняет ли tool fine-tuning необходимость validation? **Ответ:** нет, training не равен строгому runtime invariant.
3. Можно ли исполнять partial JSON из stream? **Ответ:** только по специально определённому incremental protocol; обычный незавершённый object нельзя считать готовым действием.

<!-- /unit:ba09cc9af76e8112db -->

<!-- unit:c208d7a182f7914ac7 -->
<a id="u-c208d7a182f7914ac7"></a>

4. Обязательное поле и nullable —одно и то же? **Ответ:** нет; required отвечает за наличие, null —за допустимое значение.
5. Если properties объявлены, запрещены ли другие поля? **Ответ:** не автоматически; это отдельные ограничения schema.
6. Если prefix можно завершить, уже ли он valid final object? **Ответ:** нет, нужны завершение и проверка terminal state.
7. Выражает ли стандарт зависимость полей? **Ответ:** да, например if/then/else; конкретный decoder может поддерживать subset.
8. Проверяет ли format date нужную бизнес-дату? **Ответ:** нет; сначала нужна включённая проверка формата, затем calendar/reference/timezone semantics.
9. Заменяет ли repair проверку смысла? **Ответ:** нет, после изменения candidate снова валидируют все нужные свойства.
10. Доказывает ли high schema success extraction accuracy? **Ответ:** нет, правильные поля могут содержать неверные факты.

<!-- /unit:c208d7a182f7914ac7 -->

<!-- unit:ef78c560c72a3820ff -->
<a id="u-ef78c560c72a3820ff"></a>

**Практика:** задать contract для «сумма из счёта»: value,currency,source,unknown. Написать три schema-valid semantic failures и rules, обнаруживающие их. Механика logits/sampling подробно в [инференсе](<../04-inference/infer-generation.md>).

<!-- /unit:ef78c560c72a3820ff -->

## Результат главы и переход дальше

Ответьте на вопрос: **Что гарантирует синтаксис результата и что ещё должно проверить приложение?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

Следующие темы: [Вызов инструментов и исполнение](<app-tools.md>).

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
