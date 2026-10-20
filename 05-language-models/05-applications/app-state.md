# Состояние, orchestration и граф управления

**Уровень:** основа ветки.

**Предпосылки:** [Вызов инструментов и исполнение](<app-tools.md>)

**Используют:** [RAG: BM25, dense, hybrid и multi-vector retrieval](<app-rag.md>), [Постановка обучения агента и baseline](<../06-agent-learning/agent-task.md>)

**Учебный вопрос:** Какие данные и переходы нужны для остановки, восстановления и повторного исполнения?





## Карта вычисления и границы этапов

![Состояние, orchestration и граф управления: входы, преобразования и выходы](<../../оформление/course/app-state.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Параметры модели при обычном выполнении фиксированы. Состояние запроса, cache и состояние внешнего инструмента имеют разные жизненные циклы; их нельзя смешивать с optimizer states.

<!-- unit:29f97ef075af516d52 -->
<a id="u-29f97ef075af516d52"></a>

## Какие компоненты добавляются вокруг модели

<!-- /unit:29f97ef075af516d52 -->

<!-- unit:1827a0b9cabc378855 -->
<a id="u-1827a0b9cabc378855"></a>

Один model call возвращает текст/структуру. Приложение должно получить user input, подготовить context, при необходимости найти документы, исполнить проверенные tools, сохранить state и завершить результат. Orchestration задаёт порядок/условия этих операций.

<!-- /unit:1827a0b9cabc378855 -->

<!-- unit:e4f0eeb76e7d12f5cb -->
<a id="u-e4f0eeb76e7d12f5cb"></a>

Linear chain подходит для фиксированного пути `retrieve→generate→validate`. Graph/state machine нужен, когда есть branches, retries, tool loop и early termination. Model weights от изменения graph не обновляются: runtime исполняет существующую policy, обучение policy требует отдельного objective/data.

<!-- /unit:e4f0eeb76e7d12f5cb -->

<!-- unit:380ae6ab0721c767d0 -->
<a id="u-380ae6ab0721c767d0"></a>

Пример state:

<!-- /unit:380ae6ab0721c767d0 -->

<!-- unit:fd06b2f794820a2f24 -->
<a id="u-fd06b2f794820a2f24"></a>

```text
question, authorized_scope, messages, evidence,
pending_calls, observations, attempts, budget, status.
```

<!-- /unit:fd06b2f794820a2f24 -->

<!-- unit:b3953bc5d69697fd82 -->
<a id="u-b3953bc5d69697fd82"></a>

Trusted constraints, например права на документы, задаются приложением. Их нельзя заменять модельным summary из chat history. Node получает нужные поля и возвращает определённые изменения; merge policy для parallel updates должна быть явной.

<!-- /unit:b3953bc5d69697fd82 -->

<!-- unit:ac89097a7583f9aeef -->
<a id="u-ac89097a7583f9aeef"></a>

**Схема курса:** request → state/router → model proposes tool or retrieval → validate/execute → observation/evidence → state update → stop, retry or answer

<!-- /unit:ac89097a7583f9aeef -->

<!-- unit:47dbb81efa8119603f -->
<a id="u-47dbb81efa8119603f"></a>

<!-- source-depth-inference-З -->

<!-- /unit:47dbb81efa8119603f -->

<!-- unit:3346e73caeeecefd4c -->
<a id="u-3346e73caeeecefd4c"></a>

### Три уровня, которые были смешаны в вопросах чата

<!-- /unit:3346e73caeeecefd4c -->

<!-- unit:1970072ca4035d4a53 -->
<a id="u-1970072ca4035d4a53"></a>

В S5 вопрос был конкретным: как использовать скачанную в Ollama модель для агента, циклов и инструментов? Сначала разделим веса модели, её endpoint и приложение. Веса задают обученную функцию предсказания. Inference backend загружает их, токенизирует вход, выполняет forward и decoding. Приложение хранит цель и историю, выбирает доступные операции и связывает последовательные обращения. Один продукт может объединять несколько уровней, но от этого их функции не становятся одинаковыми.

<!-- /unit:1970072ca4035d4a53 -->

<!-- unit:33b073254d4c6f4c4f -->
<a id="u-33b073254d4c6f4c4f"></a>

Путь одного решения выглядит так:

<!-- /unit:33b073254d4c6f4c4f -->

<!-- unit:0d26a9bb9fdc9a0c6b -->
<a id="u-0d26a9bb9fdc9a0c6b"></a>

```text
trusted task + current state + tool descriptions
                    ↓ serialization / chat template
           tokens → Transformer → next-token scores
                    ↓ sampling / constrained decoding
          assistant message or proposed tool calls
                    ↓ parser + application validation
        executor → real observation → updated state
                    └──────── next model request
```

<!-- /unit:0d26a9bb9fdc9a0c6b -->

<!-- unit:36949f2565dedb8005 -->
<a id="u-36949f2565dedb8005"></a>

Стрелка назад — работа runtime. Внутри одного forward нет скрытого Python-цикла, который автоматически обращается к поиску. После tool result выполняется новое обращение с обновлённым контекстом; backend может переиспользовать совместимый prefix KV cache. Это экономия вычислений, а не замена observation воображаемым результатом. Веса при каждом таком шаге остаются прежними.

<!-- /unit:36949f2565dedb8005 -->

<!-- unit:5f16e36800decff5aa -->
<a id="u-5f16e36800decff5aa"></a>

Workflow и agent различаются тем, кто задаёт следующий шаг. Для workflow разработчик заранее определил допустимые ветки, например «поиск → ответ → проверка». В agent loop модель может выбирать среди предложенных действий. В обоих случаях runtime определяет исполнимые переходы и пределы. Graph способен выразить оба варианта: наличие графа ещё не означает автономный выбор модели. Такую границу использует и [документация LangGraph о workflows и agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents).

<!-- /unit:5f16e36800decff5aa -->

<!-- unit:4682ef2f6e20c32fff -->
<a id="u-4682ef2f6e20c32fff"></a>

### State — больше, чем текст переписки

<!-- /unit:4682ef2f6e20c32fff -->

<!-- unit:73568485e64186af83 -->
<a id="u-73568485e64186af83"></a>

Messages нужны для model context. State нужен для корректного исполнения. Помимо текста, в нём могут находиться идентификатор задачи, версия данных, разрешённый scope, множество уже прочитанных документов, pending calls, результаты проверок, остаток бюджета и terminal status. Часть полей сериализуют для LLM, часть использует только программа. Пароль, объект подключения к БД и mutable client session не следует превращать в обычные слова prompt ради «полноты памяти».

<!-- /unit:73568485e64186af83 -->

<!-- unit:a2cc8a2052fea2bf60 -->
<a id="u-a2cc8a2052fea2bf60"></a>

Различаются три вида памяти. Working context — то, что модель видит сейчас. Durable state — сохранённое состояние выполнения, позволяющее продолжить после сбоя. Knowledge store — документы и индекс для retrieval. Сохранить checkpoint не означает обучить модель новым знаниям; заменить context summary не означает изменить документы; заново загрузить веса не означает восстановить потерянные результаты tools.

<!-- /unit:a2cc8a2052fea2bf60 -->

<!-- unit:035cfc76e6b5aae428 -->
<a id="u-035cfc76e6b5aae428"></a>

Особенно важна роль summary. Оно может сжать старые рассуждения и observations, но не должно самовольно менять права или статус операции. Например, фраза summary «доступ ко всем документам разрешён» не заменяет authoritative scope. Для фактов сохраняют IDs и versions, чтобы итог можно было проверить независимо от пересказа. Для pending external operation сохраняют её identity, а не только слова «кажется, сделали».

<!-- /unit:035cfc76e6b5aae428 -->

<!-- unit:2cb0f0c1d47eb3002b -->
<a id="u-2cb0f0c1d47eb3002b"></a>

### Graph как явная машина состояний

<!-- /unit:2cb0f0c1d47eb3002b -->

<!-- unit:a3a6bdcbcecd9db1ec -->
<a id="u-a3a6bdcbcecd9db1ec"></a>

В учебной машине есть состояния `READY`, `CALL_PENDING`, `OBSERVED`, `ANSWER_READY`, `DONE`, `FAILED`. Из READY модель предлагает call или answer. После проверки call переходит в CALL_PENDING; исполнитель создаёт observation, и управление возвращается к READY через OBSERVED. Candidate answer проходит проверку, после которой наступает DONE либо допустимая ветка исправления. Исчерпание бюджета переводит в FAILED/partial с объяснением, а не в фиктивный успех.

<!-- /unit:a3a6bdcbcecd9db1ec -->

<!-- unit:d2213fd7dcf38c0c05 -->
<a id="u-d2213fd7dcf38c0c05"></a>

Смысл state machine — сделать переходы проверяемыми. Из CALL_PENDING нельзя честно перейти в «tool succeeded», пока нет подтверждённого result. Из ANSWER_READY нельзя объявить наличие цитаты, если cited ID отсутствует в evidence. Программный validator может проверять эти условия даже тогда, когда модель уверенно пишет обратное.

<!-- /unit:d2213fd7dcf38c0c05 -->

<!-- unit:191ea8b297c68d9444 -->
<a id="u-191ea8b297c68d9444"></a>

Фиксированный pipeline тоже имеет state, хотя его иногда прячут в локальные переменные. Graph полезен при усложнении ветвлений, но не обязан заменять простую функцию из трёх шагов. Разница в явности control flow и persistence, а не в магической способности graph «думать лучше».

<!-- /unit:191ea8b297c68d9444 -->

<!-- unit:01a0280fb74795daba -->
<a id="u-01a0280fb74795daba"></a>

## Frameworks: abstraction по потребности

<!-- /unit:01a0280fb74795daba -->

<!-- unit:13be629bc2b102ff2c -->
<a id="u-13be629bc2b102ff2c"></a>

LangChain-style components дают templates, model/tool adapters, chains и parsers; LangGraph-style graph описывает state/transitions/checkpointing. LlamaIndex делает акцент на data/retrieval integrations. AutoGen/CrewAI исторически предлагали conversation/role coordination patterns. Их APIs меняются; выбирать нужен набор capabilities, а не обязательное прохождение каждого framework.

<!-- /unit:13be629bc2b102ff2c -->

<!-- unit:c1983dd0002bc777be -->
<a id="u-c1983dd0002bc777be"></a>

Один agent participant может иметь свой loop с tools, state и termination. Это не требует быть реализованным через LangGraph: тот же механизм можно написать другим runtime. Multi-agent coordination добавляет messages, duplicated context и termination difficulties; качество сравнивать с single-agent baseline при одинаковом total budget.

<!-- /unit:c1983dd0002bc777be -->

<!-- unit:5c359cd38136de7437 -->
<a id="u-5c359cd38136de7437"></a>

Локальные Ollama/LM Studio models могут быть model endpoints такого приложения. Ответ на исходный вопрос о локальном agent pipeline: runtime вызывает endpoint, валидирует output, исполняет tools и повторяет call. Встроенные features конкретного приложения могут часть этого делать, но supported API/tool modes проверяются отдельно.

<!-- /unit:5c359cd38136de7437 -->

<!-- unit:3f9101565a308548b6 -->
<a id="u-3f9101565a308548b6"></a>

### Ответ на вопрос об AutoGen и LangGraph

<!-- /unit:3f9101565a308548b6 -->

<!-- unit:856de0bc98ac94b0d6 -->
<a id="u-856de0bc98ac94b0d6"></a>

В S6 пользователь связал conversation-centric участников AutoGen с графами LangGraph. Совпадение общей идеи последовательных решений не означает зависимость одной реализации от другой. Participant может вызывать модель, исполнять tool, возвращать фиксированный ответ или передавать сообщение человеку. Graph node также может быть обычной deterministic функцией. Поэтому исходное «каждый агент во всех frameworks обязательно LLM + tools + loop» слишком широко.

<!-- /unit:856de0bc98ac94b0d6 -->

<!-- unit:d7d7b3c10699f0c619 -->
<a id="u-d7d7b3c10699f0c619"></a>

Есть два независимых уровня: локальная policy участника и coordinator, выбирающий, кому передать управление. Researcher может сам выбирать search/read tools, а coordinator — решать, когда передать его evidence автору ответа. Либо coordinator детерминирован, а участники только выполняют фиксированные функции. Названия «researcher», «critic», «writer» работают как описание роли, но не создают отдельного знания и не гарантируют независимость ошибок.

<!-- /unit:d7d7b3c10699f0c619 -->

<!-- unit:d08880c969763afdff -->
<a id="u-d08880c969763afdff"></a>

Вопрос о самостоятельном выборе участников разбирается на этом же уровне: делегирование может быть ещё одним разрешённым action с ограниченным входом, бюджетом и result contract. Из факта, что интерфейс показывает subagent, нельзя восстановить закрытые внутренние prompts, framework и правила маршрутизации конкретного продукта. Здесь объясняется механизм, без утверждения о внутреннем устройстве упомянутого в разбираемом примере редактора.

<!-- /unit:d08880c969763afdff -->

<!-- unit:e71a80f016263b2453 -->
<a id="u-e71a80f016263b2453"></a>

### Parallel branches: latency, state merge и общий бюджет

<!-- /unit:e71a80f016263b2453 -->

<!-- unit:460532c61582dc7c15 -->
<a id="u-460532c61582dc7c15"></a>

Для двух независимых поисков длительностью 0.8 и 1.2 секунды последовательный путь занимает 2.0, идеальный parallel —1.2 до merge, без overhead/queue. Но если второй запрос строится по результату первого, такого ускорения нет. Parallel execution уменьшает критический путь, а не сумму оплаченной/выполненной работы.

<!-- /unit:460532c61582dc7c15 -->

<!-- unit:711f808b0998623781 -->
<a id="u-711f808b0998623781"></a>

Пусть начальное evidence множество `{a}`, ветка A добавляет b, ветка B —c. Если обе возвращают полное replacement-поле, last-write-wins сохранит `{a,b}` или `{a,c}` и потеряет другой результат. Объединение множеств даст `{a,b,c}` независимо от порядка. Для evidence с versions удобнее merge по стабильному ID/revision и явная политика конфликтов. Для messages простая конкатенация сохраняет всё, но порядок A+B отличается от B+A; порядок может повлиять на следующий model call.

<!-- /unit:711f808b0998623781 -->

<!-- unit:bd0c62a26e92dc1710 -->
<a id="u-bd0c62a26e92dc1710"></a>

Две ветки не должны независимо тратить весь общий budget. При лимите 6000 tokens coordinator может зарезервировать 3000 и 2000 для двух участников, 1000 для synthesis. Это 6000 общего лимита, а не 6000 на каждого. Если участник использует больше model calls, повторённые system/tool/history tokens тоже учитываются по объявленному accounting; параллельность их не устраняет. Учебное сравнение с single-agent имеет смысл при сопоставимых data access и суммарном бюджете.

<!-- /unit:bd0c62a26e92dc1710 -->

<!-- unit:0f93f2cda904cee535 -->
<a id="u-0f93f2cda904cee535"></a>

### Локальная модель: совместимость проверяется по слоям

<!-- /unit:0f93f2cda904cee535 -->

<!-- unit:4c916a559518aa716f -->
<a id="u-4c916a559518aa716f"></a>

Скачанный checkpoint сам по себе не задаёт работающий агент. Нужны подходящие tokenizer/chat template, способность model следовать tool protocol, поддержка backend и parser, затем executor и loop. Ошибка на каждом слое проявляется по-разному: raw text вместо calls, испорченные argument types, исчезнувший tool result, неверная role serialization, повтор вызова после уже полученного результата.

<!-- /unit:4c916a559518aa716f -->

<!-- unit:90ae0534064d81c145 -->
<a id="u-90ae0534064d81c145"></a>

Нельзя переносить из разбираемого примера абсолютное «Ollama/LM Studio только генерируют, никаких loops у них нет». В документации [Ollama показан multi-turn tool loop](https://docs.ollama.com/capabilities/tool-calling); [LM Studio разделяет tool requests и исполнение в коде](https://lmstudio.ai/docs/developer/openai-compat/tools), а его [SDK .act() автоматизирует несколько rounds](https://lmstudio.ai/docs/typescript/agent/act). Где физически находится loop — в вашем коде, SDK или встроенной функции продукта — зависит от выбранного интерфейса. Это не меняет различия между model proposal и реальным effect.

<!-- /unit:90ae0534064d81c145 -->

<!-- unit:03b1a1381e98303be3 -->
<a id="u-03b1a1381e98303be3"></a>

Совместимый HTTP shape облегчает замену endpoint, но не гарантирует одинаковые supported schema, tool-choice modes, streaming events и качество. Начинать проверку разумно с read-only инструмента известного результата: предложен ли правильный call, распарсены ли arguments, реальный ли result возвращён в контекст, использован ли он в final. Нельзя лечить ошибку chat template добавлением ещё пяти агентов.

<!-- /unit:03b1a1381e98303be3 -->

<!-- unit:dcf68836e60adc0d90 -->
<a id="u-dcf68836e60adc0d90"></a>

## Chain, branch и loop

<!-- /unit:dcf68836e60adc0d90 -->

<!-- unit:6c5a9b54862b9d76b0 -->
<a id="u-6c5a9b54862b9d76b0"></a>

Chain — фиксированная последовательность stages. Branch выбирает дальнейший путь по state. Loop повторяет часть работы до condition. В Python это можно выразить functions/if/while; frameworks добавляют integrations, checkpointing, streaming и удобное описание transitions.

<!-- /unit:6c5a9b54862b9d76b0 -->

<!-- unit:91bd36da34995c3566 -->
<a id="u-91bd36da34995c3566"></a>

Пример documentation assistant:

<!-- /unit:91bd36da34995c3566 -->

<!-- unit:52fd71a34a04dcafb0 -->
<a id="u-52fd71a34a04dcafb0"></a>

```text
prepare → retrieve → inspect evidence
                    ├ enough → answer → validate → end
                    ├ retry allowed → rewrite query → retrieve
                    └ no evidence/budget → abstain → end.
```

<!-- /unit:52fd71a34a04dcafb0 -->

<!-- unit:ce460210ff967bf63d -->
<a id="u-ce460210ff967bf63d"></a>

Каждый arrow должен иметь condition. Если «model решает, искать ещё или хватит», в state должны быть observations, remaining budget и criterion enough evidence. Prompt «будь умным» не задаёт bounded algorithm.

<!-- /unit:ce460210ff967bf63d -->

<!-- unit:1b2b7960af7acc4885 -->
<a id="u-1b2b7960af7acc4885"></a>

**Схема курса:** initial state → selected node → state transition → tool/worker result or error → reducer/checkpoint → route/stop condition → observable final state

<!-- /unit:1b2b7960af7acc4885 -->

<!-- unit:de8f3cbbba0e1d3f18 -->
<a id="u-de8f3cbbba0e1d3f18"></a>

<!-- source-depth-applications-В -->

<!-- /unit:de8f3cbbba0e1d3f18 -->

<!-- unit:108896055bba2c1665 -->
<a id="u-108896055bba2c1665"></a>

### Что означает stateless и почему «чистая функция» —упрощение

<!-- /unit:108896055bba2c1665 -->

<!-- unit:04fd6b5120a1604a3c -->
<a id="u-04fd6b5120a1604a3c"></a>

В S5 два обращения разделены во времени: сначала «меня зовут Алексей», затем «как меня зовут?». Если второй input не содержит первого факта и нет внешнего memory mechanism, модель не получает основания восстановить имя именно этого собеседника. Проблема не в пяти минутах как сроке забывания: нужные данные просто не переданы в вычисление.

<!-- /unit:04fd6b5120a1604a3c -->

<!-- unit:a4c5e02fe9addc85a9 -->
<a id="u-a4c5e02fe9addc85a9"></a>

Для понимания удобно писать $y\sim p_\theta(\cdot\mid x)$: input x содержит доступный context, θ —веса. Это точнее, чем объявлять каждый API call математически чистой детерминированной функцией f(prompt). Sampling может быть stochastic, provider может хранить conversation state, backend —KV cache, а дополнительные tools/runtime —свои данные. При этом in-context память не равна изменению θ. Сохранённое имя может попасть в следующий prompt без fine-tuning.

<!-- /unit:a4c5e02fe9addc85a9 -->

<!-- unit:0fa9aaf1adedc1179d -->
<a id="u-0fa9aaf1adedc1179d"></a>

Нужен явный ответ на вопрос «что увидит следующий вызов»: весь transcript, последние сообщения, summary, найденные memory records или их смесь. Если API предоставляет server-side conversation, ответственность за выбор conversation ID и проверку изоляции остаётся у приложения. Если история хранится только в локальном Python list, restart процесса её потеряет.

<!-- /unit:0fa9aaf1adedc1179d -->

<!-- unit:ca120ab78c3713eafc -->
<a id="u-ca120ab78c3713eafc"></a>

### Пять проблем исходника: что исправляет orchestration

<!-- /unit:ca120ab78c3713eafc -->

<!-- unit:b52d79083fa2ce10fa -->
<a id="u-b52d79083fa2ce10fa"></a>

История добавляет информацию из прошлых ходов; tool execution добавляет внешние наблюдения; validator даёт критерий проверки; control flow связывает необходимые операции; error policy выбирает допустимое восстановление. Каждое средство решает свою проблему. Добавленная история не гарантирует память о всех деталях при truncation. Добавленный critic не становится источником истины. Новый graph не делает устаревший corpus актуальным.

<!-- /unit:b52d79083fa2ce10fa -->

<!-- unit:630fb4bc92aeb07953 -->
<a id="u-630fb4bc92aeb07953"></a>

Исходный пример интеграла $\int_0^\pi\sin x\,dx=2$ можно решить аналитически: первообразная $-\cos x$, значение $-\cos\pi+\cos0=2$. Поэтому фраза «для этого непременно нужен Python» неверна. Инструмент нужен, когда выбранная задача требует проверяемого исполнения/наблюдения; он полезен как внешний oracle, но нельзя объявлять каждую простую математическую задачу недоступной language model.

<!-- /unit:630fb4bc92aeb07953 -->

<!-- unit:56d925a6971f728cc8 -->
<a id="u-56d925a6971f728cc8"></a>

В другом примере $247\cdot891=220077$. Проверка calculator различает реальный результат и plausible text. Однако если модель неправильно переписала задачу в выражение, точный calculator вернёт точный ответ на неверный вопрос. Надёжность получается из правильной постановки, выполнения и использования результата, а не из одного факта наличия tool.

<!-- /unit:56d925a6971f728cc8 -->

<!-- unit:3ef906d86e46377864 -->
<a id="u-3ef906d86e46377864"></a>

### От wrapper к graph без выдуманной лестницы возможностей

<!-- /unit:3ef906d86e46377864 -->

<!-- unit:b7ff67318021b01067 -->
<a id="u-b7ff67318021b01067"></a>

Source wrapper сохранял user message, вызывал модель и дописывал assistant answer. Pipeline затем генерировал вопросы по теме «Трансформеры в NLP», искал ответы и синтезировал отчёт. Это хорошие иллюстрации разных control flows. Но pipeline в обычном Python уже содержит `for`: утверждение «на этом уровне нет циклов» противоречит самому примеру. Один tool response также может содержать несколько calls; их число не определяется ярлыком «уровень 2».

<!-- /unit:b7ff67318021b01067 -->

<!-- unit:d60209b89bb1c4daa2 -->
<a id="u-d60209b89bb1c4daa2"></a>

Если source предлагает разбивать список вопросов через `split("\n")`, нужно учесть заголовки, пустые строки и многострочный вопрос. Для интерфейса «ровно пять вопросов» естественнее отдельная структурная проверка списка. Затем оценивают содержательную полноту: пять повторений одного вопроса не покрывают пять аспектов Transformer. То же касается отчёта: много stages не доказывает, что в него попали механизмы attention, обучения и CV variants.

<!-- /unit:d60209b89bb1c4daa2 -->

<!-- unit:c0622a20b22db27110 -->
<a id="u-c0622a20b22db27110"></a>

Chain как композиция функций не запрещает ветвление внутри функции. Graph делает переходы и state явными, помогает persistence/inspection, но не создаёт новую вычислимость. «В chains практически невозможно» из сравнительных таблиц следует читать как критику удобства конкретного pattern/API в той версии, а не математический запрет. Современные framework boundaries проверяются отдельно от исторического учебного snippet.

<!-- /unit:c0622a20b22db27110 -->

<!-- unit:ba2772098e43fe870a -->
<a id="u-ba2772098e43fe870a"></a>

### Routing: decision модели и переход runtime

<!-- /unit:ba2772098e43fe870a -->

<!-- unit:5581c1cea63a70f154 -->
<a id="u-5581c1cea63a70f154"></a>

В support-примере запрос классифицируется как billing, technical или complaint, затем выбирается соответствующий путь. Classifier может быть deterministic rule или LLM. Router получает результат и проверяет, что label допустим. Если label неизвестен, нужен fallback/unknown branch; иначе typo станет попыткой перейти в несуществующий node.

<!-- /unit:5581c1cea63a70f154 -->

<!-- unit:ced1d86118717a8bf6 -->
<a id="u-ced1d86118717a8bf6"></a>

Model confidence вроде 0.9 не является автоматически calibrated вероятностью правильности. В примере «если уверена —ответь, иначе ищи» нужно определить, как confidence измеряется и почему ей можно доверять. Более наблюдаемое условие —есть ли source нужной версии, покрыты ли обязательные поля и пройдены ли validators. И эти признаки не обещают всеведения: доступный source может содержать ошибку.

<!-- /unit:ced1d86118717a8bf6 -->

<!-- unit:cc8bedee6b6718713f -->
<a id="u-cc8bedee6b6718713f"></a>

Во всех вариантах разделяются proposed route, проверка constraints и accepted route. Модель может предложить дальнейший поиск; runtime ограничивает допустимый scope и remaining budget. Модель может предложить finish; runtime решает, достаточно ли этого для terminal success по заданной задаче.

<!-- /unit:cc8bedee6b6718713f -->

<!-- unit:bb2cfa37f3b7f5b2d7 -->
<a id="u-bb2cfa37f3b7f5b2d7"></a>

## State и ответственность узлов

<!-- /unit:bb2cfa37f3b7f5b2d7 -->

<!-- unit:02c906756a129a7af0 -->
<a id="u-02c906756a129a7af0"></a>

State хранит messages, original request, authorization scope, document IDs/versions, observations, pending calls, attempts и result. Не все поля должны целиком копироваться в каждый model prompt: это снижает cost и риск смешать trusted constraints с untrusted text.

<!-- /unit:02c906756a129a7af0 -->

<!-- unit:3aa43a225f8369bbbc -->
<a id="u-3aa43a225f8369bbbc"></a>

Node contract задаёт, что читается и что обновляется. Если два parallel nodes добавляют evidence, reducer должен объединить по IDs/version, а не оставить последний список. Для scalar budget надо учитывать суммарный расход, а не minimum/last update случайно.

<!-- /unit:3aa43a225f8369bbbc -->

<!-- unit:9d1c7f7ec3359b6146 -->
<a id="u-9d1c7f7ec3359b6146"></a>

Original user constraints полезно хранить отдельно от model summary. Если summary потерял «только документы 2025 года», retrieval не должен расширить scope молча. Structured trusted state задаёт invariant, model-generated text — содержимое для обработки.

<!-- /unit:9d1c7f7ec3359b6146 -->

<!-- unit:f7c5e0b590fc068b71 -->
<a id="u-f7c5e0b590fc068b71"></a>

### TypedDict —описание, а не автоматическая защита

<!-- /unit:f7c5e0b590fc068b71 -->

<!-- unit:6be8245d4d67f54eda -->
<a id="u-6be8245d4d67f54eda"></a>

TypedDict помогает static checker обнаружить неверное имя поля и согласовать интерфейсы функций. Обычный Python dict не начинает проверять types при каждом присваивании только из-за аннотации. Если из внешнего JSON пришла строка вместо iteration integer, необходим runtime validator. И даже корректный integer -5 может нарушать invariant неотрицательного счётчика.

<!-- /unit:6be8245d4d67f54eda -->

<!-- unit:385bae59e5478609ad -->
<a id="u-385bae59e5478609ad"></a>

Node contract полезно описать через preconditions и postconditions. Retrieve читает original_question, scope, corpus_revision и search budget; возвращает observation с call ID, найденные references и расход. Generate читает подготовленный context и возвращает candidate. Validate читает candidate и evidence, выдаёт результат проверки. Authorized scope не должен становиться свободным output поле model node.

<!-- /unit:385bae59e5478609ad -->

<!-- unit:0cfb8180ffb09516e3 -->
<a id="u-0cfb8180ffb09516e3"></a>

Не стоит хранить произвольные live objects внутри durable state без serialization contract: dataframe, открытый file handle и GPU engine не обязательно восстанавливаются из checkpoint. Для большого artifact хранят stable ID/revision и metadata; данные читают по контролируемому path. Это особенно важно для source примеров с embeddings и cluster CSV.

<!-- /unit:0cfb8180ffb09516e3 -->

<!-- unit:8427ed8144f7c9d2b4 -->
<a id="u-8427ed8144f7c9d2b4"></a>

### Delta и reducer: два соглашения нельзя смешивать

<!-- /unit:8427ed8144f7c9d2b4 -->

<!-- unit:f17d2f2d2d92dd7c7d -->
<a id="u-f17d2f2d2d92dd7c7d"></a>

Пусть messages уже равны `[u1,a1]`, node получил новый a2. При replacement policy он возвращает полный `[u1,a1,a2]`. При append policy достаточно `[a2]`. Если append reducer получает полный старый список вместе с новым, получится `[u1,a1,u1,a1,a2]`. История продублирована, token cost вырос, прежние фразы могут получить лишний вес в model context.

<!-- /unit:f17d2f2d2d92dd7c7d -->

<!-- unit:deb1699eb455a96b3d -->
<a id="u-deb1699eb455a96b3d"></a>

Реальная message-aware merge policy может использовать IDs: новый ID добавляется, существующий заменяется. Тогда повторное включение того же ID не обязано дублировать сообщение. Но если при каждом replay старому содержимому выдают новые IDs, семантический дубль снова выглядит новым. Идентичность message, содержимого и business operation различаются.

<!-- /unit:deb1699eb455a96b3d -->

<!-- unit:797defcb376d53830f -->
<a id="u-797defcb376d53830f"></a>

В [Graph API LangGraph](https://docs.langchain.com/oss/python/langgraph/graph-api) default updates заменяют значение поля; пользовательский reducer определяет иное слияние, а add_messages учитывает IDs. Это причина читать конкретную схему state перед изменением node return. `Annotated` само по себе не выбирает правильный merge: приложение/runtime должно интерпретировать указанную функцию.

<!-- /unit:797defcb376d53830f -->

<!-- unit:27b5fc1012b9d5b558 -->
<a id="u-27b5fc1012b9d5b558"></a>

### Dictionary union, evidence identity и расход

<!-- /unit:27b5fc1012b9d5b558 -->

<!-- unit:544b29bfec1f1e0bd8 -->
<a id="u-544b29bfec1f1e0bd8"></a>

Исходник предлагает `operator.or_` для словаря analysis_results. При разных keys это объединяет записи. При одинаковом key правая запись заменяет левую. Если результат хранится под именем `run_statistical_test`, второй тест затрёт первый, даже если параметры и data revision различаются. Для истории результатов лучше stable call/result ID и metadata аргументов; сводка по tool name —отдельное представление.

<!-- /unit:544b29bfec1f1e0bd8 -->

<!-- unit:bea20528d3200ab8db -->
<a id="u-bea20528d3200ab8db"></a>

Merge по document ID тоже недостаточен, если нужно различать версии. `(doc_id,revision)` сохраняет два observations, а policy выбирает применимую версию. Молчаливое «последний arrival победил» не означает «самая свежая версия победила». Order доставки и порядок изменения источника —разные часы.

<!-- /unit:bea20528d3200ab8db -->

<!-- unit:acbe2b664e56434ac0 -->
<a id="u-acbe2b664e56434ac0"></a>

Budget accounting удобно строить на events расхода с уникальным usage ID. Две ветки потратили 300 и 500 tokens: общий расход 800. Если обе прочитали remaining=1000 и независимо записали остатки 700 и 500, replacement потеряет расход одной ветки. Если один usage event доставлен повторно, простое суммирование начислит его дважды; dedupe по event identity и сумма уникальных расходов дают другой invariant.

<!-- /unit:acbe2b664e56434ac0 -->

<!-- unit:204b5f5a749850c755 -->
<a id="u-204b5f5a749850c755"></a>

### Разбираем ошибки первого StateGraph snippet

<!-- /unit:204b5f5a749850c755 -->

<!-- unit:dbd50f631a41130d25 -->
<a id="u-dbd50f631a41130d25"></a>

В S5 node `call_model` и `execute_tool` зарегистрированы, route возвращает строку `end`, затем добавляется edge из `end` к END. Однако сам node `end` не зарегистрирован. Нужен прямой mapping label→END либо реальный зарегистрированный terminal node. Label результата routing не следует автоматически считать существующим graph node.

<!-- /unit:dbd50f631a41130d25 -->

<!-- unit:d590af5bf35e631594 -->
<a id="u-d590af5bf35e631594"></a>

Учебная запись mapping, без обещания совместимости с любой версией SDK:

<!-- /unit:d590af5bf35e631594 -->

<!-- unit:fef9d557306d496547 -->
<a id="u-fef9d557306d496547"></a>

```python
END = "__end__"  # учебный sentinel, в реальном SDK берётся его END
route_targets = {"execute": "execute_tool", "final": "validate_answer", "stop": END}

def route_after_model(response, remaining_tool_calls):
    calls = response.get("tool_calls", [])
    if calls:
        return "stop" if len(calls) > remaining_tool_calls else "execute"
    return "final"
```

<!-- /unit:fef9d557306d496547 -->

<!-- unit:584aa840810d24e397 -->
<a id="u-584aa840810d24e397"></a>

Если calls нет, candidate идёт на проверку ответа, а не в executor. Если предложено два calls, а осталось одно разрешённое исполнение, выбранная policy останавливает этот proposal; другое приложение может разрешить correction, но должно сохранить корректный протокол. Пропустить неизвестно какие calls молча —плохая замена явной policy.

<!-- /unit:584aa840810d24e397 -->

<!-- unit:cad394e8b36004cd88 -->
<a id="u-cad394e8b36004cd88"></a>

Source executor брал `tool_calls[0]`. При нескольких calls остальные теряются, а следующий model context может ждать их результатов. Либо framework действительно ограничивает proposal одним call, либо executor обрабатывает весь список с correlation IDs и определёнными зависимостями. Декларация list в API не означает, что код по первому элементу корректен для всех случаев.

<!-- /unit:cad394e8b36004cd88 -->

<!-- unit:2c6625984868a08fc6 -->
<a id="u-2c6625984868a08fc6"></a>

### Аналитический пример: отдельные баги и содержательные критерии

<!-- /unit:2c6625984868a08fc6 -->

<!-- unit:9d000326319cef50c5 -->
<a id="u-9d000326319cef50c5"></a>

В функции create_visualization из исходника filename вычисляется через `state['visualizations']`, хотя state не передан и не задан как локальная переменная. Это не graph feature, а неявная зависимость/ошибка области видимости. Имя artifact лучше выдаёт контролируемый executor, включающий run/call identity, а результат возвращает reference. Количество записей списка само по себе не защищает от collision двух parallel writers.

<!-- /unit:9d000326319cef50c5 -->

<!-- unit:d357699842dc29d6e5 -->
<a id="u-d357699842dc29d6e5"></a>

Source completion эвристика ищет в ответе слова «результат», «вывод», «итог». Строка «результат пока не получен» удовлетворяет такой проверке, хотя сообщает о незавершённости. В другом случае свободный текст без keywords направляется к execute_tool, а у него может не оказаться tool_calls вообще. Routing сначала должен различать фактический call и candidate answer, затем содержательно проверять результат.

<!-- /unit:d357699842dc29d6e5 -->

<!-- unit:856e8f27f7b55df496 -->
<a id="u-856e8f27f7b55df496"></a>

Поле iteration в state не ограничивает работу, если ни один обязательный cycle path его не увеличивает. Source snippets меняют счётчик в разных местах; их нельзя объединять в надежду, что одно название поля гарантирует cap. Нужно проследить каждое back edge и проверить расход лимита/условие termination. Recursion/step limit runtime —дополнительный предохранитель, а не объяснение semantic success.

<!-- /unit:856e8f27f7b55df496 -->

<!-- unit:e311ea67b7ef6c521f -->
<a id="u-e311ea67b7ef6c521f"></a>

## Checkpoint и resume

<!-- /unit:e311ea67b7ef6c521f -->

<!-- unit:504bd492a77fb845b5 -->
<a id="u-504bd492a77fb845b5"></a>

Checkpoint сохраняет состояние workflow после определённых transitions, чтобы восстановить работу. Но saved messages не автоматически доказывают, какие external tools действительно были executed. В state нужны call IDs/statuses и durable result provenance.

<!-- /unit:504bd492a77fb845b5 -->

<!-- unit:5f0e16a26f9a9e54b5 -->
<a id="u-5f0e16a26f9a9e54b5"></a>

Crash scenario: mutation выполнилась, process упал до сохранения observation. Resume из previous checkpoint может повторить operation. Поэтому checkpointing сам по себе не гарантирует exactly-once side effects; idempotency/status reconciliation остаются задачами execution layer.

<!-- /unit:5f0e16a26f9a9e54b5 -->

<!-- unit:ad4a5f7a431d409a5a -->
<a id="u-ad4a5f7a431d409a5a"></a>

Для read-only учебного поиска repeat безопаснее, но results могут измениться при новой версии corpus. Воспроизводимый replay использует saved observations/snapshot или явно отмечает, что retrieval выполнен заново по другим данным. Равенство graph topology не означает равенство inputs.

<!-- /unit:ad4a5f7a431d409a5a -->

<!-- unit:afddc9ff39681bce2b -->
<a id="u-afddc9ff39681bce2b"></a>

### История, checkpoint и snapshot источников

<!-- /unit:afddc9ff39681bce2b -->

<!-- unit:7abd8192db6bdbda7b -->
<a id="u-7abd8192db6bdbda7b"></a>

Transcript хранит разговор. Checkpoint хранит workflow state и нужную runtime информацию по контракту checkpointer. Corpus snapshot фиксирует, по каким документам искали. Model artifact фиксирует веса/adapter/tokenizer/template. Эти четыре объекта не взаимозаменяемы. Воспроизвести graph по одному transcript бывает невозможно, если из него не видно scope и pending calls.

<!-- /unit:7abd8192db6bdbda7b -->

<!-- unit:a1a4e48a4f8b9643cb -->
<a id="u-a1a4e48a4f8b9643cb"></a>

В source checkpointer example первый message —только «Привет», второй —«Как меня зовут?». Сохранённая история не содержит имени. Правильный resume должен помнить приветствие, но не выдумывать Алексея из другого учебного примера. Persistence сохраняет переданные факты; она не создаёт отсутствующие сведения.

<!-- /unit:a1a4e48a4f8b9643cb -->

<!-- unit:af4b6f5e6c39023773 -->
<a id="u-af4b6f5e6c39023773"></a>

In-memory saver живёт в памяти процесса и теряется вместе с ним. Durable backend сохраняет данные дольше, но его guarantees зависят от transactions, storage и configuration. `thread_id` выбирает историю выполнения; это не универсально безопасная замена user ID. Один пользователь может иметь несколько независимых задач, а ошибочное повторное использование чужого ID смешает histories, если application access layer этого не предотвращает.

<!-- /unit:af4b6f5e6c39023773 -->

<!-- unit:292e5f37bc1772471d -->
<a id="u-292e5f37bc1772471d"></a>

### Crash между effect и checkpoint

<!-- /unit:292e5f37bc1772471d -->

<!-- unit:bdab6fc0a8bb970d93 -->
<a id="u-bdab6fc0a8bb970d93"></a>

Представим шаг `create_report` с intent R1. Внешний сервис уже создал report Q7, затем процесс упал до сохранения observation. Последний checkpoint не содержит Q7. Resume повторяет node. Без idempotency/reconciliation возможен второй report. Checkpointer знает сохранённые transitions, а не всё, что успело произойти в другой системе.

<!-- /unit:bdab6fc0a8bb970d93 -->

<!-- unit:2933d72faa58fb3e7a -->
<a id="u-2933d72faa58fb3e7a"></a>

Для read-only lookup повтор тоже не всегда означает одинаковый observation: corpus мог обновиться с v7 до v8. Replay по saved results повторяет прежний evidence; rerun по live source получает новый. Оба режима допустимы, но выводы о воспроизводимости различаются. Чтобы сравнивать routing policies, удобно фиксировать observations; чтобы проверить актуальную систему, наоборот, нужен объявленный live input.

<!-- /unit:2933d72faa58fb3e7a -->

<!-- unit:c51ecd0b77d2e8a129 -->
<a id="u-c51ecd0b77d2e8a129"></a>

Внешний effect и checkpoint можно координировать только с учётом реальных transaction boundaries. Один local lock не делает две разные storage systems одной transaction. Стабильный operation key и status lookup обсуждаются в [главе tool execution](<app-tools.md>); повторение network call не обязано повторять business effect при поддержанном server contract.

<!-- /unit:c51ecd0b77d2e8a129 -->

<!-- unit:3d292b16225d9328c2 -->
<a id="u-3d292b16225d9328c2"></a>

### Interrupt/resume не равен паузе Python stack на любой строке

<!-- /unit:3d292b16225d9328c2 -->

<!-- unit:5db845d2805dfb4527 -->
<a id="u-5db845d2805dfb4527"></a>

В S5 смешаны static interrupt_before и вызов dynamic interrupt внутри node. Это разные точки остановки. Static breakpoint ставится относительно node; dynamic interrupt передаёт payload и получает resume value по протоколу runtime. Нельзя автоматически ожидать, что значение `resume="yes"` относится к ещё не достигнутому внутреннему interrupt только потому, что граф остановили перед node.

<!-- /unit:5db845d2805dfb4527 -->

<!-- unit:34596ad34d443fcb29 -->
<a id="u-34596ad34d443fcb29"></a>

В [LangGraph interrupt documentation](https://docs.langchain.com/oss/python/langgraph/interrupts) явно указано: при resume соответствующий node исполняется с начала, а не продолжает обычный Python stack с замороженной строки. Значит, код до interrupt может выполниться повторно. Его нужно делать безопасным для replay либо выносить эффекты в подходящие отдельные шаги с нужными guarantees. Порядок нескольких interrupts внутри node также важен для сопоставления resume values.

<!-- /unit:34596ad34d443fcb29 -->

<!-- unit:fe75bcacaa275a7467 -->
<a id="u-fe75bcacaa275a7467"></a>

Учебная трасса: `append_log(); interrupt(); create_report()`. Если append_log не idempotent, один resume добавит вторую log entry до продолжения. Если create_report стоит до interrupt, при replay может повториться и effect. Не надо путать эту особенность runtime с правилом, будто любая функция Python автоматически возобновляема после закрытия процесса.

<!-- /unit:fe75bcacaa275a7467 -->

<!-- unit:f008ca61cc293df1d7 -->
<a id="u-f008ca61cc293df1d7"></a>

Не каждую учебную операцию требуется ставить на human approval. Внешний input node нужен там, где задача действительно зависит от missing information/решения человека или выбранной policy. В нашем конспекте разбирается механизм, а не вводится новая процедура согласования текущего редактирования материалов.

<!-- /unit:f008ca61cc293df1d7 -->

<!-- unit:3d751159716e3afdef -->
<a id="u-3d751159716e3afdef"></a>

### Subgraph как модуль с интерфейсом

<!-- /unit:3d751159716e3afdef -->

<!-- unit:97e3be3198e139747e -->
<a id="u-97e3be3198e139747e"></a>

Parent state может содержать request, scope и общий budget; analysis subgraph —локальный план и результаты. Передача всего parent dict дочернему module не всегда нужна. Явный mapping входов/выходов уменьшает неоднозначность: child получает разрешённый dataset ID и лимит, возвращает references, usage events и status. Он не должен незаметно заменять parent authorization.

<!-- /unit:97e3be3198e139747e -->

<!-- unit:f9f696df7095dac467 -->
<a id="u-f9f696df7095dac467"></a>

Счётчики child и parent имеют разные единицы. Если parent допускает 3 child invocations, а каждый child делает до 5 model calls, это уже до 15 calls внутри children без synthesis. «Всего три шага» по верхнему graph не означает три обращения к модели. Глобальный accounting должен учитывать вложенную работу и не выдавать каждому child новый полный global budget.

<!-- /unit:f9f696df7095dac467 -->

<!-- unit:6d9c58596aa785fc0b -->
<a id="u-6d9c58596aa785fc0b"></a>

При failure child важно сохранить, какие outcomes подтверждены, а какие нет. Parent не должен превращать child partial в полный success только потому, что module вернул строку. Для реального replay subgraph действует контракт конкретного runtime и checkpointer; одинаковое слово graph не задаёт этих guarantees.

<!-- /unit:6d9c58596aa785fc0b -->

<!-- unit:c50b13ee66394c5103 -->
<a id="u-c50b13ee66394c5103"></a>

### Streaming, cancellation и trace

<!-- /unit:c50b13ee66394c5103 -->

<!-- unit:69aa2f2d113688a2d5 -->
<a id="u-69aa2f2d113688a2d5"></a>

Streaming может показывать model tokens, node updates или events выполнения. Это разные уровни наблюдения. Частичный answer preview не равен accepted answer; событие «tool started» не равно «tool succeeded». Для каждого event полезны run/node/call identity и статус, чтобы UI и evaluator не считали начало работы завершённым фактом.

<!-- /unit:69aa2f2d113688a2d5 -->

<!-- unit:a4601f0f267cdd000a -->
<a id="u-a4601f0f267cdd000a"></a>

Cancellation пользователем/дедлайном прекращает дальнейшие действия по policy, но уже отправленный внешний запрос может продолжить исполняться. Нужно обработать late result и ambiguous pending status. Deadline guard перед node не прерывает автоматически зависший network call внутри него; для конечного времени нужны и bounds исполнения узлов.

<!-- /unit:a4601f0f267cdd000a -->

<!-- unit:e234cfe999a9b5d7a3 -->
<a id="u-e234cfe999a9b5d7a3"></a>

Логировать следует структуру решений и нужное evidence, а не бесконтрольно копировать любой secret в каждый log. Для debugging важны версия graph/prompt/model/tools, входные references, выбранный route, расход и terminal reason. Screenshot графа показывает topology, но не доказывает, что конкретный run пошёл ожидаемым путём.

<!-- /unit:e234cfe999a9b5d7a3 -->

<!-- unit:46316042e0176d9a9f -->
<a id="u-46316042e0176d9a9f"></a>

## Frameworks и local models

<!-- /unit:46316042e0176d9a9f -->

<!-- unit:49061cd982cc6ddbee -->
<a id="u-49061cd982cc6ddbee"></a>

LangChain-style components помогают связать templates/models/tools/parsers; graph runtimes — задавать state/transitions/checkpoints; data frameworks — ingestion/retrieval. AutoGen-style participant coordination может использовать messages/handlers и внутренние tool loops. Оно не обязано быть LangGraph instance и не обязано исключать графовое описание на другом уровне.

<!-- /unit:49061cd982cc6ddbee -->

<!-- unit:88a6960be9c4929a79 -->
<a id="u-88a6960be9c4929a79"></a>

Ollama/LM Studio могут дать local inference endpoint; orchestrator отправляет context/tools, получает model output, валидирует и исполняет actions. Compatibility части API не доказывает одинаковые chat templates, tool formats, structured generation и cancellation. Выбранные model/runtime/version должны быть tested на одном простом call до сложной многоагентной схемы.

<!-- /unit:88a6960be9c4929a79 -->

<!-- unit:1dffc13701d9e6a074 -->
<a id="u-1dffc13701d9e6a074"></a>

В исходнике каждый agent описывался формулой «LLM+prompt+tools+loop». Это полезный common pattern, но не обязательное определение всех agents: бывают deterministic participants, actors без tools и другие abstractions. Содержательно важно, где decision, state, execution и termination.

<!-- /unit:1dffc13701d9e6a074 -->

<!-- unit:f86a04bcb197294288 -->
<a id="u-f86a04bcb197294288"></a>

### Абстракции LangChain: что именно переиспользуется

<!-- /unit:f86a04bcb197294288 -->

<!-- unit:c64ed3c3aa8837bfe7 -->
<a id="u-c64ed3c3aa8837bfe7"></a>

Из S5 сохраняются template, model adapter, parser, runnable composition, batch/async/stream, retriever и tool wrapper. Template параметризует messages; adapter приводит запрос/ответ к общей форме; parser разбирает candidate; composition передаёт output одного stage следующему. Эти функции полезны независимо от бренда библиотеки. Parser не становится constrained decoder —это уже подробно разобрано в [главе structured output](<app-structured.md>).

<!-- /unit:c64ed3c3aa8837bfe7 -->

<!-- unit:2bf9b12c37ecc72ffd -->
<a id="u-2bf9b12c37ecc72ffd"></a>

Batch в orchestration API может означать несколько независимых вызовов с concurrency, а не один GPU tensor batch в inference engine. Async позволяет не блокировать ожидание, но не гарантирует ускорение одного forward. Streaming уменьшает время до видимого fragment, но полный result всё равно требует завершения. Retry/fallback нужно настроить по области ошибки: повтор всей chain может повторить уже успешный mutating tool.

<!-- /unit:2bf9b12c37ecc72ffd -->

<!-- unit:1cf4fbb94b7a8203ab -->
<a id="u-1cf4fbb94b7a8203ab"></a>

В source есть `calculate(expression): eval(expression)` и произвольное выполнение сгенерированного Python. Для изучения арифметики лучше typed handler `multiply(a,b)` или явно ограниченный parser выражений. Возможность сгенерировать строку Python не означает, что её следует исполнять с произвольными правами в учебном примере. Если задача действительно требует code execution, это отдельный executor contract и среда исполнения.

<!-- /unit:1cf4fbb94b7a8203ab -->

<!-- unit:a665fdfcd132e21da0 -->
<a id="u-a665fdfcd132e21da0"></a>

### Стратегии памяти: buffer, window, summary, hybrid

<!-- /unit:a665fdfcd132e21da0 -->

<!-- unit:bc50048e994a51c088 -->
<a id="u-bc50048e994a51c088"></a>

Полный buffer сохраняет все turns, пока они помещаются. Window ограничивает выбранные последние turns/messages, но может потерять раннее условие. Summary сжимает содержание и может ошибиться. Hybrid сохраняет summary старой части и подробный недавний контекст. Это компромиссы, а не одно вечное «правильное production решение».

<!-- /unit:bc50048e994a51c088 -->

<!-- unit:9c8b10460bb46f183e -->
<a id="u-9c8b10460bb46f183e"></a>

Допустим, 40 turns дают в среднем по 250 tokens без дополнительной обвязки: 10000 tokens. Последние 8 turns занимают 2000. Summary старых 32 turns размером 600 плюс эти 2000 дают 2600, но построение summary тоже требует работы и проверки. Экономия длины следующего prompt не равна нулевой цене memory maintenance.

<!-- /unit:9c8b10460bb46f183e -->

<!-- unit:4c45662eac1c1aef32 -->
<a id="u-4c45662eac1c1aef32"></a>

Нужно определить единицу window: K messages, K pairs или token budget —разные правила. Граница не должна отрывать tool result от соответствующего call без корректного преобразования протокола. Для долгосрочного факта «использовать только документы 2025 года» summary опасно считать единственным authority: исходный constraint сохраняют отдельно. Для имени/предпочтения memory record должен иметь источник и возможность обновления, а не вечную догадку модели.

<!-- /unit:4c45662eac1c1aef32 -->

<!-- unit:840d4cb12dbb878fc7 -->
<a id="u-840d4cb12dbb878fc7"></a>

Заявление исходника «большинство серьёзных приложений делают память именно так» не подтверждено исследованием. В конспекте сохраняем механизм и trade-offs, а не неподтверждённую популярность. Конкретные memory classes и imports из старого примера не выдаются за текущий обязательный API.

<!-- /unit:840d4cb12dbb878fc7 -->

<!-- unit:399bf61ae4021724c6 -->
<a id="u-399bf61ae4021724c6"></a>

### Четыре framework approaches без жёстких границ

<!-- /unit:399bf61ae4021724c6 -->

<!-- unit:d83862188acf68aa54 -->
<a id="u-d83862188acf68aa54"></a>

Graph-oriented runtime делает explicit state и transitions. Conversation-oriented coordination описывает участников и обмен messages. Role/task-oriented интерфейс связывает исполнителей с задачами и expected outputs. Data-oriented framework собирает ingestion, index, retrieval и query interfaces. Source сопоставлял эти акценты с LangGraph, AutoGen, CrewAI и LlamaIndex; полезно понимать акценты, но нельзя из них вывести невозможность остальных patterns.

<!-- /unit:d83862188acf68aa54 -->

<!-- unit:3c7b290e206fa3c428 -->
<a id="u-3c7b290e206fa3c428"></a>

Исправляем устаревшее абсолютное противопоставление «LangChain только linear, agents только отдельный LangGraph»: [текущий overview LangChain](https://docs.langchain.com/oss/python/langchain/overview) прямо описывает agents, построенные на LangGraph. AutoGen не требует, чтобы каждый participant был экземпляром LangGraph; при этом messaging и graph model могут существовать на разных уровнях одной системы. Graph также может содержать dynamic routing и fan-out, поэтому «graph всегда статичен, messages всегда эмерджентны» неверно.

<!-- /unit:3c7b290e206fa3c428 -->

<!-- unit:400c67e400c6b8ccff -->
<a id="u-400c67e400c6b8ccff"></a>

В [AutoGen termination examples](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/termination.html) есть программные conditions, в том числе по количеству сообщений и содержимому. Значит, нельзя утверждать, что в conversation framework всякая остановка обязательно решается только LLM. И наоборот, route function в graph может использовать model decision. Вопрос «кто решает» задают к конкретному переходу.

<!-- /unit:400c67e400c6b8ccff -->

<!-- unit:d0657bce9c0eadb0c5 -->
<a id="u-d0657bce9c0eadb0c5"></a>

Оценки из source таблицы вроде «production-ready: да/средне», «отладка лёгкая» и «лучший для большинства» не являются измеренными универсальными свойствами. Reliability зависит от graph/application contract, integration versions, storage, failures и checks. Комбинация LangGraph+retriever либо graph+group conversation внутри node возможна, но каждая boundary добавляет serialization, accounting и recovery obligations.

<!-- /unit:d0657bce9c0eadb0c5 -->

<!-- unit:622d3fa49fe427a874 -->
<a id="u-622d3fa49fe427a874"></a>

### Подробный ответ о local endpoint и готовых артефактах

<!-- /unit:622d3fa49fe427a874 -->

<!-- unit:def186157174ef1ae7 -->
<a id="u-def186157174ef1ae7"></a>

Исходный локальный пример предлагал load_embeddings, describe_clusters и get_ablation_row. Первая функция читала .npy и возвращала shape/нормы; вторая читала cluster CSV; третья искала строку метрики в ablation table. Это интерфейс к заранее вычисленным данным. Он не обучает embeddings и не запускает clustering заново при каждом вопросе.

<!-- /unit:def186157174ef1ae7 -->

<!-- unit:947b2573ae02092b0e -->
<a id="u-947b2573ae02092b0e"></a>

Запрос «сравни silhouette для 5 и 10 clusters» требует rows, действительно содержащих эти configurations и сравнимые protocol/data revisions. Shape embeddings и средние расстояния до центроида сами по себе не дают silhouette. Модель должна прочитать соответствующую метрику, проверить условия сравнения и честно указать отсутствие нужных rows. Строка заглушки «результаты поиска: ...» не считается выполненным поиском.

<!-- /unit:947b2573ae02092b0e -->

<!-- unit:f9e67a871c7997564d -->
<a id="u-f9e67a871c7997564d"></a>

Registry разрешённых artifact IDs лучше произвольного model-generated filesystem path. У артефактов сохраняют dataset/version, preprocessing/encoder revision и evaluation protocol. Если cluster CSV содержит поле size, groupby count считает количество записей, а sum —сумму sizes: в зависимости от формата файла это разные статистики. Код из source нельзя объявить содержательно правильным без описания rows.

<!-- /unit:f9e67a871c7997564d -->

<!-- unit:85ea3f5ff45a5b5019 -->
<a id="u-85ea3f5ff45a5b5019"></a>

При замене endpoint сохраняют graph semantics, но отдельно проверяют template, tool serialization, parser, schema subset, context limits, streaming/cancellation и latency. «Весь агентный код всегда меняется одной строкой» —слишком сильное обещание. В Ollama/LM Studio могут быть свои SDK loops и дополнительные features; границы model/serving/orchestration относятся к функциям, а не запрету одному продукту объединять уровни.

<!-- /unit:85ea3f5ff45a5b5019 -->

<!-- unit:95dd12a6d44d6c7ba7 -->
<a id="u-95dd12a6d44d6c7ba7"></a>

Не переносим из разбираемого примера фиксированные рекомендации 7–14B, обязательный num_ctx≥16384, проценты успеха и токены/сек как закон. Память зависит от weights, KV configuration, context/concurrency и backend; latency —от реального числа calls, input/output lengths, caching и tools. Низкая temperature не исправляет отсутствующую способность или неверный template. Для учебного понимания достаточно проследить один реальный protocol cycle; запуск модели и скачивание weights здесь не заявляются.

<!-- /unit:95dd12a6d44d6c7ba7 -->

<!-- unit:831c5e8dc71c93920f -->
<a id="u-831c5e8dc71c93920f"></a>

## Multi-agent coordination и бюджет

<!-- /unit:831c5e8dc71c93920f -->

<!-- unit:139ad0b5e4b45cadcb -->
<a id="u-139ad0b5e4b45cadcb"></a>

Researcher находит sources, reviewer проверяет claims, writer оформляет результат — возможное разделение функций. Польза появляется, если роли дают разные проверяемые operations/evidence. Три одинаковых prompts одной model могут просто втрое повторить один bias.

<!-- /unit:139ad0b5e4b45cadcb -->

<!-- unit:f96dbfe67a68c87836 -->
<a id="u-f96dbfe67a68c87836"></a>

Coordinator должен определить recipients, message schema, доступный scope, handoff state и общий budget. Nested loops легко создают unbounded repeated work: каждый participant «ещё чуть-чуть поищет», но никто не проверяет global stop. По этой причине single well-defined workflow — обязательный baseline сравнения, а не предположение, что multi-agent хуже/лучше.

<!-- /unit:f96dbfe67a68c87836 -->

<!-- unit:067e7db447afd5f3b0 -->
<a id="u-067e7db447afd5f3b0"></a>

Сравнивать total model tokens/tool cost/latency и task success при equal budget. If agent A criticises agent B, это не independent truth; полезность review проверяется обнаруженными исправлениями и независимым oracle.

<!-- /unit:067e7db447afd5f3b0 -->

<!-- unit:77c61e0b56677761a4 -->
<a id="u-77c61e0b56677761a4"></a>

### Локальная policy и межагентный coordinator

<!-- /unit:77c61e0b56677761a4 -->

<!-- unit:b0a97dd0d3d5a460ab -->
<a id="u-b0a97dd0d3d5a460ab"></a>

Researcher может выбирать search/read actions, coordinator —выбирать researcher или writer. Один выбор не определяет другой. Участник может быть deterministic executor, retrieval service или human input endpoint без собственного LLM loop. Даже если все три роли используют одну модель, их context/tools могут различаться; одинаковые веса не означают автоматически полную бесполезность разделения, но и разные role names не доказывают полезную независимость.

<!-- /unit:b0a97dd0d3d5a460ab -->

<!-- unit:4baccf9d4e5b7258a1 -->
<a id="u-4baccf9d4e5b7258a1"></a>

Handoff должен передавать исходную задачу, применимый scope, evidence references, unresolved questions, статус и budget. Передать writer только красивый prose summary рискованно: он может потерять IDs и условия. Reviewer должен видеть проверяемые claims и sources; согласие двух моделей не заменяет внешний oracle. Если оба читают один ошибочный summary, они могут уверенно повторить одну ошибку.

<!-- /unit:4baccf9d4e5b7258a1 -->

<!-- unit:f70c927ad321699aa1 -->
<a id="u-f70c927ad321699aa1"></a>

Вопрос пользователя из S6 о самостоятельном выборе subagents объясняется как разрешённое действие делегирования с input/result contract. Это концептуальная возможность. Из интерфейса конкретного продукта нельзя достоверно восстановить его внутренний framework, prompts или точные лимиты. Упоминание subagents в учебном источнике не означает, что в этой работе уже создавались дополнительные агенты.

<!-- /unit:f70c927ad321699aa1 -->

<!-- unit:07f57513a935e5d0a0 -->
<a id="u-07f57513a935e5d0a0"></a>

### Стоимость координации и честное сравнение

<!-- /unit:07f57513a935e5d0a0 -->

<!-- unit:a76243e3e7cb05b411 -->
<a id="u-a76243e3e7cb05b411"></a>

Source пример распределял 100 документов между 10 researchers, по 10 каждому. При независимых задачах и достаточной capacity идеальный parallel path может сократиться примерно в 10 раз до synthesis, но суммарное чтение 100 документов не исчезает. Если каждый из 10 вызовов дополнительно повторяет 1000 tokens instructions, это 10000 instruction tokens против 1000 у одного вызова, без учёта различий context limits и числа calls. Ускорение elapsed time не тождественно сокращению token work.

<!-- /unit:a76243e3e7cb05b411 -->

<!-- unit:b005c43ab07de6e59a -->
<a id="u-b005c43ab07de6e59a"></a>

Пусть последовательная обработка занимает 100 s, а parallel research 10 s плюс synthesis 8 s при игнорировании overhead. Speedup 100/18≈5.56, а не 10. Общий выигрыш ограничен serial частью, capacity backend и очередями. Если synthesis требует перечитать всё, оно может стать новым bottleneck.

<!-- /unit:b005c43ab07de6e59a -->

<!-- unit:c9b25396b0dfe6703d -->
<a id="u-c9b25396b0dfe6703d"></a>

Сравнение требует одинаковых задач, data access, критериев correctness и суммарного бюджета. Один агент с тремя обоснованными stages может быть сильным baseline. Multi-agent может помочь независимыми tools/evidence и параллельностью, но role conflict сам по себе не гарантирует роста качества. Проверяют, какие ошибки обнаружены, какие новые внесены и насколько вырос cost per correct task.

<!-- /unit:c9b25396b0dfe6703d -->

<!-- unit:c16bde9294e02f9b15 -->
<a id="u-c16bde9294e02f9b15"></a>

### Self-correction не означает поиск значимости до победы

<!-- /unit:c16bde9294e02f9b15 -->

<!-- unit:6f9a7354c0514c0375 -->
<a id="u-6f9a7354c0514c0375"></a>

В S5 сказано: если статистический тест не показал значимость, агент может попробовать другой. Как общий пример self-correction это неверно. Исправлять неправильно выбранный тест при несоблюдённых предпосылках —одна задача. Перебирать тесты/группы/признаки, пока p-value не станет меньше 0.05, —другая: меняется процедура статистического вывода и риск ложных находок.

<!-- /unit:6f9a7354c0514c0375 -->

<!-- unit:456252040e0ff80aeb -->
<a id="u-456252040e0ff80aeb"></a>

В идеализированном случае 20 независимых true-null tests с вероятностью false positive 0.05 каждый вероятность хотя бы одного равна $1-0.95^{20}\approx0.641514$. Автоматический loop усиливает эту проблему, если критерий enough —«нашёл значимое». Для аналитического workflow нужно заранее определять hypothesis/protocol и учитывать exploratory/multiple-testing характер работы, а не выдавать удачно найденный p-value за подтверждённый вывод.

<!-- /unit:456252040e0ff80aeb -->

<!-- unit:30503855400e36eda8 -->
<a id="u-30503855400e36eda8"></a>

Точно так же `len(findings)>=5` не означает полноту исследования. Source extract-loop каждый раз проходил по всем накопленным sources и дописывал findings; один и тот же source мог учитываться повторно и искусственно достичь пяти. Нужны identity/deduplication и покрытие вопросов, а не только длина списка. Полезный progress marker —получено новое применимое evidence или устранён конкретный validation failure.

<!-- /unit:30503855400e36eda8 -->

<!-- unit:3e684206ea0fd2cf22 -->
<a id="u-3e684206ea0fd2cf22"></a>

### Три поиска: трасса успеха и трасса исчерпания

<!-- /unit:3e684206ea0fd2cf22 -->

<!-- unit:963bd412f06bb56db6 -->
<a id="u-963bd412f06bb56db6"></a>

Для учебного cap3 searches счётчик увеличивается при каждой реально начатой попытке, включая неуспешную, если это выбранная единица лимита. Первый поиск даёт нерелевантный документ, второй —нужный v8 passage: можно ответить и завершиться после validation, не расходуя третий. Если все три не дали evidence, следующий route —partial/abstain, а не четвёртый поиск под другим именем node.

<!-- /unit:963bd412f06bb56db6 -->

<!-- unit:da43f4235debafb406 -->
<a id="u-da43f4235debafb406"></a>

Три search executions не ограничивают автоматически число model calls, validation retries и время зависшего tool. Поэтому global caps задаются отдельно по нужным units. Invalid candidate после третьего поиска можно отклонить без дальнейшего retrieval; новый generation step возможен только если есть соответствующий model/token budget. В source local loop cap iteration проверялся до обработки очередного proposal: на границе model мог уже попросить tool, но graph завершался, не исполнив call. Такой terminal state должен быть обозначен как budget-exhausted/partial, а не возвращён как готовый final answer.

<!-- /unit:da43f4235debafb406 -->

<!-- unit:0e04bc3c29453e5600 -->
<a id="u-0e04bc3c29453e5600"></a>

Для каждого пути называют terminal reason: accepted answer, missing input, no applicable evidence, permanent failure, budget exhausted или cancelled. Это делает цель процесса проверяемой и позволяет читать trace как объяснение результата. Сам факт достижения END сообщает об окончании исполнения, а не автоматически об успешности задачи.

<!-- /unit:0e04bc3c29453e5600 -->

<!-- unit:5e484a90140349e7ad -->
<a id="u-5e484a90140349e7ad"></a>

## Самопроверка и практика

<!-- /unit:5e484a90140349e7ad -->

<!-- unit:1b88909984a8152870 -->
<a id="u-1b88909984a8152870"></a>

1. Чем graph отличается от обучения policy? **Ответ:** graph организует выполнение; policy training меняет параметры по data/objective.
2. Что потеряет last-write-wins evidence reducer? **Ответ:** результаты parallel branch, обновлённые раньше другого.
3. Гарантирует ли checkpoint отсутствие duplicate actions? **Ответ:** нет, external side effect и сохранение checkpoint могут быть неатомарны.

<!-- /unit:1b88909984a8152870 -->

<!-- unit:db9c1bdbf756ddd39c -->
<a id="u-db9c1bdbf756ddd39c"></a>

4. Проверяет ли TypedDict внешний JSON при исполнении? **Ответ:** сам по себе нет; нужны runtime checks.
5. Что возвращать при append reducer? **Ответ:** новый delta; полный accumulated list может задублировать историю.
6. Сохранит ли dict union все вызовы одного tool? **Ответ:** нет, совпавший key заменяется; нужны call/result identities.
7. Является ли thread_id правом доступа? **Ответ:** нет, выбор/доступ к execution history проверяет приложение.
8. Возобновляется ли dynamic interrupt с сохранённой Python строки? **Ответ:** node может исполниться с начала; pre-interrupt effects требуют правильной replay policy.
9. Значит ли END, что ответ верен? **Ответ:** нет, остановка и success —разные статусы.
10. Почему пять findings ещё не enough evidence? **Ответ:** это могут быть дубли/нерелевантные claims без покрытия задачи.
11. Исправление ли анализа —перебирать тесты до p<0.05? **Ответ:** нет; это меняет процедуру вывода и требует учёта множественных проверок.
12. Гарантирует ли graph детерминированный результат? **Ответ:** нет, model sampling, external data и parallel merge могут различаться.

<!-- /unit:db9c1bdbf756ddd39c -->

<!-- unit:ed57d95185a632cd38 -->
<a id="u-ed57d95185a632cd38"></a>

**Практика:** спроектировать graph с retrieve/rewrite/answer/validate и global cap3searches. Добавить checkpoint после search; описать crash перед сохранением observation. Показать recovery отдельно для read-only search и report-creation action. Код framework пока не нужен: сначала проверяется semantics.

<!-- /unit:ed57d95185a632cd38 -->

## Результат главы и переход дальше

Ответьте на вопрос: **Какие данные и переходы нужны для остановки, восстановления и повторного исполнения?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

Следующие темы: [RAG: BM25, dense, hybrid и multi-vector retrieval](<app-rag.md>), [Постановка обучения агента и baseline](<../06-agent-learning/agent-task.md>).

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
