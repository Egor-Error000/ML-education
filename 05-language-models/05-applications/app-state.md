# Состояние, orchestration и граф управления

**Уровень:** основа ветки.

**Предпосылки:** [Вызов инструментов и исполнение](<app-tools.md>)

**Используют:** [RAG: BM25, dense, hybrid и multi-vector retrieval](<app-rag.md>), [Постановка обучения агента и baseline](<../06-agent-learning/agent-task.md>)

**Учебный вопрос:** Какие данные и переходы нужны для остановки, восстановления и повторного исполнения?

## Карта вычисления и границы этапов

![Состояние, orchestration и граф управления: входы, преобразования и выходы](<../../оформление/course/app-state.svg>)

Orchestration (управление порядком действий и состоянием) задаёт порядок действий и сохранение состояния. Повтор после сбоя должен учитывать уже выполненные внешние эффекты, иначе одна логическая операция может исполниться дважды.

## Какие компоненты добавляются вокруг модели

Один model call возвращает текст/структуру. Приложение должно получить user input, подготовить context (доступный модели контекст), при необходимости найти документы, исполнить проверенные tools, сохранить state (состояние системы) и завершить результат. Orchestration задаёт порядок/условия этих операций.

Linear chain подходит для фиксированного пути `retrieve→generate→validate`. Graph/state machine нужен, когда есть branches, retries, tool (инструмент приложения) loop и early termination. Model weights от изменения graph не обновляются: runtime (среда исполнения модели) исполняет существующую policy, обучение policy требует отдельного objective (оптимизируемая целевая функция)/data.

Пример state:

```text
question, authorized_scope, messages, evidence,
pending_calls, observations, attempts, budget, status.
```

Trusted constraints, например права на документы, задаются приложением. Их нельзя заменять модельным summary из chat history. Node получает нужные поля и возвращает определённые изменения; merge policy для parallel updates должна быть явной.

**Схема курса:** request → state/router → model proposes tool or retrieval (поиск кандидатов) → validate/execute → observation (доступное наблюдение среды)/evidence → state update → stop, retry or answer



### Три уровня приложения: веса, исполнение и управление

Как использовать локальную модель через Ollama для цикла агента и вызова инструментов? Сначала разделим веса модели, её endpoint и приложение. Веса задают обученную функцию предсказания. Inference (применение обученной модели) backend загружает их, токенизирует вход, выполняет forward (прямой проход вычисления) и decoding. Приложение хранит цель и историю, выбирает доступные операции и связывает последовательные обращения. Один продукт может объединять несколько уровней, но от этого их функции не становятся одинаковыми.

Путь одного решения выглядит так:

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

Стрелка назад — работа runtime. Внутри одного forward нет скрытого Python-цикла, который автоматически обращается к поиску. После tool result выполняется новое обращение с обновлённым контекстом; backend может переиспользовать совместимый prefix (уже доступный префикс) KV cache (кеш повторно используемых вычислений). Это экономия вычислений, а не замена observation воображаемым результатом. Веса при каждом таком шаге остаются прежними.

Workflow и agent различаются тем, кто задаёт следующий шаг. Для workflow разработчик заранее определил допустимые ветки, например «поиск → ответ → проверка». В agent loop модель может выбирать среди предложенных действий. В обоих случаях runtime определяет исполнимые переходы и пределы. Graph способен выразить оба варианта: наличие графа ещё не означает автономный выбор модели. Такую границу использует и [документация LangGraph о workflows и agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents).

### State — больше, чем текст переписки

Messages нужны для model context. State нужен для корректного исполнения. Помимо текста, в нём могут находиться идентификатор задачи, версия данных, разрешённый scope, множество уже прочитанных документов, pending calls, результаты проверок, остаток бюджета и terminal status. Часть полей сериализуют для LLM, часть использует только программа. Пароль, объект подключения к БД и mutable client session не следует превращать в обычные слова prompt (входной запрос или инструкция) ради «полноты памяти».

Различаются три вида памяти. Working context — то, что модель видит сейчас. Durable state — сохранённое состояние выполнения, позволяющее продолжить после сбоя. Knowledge store — документы и индекс для retrieval. Сохранить checkpoint (сохранённое состояние модели) не означает обучить модель новым знаниям; заменить context summary не означает изменить документы; заново загрузить веса не означает восстановить потерянные результаты tools.

Особенно важна роль summary. Оно может сжать старые рассуждения и observations, но не должно самовольно менять права или статус операции. Например, фраза summary «доступ ко всем документам разрешён» не заменяет authoritative scope. Для фактов сохраняют IDs и versions, чтобы итог можно было проверить независимо от пересказа. Для pending external operation сохраняют её identity, а не только слова «кажется, сделали».

### Graph как явная машина состояний

В учебной машине есть состояния `READY`, `CALL_PENDING`, `OBSERVED`, `ANSWER_READY`, `DONE`, `FAILED`. Из READY модель предлагает call или answer. После проверки call переходит в CALL_PENDING; исполнитель создаёт observation, и управление возвращается к READY через OBSERVED. Candidate answer проходит проверку, после которой наступает DONE либо допустимая ветка исправления. Исчерпание бюджета переводит в FAILED/partial с объяснением, а не в фиктивный успех.

Смысл state machine — сделать переходы проверяемыми. Из CALL_PENDING нельзя честно перейти в «tool succeeded», пока нет подтверждённого result. Из ANSWER_READY нельзя объявить наличие цитаты, если cited ID отсутствует в evidence. Программный validator может проверять эти условия даже тогда, когда модель уверенно пишет обратное.

Фиксированный pipeline тоже имеет state, хотя его иногда прячут в локальные переменные. Graph полезен при усложнении ветвлений, но не обязан заменять простую функцию из трёх шагов. Разница в явности control flow и persistence, а не в магической способности graph «думать лучше».

## Frameworks: abstraction по потребности

LangChain-style components дают templates, model/tool adapters, chains и parsers; LangGraph-style graph описывает state/transitions/checkpointing. LlamaIndex делает акцент на data/retrieval integrations. AutoGen/CrewAI исторически предлагали conversation/role coordination patterns. Их APIs меняются; выбирать нужен набор capabilities, а не обязательное прохождение каждого framework.

Один agent participant может иметь свой loop с tools, state и termination. Это не требует быть реализованным через LangGraph: тот же механизм можно написать другим runtime. Multi-agent coordination добавляет messages, duplicated context и termination difficulties; качество сравнивать с single-agent baseline (базовый вариант сравнения) при одинаковом total budget.

Локальные Ollama/LM Studio models могут быть model endpoints такого приложения. Ответ на исходный вопрос о локальном agent pipeline: runtime вызывает endpoint, валидирует output, исполняет tools и повторяет call. Встроенные features (признаки) конкретного приложения могут часть этого делать, но supported API/tool modes проверяются отдельно.

### Ответ на вопрос об AutoGen и LangGraph

Участников диалоговой системы AutoGen и узлы графа LangGraph нельзя считать тождественными сущностями. Совпадение общей идеи последовательных решений не означает зависимость одной реализации от другой. Participant может вызывать модель, исполнять tool, возвращать фиксированный ответ или передавать сообщение человеку. Graph node также может быть обычной deterministic функцией. Поэтому исходное «каждый агент во всех frameworks обязательно LLM + tools + loop» слишком широко.

Есть два независимых уровня: локальная policy участника и coordinator, выбирающий, кому передать управление. Researcher может сам выбирать search/read tools, а coordinator — решать, когда передать его evidence автору ответа. Либо coordinator детерминирован, а участники только выполняют фиксированные функции. Названия «researcher», «critic», «writer» работают как описание роли, но не создают отдельного знания и не гарантируют независимость ошибок.

Вопрос о самостоятельном выборе участников разбирается на этом же уровне: делегирование может быть ещё одним разрешённым action с ограниченным входом, бюджетом и result contract. Из факта, что интерфейс показывает subagent, нельзя восстановить закрытые внутренние prompts, framework и правила маршрутизации конкретного продукта. Здесь объясняется механизм, без утверждения о внутреннем устройстве упомянутого в разбираемом примере редактора.

### Parallel branches: latency (задержка отдельного запроса), state merge и общий бюджет

Для двух независимых поисков длительностью 0.8 и 1.2 секунды последовательный путь занимает 2.0, идеальный parallel —1.2 до merge, без overhead/queue. Но если второй запрос строится по результату первого, такого ускорения нет. Parallel execution уменьшает критический путь, а не сумму оплаченной/выполненной работы.

Пусть начальное evidence множество `{a}`, ветка A добавляет b, ветка B —c. Если обе возвращают полное replacement-поле, last-write-wins сохранит `{a,b}` или `{a,c}` и потеряет другой результат. Объединение множеств даст `{a,b,c}` независимо от порядка. Для evidence с versions удобнее merge по стабильному ID/revision и явная политика конфликтов. Для messages простая конкатенация сохраняет всё, но порядок A+B отличается от B+A; порядок может повлиять на следующий model call.

Две ветки не должны независимо тратить весь общий budget. При лимите 6000 tokens (токены) coordinator может зарезервировать 3000 и 2000 для двух участников, 1000 для synthesis. Это 6000 общего лимита, а не 6000 на каждого. Если участник использует больше model calls, повторённые system/tool/history tokens тоже учитываются по объявленному accounting; параллельность их не устраняет. Учебное сравнение с single-agent имеет смысл при сопоставимых data access и суммарном бюджете.

### Локальная модель: совместимость проверяется по слоям

Скачанный checkpoint сам по себе не задаёт работающий агент. Нужны подходящие tokenizer (токенизатор)/chat template (шаблон сериализации входа), способность model следовать tool protocol, поддержка backend и parser, затем executor и loop. Ошибка на каждом слое проявляется по-разному: raw text вместо calls, испорченные argument types, исчезнувший tool result, неверная role serialization, повтор вызова после уже полученного результата.

Нельзя переносить из разбираемого примера абсолютное «Ollama/LM Studio только генерируют, никаких loops у них нет». В документации [Ollama показан multi-turn tool loop](https://docs.ollama.com/capabilities/tool-calling); [LM Studio разделяет tool requests и исполнение в коде](https://lmstudio.ai/docs/developer/openai-compat/tools), а его [SDK .act() автоматизирует несколько rounds](https://lmstudio.ai/docs/typescript/agent/act). Где физически находится loop — в вашем коде, SDK или встроенной функции продукта — зависит от выбранного интерфейса. Это не меняет различия между model proposal и реальным effect.

Совместимый HTTP shape облегчает замену endpoint, но не гарантирует одинаковые supported schema (схема допустимых данных), tool-choice modes, streaming events и качество. Начинать проверку разумно с read-only инструмента известного результата: предложен ли правильный call, распарсены ли arguments, реальный ли result возвращён в контекст, использован ли он в final. Нельзя лечить ошибку chat template добавлением ещё пяти агентов.

## Chain, branch и loop

Chain — фиксированная последовательность stages. Branch выбирает дальнейший путь по state. Loop повторяет часть работы до condition. В Python это можно выразить functions/if/while; frameworks добавляют integrations, checkpointing, streaming и удобное описание transitions.

Пример documentation assistant:

```text
prepare → retrieve → inspect evidence
                    ├ enough → answer → validate → end
                    ├ retry allowed → rewrite query → retrieve
                    └ no evidence/budget → abstain → end.
```

Каждый arrow должен иметь condition. Если «model решает, искать ещё или хватит», в state должны быть observations, remaining budget и criterion enough evidence. Prompt «будь умным» не задаёт bounded algorithm.

**Схема курса:** initial state → selected node → state transition → tool/worker result or error → reducer/checkpoint → route/stop condition → observable final state



### Что означает stateless и почему «чистая функция» —упрощение

Рассмотрим два обращения, разделённые во времени: сначала «меня зовут Алексей», затем «как меня зовут?». Если второй input не содержит первого факта и нет внешнего memory mechanism, модель не получает основания восстановить имя именно этого собеседника. Проблема не в пяти минутах как сроке забывания: нужные данные просто не переданы в вычисление.

Для понимания удобно писать $y\sim p_\theta(\cdot\mid x)$: input x содержит доступный context, θ —веса. Это точнее, чем объявлять каждый API call математически чистой детерминированной функцией f(prompt). Sampling (случайный выбор по распределению) может быть stochastic, provider может хранить conversation state, backend —KV cache, а дополнительные tools/runtime —свои данные. При этом in-context память не равна изменению θ. Сохранённое имя может попасть в следующий prompt без fine-tuning (дообучение модели).

Нужен явный ответ на вопрос «что увидит следующий вызов»: весь transcript, последние сообщения, summary, найденные memory records или их смесь. Если API предоставляет server-side conversation, ответственность за выбор conversation ID и проверку изоляции остаётся у приложения. Если история хранится только в локальном Python list, restart процесса её потеряет.

### Пять ошибок управления состоянием и переходами

История добавляет информацию из прошлых ходов; tool execution добавляет внешние наблюдения; validator даёт критерий проверки; control flow связывает необходимые операции; error policy выбирает допустимое восстановление. Каждое средство решает свою проблему. Добавленная история не гарантирует память о всех деталях при truncation. Добавленный critic не становится источником истины. Новый graph не делает устаревший corpus актуальным.

Исходный пример интеграла $\int_0^\pi\sin x\,dx=2$ можно решить аналитически: первообразная $-\cos x$, значение $-\cos\pi+\cos0=2$. Поэтому фраза «для этого непременно нужен Python» неверна. Инструмент нужен, когда выбранная задача требует проверяемого исполнения/наблюдения; он полезен как внешний oracle, но нельзя объявлять каждую простую математическую задачу недоступной language model.

В другом примере $247\cdot891=220077$. Проверка calculator различает реальный результат и plausible text. Однако если модель неправильно переписала задачу в выражение, точный calculator вернёт точный ответ на неверный вопрос. Надёжность получается из правильной постановки, выполнения и использования результата, а не из одного факта наличия tool.

### От wrapper к graph без выдуманной лестницы возможностей

Source wrapper сохранял user message, вызывал модель и дописывал assistant answer. Pipeline затем генерировал вопросы по теме «Трансформеры в NLP», искал ответы и синтезировал отчёт. Это хорошие иллюстрации разных control flows. Но pipeline в обычном Python уже содержит `for`: утверждение «на этом уровне нет циклов» противоречит самому примеру. Один tool response также может содержать несколько calls; их число не определяется ярлыком «уровень 2».

Если source предлагает разбивать список вопросов через `split("\n")`, нужно учесть заголовки, пустые строки и многострочный вопрос. Для интерфейса «ровно пять вопросов» естественнее отдельная структурная проверка списка. Затем оценивают содержательную полноту: пять повторений одного вопроса не покрывают пять аспектов Transformer. То же касается отчёта: много stages не доказывает, что в него попали механизмы attention, обучения и CV variants.

Chain как композиция функций не запрещает ветвление внутри функции. Graph делает переходы и state явными, помогает persistence/inspection, но не создаёт новую вычислимость. «В chains практически невозможно» из сравнительных таблиц следует читать как критику удобства конкретного pattern/API в той версии, а не математический запрет. Современные framework boundaries проверяются отдельно от исторического учебного snippet.

### Routing: decision модели и переход runtime

В support-примере запрос классифицируется как billing, technical или complaint, затем выбирается соответствующий путь. Classifier может быть deterministic rule или LLM. Router получает результат и проверяет, что label (метка) допустим. Если label неизвестен, нужен fallback/unknown branch; иначе typo станет попыткой перейти в несуществующий node.

Model confidence (уверенность по принятой оценке) вроде 0.9 не является автоматически calibrated вероятностью правильности. В примере «если уверена —ответь, иначе ищи» нужно определить, как confidence измеряется и почему ей можно доверять. Более наблюдаемое условие —есть ли source нужной версии, покрыты ли обязательные поля и пройдены ли validators. И эти признаки не обещают всеведения: доступный source может содержать ошибку.

Во всех вариантах разделяются proposed route, проверка constraints и accepted route. Модель может предложить дальнейший поиск; runtime ограничивает допустимый scope и remaining budget. Модель может предложить finish; runtime решает, достаточно ли этого для terminal success по заданной задаче.

## State и ответственность узлов

State хранит messages, original request, authorization scope, document (документ) IDs/versions, observations, pending calls, attempts и result. Не все поля должны целиком копироваться в каждый model prompt: это снижает cost и риск смешать trusted constraints с untrusted text.

Node contract задаёт, что читается и что обновляется. Если два parallel nodes добавляют evidence, reducer должен объединить по IDs/version, а не оставить последний список. Для scalar budget надо учитывать суммарный расход, а не minimum/last update случайно.

Original user constraints полезно хранить отдельно от model summary. Если summary потерял «только документы 2025 года», retrieval не должен расширить scope молча. Structured trusted state задаёт invariant, model-generated text — содержимое для обработки.

### TypedDict —описание, а не автоматическая защита

TypedDict помогает static checker обнаружить неверное имя поля и согласовать интерфейсы функций. Обычный Python dict не начинает проверять types при каждом присваивании только из-за аннотации. Если из внешнего JSON пришла строка вместо iteration integer, необходим runtime validator. И даже корректный integer -5 может нарушать invariant неотрицательного счётчика.

Node contract полезно описать через preconditions и postconditions. Retrieve читает original_question, scope, corpus_revision и search budget; возвращает observation с call ID, найденные references и расход. Generate читает подготовленный context и возвращает candidate. Validate читает candidate и evidence, выдаёт результат проверки. Authorized scope не должен становиться свободным output поле model node.

Не стоит хранить произвольные live objects внутри durable state без serialization contract: dataframe, открытый file handle и GPU engine не обязательно восстанавливаются из checkpoint. Для большого artifact хранят stable ID/revision и metadata; данные читают по контролируемому path. Это особенно важно для source примеров с embeddings (векторные представления) и cluster CSV.

### Delta и reducer: два соглашения нельзя смешивать

Пусть messages уже равны `[u1,a1]`, node получил новый a2. При replacement policy он возвращает полный `[u1,a1,a2]`. При append policy достаточно `[a2]`. Если append reducer получает полный старый список вместе с новым, получится `[u1,a1,u1,a1,a2]`. История продублирована, token (токен) cost вырос, прежние фразы могут получить лишний вес в model context.

Реальная message-aware merge policy может использовать IDs: новый ID добавляется, существующий заменяется. Тогда повторное включение того же ID не обязано дублировать сообщение. Но если при каждом replay старому содержимому выдают новые IDs, семантический дубль снова выглядит новым. Идентичность message, содержимого и business operation различаются.

В [Graph API LangGraph](https://docs.langchain.com/oss/python/langgraph/graph-api) default updates заменяют значение поля; пользовательский reducer определяет иное слияние, а add_messages учитывает IDs. Это причина читать конкретную схему state перед изменением node return. `Annotated` само по себе не выбирает правильный merge: приложение/runtime должно интерпретировать указанную функцию.

### Dictionary union, evidence identity и расход

Исходник предлагает `operator.or_` для словаря analysis_results. При разных keys это объединяет записи. При одинаковом key правая запись заменяет левую. Если результат хранится под именем `run_statistical_test`, второй тест затрёт первый, даже если параметры и data revision различаются. Для истории результатов лучше stable call/result ID и metadata аргументов; сводка по tool name —отдельное представление.

Merge по document ID тоже недостаточен, если нужно различать версии. `(doc_id,revision)` сохраняет два observations, а policy выбирает применимую версию. Молчаливое «последний arrival победил» не означает «самая свежая версия победила». Order доставки и порядок изменения источника —разные часы.

Budget accounting удобно строить на events расхода с уникальным usage ID. Две ветки потратили 300 и 500 tokens: общий расход 800. Если обе прочитали remaining=1000 и независимо записали остатки 700 и 500, replacement потеряет расход одной ветки. Если один usage event доставлен повторно, простое суммирование начислит его дважды; dedupe по event identity и сумма уникальных расходов дают другой invariant.

### Разбираем ошибки первого StateGraph snippet

Пусть зарегистрированы узлы `call_model` и `execute_tool`, маршрут возвращает строку `end`, а затем добавляется ребро из `end` к END. Однако сам node `end` не зарегистрирован. Нужен прямой mapping label→END либо реальный зарегистрированный terminal node. Label результата routing не следует автоматически считать существующим graph node.

Учебная запись mapping, без обещания совместимости с любой версией SDK:

```python
END = "__end__"  # учебный sentinel, в реальном SDK берётся его END
route_targets = {"execute": "execute_tool", "final": "validate_answer", "stop": END}

def route_after_model(response, remaining_tool_calls):
    calls = response.get("tool_calls", [])
    if calls:
        return "stop" if len(calls) > remaining_tool_calls else "execute"
    return "final"
```

Если calls нет, candidate идёт на проверку ответа, а не в executor. Если предложено два calls, а осталось одно разрешённое исполнение, выбранная policy останавливает этот proposal; другое приложение может разрешить correction, но должно сохранить корректный протокол. Пропустить неизвестно какие calls молча —плохая замена явной policy.

Source executor брал `tool_calls[0]`. При нескольких calls остальные теряются, а следующий model context может ждать их результатов. Либо framework действительно ограничивает proposal одним call, либо executor обрабатывает весь список с correlation IDs и определёнными зависимостями. Декларация list в API не означает, что код по первому элементу корректен для всех случаев.

### Аналитический пример: отдельные баги и содержательные критерии

В функции create_visualization из исходника filename вычисляется через `state['visualizations']`, хотя state не передан и не задан как локальная переменная. Это не graph feature (признак), а неявная зависимость/ошибка области видимости. Имя artifact лучше выдаёт контролируемый executor, включающий run/call identity, а результат возвращает reference. Количество записей списка само по себе не защищает от collision двух parallel writers.

Source completion эвристика ищет в ответе слова «результат», «вывод», «итог». Строка «результат пока не получен» удовлетворяет такой проверке, хотя сообщает о незавершённости. В другом случае свободный текст без keywords направляется к execute_tool, а у него может не оказаться tool_calls вообще. Routing сначала должен различать фактический call и candidate answer, затем содержательно проверять результат.

Поле iteration в state не ограничивает работу, если ни один обязательный cycle path его не увеличивает. Source snippets меняют счётчик в разных местах; их нельзя объединять в надежду, что одно название поля гарантирует cap. Нужно проследить каждое back edge и проверить расход лимита/условие termination. Recursion/step limit runtime —дополнительный предохранитель, а не объяснение semantic success.

## Checkpoint и resume

Checkpoint сохраняет состояние workflow после определённых transitions, чтобы восстановить работу. Но saved messages не автоматически доказывают, какие external tools действительно были executed. В state нужны call IDs/statuses и durable result provenance.

Crash scenario: mutation выполнилась, process упал до сохранения observation. Resume из previous checkpoint может повторить operation. Поэтому checkpointing сам по себе не гарантирует exactly-once side effects; idempotency/status reconciliation остаются задачами execution layer.

Для read-only учебного поиска repeat безопаснее, но results могут измениться при новой версии corpus. Воспроизводимый replay использует saved observations/snapshot или явно отмечает, что retrieval выполнен заново по другим данным. Равенство graph topology не означает равенство inputs.

### История, checkpoint и snapshot источников

Transcript хранит разговор. Checkpoint хранит workflow state и нужную runtime информацию по контракту checkpointer. Corpus snapshot фиксирует, по каким документам искали. Model artifact фиксирует веса/adapter/tokenizer/template. Эти четыре объекта не взаимозаменяемы. Воспроизвести graph по одному transcript бывает невозможно, если из него не видно scope и pending calls.

В source checkpointer example первый message —только «Привет», второй —«Как меня зовут?». Сохранённая история не содержит имени. Правильный resume должен помнить приветствие, но не выдумывать Алексея из другого учебного примера. Persistence сохраняет переданные факты; она не создаёт отсутствующие сведения.

In-memory saver живёт в памяти процесса и теряется вместе с ним. Durable backend сохраняет данные дольше, но его guarantees зависят от transactions, storage и configuration. `thread_id` выбирает историю выполнения; это не универсально безопасная замена user ID. Один пользователь может иметь несколько независимых задач, а ошибочное повторное использование чужого ID смешает histories, если application access layer этого не предотвращает.

### Crash между effect и checkpoint

Представим шаг `create_report` с intent R1. Внешний сервис уже создал report Q7, затем процесс упал до сохранения observation. Последний checkpoint не содержит Q7. Resume повторяет node. Без idempotency/reconciliation возможен второй report. Checkpointer знает сохранённые transitions, а не всё, что успело произойти в другой системе.

Для read-only lookup повтор тоже не всегда означает одинаковый observation: corpus мог обновиться с v7 до v8. Replay по saved results повторяет прежний evidence; rerun по live source получает новый. Оба режима допустимы, но выводы о воспроизводимости различаются. Чтобы сравнивать routing policies, удобно фиксировать observations; чтобы проверить актуальную систему, наоборот, нужен объявленный live input.

Внешний effect и checkpoint можно координировать только с учётом реальных transaction boundaries. Один local lock не делает две разные storage systems одной transaction. Стабильный operation key и status lookup обсуждаются в [главе tool execution](<app-tools.md>); повторение network call не обязано повторять business effect при поддержанном server contract.

### Interrupt/resume не равен паузе Python stack на любой строке

Статический `interrupt_before` и динамическое прерывание внутри узла нужно различать. Это разные точки остановки. Static breakpoint ставится относительно node; dynamic interrupt передаёт payload и получает resume value по протоколу runtime. Нельзя автоматически ожидать, что значение `resume="yes"` относится к ещё не достигнутому внутреннему interrupt только потому, что граф остановили перед node.

В [LangGraph interrupt documentation](https://docs.langchain.com/oss/python/langgraph/interrupts) явно указано: при resume соответствующий node исполняется с начала, а не продолжает обычный Python stack с замороженной строки. Значит, код до interrupt может выполниться повторно. Его нужно делать безопасным для replay либо выносить эффекты в подходящие отдельные шаги с нужными guarantees. Порядок нескольких interrupts внутри node также важен для сопоставления resume values.

Учебная трасса: `append_log(); interrupt(); create_report()`. Если append_log не idempotent, один resume добавит вторую log entry до продолжения. Если create_report стоит до interrupt, при replay может повториться и effect. Не надо путать эту особенность runtime с правилом, будто любая функция Python автоматически возобновляема после закрытия процесса.

Не каждую учебную операцию требуется ставить на human approval. Внешний input node нужен там, где задача действительно зависит от missing information/решения человека или выбранной policy. В нашем конспекте разбирается механизм, а не вводится новая процедура согласования текущего редактирования материалов.

### Subgraph как модуль с интерфейсом

Parent state может содержать request, scope и общий budget; analysis subgraph —локальный план и результаты. Передача всего parent dict дочернему module не всегда нужна. Явный mapping входов/выходов уменьшает неоднозначность: child получает разрешённый dataset ID и лимит, возвращает references, usage events и status. Он не должен незаметно заменять parent authorization.

Счётчики child и parent имеют разные единицы. Если parent допускает 3 child invocations, а каждый child делает до 5 model calls, это уже до 15 calls внутри children без synthesis. «Всего три шага» по верхнему graph не означает три обращения к модели. Глобальный accounting должен учитывать вложенную работу и не выдавать каждому child новый полный global budget.

При failure child важно сохранить, какие outcomes подтверждены, а какие нет. Parent не должен превращать child partial в полный success только потому, что module вернул строку. Для реального replay subgraph действует контракт конкретного runtime и checkpointer; одинаковое слово graph не задаёт этих guarantees.

### Streaming, cancellation и trace

Streaming может показывать model tokens, node updates или events выполнения. Это разные уровни наблюдения. Частичный answer preview не равен accepted answer; событие «tool started» не равно «tool succeeded». Для каждого event полезны run/node/call identity и статус, чтобы UI и evaluator не считали начало работы завершённым фактом.

Cancellation пользователем/дедлайном прекращает дальнейшие действия по policy, но уже отправленный внешний запрос может продолжить исполняться. Нужно обработать late result и ambiguous pending status. Deadline guard перед node не прерывает автоматически зависший network call внутри него; для конечного времени нужны и bounds исполнения узлов.

Логировать следует структуру решений и нужное evidence, а не бесконтрольно копировать любой secret в каждый log. Для debugging важны версия graph/prompt/model/tools, входные references, выбранный route, расход и terminal reason. Screenshot графа показывает topology, но не доказывает, что конкретный run пошёл ожидаемым путём.

## Frameworks и local models

LangChain-style components помогают связать templates/models/tools/parsers; graph runtimes — задавать state/transitions/checkpoints; data frameworks — ingestion/retrieval. AutoGen-style participant coordination может использовать messages/handlers и внутренние tool loops. Оно не обязано быть LangGraph instance и не обязано исключать графовое описание на другом уровне.

Ollama/LM Studio могут дать local inference endpoint; orchestrator отправляет context/tools, получает model output, валидирует и исполняет actions. Compatibility части API не доказывает одинаковые chat templates, tool formats, structured generation и cancellation. Выбранные model/runtime/version должны быть tested на одном простом call до сложной многоагентной схемы.

В исходнике каждый agent описывался формулой «LLM+prompt+tools+loop». Это полезный common pattern, но не обязательное определение всех agents: бывают deterministic participants, actors без tools и другие abstractions. Содержательно важно, где decision, state, execution и termination.

### Абстракции LangChain: что именно переиспользуется

В состав приложения могут входить template, model adapter, parser, runnable composition, batch (пакет примеров)/async/stream, retriever (компонент поиска) и tool wrapper. Template параметризует messages; adapter приводит запрос/ответ к общей форме; parser разбирает candidate; composition передаёт output одного stage следующему. Эти функции полезны независимо от бренда библиотеки. Parser не становится constrained decoder (декодер) —это уже подробно разобрано в [главе structured output](<app-structured.md>).

Batch в orchestration API может означать несколько независимых вызовов с concurrency, а не один GPU tensor batch в inference engine. Async позволяет не блокировать ожидание, но не гарантирует ускорение одного forward. Streaming уменьшает время до видимого fragment, но полный result всё равно требует завершения. Retry/fallback нужно настроить по области ошибки: повтор всей chain может повторить уже успешный mutating tool.

В source есть `calculate(expression): eval(expression)` и произвольное выполнение сгенерированного Python. Для изучения арифметики лучше typed handler `multiply(a,b)` или явно ограниченный parser выражений. Возможность сгенерировать строку Python не означает, что её следует исполнять с произвольными правами в учебном примере. Если задача действительно требует code execution, это отдельный executor contract и среда исполнения.

### Стратегии памяти: buffer, window, summary, hybrid

Полный buffer сохраняет все turns, пока они помещаются. Window ограничивает выбранные последние turns/messages, но может потерять раннее условие. Summary сжимает содержание и может ошибиться. Hybrid сохраняет summary старой части и подробный недавний контекст. Это компромиссы, а не одно вечное «правильное production решение».

Допустим, 40 turns дают в среднем по 250 tokens без дополнительной обвязки: 10000 tokens. Последние 8 turns занимают 2000. Summary старых 32 turns размером 600 плюс эти 2000 дают 2600, но построение summary тоже требует работы и проверки. Экономия длины следующего prompt не равна нулевой цене memory maintenance.

Нужно определить единицу window: K messages, K pairs или token budget —разные правила. Граница не должна отрывать tool result от соответствующего call без корректного преобразования протокола. Для долгосрочного факта «использовать только документы 2025 года» summary опасно считать единственным authority: исходный constraint сохраняют отдельно. Для имени/предпочтения memory record должен иметь источник и возможность обновления, а не вечную догадку модели.

Заявление исходника «большинство серьёзных приложений делают память именно так» не подтверждено исследованием. В конспекте сохраняем механизм и trade-offs, а не неподтверждённую популярность. Конкретные memory classes и imports из старого примера не выдаются за текущий обязательный API.

### Четыре framework approaches без жёстких границ

Graph-oriented runtime делает explicit state и transitions. Conversation-oriented coordination описывает участников и обмен messages. Role/task-oriented интерфейс связывает исполнителей с задачами и expected outputs. Data-oriented framework собирает ingestion, index, retrieval и query (запрос) interfaces. Source сопоставлял эти акценты с LangGraph, AutoGen, CrewAI и LlamaIndex; полезно понимать акценты, но нельзя из них вывести невозможность остальных patterns.

Исправляем устаревшее абсолютное противопоставление «LangChain только linear, agents только отдельный LangGraph»: [текущий overview LangChain](https://docs.langchain.com/oss/python/langchain/overview) прямо описывает agents, построенные на LangGraph. AutoGen не требует, чтобы каждый participant был экземпляром LangGraph; при этом messaging и graph model могут существовать на разных уровнях одной системы. Graph также может содержать dynamic routing и fan-out, поэтому «graph всегда статичен, messages всегда эмерджентны» неверно.

В [AutoGen termination examples](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/termination.html) есть программные conditions, в том числе по количеству сообщений и содержимому. Значит, нельзя утверждать, что в conversation framework всякая остановка обязательно решается только LLM. И наоборот, route function в graph может использовать model decision. Вопрос «кто решает» задают к конкретному переходу.

Оценки из source таблицы вроде «production-ready: да/средне», «отладка лёгкая» и «лучший для большинства» не являются измеренными универсальными свойствами. Reliability зависит от graph/application contract, integration versions, storage, failures и checks. Комбинация LangGraph+retriever либо graph+group conversation внутри node возможна, но каждая boundary добавляет serialization, accounting и recovery obligations.

### Подробный ответ о local endpoint и готовых артефактах

Исходный локальный пример предлагал load_embeddings, describe_clusters и get_ablation_row. Первая функция читала .npy и возвращала shape/нормы; вторая читала cluster CSV; третья искала строку метрики в ablation (сравнение с удалённой или изменённой частью метода) table. Это интерфейс к заранее вычисленным данным. Он не обучает embeddings и не запускает clustering заново при каждом вопросе.

Запрос «сравни silhouette для 5 и 10 clusters» требует rows, действительно содержащих эти configurations и сравнимые protocol/data revisions. Shape embeddings и средние расстояния до центроида сами по себе не дают silhouette. Модель должна прочитать соответствующую метрику, проверить условия сравнения и честно указать отсутствие нужных rows. Строка заглушки «результаты поиска: ...» не считается выполненным поиском.

Registry разрешённых artifact IDs лучше произвольного model-generated filesystem path. У артефактов сохраняют dataset/version, preprocessing/encoder (кодировщик входа) revision и evaluation protocol. Если cluster CSV содержит поле size, groupby count считает количество записей, а sum —сумму sizes: в зависимости от формата файла это разные статистики. Код из source нельзя объявить содержательно правильным без описания rows.

При замене endpoint сохраняют graph semantics, но отдельно проверяют template, tool serialization, parser, schema subset, context limits, streaming/cancellation и latency. «Весь агентный код всегда меняется одной строкой» —слишком сильное обещание. В Ollama/LM Studio могут быть свои SDK loops и дополнительные features; границы model/serving (обслуживание запросов к модели)/orchestration относятся к функциям, а не запрету одному продукту объединять уровни.

Не переносим из разбираемого примера фиксированные рекомендации 7–14B, обязательный num_ctx≥16384, проценты успеха и токены/сек как закон. Память зависит от weights, KV configuration, context/concurrency и backend; latency —от реального числа calls, input/output lengths, caching и tools. Низкая temperature (температура распределения выбора) не исправляет отсутствующую способность или неверный template. Для учебного понимания достаточно проследить один реальный protocol cycle; запуск модели и скачивание weights здесь не заявляются.

## Multi-agent coordination и бюджет

Researcher находит sources, reviewer проверяет claims, writer оформляет результат — возможное разделение функций. Польза появляется, если роли дают разные проверяемые operations/evidence. Три одинаковых prompts одной model могут просто втрое повторить один bias.

Coordinator должен определить recipients, message schema, доступный scope, handoff state и общий budget. Nested loops легко создают unbounded repeated work: каждый participant «ещё чуть-чуть поищет», но никто не проверяет global stop. По этой причине single well-defined workflow — обязательный baseline сравнения, а не предположение, что multi-agent хуже/лучше.

Сравнивать total model tokens/tool cost/latency и task success при equal budget. If agent A criticises agent B, это не independent truth; полезность review проверяется обнаруженными исправлениями и независимым oracle.

### Локальная policy и межагентный coordinator

Researcher может выбирать search/read actions, coordinator —выбирать researcher или writer. Один выбор не определяет другой. Участник может быть deterministic executor, retrieval service или human input endpoint без собственного LLM loop. Даже если все три роли используют одну модель, их context/tools могут различаться; одинаковые веса не означают автоматически полную бесполезность разделения, но и разные role names не доказывают полезную независимость.

Handoff должен передавать исходную задачу, применимый scope, evidence references, unresolved questions, статус и budget. Передать writer только красивый prose summary рискованно: он может потерять IDs и условия. Reviewer должен видеть проверяемые claims и sources; согласие двух моделей не заменяет внешний oracle. Если оба читают один ошибочный summary, они могут уверенно повторить одну ошибку.

Самостоятельный выбор помощников можно реализовать как разрешённое действие делегирования с input/result contract. Это концептуальная возможность. Из интерфейса конкретного продукта нельзя достоверно восстановить его внутренний framework, prompts или точные лимиты. Упоминание subagents в учебном источнике не означает, что в этой работе уже создавались дополнительные агенты.

### Стоимость координации и честное сравнение

Source пример распределял 100 документов между 10 researchers, по 10 каждому. При независимых задачах и достаточной capacity идеальный parallel path может сократиться примерно в 10 раз до synthesis, но суммарное чтение 100 документов не исчезает. Если каждый из 10 вызовов дополнительно повторяет 1000 tokens instructions, это 10000 instruction tokens против 1000 у одного вызова, без учёта различий context limits и числа calls. Ускорение elapsed time не тождественно сокращению token work.

Пусть последовательная обработка занимает 100 s, а parallel research 10 s плюс synthesis 8 s при игнорировании overhead. Speedup 100/18≈5.56, а не 10. Общий выигрыш ограничен serial частью, capacity backend и очередями. Если synthesis требует перечитать всё, оно может стать новым bottleneck.

Сравнение требует одинаковых задач, data access, критериев correctness и суммарного бюджета. Один агент с тремя обоснованными stages может быть сильным baseline. Multi-agent может помочь независимыми tools/evidence и параллельностью, но role conflict сам по себе не гарантирует роста качества. Проверяют, какие ошибки обнаружены, какие новые внесены и насколько вырос cost per correct task.

### Self-correction не означает поиск значимости до победы

Перебор статистических тестов до появления значимости не является корректным общим правилом исправления ошибки. Как общий пример self-correction это неверно. Исправлять неправильно выбранный тест при несоблюдённых предпосылках —одна задача. Перебирать тесты/группы/признаки, пока p-value не станет меньше 0.05, —другая: меняется процедура статистического вывода и риск ложных находок.

В идеализированном случае 20 независимых true-null tests с вероятностью false positive (ложное срабатывание) 0.05 каждый вероятность хотя бы одного равна $1-0.95^{20}\approx0.641514$. Автоматический loop усиливает эту проблему, если критерий enough —«нашёл значимое». Для аналитического workflow нужно заранее определять hypothesis/protocol и учитывать exploratory/multiple-testing характер работы, а не выдавать удачно найденный p-value за подтверждённый вывод.

Точно так же `len(findings)>=5` не означает полноту исследования. Source extract-loop каждый раз проходил по всем накопленным sources и дописывал findings; один и тот же source мог учитываться повторно и искусственно достичь пяти. Нужны identity/deduplication и покрытие вопросов, а не только длина списка. Полезный progress marker —получено новое применимое evidence или устранён конкретный validation (проверочная часть для выбора решения) failure.

### Три поиска: трасса успеха и трасса исчерпания

Для учебного cap3 searches счётчик увеличивается при каждой реально начатой попытке, включая неуспешную, если это выбранная единица лимита. Первый поиск даёт нерелевантный документ, второй —нужный v8 passage: можно ответить и завершиться после validation, не расходуя третий. Если все три не дали evidence, следующий route —partial/abstain, а не четвёртый поиск под другим именем node.

Три search executions не ограничивают автоматически число model calls, validation retries и время зависшего tool. Поэтому global caps задаются отдельно по нужным units. Invalid candidate после третьего поиска можно отклонить без дальнейшего retrieval; новый generation step возможен только если есть соответствующий model/token budget. В source local loop cap iteration проверялся до обработки очередного proposal: на границе model мог уже попросить tool, но graph завершался, не исполнив call. Такой terminal state должен быть обозначен как budget-exhausted/partial, а не возвращён как готовый final answer.

Для каждого пути называют terminal reason: accepted answer, missing input, no applicable evidence, permanent failure, budget exhausted или cancelled. Это делает цель процесса проверяемой и позволяет читать trace как объяснение результата. Сам факт достижения END сообщает об окончании исполнения, а не автоматически об успешности задачи.

## Самопроверка и практика

1. Чем graph отличается от обучения policy? **Ответ:** graph организует выполнение; policy training меняет параметры по data/objective.
2. Что потеряет last-write-wins evidence reducer? **Ответ:** результаты parallel branch, обновлённые раньше другого.
3. Гарантирует ли checkpoint отсутствие duplicate actions? **Ответ:** нет, external side effect и сохранение checkpoint могут быть неатомарны.

4. Проверяет ли TypedDict внешний JSON при исполнении? **Ответ:** сам по себе нет; нужны runtime checks.
5. Что возвращать при append reducer? **Ответ:** новый delta; полный accumulated list может задублировать историю.
6. Сохранит ли dict union все вызовы одного tool? **Ответ:** нет, совпавший key заменяется; нужны call/result identities.
7. Является ли thread_id правом доступа? **Ответ:** нет, выбор/доступ к execution history проверяет приложение.
8. Возобновляется ли dynamic interrupt с сохранённой Python строки? **Ответ:** node может исполниться с начала; pre-interrupt effects требуют правильной replay policy.
9. Значит ли END, что ответ верен? **Ответ:** нет, остановка и success —разные статусы.
10. Почему пять findings ещё не enough evidence? **Ответ:** это могут быть дубли/нерелевантные claims без покрытия задачи.
11. Исправление ли анализа —перебирать тесты до p<0.05? **Ответ:** нет; это меняет процедуру вывода и требует учёта множественных проверок.
12. Гарантирует ли graph детерминированный результат? **Ответ:** нет, model sampling, external data и parallel merge могут различаться.

**Практика:** спроектировать graph с retrieve/rewrite/answer/validate и global cap3searches. Добавить checkpoint после search; описать crash перед сохранением observation. Показать recovery отдельно для read-only search и report-creation action. Код framework пока не нужен: сначала проверяется semantics.

## Результат главы и переход дальше

Следующие темы: [RAG: BM25, dense, hybrid и multi-vector retrieval](<app-rag.md>), [Постановка обучения агента и baseline](<../06-agent-learning/agent-task.md>).

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
