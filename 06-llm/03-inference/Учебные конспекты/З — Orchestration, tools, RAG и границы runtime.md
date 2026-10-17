# З. От model forward к приложению: state, tools и retrieval

[Оглавление](README.md). Блок D.1–D.6; общая механика без привязки к одному framework API.

**Основа:** S5, строки 1038–3041; S6, 4–2197. В S6:179–181 вы прямо уточнили: сейчас изучается материал, а не создаётся агент. Поэтому ниже учебная система, без утверждений о вашем deployment. **Углубление:** явный state machine и границы ответственности.

## 1. Какие компоненты добавляются вокруг модели

Один model call возвращает текст/структуру. Приложение должно получить user input, подготовить context, при необходимости найти документы, исполнить проверенные tools, сохранить state и завершить результат. Orchestration задаёт порядок/условия этих операций.

Linear chain подходит для фиксированного пути `retrieve→generate→validate`. Graph/state machine нужен, когда есть branches, retries, tool loop и early termination. Model weights от изменения graph не обновляются: runtime исполняет существующую policy, обучение policy требует отдельного objective/data.

Пример state:

```text
question, authorized_scope, messages, evidence,
pending_calls, observations, attempts, budget, status.
```

Trusted constraints, например права на документы, задаются приложением. Их нельзя заменять модельным summary из chat history. Node получает нужные поля и возвращает определённые изменения; merge policy для parallel updates должна быть явной.

## 2. Один tool loop по шагам

Модель предлагает `search_documents(query,limit)`. Приложение проверяет known tool name, types/ranges, scope и remaining budget. После реального исполнения observation содержит найденные IDs/snippets или structured error. Это возвращается модели как tool result, а не как новая управляющая инструкция.

Повторить call можно по policy: temporary read failure допускает bounded retry, invalid args требуют correction, отсутствие права не исправляется бесконечными attempts. После enough evidence модель создаёт final answer; после лимита шагов система завершает задачу понятным partial/failure status.

Call ID связывает request и observation, особенно при нескольких parallel tools. Timeout не доказывает, что внешнее действие не произошло. Поэтому retry mutable operation требует idempotency/сверки состояния. Read-only учебная среда упрощает этот риск, но concept нужен для понимания production.

Tool description и schema — interface, не training. Если маленькая локальная модель плохо соблюдает tool format, framework не создаёт способности автоматически. Нужно проверить native template, parser, supported schema и примеры prompt, затем при необходимости адаптировать model/data.

## 3. RAG: retrieval добавляет текстовый evidence

Pipeline: corpus ingestion → chunks с IDs/provenance → sparse/dense index → query retrieval → optional reranking → context selection → generation с citations. Generator обычно получает **исходные тексты**, выбранные по vector scores, а не произвольный embedding vector вместо слов.

Это отвечает вопросу чата «почему модель поймёт наш embedding». Retriever понимает совместимое query/document space, приложение по IDs достаёт snippets, LLM читает их как text. Input token embeddings LLM и retrieval embeddings имеют разные roles/objectives.

BM25 использует term frequencies, corpus document frequencies и length normalization. Dense retriever сравнивает learned vectors. Hybrid объединяет rankings/scores по объявленной policy, reranker оценивает query-document pairs. Больше stages не гарантирует improvement: each может добавить latency и failure mode.

FAISS — библиотека vector search; полноценная система ещё хранит documents/metadata, versions, access filters, deletion/update semantics. ANN approximation оценивается относительно exact search отдельно от relevance quality. Стандартный vector index не должен выдавать document вне разрешённого scope только из-за большой similarity.

## 4. Frameworks: abstraction по потребности

LangChain-style components дают templates, model/tool adapters, chains и parsers; LangGraph-style graph описывает state/transitions/checkpointing. LlamaIndex делает акцент на data/retrieval integrations. AutoGen/CrewAI исторически предлагали conversation/role coordination patterns. Их APIs меняются; выбирать нужен набор capabilities, а не обязательное прохождение каждого framework.

Один agent participant может иметь свой loop с tools, state и termination. Это не требует быть реализованным через LangGraph: тот же механизм можно написать другим runtime. Multi-agent coordination добавляет messages, duplicated context и termination difficulties; качество сравнивать с single-agent baseline при одинаковом total budget.

Локальные Ollama/LM Studio models могут быть model endpoints такого приложения. Ответ на исходный вопрос о локальном agent pipeline: runtime вызывает endpoint, валидирует output, исполняет tools и повторяет call. Встроенные features конкретного приложения могут часть этого делать, но supported API/tool modes проверяются отдельно.

## 5. Как разделить ошибки

Retrieval failure: evidence не найдено. Context selection failure: найдено, но не передано/обрезано. Generation failure: evidence есть, ответ неверный. Tool failure: wrong tool/args или ошибка исполнения. State failure: потерян результат/ID, повторено действие. Такие категории позволяют выбрать intervention точнее, чем «агент тупит».

Oracle context experiment подаёт gold snippet вместо retrieval. Если ответ теперь correct, bottleneck вероятно upstream; если нет — prompt/model/evidence interpretation требуют разбора. JSON validity и tool HTTP success — локальные diagnostics, final task success оценивается отдельно.

1. Обучает ли graph routing веса? **Ответ:** нет, без отдельного training процесса.
2. Передаёт ли ordinary RAG vector напрямую generator? **Ответ:** обычно нет, vector выбирает IDs, затем подаётся текст evidence.
3. Обязан ли каждый AutoGen agent быть LangGraph graph? **Ответ:** нет, это разные уровни/реализации loop/coordination.

**Практика:** на бумаге проследить запрос «найди правило таймаута и процитируй» через scope→retrieval→tool observation→answer→validation. Добавить no-results, timeout и forbidden-document случаи. Более подробный прикладной материал/исходные вопросы — в [разделе приложений](<../../04-applications/План — приложения LLM.md>).
