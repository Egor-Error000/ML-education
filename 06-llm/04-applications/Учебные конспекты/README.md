# Приложения LLM — подробные конспекты по вашим чатам

[Версия для чтения с набранными формулами](<../../../Читать конспекты.html#course-9-0>) · работает без интернета.

[План](<../План — приложения LLM.md>) · [Общее оглавление](../../../Учебные%20конспекты.md) · [Предыдущая короткая заметка](../notes/llm-applications.md)

Пять пунктов плана раскрыты в пяти главах. Основной материал восстановлен из S5/S6; все пять глав А–Д дополнительно углублены по source schemas, tool examples и state/graph/memory/coordination примерам. В В разобраны ошибки исходного кода; в А добавлен контрпример локальной GBNF grammar. В Г дополнены ingestion, chunking/overlap, corrected embeddings/ANN/RRF и multi-vector cost; подробный retrieval-разбор сохраняет ваши вопросы о BM25, embeddings и размере multi-vector базы. В Д разобраны units/qrels, macro/micro/ranking metrics, citations/coverage, oracle, retry budget, judge и paired comparisons; S3:820–918 используется с явными исправлениями. Формулы оформлены в LaTeX, практические snippets не выдаются за ваши результаты.

- [А. Structured output: schema и смысл](<А — Structured output, schema и смысл результата.md>) — пункт1.
- [Б. Tool calling: execution, errors, retries, idempotency](<Б — Tool calling, execution, ошибки и idempotency.md>) — пункт2.
- [В. State, graph, checkpoints и agent participants](<В — State, orchestration, графы и участники агента.md>) — пункт3.
- [Г. RAG, BM25, dense/hybrid, reranker и multi-vector](<Г — RAG, BM25, dense, hybrid и multi-vector retrieval.md>) — пункт4.
- [Д. Evaluation retrieval/answer/citations/task success](<Д — Evaluation retrieval, ответов, citations и task success.md>) — пункт5.

## Источники и сохранённые вопросы

**S5:** [chat-LLM Study Plan Status.txt](<../../../archive/study-sources/chat-LLM Study Plan Status.txt>):120–242 — structured output;1463–1515 — templates/parser (не автоматический constrained decoder);1038–1817 — orchestration/chains;1821–2822 — state graph/local runtimes;2826–3041 — начало tool calling.

**S6:** [chat-Пункт D4.txt](<../../../archive/study-sources/chat-Пункт D4.txt>):4–178 — tools;182–833 — RAG и различие embeddings;837–1133 — BM25;1134–1373 — multi-vector storage/поиск;1377–2197 — frameworks/agent loops. Последующие deployment/scaling раскрыты в [инференсе](../../03-inference/Учебные%20конспекты/README.md).

Сохранены вопросы: где исполняется tool; как model endpoint Ollama/LM Studio связывается с loop; обязан ли participant быть LangGraph graph; зачем отдельный retrieval embedding; откуда берутся BM25 terms; какой storage overhead у нескольких representations.

**Исправления:** schema/parser/training не смешиваются; timeouts не равны empty search; idempotency/checkpointing имеют разные guarantees; значения закрытых runtime support/процентов не объявляются вечными фактами. Ваше уточнение S6:179–181 о теоретическом режиме соблюдено: готовое приложение и deployment не приписываются.

## Checkpoint

Проследить один запрос через corpus version/ACL → retrieval → context → tool call/observation → answer → validation. На каждом этапе назвать input/output, возможную ошибку и oracle. Самостоятельно пересчитать BM25, storage, MRR/nDCG и cost per success. Обучение encoder/reranker подробно относится к [плану BERT](<../../../07-bert/План — BERT и encoder-модели для NLP.md>); обучение agent trajectories — к [lifecycle](<../../05-lifecycle/План — жизненный цикл LLM и обучение агентов.md>).
