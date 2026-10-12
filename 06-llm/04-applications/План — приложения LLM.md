# План: приложения на основе LLM

План объединяет structured output, вызов инструментов, orchestration и RAG. Общий inference/serving маршрут остаётся в [плане inference](../03-inference/%D0%9F%D0%BB%D0%B0%D0%BD%20%E2%80%94%20%D0%B8%D0%BD%D1%84%D0%B5%D1%80%D0%B5%D0%BD%D1%81%20LLM.md); исторический конспект и точные источники — в [notes](notes/README.md).

## Темы

1. Structured output: schema-constrained decoding, parser, syntactic и semantic validation.
2. Tool calling: контракт аргументов, проверка прав, execution, ошибки, timeout и возврат observation.
3. Orchestration: state, loop, routing и завершение; framework abstractions без предположения, что веса модели обучаются.
4. RAG: ingestion/chunking → sparse/dense/hybrid retrieval → reranking → grounded response.
5. Оценка: отдельно retrieval, answer correctness, citations, task success и стоимость.

**Checkpoint:** проследить один запрос через retrieval и tool call, указав, где проверяются формат, смысл, права и результат.
