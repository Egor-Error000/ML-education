# Применение LLM: structured output, orchestration, RAG и production

Конспект покрывает структурированный вывод, tool calling, orchestration, retrieval и deployment. Теоретический разбор framework/API не подтверждает созданный пользователем агент или рабочую production-систему.

## 1. Structured output и вызов инструментов

Constrained decoding ограничивает допустимые следующие токены грамматикой/schema, если это поддерживает backend. Поддержка JSON Schema часто является подмножеством полной спецификации. Валидный JSON не гарантирует верные значения, уместность выбранного инструмента, безопасность аргументов или успешное завершение.

Output parser разбирает уже полученный текст и сам по себе не доказывает, что во время генерации применялась logit mask. Разделяйте schema/format validation, semantic validation, авторизацию, вызов инструмента, обработку ошибки/timeout и передачу результата обратно модели. Объявление tool в API задаёт доступный интерфейс и не обучает веса модели на каждом запросе.

Типичный цикл: модель просит действие → приложение валидирует параметры и права → приложение вызывает tool → возвращает наблюдение → модель продолжает или завершает. Нужны лимиты повторов и ясная обработка отказа, timeout и неполного ответа. (C11.)

## 2. Orchestration, frameworks и локальные модели

LangChain/LangGraph, AutoGen и CrewAI предоставляют разные abstractions для chains, state, graphs, handoffs и координации. Их роли и API меняются; названия не определяют автоматически семантику всей системы. Agent loop обычно включает state, observation, решение/действие и условие завершения, но сам loop не равен обученной agent policy.

Ollama, LM Studio и серверы с OpenAI-compatible endpoints могут иметь различия в tools, streaming, schema, cancellation и checkpoint behavior. Совместимость одного endpoint не гарантирует одинаковую поддержку остальных возможностей. Сниппеты в исходных чатах — учебные, их запуск не подтверждён. (C14.)

RAG и framework orchestration не означают, что обучен агент. Agentic training — отдельная тема: данные траекторий, среда, reward/evaluation и обучение policy.

## 3. Retrieval augmented generation

RAG получает внешние документы/фрагменты и добавляет их к контексту запроса. Пайплайн обычно включает ingestion, chunking, индексацию, retrieval, reranking/selection, сбор контекста и генерацию. Internal token embeddings модели и retrieval embeddings — разные представления и задачи; даже если используются одной моделью, их качество проверяется на retrieval task.

BM25 — lexical retrieval, основанный на совпадениях и term statistics; dense retrieval использует embedding similarity. Hybrid retrieval сочетает сигналы. Multi-vector retrieval хранит несколько представлений на документ/фрагмент и меняет storage/recall trade-off. Стоимость хранения считать как vectors + index overhead + metadata, а не по одному универсальному проценту.

Оценивать отдельно retrieval recall/precision/ranking и качество ответа/grounding/citations. Высокое качество генератора не исправляет отсутствие нужного фрагмента, а успешный retrieval не гарантирует корректный ответ. Метрики и пороги выбирать на целевой задаче; непроверенные проценты улучшения из чата не переносить как benchmark claims. (C13.)

## 4. Deployment, cold start, очереди и масштабирование

Cold start включает загрузку процесса, модели/весов, инициализацию kernels, прогрев и готовность принимать запросы. Модель в локальном volume сокращает часть загрузки, но не означает прогретые kernels или заполненный cache. Различайте startup, readiness и liveness checks: они отвечают на разные вопросы.

Scaling учитывает concurrency, очередь, throughput, latency percentiles, capacity, memory, cold-start time и hysteresis. Одного универсального threshold нет: слишком частое масштабирование может вызывать колебания; чрезмерно медленное — рост очереди. Autoscaling example/YAML остаётся иллюстративным, если не проверен на конкретной системе.

## Прогресс и происхождение

Вопросы пользователя, сохранённые по смыслу: гарантирует ли schema валидность действия; как различить retrieval и внутренние embeddings модели; что происходит при tool execution и ошибках; как не принимать runtime snippets и числа за проверенный результат.

- S5, `chat-LLM Study Plan Status.txt`: строки 120–242 (structured output), 1038–1817 (orchestration/LangChain), 1821–2822 (D3/local runtimes), 2826–3041 (начало tool calling). [Архив](../../../archive/study-sources/chat-LLM%20Study%20Plan%20Status.txt).
- S6, `chat-Пункт D4.txt`: строки 4–178 (tool execution), 182–833 (RAG), 837–1373 (BM25/multi-vector), 1377–2197 (frameworks/loops), 2201–3009 (deployment/cold start), 3013–3904 (autoscaling/queues/latency). [Архив](../../../archive/study-sources/chat-%D0%9F%D1%83%D0%BD%D0%BA%D1%82%20D4.txt).

Обсуждение D1–D6, E1–E2 подробное; пользовательское создание агента, запуск кода, deployment или измерения не подтверждены. Проверить себя: отличить schema validity от semantic correctness; нарисовать путь одного tool call с validation/timeout; объяснить, почему retrieval и answer metrics нужны отдельно.
