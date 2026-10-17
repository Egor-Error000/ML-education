# План: Инференс LLM и других моделей — от квантизации до агентных фреймворков

Подробное учебное изложение по вашим чатам: [Учебные конспекты — 10 глав](Учебные%20конспекты/README.md). Новые E.3–E.4 явно отделены от восстановленной истории.

История прохождения: [заметка по inference](notes/llm-inference.md), [приложения и RAG](../04-applications/notes/llm-applications.md), [исследование DSpark](notes/llm-inference-research.md); полный учёт и уровни обсуждения/понимания/практики — в [реестре](../migration/PROGRESS.md). Теория A/B/C/D и E.1–E.2 обсуждена; после [поправок](../migration/CORRECTIONS.md) продолжить **E.3 → E.4**, не начинать с quantization заново. Занятие — одна тема с 2–4 смысловыми разделами; лабораторные ниже предназначены для отдельного закрепления.

Принцип прохождения тот же, что и в предыдущих темах: сначала концепция/механика
в отрыве от конкретного инструмента (почему это вообще нужно, как работает на
уровне идеи), потом конкретные библиотеки/фреймворки как реализация этой идеи.
Порядок — от "внутренностей одной модели" наружу, к "экосистеме приложений
вокруг модели".

---

## Блок A — Оптимизация весов и вычислений модели

### A.1 Квантизация — основы
- Числовые форматы: FP32 → FP16/BF16 → INT8 → INT4, что теряется при переходе
  (разрядность мантиссы/экспоненты, разница FP16 vs BF16 по диапазону значений)
- Symmetric vs asymmetric quantization, per-tensor vs per-channel vs per-group
  квантизация — почему более гранулярная схема точнее, но дороже по метаданным
- Calibration — как выбираются scale/zero-point для активаций без переобучения

### A.2 Post-Training Quantization (PTQ) vs Quantization-Aware Training (QAT)
- PTQ: быстро, не требует переобучения, но теряет больше точности на агрессивных
  битностях
- QAT: имитация квантизации во время обучения (fake quantization nodes),
  точнее, но дороже вычислительно
- Когда какой подход оправдан — зависимость от бюджета времени и целевой битности

### A.3 Продвинутые схемы квантизации для LLM
- **GPTQ** — послойная квантизация с компенсацией ошибки через Hessian-based
  подход
- **AWQ (Activation-aware Weight Quantization)** — идея защиты "важных" весов
  на основе статистики активаций, а не квантизации всех весов одинаково
  агрессивно
- **GGUF** — формат хранения в экосистеме llama.cpp, поддерживающей CPU/GPU и offload; **GGML** — tensor-библиотека/историческая экосистема. Отличать формат файла от метода quantization и от
  GPTQ/AWQ по целевой платформе и структуре файла
- Сравнение по компромиссу качество/скорость/размер на практике (эмпирически,
  не только в теории)

### A.4 Pruning
- Structured pruning (каналы, головы, слои) может ускорить inference при подходящих shapes/kernels; выигрыш проверять замером, не числом удалённых параметров.
- Unstructured pruning (отдельные веса) — требует sparse-инференса для выигрыша,
  иначе прирост только в размере файла, не в скорости
- Magnitude-based pruning vs более сложные критерии важности весов
- Итеративный pruning + fine-tuning цикл

### A.5 Граница с обучением: distillation

Teacher/student, soft targets, response/reasoning/on-policy distillation описаны в [адаптации](../02-adaptation/%D0%9F%D0%BB%D0%B0%D0%BD%20%E2%80%94%20%D0%B0%D0%B4%D0%B0%D0%BF%D1%82%D0%B0%D1%86%D0%B8%D1%8F%20LLM.md). В этом плане сравнивается эксплуатационный результат student: task quality, latency, throughput и память на том же hardware/workload.

---

### A.6 KV-cache — специфика авторегрессивной генерации
- Почему attention нужно кэшировать key/value для уже сгенерированных токенов
  (не пересчитываем K/V префикса; attention нового токена к истории всё равно растёт с длиной контекста)
- Рост памяти KV-cache с длиной контекста и batch size — почему это главный
  ограничитель batch/context; bottleneck зависит от prefill/decode, hardware и workload, а не всегда только от памяти
- Multi-Query Attention (MQA) и Grouped-Query Attention (GQA) — архитектурные
  решения уменьшения размера KV-cache на уровне модели, до всякого serving-слоя
- Квантизация самого KV-cache (отдельно от весов модели)

---

## Блок B — Serving-движки: инференс на потоке запросов

### B.1 Проблема батчинга для генеративных моделей
- Почему naive static batching плохо работает для LLM: разная длина
  генерации у разных запросов в батче → простой GPU на "хвостах"
- **Continuous batching** — динамическая подстановка новых запросов в
  освобождающиеся слоты батча

### B.2 vLLM и PagedAttention
- Механика PagedAttention: KV-cache как виртуальная память со страницами,
  аналогия с paging в ОС
- Проблема фрагментации памяти при непрерывном выделении KV-cache и как paging
  её решает
- Практика: запуск vLLM, throughput vs latency trade-off, параметры батчинга

### B.3 Альтернативные serving-движки — сравнение подходов
- **TensorRT-LLM** (NVIDIA) — компиляция графа под конкретное железо, глубокая
  аппаратная оптимизация, менее гибко чем vLLM по смене моделей
- **Text Generation Inference (TGI, HuggingFace)** — концептуальная близость к
  vLLM, отличия в экосистеме и поддерживаемых моделях
- **llama.cpp / Ollama** — локальный путь с CPU/GPU/offload и GGUF; для локального
  инференса без выделенного GPU-сервера (ваш текущий кейс в фазе 4 проекта)
- Критерии выбора движка под задачу: GPU vs CPU, throughput vs простота
  деплоя, поддержка нужной архитектуры модели

### B.4 Инференс не-LLM моделей — для контраста
- ONNX Runtime — универсальный формат экспорта, независимость от исходного
  фреймворка обучения
- torch.compile и актуальный export-путь PyTorch — отдельно проверить совместимость графа и backend; TorchScript изучать для существующих проектов, сверяя статус поддержки.
- Отличие serving-паттернов для CV-моделей (обычно stateless, единичный forward
  pass) от LLM (stateful генерация, KV-cache) — почему подходы, оптимальные для
  LLM, не переносятся напрямую на CV inference

---

## Блок C — Inference-time техники ускорения и контроля генерации

### C.1 Speculative Decoding
- Механика: маленькая draft-модель генерирует несколько токенов вперёд,
  большая модель верифицирует их параллельно за один forward pass
- При корректной rejection/acceptance-схеме сохраняется целевое распределение токенов; это не означает идентичный текст при одном seed. Реальный speedup зависит от acceptance rate и стоимости draft/verifier.
- Варианты: с отдельной draft-моделью vs self-speculative подходы

Углубление по истории: acceptance/residual sampling и границы переноса на beam search. Case study [DSpark](https://arxiv.org/abs/2607.05147): semi-autoregressive draft и confidence-scheduled verification; условия измерений первоисточника сохранять. Числа из примеров чатов не считать собственным benchmark.

### C.2 Structured Output / Constrained Decoding
- Принудительное ограничение пространства генерации по грамматике
  (JSON schema, regex, context-free grammar)
- Механика на уровне сэмплирования: маскирование невалидных токенов на каждом
  шаге по текущему состоянию грамматики
- Прямое применение: надёжный tool-calling в агентных системах (относится к
  вашему LangGraph-агенту в фазе 4)

### C.3 Стратегии сэмплирования
- Greedy, beam search, top-k, top-p (nucleus), temperature — механика каждой,
  компромисс детерминированность/разнообразие
- Почему это не "инфраструктурная", а полноценная часть качества генерации

### C.4 Техники работы с длинным контекстом
- Sliding window attention — механика, где теряется информация
- Квантизация KV-cache под длинный контекст (связка с A.6)
- Context caching / prompt caching у некоторых API-провайдеров — переиспользование
  вычисленного KV-state для повторяющегося префикса промпта между запросами

---

## Блок D — Оркестрация приложений вокруг LLM

### D.1 От промпта к пайплайну — зачем нужна оркестрация вообще
- Проблема "одного вызова LLM недостаточно": нужен контекст, инструменты,
  память, повторные попытки — идея до всякого фреймворка

### D.2 LangChain
- Концепция chains — последовательные вызовы LLM/инструментов
- Prompt templates, output parsers, memory-модули (базовая, буферная,
  суммаризирующая память)
- Почему линейные chains плохо подходят для сложной ветвящейся/циклической
  логики — мотивация перехода к graph-based подходу

### D.3 LangGraph (уже частично на практике в проекте)
- Граф состояний: узлы как шаги обработки, рёбра — условные переходы
- Циклы и agentic loops — повторные попытки, self-correction паттерны
- State management между узлами графа
- Как это ложится на ваш кейс: агент как tool-calling интерфейс к готовым
  артефактам (эмбеддинги, кластеры, ablation-таблица)

### D.4 Tool / Function Calling — исполнение готовой policy

Schema и описание tool → model output → parser/schema validation → проверка допустимости действия → execution → результат в state → следующий шаг. Constrained decoding ограничивает синтаксис, но не гарантирует правильный tool, смысл аргументов и допустимость операции.

Timeouts, retries только для повторяемых операций, idempotency keys, лимиты шагов/токенов и обработка tool errors. Независимые read-only вызовы можно параллелить; зависимые действия требуют последовательности. Учебная среда — read-only инструменты.

Обучение выбора tools, SFT на траекториях, recovery и RL — в [lifecycle](../05-lifecycle/%D0%9F%D0%BB%D0%B0%D0%BD%20%E2%80%94%20%D0%B6%D0%B8%D0%B7%D0%BD%D0%B5%D0%BD%D0%BD%D1%8B%D0%B9%20%D1%86%D0%B8%D0%BA%D0%BB%20LLM%20%D0%B8%20%D0%BE%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D0%B5%20%D0%B0%D0%B3%D0%B5%D0%BD%D1%82%D0%BE%D0%B2.md). Граф/state runtime не изменяет веса модели сам по себе.

---

### D.5 RAG (Retrieval-Augmented Generation)
- Мотивация: ограниченность контекстного окна + устаревание знаний модели
- Pipeline: документы → [чанкинг и provenance](../../01-data/%D0%9F%D0%BB%D0%B0%D0%BD%20%E2%80%94%20%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D0%B5%20%D0%B8%20%D0%BA%D0%B0%D1%87%D0%B5%D1%81%D1%82%D0%B2%D0%BE%20ML-%D1%8D%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D0%B0.md) → embeddings/index → retrieval → augmented prompt. Здесь изучать retrieval и генерацию, без повторения общей обработки корпуса.
- Векторные БД: FAISS (локальная, библиотека) vs Chroma/Weaviate/Pinecone
  (полноценные БД с индексами) — разница в масштабируемости и функциональности
- Метрики качества RAG: retrieval recall/precision отдельно от качества
  финальной генерации — типичная ошибка мерить только end-to-end
- Продвинутые схемы: hybrid search (dense + sparse/BM25), re-ranking,
  query rewriting/expansion

### D.6 Агентные фреймворки — сравнительный обзор
- LangGraph vs **AutoGen** (multi-agent conversation-центричный подход) vs
  **CrewAI** (role-based агенты) — разные философии организации многоагентных
  систем
- **LlamaIndex** — акцент на data framework для RAG, пересечение и отличие
  от LangChain по зоне ответственности
- Когда оправдана multi-agent архитектура, а когда это overengineering
  поверх одного хорошо спроектированного графа

---

## Блок E — Инфраструктура и продакшн-паттерны

### E.1 Контейнеризация и деплой inference-сервисов
- Docker-паттерны для serving (ваш существующий опыт — прямое применение)
- Разделение "холодного старта" (загрузка весов) и "горячего" инференса
- GPU-шаринг между несколькими моделями/тенантами через batching

### E.2 Масштабирование под нагрузку
- Автоскейлинг inference-подов, метрики для скейлинга (latency percentiles,
  очередь запросов, а не просто CPU/GPU utilization)
- Trade-off latency vs throughput при выборе размера батча в проде

### E.3 Кэширование на уровне приложения
- Semantic caching — кэш ответов по семантической близости запроса
  (эмбеддинг запроса + similarity search), а не точному совпадению строки
- Отличие от prompt caching (Блок C.4) — уровень абстракции (кэш вычислений
  модели vs кэш готовых ответов приложения)

### E.4 Мониторинг и оценка LLM в проде
- Специфичные проблемы мониторинга генеративных систем: нет единственного
  "правильного" ответа для сравнения, в отличие от классификации
  (перекликается с темой метрик из CV/NLP-задач)
- LLM-as-judge подход, его ограничения и bias
- Отслеживание дрифта качества ответов во времени

---
## Практический порядок и углубление

[Единый маршрут](../../README.md). Ядро приложения: C.3 → C.2 → D.1/D.3/D.4 → D.5 по задаче → evaluation. Системная ветка: A.1–A.3/A.6 → B → E. Distillation изучается в адаптации; pruning/QAT и сравнение всех frameworks необязательны.

### 1. Измерение до оптимизации — ядро

Зафиксировать model revision, tokenizer/template, hardware, runtime, prompt/output lengths и concurrency. Различать prefill (обработка входа), decode (новые токены), time to first token, inter-token latency, end-to-end latency и tokens/s. Измерить p50/p95, queue time, peak memory и OOM/error rate. Указывать warm/cold run и способ синхронизации GPU.

**Практика:** короткий/длинный prompt × короткий/длинный output × concurrency; строить tradeoff latency/throughput по workload. Заранее задать минимально приемлемое качество и latency budget.

### 2. Quantization и память — ядро

Простейший affine quantizer: q=clip(round(x/s)+z), x_hat=s(q-z). На маленьком векторе сравнить symmetric/asymmetric и granularity. Calibration data должны представлять рабочие inputs; metadata и dequantization overhead входят в memory/runtime.

AWQ scaling использует activation statistics, а не обязательно хранение всех «важных» весов в высоком dtype. GPTQ и AWQ — алгоритмы; INT4/NF4 — представления; GGUF — формат. QAT не гарантирует выигрыш качества при любом budget.

Весовая память приблизительно P*b/8 байт до metadata. KV-cache приблизительно 2*L*B*T*H_kv*d_head*bytes_per_element до overhead/sharding/особенностей архитектуры. Подставить параметры выбранной модели; различать MHA/GQA/MQA.

**Практика:** FP/BF baseline vs один доступный quantization формат; task quality, format/tool correctness, VRAM, TTFT и throughput. Сжатие не гарантирует ускорения; loss качества в structured output измерять, а не предполагать универсально больший.

### 3. Sampling, контекст и speculative decoding — ядро + углубление

Вручную посчитать softmax(logits/T), top-k/top-p и нормировку; greedy — отдельный режим, а не деление на T=0. Beam search имеет свой objective/length penalty, не обязателен для диалогового sampling.

**Практика:** фиксированные задачи при нескольких decoding settings, качество и валидность схемы. Проверить truncation и «lost in the middle». Sliding window возможен при поддержке архитектуры/runtime и не является бесплатным расширением обученного контекста.

Speculative decoding — углубление: acceptance rate и стоимость draft/verifier; доказательство сохранения распределения отдельно от наблюдаемого speedup.

### 4. RAG — ядро для задач со знаниями

BM25/sparse baseline vs dense embeddings → hybrid → reranking при необходимости. ANN index: recall-speed-memory; если индекс приближённый, отдельно измерить ANN recall относительно exact search. FAISS — библиотека поиска, не полный document store/права доступа.

Фиксированный набор вопросов, релевантных документов и эталонных ответов: Recall@k/MRR/nDCG для retrieval; correctness, groundedness, citation accuracy и abstention для ответа. Контроль oracle context позволяет отделить retrieval failure от generation failure. Chunking принадлежит данным, а selection k/reranking — этому плану.

**Практика:** no-retrieval → BM25 → dense/hybrid на одном корпусе; затем reranker. Проверить отсутствие ответа, противоречивые/устаревшие источники и права доступа к документам. Инструкции из retrieved text не должны становиться управляющими инструкциями tools.

### 5. Граф исполнения, serving и эксплуатация — ядро по необходимости

Начать с простого Python-loop/state machine; LangGraph/LangChain/LlamaIndex и другие frameworks выбирать по нужным возможностям, не изучать каждый API подряд. State, checkpoint, resume, retries и budget должны иметь явную семантику.

Continuous batching, paged KV-cache и prefix caching решают разные задачи. Один memory-saving приём не гарантирует меньшую p95 latency. Совместимость vLLM/TensorRT-LLM/llama.cpp/Ollama/TGI и статус проектов сверять перед практикой, фиксировать версию и backend.

Cache: ответ приложения vs KV/prefix cache; tenant/user, версия модели, данные и tools включать в политику invalidation. Semantic cache может вернуть неверный ответ похожему, но различающемуся по смыслу запросу.

**Практика:** нагрузка с ростом concurrency, backpressure, timeout/cancellation и cold start. Проверить идемпотентность повторных действий. Candidate rollout: фиксированный regression-набор → небольшой canary → сравнение quality/latency/errors → rollback при нарушении заданных критериев.

### 6. Evaluation приложения и границы ответственности

Качество одиночных fine-tuned ответов/judge и contamination — в [адаптации](../02-adaptation/%D0%9F%D0%BB%D0%B0%D0%BD%20%E2%80%94%20%D0%B0%D0%B4%D0%B0%D0%BF%D1%82%D0%B0%D1%86%D0%B8%D1%8F%20LLM.md); task success/стоимость траекторий — в [lifecycle](../05-lifecycle/%D0%9F%D0%BB%D0%B0%D0%BD%20%E2%80%94%20%D0%B6%D0%B8%D0%B7%D0%BD%D0%B5%D0%BD%D0%BD%D1%8B%D0%B9%20%D1%86%D0%B8%D0%BA%D0%BB%20LLM%20%D0%B8%20%D0%BE%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D0%B5%20%D0%B0%D0%B3%D0%B5%D0%BD%D1%82%D0%BE%D0%B2.md). Здесь дополнительно retrieval, полная latency, throughput, errors, cache correctness и качество по workload.

**Готово:** воспроизводимое приложение, фиксированные eval/нагрузочные задачи, измерение до/после одной оптимизации и объяснение компромисса качества, памяти, скорости и цены. Валидный JSON не считается доказательством правильного ответа или вызова.
