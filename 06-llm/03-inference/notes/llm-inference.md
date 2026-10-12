# Инференс LLM: сжатие, кэш, serving и decoding

Здесь находятся общие inference-механизмы. Distillation вынесена в [заметку об адаптации](../../02-adaptation/notes/llm-adaptation.md), чтобы не вести два конкурирующих объяснения.

## 1. Квантование и pruning

Различайте четыре уровня: алгоритм (например GPTQ/AWQ), численное представление (INT4/NF4), container/file format (например GGUF) и runtime/kernel. PTQ квантует после обучения; QAT включает fake/quantization effects в обучение. STE — приближённый способ пропустить градиент через недифференцируемую операцию, а не точная производная округления.

GPTQ — семейство post-training weight quantization методов с калибровкой и компенсацией ошибок; AWQ использует activation-aware scaling для защиты чувствительных каналов. Не сводить AWQ к гарантированному хранению важных весов в более высоком dtype. Weight-only quantization, динамическое quantization activations и quantization KV cache — различные режимы с разной поддержкой.

Pruning удаляет веса, связи, головы/структурные блоки или целые слои. Ненулевые разреженные веса ускоряют модель только при поддержке соответствующих kernels и hardware; уменьшение параметров само по себе не гарантирует сокращения latency. Публичные описания не дают оснований уверенно реконструировать закрытый production-рецепт конкретной компании.

Форматы памяти считайте явно: 7B весов в 16-bit — около 14 GB весов, до дополнительной памяти. NF4 описывает формат квантуемого значения; compute dtype отдельно. GGUF — контейнерный формат экосистемы llama.cpp, CPU-only ограничение неверно. (C03, C05, C10, C13.)

## 2. KV cache и memory budget

В autoregressive decode cache хранит key/value предыдущих позиций, чтобы не вычислять их повторно. Каждый новый query всё равно взаимодействует с разрешёнными историческими позициями. Для обычного cache приблизительная память:

```text
bytes ≈ 2 × layers × batch × cached_tokens × kv_heads × head_dim × bytes_per_element
```

Здесь двойка — K и V. Учитывайте alignment, allocator/page metadata, scale factors при quantized cache и фактический dtype. MHA обычно имеет число KV heads, равное query heads; MQA разделяет одну KV head; GQA использует промежуточное число KV heads. Поэтому GQA/MQA могут снижать cache footprint и bandwidth.

Prefill обрабатывает prompt и часто compute-heavy; decode генерирует по токену и часто ограничен memory bandwidth или последовательной зависимостью. Batching увеличивает утилизацию, но меняет latency и размер cache.

## 3. Serving: batching, paging и runtime

Static batching ждёт/группирует запросы; continuous batching динамически добавляет и выпускает последовательности по мере завершения. Paged KV allocation разбивает cache на блоки, облегчая утилизацию памяти и переменную длину sequence; это механизм управления памятью, а не изменение математической функции модели.

Serving engines отличаются scheduler, kernels, batching, cache management, parallelism и совместимостью. PyTorch, ONNX Runtime, TensorRT-LLM, Triton и локальные runtime — не взаимозаменяемые обещания характеристик. API snippets и конфигурации version-sensitive; benchmark надо привязывать к модели, hardware, dtype, prompt, concurrency и decoding settings. Чатовые YAML/config примеры запуска не подтверждают.

Prompt/KV prefix caching сохраняет или повторно использует вычисления общего префикса; это не cache готовых ответов. Semantic/answer cache переиспользует ранее полученный ответ и требует проверки, что ключ покрывает смысловые ограничения, модель/версию, права доступа и актуальность данных. TTL, минимальная длина prefix и стоимость API зависят от поставщика и версии. Подробный checkpoint для E.3 — в [реестре](../../migration/PROGRESS.md); перед практикой сверять текущую документацию runtime.

## 4. Speculative decoding и beam search

Draft model предлагает несколько кандидатов, target/verifier проверяет их параллельно. Правильно реализованный speculative sampling сохраняет распределение target, но не обязан возвращать тот же текст для фиксированного seed, что и обычная генерация. При proposal `q` и target `p` acceptance probability для предложенного токена `x` равна `min(1,p(x)/q(x))`; при отказе выбирается из нормированного positive residual `(p−q)+`. Ускорение зависит от стоимости draft/verifier, сходства распределений, длины proposal и workload. Более сильный draft не автоматически быстрее.

Beam search ведёт набор гипотез с pruning по score и является другой процедурой выбора; свойства unbiased speculative sampling нельзя переносить на beam автоматически. (C12.)

Отдельный опубликованный case study DSpark доступен в [исследовательской заметке](llm-inference-research.md); его числа зависят от описанных там условий.

## Прогресс и происхождение

Вопросы пользователя, сохранённые по смыслу: что именно ускоряет KV cache; как отличать GPTQ/AWQ, формат quantization и runtime; почему temperature не меняет greedy argmax; гарантирует ли speculative decoding тот же текст и всегда ли более сильный draft быстрее.

- S4, `chat-План изучения темы(1).txt`: строки 4–2048 (quantization), 2052–2649 (PTQ/QAT/GPTQ/AWQ), 2653–3441 (pruning), 4771–4922 (KV cache/MHA/GQA), 4926–6435 (batching, paging, engines), 6439–7697 (speculative decoding/beam). [Архив](<../../../archive/study-sources/chat-%D0%9F%D0%BB%D0%B0%D0%BD%20%D0%B8%D0%B7%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D1%8F%20%D1%82%D0%B5%D0%BC%D1%8B%281%29.txt>).
- S3, `chat-Изучение Fine-tuning LLM.txt`, строки 674–715: уникальные вопросы об inference effects quantization; общая теория не дублируется. [Архив](../../../archive/study-sources/chat-%D0%98%D0%B7%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D0%B5%20Fine-tuning%20LLM.txt).

Обсуждение A1–A4, A6, B1–B4 и C1 подробно, практика deployment не подтверждена. Следующий шаг — вручную посчитать KV cache для заданных размеров и сравнить с peak memory; затем разобрать E.3/E.4 из плана inference.
