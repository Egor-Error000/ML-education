# Инференс

[Модуль «LLM и языковые модели»](<../README.md>) · [Сценарии глав](<План.md>)

- [От prompt к обычному циклу генерации](<infer-generation.md>) — общее ядро. Какие состояния меняются при генерации и какие веса остаются фиксированными?
- [Prefill, decode, KV-cache, MQA и GQA](<infer-kv.md>) — общее ядро. Какие вычисления и тензоры можно повторно использовать?
- [Контекст, sliding window и prefix caching](<infer-context.md>) — общее ядро. Что меняется при ограничении истории и повторном использовании prefix?
- [Greedy, temperature, top-p, beam и ограничения](<infer-decoding.md>) — общее ядро. Как decoding policy выбирает токен из logits?
- [Serving, batching и PagedAttention](<infer-serving.md>) — основа ветки. Как совместно обслуживать запросы при ограничениях памяти и задержки?
- [Speculative decoding и rejection sampling](<infer-speculative.md>) — углубление. Почему проверка draft сохраняет заданное распределение?
- [Квантизация: PTQ, QAT, GPTQ и AWQ](<infer-quantization.md>) — углубление. Какие массивы и ошибки меняются при сокращении precision?
- [Pruning и границы сжатия](<infer-pruning.md>) — углубление. Как удаление структуры связано с реальным ускорением?
- [Deployment, cold start, очереди и масштабирование](<infer-deployment.md>) — основа ветки. Как согласовать admission, startup и воспроизводимый rollout?
- [Кэш ответов, monitoring и качество](<infer-monitoring.md>) — основа ветки. Как доказать допустимость reuse и заметить регрессию качества?

**От генерации к serving:** [Языковая модель и её задачи](<../01-foundations/lm-task.md>) → [Токенизация: от текста к индексам](<../01-foundations/lm-tokenizer.md>) → [Embeddings и контекстные состояния](<../01-foundations/lm-states.md>) → [Attention, Q/K/V и несколько голов](<../01-foundations/lm-attention.md>) → [Позиции, causal mask и padding](<../01-foundations/lm-position.md>) → [Transformer-блок: residual, normalization и MLP](<../01-foundations/lm-transformer.md>) → [Encoder, decoder-only и encoder–decoder](<../01-foundations/lm-families.md>) → [Выход модели: LM head, logits и вероятности](<../01-foundations/lm-output.md>) → [Обучение: targets, loss и обновление весов](<../01-foundations/lm-training.md>) → [Обычная генерация по одному токену](<../01-foundations/lm-generation.md>) → [От prompt к обычному циклу генерации](<infer-generation.md>) → [Prefill, decode, KV-cache, MQA и GQA](<infer-kv.md>) → [Контекст, sliding window и prefix caching](<infer-context.md>) → [Greedy, temperature, top-p, beam и ограничения](<infer-decoding.md>) → [Serving, batching и PagedAttention](<infer-serving.md>) → [Deployment, cold start, очереди и масштабирование](<infer-deployment.md>) → [Кэш ответов, monitoring и качество](<infer-monitoring.md>)
