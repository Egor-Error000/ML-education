# Инференс LLM — подробные учебные конспекты

[Версия для чтения с набранными формулами](<../../../Читать конспекты.html#course-8-0>) · работает без интернета.

[Маршрут LLM/BERT и сценарии](<../../../План — LLM и языковые модели.md>) · [План раздела](<../План — инференс LLM.md>) · [Общее оглавление](../../../Учебные%20конспекты.md)

Блоки A–E объединены в десять связанных учебных глав. Они сохраняют содержательные вопросы исходных чатов; E.3–E.4 помечены как новый материал, продолжающий план. Буквы ниже — имена файлов, а рекомендуемый порядок чтения строит материал от обычной генерации к её оптимизации и эксплуатации.

## Рекомендуемый порядок

1. **Обычная генерация:** начните с вводного раздела в главе Е — logits задают распределение следующего токена, а decoding policy выбирает токен и условие остановки.
2. **KV-cache и цикл запроса:** В → Г — сначала prefill/decode и память, затем serving, batching и распределение KV-блоков.
3. **Оптимизация модели:** А → Б — quantization, затем pruning и отличие от distillation.
4. **Оптимизация генерации и контекста:** Д → Ж — speculative decoding, затем длинный контекст и prompt cache.
5. **Исполнение и production:** З → И → К — orchestration/tools/RAG, deployment/scaling, затем answer cache, мониторинг и оценка.

Главы сохраняют исходные буквы и происхождение материала; маршрут выше задаёт последовательность изучения.

- [Е. Greedy, temperature, top-k/top-p, beam и constrained output](<Е — Greedy, temperature, top-p, beam и constrained decoding.md>) — вводная генерация и C.2–C.3.
- [В. KV-cache, prefill/decode и MQA/GQA](<В — KV-cache, prefill, decode, MQA и GQA.md>) — A.6.
- [Г. Serving, continuous batching, PagedAttention и engines](<Г — Serving, continuous batching и PagedAttention.md>) — B.1–B.4.
- [А. Quantization, activations, PTQ/QAT, GPTQ/AWQ](<А — Квантизация, активации, PTQ, QAT, GPTQ и AWQ.md>) — A.1–A.3.
- [Б. Pruning и граница с distillation](<Б — Pruning и граница с дистилляцией.md>) — A.4–A.5.
- [Д. Speculative decoding, rejection/residual и beam](<Д — Speculative decoding, rejection sampling и beam search.md>) — C.1; case study DSpark.
- [Ж. Long context, sliding window и prompt caching](<Ж — Длинный контекст, sliding window и prompt caching.md>) — C.4.
- [З. State, orchestration, tools, RAG и frameworks](<З — Orchestration, tools, RAG и границы runtime.md>) — D.1–D.6.
- [И. Deployment, cold start, queue и autoscaling](<И — Deployment, cold start, очереди и масштабирование.md>) — E.1–E.2.
- [К. Answer/semantic cache и quality monitoring](<К — Кэш ответов, semantic cache, мониторинг и качество.md>) — E.3–E.4, дополнительные новые темы.

Основа — S4 [chat-План изучения темы(1).txt](<../../../archive/study-sources/chat-План изучения темы(1).txt>), продолжение S5 [LLM Study Plan Status](<../../../archive/study-sources/chat-LLM Study Plan Status.txt>) и S6 [Пункт D4](<../../../archive/study-sources/chat-Пункт D4.txt>). Диапазоны происхождения указаны в главах; [карта миграции](../../migration/MIGRATION_PLAN.md) сохраняет последовательность обсуждений.

## Покрытие практических протоколов

1. Измерение до оптимизации — Г/И: workload, TTFT, decode latency, p50/p95, queue, cold/warm, memory/errors.
2. Quantization и memory — А/В: affine example, calibration, output errors, weight/KV budget.
3. Sampling/context/speculation — Д/Е/Ж: ручные distributions, acceptance/residual, context position checks.
4. RAG — З и отдельный [план приложений](<../../04-applications/План — приложения LLM.md>): retrieval→evidence→answer, oracle context и разграничение ошибок; подробная механика поиска изучается там.
5. Graph/serving/эксплуатация — З/Г/И/К: state, retries, scheduling, cache correctness и rollout.
6. Evaluation — К плюс [главы адаптации](<../../02-adaptation/Учебные конспекты/З — Evaluation, perplexity, judge и экспорт.md>): качество, judge, regression и стоимость успешной задачи.

Distillation имеет одно основное подробное объяснение в [адаптации](<../../02-adaptation/Учебные конспекты/К — Дистилляция, teacher annealing и обучение на своих траекториях.md>). Она не размножается как второй курс в A.5.

Вопросы для самостоятельной проверки: посчитать KV512MiB example; получить target distribution из p/q residual; показать неизменность greedy при T>0; отличить prefix caching от answer cache; проследить tool error/timeout и termination. Теоретический текст и assistant snippets не подтверждают ваш запуск engine, training или production benchmark.
