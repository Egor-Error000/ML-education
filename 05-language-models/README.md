# LLM и языковые модели

[Все модули](<../README.md>) · [Открыть в читателе](<../Читать конспекты.html#module-lm>) · [План и сценарии](<План.md>)

Общее ядро образует маршрут подраздела. Основа ветки нужна для выбранной задачи; углубление имеет отдельный вход и возврат к продолжению. Предпосылки каждой главы ведут к конкретным материалам.

## Основы

[Оглавление подраздела](<01-foundations/README.md>)

- [Языковая модель и её задачи](<01-foundations/lm-task.md>) — общее ядро. Что моделирует вероятность последовательности и какие задачи используют это распределение?
- [Токенизация: от текста к индексам](<01-foundations/lm-tokenizer.md>) — общее ядро. Как строка становится последовательностью ID и где заканчивается токенизатор?
- [Embeddings и контекстные состояния](<01-foundations/lm-states.md>) — общее ядро. Чем различаются ID, embedding и скрытое состояние?
- [Attention, Q/K/V и несколько голов](<01-foundations/lm-attention.md>) — общее ядро. Как одна позиция получает взвешенную информацию от других?
- [Позиции, causal mask и padding](<01-foundations/lm-position.md>) — общее ядро. Как независимо задать порядок токенов и доступ к информации?
- [Transformer-блок: residual, normalization и MLP](<01-foundations/lm-transformer.md>) — общее ядро. Как attention, MLP и остаточные связи образуют повторяемый блок?
- [Encoder, decoder-only и encoder–decoder](<01-foundations/lm-families.md>) — общее ядро. Как допустимые связи и источник Q/K/V различают семейства архитектур?
- [Выход модели: LM head, logits и вероятности](<01-foundations/lm-output.md>) — общее ядро. Как скрытое состояние превращается в распределение следующего токена?
- [Обучение: targets, loss и обновление весов](<01-foundations/lm-training.md>) — общее ядро. Как построить targets и проследить путь ошибки до обучаемых параметров?
- [Обычная генерация по одному токену](<01-foundations/lm-generation.md>) — общее ядро. Как распределение превращается в продолжение и почему обучение отличается от применения?
- [BERT в общем ядре: encoder, MLM и задачи](<01-foundations/lm-bert.md>) — общее ядро. Как двунаправленный encoder обучается на скрытых позициях и используется в задачах?
- [Углубление: RNN и рекуррентное состояние](<01-foundations/lm-rnn.md>) — углубление. Как одна и та же ячейка связывает состояния и градиенты во времени?
- [Углубление: точный backward attention](<01-foundations/lm-backward.md>) — углубление. Как производные проходят через softmax, Q/K/V и общие параметры?
- [Углубление: интерпретация представлений](<01-foundations/lm-interpretation.md>) — углубление. Какие проверки нужны для утверждения о смысле слоя или embedding?

## BERT и encoder-модели

[Оглавление подраздела](<02-bert-encoders/README.md>)

- [Encoder, tokenizer, attention и pooling](<02-bert-encoders/bert-interface.md>) — основа ветки. Как получить состояния и корректно агрегировать их в представление текста?
- [MLM, альтернативные objectives и семейства encoder](<02-bert-encoders/bert-objectives.md>) — основа ветки. Какие позиции дают loss и как цель обучения меняет представления?
- [Данные, task heads, fine-tuning и DAPT](<02-bert-encoders/bert-adaptation.md>) — основа ветки. Как согласовать разметку, форму выхода и objective encoder?
- [Classification, NER, spans и extractive QA](<02-bert-encoders/bert-tasks.md>) — основа ветки. Как преобразовать текстовую задачу в targets и восстановить ответ в исходных координатах?
- [Sentence embeddings и contrastive learning](<02-bert-encoders/bert-embeddings.md>) — основа ветки. Как обучить пространство, где близость полезна для конкретной задачи?
- [Dense retrieval, cross-encoder, SPLADE и ColBERT](<02-bert-encoders/bert-retrieval.md>) — основа ветки. Как выбрать место взаимодействия query и document и оценить полный budget?
- [Языки, длинные документы, домены и layout](<02-bert-encoders/bert-domains.md>) — основа ветки. Какие части контракта меняются при переносе на новый язык или тип документа?
- [Метрики, калибровка, OOD и интерпретация](<02-bert-encoders/bert-evaluation.md>) — основа ветки. Как оценить encoder на правильной единице задачи и проверить отказ?
- [Сжатие, сервис encoder и жизненный цикл индекса](<02-bert-encoders/bert-operations.md>) — углубление. Как обновить encoder и индекс, сохраняя согласованность представлений?

## Адаптация

[Оглавление подраздела](<03-adaptation/README.md>)

- [Задача адаптации, baseline и выбор вмешательства](<03-adaptation/adapt-choice.md>) — основа ветки. Как отличить нехватку знаний, ошибку поведения и проблему исполнения?
- [SFT: данные, шаблоны, masks и objective](<03-adaptation/adapt-sft.md>) — основа ветки. Какие позиции становятся целями supervised fine-tuning?
- [Полное обучение или PEFT](<03-adaptation/adapt-regimes.md>) — основа ветки. Какие параметры обновляются и почему это не меняет смысл SFT?
- [LoRA: low-rank обновление и градиенты](<03-adaptation/adapt-lora.md>) — основа ветки. Как low-rank ветвь обновляет конкретную линейную проекцию?
- [DoRA, AdaLoRA, LoRA+ и QLoRA](<03-adaptation/adapt-variants.md>) — углубление. Как различаются изменяемый объект, бюджет ранга и формат хранения базы?
- [Adapters, prefix tuning, soft prompts и BitFit](<03-adaptation/adapt-alternatives.md>) — углубление. Где добавляются обучаемые параметры и что меняется при inference?
- [RLHF, PPO, DPO, IPO, KTO и ORPO](<03-adaptation/adapt-preference.md>) — основа ветки. Как предпочтения превращаются в objective и каких данных он требует?
- [Model soup, TIES, DARE и переключение адаптеров](<03-adaptation/adapt-merging.md>) — углубление. Когда операции над весами совместимы и как проверить итоговую функцию?
- [Дистилляция и обучение на траекториях](<03-adaptation/adapt-distillation.md>) — углубление. Какой сигнал teacher передаёт student и как проверить перенос?
- [AI feedback и continual learning](<03-adaptation/adapt-continual.md>) — углубление. Как обновлять поведение, контролируя источник feedback и забывание?
- [Оценка, perplexity, judge и экспорт](<03-adaptation/adapt-evaluation.md>) — основа ветки. Как сравнить candidate с baseline и подтвердить сохранение поведения после экспорта?

## Инференс

[Оглавление подраздела](<04-inference/README.md>)

- [От prompt к обычному циклу генерации](<04-inference/infer-generation.md>) — общее ядро. Какие состояния меняются при генерации и какие веса остаются фиксированными?
- [Prefill, decode, KV-cache, MQA и GQA](<04-inference/infer-kv.md>) — общее ядро. Какие вычисления и тензоры можно повторно использовать?
- [Контекст, sliding window и prefix caching](<04-inference/infer-context.md>) — общее ядро. Что меняется при ограничении истории и повторном использовании prefix?
- [Greedy, temperature, top-p, beam и ограничения](<04-inference/infer-decoding.md>) — общее ядро. Как decoding policy выбирает токен из logits?
- [Serving, batching и PagedAttention](<04-inference/infer-serving.md>) — основа ветки. Как совместно обслуживать запросы при ограничениях памяти и задержки?
- [Speculative decoding и rejection sampling](<04-inference/infer-speculative.md>) — углубление. Почему проверка draft сохраняет заданное распределение?
- [Квантизация: PTQ, QAT, GPTQ и AWQ](<04-inference/infer-quantization.md>) — углубление. Какие массивы и ошибки меняются при сокращении precision?
- [Pruning и границы сжатия](<04-inference/infer-pruning.md>) — углубление. Как удаление структуры связано с реальным ускорением?
- [Deployment, cold start, очереди и масштабирование](<04-inference/infer-deployment.md>) — основа ветки. Как согласовать admission, startup и воспроизводимый rollout?
- [Кэш ответов, monitoring и качество](<04-inference/infer-monitoring.md>) — основа ветки. Как доказать допустимость reuse и заметить регрессию качества?

## Приложения

[Оглавление подраздела](<05-applications/README.md>)

- [Structured output: schema и смысл результата](<05-applications/app-structured.md>) — основа ветки. Что гарантирует синтаксис результата и что ещё должно проверить приложение?
- [Вызов инструментов и исполнение](<05-applications/app-tools.md>) — основа ветки. Где заканчивается выбор модели и начинается действие приложения?
- [Состояние, orchestration и граф управления](<05-applications/app-state.md>) — основа ветки. Какие данные и переходы нужны для остановки, восстановления и повторного исполнения?
- [RAG: BM25, dense, hybrid и multi-vector retrieval](<05-applications/app-rag.md>) — основа ветки. Как найти evidence и связать его с ответом без обучения собственного retriever?
- [Оценка retrieval, ответов и task success](<05-applications/app-evaluation.md>) — основа ветки. Как локализовать ошибку приложения и измерить стоимость успешной задачи?

## Обучение агентов

[Оглавление подраздела](<06-agent-learning/README.md>)

- [Постановка обучения агента и baseline](<06-agent-learning/agent-task.md>) — основа ветки. Как выбрать вмешательство по измеренному типу ошибки?
- [Среда, observations, actions и horizon](<06-agent-learning/agent-environment.md>) — основа ветки. Как описать доступную агенту информацию и последствия действия?
- [Оценка траекторий и воспроизводимость](<06-agent-learning/agent-evaluation.md>) — основа ветки. Что считать успехом целой попытки и как учитывать retries и стоимость?
- [Behavior cloning и SFT на демонстрациях](<06-agent-learning/agent-demonstrations.md>) — основа ветки. Как из траектории получить targets для обучения политики?
- [Distribution shift, recovery и DAgger](<06-agent-learning/agent-recovery.md>) — основа ветки. Как научиться действовать в состояниях, порождённых собственными ошибками?
- [Обучение через взаимодействие: PPO и GRPO](<06-agent-learning/agent-rl.md>) — основа ветки. Как отложенный reward меняет вероятность последовательности действий?
- [Углубление: reasoning и верификация](<06-agent-learning/agent-reasoning.md>) — углубление. Когда дополнительный поиск и проверка улучшают результат?
- [Углубление: память и длинный горизонт](<06-agent-learning/agent-memory.md>) — углубление. Какую информацию сохранить и как измерить её полезность?
- [Углубление: предпочтения траекторий](<06-agent-learning/agent-preferences.md>) — углубление. Как сравнивать полные попытки с разным бюджетом и исходом?
- [Углубление: роли и обучение координации](<06-agent-learning/agent-coordination.md>) — углубление. При каких условиях несколько участников дают измеримый выигрыш?

## Маршруты

**Общее ядро языковых моделей:** [Языковая модель и её задачи](<01-foundations/lm-task.md>) → [Токенизация: от текста к индексам](<01-foundations/lm-tokenizer.md>) → [Embeddings и контекстные состояния](<01-foundations/lm-states.md>) → [Attention, Q/K/V и несколько голов](<01-foundations/lm-attention.md>) → [Позиции, causal mask и padding](<01-foundations/lm-position.md>) → [Transformer-блок: residual, normalization и MLP](<01-foundations/lm-transformer.md>) → [Encoder, decoder-only и encoder–decoder](<01-foundations/lm-families.md>) → [Выход модели: LM head, logits и вероятности](<01-foundations/lm-output.md>) → [Обучение: targets, loss и обновление весов](<01-foundations/lm-training.md>) → [Обычная генерация по одному токену](<01-foundations/lm-generation.md>) → [BERT в общем ядре: encoder, MLM и задачи](<01-foundations/lm-bert.md>)

**BERT без генеративной адаптации:** [Постановка задачи и паспорт данных](<../01-data-features/01-reliability/df-task.md>) → [Split, cross-validation и утечки](<../01-data-features/01-reliability/df-split.md>) → [Dataset, batching и воспроизводимость](<../01-data-features/01-reliability/df-loading.md>) → [Текст, токенизация и разбиение документов](<../01-data-features/03-modalities/df-text.md>) → [Языковая модель и её задачи](<01-foundations/lm-task.md>) → [Токенизация: от текста к индексам](<01-foundations/lm-tokenizer.md>) → [Embeddings и контекстные состояния](<01-foundations/lm-states.md>) → [Attention, Q/K/V и несколько голов](<01-foundations/lm-attention.md>) → [Позиции, causal mask и padding](<01-foundations/lm-position.md>) → [Transformer-блок: residual, normalization и MLP](<01-foundations/lm-transformer.md>) → [Encoder, decoder-only и encoder–decoder](<01-foundations/lm-families.md>) → [Выход модели: LM head, logits и вероятности](<01-foundations/lm-output.md>) → [Обучение: targets, loss и обновление весов](<01-foundations/lm-training.md>) → [BERT в общем ядре: encoder, MLM и задачи](<01-foundations/lm-bert.md>) → [Encoder, tokenizer, attention и pooling](<02-bert-encoders/bert-interface.md>) → [MLM, альтернативные objectives и семейства encoder](<02-bert-encoders/bert-objectives.md>) → [Данные, task heads, fine-tuning и DAPT](<02-bert-encoders/bert-adaptation.md>) → [Classification, NER, spans и extractive QA](<02-bert-encoders/bert-tasks.md>) → [Метрики, калибровка, OOD и интерпретация](<02-bert-encoders/bert-evaluation.md>)

**Приложение без LoRA и обучения retrieval:** [Постановка задачи и паспорт данных](<../01-data-features/01-reliability/df-task.md>) → [Split, cross-validation и утечки](<../01-data-features/01-reliability/df-split.md>) → [Dataset, batching и воспроизводимость](<../01-data-features/01-reliability/df-loading.md>) → [Текст, токенизация и разбиение документов](<../01-data-features/03-modalities/df-text.md>) → [Языковая модель и её задачи](<01-foundations/lm-task.md>) → [Токенизация: от текста к индексам](<01-foundations/lm-tokenizer.md>) → [Embeddings и контекстные состояния](<01-foundations/lm-states.md>) → [Attention, Q/K/V и несколько голов](<01-foundations/lm-attention.md>) → [Позиции, causal mask и padding](<01-foundations/lm-position.md>) → [Transformer-блок: residual, normalization и MLP](<01-foundations/lm-transformer.md>) → [Encoder, decoder-only и encoder–decoder](<01-foundations/lm-families.md>) → [Выход модели: LM head, logits и вероятности](<01-foundations/lm-output.md>) → [Обучение: targets, loss и обновление весов](<01-foundations/lm-training.md>) → [Обычная генерация по одному токену](<01-foundations/lm-generation.md>) → [Structured output: schema и смысл результата](<05-applications/app-structured.md>) → [Вызов инструментов и исполнение](<05-applications/app-tools.md>) → [Состояние, orchestration и граф управления](<05-applications/app-state.md>) → [RAG: BM25, dense, hybrid и multi-vector retrieval](<05-applications/app-rag.md>) → [Оценка retrieval, ответов и task success](<05-applications/app-evaluation.md>)

**SFT перед LoRA:** [Постановка задачи и паспорт данных](<../01-data-features/01-reliability/df-task.md>) → [Split, cross-validation и утечки](<../01-data-features/01-reliability/df-split.md>) → [Dataset, batching и воспроизводимость](<../01-data-features/01-reliability/df-loading.md>) → [Текст, токенизация и разбиение документов](<../01-data-features/03-modalities/df-text.md>) → [Языковая модель и её задачи](<01-foundations/lm-task.md>) → [Токенизация: от текста к индексам](<01-foundations/lm-tokenizer.md>) → [Embeddings и контекстные состояния](<01-foundations/lm-states.md>) → [Attention, Q/K/V и несколько голов](<01-foundations/lm-attention.md>) → [Позиции, causal mask и padding](<01-foundations/lm-position.md>) → [Transformer-блок: residual, normalization и MLP](<01-foundations/lm-transformer.md>) → [Encoder, decoder-only и encoder–decoder](<01-foundations/lm-families.md>) → [Выход модели: LM head, logits и вероятности](<01-foundations/lm-output.md>) → [Обучение: targets, loss и обновление весов](<01-foundations/lm-training.md>) → [Обычная генерация по одному токену](<01-foundations/lm-generation.md>) → [Задача адаптации, baseline и выбор вмешательства](<03-adaptation/adapt-choice.md>) → [SFT: данные, шаблоны, masks и objective](<03-adaptation/adapt-sft.md>) → [Полное обучение или PEFT](<03-adaptation/adapt-regimes.md>) → [LoRA: low-rank обновление и градиенты](<03-adaptation/adapt-lora.md>) → [Оценка, perplexity, judge и экспорт](<03-adaptation/adapt-evaluation.md>)

**От генерации к serving:** [Языковая модель и её задачи](<01-foundations/lm-task.md>) → [Токенизация: от текста к индексам](<01-foundations/lm-tokenizer.md>) → [Embeddings и контекстные состояния](<01-foundations/lm-states.md>) → [Attention, Q/K/V и несколько голов](<01-foundations/lm-attention.md>) → [Позиции, causal mask и padding](<01-foundations/lm-position.md>) → [Transformer-блок: residual, normalization и MLP](<01-foundations/lm-transformer.md>) → [Encoder, decoder-only и encoder–decoder](<01-foundations/lm-families.md>) → [Выход модели: LM head, logits и вероятности](<01-foundations/lm-output.md>) → [Обучение: targets, loss и обновление весов](<01-foundations/lm-training.md>) → [Обычная генерация по одному токену](<01-foundations/lm-generation.md>) → [От prompt к обычному циклу генерации](<04-inference/infer-generation.md>) → [Prefill, decode, KV-cache, MQA и GQA](<04-inference/infer-kv.md>) → [Контекст, sliding window и prefix caching](<04-inference/infer-context.md>) → [Greedy, temperature, top-p, beam и ограничения](<04-inference/infer-decoding.md>) → [Serving, batching и PagedAttention](<04-inference/infer-serving.md>) → [Deployment, cold start, очереди и масштабирование](<04-inference/infer-deployment.md>) → [Кэш ответов, monitoring и качество](<04-inference/infer-monitoring.md>)

**Оценка агента перед обучением:** [Постановка обучения агента и baseline](<06-agent-learning/agent-task.md>) → [Среда, observations, actions и horizon](<06-agent-learning/agent-environment.md>) → [Оценка траекторий и воспроизводимость](<06-agent-learning/agent-evaluation.md>) → [Behavior cloning и SFT на демонстрациях](<06-agent-learning/agent-demonstrations.md>) → [Distribution shift, recovery и DAgger](<06-agent-learning/agent-recovery.md>) → [Обучение через взаимодействие: PPO и GRPO](<06-agent-learning/agent-rl.md>)

Вход в маршрут: [Состояние, orchestration и граф управления](<05-applications/app-state.md>), [Оценка retrieval, ответов и task success](<05-applications/app-evaluation.md>), [SFT: данные, шаблоны, masks и objective](<03-adaptation/adapt-sft.md>).
