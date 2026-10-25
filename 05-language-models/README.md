# LLM и языковые модели

[Все модули](<../README.md>) · [Открыть в читателе](<../Читать конспекты.html#module-lm>) · [План и сценарии](<План.md>)

Общее ядро образует маршрут подраздела. Основа ветки нужна для выбранной задачи; углубление имеет отдельный вход и возврат к продолжению. Предпосылки каждой главы ведут к конкретным материалам.

## Основы

[Оглавление подраздела](<01-foundations/README.md>)

- [Языковая модель и её задачи](<01-foundations/lm-task.md>) — общее ядро. Что моделирует вероятность последовательности и какие задачи используют это распределение?
- [Токенизация: от текста к индексам](<01-foundations/lm-tokenizer.md>) — общее ядро. Как превратить текст в IDs, сохранив связь с исходными символами?
- [Embeddings и контекстные состояния](<01-foundations/lm-states.md>) — общее ядро. Чем отличаются ID, строка таблицы embeddings, контекстное состояние и оценка слова?
- [Attention, Q/K/V и несколько голов](<01-foundations/lm-attention.md>) — общее ядро. Как одна позиция получает взвешенную информацию от других?
- [Позиции, causal mask и padding](<01-foundations/lm-position.md>) — общее ядро. Как модель различает расположение токенов и ограничивает доступ к контексту?
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
- [Цели предобучения BERT и варианты кодировщиков](<02-bert-encoders/bert-objectives.md>) — основа ветки. Что модель видит, что должна предсказать и какие параметры меняет ошибка?
- [Данные, головы задач и адаптация BERT](<02-bert-encoders/bert-adaptation.md>) — основа ветки. Какие данные, выход и параметры следует выбрать для адаптации BERT?
- [Классификация, сущности и извлечение ответов](<02-bert-encoders/bert-tasks.md>) — основа ветки. Как разметка превращается в голову классификации, сущностей или извлечения ответа?
- [Векторы предложений и контрастивное обучение](<02-bert-encoders/bert-embeddings.md>) — основа ветки. Как обучить и проверить векторы предложений для сравнения текстов?
- [Поиск: плотные векторы, BM25, SPLADE и ColBERT](<02-bert-encoders/bert-retrieval.md>) — основа ветки. Как устроены кандидаты, переранжирование и разные представления поискового документа?
- [Языки, длинные документы и пространственная структура](<02-bert-encoders/bert-domains.md>) — основа ветки. Что меняется при переносе encoder между языками, доменами и видами документа?
- [Метрики, калибровка и границы интерпретации](<02-bert-encoders/bert-evaluation.md>) — основа ветки. Какие выводы позволяют метрики, калибровка и объяснения encoder?
- [Сжатие, служба кодировщика и обновление индекса](<02-bert-encoders/bert-operations.md>) — углубление. Как уменьшить стоимость encoder и безопасно для качества обновить служебный комплект?

## Адаптация

[Оглавление подраздела](<03-adaptation/README.md>)

- [Адаптация: задача, базовая система и бюджет](<03-adaptation/adapt-choice.md>) — основа ветки. Как выбрать изменение, которое исправляет наблюдаемую ошибку системы?
- [SFT: демонстрации, шаблоны и обучающие цели](<03-adaptation/adapt-sft.md>) — основа ветки. Какие токены учат модель нужному поведению и с каким весом?
- [Полное обучение, замороженная основа и PEFT](<03-adaptation/adapt-regimes.md>) — основа ветки. Какая часть модели должна получать обновления при выбранной обучающей цели?
- [LoRA: низкоранговое обновление и его градиенты](<03-adaptation/adapt-lora.md>) — основа ветки. Куда добавляются факторы A/B и как они начинают обучаться?
- [DoRA, AdaLoRA, LoRA+ и QLoRA: что именно меняется](<03-adaptation/adapt-variants.md>) — углубление. Как различить норму весов, бюджет компонентов, скорость факторов и хранение базы?
- [Адаптерные блоки, мягкие запросы, префиксы и BitFit](<03-adaptation/adapt-alternatives.md>) — углубление. Где находятся обучаемые величины и что добавляется к вычислению при применении?
- [Предпочтения: награда, PPO, DPO и альтернативные цели](<03-adaptation/adapt-preference.md>) — основа ветки. Как сравнение ответов превращается в обучающий сигнал и что оно не устанавливает?
- [Слияние моделей и переключение адаптеров](<03-adaptation/adapt-merging.md>) — углубление. Какой объект объединяется и при каких условиях сохраняется его смысл?
- [Дистилляция: распределения, ответы и состояния ученика](<03-adaptation/adapt-distillation.md>) — углубление. Какой сигнал передаёт учитель и по какой цене ученик его воспроизводит?
- [Обратная связь моделей и последовательное обучение](<03-adaptation/adapt-continual.md>) — углубление. Как обновлять модель, сохраняя источник оценки и проверку прежних задач?
- [Оценка адаптации: задача, неопределённость и экспорт](<03-adaptation/adapt-evaluation.md>) — основа ветки. Что именно улучшилось и сохраняет ли это загруженная итоговая модель?

## Инференс

[Оглавление подраздела](<04-inference/README.md>)

- [Инференс: от входа к обычному циклу генерации](<04-inference/infer-generation.md>) — общее ядро. Что пересчитывается на каждом шаге и что нужно измерить до оптимизаций?
- [Prefill, decode и KV-кеш: вычисление и память](<04-inference/infer-kv.md>) — общее ядро. Почему можно сохранить прошлые ключи и значения и чего это не устраняет?
- [Контекст: длина, локальное окно и повторение префикса](<04-inference/infer-context.md>) — общее ядро. Какая информация остаётся доступной и какие вычисления действительно повторяются?
- [Выбор токенов: температура, отсечение, поиск и ограничения](<04-inference/infer-decoding.md>) — общее ядро. Как правило выбора меняет распределение и какие гарантии даёт грамматика?
- [Обслуживание запросов: планировщик, пакеты и страницы KV](<04-inference/infer-serving.md>) — основа ветки. Как совместить запросы, сохранив их состояние, задержку и границы памяти?
- [Спекулятивная генерация: черновик, коррекция и состояние](<04-inference/infer-speculative.md>) — углубление. Как принять дешёвое предложение, сохранив распределение целевой модели?
- [Квантизация: представление чисел, калибровка и ошибка задачи](<04-inference/infer-quantization.md>) — углубление. Какие массивы становятся малобитными и как их ошибка влияет на вычисление?
- [Удаление весов и структур: границы сжатия модели](<04-inference/infer-pruning.md>) — углубление. Как отличить нулевой вес, меньший граф и реальное ускорение?
- [Развёртывание: готовность, очередь и масштабирование](<04-inference/infer-deployment.md>) — основа ветки. Как выпустить согласованный комплект и принимать только исполнимую работу?
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

**BERT без генеративной адаптации:** [Постановка задачи и паспорт данных](<../01-data-features/01-reliability/df-task.md>) → [Split, cross-validation и утечки](<../01-data-features/01-reliability/df-split.md>) → [Dataset, batching и воспроизводимость](<../01-data-features/01-reliability/df-loading.md>) → [Текст, токенизация и разбиение документов](<../01-data-features/03-modalities/df-text.md>) → [Языковая модель и её задачи](<01-foundations/lm-task.md>) → [Токенизация: от текста к индексам](<01-foundations/lm-tokenizer.md>) → [Embeddings и контекстные состояния](<01-foundations/lm-states.md>) → [Attention, Q/K/V и несколько голов](<01-foundations/lm-attention.md>) → [Позиции, causal mask и padding](<01-foundations/lm-position.md>) → [Transformer-блок: residual, normalization и MLP](<01-foundations/lm-transformer.md>) → [Encoder, decoder-only и encoder–decoder](<01-foundations/lm-families.md>) → [Выход модели: LM head, logits и вероятности](<01-foundations/lm-output.md>) → [Обучение: targets, loss и обновление весов](<01-foundations/lm-training.md>) → [BERT в общем ядре: encoder, MLM и задачи](<01-foundations/lm-bert.md>) → [Encoder, tokenizer, attention и pooling](<02-bert-encoders/bert-interface.md>) → [Цели предобучения BERT и варианты кодировщиков](<02-bert-encoders/bert-objectives.md>) → [Данные, головы задач и адаптация BERT](<02-bert-encoders/bert-adaptation.md>) → [Классификация, сущности и извлечение ответов](<02-bert-encoders/bert-tasks.md>) → [Метрики, калибровка и границы интерпретации](<02-bert-encoders/bert-evaluation.md>)

**Приложение без LoRA и обучения retrieval:** [Постановка задачи и паспорт данных](<../01-data-features/01-reliability/df-task.md>) → [Split, cross-validation и утечки](<../01-data-features/01-reliability/df-split.md>) → [Dataset, batching и воспроизводимость](<../01-data-features/01-reliability/df-loading.md>) → [Текст, токенизация и разбиение документов](<../01-data-features/03-modalities/df-text.md>) → [Языковая модель и её задачи](<01-foundations/lm-task.md>) → [Токенизация: от текста к индексам](<01-foundations/lm-tokenizer.md>) → [Embeddings и контекстные состояния](<01-foundations/lm-states.md>) → [Attention, Q/K/V и несколько голов](<01-foundations/lm-attention.md>) → [Позиции, causal mask и padding](<01-foundations/lm-position.md>) → [Transformer-блок: residual, normalization и MLP](<01-foundations/lm-transformer.md>) → [Encoder, decoder-only и encoder–decoder](<01-foundations/lm-families.md>) → [Выход модели: LM head, logits и вероятности](<01-foundations/lm-output.md>) → [Обучение: targets, loss и обновление весов](<01-foundations/lm-training.md>) → [Обычная генерация по одному токену](<01-foundations/lm-generation.md>) → [Structured output: schema и смысл результата](<05-applications/app-structured.md>) → [Вызов инструментов и исполнение](<05-applications/app-tools.md>) → [Состояние, orchestration и граф управления](<05-applications/app-state.md>) → [RAG: BM25, dense, hybrid и multi-vector retrieval](<05-applications/app-rag.md>) → [Оценка retrieval, ответов и task success](<05-applications/app-evaluation.md>)

**SFT перед LoRA:** [Постановка задачи и паспорт данных](<../01-data-features/01-reliability/df-task.md>) → [Split, cross-validation и утечки](<../01-data-features/01-reliability/df-split.md>) → [Dataset, batching и воспроизводимость](<../01-data-features/01-reliability/df-loading.md>) → [Текст, токенизация и разбиение документов](<../01-data-features/03-modalities/df-text.md>) → [Языковая модель и её задачи](<01-foundations/lm-task.md>) → [Токенизация: от текста к индексам](<01-foundations/lm-tokenizer.md>) → [Embeddings и контекстные состояния](<01-foundations/lm-states.md>) → [Attention, Q/K/V и несколько голов](<01-foundations/lm-attention.md>) → [Позиции, causal mask и padding](<01-foundations/lm-position.md>) → [Transformer-блок: residual, normalization и MLP](<01-foundations/lm-transformer.md>) → [Encoder, decoder-only и encoder–decoder](<01-foundations/lm-families.md>) → [Выход модели: LM head, logits и вероятности](<01-foundations/lm-output.md>) → [Обучение: targets, loss и обновление весов](<01-foundations/lm-training.md>) → [Обычная генерация по одному токену](<01-foundations/lm-generation.md>) → [Адаптация: задача, базовая система и бюджет](<03-adaptation/adapt-choice.md>) → [SFT: демонстрации, шаблоны и обучающие цели](<03-adaptation/adapt-sft.md>) → [Полное обучение, замороженная основа и PEFT](<03-adaptation/adapt-regimes.md>) → [LoRA: низкоранговое обновление и его градиенты](<03-adaptation/adapt-lora.md>) → [Оценка адаптации: задача, неопределённость и экспорт](<03-adaptation/adapt-evaluation.md>)

**От генерации к serving:** [Языковая модель и её задачи](<01-foundations/lm-task.md>) → [Токенизация: от текста к индексам](<01-foundations/lm-tokenizer.md>) → [Embeddings и контекстные состояния](<01-foundations/lm-states.md>) → [Attention, Q/K/V и несколько голов](<01-foundations/lm-attention.md>) → [Позиции, causal mask и padding](<01-foundations/lm-position.md>) → [Transformer-блок: residual, normalization и MLP](<01-foundations/lm-transformer.md>) → [Encoder, decoder-only и encoder–decoder](<01-foundations/lm-families.md>) → [Выход модели: LM head, logits и вероятности](<01-foundations/lm-output.md>) → [Обучение: targets, loss и обновление весов](<01-foundations/lm-training.md>) → [Обычная генерация по одному токену](<01-foundations/lm-generation.md>) → [Инференс: от входа к обычному циклу генерации](<04-inference/infer-generation.md>) → [Prefill, decode и KV-кеш: вычисление и память](<04-inference/infer-kv.md>) → [Контекст: длина, локальное окно и повторение префикса](<04-inference/infer-context.md>) → [Выбор токенов: температура, отсечение, поиск и ограничения](<04-inference/infer-decoding.md>) → [Обслуживание запросов: планировщик, пакеты и страницы KV](<04-inference/infer-serving.md>) → [Развёртывание: готовность, очередь и масштабирование](<04-inference/infer-deployment.md>) → [Кэш ответов, monitoring и качество](<04-inference/infer-monitoring.md>)

**Оценка агента перед обучением:** [Постановка обучения агента и baseline](<06-agent-learning/agent-task.md>) → [Среда, observations, actions и horizon](<06-agent-learning/agent-environment.md>) → [Оценка траекторий и воспроизводимость](<06-agent-learning/agent-evaluation.md>) → [Behavior cloning и SFT на демонстрациях](<06-agent-learning/agent-demonstrations.md>) → [Distribution shift, recovery и DAgger](<06-agent-learning/agent-recovery.md>) → [Обучение через взаимодействие: PPO и GRPO](<06-agent-learning/agent-rl.md>)

Вход в маршрут: [Состояние, orchestration и граф управления](<05-applications/app-state.md>), [Оценка retrieval, ответов и task success](<05-applications/app-evaluation.md>), [SFT: демонстрации, шаблоны и обучающие цели](<03-adaptation/adapt-sft.md>).
