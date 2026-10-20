# BERT и encoder-модели

[Модуль «LLM и языковые модели»](<../README.md>) · [Сценарии глав](<План.md>)

- [Encoder, tokenizer, attention и pooling](<bert-interface.md>) — основа ветки. Как получить состояния и корректно агрегировать их в представление текста?
- [MLM, альтернативные objectives и семейства encoder](<bert-objectives.md>) — основа ветки. Какие позиции дают loss и как цель обучения меняет представления?
- [Данные, task heads, fine-tuning и DAPT](<bert-adaptation.md>) — основа ветки. Как согласовать разметку, форму выхода и objective encoder?
- [Classification, NER, spans и extractive QA](<bert-tasks.md>) — основа ветки. Как преобразовать текстовую задачу в targets и восстановить ответ в исходных координатах?
- [Sentence embeddings и contrastive learning](<bert-embeddings.md>) — основа ветки. Как обучить пространство, где близость полезна для конкретной задачи?
- [Dense retrieval, cross-encoder, SPLADE и ColBERT](<bert-retrieval.md>) — основа ветки. Как выбрать место взаимодействия query и document и оценить полный budget?
- [Языки, длинные документы, домены и layout](<bert-domains.md>) — основа ветки. Какие части контракта меняются при переносе на новый язык или тип документа?
- [Метрики, калибровка, OOD и интерпретация](<bert-evaluation.md>) — основа ветки. Как оценить encoder на правильной единице задачи и проверить отказ?
- [Сжатие, сервис encoder и жизненный цикл индекса](<bert-operations.md>) — углубление. Как обновить encoder и индекс, сохраняя согласованность представлений?

**BERT без генеративной адаптации:** [Постановка задачи и паспорт данных](<../../01-data-features/01-reliability/df-task.md>) → [Split, cross-validation и утечки](<../../01-data-features/01-reliability/df-split.md>) → [Dataset, batching и воспроизводимость](<../../01-data-features/01-reliability/df-loading.md>) → [Текст, токенизация и разбиение документов](<../../01-data-features/03-modalities/df-text.md>) → [Языковая модель и её задачи](<../01-foundations/lm-task.md>) → [Токенизация: от текста к индексам](<../01-foundations/lm-tokenizer.md>) → [Embeddings и контекстные состояния](<../01-foundations/lm-states.md>) → [Attention, Q/K/V и несколько голов](<../01-foundations/lm-attention.md>) → [Позиции, causal mask и padding](<../01-foundations/lm-position.md>) → [Transformer-блок: residual, normalization и MLP](<../01-foundations/lm-transformer.md>) → [Encoder, decoder-only и encoder–decoder](<../01-foundations/lm-families.md>) → [Выход модели: LM head, logits и вероятности](<../01-foundations/lm-output.md>) → [Обучение: targets, loss и обновление весов](<../01-foundations/lm-training.md>) → [BERT в общем ядре: encoder, MLM и задачи](<../01-foundations/lm-bert.md>) → [Encoder, tokenizer, attention и pooling](<bert-interface.md>) → [MLM, альтернативные objectives и семейства encoder](<bert-objectives.md>) → [Данные, task heads, fine-tuning и DAPT](<bert-adaptation.md>) → [Classification, NER, spans и extractive QA](<bert-tasks.md>) → [Метрики, калибровка, OOD и интерпретация](<bert-evaluation.md>)


## Сквозные проекты

- [P1. Классификация обращений — минимальный практический проект](<bert-tasks.md#u-99bcd89231dba8f250>)
- [P2. NER и structured extraction — следующая ступень](<bert-tasks.md#u-9b1d2c67a46c142f66>)
- [P3. Поиск по документам — основная retrieval-практика](<bert-retrieval.md#u-a5d70d09aa7c4a04ca>)
- [P4. Вопросы по русским документам — композиция задач](<bert-tasks.md#u-7c7173126907038e85>)
- [P5. Малый бюджет разметки и domain shift — углубление](<bert-domains.md#u-9ed65ac48814aaa902>)
- [P6. Компактный encoder service — системная ветка](<bert-operations.md#u-10c22e85c88da64062>)
