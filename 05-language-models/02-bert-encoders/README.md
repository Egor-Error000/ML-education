# BERT и encoder-модели

[Модуль «LLM и языковые модели»](<../README.md>) · [Сценарии глав](<План.md>)

- [Encoder, tokenizer, attention и pooling](<bert-interface.md>) — основа ветки. Как получить состояния и корректно агрегировать их в представление текста?
- [Цели предобучения BERT и варианты кодировщиков](<bert-objectives.md>) — основа ветки. Что модель видит, что должна предсказать и какие параметры меняет ошибка?
- [Данные, головы задач и адаптация BERT](<bert-adaptation.md>) — основа ветки. Какие данные, выход и параметры следует выбрать для адаптации BERT?
- [Классификация, сущности и извлечение ответов](<bert-tasks.md>) — основа ветки. Как разметка превращается в голову классификации, сущностей или извлечения ответа?
- [Векторы предложений и контрастивное обучение](<bert-embeddings.md>) — основа ветки. Как обучить и проверить векторы предложений для сравнения текстов?
- [Поиск: плотные векторы, BM25, SPLADE и ColBERT](<bert-retrieval.md>) — основа ветки. Как устроены кандидаты, переранжирование и разные представления поискового документа?
- [Языки, длинные документы и пространственная структура](<bert-domains.md>) — основа ветки. Что меняется при переносе encoder между языками, доменами и видами документа?
- [Метрики, калибровка и границы интерпретации](<bert-evaluation.md>) — основа ветки. Какие выводы позволяют метрики, калибровка и объяснения encoder?
- [Сжатие, служба кодировщика и обновление индекса](<bert-operations.md>) — углубление. Как уменьшить стоимость encoder и безопасно для качества обновить служебный комплект?

**BERT без генеративной адаптации:** [Постановка задачи и паспорт данных](<../../01-data-features/01-reliability/df-task.md>) → [Разбиение данных, перекрёстная проверка и утечки](<../../01-data-features/01-reliability/df-split.md>) → [Доставка данных: выборка, пакет и воспроизводимость](<../../01-data-features/01-reliability/df-loading.md>) → [Текст, токенизация и разбиение документов](<../../01-data-features/03-modalities/df-text.md>) → [Языковая модель и её задачи](<../01-foundations/lm-task.md>) → [Токенизация: от текста к индексам](<../01-foundations/lm-tokenizer.md>) → [Embeddings и контекстные состояния](<../01-foundations/lm-states.md>) → [Attention, Q/K/V и несколько голов](<../01-foundations/lm-attention.md>) → [Позиции, causal mask и padding](<../01-foundations/lm-position.md>) → [Transformer-блок: residual, normalization и MLP](<../01-foundations/lm-transformer.md>) → [Encoder, decoder-only и encoder–decoder](<../01-foundations/lm-families.md>) → [Выход модели: LM head, logits и вероятности](<../01-foundations/lm-output.md>) → [Обучение: targets, loss и обновление весов](<../01-foundations/lm-training.md>) → [BERT в общем ядре: encoder, MLM и задачи](<../01-foundations/lm-bert.md>) → [Encoder, tokenizer, attention и pooling](<bert-interface.md>) → [Цели предобучения BERT и варианты кодировщиков](<bert-objectives.md>) → [Данные, головы задач и адаптация BERT](<bert-adaptation.md>) → [Классификация, сущности и извлечение ответов](<bert-tasks.md>) → [Метрики, калибровка и границы интерпретации](<bert-evaluation.md>)


## Сквозные проекты

- [P1. Классификация обращений — минимальный практический проект](<bert-tasks.md#u-99bcd89231dba8f250>)
- [P2. NER и structured extraction — следующая ступень](<bert-tasks.md#u-9b1d2c67a46c142f66>)
- [P3. Поиск по документам — основная retrieval-практика](<bert-retrieval.md#u-a5d70d09aa7c4a04ca>)
- [P4. Вопросы по русским документам — композиция задач](<bert-tasks.md#u-7c7173126907038e85>)
- [P5. Малый бюджет разметки и domain shift — углубление](<bert-domains.md#u-9ed65ac48814aaa902>)
- [P6. Компактный encoder service — системная ветка](<bert-operations.md#u-10c22e85c88da64062>)
