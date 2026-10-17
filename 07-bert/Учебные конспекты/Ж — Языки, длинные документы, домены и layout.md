# Ж. Когда task меняется вместе с языком, длиной и модальностью

[Оглавление](README.md). Группа G, модули **30–34**. **Происхождение:** углубление нового BERT-плана; общие long-context вопросы из S5 здесь применены к encoder tasks, остальные специализированные ветки — дополнение.

## 1. Русский и multilingual transfer

Morphology, casing, ё/е, омографы, transliteration и code switching меняют tokenization и entity boundaries. Token inflation — отношение token counts одного tokenizer к другому на одних texts — помогает оценить context/cost, но не является direct quality metric. Для 120 против 80 tokens ratio 1.5; это не «на 50% хуже semantics».

Сравнивают task-appropriate Russian checkpoint, mBERT/XLM-R и modern multilingual candidate при same ontology/data. [RuBERT model card](https://huggingface.co/ai-forever/ruBert-base) описывает конкретный checkpoint, но не гарантирует готовый NER/retriever head. Monolingual capacity и multilingual transfer имеют разные trade-offs; число языков pretraining не равно качеству каждого.

Cross-lingual retrieval отдельно задаёт language query и document. English-only fine-tuning с test на русском, translated supervision и multilingual task training — разные protocols. Translation-equivalent texts в разных splits дают leakage даже при разных surface forms. Перевод negation, имен и gender может менять labels; human audit нужен на relevant slices.

Report показывает each-language quality, code-switch, rare entities, calibration и fragmentation. Средний multilingual score способен скрыть плохое качество нужного языка. Если выбирать checkpoint по target Russian validation, это task adaptation selection, а не truly untouched zero-shot benchmark.

## 2. Длинный document не равен набору одинаково размеченных chunks

Truncation проста, но удаляет evidence. Sliding windows сохраняют coverage за extra compute; hierarchical encoder сначала кодирует chunks, затем агрегирует их vectors. Если document label «есть нарушение» определяется двумя из восьми chunks, присвоение positive всем восьми создаёт weak/noisy labels.

Mean aggregation размывает редкий signal, max чувствителен к одному false alarm, learned attention требует supervision и может использовать shortcuts. Для chunk scores $(0,0,0,0,0,0,1,1)$ mean 0.25, max 1; значение label не определяется одной formula без task assumptions. Для evidence extraction важно вернуть source offsets, а не только pooled decision.

Sparse/global attention, например [Longformer](https://arxiv.org/abs/2004.05150), меняет interaction pattern, позволяя selected global tokens и local windows. Для window width $w$ local attention work порядка $Lw$ плюс global interactions, в отличие от full $L^2$. Но ограниченная connectivity требует layers/global routes для distant dependencies. ModernBERT сочетает local/global blocks по своей architecture.

Learned positions классического BERT имеют training length/config. Изменить число в config без обучения новых rows и проверки checkpoint недостаточно. RoPE/native context тоже не гарантирует retrieval distant evidence. Сравнение truncation/hierarchy/native-long-context использует full documents, length/evidence-position slices, same task и total cost.

Ответ на границе windows может быть разделён так, что ни одно окно его не видит целиком. Overlap решает это только для spans ограниченной длины при нужном chunk policy; document-level multi-hop evidence может быть разнесено гораздо дальше. Нужно проверять actual information coverage.

## 3. Domain encoders: наука, специальные тексты и код

[SciBERT](https://aclanthology.org/D19-1371/) исследует scientific text representations, [CodeBERT](https://arxiv.org/abs/2002.08155) — natural/programming language. Domain pretraining меняет data/terminology/objective; название не доказывает correctness любой medical/legal/financial application.

Сначала определяется task: entity/relation extraction, classification, matching или search. Domain shift features отличается от изменения annotation policy: прежний label «положительный» может иметь другую definition. Generic task-trained checkpoint может превзойти domain MLM, которому ещё не обучили task geometry/head.

Сравнение generic task model, domain checkpoint и DAPT проводится с comparable tuning budget и general-domain regression. Кодовые tasks split по repositories/time/clone families: почти одинаковые функции в train/test делают score завышенным. Lexical token similarity не доказывает functional equivalence; нужен task oracle или execution tests, если именно equivalence является целью.

## 4. Aspect sentiment, linking и structured understanding

«Экран отличный, батарея слабая» содержит два аспекта с противоположной polarity. Один overall sentiment label теряет структуру. Можно извлекать aspect spans и классифицировать polarity в их context либо jointly предсказывать pairs. Implicit aspects требуют separate policy: отсутствие explicit span не всегда означает отсутствие opinion.

Entity linking: mention detection → candidate retrieval → context disambiguation → entity ID или NIL. «Мир» может быть организацией, картой/брендом или обычным словом; surface form не определяет ID. Candidate recall ограничивает disambiguator, а ошибки mention boundaries добавляют upstream losses.

Оценка gold mentions показывает conditional quality, predicted mentions — всей системы. Нужны seen/unseen entities, alias shift и NIL cases. Entity typing, relation classification и intent+slots имеют свои schemas и metrics. Returned artifact сохраняет offsets/IDs/evidence; general sentiment classifier не заменяет aspect-conditioned extraction.

## 5. OCR, layout и таблицы

Два бланка могут содержать один bag OCR words «Имя Анна Город Москва», но разное spatial association fields. Plain serialized text может потерять соответствие key/value. [LayoutLM](https://arxiv.org/abs/1912.13318) соединяет text и layout signals; новые variants могут добавлять visual components по своей architecture. Text-only BERT не получает изображение лишь потому, что tokens пришли из OCR.

Для table важны row/column relations, cell coordinates, headers и numerical operations. Serialization — baseline с определённым order, а не universal preservation структуры. Если задача требует сравнить суммы, encoder label head не гарантирует точную арифметику без data/objective/oracle.

Разделяют OCR recognition error, layout association, span extraction и final field correctness. Пропущенное OCR слово downstream head не обязан восстановить из пустоты. Сравнение OCR+text и layout-aware pipeline должно использовать same documents и учитывать цену preprocessing/annotations.

## 6. Контроль и практика модулей 30–34

1. Доказывает ли language coverage качественный Russian NER? **Ответ:** нет; нужен task/language evaluation и alignment.
2. Почему positive document не даёт positive label каждому chunk? **Ответ:** evidence может находиться только в части document.
3. Чем gold-mention linking отличается от end-to-end? **Ответ:** gold исключает ошибки mention detector и потому измеряет условную задачу.
4. Чего нет у text-only OCR sequence? **Ответ:** она может потерять 2D structure и visual evidence, если их явно не передали.

**Практика:** измерьте token counts одной multilingual фразы, разметьте aspect spans и NIL linking case. Для восьми chunks выберите aggregation и test evidence positions. Нарисуйте две формы с одинаковыми словами, но разными layouts; определите недостающие признаки. Обоснуйте shortlist для русского NER и multilingual long retrieval без выбора по одному leaderboard.
