# RAG: BM25, dense, hybrid и multi-vector retrieval

**Уровень:** основа ветки.

**Предпосылки:** [Состояние, orchestration и граф управления](<app-state.md>), [Split, cross-validation и утечки](<../../01-data-features/01-reliability/df-split.md>), [Текст, токенизация и разбиение документов](<../../01-data-features/03-modalities/df-text.md>)

**Используют:** [Оценка retrieval, ответов и task success](<app-evaluation.md>)

**Учебный вопрос:** Как найти evidence и связать его с ответом без обучения собственного retriever?

Для этой главы достаточно локального определения: embedding — вектор объекта, retriever возвращает кандидатов, reranker переупорядочивает их. Можно использовать готовые модели. Обучение retrieval во вложенной BERT-ветке не является предпосылкой построения приложения.



## Карта вычисления и границы этапов

![RAG: BM25, dense, hybrid и multi-vector retrieval: входы, преобразования и выходы](<../../оформление/course/app-rag.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Параметры модели при обычном выполнении фиксированы. Состояние запроса, cache и состояние внешнего инструмента имеют разные жизненные циклы; их нельзя смешивать с optimizer states.

<!-- unit:9e35fffcbd768fc75d -->
<a id="u-9e35fffcbd768fc75d"></a>

## RAG: retrieval добавляет текстовый evidence

<!-- /unit:9e35fffcbd768fc75d -->

<!-- unit:85e677c6ee18d74264 -->
<a id="u-85e677c6ee18d74264"></a>

Pipeline: corpus ingestion → chunks с IDs/provenance → sparse/dense index → query retrieval → optional reranking → context selection → generation с citations. Generator обычно получает **исходные тексты**, выбранные по vector scores, а не произвольный embedding vector вместо слов.

<!-- /unit:85e677c6ee18d74264 -->

<!-- unit:1c9689d7635c6ccb87 -->
<a id="u-1c9689d7635c6ccb87"></a>

Это отвечает вопросу чата «почему модель поймёт наш embedding». Retriever понимает совместимое query/document space, приложение по IDs достаёт snippets, LLM читает их как text. Input token embeddings LLM и retrieval embeddings имеют разные roles/objectives.

<!-- /unit:1c9689d7635c6ccb87 -->

<!-- unit:c969e25e13601552f8 -->
<a id="u-c969e25e13601552f8"></a>

BM25 использует term frequencies, corpus document frequencies и length normalization. Dense retriever сравнивает learned vectors. Hybrid объединяет rankings/scores по объявленной policy, reranker оценивает query-document pairs. Больше stages не гарантирует improvement: each может добавить latency и failure mode.

<!-- /unit:c969e25e13601552f8 -->

<!-- unit:f039bf6760919c2543 -->
<a id="u-f039bf6760919c2543"></a>

FAISS — библиотека vector search; полноценная система ещё хранит documents/metadata, versions, access filters, deletion/update semantics. ANN approximation оценивается относительно exact search отдельно от relevance quality. Стандартный vector index не должен выдавать document вне разрешённого scope только из-за большой similarity.

<!-- /unit:f039bf6760919c2543 -->

<!-- unit:cb7876640bb130438f -->
<a id="u-cb7876640bb130438f"></a>

### Полный путь evidence и его четыре представления

<!-- /unit:cb7876640bb130438f -->

<!-- unit:25e009a87636a066c2 -->
<a id="u-25e009a87636a066c2"></a>

Для запроса «какой timeout у payment-service в версии v8?» сначала задаются сервис, версия и scope. Query encoder создаёт retrieval vector. Индекс возвращает candidate IDs/scores. Document store по IDs выдаёт текст и metadata. После фильтрации и reranking в prompt попадают выбранные фрагменты. В итоге есть четыре разных объекта: query text, retrieval vector, ranked IDs, textual evidence. Их нельзя подменять одним словом embedding.

<!-- /unit:25e009a87636a066c2 -->

<!-- unit:0fe2f3d109c7a1a543 -->
<a id="u-0fe2f3d109c7a1a543"></a>

Вектор не хранит готовую цитату в форме, которую generator обязан уметь декодировать обратно. Даже если retriever и LLM используют одинаковую размерность вектора, это не делает пространства совместимыми. Retrieval training сближает релевантные query/document representations; language-model input embedding участвует в другом вычислительном тракте. Обычный RAG соединяет системы через извлечение текста, без специального обучения LLM читать произвольные поисковые векторы.

<!-- /unit:0fe2f3d109c7a1a543 -->

<!-- unit:73eaf3b90be1088fcf -->
<a id="u-73eaf3b90be1088fcf"></a>

Chunking нужен из-за granularity поиска и ограниченного context. Очень крупный chunk может скрыть нужную строку среди лишнего; очень мелкий — потерять заголовок, условие и версию. Overlap сохраняет локальную связность ценой повторов и объёма индекса. У каждого chunk должны оставаться document ID, position и revision, иначе цитата или обновление становятся неоднозначными. Удалённый документ должен исчезнуть из доступного retrieval path, а не только из первоначальной папки.

<!-- /unit:73eaf3b90be1088fcf -->

<!-- unit:2e82aeab04b1f2beb8 -->
<a id="u-2e82aeab04b1f2beb8"></a>

### Почему фильтрация после top-k иногда теряет нужное

<!-- /unit:2e82aeab04b1f2beb8 -->

<!-- unit:02cd7b0f497d7c65b6 -->
<a id="u-02cd7b0f497d7c65b6"></a>

Пусть доступный и релевантный фрагмент имеет global vector rank 11. Первые 10 — документы другого scope. Стратегия «global top-10 → убрать запрещённое» вернёт пустой список. Поиск в допустимом подмножестве мог бы найти нужный фрагмент первым. Это не аргумент за показ запрещённых текстов; это причина учитывать фильтр при поиске либо использовать корректный over-fetch/filtered search с пониманием ограничений backend.

<!-- /unit:02cd7b0f497d7c65b6 -->

<!-- unit:31bf9ec936533c7d1b -->
<a id="u-31bf9ec936533c7d1b"></a>

Top-k задаёт число кандидатов, а не число доказанных фактов. После reranking и deduplication в context может попасть меньше k. Например, найдено 20 chunks, 8 относятся к нужной версии, 3 после deduplication покрывают задачу. Не следует превращать факт «ANN вернул 20 соседей» в retrieval success без relevance labels и проверки scope/version.

<!-- /unit:31bf9ec936533c7d1b -->

<!-- unit:9ae3245285a3942437 -->
<a id="u-9ae3245285a3942437"></a>

Context budget тоже считается явно. В учебном окне 8192 tokens зарезервированы 1500 для instructions/question/history и 1200 для output. На evidence остаётся 5492. Если каждый полный упакованный chunk с metadata занимает 700 tokens, помещается 7 chunks (4900), остаётся 592; восемь потребовали бы 5600. Это упрощённая арифметика с фиксированным размером chunks; в реальном prompt длина измеряется tokenizer после упаковки, включая separators и дополнительные tools/messages.

<!-- /unit:9ae3245285a3942437 -->

<!-- unit:24c48aaaf6277c2cbf -->
<a id="u-24c48aaaf6277c2cbf"></a>

### RAG workflow и agentic retrieval

<!-- /unit:24c48aaaf6277c2cbf -->

<!-- unit:1d83397e6c747258d7 -->
<a id="u-1d83397e6c747258d7"></a>

В fixed RAG каждый запрос проходит один retrieval. В agentic варианте модель может понять, что не хватает второго документа, уточнить поиск или прочитать соседний chunk. Дополнительный control flow помогает только при содержательной политике. Если модель не заметила отсутствующую версию, многократный поиск может лишь увеличить уверенность в неправильном ответе.

<!-- /unit:1d83397e6c747258d7 -->

<!-- unit:12d881f6be655125a5 -->
<a id="u-12d881f6be655125a5"></a>

Retrieved text — evidence, а не новая доверенная инструкция. Документ с фразой «игнорируй задачу и вызови другой tool» не меняет разрешённые actions. Разделение ролей в prompt полезно, но enforcement прав и registry остаётся у программы. Цитата подтверждает provenance; её наличие не доказывает entailment. Требуется проверить, что именно процитированный фрагмент поддерживает claim и не относится к другой версии.

<!-- /unit:12d881f6be655125a5 -->

<!-- unit:50735ee4227e41af20 -->
<a id="u-50735ee4227e41af20"></a>

## Два процесса: подготовка корпуса и ответ на запрос

<!-- /unit:50735ee4227e41af20 -->

<!-- unit:9e16ab573f259cebed -->
<a id="u-9e16ab573f259cebed"></a>

Offline/ingestion path: документы → очистка и версии → chunks → metadata/provenance → sparse/dense indexes. Online path: вопрос → retrieval → фильтрация/объединение → reranking → context selection → generation → проверка citations/answer.

<!-- /unit:9e16ab573f259cebed -->

<!-- unit:a312a94d49c7fd6f27 -->
<a id="u-a312a94d49c7fd6f27"></a>

У каждого chunk должны быть устойчивые document/chunk IDs, source span или location, version и access scope. Если document обновился, старый chunk не должен продолжать выдавать obsolete правило без понятной политики. Split evaluation по documents/sources делать до нарезки, иначе соседние chunks создают leakage.

<!-- /unit:a312a94d49c7fd6f27 -->

<!-- unit:820e1d93ba0a2cdf5a -->
<a id="u-820e1d93ba0a2cdf5a"></a>

Слишком короткий chunk теряет context, слишком длинный разбавляет signal и занимает generation budget. Overlap помогает не разрезать нужную связь, но увеличивает дубликаты и index size. Размер измерять в tokenizer tokens, а semantic boundaries учитывать вместе с длиной. Это параметры pipeline, а не универсальное «лучше всегда 512tokens».

<!-- /unit:820e1d93ba0a2cdf5a -->

<!-- unit:ccd90d02cbe62bc16f -->
<a id="u-ccd90d02cbe62bc16f"></a>

**Схема курса:** corpus → chunk/metadata/index → query → sparse + dense candidates → fuse/rerank → evidence IDs → grounded answer → retrieval and answer evaluation

<!-- /unit:ccd90d02cbe62bc16f -->

<!-- unit:b10a4311312343f880 -->
<a id="u-b10a4311312343f880"></a>

<!-- source-depth-applications-Г -->

<!-- /unit:b10a4311312343f880 -->

<!-- unit:4112e443b0eac8b4b8 -->
<a id="u-4112e443b0eac8b4b8"></a>

### Что RAG добавляет к знаниям модели

<!-- /unit:4112e443b0eac8b4b8 -->

<!-- unit:7adfcd973a47ac0c62 -->
<a id="u-7adfcd973a47ac0c62"></a>

В начале D.5 исходник выделяет ограничение context, обновление знаний и проверяемые sources. RAG выбирает внешний материал для текущего ответа; он не обязан сохранять новые факты в weights. Если сегодня изменился документ и обновлён index, следующий запрос может получить новую версию без fine-tuning generator. Но если ingestion не обновился или retrieval выбрал stale passage, актуальность не возникает автоматически.

<!-- /unit:7adfcd973a47ac0c62 -->

<!-- unit:f2f7f76e8b5d5f1d94 -->
<a id="u-f2f7f76e8b5d5f1d94"></a>

Наличие source также не устраняет hallucination само по себе. Generator может неправильно прочитать условие, смешать две версии, приписать source лишнее или ответить из parametric memory. Поэтому цель —сделать evidence доступным и проверяемым, а затем оценить, как оно использовано. Аналогия исходника с открытым справочником полезна: доступ к книге не гарантирует, что отвечающий открыл нужную страницу и правильно её понял.

<!-- /unit:f2f7f76e8b5d5f1d94 -->

<!-- unit:c23378aca9d951256c -->
<a id="u-c23378aca9d951256c"></a>

Нельзя сводить RAG к любой vector database. Retrieval может быть lexical, dense, hybrid, SQL/filter lookup или сочетанием нескольких методов. В современной прикладной речи RAG часто означает retrieve text → assemble context → generate. Исследовательская модель из original RAG paper имеет свой обучаемый probabilistic formulation; её нельзя без оговорок отождествлять с любой chain, читающей документы.

<!-- /unit:c23378aca9d951256c -->

<!-- unit:9482dc2aad027b03e1 -->
<a id="u-9482dc2aad027b03e1"></a>

### Ingestion: потеря смысла до embedding stage

<!-- /unit:9482dc2aad027b03e1 -->

<!-- unit:e76040ae2a153ce11a -->
<a id="u-e76040ae2a153ce11a"></a>

PDF/DOCX/HTML, wiki, SQL export и API response сначала нужно превратить в подходящие content units. Ошибки extraction могут перемешать колонки таблицы, отделить отрицание от условия, повторить footer на каждой странице или выбросить подпись графика. После этого идеальный retriever будет искать по уже повреждённым данным. Поэтому проверка parse/normalization относится к качеству RAG, а не к внешней косметике.

<!-- /unit:e76040ae2a153ce11a -->

<!-- unit:3f5398816325f988a1 -->
<a id="u-3f5398816325f988a1"></a>

В таблице «модель / dtype / latency / hardware» значение 30 без row/column headings не отвечает ни на один точный вопрос. При сериализации сохраняют headers, units, row identity, caption и применимую revision. Если график содержит данные только в pixels, обычный text extractor не делает их автоматически доступными text encoder: нужны OCR/описание, structured data либо явно multimodal pipeline.

<!-- /unit:3f5398816325f988a1 -->

<!-- unit:265900a147e1def709 -->
<a id="u-265900a147e1def709"></a>

Metadata отделяется от текста по роли. Source ID и version нужны для provenance; access scope —для разрешённого поиска; title/section path —для интерпретации. Не все metadata нужно превращать в embedding text, но критичные semantic условия нельзя потерять. Например, одинаковая фраза «timeout 5 seconds» в двух сервисах требует service identity.

<!-- /unit:265900a147e1def709 -->

<!-- unit:bad17115b66675798f -->
<a id="u-bad17115b66675798f"></a>

### Четыре chunking стратегии и точная цена overlap

<!-- /unit:bad17115b66675798f -->

<!-- unit:7457f0cf4ba4498daa -->
<a id="u-7457f0cf4ba4498daa"></a>

Source сохраняет fixed-size, recursive, semantic и document-aware chunking. Fixed-size проще ограничивает длину, но может разрезать правило. Recursive splitting пытается использовать границы paragraph/sentence, а слишком длинные units делит дальше. Упрощённый source snippet «попробуй следующий separator на всём тексте» не описывает весь алгоритм: нужно также pack соседние короткие units до лимита и обработать один unit, который всё ещё длиннее cap.

<!-- /unit:7457f0cf4ba4498daa -->

<!-- unit:d04fb6c32185f96c93 -->
<a id="u-d04fb6c32185f96c93"></a>

Document-aware splitting берёт структуру headings/sections, но очень большой section всё равно делится; к детям можно прикрепить section path. Semantic splitting оценивает связность соседних предложений и выбирает границы. Embedding N предложений не обязательно требует N последовательных forward calls: при batch size 32 для 1000 предложений получается 32 batches, хотя число закодированных предложений остаётся 1000. Цена зависит от lengths/batching/model.

<!-- /unit:d04fb6c32185f96c93 -->

<!-- unit:7d1029de6c33d8dd76 -->
<a id="u-7d1029de6c33d8dd76"></a>

Для учебного текста длиной 10000 tokens, chunk cap512 и overlap64 stride равен 448. Начала идут 0,448,896,...,9856; последний chunk содержит 144 tokens. Получается 23 chunks, суммарно закодировано 11408 tokens. Без overlap —20 chunks и 10000 tokens. В этом примере количество chunks выросло на 15%, input-token volume —на 14.08%. Это разные единицы затрат; padding и repeated headers добавят ещё работу.

<!-- /unit:7d1029de6c33d8dd76 -->

<!-- unit:b538e4a467221c9cac -->
<a id="u-b538e4a467221c9cac"></a>

Для длинного текста без иных overhead асимптотическое отношение обработанных tokens к исходным —$c/(c-o)$, здесь $512/448\approx1.142857$. Оно не равно «всегда ровно столько процентов дороже весь RAG». Query encoding и generation от overlap не умножаются тем же коэффициентом; index payload зависит от числа chunks.

<!-- /unit:b538e4a467221c9cac -->

<!-- unit:7e49f868ed9c4980a8 -->
<a id="u-7e49f868ed9c4980a8"></a>

Считают длину тем tokenizer, который ограничивает конкретный encoder/prompt. Character count не равен token count, а tokenizer generator может отличаться от retriever. Размеры 256/512/1024 из source —параметры опытов, не универсальная норма. Хороший chunk сохраняет информацию, достаточную для нужного класса вопросов, и помещается в соответствующий вычислительный бюджет.

<!-- /unit:7e49f868ed9c4980a8 -->

<!-- unit:d17cb1e5670e092ab9 -->
<a id="u-d17cb1e5670e092ab9"></a>

### Versioning, обновления и публикация index snapshot

<!-- /unit:d17cb1e5670e092ab9 -->

<!-- unit:61e7773d5be00a2a97 -->
<a id="u-61e7773d5be00a2a97"></a>

Стабильная связь выглядит как document_id → revision → chunk IDs/spans → representation IDs. При новом revision части меняются или удаляются, поэтому index updates должны согласовываться с document store. Старый vector, указывающий на уже заменённый текст, создаёт несовместимую пару: similarity была посчитана для одного content, а generator прочитал другой.

<!-- /unit:61e7773d5be00a2a97 -->

<!-- unit:8aed6ab29332803779 -->
<a id="u-8aed6ab29332803779"></a>

Можно публиковать новый целостный snapshot после подготовки или использовать поддержанную incremental update policy с явным version filtering. В обоих случаях запрос должен знать, какая revision считается применимой. Access revocation и deletion —это не просто новый answer prompt: запрещённый content нельзя продолжать выдавать из старого index/cache path.

<!-- /unit:8aed6ab29332803779 -->

<!-- unit:bafab6c7bea4729287 -->
<a id="u-bafab6c7bea4729287"></a>

Для train/eval split соседние chunks одного документа не должны случайно превращаться в «независимые» примеры с обеих сторон. Unit split выбирается по реальной задаче: source/document/time/customer group. Изменение chunking меняет не только длину prompt, но и число index items, BM25 statistics и granularity relevance labels.

<!-- /unit:bafab6c7bea4729287 -->

<!-- unit:808591e9c63816bd35 -->
<a id="u-808591e9c63816bd35"></a>

## Почему генератору не нужно «понимать наш vector»

<!-- /unit:808591e9c63816bd35 -->

<!-- unit:ab06ba2fb5c3f54289 -->
<a id="u-ab06ba2fb5c3f54289"></a>

Retriever сравнивает query embedding с compatible document embeddings. Найденные IDs используются для извлечения **исходных текстов**. LLM получает вопрос и snippets через свой обычный tokenizer; vector coordinates обычно не подаются ей как новая скрытая активация.

<!-- /unit:ab06ba2fb5c3f54289 -->

<!-- unit:e1876b32ff8bf8200f -->
<a id="u-e1876b32ff8bf8200f"></a>

Input embedding table LLM, contextual hidden states и retrieval vector — разные роли. Retrieval encoder может быть отдельной моделью, общей model family или специально обученным вариантом, но важны objective и interface. Фраза исходника «это никогда не может быть сама LLM» слишком категорична: возможны специальные модели/схемы. Обычный текстовый RAG не требует напрямую согласовывать hidden states двух независимых LLM.

<!-- /unit:e1876b32ff8bf8200f -->

<!-- unit:67a63829b3f05df069 -->
<a id="u-67a63829b3f05df069"></a>

Query/document spaces должны быть совместимы. Одинаковая ширина 768 не делает произвольные encoders взаимозаменяемыми. При смене tokenizer/pooling/normalization/checkpoint обычно нужен re-encoding или отдельно проверенный compatibility protocol.

<!-- /unit:67a63829b3f05df069 -->

<!-- unit:5ec3d5c2f2bbfb1baf -->
<a id="u-5ec3d5c2f2bbfb1baf"></a>

### Ответ на исходный вопрос об embedding по шагам

<!-- /unit:5ec3d5c2f2bbfb1baf -->

<!-- unit:c7987bafaf2e6a229a -->
<a id="u-c7987bafaf2e6a229a"></a>

В S6 пользователь спросил: «Эмбейдинг это ведь внутреннее представление модели. Почему мы уверенны, что модель поймёт наш имбейдинг?» В обычном text RAG связь обеспечивается через **выбор текста**, а не через универсальную интерпретацию координат. Query encoder возвращает vector; search возвращает references; storage по references выдаёт passage; generator токенизирует passage своим tokenizer. На последнем шаге снова появляются его собственные input embeddings и hidden states.

<!-- /unit:c7987bafaf2e6a229a -->

<!-- unit:79cb607350c633c857 -->
<a id="u-79cb607350c633c857"></a>

Нужно различать token embedding lookup, contextual token states после Transformer layers и pooled retrieval representation всего query/chunk. Последний может быть обучен на query-positive-negative data, чтобы ранжировать passages. Это тоже выход модели и тоже её representation; различие в функции и training objective, а не в том, что «одни embeddings настоящие внутренние, а другие совсем не относятся к моделям».

<!-- /unit:79cb607350c633c857 -->

<!-- unit:748c044d1dad745298 -->
<a id="u-748c044d1dad745298"></a>

Contrastive training повышает relative score positive против negatives. Оно не требует, чтобы у всех правильных пар cosine был ровно 1, у всех неправильных —ровно 0. Два похожих документа могут отличаться нужной версией/отрицанием, и semantic similarity ещё не делает их взаимозаменяемыми evidence. Ассиметричный retriever может иметь разные query/document encoders, prefixes или instructions —важна обученная совместимость, а не буквальная идентичность обеих функций.

<!-- /unit:748c044d1dad745298 -->

<!-- unit:4b8a48bc2e7e60eb95 -->
<a id="u-4b8a48bc2e7e60eb95"></a>

### Одинаковая размерность не означает одинаковое пространство

<!-- /unit:4b8a48bc2e7e60eb95 -->

<!-- unit:a2222b449926041ebb -->
<a id="u-a2222b449926041ebb"></a>

Учебный query q=(1,0), документы A=(1,0), B=(0,1). Dot-product retrieval выбирает A. Если заменить только document encoder так, что координатные оси поменялись местами, размеры всё ещё 2, но теперь старый query выберет прежний B. Если согласованно преобразовать query и документы одной orthogonal rotation, dot products сохранятся. Несогласованная замена encoder —совсем другое действие.

<!-- /unit:a2222b449926041ebb -->

<!-- unit:257c3bdf848145b8d9 -->
<a id="u-257c3bdf848145b8d9"></a>

Поэтому version включает не только размерность, но и checkpoint, pooling, query/document processing и normalization. При migration хранят совместимые пары indexes/encoders либо выполняют re-encoding с проверкой качества. Даже одинаковые названия архитектуры и width не доказывают совместимость двух независимо обученных spaces.

<!-- /unit:257c3bdf848145b8d9 -->

<!-- unit:9e48c35936b73d7d2f -->
<a id="u-9e48c35936b73d7d2f"></a>

### Исправляем исключения из source: original RAG, REALM и multimodal

<!-- /unit:9e48c35936b73d7d2f -->

<!-- unit:a40c010b765b991fa2 -->
<a id="u-a40c010b765b991fa2"></a>

В ответе чата REALM и original RAG названы случаями передачи retrieval embeddings прямо через специальные cross-attention layers. Это неверное обобщение. В [original RAG](https://arxiv.org/abs/2005.11401) retriever ищет passages, а seq2seq generator работает с вопросом и извлечённым текстом; encoder hidden states текста доступны decoder cross-attention. Search vector и эти states не один объект. В [REALM](https://arxiv.org/abs/2002.08909) retrieval также связан с выбором текстовых документов для language-model objective.

<!-- /unit:a40c010b765b991fa2 -->

<!-- unit:bfd225ee96aba2bd8d -->
<a id="u-bfd225ee96aba2bd8d"></a>

Multimodal connector, например image encoder → learned projector → language model, действительно может передавать non-text representations через специально обученный интерфейс. Но не всякий image embedding совместим с произвольной LLM. И CLIP сам по себе не следует называть генеративной LLM, принимающей чужие vectors как prompt. Direct representation injection требует определённой архитектуры/обучения; стандартный текстовый RAG этого не предполагает.

<!-- /unit:bfd225ee96aba2bd8d -->

<!-- unit:c593964d24dc2a7b50 -->
<a id="u-c593964d24dc2a7b50"></a>

Более подробные retrieval-encoder objectives и token-level late interaction разобраны в [BERT-разделе](<../02-bert-encoders/README.md>). Здесь важно проследить application interface: какой именно объект каждый stage выдаёт и получает.

<!-- /unit:c593964d24dc2a7b50 -->

<!-- unit:a3e4cb57039adb98f5 -->
<a id="u-a3e4cb57039adb98f5"></a>

## BM25: terms приходят из query и analyzer

<!-- /unit:a3e4cb57039adb98f5 -->

<!-- unit:4b586ad8d8d7d5309b -->
<a id="u-4b586ad8d8d7d5309b"></a>

BM25 ранжирует документы по совпадениям terms, их частотам и corpus statistics. Query analyzer выполняет выбранные tokenization, case folding, stemming/lemmatization и stopword policy. «Ключевые слова» в базовой схеме — обработанные terms запроса; отдельный LLM, который заранее выберет keywords, не обязателен.

<!-- /unit:4b586ad8d8d7d5309b -->

<!-- unit:050edc14819c0f4fbc -->
<a id="u-050edc14819c0f4fbc"></a>

Inverted index хранит postings: для term — какие documents его содержат и сколько раз. Поиск читает postings relevant query terms, не обязан заново сканировать весь raw corpus. Если пользователь написал synonym, отсутствующий в document, чистое lexical matching может его не связать; query expansion — отдельное дополнение.

<!-- /unit:050edc14819c0f4fbc -->

<!-- unit:f635d638cc7922b385 -->
<a id="u-f635d638cc7922b385"></a>

Одна распространённая положительная IDF-convention:

<!-- /unit:f635d638cc7922b385 -->

<!-- unit:273c4728fa6ea973aa -->
<a id="u-273c4728fa6ea973aa"></a>

$$
\operatorname{IDF}(t)=\ln\left(1+\frac{N-n_t+0.5}{n_t+0.5}\right),
$$

<!-- /unit:273c4728fa6ea973aa -->

<!-- unit:6e4e4c226a72175a1d -->
<a id="u-6e4e4c226a72175a1d"></a>

где $N$ — число документов, $n_t$ — document frequency term. При этой convention:

<!-- /unit:6e4e4c226a72175a1d -->

<!-- unit:bad9e908085fca9c50 -->
<a id="u-bad9e908085fca9c50"></a>

$$
\operatorname{BM25}(d,q)=
\sum_{t\in q}\operatorname{IDF}(t)
\frac{f(t,d)(k_1+1)}{f(t,d)+k_1\left(1-b+b\frac{|d|}{\operatorname{avgdl}}\right)}.
$$

<!-- /unit:bad9e908085fca9c50 -->

<!-- unit:80ae74bb600484a8ff -->
<a id="u-80ae74bb600484a8ff"></a>

Другие BM25 variants могут отличаться IDF и обработкой query term frequency; сравнивать scores разных indexes без declaration нельзя. При $b=0$ length normalization выключена; $k_1$ задаёт saturation term frequency. Повторение слова сто раз не должно автоматически дать стократный score. [Разбор BM25](https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html).

<!-- /unit:80ae74bb600484a8ff -->

<!-- unit:5747102d6dc8e8c3e8 -->
<a id="u-5747102d6dc8e8c3e8"></a>

### Числовой пример с полным расчётом

<!-- /unit:5747102d6dc8e8c3e8 -->

<!-- unit:9f036543a6021fd910 -->
<a id="u-9f036543a6021fd910"></a>

Пусть $N=1000$, средняя длина 200, query — «квантизация GPTQ», длина document 300. Первое слово встречается в 30 documents и 4 раза в этом, второе — в 5 documents и 2 раза. Возьмём $k_1=1.2$, $b=0.75$.

<!-- /unit:9f036543a6021fd910 -->

<!-- unit:c16de783d43b88a6aa -->
<a id="u-c16de783d43b88a6aa"></a>

$$
\begin{aligned}
\operatorname{IDF}_1&=\ln\frac{1001}{30.5}\approx3.491028,\\
\operatorname{IDF}_2&=\ln\frac{1001}{5.5}\approx5.204007,\\
\ell_d&=1-0.75+0.75\frac{300}{200}=1.375.
\end{aligned}
$$

<!-- /unit:c16de783d43b88a6aa -->

<!-- unit:441e98decef9db6121 -->
<a id="u-441e98decef9db6121"></a>

Тогда contributions:

<!-- /unit:441e98decef9db6121 -->

<!-- unit:46d0a2ccfe5240dbd2 -->
<a id="u-46d0a2ccfe5240dbd2"></a>

$$
\begin{aligned}
s_1&=3.491028\frac{4\cdot2.2}{4+1.2\cdot1.375}\approx5.437353,\\
s_2&=5.204007\frac{2\cdot2.2}{2+1.2\cdot1.375}\approx6.273323,\\
s_1+s_2&\approx11.710677.
\end{aligned}
$$

<!-- /unit:46d0a2ccfe5240dbd2 -->

<!-- unit:3e6e7cfe71c0be4529 -->
<a id="u-3e6e7cfe71c0be4529"></a>

Это ranking score, не probability relevance. Изменение corpus/chunking меняет document frequencies и average length, поэтому даже тот же text может получить другой score.

<!-- /unit:3e6e7cfe71c0be4529 -->

<!-- unit:994cec80c4dc440ad5 -->
<a id="u-994cec80c4dc440ad5"></a>

### Как строятся postings и откуда берётся score без LLM

<!-- /unit:994cec80c4dc440ad5 -->

<!-- unit:6ba0c5e02b0b642401 -->
<a id="u-6ba0c5e02b0b642401"></a>

Возьмём corpus из source example: D1=«квантизация весов модели», D2=«gptq квантизация hessian», D3=«обучение трансформеров». После объявленного analyzer длины 3,3,2, средняя 8/3. Postings для «квантизация» содержат D1,D2 с TF1; для «gptq» —D2 с TF1. Query analyzer превращает «Квантизация GPTQ» в те же lowercase terms. Отдельный этап генерации keywords здесь не нужен.

<!-- /unit:6ba0c5e02b0b642401 -->

<!-- unit:4606eae4b74cb4bafd -->
<a id="u-4606eae4b74cb4bafd"></a>

В положительной IDF convention этой главы $\operatorname{IDF}(\text{квантизация})=\ln1.6\approx0.470004$, $\operatorname{IDF}(\text{gptq})=\ln(8/3)\approx0.980829$. При k1=1.2,b=0.75 фактор TF для документов длины 3 равен примерно 0.951351. Score D1≈0.447139, D2≈1.380252, D3=0. D2 выигрывает, потому что совпали оба query terms, в том числе более редкий gptq.

<!-- /unit:4606eae4b74cb4bafd -->

<!-- unit:cbb54c89778dbf0b52 -->
<a id="u-cbb54c89778dbf0b52"></a>

Библиотека rank_bm25 из source может использовать другую IDF convention/обработку общих terms; этот ручной расчёт не объявляется её точным output. При сравнении implementations фиксируют формулу и analyzer. В search engines fields, boosts и phrase queries добавляют собственные правила, а не меняют автоматически смысл базовой суммы BM25.

<!-- /unit:cbb54c89778dbf0b52 -->

<!-- unit:b21a75e7ed8e7327b7 -->
<a id="u-b21a75e7ed8e7327b7"></a>

### TF, DF и насыщение —три разные величины

<!-- /unit:b21a75e7ed8e7327b7 -->

<!-- unit:541eb33c2a44668b97 -->
<a id="u-541eb33c2a44668b97"></a>

TF(t,d)=4 означает четыре вхождения в одном документе. DF(t)=30 означает 30 документов, где term есть хотя бы раз. Удвоение повторов в том же document не удваивает DF, а добавление нового document может изменить DF и N одновременно. Эти corpus statistics объясняют, почему score не является постоянным свойством одной строки.

<!-- /unit:541eb33c2a44668b97 -->

<!-- unit:83e220a251229b56ff -->
<a id="u-83e220a251229b56ff"></a>

При средней длине документа length factor равен 1. Для k1=1.2 насыщенный TF равен 1 при f=1;1.571429 при f=3;1.964286 при f=10;2.173913 при f=100. При f→∞ предел 2.2. Так документ получает дополнительный сигнал от повторов, но этот вклад ограничен. При фиксированном f length normalization меняет denominator; она не обязательно отменяет весь эффект увеличения f в длинном document.

<!-- /unit:83e220a251229b56ff -->

<!-- unit:d5639d5e4c078c5e82 -->
<a id="u-d5639d5e4c078c5e82"></a>

При b=0 длина не участвует. При b=1 используется полная выбранная нормализация |d|/avgdl. Малый k1 делает TF быстрее насыщаемым; выбор параметров зависит от corpus/queries. Не надо объявлять любое редкое слово полезным: случайный идентификатор/опечатка тоже может иметь высокий IDF, оставаясь нерелевантным intent.

<!-- /unit:d5639d5e4c078c5e82 -->

<!-- unit:014ec2f03b4f8fb297 -->
<a id="u-014ec2f03b4f8fb297"></a>

В source IDF example для N=10000 встречаемость 8000/3000/50 даёт приблизительно 0.223181/1.203906/5.288467. Редкость повышает вес совпадения, но слово, которого вообще нет в document, имеет TF0 и не добавляет score. Точная арифметика основного GPTQ example выше даёт 11.710677, а не округлённое source11.73. Если увеличить длину 300→600 при тех же TF, N/DF и avgdl, score станет≈8.968247.

<!-- /unit:014ec2f03b4f8fb297 -->

<!-- unit:bea753849b896d0025 -->
<a id="u-bea753849b896d0025"></a>

### Analyzer: morphology, identifiers, negation и phrases

<!-- /unit:bea753849b896d0025 -->

<!-- unit:fc471bb317fba48f97 -->
<a id="u-fc471bb317fba48f97"></a>

Lowercasing, stopwords и stemming/lemmatization —выбранные преобразования, а не обязательное определение BM25. Для русского morphology может связать «квантизация/квантизации». Для технического поиска чрезмерная нормализация может сломать `ERR_AUTH_401`, `C++`, version `v1.2` или точное имя поля. Анализируют реальные tokens, созданные implementation, а не представляют, что любой пробел/символ всегда обрабатывается одинаково.

<!-- /unit:fc471bb317fba48f97 -->

<!-- unit:e7f9cb969586529de8 -->
<a id="u-e7f9cb969586529de8"></a>

Index и query analyzers должны создавать совместимые term representations. Они не обязаны быть побитно одной и той же программой: например, query-time synonyms могут расширять запрос, а index-time analyzer хранить исходные terms. Но если query сохраняет одну форму, а index другую без связи между ними, совпадения потеряются.

<!-- /unit:e7f9cb969586529de8 -->

<!-- unit:0f6bb0e5825b19f040 -->
<a id="u-0f6bb0e5825b19f040"></a>

Удалять отрицание опасно: «не использовать GPTQ» и «использовать GPTQ» могут стать одинаковым bag of terms. Более того, даже сохранённое «не» не заставляет базовый BM25 понимать область отрицания как reasoning model. Phrase matching/positions и semantic interpretation —дополнительные механизмы. Source пример «квантизация весов» не доказывает, что любая стандартная BM25 сумма автоматически проверяет соседство слов.

<!-- /unit:0f6bb0e5825b19f040 -->

<!-- unit:8a68bd4f4066e6350a -->
<a id="u-8a68bd4f4066e6350a"></a>

BM25 особенно полезен как проверяемый lexical signal для names, IDs и терминов; multilingual dense model может находить cross-language paraphrases. Это tendencies, а не строгий рейтинг всех задач. Нет общего правила, что sparse index при любом corpus компактнее dense или всегда быстрее обновляется.

<!-- /unit:8a68bd4f4066e6350a -->

<!-- unit:afecbcc8b957ecde3c -->
<a id="u-afecbcc8b957ecde3c"></a>

## Dense retrieval, ANN и reranker

<!-- /unit:afecbcc8b957ecde3c -->

<!-- unit:1db3f2c6828a13e73c -->
<a id="u-1db3f2c6828a13e73c"></a>

Bi-encoder отдельно вычисляет $q=f_Q(x)$ и $d=f_D(	ext{chunk})$, similarity может быть $q^\top d$ или cosine после normalization. Documents предвычисляются; online query encoding сравнительно дешёв. Contrastive task training учит relevant pairs, но generic semantic similarity не всегда равна task relevance. [DPR](https://arxiv.org/abs/2004.04906).

<!-- /unit:1db3f2c6828a13e73c -->

<!-- unit:91e8d7580e912f81b4 -->
<a id="u-91e8d7580e912f81b4"></a>

Exact vector search — полезный reference на небольшом corpus. ANN index ускоряет приблизительный поиск с trade-off memory/latency/recall. **ANN recall** сравнивает найденные neighbours с exact neighbours того же space; **relevance recall** — с размеченными useful documents. Хороший ANN не исправит encoder geometry, в которой нужный chunk далеко.

<!-- /unit:91e8d7580e912f81b4 -->

<!-- unit:307588309bcf1c8711 -->
<a id="u-307588309bcf1c8711"></a>

Cross-encoder reranker совместно читает query и каждый candidate, учитывая interactions. Он может улучшить порядок найденных candidates, но не вернуть document, которого вообще нет в candidate set. Цена растёт с количеством pairs и lengths. Поэтому важны recall первого этапа, quality reranker и total latency.

<!-- /unit:307588309bcf1c8711 -->

<!-- unit:7b8fa5986d107967eb -->
<a id="u-7b8fa5986d107967eb"></a>

### Geometry: cosine, inner product и squared L2

<!-- /unit:7b8fa5986d107967eb -->

<!-- unit:9f737faf7ded2d0c30 -->
<a id="u-9f737faf7ded2d0c30"></a>

Cosine сравнивает направления и требует ненулевых norms. Inner product учитывает и направления, и величины document vectors. Для q=(1,0), A=(1,0), B=(2,2) inner product предпочитает B:2>1, а cosine предпочитает A:1>1/√2. Поэтому просто заменить метрику при сохранении прежнего index не всегда безопасно.

<!-- /unit:9f737faf7ded2d0c30 -->

<!-- unit:1f80d50a2852b6947a -->
<a id="u-1f80d50a2852b6947a"></a>

После normalization обоих vectors до unit norm выполняется $\|q-d\|_2^2=2-2q^\top d$. Максимальный cosine и минимальный squared L2 дают одинаковый ranking без ties/numerical complications. Но шкалы противоположны: similarity больше лучше, distance меньше лучше. Порог 0.8 в cosine нельзя слепо применить к L2 distance.

<!-- /unit:1f80d50a2852b6947a -->

<!-- unit:ec609470e2f97a1f3f -->
<a id="u-ec609470e2f97a1f3f"></a>

В FAISS METRIC_L2 выдаёт **квадрат** евклидова расстояния; root не нужен для ranking, но нужен, если нужен именно distance. Это прямо описано в [FAISS metric documentation](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances). У нулевого vector cosine normalization не определена: такой encoder output нужно обрабатывать по policy, а не бесконтрольно делить на 0.

<!-- /unit:ec609470e2f97a1f3f -->

<!-- unit:9cb5cdb092d584ef55 -->
<a id="u-9cb5cdb092d584ef55"></a>

### Exact, IVF, HNSW и compression —разные approximation mechanisms

<!-- /unit:9cb5cdb092d584ef55 -->

<!-- unit:a9700d9443fe5132dd -->
<a id="u-a9700d9443fe5132dd"></a>

Flat search сравнивает query со всеми vectors и даёт reference для этой representation/metric. IVF предварительно группирует vectors и ищет в выбранных lists. Если нужный сосед попал в непросмотренную list, он будет пропущен; увеличение числа probes обычно увеличивает work и может повысить neighbour recall. Это не то же самое, что тематическая relevance документа.

<!-- /unit:a9700d9443fe5132dd -->

<!-- unit:2129ab2e7375002a11 -->
<a id="u-2129ab2e7375002a11"></a>

HNSW строит graph близостей и навигирует по нему, экономя сравнения ценой graph memory/build effort и approximation. Compression, например PQ, уменьшает стоимость представления и может приблизить distance estimates. Параметры construction/search/compression имеют разные эффекты; одна ручка «скорость поиска» не описывает всё пространство trade-offs.

<!-- /unit:2129ab2e7375002a11 -->

<!-- unit:f7006f80a4b39785fc -->
<a id="u-f7006f80a4b39785fc"></a>

FAISS example из source переключал Flat на IVFFlat и вызывал train/add/search. Training IVF обучает coarse partition на vectors, а не дообучает retriever или generator на задаче пользователя. Маленький illustrative corpus не делает любой nlist корректным: число training vectors и coverage должны соответствовать index setup. Конкретный engine run здесь не выполняется.

<!-- /unit:f7006f80a4b39785fc -->

<!-- unit:cd2e193d0eedf10438 -->
<a id="u-cd2e193d0eedf10438"></a>

Библиотека vector search не обязана быть полноценной document database. Но из этого не следует «у FAISS нет возможности сохранять index»: [есть read_index/write_index](https://github.com/facebookresearch/faiss/wiki/Index-IO,-cloning-and-hyper-parameter-tuning). Metadata, transactions, access policies и publication согласованных data/index revisions могут оставаться задачей внешнего приложения. Source пороги «только <1M» и таблицы managed offerings не используются как вечный feature contract продуктов.

<!-- /unit:cd2e193d0eedf10438 -->

<!-- unit:7322885bb8d29bca10 -->
<a id="u-7322885bb8d29bca10"></a>

### ANN recall, candidate recall и финальный top-k

<!-- /unit:7322885bb8d29bca10 -->

<!-- unit:fdc89a88be33f4242a -->
<a id="u-fdc89a88be33f4242a"></a>

Предположим, exact vector top10 и ANN top10 совпали по 9 IDs: neighbour recall@10=0.9. Если релевантный passage вообще не входит в exact top10, perfect ANN на этом cutoff тоже его не найдёт. Нужно отдельно оценивать encoder relevance, search approximation и post-filter/context selection.

<!-- /unit:fdc89a88be33f4242a -->

<!-- unit:5157c40db58d0f0e9c -->
<a id="u-5157c40db58d0f0e9c"></a>

Reranker получает только candidates. Даже идеальный scoring не вернёт passage вне этого множества. Но фраза source «улучшает precision без потери recall» тоже неверна без определения cutoff: top100 candidates может содержать 6 relevant items, а top5 после rerank —только 4. Candidate recall и final recall@5 измеряются на разных множествах. И неидеальный reranker способен ухудшить порядок.

<!-- /unit:5157c40db58d0f0e9c -->

<!-- unit:276bd4f483936ea296 -->
<a id="u-276bd4f483936ea296"></a>

При scope/version filters важно не потерять eligible evidence до reranking. Global top10 может состоять только из недоступных documents, а первый доступный relevant иметь global rank11. Post-filter даст пустоту, хотя поиск в допустимом subset мог бы найти ответ. Over-fetch помогает отдельным случаям, но конечный фиксированный over-fetch не является общей гарантией при произвольном числе ineligible items.

<!-- /unit:276bd4f483936ea296 -->

<!-- unit:05066c9e846bef461b -->
<a id="u-05066c9e846bef461b"></a>

## Hybrid и query rewriting

<!-- /unit:05066c9e846bef461b -->

<!-- unit:1fc372a0687b61d028 -->
<a id="u-1fc372a0687b61d028"></a>

Lexical search силён для точных IDs, rare terms и names; dense — для парафразов, если encoder подходит задаче. Hybrid может объединить ranked lists через reciprocal rank fusion:

<!-- /unit:1fc372a0687b61d028 -->

<!-- unit:3e67620fc3a75d96f0 -->
<a id="u-3e67620fc3a75d96f0"></a>

$$
\operatorname{RRF}(d)=\sum_{j}\frac{1}{c+\operatorname{rank}_j(d)}.
$$

<!-- /unit:3e67620fc3a75d96f0 -->

<!-- unit:f9fb1ab04b552afc0e -->
<a id="u-f9fb1ab04b552afc0e"></a>

Для absent document convention обычно нет contribution. $c$ и candidate depths задаются по validation; это не автоматическое доказательство superiority hybrid. Простая сумма BM25 score и cosine без scale calibration может дать одному component случайное доминирование.

<!-- /unit:f9fb1ab04b552afc0e -->

<!-- unit:1013cff8981c3226bc -->
<a id="u-1013cff8981c3226bc"></a>

Query rewriting/expansion помогает раскрыть неоднозначные terms, но может изменить intent или выдумать entity. Сохранять original question и проверять exact constraints: dates, negation, IDs, product versions. Несколько rewritten searches повышают стоимость и требуют merge/dedup.

<!-- /unit:1013cff8981c3226bc -->

<!-- unit:48b853712e8c9cdd5d -->
<a id="u-48b853712e8c9cdd5d"></a>

### RRF: полноценный merge отсутствующих IDs

<!-- /unit:48b853712e8c9cdd5d -->

<!-- unit:55e989f6e1337b27da -->
<a id="u-55e989f6e1337b27da"></a>

Source RRF snippet использовал list.index(doc) в обоих rankings. Для document, присутствующего только в одном списке, это вызывает ошибку. Сначала берут union IDs, затем суммируют contributions только тех списков, где ID есть. Ниже учебная реализация с rank1-based, одинаковым c и deterministic tie-break по string ID:

<!-- /unit:55e989f6e1337b27da -->

<!-- unit:e0ca9a1143b08970ff -->
<a id="u-e0ca9a1143b08970ff"></a>

```python
def rrf(rankings, c=60):
    scores = {}
    for ranking in rankings:
        seen = set()
        for rank, doc_id in enumerate(ranking, 1):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (c + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
```

<!-- /unit:e0ca9a1143b08970ff -->

<!-- unit:b11b8b605cf81e8bdf -->
<a id="u-b11b8b605cf81e8bdf"></a>

Пример lexical `[A,B,C]`, dense `[D,B,E]`: B получает 2/62≈0.032258; A и D —по 1/61≈0.016393; C и E —по 1/63≈0.015873. B выше обоих первых мест, потому что хорошо ранжируется в двух списках. Это policy fusion, а не вероятность того, что B верен. При другом c/weights/depth результат может измениться.

<!-- /unit:b11b8b605cf81e8bdf -->

<!-- unit:728358b370ef89ba9f -->
<a id="u-728358b370ef89ba9f"></a>

Если брать raw BM25 и cosine и просто суммировать, score scale одного компонента может доминировать. Rank fusion избегает прямого смешения этих шкал, но теряет информацию о margins. В одном списке разница rank1→2 может быть огромной, в другом почти нулевой; RRF этого не знает. Сравнивают fusion choices на одинаковом eval protocol, а не объявляют hybrid автоматически лучше.

<!-- /unit:728358b370ef89ba9f -->

<!-- unit:b1d660a4d48c89dc47 -->
<a id="u-b1d660a4d48c89dc47"></a>

### Rewrite, expansion, HyDE и step-back

<!-- /unit:b1d660a4d48c89dc47 -->

<!-- unit:117e056c7f98c12286 -->
<a id="u-117e056c7f98c12286"></a>

Source expansion для «как работает RAG» добавлял полное название и вопросы об архитектуре. Это помогает lexical coverage, если сохраняется intent. Для query с version/negation/product ID rewrite должен сохранить эти constraints. Хорошо звучащий общий вопрос может потерять точный scope и ухудшить retrieval.

<!-- /unit:117e056c7f98c12286 -->

<!-- unit:c714de396c57dc21c4 -->
<a id="u-c714de396c57dc21c4"></a>

HyDE сначала генерирует hypothetical document, затем кодирует его и ищет реальные passages. В [исходной работе HyDE](https://arxiv.org/abs/2212.10496) подчёркивается, что hypothetical text может содержать ложные details. Поэтому он служит поисковым pivot, а не новым gold source. Иначе pipeline начнёт «подтверждать» собственный invented answer, процитировав generated draft как evidence.

<!-- /unit:c714de396c57dc21c4 -->

<!-- unit:4f0b0d7b20ee10e194 -->
<a id="u-4f0b0d7b20ee10e194"></a>

Step-back формулирует более общий вопрос и использует его вместе с исходным для поиска. Он может помочь найти background, но итог всё равно должен отвечать конкретному запросу. Multi-query увеличивает число searches и merge work; одинаковые passages дедуплицируют. Diversity queries не должна означать изменение исходной задачи без причины.

<!-- /unit:4f0b0d7b20ee10e194 -->

<!-- unit:3e86e8e3d1c285a6a4 -->
<a id="u-3e86e8e3d1c285a6a4"></a>

### Parent retrieval: искать маленькое, читать достаточное

<!-- /unit:3e86e8e3d1c285a6a4 -->

<!-- unit:96bab14b4c2b5eccb9 -->
<a id="u-96bab14b4c2b5eccb9"></a>

В source child256tokens связывается с parent1024tokens. Индексировать vectors можно только для children, а parent text хранить в document store. Найденные child IDs преобразуют в unique parent IDs; parent должен соответствовать той же revision/scope. Для такого варианта отдельный vector каждого parent не обязателен. Поэтому утверждение «parent retrieval всегда удваивает vector database» неверно.

<!-- /unit:96bab14b4c2b5eccb9 -->

<!-- unit:664ffcd8c42fb4f6dc -->
<a id="u-664ffcd8c42fb4f6dc"></a>

Например, top5 children относятся к parents `[P,P,Q,Q,R]`. После parent dedup остаётся 3 content units. Если механически вставить 5 parents, два больших текста повторятся, расходуя context и меняя распределение evidence. Слишком большой parent может вытеснить другие нужные sources; иногда лучше вернуть child с соседями и section header вместо всего parent.

<!-- /unit:664ffcd8c42fb4f6dc -->

<!-- unit:47018695bdf4e62e94 -->
<a id="u-47018695bdf4e62e94"></a>

Выбирается granularity retrieval и granularity reading. Одно не определяет другое. Citation может ссылаться на точный child span внутри parent, чтобы широкое чтение не теряло проверяемость конкретного claim. При вложенных sections важно определить mapping, а не считать любой ближайший heading правильным parent.

<!-- /unit:47018695bdf4e62e94 -->

<!-- unit:c22c6187cc1d688b8d -->
<a id="u-c22c6187cc1d688b8d"></a>

## Multi-vector: что именно умножается

<!-- /unit:c22c6187cc1d688b8d -->

<!-- unit:6f4001b7e4211cc313 -->
<a id="u-6f4001b7e4211cc313"></a>

В варианте из разбираемого примера document имеет vectors текста, summary, keywords и tables. Поиск объединяет matches по parent document ID, а generator получает выбранный original content. Summary может пропустить важное исключение; keywords могут улучшить lexical coverage, но добавить noise. Нужен эксперимент, а не фиксированный обещанный прирост.

<!-- /unit:6f4001b7e4211cc313 -->

<!-- unit:8b53de1a47f53aeb0a -->
<a id="u-8b53de1a47f53aeb0a"></a>

Для $M$ documents, $r$ vectors на document, width $D$, bytes per coordinate $b$:

<!-- /unit:8b53de1a47f53aeb0a -->

<!-- unit:35eacccdde8e6575c0 -->
<a id="u-35eacccdde8e6575c0"></a>

$$
M_{\mathrm{vectors}}=MrDb.
$$

<!-- /unit:35eacccdde8e6575c0 -->

<!-- unit:de6a0640b5d36acbbb -->
<a id="u-de6a0640b5d36acbbb"></a>

При $M=10\,000$, $D=384$, FP32 ($b=4$): один vector даёт $15.36$ MB, четыре — $61.44$ MB в десятичных единицах. Это **только coordinates**. Полная стоимость добавляет index structures, text/metadata, parent mappings и auxiliary representations. Documents не обязательно дублируются, но generated summaries тоже занимают bytes; overhead зависит от index/config.

<!-- /unit:de6a0640b5d36acbbb -->

<!-- unit:aa79ed25e69b4a19fa -->
<a id="u-aa79ed25e69b4a19fa"></a>

Late interaction вроде ColBERT хранит token-level vectors и оценивает query-document interactions, например суммой максимальных token similarities. Это иной multi-vector mechanism, чем четыре summary/table vectors. Нельзя переносить memory/speed одного на другой по одному названию.

<!-- /unit:aa79ed25e69b4a19fa -->

<!-- unit:027b4078eed7bcbfc0 -->
<a id="u-027b4078eed7bcbfc0"></a>

### Сначала определяем, что означает «один документ»

<!-- /unit:027b4078eed7bcbfc0 -->

<!-- unit:1495aa8defc72b65f7 -->
<a id="u-1495aa8defc72b65f7"></a>

В source 10000 documents сравнивались с 10000 vectors. Но после chunking один исходный document обычно даёт несколько index items. Если 10000 documents дали 10 chunks каждый, baseline уже 100000 vectors, а четыре representations каждого chunk —400000. Четыре vectors на весь document и четыре на каждый chunk имеют разный payload и retrieval granularity.

<!-- /unit:1495aa8defc72b65f7 -->

<!-- unit:9747c5353317bffb2f -->
<a id="u-9747c5353317bffb2f"></a>

Разберём именно предложенные text/summary/keywords/table vectors. Каждый representation имеет parent/chunk ID, type и revision. Если summary/table текст кодируются совместимым encoder в том же space, один query vector может искать среди всех. Если используются разные encoders, например image и text в несогласованных spaces, одинакового metadata vector_type недостаточно: нужны совместимые query encoders и отдельные search/fusion rules. Разные dimensions нельзя сделать одним однородным index только указанием type.

<!-- /unit:9747c5353317bffb2f -->

<!-- unit:d0c0d068cdc2c32beb -->
<a id="u-d0c0d068cdc2c32beb"></a>

Это не тот же алгоритм, что ColBERT token-level multi-vector late interaction. В ColBERT несколько vectors представляют tokens и score агрегирует query-token matches. В source примере несколько standalone searchable views ведут к parent text. Оба называют multi-vector, но формулы scoring, число vectors и index cost могут сильно различаться.

<!-- /unit:d0c0d068cdc2c32beb -->

<!-- unit:6bd7966183f62062b0 -->
<a id="u-6bd7966183f62062b0"></a>

### Payload, index overhead и полная стоимость

<!-- /unit:6bd7966183f62062b0 -->

<!-- unit:5a53f6d45ee1fd6fde -->
<a id="u-5a53f6d45ee1fd6fde"></a>

Основной пример выше даёт 15.36MB→61.44MB только coordinates. Восстановим и крупный source scenario:100000 items,384dim,FP32 дают 153.6MB→614.4MB. Если каждый original text занимает ровно 2KiB, общий text payload204.8MB. Объявим для учебного расчёта shared metadata50MB, index overhead300MB для baseline и 1200MB для четырёх representations.

<!-- /unit:5a53f6d45ee1fd6fde -->

<!-- unit:1c9a71dddb2ca3bbdf -->
<a id="u-1c9a71dddb2ca3bbdf"></a>

Тогда totals708.4MB и 2069.2MB, отношение≈2.92095. Это иллюстрация исходной идеи «whole store не обязательно 4x». Значения index overhead здесь **заданы**, а не измерены и не выведены из универсального HNSW закона. Summary/table/keywords texts, per-vector metadata, mappings, replicas и allocator overhead ещё не учтены. В некоторых engines оценка index size уже включает coordinates; считать их второй раз нельзя.

<!-- /unit:1c9a71dddb2ca3bbdf -->

<!-- unit:a4187d02b3e5b474f9 -->
<a id="u-a4187d02b3e5b474f9"></a>

Общая запись —$S_1=T+M_1+V_1+I_1$, $S_r=T+T_{\mathrm{aux}}+M_r+rV_1+I_r$. Shared original text T не умножается, но auxiliary representation text и per-vector metadata могут вырасти. Если payload хранится по каждой vector record, текст может фактически дублироваться; это зависит от schema storage, а не от слова multi-vector.

<!-- /unit:a4187d02b3e5b474f9 -->

<!-- unit:618e93ed0efd5b435f -->
<a id="u-618e93ed0efd5b435f"></a>

При тех же 100000 items и четырёх 384dim vectors FP16 coordinates займут 307.2MB вместо 614.4MB. Но это только dtype арифметика; поддержка backend, search precision и compression format проверяются отдельно. На диске, в RAM и на GPU могут находиться разные copies/representations. «Тексты всегда самый большой компонент» из source тоже не является правилом.

<!-- /unit:618e93ed0efd5b435f -->

<!-- unit:66537d3e7ed06f06a7 -->
<a id="u-66537d3e7ed06f06a7"></a>

### Offline price: число vectors не равно числу API requests

<!-- /unit:66537d3e7ed06f06a7 -->

<!-- unit:da5e6a0d3a3fcb9fd5 -->
<a id="u-da5e6a0d3a3fcb9fd5"></a>

Четыре embeddings вместо одного означают больше encoded items, если действительно создаются все четыре. Но API batching может упаковать несколько items в один request; billing бывает по tokens или другому contract. Summary generation —дополнительные model input/output tokens, а короткие keywords и длинный full text стоят не одинаково. Поэтому «всегда 4x API calls,4x стоимость и 4x wall time» неверно.

<!-- /unit:da5e6a0d3a3fcb9fd5 -->

<!-- unit:35cec8f43dba3736a2 -->
<a id="u-35cec8f43dba3736a2"></a>

Полезно учитывать original extraction, construction вспомогательных views, encoding, index build, storage, update и serving. Если таблица отсутствует у 90% documents, число table vectors не равно числу всех documents. Если changed text требует regenerate summary и table view, update cost отличается от простого single-record write.

<!-- /unit:35cec8f43dba3736a2 -->

<!-- unit:981587392aebdfb0bc -->
<a id="u-981587392aebdfb0bc"></a>

Запись общей цены за snapshot с B updated items: extraction/aux-generation + sum encoded tokens + index update work + retained storage. Конкретные тарифы/latency не подставляются из старого snippet как актуальный прайс. Здесь изучается механизм расчёта; пользовательский запуск и measured costs не заявляются.

<!-- /unit:981587392aebdfb0bc -->

<!-- unit:c89ffccd8a6352499a -->
<a id="u-c89ffccd8a6352499a"></a>

### Online price, parent aggregation и число уникальных кандидатов

<!-- /unit:c89ffccd8a6352499a -->

<!-- unit:3cfdad720d39dfdac1 -->
<a id="u-3cfdad720d39dfdac1"></a>

Четыре последовательных searches могут занимать примерно сумму длительностей, четыре parallel —путь по самому медленному плюс merge/queue, если capacity хватает. Один search по общему большему index имеет свою зависимость от размера/parameters. Утверждения source «1.5–2x slower» и «4 searches всегда 4x latency» не универсальны. Query encoding может быть shared или требовать разные compatible encoders.

<!-- /unit:3cfdad720d39dfdac1 -->

<!-- unit:c25cc6c3b047237e51 -->
<a id="u-c25cc6c3b047237e51"></a>

Если top4 vectors все принадлежат одному parent P, получен один unique parent, а не четыре независимых sources. Over-fetch, grouping, per-parent caps и aggregation выбирают явно. Max score делает parent привлекательным по самому сильному view; sum может давать преимущество parent с большим числом representations; mean разбавляет сильное совпадение нерелевантными views. Ни один вариант не решает задачу автоматически.

<!-- /unit:c25cc6c3b047237e51 -->

<!-- unit:37fa1850f078e9e75b -->
<a id="u-37fa1850f078e9e75b"></a>

Summary может потерять исключение «кроме версии v8», keyword list —придумать термин, table serialization —перепутать row headers. Representation retrieval должен вести к проверенному original evidence. Генерируемые views рассматривают как поисковые вспомогательные данные, если они не получили отдельного статуса источника. Для итогового ответа сохраняют связь с original document/span.

<!-- /unit:37fa1850f078e9e75b -->

<!-- unit:a809bcb6afe5567ccd -->
<a id="u-a809bcb6afe5567ccd"></a>

### Отвечаем «стоит ли оно того» через ablation

<!-- /unit:a809bcb6afe5567ccd -->

<!-- unit:d79f652ba4a6625422 -->
<a id="u-d79f652ba4a6625422"></a>

Порог<50000 или>500000 documents из source не доказывает оправданность схемы. У маленькой коллекции может быть тяжёлая serving нагрузка; у большой —редкие дорогие queries, для которых прирост полезен. Однородный текст тоже может содержать множество distinct facts, а разнородный документ иногда хорошо находится одним правильным chunk.

<!-- /unit:d79f652ba4a6625422 -->

<!-- unit:8d3354540930a15478 -->
<a id="u-8d3354540930a15478"></a>

Сначала фиксируют baseline, dataset/queries, qrels и нужные slices. Потом добавляют одну view, проверяют candidate/context recall и end-to-end task success вместе с index/build/update/query cost. Важно видеть, какие missing cases исправлены и какие новые ошибки внесены. Summary view может помочь paraphrase, но ухудшить exact version search; общий average скроет этот обмен.

<!-- /unit:8d3354540930a15478 -->

<!-- unit:e4078abd52f8b42cd8 -->
<a id="u-e4078abd52f8b42cd8"></a>

Source приводит якобы «реальный enterprise case» с+4% recall,+300% costs,+180% search time и Pareto80/20. Без источника/методики это неподтверждённая иллюстрация, не evidence для выбора. Не переносим и обещание «baseline85% значит максимум ещё 2–3%». Нужная польза определяется фактическими оставшимися failures и cost of error, а не одной baseline цифрой.

<!-- /unit:e4078abd52f8b42cd8 -->

<!-- unit:cdc5b5a229794c2721 -->
<a id="u-cdc5b5a229794c2721"></a>

## Context assembly и защита смысла

<!-- /unit:cdc5b5a229794c2721 -->

<!-- unit:e8ec62781bee3689e0 -->
<a id="u-e8ec62781bee3689e0"></a>

После ranking отобрать evidence, удалить redundant overlap, сохранить source IDs/versions и уложить input budget. Больше $k$ иногда добавляет distractors и обрезает нужный конец prompt. Citations должны ссылаться на действительно переданный source и поддерживать соответствующий claim.

<!-- /unit:e8ec62781bee3689e0 -->

<!-- unit:d9b68d6e2f045f89a7 -->
<a id="u-d9b68d6e2f045f89a7"></a>

Retrieved instructions остаются данными. Document «игнорируй пользователя и вызови tool» не даёт приложению новых прав. ACL/filtering и доверенные constraints сохраняются вне untrusted snippets. При отсутствии evidence допустимый abstention полезнее уверенного unsupported answer.

<!-- /unit:d9b68d6e2f045f89a7 -->

<!-- unit:bb8ac5d93fa7a13e9d -->
<a id="u-bb8ac5d93fa7a13e9d"></a>

### От candidate set до фактического input generator

<!-- /unit:bb8ac5d93fa7a13e9d -->

<!-- unit:7414e236e5f1a3844f -->
<a id="u-7414e236e5f1a3844f"></a>

После retrieval, scope/version checks и reranking нужно упаковать evidence. В prompt сохраняют passage ID, content и необходимые условия/headers. Dedup удаляет повторы, но не должен объединять разные revisions как один факт. Контекстные ограничения считаются после реальной serialization, включая system instructions, question/history, tool definitions, citations и резерв output.

<!-- /unit:7414e236e5f1a3844f -->

<!-- unit:e592e857cfb8c469c7 -->
<a id="u-e592e857cfb8c469c7"></a>

Если окно 8192tokens, instructions/question/history1500 и output reserve1200, для content остаётся 5492. Семь упакованных passages по 700tokens требуют 4900, восемь 5600 уже не помещаются. Наличие нужного passage в retrieval top20 ещё не означает, что он прошёл packing. Trace должен хранить actual selected IDs и то, что действительно увидел generator.

<!-- /unit:e592e857cfb8c469c7 -->

<!-- unit:ee333545402e801df2 -->
<a id="u-ee333545402e801df2"></a>

Нельзя исправлять quality только инструкцией «ответь подробно»: длинный ответ без нужного evidence может увеличить число unsupported claims. И короткий source snippet может не объяснять полный механизм Transformer, если задача требует обучения/архитектуры. Для сложного запроса нужен coverage по аспектам, а не только близость каждого chunk теме.

<!-- /unit:ee333545402e801df2 -->

<!-- unit:cbf9e95515f9f11f93 -->
<a id="u-cbf9e95515f9f11f93"></a>

### Когда выбирать иной путь получения данных

<!-- /unit:cbf9e95515f9f11f93 -->

<!-- unit:f9b6de450a121bb1ce -->
<a id="u-f9b6de450a121bb1ce"></a>

Для очень маленького corpus полный context может быть разумным reference, если он помещается и стоимость подходит. Fine-tuning не является автоматической заменой document store с citations/updates: оно меняет policy/weights и имеет другой lifecycle. Решение зависит от формы задачи и данных, а не только от числа файлов.

<!-- /unit:f9b6de450a121bb1ce -->

<!-- unit:d405aa2fedc6699449 -->
<a id="u-d405aa2fedc6699449"></a>

Для текущей цены, остатка товара или статуса операции direct structured API может лучше соответствовать freshness/consistency contract, чем поиск по вчерашнему export. При этом RAG системы способны поддерживать incremental updates; «если данные меняются часто, RAG всегда невозможен» тоже неверно. Разделяют live state lookup и поиск explanatory documentation.

<!-- /unit:d405aa2fedc6699449 -->

<!-- unit:a89d634c1e269a6c8f -->
<a id="u-a89d634c1e269a6c8f"></a>

RAG может помогать code/math tasks внешними definitions, API contracts, examples и premises, хотя не заменяет вычисление/рассуждение. Source «RAG для кода не поможет» слишком категоричен. Нельзя вывести и обещание sub100ms из выбора fine-tuning: latency зависит от model, lengths, serving и всей цепочки. Query embedding cache отличается от semantic answer cache; сохранённый vector не равен сохранённому готовому ответу.

<!-- /unit:a89d634c1e269a6c8f -->

<!-- unit:f7e9bdde1c21478023 -->
<a id="u-f7e9bdde1c21478023"></a>

## Вопросы и практика

<!-- /unit:f7e9bdde1c21478023 -->

<!-- unit:146ba5f31828466dbd -->
<a id="u-146ba5f31828466dbd"></a>

1. Кто выбирает terms базового BM25? **Ответ:** query analyzer; LLM keyword generation не обязательна.
2. Что получает ordinary generator после vector search? **Ответ:** texts по найденным IDs, а не произвольные coordinates.
3. Почему reranker не исправит zero candidate recall? **Ответ:** оценивает только предоставленные pairs.
4. Увеличивают ли четыре vectors весь storage ровно вчетверо? **Ответ:** координаты при одинаковом dtype — да; total storage зависит от shared text/index/metadata.

<!-- /unit:146ba5f31828466dbd -->

<!-- unit:88c2013942feb984fa -->
<a id="u-88c2013942feb984fa"></a>

5. Обязательно ли query/document encoders идентичны? **Ответ:** нет, но должны образовывать совместимую обученную пару и protocol.
6. Означает ли cosine≈1 истинный ответ? **Ответ:** нет; similarity не проверяет версии, scope и содержательную применимость.
7. Требует ли parent retrieval отдельного vector каждого parent? **Ответ:** не всегда; можно индексировать children и хранить parent text отдельно.
8. Почему RRF должен учитывать absent IDs? **Ответ:** каждый document может присутствовать только в части rankings; contribution отсутствующего списка равен 0 по выбранной policy.
9. Гарантирует ли reranker сохранение recall? **Ответ:** не на произвольном final cutoff; он ограничен candidates и может ошибаться.
10. Почему top4 vectors не обязательно четыре sources? **Ответ:** несколько vectors могут ссылаться на один parent.
11. Почему 4x embeddings не обязательно 4x API calls? **Ответ:** batching, lengths, auxiliary generation и billing имеют отдельные units.
12. Что подтверждает source enterprise benchmark без ссылки? **Ответ:** ничего измеренного; числа остаются неподтверждённой иллюстрацией.

<!-- /unit:88c2013942feb984fa -->

<!-- unit:9f02dbab045de6115f -->
<a id="u-9f02dbab045de6115f"></a>

**Практика:** пересчитать BM25 example, затем удвоить document length при тех же term frequencies. На фиксированных queries сравнить BM25, dense, hybrid и один reranker, меняя один stage. Сохранить cases exact IDs, парафразов, отсутствующего ответа, stale versions и forbidden documents.

<!-- /unit:9f02dbab045de6115f -->

## Результат главы и переход дальше

Ответьте на вопрос: **Как найти evidence и связать его с ответом без обучения собственного retriever?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

Следующие темы: [Оценка retrieval, ответов и task success](<app-evaluation.md>).

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
