# Ж. Когда task меняется вместе с языком, длиной и модальностью

[Оглавление](README.md). Группа G, модули **30–34**. **Происхождение:** углубление нового BERT-плана; общие long-context вопросы из S5 здесь применены к encoder tasks, остальные специализированные ветки — дополнение.

## 1. Русский и multilingual transfer

Morphology, casing, ё/е, омографы, transliteration и code switching меняют tokenization и entity boundaries. Token inflation — отношение token counts одного tokenizer к другому на одних texts — помогает оценить context/cost, но не является direct quality metric. Для 120 против 80 tokens ratio 1.5; это не «на 50% хуже semantics».

Сравнивают task-appropriate Russian checkpoint, mBERT/XLM-R и modern multilingual candidate при same ontology/data. [RuBERT model card](https://huggingface.co/ai-forever/ruBert-base) описывает конкретный checkpoint, но не гарантирует готовый NER/retriever head. Monolingual capacity и multilingual transfer имеют разные trade-offs; число языков pretraining не равно качеству каждого.

Cross-lingual retrieval отдельно задаёт language query и document. English-only fine-tuning с test на русском, translated supervision и multilingual task training — разные protocols. Translation-equivalent texts в разных splits дают leakage даже при разных surface forms. Перевод negation, имен и gender может менять labels; human audit нужен на relevant slices.

Report показывает each-language quality, code-switch, rare entities, calibration и fragmentation. Средний multilingual score способен скрыть плохое качество нужного языка. Если выбирать checkpoint по target Russian validation, это task adaptation selection, а не truly untouched zero-shot benchmark.


<!-- source-depth-bert-Ж -->
### 1.1. Язык влияет на вход, supervision и оценку разными путями

Русский текст может распадаться на большее число subwords из-за словаря, морфологии или редких имён. Это увеличивает длину и расходует context budget. Но модель с более дробной токенизацией всё ещё может иметь полезные representations; число pieces не является самостоятельной метрикой понимания. Если один tokenizer даёт 120 tokens, другой 80, ratio=1.5 говорит о длине именно этих входов, а не о падении качества на 50%.

Если одинаковая по содержанию версия текста занимает 400 tokens на одном языке и 600 на другом, при доступных 510 content positions вторая версия будет обрезана. Плохое качество может быть связано с потерянным evidence, а не только с multilingual representation. Поэтому отдельно измеряют fragmentation, долю truncation и качество на текстах, которые обе модели видели целиком.

Нормализация меняет смысл в зависимости от задачи. Замена ё/е может быть допустима для темы письма, но важна для точного имени. Внешне похожие латинские и кириллические символы способны означать разные IDs в коде товара. Транслитерация может облегчить поиск alias и одновременно испортить точный span. Правило выбирается по target, а не как универсальная «очистка русского текста».

### 1.2. Multilingual average зависит от состава evaluation

Пусть в тесте 900 English examples с accuracy 0.95 и 100 Russian с accuracy 0.60. Средняя по всем примерам accuracy равна 0.915. Среднее двух language accuracies — 0.775. Это два разных знаменателя, и первое число скрывает слабое качество на русском. Речь здесь об accuracy, а не о macro-F1 по классам.

Для cross-lingual retrieval нужны пары направлений: например, русский query → English documents и наоборот. Успешный поиск внутри одного языка не доказывает alignment разных языков. Переведённые тексты должны сохранять labels, имена, отрицания и числовые условия; дубликаты переводов группируют до split. Смена языка не создаёт новый независимый источник, если содержание осталось тем же.


## 2. Длинный document не равен набору одинаково размеченных chunks

Truncation проста, но удаляет evidence. Sliding windows сохраняют coverage за extra compute; hierarchical encoder сначала кодирует chunks, затем агрегирует их vectors. Если document label «есть нарушение» определяется двумя из восьми chunks, присвоение positive всем восьми создаёт weak/noisy labels.

Mean aggregation размывает редкий signal, max чувствителен к одному false alarm, learned attention требует supervision и может использовать shortcuts. Для chunk scores $(0,0,0,0,0,0,1,1)$ mean 0.25, max 1; значение label не определяется одной formula без task assumptions. Для evidence extraction важно вернуть source offsets, а не только pooled decision.

Sparse/global attention, например [Longformer](https://arxiv.org/abs/2004.05150), меняет interaction pattern, позволяя selected global tokens и local windows. Для window width $w$ local attention work порядка $Lw$ плюс global interactions, в отличие от full $L^2$. Но ограниченная connectivity требует layers/global routes для distant dependencies. ModernBERT сочетает local/global blocks по своей architecture.

Learned positions классического BERT имеют training length/config. Изменить число в config без обучения новых rows и проверки checkpoint недостаточно. RoPE/native context тоже не гарантирует retrieval distant evidence. Сравнение truncation/hierarchy/native-long-context использует full documents, length/evidence-position slices, same task и total cost.

Ответ на границе windows может быть разделён так, что ни одно окно его не видит целиком. Overlap решает это только для spans ограниченной длины при нужном chunk policy; document-level multi-hop evidence может быть разнесено гораздо дальше. Нужно проверять actual information coverage.


### 2.1. Три способа чтения длинного документа решают разные задачи

Truncation передаёт только выбранный кусок; утерянные tokens недоступны всему последующему stack. Sliding windows передают несколько перекрывающихся кусков и объединяют результаты. Hierarchical model сначала строит chunk representations, затем обрабатывает их последовательность отдельным агрегатором. В последнем случае взаимодействие между исходными tokens разных chunks проходит через bottleneck их chunk vectors.

Если document label означает «хотя бы один фрагмент содержит нужное событие», присвоение positive всем chunks создаёт ложные token/chunk targets. Можно обучать document-level aggregation, не выдавая неизвестные chunk labels за разметку. Но тогда важны shortcuts: модель может обнаружить тип источника вместо события. Для доказуемого evidence нужны соответствующие labels или отдельная проверка извлечённых фрагментов.

Mean в исходном примере даёт 0.25, max — 1. Ни одно из этих чисел нельзя автоматически назвать вероятностью document label. Если chunk probabilities действительно относятся к независимым событиям «в этом chunk есть evidence», можно получить noisy-OR $1-\prod_k(1-p_k)$. Для двух независимых p=0.1 это 0.19. Если оба overlapping chunks описывают одно и то же событие, независимость нарушена: повторение одного evidence не должно само по себе повышать уверенность до 0.19.

### 2.2. Когда overlap гарантирует, что span не будет разрезан

Пусть каждое окно содержит C content tokens, шаг между началами S, overlap O=C−S. Indices целочисленные, первое окно начинается с 0, хвост документа тоже покрыт. Для гарантии покрытия любого непрерывного span длины не больше A достаточно $O\ge A-1$.

Почему? Возьмём последнее начало окна kS, не превосходящее начало ответа a. Тогда $a-kS\le S-1$. Последняя позиция ответа не дальше $a+A-1\le kS+S+A-2$. Она помещается в окно до $kS+C-1$, если $S+A-1\le C$, то есть O≥A−1. Это вывод про покрытие **одним** окном, а не утверждение, что модель правильно извлечёт ответ.

При C=8, O=2, S=6 любой span длины до 3 покрыт. Для длины 4 уже возможна ошибка: span [5,9) не входит ни в первое окно [0,8), ни во второе [6,14). Если часть model budget занята вопросом и special tokens, C нужно считать после их вычитания. Нерегулярные границы chunks и удаление отдельных фрагментов требуют нового анализа покрытия.

Multi-hop evidence может быть разнесено между несколькими далёкими разделами. Гарантия для короткого span не делает такое evidence доступным совместно. Для него нужны другая aggregation, retrieval нескольких частей или long-context architecture, и отдельные тесты на расположение evidence.

![Local и global связи показывают доступность контекста по слоям, а не фиксированную силу внимания.](../../оформление/transformer/encoder-local-global.svg)

Local и global связи показывают доступность контекста по слоям, а не фиксированную силу внимания.

### 2.3. Local и global attention: доступность по слоям

У local attention с радиусом r одна позиция читает максимум 2r+1 соседей в одном слое. После K таких слоёв влияние может распространиться на расстояние Kr, если нет дополнительных связей. Например, r=2, K=3 дают максимум 13 позиций в полном внутреннем receptive field. Norm и MLP сами не добавляют межпозиционные рёбра.

Global position добавляет дальние маршруты. В упрощённой схеме global query читает все позиции, а остальные queries могут читать global key. Информация a→global→b проходит два attention слоя: выходы одного слоя вычисляются из состояний предыдущего, поэтому b не читает уже обновлённый global output того же слоя. Если task head читает сам global token, для сбора информации ему может хватить одного такого слоя. Механизм local/global attention — основа [Longformer](https://arxiv.org/abs/2004.05150).

При длине L, local width w и g global positions число разрешённых пар имеет порядок Lw+Lg вместо L² при фиксированных w,g. Для L=1000,w=33,g=2 верхняя грубая сумма local и двусторонних global связей — 33000+4000=37000 против миллиона dense pairs; пересечения и границы могут уменьшить это число. Стоимость проекций и MLP при этом не исчезла. Важны и граф связей, и поддержка эффективного kernel.


## 3. Domain encoders: наука, специальные тексты и код

[SciBERT](https://aclanthology.org/D19-1371/) исследует scientific text representations, [CodeBERT](https://arxiv.org/abs/2002.08155) — natural/programming language. Domain pretraining меняет data/terminology/objective; название не доказывает correctness любой medical/legal/financial application.

Сначала определяется task: entity/relation extraction, classification, matching или search. Domain shift features отличается от изменения annotation policy: прежний label «положительный» может иметь другую definition. Generic task-trained checkpoint может превзойти domain MLM, которому ещё не обучили task geometry/head.

Сравнение generic task model, domain checkpoint и DAPT проводится с comparable tuning budget и general-domain regression. Кодовые tasks split по repositories/time/clone families: почти одинаковые функции в train/test делают score завышенным. Lexical token similarity не доказывает functional equivalence; нужен task oracle или execution tests, если именно equivalence является целью.


### 3.1. Что именно меняется при переходе в домен

В специализированных текстах могут измениться термины, структура предложений, частоты классов и само правило разметки. Если новый источник пишет «отклонено» там, где прежний писал «не подтверждено», это lexical/domain shift. Если новая инструкция иначе определяет positive label, меняется supervision. Смешение этих причин мешает выбрать полезное вмешательство.

Domain MLM encoder ещё не обязан иметь sentence retrieval geometry или NER head. Он учился выбранному pretraining objective, а прикладная задача требует своей supervision. Поэтому сравнивают не только названия checkpoints, но и полный путь domain/generic encoder → одинаковый task training → независимый test. DAPT полезно оценивать вместе с regression на прежних источниках, а не только по MLM loss нового корпуса.

В коде почти одинаковые функции могут различаться идентификаторами и форматированием, а близкий по tokens код — поведением на граничном input. Для semantic search такие near-clones могут быть полезны; для доказательства эквивалентности нужны другие oracle. Split по repository, времени и clone family предотвращает ситуацию, когда тест измеряет узнавание почти той же функции. Случайное разбиение строк кода особенно легко нарушает независимость.


## 4. Aspect sentiment, linking и structured understanding

«Экран отличный, батарея слабая» содержит два аспекта с противоположной polarity. Один overall sentiment label теряет структуру. Можно извлекать aspect spans и классифицировать polarity в их context либо jointly предсказывать pairs. Implicit aspects требуют separate policy: отсутствие explicit span не всегда означает отсутствие opinion.

Entity linking: mention detection → candidate retrieval → context disambiguation → entity ID или NIL. «Мир» может быть организацией, картой/брендом или обычным словом; surface form не определяет ID. Candidate recall ограничивает disambiguator, а ошибки mention boundaries добавляют upstream losses.

Оценка gold mentions показывает conditional quality, predicted mentions — всей системы. Нужны seen/unseen entities, alias shift и NIL cases. Entity typing, relation classification и intent+slots имеют свои schemas и metrics. Returned artifact сохраняет offsets/IDs/evidence; general sentiment classifier не заменяет aspect-conditioned extraction.


### 4.1. Один текст может требовать нескольких структурированных ответов

Для «Экран отличный, батарея слабая» два ответа — (экран, positive) и (батарея, negative). Overall sentiment может быть mixed или зависеть от guideline, но он теряет привязку оценок к аспектам. Если aspect spans уже даны, classifier решает условную polarity задачу; если их надо найти, добавляется extraction. Нельзя сравнивать эти режимы как одну и ту же сложность.

Entity linking тоже отделяет поиск от решения. Пусть mention detector нашёл правильный span «Мир», а candidate stage вернул только финансовый бренд и организацию, хотя в данном тексте подразумевался иной entity. Disambiguator не сможет выбрать отсутствующий ID. NIL означает, что подходящего entity нет в допустимой knowledge base, а не просто низкую уверенность любой головы.

Gold-mention evaluation исключает ошибки детектора. End-to-end учитывает и boundaries, и retrieval candidates, и выбранный ID. Если выдавать только последнее число без условий, можно принять улучшение mention detector за улучшение disambiguation. Схема ответа хранит original offsets, ID/version knowledge base и evidence; одинаковая surface string не заменяет эту связь.


## 5. OCR, layout и таблицы

Два бланка могут содержать один bag OCR words «Имя Анна Город Москва», но разное spatial association fields. Plain serialized text может потерять соответствие key/value. [LayoutLM](https://arxiv.org/abs/1912.13318) соединяет text и layout signals; новые variants могут добавлять visual components по своей architecture. Text-only BERT не получает изображение лишь потому, что tokens пришли из OCR.

Для table важны row/column relations, cell coordinates, headers и numerical operations. Serialization — baseline с определённым order, а не universal preservation структуры. Если задача требует сравнить суммы, encoder label head не гарантирует точную арифметику без data/objective/oracle.

Разделяют OCR recognition error, layout association, span extraction и final field correctness. Пропущенное OCR слово downstream head не обязан восстановить из пустоты. Сравнение OCR+text и layout-aware pipeline должно использовать same documents и учитывать цену preprocessing/annotations.


![Текст и пространственные признаки должны сохранять соответствие одному исходному слову; схема fusion зависит от модели.](../../оформление/transformer/layout-text-boxes.svg)

Текст и пространственные признаки должны сохранять соответствие одному исходному слову; схема fusion зависит от модели.

### 5.1. Что даёт двумерная позиция сверх строки OCR

OCR может вернуть слова и bounding boxes. Если preprocessing оставил только bag of words, два бланка с теми же словами способны потерять соответствие «поле — значение». Последовательный порядок чтения сохраняет часть структуры, но не гарантирует сохранения колонок, таблиц и пространственных групп. Layout-aware model получает дополнительные признаки расположения, а конкретная fusion зависит от архитектуры. Общая идея показана в [LayoutLM](https://arxiv.org/abs/1912.13318).

Для учебного bbox в пикселях (100,400,500,600) на странице ширины 1000 и высоты 2000 переход к координатам 0–1000 по каждой оси даёт (100,200,500,300). Горизонталь и вертикаль масштабируются по своим размерам. Это объявленная схема нормировки, не универсальный формат всех models. Нужны согласованные начало координат, размеры страницы, page IDs и порядок x0≤x1,y0≤y1.

Если слово разбилось на несколько subwords, каждому нужно согласованное соответствие bbox/word ID. Изменение порядка OCR words без перестановки boxes нарушает пары признаков при правильных shapes. Для нескольких страниц один и тот же bbox не означает одно место: нужен контекст страницы или предусмотренная архитектурой обработка.

Таблица добавляет отношения rows/columns и headers. Последовательность «товар A 2 10 товар B 3 7» не сообщает сама по себе, где количество, где цена и какой итог требуется. Serialization должна явно сохранять выбранную структуру. Если задача требует точного умножения и суммирования, наличие contextual vectors не заменяет проверку вычисления или предназначенный для этого компонент.

### 5.2. Где искать ошибку документа

Если OCR распознал 100 как 700, правильное извлечение распознанного span ещё не даёт правильное поле исходного изображения. Если текст распознан верно, но значение привязано к соседнему header, ошибка в layout association. Если span и field верны, но postprocessing неверно интерпретировал единицы, ошибка уже после encoder. Раздельная оценка этих этапов показывает, что нужно исправлять.

Сравнивая text-only и layout-aware модель, фиксируют те же страницы, OCR policy и supervision. Дополнительные boxes, pixels и labels — дополнительная информация и стоимость, а не бесплатное преимущество другого encoder. Итоговая метрика оценивает нужные поля на исходных документах, а не только token-level accuracy OCR-последовательности.


## 6. Контроль и практика модулей 30–34

1. Доказывает ли language coverage качественный Russian NER? **Ответ:** нет; нужен task/language evaluation и alignment.
2. Почему positive document не даёт positive label каждому chunk? **Ответ:** evidence может находиться только в части document.
3. Чем gold-mention linking отличается от end-to-end? **Ответ:** gold исключает ошибки mention detector и потому измеряет условную задачу.
4. Чего нет у text-only OCR sequence? **Ответ:** она может потерять 2D structure и visual evidence, если их явно не передали.

5. Какой overlap гарантирует покрытие span длины до A при регулярных окнах? **Ответ:** O≥A−1 при объявленных целочисленных границах и покрытом хвосте; это не гарантия правильного QA и не покрытие multi-hop evidence.
6. Почему local→global→local обычно требует двух слоёв? **Ответ:** queries одного слоя читают states предыдущего слоя, а не уже обновлённый global output того же слоя.
7. Равен ли больший token count худшему пониманию? **Ответ:** нет. Он меняет length/cost и вероятность truncation; качество проверяется по task labels.
8. Почему bbox нужно нормировать отдельно по ширине и высоте? **Ответ:** это разные оси страницы; деление обеих на одну длину исказит геометрию прямоугольной страницы.

**Практика:** измерьте token counts одной multilingual фразы, разметьте aspect spans и NIL linking case. Для восьми chunks выберите aggregation и test evidence positions. Нарисуйте две формы с одинаковыми словами, но разными layouts; определите недостающие признаки. Обоснуйте shortlist для русского NER и multilingual long retrieval без выбора по одному leaderboard.
