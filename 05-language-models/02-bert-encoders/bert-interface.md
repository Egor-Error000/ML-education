# Encoder, tokenizer, attention и pooling

**Уровень:** основа ветки.

**Предпосылки:** [BERT в общем ядре: encoder, MLM и задачи](<../01-foundations/lm-bert.md>), [Split, cross-validation и утечки](<../../01-data-features/01-reliability/df-split.md>), [Текст, токенизация и разбиение документов](<../../01-data-features/03-modalities/df-text.md>)

**Используют:** [MLM, альтернативные objectives и семейства encoder](<bert-objectives.md>), [Данные, task heads, fine-tuning и DAPT](<bert-adaptation.md>), [Classification, NER, spans и extractive QA](<bert-tasks.md>), [Sentence embeddings и contrastive learning](<bert-embeddings.md>), [Dense retrieval, cross-encoder, SPLADE и ColBERT](<bert-retrieval.md>), [Языки, длинные документы, домены и layout](<bert-domains.md>), [Метрики, калибровка, OOD и интерпретация](<bert-evaluation.md>), [Сжатие, сервис encoder и жизненный цикл индекса](<bert-operations.md>)

**Учебный вопрос:** Как получить состояния и корректно агрегировать их в представление текста?

Вводная глава BERT уже определила encoder, MLM и task head. Здесь подробно разбираются offsets, формы входов и pooling: они нужны для точного согласования текста с предсказаниями, независимо от генеративной адаптации.



## Карта вычисления и границы этапов

![Encoder, tokenizer, attention и pooling: входы, преобразования и выходы](<../../оформление/course/bert-interface.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Сравните формы и смысл каждого перехода с формулами главы. Выход преобразования и доказательство его качества — разные объекты: последняя проверка использует отдельный протокол оценки.

<!-- unit:d723e0c6e7f5c43995 -->
<a id="u-d723e0c6e7f5c43995"></a>

## Что возвращает encoder

<!-- /unit:d723e0c6e7f5c43995 -->

<!-- unit:ab40c478a53e48f689 -->
<a id="u-ab40c478a53e48f689"></a>

Encoder преобразует уже данный текст в contextual token states. Он может видеть разрешённые tokens с обеих сторон позиции. Decoder-only causal model ограничивает attention прошлым; encoder-decoder добавляет отдельное encoding входа и генерацию output. Architecture mask и training objective нужно рассматривать вместе.

<!-- /unit:ab40c478a53e48f689 -->

<!-- unit:ebfe903016b23d993d -->
<a id="u-ebfe903016b23d993d"></a>

Для batch size $B$, length $L$, hidden width $D$ encoder выдаёт $H\in\mathbb R^{B\times L\times D}$. Дальнейший смысл задаёт head: тема письма — logits $B\times C$; независимые moderation labels — тоже $B\times C$, но с другим loss; NER — $B\times L\times C$; extractive QA — два массива $B\times L$; retrieval — vectors $B\times d_e$. Если нужен новый произвольный ответный текст, обычный MLM BERT не становится chat generator от одного prompt.

<!-- /unit:ebfe903016b23d993d -->

<!-- unit:af025c3e627572648c -->
<a id="u-af025c3e627572648c"></a>

Сначала выбирается prediction unit и baseline: правила/TF-IDF, frozen encoder+head, затем tuning. Advantage encoder зависит от task/data/cost, а не от универсального превосходства над всеми generative systems.

<!-- /unit:af025c3e627572648c -->

<!-- unit:cd5fbeacfe07d2c5fd -->
<a id="u-cd5fbeacfe07d2c5fd"></a>

<!-- source-depth-bert-А -->

<!-- /unit:cd5fbeacfe07d2c5fd -->

<!-- unit:1558ba05f3d4db892d -->
<a id="u-1558ba05f3d4db892d"></a>

### Один текст даёт много состояний, а голова выбирает единицу ответа

<!-- /unit:1558ba05f3d4db892d -->

<!-- unit:60ef29756422d4a6f0 -->
<a id="u-60ef29756422d4a6f0"></a>

Пусть модель получает письмо «Доставка задержалась, прошу вернуть деньги». Токенизация создаёт несколько IDs; encoder выдаёт отдельный вектор каждой позиции. Он не обязан сразу выбрать один «главный вектор письма». Если задача — тема обращения, голова агрегирует последовательность и выдаёт распределение классов. Если задача — найти упоминания товара, нужны выходы по позициям или spans. Если задача — сравнить письмо с шаблоном ответа, понадобятся парная постановка либо специально обученные embeddings.

<!-- /unit:60ef29756422d4a6f0 -->

<!-- unit:10f04caf5deee5a409 -->
<a id="u-10f04caf5deee5a409"></a>

Форма B×L×D — интерфейс представления, а не набор готовых labels. Координата 17 не обязана означать «возврат», координата 18 — «доставка». Прикладная голова и функция потерь обучают связь представления с нужными ответами. Поэтому нельзя взять первые C координат hidden state и объявить их вероятностями C классов. У hidden state нет автоматической softmax-нормировки по прикладным меткам.

<!-- /unit:10f04caf5deee5a409 -->

<!-- unit:e2cb74dfee62ad7986 -->
<a id="u-e2cb74dfee62ad7986"></a>

Все примеры batch используют одни веса encoder, но разные входные IDs и контексты. Между примерами обычного batch attention нет: сообщение первого клиента не становится контекстом второго только потому, что они обрабатываются одновременно. Ось B — независимые последовательности; обмен идёт по оси L внутри каждого примера, если явно не введена иная архитектура.

<!-- /unit:e2cb74dfee62ad7986 -->

<!-- unit:424b4646f813324116 -->
<a id="u-424b4646f813324116"></a>

Bidirectional означает доступ к обеим сторонам **внутри переданного входа**. Encoder не видит будущее сообщение клиента, которого нет в массиве, и не восстанавливает автоматически обрезанное окончание письма. Это доступность наблюдаемых позиций, а не знание будущих событий.

<!-- /unit:424b4646f813324116 -->

<!-- unit:4f6f8c28c9a27e9748 -->
<a id="u-4f6f8c28c9a27e9748"></a>

## Tokenizer и связь с исходными characters

<!-- /unit:4f6f8c28c9a27e9748 -->

<!-- unit:0912c0eeb666a5d2a7 -->
<a id="u-0912c0eeb666a5d2a7"></a>

У классического BERT WordPiece vocabulary; cased/uncased checkpoints имеют разные normalization contracts. Другие encoders могут использовать BPE или SentencePiece. Совпадение vocabulary size не делает token IDs семантически совместимыми: row 100 одного checkpoint может обозначать другой token.

<!-- /unit:0912c0eeb666a5d2a7 -->

<!-- unit:62a39657003e9f7401 -->
<a id="u-62a39657003e9f7401"></a>

Для пары классическая layout-схема: CLS, tokens A, SEP, tokens B, SEP. Input IDs определяют lookup, attention mask отмечает valid positions, token-type IDs различают segments A/B. Наличие token-type embeddings не universal: config конкретной модели определяет fields. SEP не запрещает cross-segment attention.

<!-- /unit:62a39657003e9f7401 -->

<!-- unit:0d3c5786aa3081b92e -->
<a id="u-0d3c5786aa3081b92e"></a>

Рассмотрим условное, не реальное WordPiece-разбиение: «Анна не Иванова-Петрова» → «Анна», «не», «Иван», «##ова», «-», «Петр», «##ова». Для NER нужно знать word IDs и char offsets исходной фразы; continuation pieces не являются отдельными новыми entities. Normalization может изменить lowercase/ё/е/омографы и offsets mapping. Поэтому восстанавливать spans по длинам token strings ненадёжно: используются offsets tokenizer по его contract.

<!-- /unit:0d3c5786aa3081b92e -->

<!-- unit:4462b4dc4e570b8b26 -->
<a id="u-4462b4dc4e570b8b26"></a>

До training измеряются tokens-per-word, UNK/fragmentation, truncation rate и language slices. Truncation выбрасывает информацию; увеличение `max_length` не исправляет потерянный конец текста, если модель/positions не поддерживают его.

<!-- /unit:4462b4dc4e570b8b26 -->

<!-- unit:9e7b0558a093bdfe0a -->
<a id="u-9e7b0558a093bdfe0a"></a>

### ID, кусочек строки и span исходного текста — три разных объекта

<!-- /unit:9e7b0558a093bdfe0a -->

<!-- unit:59e27df73b3096f199 -->
<a id="u-59e27df73b3096f199"></a>

Возьмём исходную строку «Анна не Иванова-Петрова». В индексах Unicode-символов Python фамилия занимает полуоткрытый интервал [8,23): начальный символ входит, конечный индекс не входит. В условном разбиении части фамилии имеют offsets [8,12), [12,15), [15,16), [16,20), [20,23). Объединение непрерывного span возвращает исходную подстроку [8,23), включая дефис. Нельзя суммировать длины строк «##ова»: служебные решётки не являются символами исходной фамилии.

<!-- /unit:59e27df73b3096f199 -->

<!-- unit:cfb329b404926c9261 -->
<a id="u-cfb329b404926c9261"></a>

ID — номер строки embedding-таблицы. Строковое представление subword нужно для чтения токенизации; offset связывает токен с исходным текстом. Специальный CLS может иметь свой ID и скрытое состояние, но не соответствовать ни одному реальному символу документа. Поэтому special-token mask и offsets проверяются отдельно. Описания полей encoding приведены в [документации tokenizer](https://huggingface.co/docs/tokenizers/api/encoding).

<!-- /unit:cfb329b404926c9261 -->

<!-- unit:f8f3253e81449c9832 -->
<a id="u-f8f3253e81449c9832"></a>

Если задаче нужны word-level labels, несколько subwords одного слова требуют правила выравнивания. Можно начислять loss только первому subword и игнорировать остальные; можно размечать все части с согласованным BIO-продолжением. Эти варианты задают разные числа обучаемых позиций. Нельзя случайно приписать `B-PER` каждой части одной фамилии и считать, что boundaries не изменились. Для span extraction часто удобнее сначала получить точные границы по offsets, а затем оценивать сущности целиком.

<!-- /unit:f8f3253e81449c9832 -->

<!-- unit:f9b8d9a56fab8e7639 -->
<a id="u-f9b8d9a56fab8e7639"></a>

Нормализация тоже часть контракта. Если приложение изменило текст до tokenizer, offsets относятся к тому тексту, который tokenizer фактически получил. После удаления пробелов или преобразования Unicode нельзя применять старые индексы к новой строке без отображения между версиями. Особенно важно согласовать единицы индексов при передаче между системами: символы, байты и UTF-16 code units в общем случае не совпадают.

<!-- /unit:f9b8d9a56fab8e7639 -->

<!-- unit:0f352484d6527737ef -->
<a id="u-0f352484d6527737ef"></a>

### Два сегмента не превращаются в две независимые модели

<!-- /unit:0f352484d6527737ef -->

<!-- unit:497e16f1e7e8ace426 -->
<a id="u-497e16f1e7e8ace426"></a>

Для пары «вопрос — документ» классический формат `[CLS] вопрос [SEP] документ [SEP]` даёт одну общую последовательность. Type IDs помечают первую и вторую части; learned type embeddings прибавляются к признакам. Они не являются attention mask. При обычном bidirectional attention слово вопроса может получать информацию из документа и наоборот.

<!-- /unit:497e16f1e7e8ace426 -->

<!-- unit:04c05fa822d5452079 -->
<a id="u-04c05fa822d5452079"></a>

Это полезно для cross-encoder, который совместно оценивает конкретную пару, но меняет возможность кеширования. Представление документа внутри этой пары зависит от вопроса, поэтому нельзя один раз сохранить его как независимый document embedding и затем получить тот же результат для любого вопроса. Bi-encoder отдельно кодирует две стороны, а взаимодействие переносит в сравнение embeddings; подробное сравнение находится в главе [«Dense retrieval, cross-encoder, SPLADE и ColBERT»](<bert-retrieval.md>).

<!-- /unit:04c05fa822d5452079 -->

<!-- unit:ad1d097cc3cc836fc2 -->
<a id="u-ad1d097cc3cc836fc2"></a>

Ограничение длины относится ко всей собранной последовательности, включая специальные токены. При бюджете 12 позиций и классическом парном формате три позиции заняты CLS/SEP/SEP; на содержимое двух частей остаётся 9. Если вопрос занимает 4, документу остаётся 5. Политика обрезки должна задавать, какая сторона сохраняется и какой смысл теряется. Повторная токенизация с другим tokenizer может изменить этот бюджет даже при тех же исходных словах.

<!-- /unit:ad1d097cc3cc836fc2 -->

<!-- unit:d5023b07ebdf373135 -->
<a id="u-d5023b07ebdf373135"></a>

## Входы и один block: согласованные shapes

<!-- /unit:d5023b07ebdf373135 -->

<!-- unit:7dd4fb3a82dd2ce1e0 -->
<a id="u-7dd4fb3a82dd2ce1e0"></a>

Классическая embedding layer складывает token, learned position и token-type vectors:

<!-- /unit:7dd4fb3a82dd2ce1e0 -->

<!-- unit:d54a2f5ac0debc59fa -->
<a id="u-d54a2f5ac0debc59fa"></a>

$$
X_{b,t}=\operatorname{Dropout}\!\left(
\operatorname{LN}(E_{x_{b,t}}+P_t+S_{s_{b,t}})\right).
$$

<!-- /unit:d54a2f5ac0debc59fa -->

<!-- unit:537423d09864a7b18b -->
<a id="u-537423d09864a7b18b"></a>

Для row-oriented tokens проекции $W_Q,W_K,W_V\in\mathbb R^{D\times D}$ дают tensors $B\times L\times D$, после reshape — $B\times H_a\times L\times d_h$, где $D=H_a d_h$. Одна attention head:

<!-- /unit:537423d09864a7b18b -->

<!-- unit:d0fb6d362bb5121086 -->
<a id="u-d0fb6d362bb5121086"></a>

$$
A=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}+M\right),
\qquad O=AV.
$$

<!-- /unit:d0fb6d362bb5121086 -->

<!-- unit:a3d6457461ba97b3ff -->
<a id="u-a3d6457461ba97b3ff"></a>

Softmax по key axis, padding keys имеют additive mask $-\infty$; finite approximation зависит от dtype/backend. Не нужно маскировать valid будущие keys causal mask, если задача — обычный bidirectional encoder. Полностью masked row требует корректной обработки, иначе softmax может дать NaN.

<!-- /unit:a3d6457461ba97b3ff -->

<!-- unit:74fb5ba945cfeb57e6 -->
<a id="u-74fb5ba945cfeb57e6"></a>

После concat heads и $W_O$ исходный post-LN block имеет residual+LN после attention, затем FFN и снова residual+LN. FFN применяет одну и ту же нелинейную двухслойную функцию отдельно к token positions. В pre-LN нормализация стоит перед sublayer, меняя путь gradients; это architecture change, не перестановка без последствий.

<!-- /unit:74fb5ba945cfeb57e6 -->

<!-- unit:f7d1588385973356d8 -->
<a id="u-f7d1588385973356d8"></a>

Для $B=2,L=5,D=12,H_a=3$ head dimension 4; scores shape $2\times3\times5\times5$, после concat output $2\times5\times12$. Если все four projection matrices имеют biases, parameters:

<!-- /unit:f7d1588385973356d8 -->

<!-- unit:a9ba0b8e8e5a8e9618 -->
<a id="u-a9ba0b8e8e5a8e9618"></a>

$$
N_{QKVO}=4(D^2+D)=624.
$$

<!-- /unit:a9ba0b8e8e5a8e9618 -->

<!-- unit:e88a2ccd6360331844 -->
<a id="u-e88a2ccd6360331844"></a>

При $D_{ff}=48$ FFN с two biases содержит $12\cdot48+48+48\cdot12+12=1212$. Two LayerNorm с scale/bias добавляют $4D=48$. Это toy block count без embedding tables и других architecture choices.

<!-- /unit:e88a2ccd6360331844 -->

<!-- unit:3bf6909dc2355077e9 -->
<a id="u-3bf6909dc2355077e9"></a>

Исторические base/large configs — depth 12/24, widths 768/1024, heads 12/16 соответственно; это конкретные версии, не определение всех encoders. [Официальные BERT configs](https://github.com/google-research/bert).

<!-- /unit:3bf6909dc2355077e9 -->

<!-- unit:ed7e35ae59f8652d26 -->
<a id="u-ed7e35ae59f8652d26"></a>

### Что складывается на входе и что меняется по глубине

<!-- /unit:ed7e35ae59f8652d26 -->

<!-- unit:f048dfaeaad79fcbef -->
<a id="u-f048dfaeaad79fcbef"></a>

Для позиции t выбирается строка token embedding E по ID, строка position embedding P по индексу t и строка type embedding S по номеру сегмента. Все три вектора имеют ширину D. Их **складывают**, а не конкатенируют в вектор ширины 3D. После LayerNorm и dropout внешний размер остаётся D, поэтому первый Transformer-блок получает B×L×D.

<!-- /unit:f048dfaeaad79fcbef -->

<!-- unit:ec3d9973e42b1e5884 -->
<a id="u-ec3d9973e42b1e5884"></a>

Например, при D=3 и учебных $E=(1,0,2)$, $P=(0,1,-1)$, $S=(1,0,0)$ сумма равна (2,1,1). Если заменить ID при той же позиции и сегменте, изменится E; если переместить тот же ID, изменится P. После первого attention состояние уже зависит от доступных соседей. В последующих слоях снова искать новый ID по этому состоянию не требуется: скрытые векторы проходят через непрерывные преобразования, а исходные IDs служили lookup на входе.

<!-- /unit:ec3d9973e42b1e5884 -->

<!-- unit:b490a096aa6f32ec35 -->
<a id="u-b490a096aa6f32ec35"></a>

Матрица embeddings и состояния конкретного batch — разные объекты. Таблица является параметром модели; массив B×L×D — вычисленным результатом для текущих данных. Можно получить gradient по обоим, но optimizer обновляет выбранные параметры, а не сохраняет каждый hidden state как новую строку словаря.

<!-- /unit:b490a096aa6f32ec35 -->

<!-- unit:4dc85a98bfac2c4b45 -->
<a id="u-4dc85a98bfac2c4b45"></a>

### Почему post-LN нельзя заменить pre-LN перестановкой строк кода

<!-- /unit:4dc85a98bfac2c4b45 -->

<!-- unit:a378890de29c2da9ba -->
<a id="u-a378890de29c2da9ba"></a>

Для одной residual-ветви post-LN даёт $y=\operatorname{LN}(x+F(x))$, pre-LN — $y=x+F(\operatorname{LN}(x))$. В первом случае вся сумма нормируется; во втором исходный x идёт по прямому пути без этой нормировки. Полный блок добавляет аналогичную вторую ветвь FFN. Исходный BERT использует post-LN; устройство embedding layer, pooler и блоков можно сверить с [кодом модели](https://github.com/google-research/bert/blob/master/modeling.py).

<!-- /unit:a378890de29c2da9ba -->

<!-- unit:a4597abf612e1481cc -->
<a id="u-a4597abf612e1481cc"></a>

Для наглядного контрпримера возьмём x=(1,2,3), постоянную учебную ветвь $F(\cdot)=(1,0,-1)$ и LayerNorm с единичным scale, нулевым bias. В post-LN сумма равна (2,2,2), а нормированный результат — (0,0,0), поскольку все координаты равны. В pre-LN та же постоянная ветвь добавляется к x и даёт (2,2,2). Такая ветвь выбрана специально для простой арифметики; пример доказывает, что схемы не тождественны даже до обсуждения обучения.

<!-- /unit:a4597abf612e1481cc -->

<!-- unit:810f3f6a60458729f2 -->
<a id="u-810f3f6a60458729f2"></a>

В backward pre-LN residual имеет отдельное единичное слагаемое к производной. У post-LN сигнал всей суммы сначала проходит через Jacobian LayerNorm. Это одна из причин, почему выбор norm placement связан со стабильностью обучения глубоких моделей. Но из одного локального рассуждения не следует, что любая pre-LN модель гарантированно обучится или что любой post-LN checkpoint хуже. Смена схемы уже обученной модели меняет её forward и требует отдельного обоснования.

<!-- /unit:810f3f6a60458729f2 -->

<!-- unit:629509945707c53a87 -->
<a id="u-629509945707c53a87"></a>

### Где параметры общие, а где разные

<!-- /unit:629509945707c53a87 -->

<!-- unit:e87a4579acedb463ad -->
<a id="u-e87a4579acedb463ad"></a>

В одном FFN матрицы общие для всех L позиций и всех B примеров. У следующего encoder-блока обычно другой набор матриц. Поэтому 12 блоков — не 12 отдельных токенов и не 12 attention heads. Это глубина последовательных преобразований; несколько heads находятся внутри каждого attention sublayer.

<!-- /unit:e87a4579acedb463ad -->

<!-- unit:5c7ac617d4e8f57bbc -->
<a id="u-5c7ac617d4e8f57bbc"></a>

Приведённый выше блок D=12 содержит 624 параметра Q/K/V/O, 1212 FFN и 48 двух LayerNorm, всего 1884. При удвоении длины с 5 до 10 это число параметров не меняется: те же операторы применяются к большему числу позиций. Вычисления и активации растут. При удвоении ширины D матрицы, напротив, становятся больше; длина примера и размер модели влияют на ресурсы разными путями.

<!-- /unit:5c7ac617d4e8f57bbc -->

<!-- unit:fcfcae4790288a4484 -->
<a id="u-fcfcae4790288a4484"></a>

## Mask и стоимость

<!-- /unit:fcfcae4790288a4484 -->

<!-- unit:01291510765997c4a7 -->
<a id="u-01291510765997c4a7"></a>

![Bidirectional BERT использует все реальные позиции; causal mask нужна для другой постановки](../../archive/2026-10-20-before-rebuild/оформление/transformer/attention-masks.svg)

<!-- /unit:01291510765997c4a7 -->

<!-- unit:fc3a8a2f05cf5f6bf9 -->
<a id="u-fc3a8a2f05cf5f6bf9"></a>

Правая часть схемы показывает обычный bidirectional encoder. Цвет обозначает доступность связи, а не её вычисленный вес; PAD дополнительно исключается из источников.

<!-- /unit:fc3a8a2f05cf5f6bf9 -->

<!-- unit:683594e8064e776b62 -->
<a id="u-683594e8064e776b62"></a>

Dense attention score storage порядка $BH_aL^2$, attention matmul work — $BL^2D$; projections/FFN при proportional width — $BLD^2$. При переходе 128→512 length linear parts растут в 4 раза, quadratic — в 16. Actual memory также включает weights, saved activations, workspace и backend.

<!-- /unit:683594e8064e776b62 -->

<!-- unit:6308c05f3b6dc1eb68 -->
<a id="u-6308c05f3b6dc1eb68"></a>

FlashAttention-подобные algorithms меняют порядок обработки и memory traffic, позволяя не materialize все scores одновременно, но dense attention математически остаётся полной. Один long sample при padding до batch maximum увеличивает work всех shorter samples. Dynamic padding и bucketing помогают, если runtime действительно использует уменьшенные lengths.

<!-- /unit:6308c05f3b6dc1eb68 -->

<!-- unit:161425382168bae6e9 -->
<a id="u-161425382168bae6e9"></a>

Padding query states не обязательно нули: даже при masked keys valid states могут агрегироваться в padded queries. Для pooling и token loss padding исключается своими masks. Одна attention mask не означает автоматическое правильное masking каждой последующей операции.

<!-- /unit:161425382168bae6e9 -->

<!-- unit:52aa551c913d494706 -->
<a id="u-52aa551c913d494706"></a>

### Почему PAD может иметь ненулевое состояние

<!-- /unit:52aa551c913d494706 -->

<!-- unit:8dd4bee71d2321d9cb -->
<a id="u-8dd4bee71d2321d9cb"></a>

Attention mask часто запрещает использовать PAD как key/source. Но query на PAD-позиции всё ещё может иметь ненулевой вход: к embedding добавились position/type vectors, затем прошли residual и norm. Если эта query обращается к разрешённым реальным keys, её output тоже может быть ненулевым. Следовательно, проверка «мы уже передали attention mask, значит hidden states PAD равны нулю» неверна.

<!-- /unit:8dd4bee71d2321d9cb -->

<!-- unit:44a471ec0aebd7b54c -->
<a id="u-44a471ec0aebd7b54c"></a>

Для классификации с mean pooling PAD снова исключается отдельной pooling mask. Для token classification его label снова исключается из loss. Для оценки метрик PAD не должен превращаться в дополнительный правильный класс O. Один бит valid/invalid может послужить основой нескольких масок, но смысл каждой операции должен быть задан отдельно.

<!-- /unit:44a471ec0aebd7b54c -->

<!-- unit:39503a1eeaba2f29f4 -->
<a id="u-39503a1eeaba2f29f4"></a>

### Dynamic padding и bucketing решают разные части задачи

<!-- /unit:39503a1eeaba2f29f4 -->

<!-- unit:2f558cdcacd4c5a49b -->
<a id="u-2f558cdcacd4c5a49b"></a>

Пусть есть восемь примеров: четыре длины 10 и четыре длины 100. Если в каждом из двух batches по четыре примера смешаны две короткие и две длинные строки, padding до batch maximum даёт $2\cdot4\cdot100^2=80000$ attention pairs на голову в сумме. Dynamic padding не сократит их: максимум каждого batch всё равно 100.

<!-- /unit:2f558cdcacd4c5a49b -->

<!-- unit:9d66255e61f3a6ee37 -->
<a id="u-9d66255e61f3a6ee37"></a>

Если собрать один batch из четырёх коротких и другой из четырёх длинных, число пар станет $4\cdot10^2+4\cdot100^2=40400$. Порядок примеров изменился, но их состав тот же. Это учебное сравнение плотного attention по padded lengths, а не измеренный выигрыш времени. Длина после tokenizer и добавления специальных токенов должна учитываться целиком.

<!-- /unit:9d66255e61f3a6ee37 -->

<!-- unit:0caec51a738933e0f9 -->
<a id="u-0caec51a738933e0f9"></a>

Bucketing уменьшает разброс длин внутри batches. Он требует согласования с перемешиванием данных: если длина связана с классом или источником, строго отсортированные проходы могут создать нежелательный порядок обучения. При переменных batch sizes нужно согласовать нормировку loss и число обработанных примеров. Ускорение не должно незаметно поменять относительный вес длинных документов.

<!-- /unit:0caec51a738933e0f9 -->

<!-- unit:39fe92e6c4b0d8ccf5 -->
<a id="u-39fe92e6c4b0d8ccf5"></a>

## CLS, pooler и masked pooling

<!-- /unit:39fe92e6c4b0d8ccf5 -->

<!-- unit:36f581627921bb06e7 -->
<a id="u-36f581627921bb06e7"></a>

Raw CLS — token state $h_0$. Классический `pooler_output` — learned affine transform CLS с tanh, а не идентичный vector. `last_hidden_state` содержит все positions. [Interfaces outputs](https://huggingface.co/docs/transformers/main/model_doc/bert).

<!-- /unit:36f581627921bb06e7 -->

<!-- unit:ee09f2d379f32a2241 -->
<a id="u-ee09f2d379f32a2241"></a>

Masked mean с выбранным set valid/special-token positions:

<!-- /unit:ee09f2d379f32a2241 -->

<!-- unit:de54c93e003706fb47 -->
<a id="u-de54c93e003706fb47"></a>

$$
z=\frac{\sum_t m_th_t}{\sum_t m_t}.
$$

<!-- /unit:de54c93e003706fb47 -->

<!-- unit:ecb25614774ec3abf4 -->
<a id="u-ecb25614774ec3abf4"></a>

Для rows $(1,0,0),(0,2,0),(0,0,3),(9,9,9)$ и mask $(1,1,1,0)$ mean $(1/3,2/3,1)$. Если padding ошибочно включён, mean $(2.5,2.75,3)$ и direction меняется. Padding здесь намеренно nonzero: это проверяет реальную необходимость mask. Empty denominator обрабатывается отдельно.

<!-- /unit:ecb25614774ec3abf4 -->

<!-- unit:998641a47b6f548b54 -->
<a id="u-998641a47b6f548b54"></a>

Masked max также исключает padding, но выбор нулей вместо $-\infty$ может ошибочно победить отрицательные valid coordinates. Выбор и объединение слоёв, projection и normalization входят в сохраняемый контракт. MLM pretraining не гарантирует хорошую геометрию cosine similarity предложений: обучение sentence embeddings рассматривается в главе [«Sentence embeddings и contrastive learning»](<bert-embeddings.md>), retrieval — в Е.

<!-- /unit:998641a47b6f548b54 -->

<!-- unit:2b41633974d922c289 -->
<a id="u-2b41633974d922c289"></a>

### Pooling задаёт способ чтения encoder, а не новую информацию

<!-- /unit:2b41633974d922c289 -->

<!-- unit:596ef369347f2b44c5 -->
<a id="u-596ef369347f2b44c5"></a>

CLS pooling выбирает одну специально обучаемую позицию, которая могла агрегировать контекст через attention. Mean pooling усредняет выбранные состояния. Max pooling выбирает максимум каждой координаты, причём разные координаты могут прийти от разных токенов. Это три разных функции над одной последовательностью. Ни одна из них автоматически не превращает обычный MLM encoder в хорошую модель семантического сходства: полезная геометрия зависит от обучения.

<!-- /unit:596ef369347f2b44c5 -->

<!-- unit:9578e84e6907362ba7 -->
<a id="u-9578e84e6907362ba7"></a>

Классический pooler дополнительно преобразует CLS: $z=\tanh(h_0W+b)$ при row-vector записи. Даже если W единичная и b=0, вектор (1,−1) превращается примерно в (0.761594,−0.761594). Если W меняет разные координаты по-разному, меняется и направление. Поэтому raw CLS и pooler output нельзя незаметно подменять при переносе обученной task head.

<!-- /unit:9578e84e6907362ba7 -->

<!-- unit:123469d32f735fdbfe -->
<a id="u-123469d32f735fdbfe"></a>

### Градиент masked mean показывает, какие позиции обучаются напрямую

<!-- /unit:123469d32f735fdbfe -->

<!-- unit:c60dcf92efffcad713 -->
<a id="u-c60dcf92efffcad713"></a>

Пусть $M=\sum_tm_t>0$, маска фиксирована, $z=\sum_tm_th_t/M$, а downstream gradient равен g. Тогда

<!-- /unit:c60dcf92efffcad713 -->

<!-- unit:0901c1a68aa3bdf1a2 -->
<a id="u-0901c1a68aa3bdf1a2"></a>

$$
\frac{\partial\mathcal L}{\partial h_t}=\frac{m_t}{M}\,g.
$$

<!-- /unit:0901c1a68aa3bdf1a2 -->

<!-- unit:f51e6f2f0d761a69e5 -->
<a id="u-f51e6f2f0d761a69e5"></a>

Три включённых токена получают по трети градиента pooling-ветви, исключённый PAD — ноль от этой ветви. Но если исключена реальная позиция, её раннее представление всё ещё могло участвовать в attention включённых токенов. Нулевая производная непосредственного pooling по последнему h_t не доказывает отсутствие всех градиентных путей к этой позиции в предыдущих слоях.

<!-- /unit:f51e6f2f0d761a69e5 -->

<!-- unit:334d699415e0dd9355 -->
<a id="u-334d699415e0dd9355"></a>

Для masked max нельзя подставлять ноль вместо исключённого значения, если реальные координаты отрицательны. Например, valid числа (−3,−2), PAD=0: максимум без правильной маски равен 0, а должен быть −2. Математически запрещённые элементы получают −∞ перед max; полностью пустой набор требует явного решения, а не передачи бесконечностей в следующую голову.

<!-- /unit:334d699415e0dd9355 -->

<!-- unit:a2207a40c00f0df3c5 -->
<a id="u-a2207a40c00f0df3c5"></a>

Сохраняемый embedding pipeline должен включать tokenizer, выбор слоя, pooling mask, projection и нормировку. Если обучить голову на masked mean, а при inference перейти к CLS, совпадение ширины D не обеспечит совместимость. Для retrieval к этому контракту добавляются метрика сравнения и способ построения индекса, разобранные в главе [«Dense retrieval, cross-encoder, SPLADE и ColBERT»](<bert-retrieval.md>).

<!-- /unit:a2207a40c00f0df3c5 -->

## Архитектуры: подробные схемы

### Encoder и интерфейс головы

![Encoder и интерфейс головы](<../../оформление/course/arch-bert.svg>)

Входная маска исключает PAD как источник контекста. Выход encoder — H, а не готовый универсальный embedding текста. Разные головы используют H по-разному: pooling, отдельные позиции или границы. Для mean pooling знаменатель считает только допустимые позиции.

<!-- unit:95792098be8f6216e4 -->
<a id="u-95792098be8f6216e4"></a>

## Контроль и практика

<!-- /unit:95792098be8f6216e4 -->

<!-- unit:d620ba507f9456ebb3 -->
<a id="u-d620ba507f9456ebb3"></a>

1. Можно ли видеть обе части pair через SEP? **Ответ:** да; SEP boundary marker сам по себе не attention barrier.
2. Почему pooler не равен CLS? **Ответ:** есть дополнительные learned transform и tanh.
3. Что растёт в 16 раз при length 128→512? **Ответ:** quadratic attention terms, при fixed прочих dimensions.
4. Почему token loss mask отдельна от attention mask? **Ответ:** первая выбирает supervised targets, вторая — допустимые attention interactions.

<!-- /unit:d620ba507f9456ebb3 -->

<!-- unit:00a8d2ca21d06d3049 -->
<a id="u-00a8d2ca21d06d3049"></a>

5. Почему замена CLS на mean может сломать готовую голову при той же ширине D? **Ответ:** меняется распределение входных признаков и способ чтения контекста; совпадение формы не означает совпадения функции.
6. Почему pooling PAD нужно маскировать отдельно? **Ответ:** attention mask обычно закрывает PAD как источник, но не гарантирует нулевые состояния PAD queries.
7. Разделяет ли ось batch тексты одного документа? **Ответ:** каждый batch item — отдельная последовательность; обычный self-attention не связывает items друг с другом. Связи внутри длинного документа нужно организовать отдельно.
8. Меняется ли число весов блока при удвоении длины? **Ответ:** обычные общие проекции и FFN сохраняют число параметров; растут вычисления и активации, а допустимая длина также ограничена позицией и конфигурацией.

<!-- /unit:00a8d2ca21d06d3049 -->

<!-- unit:43a87199a3122ad39f -->
<a id="u-43a87199a3122ad39f"></a>

**Практика:** для пяти задач запишите inputs/labels/output shapes. Повторите block shapes при $B=2,L=7,D=24,H_a=4$; посчитайте projections/FFN. Вручную восстановите NER char span из условных pieces и вычислите masked mean. Для реального checkpoint проверьте tokenizer/config/train-eval outputs, не объявляя toy calculations результатом training.

<!-- /unit:43a87199a3122ad39f -->

## Результат главы и переход дальше

Ответьте на вопрос: **Как получить состояния и корректно агрегировать их в представление текста?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

Следующие темы: [MLM, альтернативные objectives и семейства encoder](<bert-objectives.md>), [Данные, task heads, fine-tuning и DAPT](<bert-adaptation.md>), [Classification, NER, spans и extractive QA](<bert-tasks.md>), [Sentence embeddings и contrastive learning](<bert-embeddings.md>), [Dense retrieval, cross-encoder, SPLADE и ColBERT](<bert-retrieval.md>), [Языки, длинные документы, домены и layout](<bert-domains.md>), [Метрики, калибровка, OOD и интерпретация](<bert-evaluation.md>), [Сжатие, сервис encoder и жизненный цикл индекса](<bert-operations.md>).

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
