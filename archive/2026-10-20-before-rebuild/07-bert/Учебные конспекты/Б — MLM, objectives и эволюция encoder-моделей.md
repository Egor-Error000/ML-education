# Б. Чему учится encoder до появления прикладной головы

[Оглавление](README.md). Модули **6–9**. **Происхождение:** полный BERT-pretraining — дополнение по плану. Чаты про next-token LLM дают полезный контраст, но не заменяют MLM. Каталог ниже — объяснение mechanisms и критериев, а не рейтинг checkpoint для вашей задачи.

**Уровень:** Общее ядро для pretraining route. **Предпосылки:** B1. **Используют:** B3/B4/B8; BERT model choice.

## 1. MLM: labels выбраны отдельно от corruption

В masked language modeling выбирается set target positions $\mathcal M$. Input повреждается, encoder получает $\widetilde x$, а target остаётся исходный token $x_t$:

$$
L_{\mathrm{MLM}}=-\frac1{|\mathcal M|}
\sum_{t\in\mathcal M}\log p_\theta(x_t\mid\widetilde x).
$$

Исторический BERT выбирал 15% positions; среди выбранных 80% заменял MASK, 10% случайным token, 10% оставлял unchanged. Последние тоже дают loss: selection mask и actual replacement не одно и то же. На невыбранных positions labels обычно ignored. Attention mask отвечает за context validity, а не за set MLM targets. [Исходный BERT](https://aclanthology.org/N19-1423/).

MLM head преобразует token states, затем проецирует в vocabulary; классический recipe включает transform/nonlinearity/LN перед projection и может использовать shared input/output embeddings. Logits shape $B\times L\times V$; реализация может вычислять только selected positions для экономии. Tying не отменяет gradient из output path.


<!-- source-depth-bert-Б -->
### 1.1. Один batch: что именно испорчено и за что начисляется ошибка

![Исходная строка, повреждённый вход и отдельные цели BERT MLM](../../оформление/transformer/bert-mlm-targets.svg)

Возьмём условную токенизацию `[CLS] кот спит на мягком диване [SEP]`. Реальный WordPiece может дробить слова, поэтому здесь заранее считаем каждое слово отдельным ID. Индексы: CLS=0, «кот»=1, «спит»=2, «на»=3, «мягком»=4, «диване»=5, SEP=6. Выберем цели на позициях 2, 4, 5. Позицию 2 заменим MASK, позицию 4 — случайным словом «старом», позицию 5 оставим прежней. Три разных действия выбраны специально для иллюстрации; доля выбранных позиций в этом маленьком примере не равна историческим 15%.

Вход encoder теперь: `[CLS] кот [MASK] на старом диване [SEP]`. Целевые labels: `[ignore, ignore, спит, ignore, мягком, диване, ignore]`. Предсказывать на позиции 4 нужно **«мягком»**, хотя вход содержит «старом». На позиции 5 нужно предсказывать «диване», хотя оно не было заменено. Если собирать labels после изменения входа, можно нечаянно учить модель воспроизводить corruption вместо исходного текста.

Все семь непустых позиций доступны bidirectional self-attention. Нельзя использовать MLM selection mask в качестве attention mask: это удалило бы весь контекст, кроме выбранных мест, или изменило задачу другим ошибочным способом. Нельзя также закрывать MASK от всех queries только потому, что он специальный. MASK — реальный входной токен с обучаемым embedding; PAD — заполнитель. Их роли различаются.

Здесь нет next-token shift. Строка logits **позиции 2** сравнивается с исходным ID позиции 2. Сдвиг на один токен, полезный в causal LM, сделал бы эти labels несогласованными. При этом MLM не является простым копированием всех входов: часть выбранных целей отсутствует во входе или заменена, а невыбранные позиции не дают прямого MLM loss.

### 1.2. Почему 15%, 80%, 10% и 10% имеют разные знаменатели

Исторические 80/10/10 относятся к **уже выбранным** MLM positions. В упрощённой вероятностной модели, где каждая допустимая позиция выбирается с вероятностью 0.15, на 10 000 таких позиций ожидаются 1500 целей: 1200 MASK, 150 random replacements и 150 unchanged. Оставшиеся 8500 позиций не являются целями данного прохода. Это ожидания, а не требование получить ровно эти количества в каждом batch.

Случайная замена может совпасть с исходным ID. При равномерной выборке из V токенов вероятность совпадения равна 1/V; ожидаемое число таких совпадений среди 150 random events — 150/V. Следовательно, «random branch выбрана» и «значение ID действительно изменилось» не одно событие. Для MLM label всё равно остаётся исходным ID, а для RTD важен факт совпадения с исходником.

В реальном preprocessing используются целое число выбранных позиций, ограничения на максимум целей и исключение специальных токенов; whole-word masking может объединять subwords. Поэтому для короткой строки точную долю нужно смотреть в реализации, а не обещать ровно 15%. Исторические правила и подготовку labels можно сверить с [кодом BERT](https://github.com/google-research/bert/blob/master/create_pretraining_data.py).

Наличие unchanged targets сокращает разрыв между специальным MASK-входом предобучения и обычными входами прикладных задач. Но из этого не следует, что модель не может использовать простое копирование в части случаев. Это осознанная смесь обучающих ситуаций. Если поменять доли, меняется objective distribution; сравнение losses при разных corruption policies уже не является чистым сравнением качества представления.

### 1.3. Полный forward: embeddings, encoder, MLM head

Для B примеров длины L lookup создаёт массив B×L×D. В классическом BERT к token embeddings добавляются position и segment/type embeddings. Bidirectional Transformer encoder преобразует их в contextual states той же внешней формы. MLM head строит logits по словарю для выбранных мест. Дополнительная transform/nonlinearity/normalization перед vocabulary projection — часть конкретного устройства head; она не превращает encoder в авторегрессионный decoder.

Если нужны только M выбранных состояний, их можно собрать в массив M×D перед head. Тогда logits будут M×V, а не B×L×V; смысл loss на тех же целях сохраняется. Однако encoder всё равно обработал весь доступный контекст. Такая экономия выходной проекции не означает, что Transformer stack прошёл только по выбранным токенам.

При общей входной и выходной embedding-таблице $E\in\mathbb R^{V\times D}$ logits после преобразования состояния u можно записать $z=uE^\top+b$. Один параметр E используется в двух ролях: lookup входных ID и проекция в словарь. Градиенты этих использований суммируются. Поэтому строка E для отсутствующего во входе слова всё равно может получить градиент через выходной softmax. При отдельной output matrix это будут другие параметры.

### 1.4. Чему именно учит контекстный градиент

Пусть на MASK-позиции модель предсказывает «спит» с вероятностью 0.2, а конкурирующие варианты получают 0.5 и 0.3. Для трёхклассовой учебной задачи градиент logits равен $(-0.8,0.5,0.3)$, если правильный класс записан первым. Величина ошибки определяется всей нормировкой softmax, а не только одним правильным logit. Градиент проходит к состоянию MASK, затем через attention к «кот», «на», «диване» и другим доступным состояниям.

Позиция «кот» не включена в labels, но её K/V могли помочь предсказать «спит». Через этот путь она получает градиент. С другой стороны, сам факт доступности позиции не гарантирует большой градиент: вклад зависит от весов, значений и дальнейшей вычислительной цепи. Нет отдельного supervised указания «глагол должен смотреть на подлежащее»; полезные зависимости отбираются через ошибку восстановления токенов.

Две цели с вероятностями 0.6 и 0.5 дают среднюю ошибку около 0.601986. Если объединить его с другим примером, где одна цель имеет вероятность 0.9, среднее по **трём целям** равно $-(\log0.6+\log0.5+\log0.9)/3\approx0.436444$. Среднее двух sequence means равно приблизительно 0.353673. В первом случае каждый target token имеет одинаковый вес; во втором одинаковый вес имеют примеры. Нужно выбрать нормировку явно, особенно при разном числе MLM targets и gradient accumulation.

### 1.5. Dynamic masking и что считать одним проходом корпуса

Если одна и та же строка каждый раз приходит с одной и той же заменой, модель встречает ограниченный набор задач восстановления. При dynamic masking выбранные места и corruption могут меняться при новом предъявлении. Это даёт разные вопросы к тому же тексту, но не добавляет новых независимых документов. Train/validation split всё равно нужно делать по исходным документам или связанным группам до размножения masked variants.

В идеализированной модели с независимым выбором позиции с вероятностью p в каждом из k предъявлений вероятность хотя бы раз получить её как цель равна $1-(1-p)^k$. При p=0.15 и k=10 это около 0.803126. Среднее число выборов равно kp=1.5, а не 10. Реальные ограничения внутри последовательности создают зависимости между позициями; формула показывает поведение отдельно взятой позиции при объявленных предположениях.

Неподвижный validation masking seed делает сравнение шагов более воспроизводимым. Если каждый раз случайно меняются цели, колебания loss включают изменение самих заданий. Это не обязательно неверный протокол, но его дисперсию нужно учитывать. Оценка должна явно определять, какие позиции, замены, длины и denominators использованы.

### 1.6. Где в совместном pretraining появляется CLS и NSP

MLM даёт предсказания выбранных token positions. Исторический NSP добавляет задачу на паре сегментов: связан ли B с продолжением A в подготовленных данных. Вход может выглядеть как `[CLS] A [SEP] B [SEP]`; segment IDs отмечают принадлежность, а attention обычно разрешён между обоими сегментами. Это не два изолированных encoder forward, если mask не задаёт такой изоляции.

Для бинарного NSP head используют итоговое представление CLS с предусмотренным преобразованием. При правильной метке и вероятности 0.8 соответствующая cross-entropy равна $-\log0.8\approx0.223144$. Совместный loss суммирует token-level MLM и sequence-level NSP с выбранными весами. Их нормировки важны: сумма по сотням MLM targets и среднее по примерам задают другое относительное влияние, чем два отдельно усреднённых losses.

CLS участвует в общем attention, поэтому его loss способен обучать другие позиции; MLM в свою очередь способен влиять на CLS через зависимости в стеке. Но одно наличие CLS не гарантирует, что его cosine distance станет качественной метрикой сходства предложений. NSP labels, MLM targets и contrastive sentence training решают разные задачи. Для sentence embeddings нужен отдельный разбор геометрии и обучения в главе Д.

### 1.7. Что уносится в downstream модель

После pretraining обычно сохраняют tokenizer, embedding tables и encoder weights. MLM/NSP heads нужны, если продолжается та же задача, например domain-adaptive MLM. Для классификации, NER или extractive QA добавляют соответствующую голову и обучают на её labels. Замена головы не «стирает язык» из encoder; но случайно инициализированная новая голова сама ещё не умеет решать задачу до обучения.

Pretraining на текстах без ручной разметки всё равно имеет targets: их автоматически строят из исходного текста. Поэтому self-supervised не означает «loss отсутствует» или «веса сами меняются от чтения». Каждое обновление по-прежнему состоит из forward, вычисления ошибки, backward и шага optimizer. Bidirectional mask определяет доступный контекст, а objective определяет, какие предсказания считаются полезными.

## 2. CE и точный gradient

Пусть vocabulary size 4. Для selected position 1 prediction $(0.1,0.6,0.2,0.1)$, correct ID 2; для position 2 — $(0.5,0.2,0.2,0.1)$, correct ID 1. Mean CE:

$$
L=-\frac{\log0.6+\log0.5}{2}\approx0.601986.
$$

Derivative softmax-CE по logits одного example — $p-e_y$; после mean по двум positions делим на 2. First row gradient $(0.05,-0.20,0.10,0.05)$, second $(-0.25,0.10,0.10,0.05)$. Sums gradients равны нулю, что соответствует invariance softmax к common logit shift.

Выбранный unchanged token доступен модели в input: objective становится проще, но позиция всё равно supervised по recipe. Это не повод удалить label автоматически. Слишком много unchanged targets способно изменить task difficulty; corruption policy — часть experiment.

MLM loss не factorizes ordinary autoregressive joint likelihood $\prod_t p(x_t\mid x_{<t})$. Pseudo-log-likelihood отдельно маскирует positions и оценивает conditional scores; её exp нельзя без объявления назвать той же perplexity, что causal LM. Different masking/evaluation recipes тоже меняют score.

## 3. NSP, SOP, spans и RTD

NSP обучает отличать настоящий следующий segment от sampled negative. Если negatives случайно приходят из другого domain, auxiliary task может решаться по style вместо discourse relation. SOP в [ALBERT](https://arxiv.org/abs/1909.11942) сравнивает правильный и переставленный порядок segments, изменяя именно supervision coherence. Для «A: оплатите счёт; B: уже оплатил» swapped order и unrelated sentence являются разными negatives.

[RoBERTa](https://arxiv.org/abs/1907.11692) пересматривает training recipe, включая dynamic masking и отсутствие NSP. Controlled conclusion требует учитывать одновременно data, batch и compute; из этого не следует, что NSP всегда вреден для любой архитектуры/задачи.

[SpanBERT](https://arxiv.org/abs/1907.10529) маскирует contiguous spans и добавляет span-boundary supervision. Идея — обучить representations восстанавливать span information по boundaries, что соответствует span-oriented tasks. Это отличается от independent token masking и не даёт автоматически готовые NER labels.

[ELECTRA](https://arxiv.org/abs/2003.10555) использует generator для corruption, а discriminator предсказывает, какие resulting tokens заменены. Если sampled token совпал с original, он не становится incorrect merely because sampling happened. RTD supervises binary decisions по actual input/original relation; generator и discriminator имеют разные objectives. Это не GAN-discriminator, который обязан работать как realtime answer verifier.

Для всех objectives проверяется negative sampling: pair leakage, источник, length и easy artifacts. Сравнение mechanism полезно даже без дорогого собственного pretraining; опубликованные ablations нужно читать с их compute/data constraints.

## 4. Семейства: какую проблему меняют

ALBERT factorizes vocabulary embedding: $V\times E$ плюс $E\times D$ вместо $V\times D$ при $E<D$. Layer parameter sharing уменьшает число unique weights, но shared block всё равно исполняется несколько раз. Memory weights и FLOPs не одно число.

DistilBERT переносит teacher behaviour через distillation и compact architecture. Это обучение student по нескольким signals, а не просто удаление каждого второго слоя без adaptation. Detailed task-specific compression находится в И.

DeBERTa отделяет content/relative-position contributions attention. DeBERTa-v3 использует RTD и gradient-disentangled embedding-sharing recipe; это не просто BERT с другим количеством layers. [DeBERTa-v3](https://arxiv.org/abs/2111.09543). Для реализации смотрят actual config/tokenizer/modules, а не совпадение общего слова Transformer.

XLM-R — multilingual pretraining с общей model/vocabulary; potential transfer сочетается с конкуренцией языков за capacity. Language coverage в corpus не доказывает одинаковое task quality каждого языка. [XLM-R](https://arxiv.org/abs/1911.02116).

## 5. Современный срез: 2024–2026

**ModernBERT** сочетает RoPE, local/global attention и современный FFN/efficiency recipe. Опубликован native context 8192. Вместимость не гарантирует успешные distant dependencies или русскоязычную задачу; нужен соответствующий task-trained checkpoint. [Работа](https://arxiv.org/abs/2412.13663).

**NeoBERT** исследует depth/width и modern recipe; в публикации описан encoder примерно 250M с context 4096 и unified fine-tuning comparisons. Это useful controlled case study, а не доказательство лучшего качества каждого downstream task. [Работа](https://arxiv.org/abs/2502.19587).

**mmBERT** описывает multilingual encoder с phased language inclusion и annealing; публикация ICML 2026 позволяет изучить timing language exposure. При выборе оценивают нужный язык, morphology и target objective отдельно. [Первичная публикация](https://proceedings.mlr.press/v306/marone26a.html).

**Ettin** — paired encoder/decoder suite для более контролируемого сравнения objectives/data/compute. Использовать как пример experiment design: сравнение двух randomly выбранных checkpoints с разными corpora не изолирует влияние mask/objective. [Seq vs Seq](https://arxiv.org/abs/2507.11412).

## 6. Как выбрать кандидатов

Для русского NER важны tokenizer fragmentation, language/task adaptation и span alignment. Для длинной English classification — native length и actual evidence coverage. Для multilingual retrieval нужен retrieval-trained geometry и cross-language evaluation, а не только multilingual MLM backbone.

Shortlist включает model/revision/license, objective, width/depth, position mechanism, context, language/domain и runtime support. Затем один split, fair tuning budget, cost и error slices. Catalog помогает предложить hypothesis; validation выбирает итог.

## 7. Контроль и практика модулей 6–9

1. Даёт ли selected unchanged position loss? **Ответ:** да, если она входит в MLM target set.
2. Почему ALBERT sharing не равно пропуску layers? **Ответ:** weights общие, но вычисления повторяются на каждом применении.
3. Является ли sampled original token заменённым для RTD? **Ответ:** label определяется совпадением с original, а не фактом запуска sampler.
4. Доказывает ли native length качество длинного reasoning? **Ответ:** нет; это capacity context, которое проверяют задачей.

5. Сдвигаются ли MLM labels на следующий токен? **Ответ:** нет. Цель выбранной позиции — исходный ID той же позиции; это отличается от causal next-token loss.
6. Нужно ли исключать MASK из доступного контекста? **Ответ:** нет. MASK является реальным обучаемым токеном; padding и MLM corruption имеют разные роли.
7. Предсказывает ли MLM head ID случайной замены? **Ответ:** нет, исходный ID. Если labels собраны после corruption без сохранения оригинала, задача подготовлена неверно.
8. Является ли меньшее MLM loss при другом masking recipe доказательством лучшего encoder? **Ответ:** нет. Изменились задания и их трудность; нужен сопоставимый evaluation и downstream контроль.

**Практика:** повторите CE/gradients и составьте batch с MASK/random/unchanged selected tokens. Для короткого диалога создайте NSP/SOP/hard-negative cases. По двум configs сравните механизм, data/objective и budget. Для трёх прикладных сценариев выше выберите критерии shortlist, не объявляя winner без evaluation.
