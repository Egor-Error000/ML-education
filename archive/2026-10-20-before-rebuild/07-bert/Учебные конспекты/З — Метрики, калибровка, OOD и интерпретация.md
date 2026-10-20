# З. Как оценить encoder без переоценки его возможностей

[Оглавление](README.md). Группа H, модули **35–37**. **Основа:** evaluation из S3 и retrieval из S6; task-specific statistics и interpretation — расширение по плану. Полные общие разделы находятся в [данных](../../01-data/Учебные%20 конспекты/README.md).

**Уровень:** Общее ядро до утверждений о качестве. **Предпосылки:** B3 and chosen B4/B5/B6 task. **Используют:** B9; E5 application evaluation.

## 1. Метрика следует единице задачи

Classification: macro/micro-F1, PR-AUC и cost threshold. NER: strict entity F1 с type/boundaries. QA: exact match и token-overlap F1 с declared normalization плюс unanswerable quality. STS: Pearson linear association и Spearman rank correlation — разные properties. High correlation не означает calibrated absolute scores.

Retrieval для queries $q$:

$$
\operatorname{MRR}=\frac1{|Q|}\sum_q\frac1{r_q},
$$

где вклад 0, если relevant не найден в cutoff. Для graded labels:

$$
\operatorname{DCG}@k=\sum_{i=1}^k\frac{2^{g_i}-1}{\log_2(i+1)},
\qquad
\operatorname{nDCG}@k=\frac{\operatorname{DCG}@k}{\operatorname{IDCG}@k}.
$$

Если IDCG=0, нужна declared convention. Для grades (2,0,1) DCG@3=3.5, ideal (2,1,0) даёт $3+1/\log_2 3$, nDCG около 0.96394. MRR этого ranking равен 1, потому что первый result relevant; MRR не видит ухудшение порядка остальных релевантных docs.

Recall@k относительно qrels зависит от полноты judgements. Unjudged document не обязательно irrelevant. Aggregate queries без judged relevant отдельно определяются protocol, иначе denominators несопоставимы. Classification, retrieval и ANN approximation scores нельзя объединить в одну «accuracy encoder».




**Схема курса:** held-out examples → task metric and subgroup slices → calibration/OOD score → accept, abstain or review → error and uncertainty report

<!-- source-depth-bert-З -->
### 1.1. Почему accuracy может скрыть провал редкого класса

Пусть из 100 текстов 95 принадлежат классу A, 5 — B, а модель всегда предсказывает A. Accuracy=0.95. Для A имеем TP=95,FP=5,FN=0, F1=190/195≈0.974359. Для B F1=0, поэтому macro-F1≈0.487179. Высокая accuracy здесь не означает, что модель научилась распознавать B.

Для multiclass single-label задачи micro-F1 по всем классам совпадёт с accuracy при обычном подсчёте всех decisions. Но per-class recall и macro average показывают другую часть поведения. Если часть labels отсутствует или используются особые masks, определения denominators нужно проверить заново. Число с названием F1 без единицы задачи и правила усреднения недостаточно.

### 1.2. Что видят MRR, nDCG и Recall

Для трёх queries первые relevant ranks 1,2 и «не найдено в cutoff» дают MRR=(1+1/2+0)/3=0.5. Нельзя убрать неудачный query из знаменателя, если protocol считает его частью задачи. Иначе сложные случаи исчезнут из метрики.

MRR не различает два rankings, если первый relevant стоит на одном месте, хотя остальные relevant documents могут быть потеряны. Recall@k учитывает, какую долю известных relevant нашли. nDCG дополнительно учитывает grades и позиции. У одной модели могут быть выше MRR и ниже Recall: она быстро находит один хороший ответ, но хуже покрывает несколько нужных источников.

В исходном nDCG примере gains=(3,0,1), discounts=(1,1/log2(3),1/2), поэтому DCG=3.5. Идеальный порядок даёт 3+1/log2(3). Это определение с gain 2^grade−1; другой gain convention даёт другие значения. Queries без judged relevant и неполные qrels требуют отдельного правила, чтобы сравниваемые методы имели одинаковую основу.


## 2. Benchmarks и uncertainty

[BEIR](https://arxiv.org/abs/2104.08663) предлагает heterogeneous retrieval tasks, [MTEB](https://arxiv.org/abs/2210.07316) — разнообразие embedding tasks, [MMTEB](https://arxiv.org/abs/2502.13595) — multilingual expansion. Средний leaderboard score относится к versions/subsets/protocol, не универсальному deployment ranking.

Для честного сравнения фиксируют dataset revisions, split, encoder/pooling/prefixes и search budget. Парное сравнение на одних queries учитывает общую difficulty. Bootstrap resamples query/document groups, когда именно они независимы; correlated token rows одного document не создают тысячи independent trials.

Seeds отражают training variation, dataset intervals — sampling uncertainty. Report effect size, slices и cost. Выбор maximum из множества configurations/seed по test делает test development set. Небольшой средний gain не отменяет серьёзную regression критичного rare slice.


### 2.1. Что именно должно оставаться парным

Если сравниваются encoders A и B, каждый query должен обрабатываться обоими по одному protocol. Difficulty конкретного query тогда влияет на обе оценки. Для bootstrap выбирают query indices с возвращением и используют **те же выбранные indices** для A и B. Независимо перемешать результаты двух методов означало бы потерять парную структуру сравнения.

Если один исходный документ породил сто похожих chunks, случайное resampling chunks не делает их независимыми. Для оценки переноса на новые документы resample выполняют по исходным группам. Аналогично для диалогов и переводов. Единица uncertainty следует единице случайности, которую предполагается обобщать.

Несколько seeds показывают вариативность training pipeline, но не заменяют разнообразие test cases. Один seed на большом test и десять seeds на пяти почти одинаковых примерах отвечают на разные вопросы. Report содержит разницу методов, её вариативность, важные slices и цену; значение «лучший seed» само по себе не является воспроизводимым эффектом.

### 2.2. Benchmark average — результат конкретного набора задач

Если одна версия набора включает больше English classification, а другая — multilingual retrieval, одинаковое имя агрегата ещё не делает scores сопоставимыми. Сохраняют dataset revisions, task weights, corpus, query preprocessing, pooling, search parameters и фильтры. При сравнении нового encoder с другим corpus невозможно отделить качество representation от доступности документов.

Многократный выбор по final test переносит в решение информацию о test, даже если его labels не использовались в backprop. Validation служит для выбора architecture/thresholds/config, test — для независимой оценки уже выбранного pipeline. Для редких критичных случаев лучше заранее объявить отдельный slice, чем объяснять его падение после усреднения.


## 3. Calibration и отказ

Temperature scaling classifier logits использует $p_T=\operatorname{softmax}(z/T)$, $T>0$, fit по calibration NLL. Положительный scalar T сохраняет argmax и ranking classes данного example; он не исправляет wrong class ordering или missing evidence. Для independent labels protocol calibration может отличаться.

Brier/NLL и reliability bins проверяют probabilities, а не только accuracy. Softmax confidence не universal OOD detector: unfamiliar input может получить confident prediction. Нужны held-out unknown/domain/time cases, а не только in-distribution histogram.

Selective prediction принимает set $\mathcal A_\tau=\{i:c_i\ge\tau\}$:

$$
\operatorname{Coverage}(\tau)=\frac{|\mathcal A_\tau|}n,
\qquad
\operatorname{Risk}(\tau)=
\frac{\sum_{i\in\mathcal A_\tau}\mathbf1[\widehat y_i\ne y_i]}{|\mathcal A_\tau|}.
$$

При пустом set risk не определён. Если из 100 inputs принято 60 и в них 3 ошибки, coverage 0.6, risk 0.05. Отказ от остальных 40 требует operational route: review, additional retrieval или явный no-answer. Нельзя сообщать только 95% accepted accuracy без coverage.


### 3.1. Две модели с одинаковыми решениями могут иметь разные probabilities

Возьмём два бинарных примера: правильные labels 1 и 0, но модель в обоих выдаёт p(class1)=0.9. Accuracy=0.5. Средняя NLL равна $-(\log0.9+\log0.1)/2\approx1.203973$, binary Brier — $(0.1^2+0.9^2)/2=0.41$. Если softened probabilities обе равны 0.6, решения и accuracy прежние, но NLL≈0.713558, Brier=0.26.

Это пример, где снижение уверенности полезно. Если все уверенные предсказания были правильными, то такое же смягчение может ухудшить NLL. Поэтому T подбирают на отдельной calibration части, а не объявляют «чем мягче, тем лучше». Temperature scaling меняет шкалу probabilities, сохраняя argmax классов каждого примера; исходный метод обсуждается в [работе о calibration](https://arxiv.org/abs/1706.04599).

### 3.2. Внутрипримерный ranking и порядок confidence разных примеров

Для T>0 деление logits на T сохраняет их порядок **в одном примере**. Но порядок max-softmax confidence разных multiclass примеров может измениться. Возьмём A=(4,0,0), B=(3,2.9,−10). При T=1 confidence A выше B; при T=10 confidence B выше A, хотя первый класс остаётся argmax в обоих. Распределение остальных logits влияет на знаменатель.

Это важно для abstention: новый T способен изменить не только численное значение порога, но и то, какие examples попадут в accepted set. После calibration coverage-risk проверяется заново. Общий scalar T не обучает другой class boundary, не возвращает обрезанный evidence и не гарантирует корректную вероятность на новом domain.

### 3.3. Reliability и OOD требуют labels или ожидаемого outcome

В reliability bin сравнивают среднюю confidence с фактической долей правильных решений. Малые bins дают шумную оценку, а крупные могут скрыть разные подгруппы. Один низкий calibration error на in-distribution test не доказывает надёжность на неизвестном языке или новом источнике.

Для отказа при no-answer/OOD нужны случаи такого типа в evaluation. Высокая confidence на бессмысленном input — возможное поведение модели, а не логическое противоречие softmax. Risk при пустом accepted set не определён; система с нулём ответов не получает полезную «нулевую ошибку» без отчёта coverage и правила обработки отказов.


## 4. Robustness tests с ожидаемым поведением

Typos, OCR noise, code switching, длинный context, числа, quotations и negation проверяют разные risks. Harmless spelling change должен часто сохранять label, смена «есть» на «нет» может обязана его изменить. Blanket invariance ко всем perturbations неверна.

Test cases фиксируют expected outcome и reason. Если annotation policy неоднозначна, сначала разрешается label issue, а не подбор model confidence. Изменение time/domain distribution, missing classes и new entities оценивается отдельно; one temperature не устраняет произвольный concept drift.


### 4.1. Инвариантность нужна только там, где target действительно сохраняется

Для темы обращения небольшая опечатка может не менять label. Но замена одного символа в коде ошибки A12→A13 может изменить нужный документ, хотя с точки зрения edit distance это похожее возмущение. Удаление «не» меняет отрицание; перестановка аргументов в relation extraction может менять roles. Нельзя требовать одинаковый prediction после любого малого редактирования.

Каждый test case задаёт original input, преобразование, ожидаемую связь ответов и причину. Для label-preserving paraphrase сравнивают стабильность, для meaning-changing pair — чувствительность. Если ожидаемый label неоднозначен, это сначала проблема annotation policy. Confidence не должна определять истинную разметку задним числом.

К изменениям языка, источника и времени применяют тот же принцип. Сдвиг частот labels отличается от нового определения labels; оба могут ухудшить quality, но требуют разных исследований. Один temperature parameter не исправляет произвольную смену target function.


## 5. Attribution, probes и causal diagnosis

Gradient/saliency показывает local sensitivity, attention inspection — внутренние weights выбранного layer/head. Ни одно не является готовым человеческим объяснением causal decision. Probe может декодировать feature hidden state, хотя main head его не использует.

Integrated gradients для baseline $x'$ и input $x$ задаёт attribution:

$$
\operatorname{IG}_i(x)=(x_i-x_i')
\int_0^1\frac{\partial F(x'+\alpha(x-x'))}{\partial x_i}\,d\alpha.
$$

Для text interpolation обычно идёт в embedding space, а не между integer IDs; baseline/path определяют interpretation. Нужны controls и actual perturbations: source metadata removal, matched negation pairs, label-preserving paraphrase. Decodability, sensitivity и causal use различаются. Первичный метод: [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365).

Shortcut может быть длиной, source или entity name. Высокое качество random split тогда не переносится на new source. Для проверки гипотезы «использует отрицание» сравнивают minimally differing pairs и контроль без semantic change, а не только наличие token «не» среди ярких saliency features.


### 5.1. Endpoint gradient, integrated gradients и вмешательство

Для учебной функции $F(x_1,x_2)=x_1x_2$, baseline (0,0) и input (2,3) градиент в input равен (3,2). Простое gradient×input даёт (6,6), сумма 12. Integrated gradients по прямому пути даёт (3,3), сумма 6=F(2,3)−F(0,0). Производная по x1 вдоль пути равна 3α: интеграл от 0 до 1 равен 1.5, затем умножается на разность x1=2. Для второй координаты получается тот же вклад 3.

Пример показывает, почему endpoint sensitivity и path attribution — разные объекты. У IG при нужных условиях есть completeness относительно baseline/path; численная интеграция приближает её. Это не означает, что каждый вклад является доказанной человеческой причиной ответа или что chosen baseline соответствует отсутствию смысла.

У текста integer token IDs нельзя линейно интерполировать как признаки с осмысленной геометрией. Обычно интерполируют embeddings. Нулевой embedding или другой baseline может не соответствовать реальному предложению. Удаление слова также меняет tokenization и контекст; это другое вмешательство, которое нужно оценивать отдельно.

### 5.2. Что доказывает probe и чего не доказывает

Если отдельный classifier может восстановить часть речи из hidden states, эта информация доступна выбранному decoder/probe. Но исходная task head могла её не использовать. Если attention map выделяет слово, оно участвовало в одной mixture с выбранными V, однако конечный ответ зависит от residual, других heads и последующих слоёв. Красивое выделение не заменяет причинную проверку.

Гипотезу о shortcut проверяют контролируемым изменением: убрать source marker при сохранённом содержании, заменить имя на другое с тем же label, сравнить отрицание с контрольной опечаткой. Важно не разрушить target вместе с якобы лишним признаком. После вмешательства смотрят actual task output и качество, а не только новую saliency картинку.


## 6. Data access и privacy в прикладном contract

Retriever и PII extractor тоже обрабатывают adversarial/untrusted input. Embedded instructions или special-token-like strings требуют task tests; encoder-only architecture не даёт абсолютной защищённости любой обвязки. ACL/filter применяется до выдачи недоступных results по contract системы.

PII head может пропустить entity. Удаление известных имён не доказывает полной anonymization: косвенные identifiers и context остаются. Для реального использования задаются retention/logging/masking policies и audit соответствующих failures. Это конкретные properties pipeline, а не автоматически получаемое качество checkpoint.

## 7. Контроль и практика модулей 35–37

1. Почему MRR может быть 1 при nonideal ranking? **Ответ:** он учитывает только first relevant rank.
2. Исправляет ли T scaling wrong argmax? **Ответ:** positive global temperature сохраняет argmax.
3. Почему probe не доказывает causal use? **Ответ:** информация может присутствовать в state, но не влиять на actual head decision.
4. Почему accepted accuracy без coverage неполна? **Ответ:** model могла отвергнуть большую часть трудных inputs.

5. Может ли T изменить порядок confidence разных multiclass примеров? **Ответ:** да. Сохраняется порядок классов внутри примера; max-softmax разных примеров зависит и от других logits.
6. Равны ли gradient×input и IG? **Ответ:** в общем случае нет. Для F=x1x2 и input(2,3) первый даёт(6,6), IG от нулевого baseline —(3,3).
7. Почему bootstrap выполняют по исходным query/document groups? **Ответ:** множество связанных chunks не образует столько же независимых trials; resampling должен сохранять единицу зависимости.
8. Исправляет ли calibration обрезанный evidence? **Ответ:** нет. Она меняет шкалу probabilities при прежних представлениях и информации входа.

**Практика:** посчитайте MRR/nDCG двух rankings, постройте coverage-risk для шести scores. Предложите три robustness pairs с expected labels и probe control. Подготовьте paired evaluation protocol с независимой calibration частью и объяснением limits, без универсального вывода из одного benchmark.
