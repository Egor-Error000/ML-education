# Метрики, калибровка, OOD и интерпретация

**Уровень:** основа ветки.

**Предпосылки:** [Encoder, tokenizer, attention и pooling](<bert-interface.md>), [Данные, task heads, fine-tuning и DAPT](<bert-adaptation.md>)

**Используют:** [Сжатие, сервис encoder и жизненный цикл индекса](<bert-operations.md>)

**Учебный вопрос:** Как оценить encoder на правильной единице задачи и проверить отказ?





## Карта вычисления и границы этапов

![Метрики, калибровка, OOD и интерпретация: входы, преобразования и выходы](<../../оформление/course/bert-evaluation.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Сравните формы и смысл каждого перехода с формулами главы. Выход преобразования и доказательство его качества — разные объекты: последняя проверка использует отдельный протокол оценки.

<!-- unit:6d6e66e390fc51628b -->
<a id="u-6d6e66e390fc51628b"></a>

## Метрика следует единице задачи

<!-- /unit:6d6e66e390fc51628b -->

<!-- unit:946b7a606d7dd88e32 -->
<a id="u-946b7a606d7dd88e32"></a>

Classification: macro/micro-F1, PR-AUC и cost threshold. NER: strict entity F1 с type/boundaries. QA: exact match и token-overlap F1 с declared normalization плюс unanswerable quality. STS: Pearson linear association и Spearman rank correlation — разные properties. High correlation не означает calibrated absolute scores.

<!-- /unit:946b7a606d7dd88e32 -->

<!-- unit:376fef518a5af04271 -->
<a id="u-376fef518a5af04271"></a>

Retrieval для queries $q$:

<!-- /unit:376fef518a5af04271 -->

<!-- unit:4873d4da858bdb0954 -->
<a id="u-4873d4da858bdb0954"></a>

$$
\operatorname{MRR}=\frac1{|Q|}\sum_q\frac1{r_q},
$$

<!-- /unit:4873d4da858bdb0954 -->

<!-- unit:87e6ac781912b670c4 -->
<a id="u-87e6ac781912b670c4"></a>

где вклад 0, если relevant не найден в cutoff. Для graded labels:

<!-- /unit:87e6ac781912b670c4 -->

<!-- unit:c1ff3d0b808601cc31 -->
<a id="u-c1ff3d0b808601cc31"></a>

$$
\operatorname{DCG}@k=\sum_{i=1}^k\frac{2^{g_i}-1}{\log_2(i+1)},
\qquad
\operatorname{nDCG}@k=\frac{\operatorname{DCG}@k}{\operatorname{IDCG}@k}.
$$

<!-- /unit:c1ff3d0b808601cc31 -->

<!-- unit:2b726c3e14ff031a44 -->
<a id="u-2b726c3e14ff031a44"></a>

Если IDCG=0, нужна declared convention. Для grades (2,0,1) DCG@3=3.5, ideal (2,1,0) даёт $3+1/\log_2 3$, nDCG около 0.96394. MRR этого ranking равен 1, потому что первый result relevant; MRR не видит ухудшение порядка остальных релевантных docs.

<!-- /unit:2b726c3e14ff031a44 -->

<!-- unit:71c0771bae687ca625 -->
<a id="u-71c0771bae687ca625"></a>

Recall@k относительно qrels зависит от полноты judgements. Unjudged document не обязательно irrelevant. Aggregate queries без judged relevant отдельно определяются protocol, иначе denominators несопоставимы. Classification, retrieval и ANN approximation scores нельзя объединить в одну «accuracy encoder».

<!-- /unit:71c0771bae687ca625 -->

<!-- unit:0704d61c7eb57e34ea -->
<a id="u-0704d61c7eb57e34ea"></a>

**Схема курса:** held-out examples → task metric and subgroup slices → calibration/OOD score → accept, abstain or review → error and uncertainty report

<!-- /unit:0704d61c7eb57e34ea -->

<!-- unit:1cd7eef88aa8af2266 -->
<a id="u-1cd7eef88aa8af2266"></a>

<!-- source-depth-bert-З -->

<!-- /unit:1cd7eef88aa8af2266 -->

<!-- unit:b9e64155941587ddf7 -->
<a id="u-b9e64155941587ddf7"></a>

### Почему accuracy может скрыть провал редкого класса

<!-- /unit:b9e64155941587ddf7 -->

<!-- unit:bd2218bbf2a70af4b5 -->
<a id="u-bd2218bbf2a70af4b5"></a>

Пусть из 100 текстов 95 принадлежат классу A, 5 — B, а модель всегда предсказывает A. Accuracy=0.95. Для A имеем TP=95,FP=5,FN=0, F1=190/195≈0.974359. Для B F1=0, поэтому macro-F1≈0.487179. Высокая accuracy здесь не означает, что модель научилась распознавать B.

<!-- /unit:bd2218bbf2a70af4b5 -->

<!-- unit:69940990c1c46ca6a0 -->
<a id="u-69940990c1c46ca6a0"></a>

Для multiclass single-label задачи micro-F1 по всем классам совпадёт с accuracy при обычном подсчёте всех decisions. Но per-class recall и macro average показывают другую часть поведения. Если часть labels отсутствует или используются особые masks, определения denominators нужно проверить заново. Число с названием F1 без единицы задачи и правила усреднения недостаточно.

<!-- /unit:69940990c1c46ca6a0 -->

<!-- unit:f43ae1c9dc25496079 -->
<a id="u-f43ae1c9dc25496079"></a>

### Что видят MRR, nDCG и Recall

<!-- /unit:f43ae1c9dc25496079 -->

<!-- unit:9fea2ace82881ff7f2 -->
<a id="u-9fea2ace82881ff7f2"></a>

Для трёх queries первые relevant ranks 1,2 и «не найдено в cutoff» дают MRR=(1+1/2+0)/3=0.5. Нельзя убрать неудачный query из знаменателя, если protocol считает его частью задачи. Иначе сложные случаи исчезнут из метрики.

<!-- /unit:9fea2ace82881ff7f2 -->

<!-- unit:d2c37cb91a8da267bb -->
<a id="u-d2c37cb91a8da267bb"></a>

MRR не различает два rankings, если первый relevant стоит на одном месте, хотя остальные relevant documents могут быть потеряны. Recall@k учитывает, какую долю известных relevant нашли. nDCG дополнительно учитывает grades и позиции. У одной модели могут быть выше MRR и ниже Recall: она быстро находит один хороший ответ, но хуже покрывает несколько нужных источников.

<!-- /unit:d2c37cb91a8da267bb -->

<!-- unit:ec6081f8ef00a9e3c7 -->
<a id="u-ec6081f8ef00a9e3c7"></a>

В исходном nDCG примере gains=(3,0,1), discounts=(1,1/log2(3),1/2), поэтому DCG=3.5. Идеальный порядок даёт 3+1/log2(3). Это определение с gain 2^grade−1; другой gain convention даёт другие значения. Queries без judged relevant и неполные qrels требуют отдельного правила, чтобы сравниваемые методы имели одинаковую основу.

<!-- /unit:ec6081f8ef00a9e3c7 -->

<!-- unit:63183fdd9fe1bd1ceb -->
<a id="u-63183fdd9fe1bd1ceb"></a>

## Benchmarks и uncertainty

<!-- /unit:63183fdd9fe1bd1ceb -->

<!-- unit:6af3be661d73870758 -->
<a id="u-6af3be661d73870758"></a>

[BEIR](https://arxiv.org/abs/2104.08663) предлагает heterogeneous retrieval tasks, [MTEB](https://arxiv.org/abs/2210.07316) — разнообразие embedding tasks, [MMTEB](https://arxiv.org/abs/2502.13595) — multilingual expansion. Средний leaderboard score относится к versions/subsets/protocol, не универсальному deployment ranking.

<!-- /unit:6af3be661d73870758 -->

<!-- unit:349ff4aaf7c33bcdca -->
<a id="u-349ff4aaf7c33bcdca"></a>

Для честного сравнения фиксируют dataset revisions, split, encoder/pooling/prefixes и search budget. Парное сравнение на одних queries учитывает общую difficulty. Bootstrap resamples query/document groups, когда именно они независимы; correlated token rows одного document не создают тысячи independent trials.

<!-- /unit:349ff4aaf7c33bcdca -->

<!-- unit:dbfe096ff6c3f3936c -->
<a id="u-dbfe096ff6c3f3936c"></a>

Seeds отражают training variation, dataset intervals — sampling uncertainty. Report effect size, slices и cost. Выбор maximum из множества configurations/seed по test делает test development set. Небольшой средний gain не отменяет серьёзную regression критичного rare slice.

<!-- /unit:dbfe096ff6c3f3936c -->

<!-- unit:f3077c080242ae6f81 -->
<a id="u-f3077c080242ae6f81"></a>

### Что именно должно оставаться парным

<!-- /unit:f3077c080242ae6f81 -->

<!-- unit:b39c81f5c530ad6fac -->
<a id="u-b39c81f5c530ad6fac"></a>

Если сравниваются encoders A и B, каждый query должен обрабатываться обоими по одному protocol. Difficulty конкретного query тогда влияет на обе оценки. Для bootstrap выбирают query indices с возвращением и используют **те же выбранные indices** для A и B. Независимо перемешать результаты двух методов означало бы потерять парную структуру сравнения.

<!-- /unit:b39c81f5c530ad6fac -->

<!-- unit:040cbfa7e7d9c8e1f9 -->
<a id="u-040cbfa7e7d9c8e1f9"></a>

Если один исходный документ породил сто похожих chunks, случайное resampling chunks не делает их независимыми. Для оценки переноса на новые документы resample выполняют по исходным группам. Аналогично для диалогов и переводов. Единица uncertainty следует единице случайности, которую предполагается обобщать.

<!-- /unit:040cbfa7e7d9c8e1f9 -->

<!-- unit:5683492a7599fadca4 -->
<a id="u-5683492a7599fadca4"></a>

Несколько seeds показывают вариативность training pipeline, но не заменяют разнообразие test cases. Один seed на большом test и десять seeds на пяти почти одинаковых примерах отвечают на разные вопросы. Report содержит разницу методов, её вариативность, важные slices и цену; значение «лучший seed» само по себе не является воспроизводимым эффектом.

<!-- /unit:5683492a7599fadca4 -->

<!-- unit:f9ec9191df741e0c58 -->
<a id="u-f9ec9191df741e0c58"></a>

### Benchmark average — результат конкретного набора задач

<!-- /unit:f9ec9191df741e0c58 -->

<!-- unit:401c784bf3db6f43a1 -->
<a id="u-401c784bf3db6f43a1"></a>

Если одна версия набора включает больше English classification, а другая — multilingual retrieval, одинаковое имя агрегата ещё не делает scores сопоставимыми. Сохраняют dataset revisions, task weights, corpus, query preprocessing, pooling, search parameters и фильтры. При сравнении нового encoder с другим corpus невозможно отделить качество representation от доступности документов.

<!-- /unit:401c784bf3db6f43a1 -->

<!-- unit:4dbaf728debc13c61c -->
<a id="u-4dbaf728debc13c61c"></a>

Многократный выбор по final test переносит в решение информацию о test, даже если его labels не использовались в backprop. Validation служит для выбора architecture/thresholds/config, test — для независимой оценки уже выбранного pipeline. Для редких критичных случаев лучше заранее объявить отдельный slice, чем объяснять его падение после усреднения.

<!-- /unit:4dbaf728debc13c61c -->

<!-- unit:b6fe31567cab6d5cd0 -->
<a id="u-b6fe31567cab6d5cd0"></a>

## Calibration и отказ

<!-- /unit:b6fe31567cab6d5cd0 -->

<!-- unit:8923e2a5a31c57bd7a -->
<a id="u-8923e2a5a31c57bd7a"></a>

Temperature scaling classifier logits использует $p_T=\operatorname{softmax}(z/T)$, $T>0$, fit по calibration NLL. Положительный scalar T сохраняет argmax и ranking classes данного example; он не исправляет wrong class ordering или missing evidence. Для independent labels protocol calibration может отличаться.

<!-- /unit:8923e2a5a31c57bd7a -->

<!-- unit:c1926f3294ebf1e1e8 -->
<a id="u-c1926f3294ebf1e1e8"></a>

Brier/NLL и reliability bins проверяют probabilities, а не только accuracy. Softmax confidence не universal OOD detector: unfamiliar input может получить confident prediction. Нужны held-out unknown/domain/time cases, а не только in-distribution histogram.

<!-- /unit:c1926f3294ebf1e1e8 -->

<!-- unit:157863a659560bf079 -->
<a id="u-157863a659560bf079"></a>

Selective prediction принимает set $\mathcal A_\tau=\{i:c_i\ge\tau\}$:

<!-- /unit:157863a659560bf079 -->

<!-- unit:35a65f9cd055d5c067 -->
<a id="u-35a65f9cd055d5c067"></a>

$$
\operatorname{Coverage}(\tau)=\frac{|\mathcal A_\tau|}n,
\qquad
\operatorname{Risk}(\tau)=
\frac{\sum_{i\in\mathcal A_\tau}\mathbf1[\widehat y_i\ne y_i]}{|\mathcal A_\tau|}.
$$

<!-- /unit:35a65f9cd055d5c067 -->

<!-- unit:2827ce622d244449c5 -->
<a id="u-2827ce622d244449c5"></a>

При пустом set risk не определён. Если из 100 inputs принято 60 и в них 3 ошибки, coverage 0.6, risk 0.05. Отказ от остальных 40 требует operational route: review, additional retrieval или явный no-answer. Нельзя сообщать только 95% accepted accuracy без coverage.

<!-- /unit:2827ce622d244449c5 -->

<!-- unit:6eb59937234929f0e7 -->
<a id="u-6eb59937234929f0e7"></a>

### Две модели с одинаковыми решениями могут иметь разные probabilities

<!-- /unit:6eb59937234929f0e7 -->

<!-- unit:80c1e7b7bd5e91777d -->
<a id="u-80c1e7b7bd5e91777d"></a>

Возьмём два бинарных примера: правильные labels 1 и 0, но модель в обоих выдаёт p(class1)=0.9. Accuracy=0.5. Средняя NLL равна $-(\log0.9+\log0.1)/2\approx1.203973$, binary Brier — $(0.1^2+0.9^2)/2=0.41$. Если softened probabilities обе равны 0.6, решения и accuracy прежние, но NLL≈0.713558, Brier=0.26.

<!-- /unit:80c1e7b7bd5e91777d -->

<!-- unit:5d619562122dfedfcf -->
<a id="u-5d619562122dfedfcf"></a>

Это пример, где снижение уверенности полезно. Если все уверенные предсказания были правильными, то такое же смягчение может ухудшить NLL. Поэтому T подбирают на отдельной calibration части, а не объявляют «чем мягче, тем лучше». Temperature scaling меняет шкалу probabilities, сохраняя argmax классов каждого примера; исходный метод обсуждается в [работе о calibration](https://arxiv.org/abs/1706.04599).

<!-- /unit:5d619562122dfedfcf -->

<!-- unit:260c856d79a6c0e730 -->
<a id="u-260c856d79a6c0e730"></a>

### Внутрипримерный ranking и порядок confidence разных примеров

<!-- /unit:260c856d79a6c0e730 -->

<!-- unit:de82c8648ba92b19b7 -->
<a id="u-de82c8648ba92b19b7"></a>

Для T>0 деление logits на T сохраняет их порядок **в одном примере**. Но порядок max-softmax confidence разных multiclass примеров может измениться. Возьмём A=(4,0,0), B=(3,2.9,−10). При T=1 confidence A выше B; при T=10 confidence B выше A, хотя первый класс остаётся argmax в обоих. Распределение остальных logits влияет на знаменатель.

<!-- /unit:de82c8648ba92b19b7 -->

<!-- unit:88f066a983dd4eee4d -->
<a id="u-88f066a983dd4eee4d"></a>

Это важно для abstention: новый T способен изменить не только численное значение порога, но и то, какие examples попадут в accepted set. После calibration coverage-risk проверяется заново. Общий scalar T не обучает другой class boundary, не возвращает обрезанный evidence и не гарантирует корректную вероятность на новом domain.

<!-- /unit:88f066a983dd4eee4d -->

<!-- unit:0d790b1e210a3c0771 -->
<a id="u-0d790b1e210a3c0771"></a>

### Reliability и OOD требуют labels или ожидаемого outcome

<!-- /unit:0d790b1e210a3c0771 -->

<!-- unit:238b85a4838891da45 -->
<a id="u-238b85a4838891da45"></a>

В reliability bin сравнивают среднюю confidence с фактической долей правильных решений. Малые bins дают шумную оценку, а крупные могут скрыть разные подгруппы. Один низкий calibration error на in-distribution test не доказывает надёжность на неизвестном языке или новом источнике.

<!-- /unit:238b85a4838891da45 -->

<!-- unit:f0f6b69f811a71456f -->
<a id="u-f0f6b69f811a71456f"></a>

Для отказа при no-answer/OOD нужны случаи такого типа в evaluation. Высокая confidence на бессмысленном input — возможное поведение модели, а не логическое противоречие softmax. Risk при пустом accepted set не определён; система с нулём ответов не получает полезную «нулевую ошибку» без отчёта coverage и правила обработки отказов.

<!-- /unit:f0f6b69f811a71456f -->

<!-- unit:4f9b4237bea57b3f79 -->
<a id="u-4f9b4237bea57b3f79"></a>

## Robustness tests с ожидаемым поведением

<!-- /unit:4f9b4237bea57b3f79 -->

<!-- unit:8884ad52ae2617cfc2 -->
<a id="u-8884ad52ae2617cfc2"></a>

Typos, OCR noise, code switching, длинный context, числа, quotations и negation проверяют разные risks. Harmless spelling change должен часто сохранять label, смена «есть» на «нет» может обязана его изменить. Blanket invariance ко всем perturbations неверна.

<!-- /unit:8884ad52ae2617cfc2 -->

<!-- unit:b70b729e9862c4fbb5 -->
<a id="u-b70b729e9862c4fbb5"></a>

Test cases фиксируют expected outcome и reason. Если annotation policy неоднозначна, сначала разрешается label issue, а не подбор model confidence. Изменение time/domain distribution, missing classes и new entities оценивается отдельно; one temperature не устраняет произвольный concept drift.

<!-- /unit:b70b729e9862c4fbb5 -->

<!-- unit:52a98a1fd6ab9116c2 -->
<a id="u-52a98a1fd6ab9116c2"></a>

### Инвариантность нужна только там, где target действительно сохраняется

<!-- /unit:52a98a1fd6ab9116c2 -->

<!-- unit:2566149e95b9135ba9 -->
<a id="u-2566149e95b9135ba9"></a>

Для темы обращения небольшая опечатка может не менять label. Но замена одного символа в коде ошибки A12→A13 может изменить нужный документ, хотя с точки зрения edit distance это похожее возмущение. Удаление «не» меняет отрицание; перестановка аргументов в relation extraction может менять roles. Нельзя требовать одинаковый prediction после любого малого редактирования.

<!-- /unit:2566149e95b9135ba9 -->

<!-- unit:1f9cf6849f672a258f -->
<a id="u-1f9cf6849f672a258f"></a>

Каждый test case задаёт original input, преобразование, ожидаемую связь ответов и причину. Для label-preserving paraphrase сравнивают стабильность, для meaning-changing pair — чувствительность. Если ожидаемый label неоднозначен, это сначала проблема annotation policy. Confidence не должна определять истинную разметку задним числом.

<!-- /unit:1f9cf6849f672a258f -->

<!-- unit:fa21e4df0d1f80b999 -->
<a id="u-fa21e4df0d1f80b999"></a>

К изменениям языка, источника и времени применяют тот же принцип. Сдвиг частот labels отличается от нового определения labels; оба могут ухудшить quality, но требуют разных исследований. Один temperature parameter не исправляет произвольную смену target function.

<!-- /unit:fa21e4df0d1f80b999 -->

<!-- unit:9f97c90c1f123ee3d5 -->
<a id="u-9f97c90c1f123ee3d5"></a>

## Attribution, probes и causal diagnosis

<!-- /unit:9f97c90c1f123ee3d5 -->

<!-- unit:85f4c49b2c89d25bf8 -->
<a id="u-85f4c49b2c89d25bf8"></a>

Gradient/saliency показывает local sensitivity, attention inspection — внутренние weights выбранного layer/head. Ни одно не является готовым человеческим объяснением causal decision. Probe может декодировать feature hidden state, хотя main head его не использует.

<!-- /unit:85f4c49b2c89d25bf8 -->

<!-- unit:6b91ade924f3c59205 -->
<a id="u-6b91ade924f3c59205"></a>

Integrated gradients для baseline $x'$ и input $x$ задаёт attribution:

<!-- /unit:6b91ade924f3c59205 -->

<!-- unit:4298b1f1eb5eb71d24 -->
<a id="u-4298b1f1eb5eb71d24"></a>

$$
\operatorname{IG}_i(x)=(x_i-x_i')
\int_0^1\frac{\partial F(x'+\alpha(x-x'))}{\partial x_i}\,d\alpha.
$$

<!-- /unit:4298b1f1eb5eb71d24 -->

<!-- unit:cf44a6a81cfa15a932 -->
<a id="u-cf44a6a81cfa15a932"></a>

Для text interpolation обычно идёт в embedding space, а не между integer IDs; baseline/path определяют interpretation. Нужны controls и actual perturbations: source metadata removal, matched negation pairs, label-preserving paraphrase. Decodability, sensitivity и causal use различаются. Первичный метод: [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365).

<!-- /unit:cf44a6a81cfa15a932 -->

<!-- unit:a144e11a1829cfa543 -->
<a id="u-a144e11a1829cfa543"></a>

Shortcut может быть длиной, source или entity name. Высокое качество random split тогда не переносится на new source. Для проверки гипотезы «использует отрицание» сравнивают minimally differing pairs и контроль без semantic change, а не только наличие token «не» среди ярких saliency features.

<!-- /unit:a144e11a1829cfa543 -->

<!-- unit:2248954926ceda56aa -->
<a id="u-2248954926ceda56aa"></a>

### Endpoint gradient, integrated gradients и вмешательство

<!-- /unit:2248954926ceda56aa -->

<!-- unit:957371d05f35556606 -->
<a id="u-957371d05f35556606"></a>

Для учебной функции $F(x_1,x_2)=x_1x_2$, baseline (0,0) и input (2,3) градиент в input равен (3,2). Простое gradient×input даёт (6,6), сумма 12. Integrated gradients по прямому пути даёт (3,3), сумма 6=F(2,3)−F(0,0). Производная по x1 вдоль пути равна 3α: интеграл от 0 до 1 равен 1.5, затем умножается на разность x1=2. Для второй координаты получается тот же вклад 3.

<!-- /unit:957371d05f35556606 -->

<!-- unit:d3d1501900b211a03d -->
<a id="u-d3d1501900b211a03d"></a>

Пример показывает, почему endpoint sensitivity и path attribution — разные объекты. У IG при нужных условиях есть completeness относительно baseline/path; численная интеграция приближает её. Это не означает, что каждый вклад является доказанной человеческой причиной ответа или что chosen baseline соответствует отсутствию смысла.

<!-- /unit:d3d1501900b211a03d -->

<!-- unit:dbf0ac8128134a8662 -->
<a id="u-dbf0ac8128134a8662"></a>

У текста integer token IDs нельзя линейно интерполировать как признаки с осмысленной геометрией. Обычно интерполируют embeddings. Нулевой embedding или другой baseline может не соответствовать реальному предложению. Удаление слова также меняет tokenization и контекст; это другое вмешательство, которое нужно оценивать отдельно.

<!-- /unit:dbf0ac8128134a8662 -->

<!-- unit:f0b7b5052f40fe9342 -->
<a id="u-f0b7b5052f40fe9342"></a>

### Что доказывает probe и чего не доказывает

<!-- /unit:f0b7b5052f40fe9342 -->

<!-- unit:88d08905c6fa2c8cd9 -->
<a id="u-88d08905c6fa2c8cd9"></a>

Если отдельный classifier может восстановить часть речи из hidden states, эта информация доступна выбранному decoder/probe. Но исходная task head могла её не использовать. Если attention map выделяет слово, оно участвовало в одной mixture с выбранными V, однако конечный ответ зависит от residual, других heads и последующих слоёв. Красивое выделение не заменяет причинную проверку.

<!-- /unit:88d08905c6fa2c8cd9 -->

<!-- unit:e87bc6b209115a6c87 -->
<a id="u-e87bc6b209115a6c87"></a>

Гипотезу о shortcut проверяют контролируемым изменением: убрать source marker при сохранённом содержании, заменить имя на другое с тем же label, сравнить отрицание с контрольной опечаткой. Важно не разрушить target вместе с якобы лишним признаком. После вмешательства смотрят actual task output и качество, а не только новую saliency картинку.

<!-- /unit:e87bc6b209115a6c87 -->

<!-- unit:dcec97af763dde9da1 -->
<a id="u-dcec97af763dde9da1"></a>

## Data access и privacy в прикладном contract

<!-- /unit:dcec97af763dde9da1 -->

<!-- unit:fb16e871dd3db3e26e -->
<a id="u-fb16e871dd3db3e26e"></a>

Retriever и PII extractor тоже обрабатывают adversarial/untrusted input. Embedded instructions или special-token-like strings требуют task tests; encoder-only architecture не даёт абсолютной защищённости любой обвязки. ACL/filter применяется до выдачи недоступных results по contract системы.

<!-- /unit:fb16e871dd3db3e26e -->

<!-- unit:9d5b8b764f9d1c0596 -->
<a id="u-9d5b8b764f9d1c0596"></a>

PII head может пропустить entity. Удаление известных имён не доказывает полной anonymization: косвенные identifiers и context остаются. Для реального использования задаются retention/logging/masking policies и audit соответствующих failures. Это конкретные properties pipeline, а не автоматически получаемое качество checkpoint.

<!-- /unit:9d5b8b764f9d1c0596 -->

<!-- unit:95d257e1d431b9d998 -->
<a id="u-95d257e1d431b9d998"></a>

## Контроль и практика

<!-- /unit:95d257e1d431b9d998 -->

<!-- unit:27cad392679244670d -->
<a id="u-27cad392679244670d"></a>

1. Почему MRR может быть 1 при nonideal ranking? **Ответ:** он учитывает только first relevant rank.
2. Исправляет ли T scaling wrong argmax? **Ответ:** positive global temperature сохраняет argmax.
3. Почему probe не доказывает causal use? **Ответ:** информация может присутствовать в state, но не влиять на actual head decision.
4. Почему accepted accuracy без coverage неполна? **Ответ:** model могла отвергнуть большую часть трудных inputs.

<!-- /unit:27cad392679244670d -->

<!-- unit:4188ab54f29ff95ea7 -->
<a id="u-4188ab54f29ff95ea7"></a>

5. Может ли T изменить порядок confidence разных multiclass примеров? **Ответ:** да. Сохраняется порядок классов внутри примера; max-softmax разных примеров зависит и от других logits.
6. Равны ли gradient×input и IG? **Ответ:** в общем случае нет. Для F=x1x2 и input(2,3) первый даёт(6,6), IG от нулевого baseline —(3,3).
7. Почему bootstrap выполняют по исходным query/document groups? **Ответ:** множество связанных chunks не образует столько же независимых trials; resampling должен сохранять единицу зависимости.
8. Исправляет ли calibration обрезанный evidence? **Ответ:** нет. Она меняет шкалу probabilities при прежних представлениях и информации входа.

<!-- /unit:4188ab54f29ff95ea7 -->

<!-- unit:fb33be9a788aff9bdf -->
<a id="u-fb33be9a788aff9bdf"></a>

**Практика:** посчитайте MRR/nDCG двух rankings, постройте coverage-risk для шести scores. Предложите три robustness pairs с expected labels и probe control. Подготовьте paired evaluation protocol с независимой calibration частью и объяснением limits, без универсального вывода из одного benchmark.

<!-- /unit:fb33be9a788aff9bdf -->

## Проектная практика ветки

<!-- unit:9f754a1c2080e50fb7 -->
<a id="u-9f754a1c2080e50fb7"></a>

## Контрольные задачи для углублённого понимания

<!-- /unit:9f754a1c2080e50fb7 -->

<!-- unit:3f5cd8bd37a441087c -->
<a id="u-3f5cd8bd37a441087c"></a>

1. Вывести shapes encoder слоя при B=2, L=7, D=24, H=4; объяснить разницу padding и causal mask.

<!-- /unit:3f5cd8bd37a441087c -->

<!-- unit:2ac740e849e8e18685 -->
<a id="u-2ac740e849e8e18685"></a>

2. Для vocabulary size 5 и двух MLM positions посчитать CE/gradient; показать loss на unchanged corruption target.

<!-- /unit:2ac740e849e8e18685 -->

<!-- unit:59adfffde1dda3b581 -->
<a id="u-59adfffde1dda3b581"></a>

3. Согласовать classification head, labels и loss для single-label и multi-label задачи.

<!-- /unit:59adfffde1dda3b581 -->

<!-- unit:d405251fb387e3874a -->
<a id="u-d405251fb387e3874a"></a>

4. Превратить word BIO labels в subword targets и восстановить исходные char spans.

<!-- /unit:d405251fb387e3874a -->

<!-- unit:0eb7de2be06a9d6199 -->
<a id="u-0eb7de2be06a9d6199"></a>

5. Выбрать валидный QA span при конфликтующих start/end maxima и unanswerable threshold.

<!-- /unit:0eb7de2be06a9d6199 -->

<!-- unit:0c39f34180b0ff9d45 -->
<a id="u-0c39f34180b0ff9d45"></a>

6. Посчитать одну строку contrastive loss, затем повторить с false negative; объяснить изменение сигнала.

<!-- /unit:0c39f34180b0ff9d45 -->

<!-- unit:75b96e2300b2191248 -->
<a id="u-75b96e2300b2191248"></a>

7. Сравнить mean-vector similarity, MaxSim и pairwise cross-encoder по вычислениям/storage.

<!-- /unit:75b96e2300b2191248 -->

<!-- unit:5e6f633bba52d912e5 -->
<a id="u-5e6f633bba52d912e5"></a>

8. Посчитать MRR/nDCG и показать, почему улучшение ANN recall не обязательно повышает relevance recall.

<!-- /unit:5e6f633bba52d912e5 -->

<!-- unit:24d358ff27547e6c98 -->
<a id="u-24d358ff27547e6c98"></a>

9. Обосновать encoder shortlist для русского short-text NER и длинного multilingual retrieval.

<!-- /unit:24d358ff27547e6c98 -->

<!-- unit:d310280036977bc34b -->
<a id="u-d310280036977bc34b"></a>

10. Спроектировать reindex/retraining procedure после смены tokenizer/pooling и указать критерии rollback.

<!-- /unit:d310280036977bc34b -->

## Результат главы и переход дальше

Ответьте на вопрос: **Как оценить encoder на правильной единице задачи и проверить отказ?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

Следующие темы: [Сжатие, сервис encoder и жизненный цикл индекса](<bert-operations.md>).

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
