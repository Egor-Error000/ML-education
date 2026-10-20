# Г. Когда representations категорий обучаются вместе с задачей

[Оглавление](README.md). Пункт 4 и расчёт параметров из практики. **Основа:** S2:1621–1708 и 1830–1880 объясняет lookup token embeddings и переход к contextual states; перенос принципа на табличные entities и transfer между tasks — дополнение.

**Уровень:** основа ветки deep tabular. **Предпосылки:** F3, neural embeddings, backprop; D2. **Используют:** F5/F8, clustering, retrieval и downstream tasks.

## 1. Lookup вместо огромного one-hot

Для $K$ categories embedding matrix $E\in\mathbb R^{K\times d}$ возвращает row $e_c$. Математически lookup эквивалентен $E^\top u_c$ для one-hot column $u_c$, но реализация не обязана создавать весь one-hot vector. Обучаются $Kd$ parameters; для $K=10\,000,d=16$ это 160 000, около 0.64 MB float32 weights без optimizer states.

Несколько fields обычно имеют собственные vocabularies и matrices: ID города 7 и ID телефона 7 не один entity. Их vectors concatenated или взаимодействуют по architecture. Dimensions — hyperparameters: большее $d$ повышает capacity, но редкие categories не получают от этого больше observations.

## 2. Как приходит gradient

Пусть scalar predictor $\widehat y=v^\top e_c+b$ и loss $L=\tfrac12(\widehat y-y)^2$. Тогда:

$$
\frac{\partial L}{\partial e_c}=(\widehat y-y)v,
\qquad
\frac{\partial L}{\partial v}=(\widehat y-y)e_c.
$$

В этом example только выбранная row получает data gradient; при batch contributions суммируются по occurrences. Dense regularization/optimizer details могут обновлять и другие parameters, поэтому «невыбранные rows вообще никогда не меняются» слишком сильное утверждение.

Числа: $e_A=(1,2),v=(0.5,-1),b=0,y=0$. Prediction $-1.5$, gradient embedding $(-0.75,1.5)$. При LR 0.1 и обновлении только embedding получаем $(1.075,1.85)$, prediction $-1.3125$, loss уменьшается с $1.125$ до $0.861328125$. Если одновременно обновить $v$, результат другой; в ручном примере это явно зафиксировано.

## 3. Что означает близость

Entity embeddings выучивают полезное для objective, а не универсальную semantic geography. Если target — CTR, города могут сближаться по advertising behaviour, а не по расстоянию на карте. В другой задаче тот же geometry может быть неуместен.

Representation не единственно идентифицируема: для invertible transform $A$ можно заменить embeddings на $Ae_c$, а downstream linear weights на $A^{-\top}v$ и сохранить predictions. Euclidean distances при произвольном transform меняются. Поэтому интерпретировать absolute geometry нужно осторожно, с учетом architecture и regularization.

При embedding+linear output без нелинейных interactions expressive power одного field может быть такой же, как отдельных category coefficients. Польза compact vector проявляется через shared factors, interactions, multitask transfer и data structure, но не гарантируется самим словом «embedding».

## 4. Rare/unknown и regularization

Rare categories получают мало gradient updates. Помогают rare grouping, smaller dimension, weight decay, norm constraints или shared metadata features, если это подтверждается validation. Unknown row должна иметь training signal, например через контролируемое category dropout/rare mapping. Отдельная случайно инициализированная unseen row не знает значение нового entity.

Словарь сохраняется вместе с weights. Перестановка IDs при loading делает lookup wrong без shape error. Missing, PAD и UNK могут требовать разные rows; padding row часто имеет special update rules, которые надо проверить в implementation.

## 5. Transfer и extraction

Embeddings можно freeze и передать другому model, включая boosting. Но если они обучены по labels всего dataset до CV, extracted features уже содержат test information. Внешняя validation должна включать training encoder на своей training части либо использовать genuinely independent pretrained model.

При переносе между tasks нужны стабильные entity IDs, совместимые vocabularies, доступность источников и проверка negative transfer. Source task мог выучить bias старого периода; fine-tuning на новой задаче может забыть полезное. Freeze против update сравниваются на одном split, с отдельным анализом редких и новых entities.

## 6. Контроль и практика

1. Почему lookup не требует dense one-hot? **Ответ:** он сразу выбирает row, хотя математически эквивалентен соответствующему умножению.
2. Является ли embedding distance объективной semantic truth? **Ответ:** нет; она зависит от objective, parametrization и training data.
3. Можно ли сначала обучить embeddings на всех labels, затем CV бустинга? **Ответ:** это утечка; encoder является частью supervised pipeline.

**Практика:** повторите gradient step, затем обновите и embedding, и head одновременно, используя исходные gradients. Посчитайте weights плюс optimizer states для нескольких dimensions. Составьте transfer protocol с независимым source dataset и stable ID mapping.


<!-- source-depth-categorical-Г -->

## 7. От числового ID к обучаемой функции

### 7.1. Что переносится из исходного объяснения LLM

В S2:1621–1708 разобрана микро-модель со словарём 1000, двумя tokens и hidden size 4: IDs 42 и 105 выбирают rows (0.1,−0.5,0.8,0.2) и (0.3,0.1,−0.2,0.9). Получается matrix 2×4. Этот lookup точно так же работает для двух категориальных IDs, но сам по себе ещё не смешивает информацию между ними. Табличный field «город» не превращается в token sequence без отдельного решения об architecture.

После Transformer block в исходном чате те же позиции уже содержат contextual states. Это различие важно при extraction: row исходной embedding matrix — статическая функция ID, а hidden state зависит ещё от остальных inputs, masks и слоя. Подставленные в чате числа после block иллюстрируют изменение представления; они не являются вычисленным forward без заданных weights. Здесь переносится механизм lookup и backprop, а Transformer mechanisms и training подробно раскрыты в курсе основ LLM.

Для batch из B строк одного field IDs имеют shape B, lookup — B×d, matrix E — K×d. Для F fields с dimensions $d_1,\ldots,d_F$ concatenation имеет shape $B\times\sum_f d_f$. Padding, masks и pooling нужны только если присутствуют наборы или sequences; один scalar category не нуждается в искусственном PAD token на каждом примере.

### 7.2. Почему d=1 иногда уже достаточно

Для одного field и scalar linear head prediction равен $v^\top e_c+b$. При d=1 и ненулевом v можно выразить произвольный scalar coefficient каждой category: выбрать $e_c=(a_c-b)/v$. Поэтому замена K one-hot coefficients на K scalar embeddings не является сама по себе сильным ограничением класса scalar functions. Важны regularization, sharing и структура последующих layers.

При M outputs с общей линейной head H размером d×M category effects образуют matrix $W=EH$ ранга не выше d. Здесь малое d действительно связывает разные tasks/outputs. Учебная matrix effects для трёх categories и двух outputs с rows (1,0), (0,1), (1,1) имеет rank 2; factorization с d=1 не воспроизведёт её точно, а d=2 может. Отдельный output bias меняет формулировку rank: ограничена matrix effects после вычитания этого bias, а не обязательно сырые predictions.

Если далее идёт nonlinear MLP, low-rank statement о линейной W уже не описывает весь predictor. Concatenation embeddings с одной linear head остаётся additive по fields. Для двух binary fields XOR невозможно выразить одной thresholded additive score: требования к четырём outcomes противоречат суммированию двух effects. Нужны interactions, нелинейность или другой подход; сам lookup не добавляет interaction term.

## 8. Полный backward и один согласованный optimizer step

### 8.1. Gradient — это сумма по обращениям к row

Пусть row $e_c$ трактуется как column vector, а upstream gradient для lookup occurrence i равен $g_i$. Тогда

$$
\nabla_{e_c}L=\sum_{i:c_i=c}g_i.
$$

Это scatter-add: несколько occurrences одного ID пишут в одну row gradient. Если batch loss усреднён по B observations, upstream gradients уже содержат factor 1/B; повторно усреднять их ещё раз нельзя. При padding/masked examples denominator определяется реальным loss, а не просто числом positions в padded tensor.

Для predictor раздела 2 возьмём batch A,A,B, начальные $e_A=(1,2)$, $e_B=(-1,0)$, $v=(0.5,-1)$ и labels 0,1,−1. Predictions −1.5,−1.5,−0.5, residuals −1.5,−2.5,0.5. При mean half-squared loss gradients равны

$$
\nabla_{e_A}L=(-2/3,4/3),\qquad
\nabla_{e_B}L=(1/12,-1/6),
$$
$$
\nabla_vL=(-1.5,-8/3),\qquad
\nabla_bL=-7/6.
$$

Row A получает сумму двух разных ошибок, а не gradient только последней строки. Если framework умеет делить gradients на frequency occurrences, это отдельное правило оптимизации и другая величина update; оно не тождественно gradient исходного mean loss. Такой режим нельзя включить молча и продолжить ручной вывод как будто ничего не изменилось.

### 8.2. Совместный update embedding и head

Вернёмся к одному примеру $e=(1,2),v=(0.5,-1),b=0,y=0$. Все gradients вычисляются по одному исходному состоянию: $\nabla_e=(-0.75,1.5)$, $\nabla_v=(-1.5,-3)$, $\nabla_b=-1.5$. При LR 0.1 совместный SGD step даёт e'=(1.075,1.85), v'=(0.65,−0.7), b'=0.15. Prediction становится −0.44625, half-squared loss 0.09956953125.

Это другой результат, чем обновление только embedding из раздела 2. Если сначала поменять embedding, затем пересчитать gradient head, получится последовательная block update scheme. Она может быть осмысленным отдельным алгоритмом, но не тем же simultaneous SGD step. Для конечного LR decrease не гарантирован произвольной nonlinear model; хороший результат этого маленького примера не превращается в universal theorem.

### 8.3. Нулевой data gradient не всегда означает неподвижную row

Когда ID не встречался в batch, его lookup contribution к gradient равен нулю. Однако weight decay, momentum от прежних batches, optimizer states, shared/tied weights и операции norm constraint могут менять row и без нового lookup occurrence. В dense Adam старые moments способны дать update при текущем нулевом gradient; sparse optimizer может иметь другой контракт того, какие rows вообще посещаются.

PAD row, отсутствующая из loss, и UNK row выполняют разные функции. PAD нужна для формы tensor и mask; UNK должна давать полезный fallback для неизвестного ID. Запрет lookup gradient для padding row не является доказательством, что любая внешняя операция над weights её сохранит. Замораживание всей matrix, mask gradients отдельных rows и исключение parameters из optimizer тоже не эквивалентны. Проверяют фактические updates и сохранённые optimizer states выбранного training code.

## 9. Память, частота обновлений и cold start

### 9.1. Обучаемая таблица — не сжатое хранение one-hot input

Для K=10 000,d=16 имеем 160 000 parameters и 640 000 bytes float32 weights. Dense gradient и два float32 Adam moments добавят ещё 1 920 000 bytes; вместе получится 2 560 000 bytes, без activations, allocator overhead и других layers. При mixed precision набор master weights и gradient dtype меняет расчёт, поэтому одного размера весов недостаточно для training memory.

One-hot input с одной активной category на observation можно хранить sparse; scalar linear model по нему имеет лишь K coefficients. Embedding matrix с d=16 содержит в 16 раз больше coefficients, чем такой head, хотя выдаёт компактный vector на example. При K=1 000 000,d=64 только float32 table занимает 256 MB в decimal units; weights+gradient+два moments — 1.024 GB. «Компактное представление» относится к выбранной операции и downstream model, а не обещает меньшую память любого решения.

Batch B×d activations зависят от числа lookup occurrences, таблица — от Kd. Sparse gradients могут сократить работу по touched rows, но не делают сами weights бесплатными. Sharding или offloading переносят часть затрат в communication и latency; это инженерная часть большой системы, а не следствие математического равенства lookup и one-hot multiplication.

### 9.2. Размерность не заменяет наблюдения редкого ID

Если category встречается r раз, её row получает прямой data signal примерно на этих r occurrences за один проход, с поправкой на sampling. Увеличение d с 8 до 128 создаёт больше свободных coordinates, а не больше evidence. При частоте p вероятность не встретить ID в B независимых draws равна $(1-p)^B$. Для p=0.0001 и B=256 она около 0.9747: большинство таких batches не несут для этого ID прямой lookup gradient.

Oversampling rare categories меняет частоту updates и при некорректном weighting может изменить objective population. Weight decay к zero — простой prior, но zero vector не обязательно разумный mean behaviour. Shared metadata может дать более полезную опору: $e_c=g(x_c)+r_c$, где $g$ строит vector из доступных свойств entity, а residual $r_c$ выучивает ID-specific correction. Для нового ID можно задать residual zero и использовать metadata path, если metadata присутствует и её значения доступны в момент prediction.

### 9.3. Unknown row нужно чему-то научить

При category dropout во время training часть известных IDs заменяется на UNK. Тогда UNK row обучается на распределении скрытых examples, а не получает «смысл всех будущих entities». Если masking uniform по строкам, frequent IDs доминируют в этом signal; если uniform по categories, objective отличается. Rate, train-only randomness и выбранная схема сравниваются на validation с отдельно определёнными known/rare/new slices.

Для новой category со stable metadata возможен inductive fallback; для чистого arbitrary ID без metadata индивидуальную информацию получить неоткуда. Нельзя создавать новую случайную row на каждом inference и ожидать осмысленную персонализацию. Добавление rows в vocabulary требует versioned mapping, initialization policy и, если training продолжается, расширения optimizer states. Перестановка старых rows при расширении разрушает уже выученное соответствие.

## 10. Геометрия не идентифицируется одним prediction loss

Для column vector e и linear head v преобразование $e'=Ae$, $v'=A^{-\top}v$ сохраняет scalar prediction. Возьмём e1=(1,0), e2=(0,1), A=diag(10,1). Distance между embeddings меняется с $\sqrt{2}$ до $\sqrt{101}$, но после соответствующей замены head predictions сохраняются. Это точный контрпример интерпретации расстояния как автоматически установленного свойства entities.

Cosine тоже не сохраняется при произвольном invertible A. Для u=(1,1), v=(1,−1) исходный cosine 0, а после того же A — 99/101≈0.9802. Orthogonal rotation сохраняет distances/cosine, но не произвольный rescaling осей. Norm constraints, normalization и contrastive objectives ограничивают допустимую геометрию; они являются частью метода, а не бесплатным свойством lookup.

Практически геометрия всё же может оказаться полезной: похожие tasks и regularization подталкивают sharing, nearest neighbours могут иметь понятную структуру. Но проверка требует конкретного downstream criterion или retrieval task. Выбор красивой 2D projection не доказывает полезности исходных distances; projection меняет их и допускает визуальные artifacts.

Для двух fields dot-product score $e_u^\top f_i$ задаёт low-rank interaction matrix. Он связывает predictions для ещё не наблюдавшихся pairs через общие factors, но не может персонализировать совсем новую row без signal. Это иной источник inductive bias, чем отдельный scalar main effect для каждого ID. Механика pair interactions продолжена в главе Е.

## 11. Extraction и transfer как часть supervised pipeline

### 11.1. Почему обучение embeddings на train допустимо

Embedding — обычный trainable parameter модели. Его обучение по training labels не требует автоматически исключать каждую строку из собственного neural training gradient: иначе пришлось бы запретить и обычное supervised learning. Честность generalization проверяет внешний validation, labels которого не участвовали в fit таблицы, head или выборе hyperparameters.

Если embedding используется как generated feature для второго estimator, encoder становится ещё одним supervised этапом. Training encoder на всём разрешённом внешнем train и fit второго estimator на его representations может быть определённым pipeline, но второй estimator видит in-sample representations, что допускает memorization и mismatch. Если он должен обучаться на out-of-sample behaviour первой модели, нужен cross-fitting с учётом проблемы coordinate alignment ниже. В любом случае обучение encoder по внешним validation/test labels до оценки второго estimator является leakage.

### 11.2. Почему нельзя просто склеить OOF embedding coordinates

У independently trained folds latent axes могут поворачиваться, менять scale и переставляться. В одном fold category A представлена (1,0), а в другом — (0,1), с согласованными heads и теми же predictions. Склейка этих координат в одну OOF matrix создаёт несогласованный input space для второго estimator. Это отличается от OOF scalar predicted probability с фиксированным смыслом positive class.

Возможны разные явные решения: общий заранее frozen independent encoder; fixed-semantics model outputs вместо latent vectors; отдельный downstream model на каждый encoder с ensemble predictions; согласованная alignment procedure, fit только на разрешённых anchors. Ни один вариант не следует автоматически из слова cross-fitting. Procrustes alignment исправляет orthogonal rotations при подходящих anchors, но не гарантирует согласование произвольных nonlinear representations и не разрешает использовать held-out targets.

### 11.3. Frozen против fine-tuned transfer

Для переноса фиксируют source entity definition, mapping, cutoff и objective. Один и тот же numeric ID в разных systems может относиться к разным entities; одинаковая shape не даёт semantic compatibility. Source labels из будущего target evaluation периода могут нарушать time protocol даже при формально отдельном dataset. Independent pretrained означает независимость относительно заявленного evaluation, а не просто другое имя файла.

Frozen table сохраняет source representation, но target head ограничена доступной информацией. Fine-tuning адаптирует known rows и может ухудшить полезную геометрию или забыть source structure. Сравнение с random initialization и metadata-only baseline помогает отделить полезный transfer от эффекта дополнительной capacity. Known entities и новые entities оценивают отдельно; если target task содержит только знакомые IDs, это ещё не доказательство cold-start качества.

В [работе об entity embeddings](https://arxiv.org/abs/1604.06737) предложено обучать representations категориальных variables и применять их дальше в моделях. Это источник метода, а не гарантия выигрыша на конкретном пользовательском dataset. Для implementation lookup/gradient rules сверяют с [API Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html) установленной версии; числовые примеры этой главы определены формулами, без предположения о library defaults.

**Расширенная практика:** воспроизведите batch scatter-add и simultaneous step; проверьте gradient finite differences; посчитайте training memory; сравните distances до и после invertible transform при неизменных predictions. Для transfer опишите разрешённые source data, axes/mapping contract и known/new evaluation. Факт наличия этого задания не означает, что обучение или benchmark уже выполнены.

## 12. Learned representations для текста, изображений и смешанных объектов

Идея embeddings распространяется за пределы категориального ID: объект $x$ проходит через encoder $f_\theta$ и pooling, после чего получается вектор $z=f_\theta(x)\in\mathbb R^d$. Например, текстовый encoder может объединить token states в один vector, а image encoder — spatial feature map. Геометрия зависит от pretraining objective, pooling, normalization и domain; одинаковая размерность двух моделей не гарантирует совпадающие axes.

Frozen encoder преобразует train/validation/new objects одной фиксированной функцией. Это дешёвый baseline, когда checkpoint независим от evaluation labels, но domain mismatch может ограничить качество. Fine-tuning меняет $\theta$ под target task; он способен адаптировать features и одновременно забыть полезную source structure. Если encoder обучен на labels из всей CV-выборки до split, готовые vectors уже несут test information. Encoder и любые trainable normalization/PCA stages должны находиться внутри соответствующей training fold либо быть независимо предобучены по объявленному протоколу.

Пример: $z_1=(1,1)$ и $z_2=(2,0)$ имеют cosine $1/\sqrt2\approx0.707$, но Euclidean distance $\sqrt2$. После L2-normalization Euclidean distance становится $\sqrt{2(1-0.707)}\approx0.765$. Значит, выбор cosine/Euclidean и normalization — часть feature contract. Проверяйте neighbors по известным labels/slices и downstream task; близость сама по себе не доказывает семантическое объяснение.

Для out-of-sample применения сохраняют model revision, preprocessing, pooling, vector dimension, normalization и правила invalid/missing input. Если после encoder применяется train-fitted PCA или projection, её fit parameters также сериализуются и не пересчитываются на новых objects.

**Схема курса:** raw object → saved preprocessing → frozen или fine-tuned encoder → pooling/normalization → versioned vector → nearest-neighbor или downstream estimator.

**Практика:** на одном split сравните frozen encoder + linear head, fine-tuning и simple feature baseline. Для каждого зафиксируйте training boundary; проверьте neighbor examples, unseen slice и transform parity на новых объектах. Отдельно укажите, какой вывод подтверждает validation metric и чего он не говорит о внутреннем смысле coordinates.
