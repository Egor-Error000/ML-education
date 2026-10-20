# Label smoothing, MixUp, CutMix и аугментации

**Уровень:** общее ядро.

**Предпосылки:** [Обобщение, диагностика и границы теории](<reg-diagnosis.md>)

**Используют:** Выбранная прикладная задача и итоговая проверка результата.

**Учебный вопрос:** Как изменения inputs и targets задают предположения о задаче?





## Карта вычисления и границы этапов

![Label smoothing, MixUp, CutMix и аугментации: входы, преобразования и выходы](<../оформление/course/reg-augmentation.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Сравните формы и смысл каждого перехода с формулами главы. Выход преобразования и доказательство его качества — разные объекты: последняя проверка использует отдельный протокол оценки.

<!-- unit:59a1dc18e0f7548125 -->
<a id="u-59a1dc18e0f7548125"></a>

## Предположения аугментаций

<!-- /unit:59a1dc18e0f7548125 -->

<!-- unit:0ddcec20d9aa7db1c4 -->
<a id="u-0ddcec20d9aa7db1c4"></a>

Обучение с transforms минимизирует риск по выбранному распределению $T$:

<!-- /unit:0ddcec20d9aa7db1c4 -->

<!-- unit:5df14760f207b571ef -->
<a id="u-5df14760f207b571ef"></a>

$$
\mathcal L(\theta)=
\mathbb E_{(x,y),T}[\ell(f_\theta(Tx),T_y(y))].
$$

<!-- /unit:5df14760f207b571ef -->

<!-- unit:9238f7f088535efee8 -->
<a id="u-9238f7f088535efee8"></a>

Для классификации $T_y$ иногда identity, для detection/segmentation — согласованное геометрическое преобразование. Label-preserving transformations должны быть обоснованы задачей. Усиление augmentation меняет objective и difficulty, поэтому при сравнении фиксируются steps, sampling и остальные настройки.

<!-- /unit:9238f7f088535efee8 -->

<!-- unit:702ac9b0bd4b43cd92 -->
<a id="u-702ac9b0bd4b43cd92"></a>

## Label smoothing и soft targets

<!-- /unit:702ac9b0bd4b43cd92 -->

<!-- unit:940d8247eeed4c8898 -->
<a id="u-940d8247eeed4c8898"></a>

Одна convention смешивает hard label с uniform по всем $K$ classes:

<!-- /unit:940d8247eeed4c8898 -->

<!-- unit:4e0ffa4bfe2f5d53a8 -->
<a id="u-4e0ffa4bfe2f5d53a8"></a>

$$
q_k=(1-\epsilon)\mathbf1[k=y]+\frac\epsilon K,
\qquad
L=-\sum_{k=1}^K q_k\log p_k.
$$

<!-- /unit:4e0ffa4bfe2f5d53a8 -->

<!-- unit:1be997635689dae479 -->
<a id="u-1be997635689dae479"></a>

При $K=3,\epsilon=0.1$ target для class 1 равен $(0.9333\ldots,0.0333\ldots,0.0333\ldots)$. Другая convention оставляет true class $1-\epsilon$ и делит $\epsilon$ только между остальными: $(0.9,0.05,0.05)$. Одинаковое epsilon не делает их одинаковыми.

<!-- /unit:1be997635689dae479 -->

<!-- unit:042fbf7d8657b8098d -->
<a id="u-042fbf7d8657b8098d"></a>

Для softmax logits gradient равен $\partial L/\partial z_k=p_k-q_k$. Поэтому smoothing ослабляет стимул бесконечно увеличивать true logit относительно остальных. Это может уменьшить overconfidence, но не гарантирует лучшую calibration или ranking в каждой задаче. Неправильно подобранный smoothing способен мешать learning редких/сложных classes.

<!-- /unit:042fbf7d8657b8098d -->

<!-- unit:4c94d92771f958c9c2 -->
<a id="u-4c94d92771f958c9c2"></a>

Uniform prior тоже не всегда соответствует task structure: соседние classes и заведомо невозможные alternatives различаются. Более сложные soft targets должны иметь собственное обоснование и independent validation.

<!-- /unit:4c94d92771f958c9c2 -->

<!-- unit:7993bd66b521088538 -->
<a id="u-7993bd66b521088538"></a>

### Вывод gradient cross-entropy

<!-- /unit:7993bd66b521088538 -->

<!-- unit:2603367b17c1db779a -->
<a id="u-2603367b17c1db779a"></a>

Для logits z и soft target q с суммой 1 запишем $L=-\sum_k q_kz_k+\log\sum_j e^{z_j}$. Первая часть дифференцируется в −q, вторая — в softmax p, поэтому gradient равен p−q. Это относится и к hard one-hot, и к MixUp, и к smoothed labels. Если target не нормирован, коэффициент перед log-sum-exp меняется; проверка суммы q является частью data contract.

<!-- /unit:2603367b17c1db779a -->

<!-- unit:d581a6a2771c5c1bb3 -->
<a id="u-d581a6a2771c5c1bb3"></a>

Пусть K=3, истинный класс первый, p=(0.98,0.01,0.01). Для hard target gradient равен (−0.02,0.01,0.01): gradient descent продолжает увеличивать true logit. При smoothing epsilon=0.1 с uniform по всем классам q=(28/30,1/30,1/30), gradient становится примерно (0.04667,−0.02333,−0.02333). Теперь true logit уменьшается: модель уже увереннее, чем требует изменённый target.

<!-- /unit:d581a6a2771c5c1bb3 -->

<!-- unit:4ee7996a6cb56a8a47 -->
<a id="u-4ee7996a6cb56a8a47"></a>

В идеальной свободно настраиваемой модели минимум CE для одного soft target достигается при p=q. Для указанного target отношение true probability к каждой другой равно 28, поэтому нужная разность logits — $\log28\approx3.3322$. Для альтернативного соглашения q=(0.9,0.05,0.05) это $\log18\approx2.8904$. Одинаковое название smoothing и одинаковое epsilon задают разные оптимумы. При hard target без других ограничений улучшение CE может продолжаться с ростом margin к бесконечности.

<!-- /unit:4ee7996a6cb56a8a47 -->

<!-- unit:ee42432e7b4d22ade3 -->
<a id="u-ee42432e7b4d22ade3"></a>

Минимальная CE soft target равна его entropy и обычно положительна. Поэтому ненулевой train loss не обязательно означает, что модель не смогла выучить такие данные. Сравнивать raw train CE экспериментов с разными targets без оговорки тоже нельзя. Мотивация label smoothing представлена в [Rethinking the Inception Architecture](https://arxiv.org/abs/1512.00567).

<!-- /unit:ee42432e7b4d22ade3 -->

<!-- unit:630e1a2d839b34258c -->
<a id="u-630e1a2d839b34258c"></a>

### Что означает uniform-компонента

<!-- /unit:630e1a2d839b34258c -->

<!-- unit:d96674dced543878f1 -->
<a id="u-d96674dced543878f1"></a>

Обозначим uniform distribution u. Тогда smoothing loss равен $(1-\epsilon)CE(y,p)+\epsilon CE(u,p)$, а $CE(u,p)=H(u)+KL(u\Vert p)$. Добавка предпочитает не делать альтернативные probabilities слишком малыми. Это не тот же objective, что entropy bonus модели: $-H(p)$ связан с $KL(p\Vert u)$, где направление KL противоположно.

<!-- /unit:d96674dced543878f1 -->

<!-- unit:510b7412317f5b6156 -->
<a id="u-510b7412317f5b6156"></a>

Если истинная conditional probability равна p*, средний smoothed target в идеальном случае равен $(1-\epsilon)p^*+\epsilon u$. Именно его и выучивает свободная модель при таком objective. Например, для бинарного p*=0.9 и epsilon=0.1 получаем 0.86. Значит, label smoothing намеренно меняет вероятностную цель. Оно может компенсировать переуверенность конкретной конечной модели, но не гарантирует калибровку для исходной population probability.

<!-- /unit:510b7412317f5b6156 -->

<!-- unit:dffe775b41d6847e6f -->
<a id="u-dffe775b41d6847e6f"></a>

Uniform по vocabulary, uniform по допустимым actions и uniform по классам отдельной задачи — разные distributions. Для NLP надо решить, включены ли special tokens и как обрабатываются padding/ignored positions. Для классов с известной близостью можно рассматривать другой prior, но его правильность и coefficient требуют независимой проверки. «Мягкая» метка не становится достовернее только из-за отсутствия точных нулей.

<!-- /unit:dffe775b41d6847e6f -->

<!-- unit:c43287b39264595c68 -->
<a id="u-c43287b39264595c68"></a>

## MixUp в inputs и representations

<!-- /unit:c43287b39264595c68 -->

<!-- unit:ff995ef2ca56300aae -->
<a id="u-ff995ef2ca56300aae"></a>

MixUp строит:

<!-- /unit:ff995ef2ca56300aae -->

<!-- unit:893683f5687662f445 -->
<a id="u-893683f5687662f445"></a>

$$
\widetilde x=\lambda x_i+(1-\lambda)x_j,
\qquad
\widetilde y=\lambda y_i+(1-\lambda)y_j,
\qquad
\lambda\sim\operatorname{Beta}(\alpha,\alpha).
$$

<!-- /unit:893683f5687662f445 -->

<!-- unit:6bacc18fc40c8851c5 -->
<a id="u-6bacc18fc40c8851c5"></a>

При classes A/B и $\lambda=0.3$ target содержит 0.3 для A и 0.7 для B. Cross-entropy с таким target равен weighted sum двух class losses. Смешивать class IDs численно нельзя: label 2 и label 6 не дают semantic class 4. Смешиваются one-hot/distributions.

<!-- /unit:6bacc18fc40c8851c5 -->

<!-- unit:219400b574f3e74309 -->
<a id="u-219400b574f3e74309"></a>

Linear interpolation задаёт prior о поведении между examples. Не все mixed inputs естественны; польза метода эмпирическая и task-dependent. Для unrelated tasks или incompatible modalities механическая смесь может не иметь смысла.

<!-- /unit:219400b574f3e74309 -->

<!-- unit:9a6a54a94cb47752e8 -->
<a id="u-9a6a54a94cb47752e8"></a>

Manifold Mixup смешивает hidden representations на выбранном layer, затем продолжает forward с mixed target. Простое смешивание двух decoded слов или integer tokens не эквивалентно этому. При frozen encoder нужно понять, на каком уровне trainable model может воспользоваться смешанным representation. Batch pairing, distribution lambda и выбранные layers являются частью experiment config.

<!-- /unit:9a6a54a94cb47752e8 -->

<!-- unit:546a992349fd12eaef -->
<a id="u-546a992349fd12eaef"></a>

### Один forward и две составляющие loss

<!-- /unit:546a992349fd12eaef -->

<!-- unit:80916043a8fd7a8177 -->
<a id="u-80916043a8fd7a8177"></a>

Учебные входы $x_i=(0,1)$ и $x_j=(1,0)$ при lambda=0.3 дают mixed input (0.7,0.3). Если их классы A и B, target для трёх классов равен (0.3,0.7,0). Пусть prediction на mixed input p=(0.4,0.5,0.1). Тогда

<!-- /unit:80916043a8fd7a8177 -->

<!-- unit:a9a75e28d003f52a74 -->
<a id="u-a9a75e28d003f52a74"></a>

$$
L=-0.3\log0.4-0.7\log0.5\approx0.76009,
\qquad \nabla_zL=(0.1,-0.2,0.1).
$$

<!-- /unit:a9a75e28d003f52a74 -->

<!-- unit:40860f6a43b7ce452b -->
<a id="u-40860f6a43b7ce452b"></a>

Это равно $0.3CE(A,p)+0.7CE(B,p)$ для одного и того же p на mixed input. Оно не равно автоматически weighted losses двух отдельных forwards на исходных xi и xj. Функция сети нелинейна, поэтому её response на смесь является дополнительным требованием. MixUp обучает предпочтению относительно простого поведения между выбранными примерами, как сформулировано в [исходной работе](https://arxiv.org/abs/1710.09412).

<!-- /unit:40860f6a43b7ce452b -->

<!-- unit:10005f2fe95c66c04f -->
<a id="u-10005f2fe95c66c04f"></a>

Численно усреднять nominal class IDs нельзя. Пример IDs 2 и 6 не создаёт «истинный класс 4»: арифметика относится к vectors distributions. Входы также должны иметь совместимые shape и representation. Интерполяция двух пиксельных изображений не обязательно является фотографией реального объекта; метод использует обучающий prior, а не доказанную физическую модель композиции.

<!-- /unit:10005f2fe95c66c04f -->

<!-- unit:ec0127ca37febd95c4 -->
<a id="u-ec0127ca37febd95c4"></a>

### Как alpha управляет силой смешивания

<!-- /unit:ec0127ca37febd95c4 -->

<!-- unit:c2712c6763b0730628 -->
<a id="u-c2712c6763b0730628"></a>

Для $\lambda\sim Beta(\alpha,\alpha)$ среднее равно 0.5, variance равна $1/[4(2\alpha+1)]$. При alpha<1 плотность больше у краёв, поэтому часто доминирует один исходник; при alpha=1 распределение равномерно; при большом alpha коэффициенты концентрируются около 0.5. Например, variance для alpha=1 равна 1/12, а для alpha=10 — 1/84.

<!-- /unit:c2712c6763b0730628 -->

<!-- unit:b79efe7a004046ec3c -->
<a id="u-b79efe7a004046ec3c"></a>

Сохранение одного и того же среднего lambda не означает одинаковой силы регуляризации. В одной реализации lambda выбирают на весь batch, в другой на каждую пару; pairing может допускать совпадение с самим собой или ограничиваться группами. Эти решения задают разное распределение обучающих задач. При class imbalance случайное pairing и sampling также меняют, какие редкие признаки чаще смешиваются с большинством.

<!-- /unit:b79efe7a004046ec3c -->

<!-- unit:c93e77272ed6f936dc -->
<a id="u-c93e77272ed6f936dc"></a>

MixUp и fixed uniform label smoothing являются affine-преобразованиями targets и коммутируют при одинаковом epsilon. Для q=(0.3,0.7,0), epsilon=0.1 получаем (0.30333,0.66333,0.03333) независимо от того, сгладили исходные labels до смеси или смешали сначала. Это алгебра targets; совместное влияние методов на обучение не обязано быть полезным. Слишком сильное размягчение может убрать нужный сигнал.

<!-- /unit:c93e77272ed6f936dc -->

<!-- unit:14419e5cac6fed37f7 -->
<a id="u-14419e5cac6fed37f7"></a>

### Manifold Mixup и frozen representation

<!-- /unit:14419e5cac6fed37f7 -->

<!-- unit:63d38554e5c85351f8 -->
<a id="u-63d38554e5c85351f8"></a>

Представим сеть как $f=g\circ h$. Manifold Mixup создаёт $\widetilde h=\lambda h(x_i)+(1-\lambda)h(x_j)$ и продолжает вычисление через g с mixed target. Не надо декодировать mixed hidden vector в «новое слово» или изображение: требование относится к последующему предсказателю.

<!-- /unit:63d38554e5c85351f8 -->

<!-- unit:d26a407f94995ec36b -->
<a id="u-d26a407f94995ec36b"></a>

Если h обучаем, gradients могут менять геометрию representation вместе с g. Если h заморожен, можно обучать g на смеси его фиксированных outputs, но h не научится новым инвариантностям этим путём. Совместимость hidden states, layer choice, paired examples и task labels остаётся предпосылкой. Для sequence tensors нужны ещё выравнивание позиций и padding masks; одинаковый shape не означает одинаковое смысловое место. Механизм исследуется в [Manifold Mixup](https://arxiv.org/abs/1806.05236).

<!-- /unit:d26a407f94995ec36b -->

<!-- unit:86dfe1efbaf9883bc3 -->
<a id="u-86dfe1efbaf9883bc3"></a>

## CutMix и удаление областей

<!-- /unit:86dfe1efbaf9883bc3 -->

<!-- unit:ea3a31d4f8d6da0c65 -->
<a id="u-ea3a31d4f8d6da0c65"></a>

CutMix заменяет region image A region из B, а weights targets определяются фактической заменённой площадью:

<!-- /unit:ea3a31d4f8d6da0c65 -->

<!-- unit:025215fc839462487b -->
<a id="u-025215fc839462487b"></a>

$$
\lambda_{\mathrm{eff}}=1-\frac{A_{\mathrm{patch}}}{A_{\mathrm{image}}}.
$$

<!-- /unit:025215fc839462487b -->

<!-- unit:d8cb60682aa4f6b745 -->
<a id="u-d8cb60682aa4f6b745"></a>

Если в image $100\times100$ вставлен patch $20\times40$, исходному class остаётся weight 0.92, вставленному — 0.08. После clipping patch по границам нужно пересчитать actual area, а не сохранять исходную sampled lambda. При detection/segmentation simple class mixture недостаточна: обновляются spatial labels, видимость и правила для перекрытых objects.

<!-- /unit:d8cb60682aa4f6b745 -->

<!-- unit:6febb57d418484bd11 -->
<a id="u-6febb57d418484bd11"></a>

Cutout/Random Erasing удаляют/заменяют region без добавления label другого объекта. Они учат устойчивости к occlusion только пока оставшееся evidence достаточно. Если entire small object исчез, сохранение label может стать noise. DropBlock из главы [«Dropout, DropConnect, DropBlock и случайные ветви»](<reg-stochastic.md>) удаляет activations внутри network, а не входные pixels; эффекты и настройки различаются.

<!-- /unit:6febb57d418484bd11 -->

<!-- unit:9d7bd4c696615707a1 -->
<a id="u-9d7bd4c696615707a1"></a>

AutoAugment ищет политику transformations в заданном search space, RandAugment упрощает управление выбором и strength. Это способы организовать augmentation policy, а не гарантированно безопасные для любой annotation presets. Подбор policy по test недопустим.

<!-- /unit:9d7bd4c696615707a1 -->

<!-- unit:efb8b71d9ce41c1eaf -->
<a id="u-efb8b71d9ce41c1eaf"></a>

### Реальная площадь после clipping

<!-- /unit:efb8b71d9ce41c1eaf -->

<!-- unit:c2e1ab2ec9e679b438 -->
<a id="u-c2e1ab2ec9e679b438"></a>

Пусть изображение 100×100, nominal patch 40×40 с центром (5,10). Его half-open границы по x равны [−15,25), по y [−10,30). После пересечения с изображением остаётся область [0,25)×[0,30), площадь 750, а не 1600. При area-weighted CutMix target исходного класса имеет weight 0.925, донорского — 0.075. Использование nominal площади дало бы ошибочный weight 0.84.

<!-- /unit:c2e1ab2ec9e679b438 -->

<!-- unit:c5ab09e693e95f6ded -->
<a id="u-c5ab09e693e95f6ded"></a>

Геометрические границы зависят от округления, индексации и способа выбора patch. Coefficient пересчитывают по фактически заменённым pixels. Если область пустая, input и target должны согласованно остаться исходными. Само правило площади является prior классификационного [CutMix](https://arxiv.org/abs/1905.04899), а не измерением доли содержательного evidence каждого класса: 7.5% картинки могут содержать весь маленький объект или только фон.

<!-- /unit:c5ab09e693e95f6ded -->

<!-- unit:3c1f42fc83ed13ae73 -->
<a id="u-3c1f42fc83ed13ae73"></a>

### Почему segmentation и detection нельзя свести к одной смеси классов

<!-- /unit:3c1f42fc83ed13ae73 -->

<!-- unit:698d228b054b16a9b6 -->
<a id="u-698d228b054b16a9b6"></a>

Для semantic segmentation label в области donor обычно берётся из donor mask, вне области — из recipient mask, с сохранением ignore policy. Это пространственная композиция targets, а не один global vector 0.925/0.075. Для instance segmentation сохраняют identity и видимую часть каждого объекта, не смешивая instance IDs как непрерывные числа.

<!-- /unit:698d228b054b16a9b6 -->

<!-- unit:5f018d753eab84af77 -->
<a id="u-5f018d753eab84af77"></a>

Для detection donor objects могут быть обрезаны patch, а recipient objects — частично закрыты. Нужно выбрать, описывает target видимую часть или полный объект, и как учитывается occlusion. Иногда надо обновить bbox, mask и visibility либо исключить объект по явному правилу. Если удалить label, оставив хорошо видимый объект как фон, получится новый источник противоречивого supervision. Связанные поля фильтруют одним object mapping.

<!-- /unit:5f018d753eab84af77 -->

<!-- unit:4e6f3b9ce76cda4b4f -->
<a id="u-4e6f3b9ce76cda4b4f"></a>

Cutout и Random Erasing не добавляют другой класс: область удаляется либо заполняется по заданному правилу, а target сохраняется только при допустимой постановке. Их параметры включают долю площади, aspect ratio, число областей и распределение заполнения. DropBlock применяется к feature map внутри сети, поэтому похожая геометрическая картинка маски не делает эти методы одним оператором.

<!-- /unit:4e6f3b9ce76cda4b4f -->

<!-- unit:ad0a6737171edc4604 -->
<a id="u-ad0a6737171edc4604"></a>

## Политики и total variation

<!-- /unit:ad0a6737171edc4604 -->

<!-- unit:fd028d65ab636af23f -->
<a id="u-fd028d65ab636af23f"></a>

Для реконструкции image prior гладкости может задаваться anisotropic TV:

<!-- /unit:fd028d65ab636af23f -->

<!-- unit:575ac0a7a52f42816e -->
<a id="u-575ac0a7a52f42816e"></a>

$$
\operatorname{TV}(u)=
\sum_{i,j}\bigl(|u_{i+1,j}-u_{i,j}|+|u_{i,j+1}-u_{i,j}|\bigr),
$$

<!-- /unit:575ac0a7a52f42816e -->

<!-- unit:c5c24aab3c190ea411 -->
<a id="u-c5c24aab3c190ea411"></a>

где границы сумм ограничены существующими neighbours. Isotropic variant использует square root суммы квадратов directional differences. TV предпочитает кусочно-гладкие images, но может стирать texture и мелкие детали. Это prior на восстанавливаемое поле, а не универсальный penalty каждого CV classifier.

<!-- /unit:c5c24aab3c190ea411 -->

<!-- unit:aceb4e77d6f36d2055 -->
<a id="u-aceb4e77d6f36d2055"></a>

### Поиск политики тоже использует данные

<!-- /unit:aceb4e77d6f36d2055 -->

<!-- unit:ed300355c7b2457963 -->
<a id="u-ed300355c7b2457963"></a>

AutoAugment задаёт пространство subpolicies: упорядоченные операции, вероятность применения каждой и magnitude. Policy search оценивает качество обученных кандидатов на development-данных. Поиск может быть дорогим; если его стоимость исключить из сравнения, итоговый бюджет будет неполным. Зафиксированная policy затем применяется при training, но её переносимость на новые источники и annotations требует проверки. Устройство поиска описано в [AutoAugment](https://arxiv.org/abs/1805.09501).

<!-- /unit:ed300355c7b2457963 -->

<!-- unit:b22ad416ad15281a27 -->
<a id="u-b22ad416ad15281a27"></a>

RandAugment уменьшает пространство управления до небольшого числа параметров, обычно количества выбираемых операций N и общего magnitude M. Один M преобразуется в разные физические единицы для rotation, contrast и других операций; M не является универсальным числом градусов или процентов. Конкретный банк операций, sampling и диапазоны принадлежат реализации. Более простая настройка в [RandAugment](https://arxiv.org/abs/1909.13719) не устраняет необходимость аудита меток.

<!-- /unit:b22ad416ad15281a27 -->

<!-- unit:2889745c0c9172ef8c -->
<a id="u-2889745c0c9172ef8c"></a>

Политика, хорошая для предметной классификации, может испортить OCR, pose или задачу различения диагностического цвета. Проверяют редкие срезы и композиции transforms. Validation, использованная для выбора policy, не является независимым доказательством её финального выигрыша; test сохраняют для оценки выбранного pipeline.

<!-- /unit:2889745c0c9172ef8c -->

<!-- unit:340b548e400db36bf9 -->
<a id="u-340b548e400db36bf9"></a>

### Total variation — prior на поле

<!-- /unit:340b548e400db36bf9 -->

<!-- unit:7e766e35a3f3c24c23 -->
<a id="u-7e766e35a3f3c24c23"></a>

Для 2×2 изображения со строками (0,1) и (1,1) сумма модулей всех горизонтальных и вертикальных соседних различий равна 2. Если isotropic TV считать по forward differences, задавая отсутствующие на границе разности нулём, единственный ненулевой совместный gradient имеет компоненты (1,1), поэтому TV равна $\sqrt2$. Разные conventions дают разные значения и gradients.

<!-- /unit:7e766e35a3f3c24c23 -->

<!-- unit:79dd086f343d36f143 -->
<a id="u-79dd086f343d36f143"></a>

TV поощряет небольшую общую величину пространственных изменений и может сохранять резкие границы между кусочно-постоянными областями. Но мелкая текстура или узкая структура тоже увеличивает TV; сильный коэффициент способен их стереть и создавать ступенчатые участки. Для реконструкции penalty сопоставляют с data fidelity, например squared error наблюдений. Для обычного classifier нет универсального смысла штрафовать TV любого внутреннего tensor без отдельной гипотезы.

<!-- /unit:79dd086f343d36f143 -->

<!-- unit:fa80973ac8ac0adb8a -->
<a id="u-fa80973ac8ac0adb8a"></a>

При изменении resolution меняются число соседних пар и физический шаг сетки. Сумма и среднее TV — разные масштабы objective; их lambda нельзя переносить без пересчёта. Восстанавливаемая intensity map и дискретная class mask также требуют разных интерпретаций гладкости.

<!-- /unit:fa80973ac8ac0adb8a -->

<!-- unit:b88d3c0e15690176be -->
<a id="u-b88d3c0e15690176be"></a>

## Контроль и практика

<!-- /unit:b88d3c0e15690176be -->

<!-- unit:5f2b8f4cbfdebcc994 -->
<a id="u-5f2b8f4cbfdebcc994"></a>

1. Почему label smoothing conventions дают разные targets? **Ответ:** epsilon распределяется либо по всем classes, либо только по incorrect classes.
2. Почему CutMix использует actual area? **Ответ:** crop/clipping меняют количество реально заменённых pixels.
3. Можно ли MixUp считать линейной смесью class IDs? **Ответ:** нет; IDs nominal, смешиваются target distributions.

<!-- /unit:5f2b8f4cbfdebcc994 -->

<!-- unit:99fdbcdd00e6978ebf -->
<a id="u-99fdbcdd00e6978ebf"></a>

**Практика:** вручную вычислите оба smoothed target при четырёх classes. Сделайте MixUp и CutMix двух examples, проверьте сумму target weights. Для small-object image найдите strength, при которой label уже не оправдан; включите этот случай в ручной аудит до training comparison.

<!-- /unit:99fdbcdd00e6978ebf -->

<!-- unit:4bad9117a7f3528534 -->
<a id="u-4bad9117a7f3528534"></a>

**Схема курса:** example + label → smoothing distribution or paired MixUp/CutMix transform → aligned soft target → loss → metric under declared augmentation policy

<!-- /unit:4bad9117a7f3528534 -->

<!-- unit:859c52882f5d63063c -->
<a id="u-859c52882f5d63063c"></a>

<!-- source-depth-regularization-Ж -->

<!-- /unit:859c52882f5d63063c -->

<!-- unit:0e8bac06428ec9c8b2 -->
<a id="u-0e8bac06428ec9c8b2"></a>

1. **Почему smoothing может уменьшить вероятность правильного класса при корректном обучении?** Оптимум изменённого target отличается от исходной conditional probability.
2. **Где вычисляются две CE составляющие MixUp?** На одном prediction для mixed input, а не обязательно на двух исходных images.
3. **Почему площадь CutMix не равна доле semantic evidence?** Содержательная информация распределена по pixels неравномерно.
4. **Почему uniform smoothing не тождественно entropy bonus?** Они соответствуют разным направлениям KL и разным целям.

<!-- /unit:0e8bac06428ec9c8b2 -->

<!-- unit:6bdbb63eec51064003 -->
<a id="u-6bdbb63eec51064003"></a>

Практика: вывести gradient soft CE, рассчитать оба smoothing conventions и один MixUp forward. Для CutMix записать nominal и clipped границы, обновить targets выбранной CV-задачи. Отдельно сравнить одиночные изменения и их сочетание при понятном бюджете. Все inputs, probabilities и площади главы учебные; результаты реального обучения и policy search здесь не заявляются.

<!-- /unit:6bdbb63eec51064003 -->

## Результат главы и переход дальше

Ответьте на вопрос: **Как изменения inputs и targets задают предположения о задаче?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

[Маршрут модуля](<README.md>) · [Происхождение и точные места переноса](<../editorial/coverage.md>)
