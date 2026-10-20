# Pose: keypoints, heatmaps и группировка

**Уровень:** основа ветки.

**Предпосылки:** [Изображение, карты признаков и блоки модели](<cv-input.md>), [Классификация, CNN и residual learning](<cv-classification.md>)

**Используют:** Выбранная прикладная задача и итоговая проверка результата.

**Учебный вопрос:** Как связать положение ключевых точек с конкретным объектом?





## Карта вычисления и границы этапов

![Pose: keypoints, heatmaps и группировка: входы, преобразования и выходы](<../оформление/course/cv-pose.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Сравните формы и смысл каждого перехода с формулами главы. Выход преобразования и доказательство его качества — разные объекты: последняя проверка использует отдельный протокол оценки.

<!-- unit:16e31c318af7e6f4f8 -->
<a id="u-16e31c318af7e6f4f8"></a>

## Что должна предсказать модель

<!-- /unit:16e31c318af7e6f4f8 -->

<!-- unit:95e42d1949bf954fb8 -->
<a id="u-95e42d1949bf954fb8"></a>

Для каждого человека задан фиксированный словарь joints: например, плечи, локти, запястья. Output содержит координаты, confidence/visibility и принадлежность человеку. Это не arbitrary object mask и не само по себе 3D reconstruction.

<!-- /unit:95e42d1949bf954fb8 -->

<!-- unit:30b61630995f4a17df -->
<a id="u-30b61630995f4a17df"></a>

Аннотация должна различать отсутствие разметки и occlusion. В COCO-подобной convention $v=0$ — не размечено, $v=1$ — размечено, но закрыто, $v=2$ — видно. Метрики с indicator(v>0) учитывают и $v=1$, и v=2. В исходном тексте это называлось «только видимые», что теряет важное различие.

<!-- /unit:30b61630995f4a17df -->

<!-- unit:99a4aa3db72242abe0 -->
<a id="u-99a4aa3db72242abe0"></a>

Левая/правая сторона обычно определяются по телу объекта, а не по viewer. Horizontal flip требует не только $x\mapsto W-1-x$ при pixel-centre convention, но и перестановки labels left/right. Иначе image выглядит корректно, а targets становятся противоположными.

<!-- /unit:99a4aa3db72242abe0 -->

<!-- unit:b80d120b273227cb34 -->
<a id="u-b80d120b273227cb34"></a>

### Словарь суставов — часть контракта данных

<!-- /unit:b80d120b273227cb34 -->

<!-- unit:4507f0b37cc0c801f5 -->
<a id="u-4507f0b37cc0c801f5"></a>

Сохраним порядок COCO из рассматриваемого примера: nose; left/right eye; left/right ear; left/right shoulder; left/right elbow; left/right wrist; left/right hip; left/right knee; left/right ankle. В нулевой индексации это 17 позиций: нос 0, глаза 1/2, уши 3/4, плечи 5/6, локти 7/8, запястья 9/10, бёдра 11/12, колени 13/14, лодыжки 15/16. Flat annotation содержит 51 значение x,y,v, однако значения v — annotation flags, а не обязательно те же числа, которые должна предсказывать confidence head.

<!-- /unit:4507f0b37cc0c801f5 -->

<!-- unit:08c876de5b011f080c -->
<a id="u-08c876de5b011f080c"></a>

Поле `num_keypoints` не следует объяснять как только число видимых v=2. Существенно число размеченных точек, включая допустимые закрытые: правило v>0 отличается от «видно на картинке». Точка v=0 не обязана означать физически отсутствующий сустав; её могли не размечать. Нулевые координаты в таком slot без корректной mask нельзя использовать как настоящую цель в углу изображения.

<!-- /unit:08c876de5b011f080c -->

<!-- unit:9c8376736a6924c945 -->
<a id="u-9c8376736a6924c945"></a>

Для руки типичен другой словарь, для лица — свой набор landmarks, а 3D datasets могут менять topology и порядок даже при тех же 17 joints. Совпадение числа точек не доказывает совместимость output. Skeleton edges для визуализации и PAF connections тоже задаются отдельно: 17 joints не означает автоматически 17 limbs, а body skeleton не включает все hand/face landmarks просто из-за имени OpenPose.

<!-- /unit:9c8376736a6924c945 -->

<!-- unit:705e22009df5e9581d -->
<a id="u-705e22009df5e9581d"></a>

### Геометрические transforms и правильное восстановление

<!-- /unit:705e22009df5e9581d -->

<!-- unit:4d9b45adf9741a1f3d -->
<a id="u-4d9b45adf9741a1f3d"></a>

Пусть crop начинается в (100,50), имеет размер 200×400 и resized до 100×200. Точка crop=(25,100) соответствует original=(150,250) в простой continuous-coordinate схеме. Если к crop добавили padding 10 слева и 20 сверху, сначала надо вычесть padding, затем применить inverse scale. Переставленные операции дадут систематический сдвиг всех суставов.

<!-- /unit:4d9b45adf9741a1f3d -->

<!-- unit:ecd89925d1903a3823 -->
<a id="u-ecd89925d1903a3823"></a>

При horizontal flip меняется не только x. Левая кисть анатомически остаётся левой кистью человека, хотя переехала на другую сторону изображения. Поэтому training target или TTA result проходит перестановку соответствующих channels. Для J=17 COCO слоты 9 и 10 меняются местами, нос остаётся 0. Скелет может выглядеть почти правдоподобно даже при ошибке перестановки, поэтому отдельно проверяют labels, а не только соединённые линии.

<!-- /unit:ecd89925d1903a3823 -->

<!-- unit:6e029a73d038102b88 -->
<a id="u-6e029a73d038102b88"></a>

## Top-down, bottom-up и one-stage

<!-- /unit:6e029a73d038102b88 -->

<!-- unit:14d92b35db386a1aea -->
<a id="u-14d92b35db386a1aea"></a>

Top-down pipeline сначала находит человека, вырезает region, затем отдельная pose model обрабатывает crop. Это делает scale более стандартизированным, но missed detection теряет pose; compute обычно зависит от количества crops. HRNet-подобный estimator может использоваться в такой системе.

<!-- /unit:14d92b35db386a1aea -->

<!-- unit:5861aa56e0ecd8ad0e -->
<a id="u-5861aa56e0ecd8ad0e"></a>

Bottom-up находит все joints и grouping cues, потом связывает точки по людям. OpenPose использует part affinity fields — vector fields вдоль связей тела. Candidate limbs оцениваются по согласованности направления сегмента с field, затем собираются skeletons. [OpenPose](https://arxiv.org/abs/1611.08050).

<!-- /unit:5861aa56e0ecd8ad0e -->

<!-- unit:329d0b81ba0f86fda2 -->
<a id="u-329d0b81ba0f86fda2"></a>

One-stage detector с keypoint head совместно выдаёт boxes и points. YOLO-pose относится к такому типу task-specific prediction; наличие boxes не превращает его автоматически в двухэтапный crop pipeline. Вычислительный граф и assignment нужно смотреть у конкретной модели.

<!-- /unit:329d0b81ba0f86fda2 -->

<!-- unit:b56d21d46fb302c8a7 -->
<a id="u-b56d21d46fb302c8a7"></a>

HRNet сохраняет high-resolution branch и обменивает information между resolutions. Это помогает точной localisation, но width/resolution стоят памяти. Хорошая local coordinate accuracy не решает grouping нескольких людей сама по себе.

<!-- /unit:b56d21d46fb302c8a7 -->

<!-- unit:a080406e7372460c47 -->
<a id="u-a080406e7372460c47"></a>

### Почему стоимость top-down зависит от людей, а bottom-up — тоже не константа

<!-- /unit:a080406e7372460c47 -->

<!-- unit:74181bcf54425b7537 -->
<a id="u-74181bcf54425b7537"></a>

Для top-down есть detector pass и pose processing N crops. Crops можно объединять в batch, поэтому «N forward calls» — не обязательное число последовательных запусков, но объём работы и памяти всё равно зависит от N и размера crops. Потерянный detector box лишает pipeline возможности восстановить соответствующую позу, даже если single-person estimator идеален.

<!-- /unit:74181bcf54425b7537 -->

<!-- unit:96342e77428b9ccac7 -->
<a id="u-96342e77428b9ccac7"></a>

Bottom-up вычисляет joint maps для всей картинки, затем ищет несколько peaks каждого типа и группирует их. Neural forward может быть почти фиксированным для данного resolution, однако число candidates и стоимость matching растут в толпе. Формулировка чата «скорость постоянная при любом числе людей» относится только к части pipeline и потому неполна.

<!-- /unit:96342e77428b9ccac7 -->

<!-- unit:380f11da71b6af9c4f -->
<a id="u-380f11da71b6af9c4f"></a>

В PAF подходе для candidate плеча a и локтя b направление $u=(b-a)/\lVert b-a\rVert$. Связь оценивается усреднением $L(a+t(b-a))\cdot u$ по t∈[0,1]. При field, совпадающем с u на всём limb, score близок к 1; при противоположном — к −1; при перпендикулярном — около 0. На практике интеграл заменяют sampling, учитывая coverage и допустимые расстояния. Много сильных elbow peaks ещё не говорит, какой относится к какому плечу: это и решает grouping.

<!-- /unit:380f11da71b6af9c4f -->

<!-- unit:c297b21c416e486157 -->
<a id="u-c297b21c416e486157"></a>

Associative embedding задаёт иной cue: точки одного человека должны иметь близкие tags, разных — различимые. Ошибка такого метода может быть не в localization, а в смешивании двух скелетов. HRNet поддерживает точные пространственные features, но сам по себе не задаёт ни PAF, ни grouping tags; архитектуру backbone и задачу head нужно разделять.

<!-- /unit:c297b21c416e486157 -->

<!-- unit:f21f4ae60c56204b34 -->
<a id="u-f21f4ae60c56204b34"></a>

## Heatmaps и coordinate regression

<!-- /unit:f21f4ae60c56204b34 -->

<!-- unit:b4e7b52da3d810b206 -->
<a id="u-b4e7b52da3d810b206"></a>

Direct regression возвращает x/y, например нормированные внутри crop. Loss может быть L1/L2, robust или probabilistic variant; visibility weights определяют supervision. При двух допустимых positions MSE может усреднить их в несуществующее место, поэтому uncertainty/distribution важны.

<!-- /unit:b4e7b52da3d810b206 -->

<!-- unit:8ccded278851a6290e -->
<a id="u-8ccded278851a6290e"></a>

Heatmap head создаёт map $B\times J\times H_{h}\times W_{h}$. Для GT point $(u,v)$ target можно задать Gaussian:

<!-- /unit:8ccded278851a6290e -->

<!-- unit:63da812930cc65cc37 -->
<a id="u-63da812930cc65cc37"></a>

$$
G(x,y)=\exp\!\left(-\frac{(x-u)^2+(y-v)^2}{2\sigma^2}\right).
$$

<!-- /unit:63da812930cc65cc37 -->

<!-- unit:15d67e55c3e28feae1 -->
<a id="u-15d67e55c3e28feae1"></a>

При sigma1 значение на centre1, на расстоянии1 — e^-0.5≈0.6065, на расстоянии2 — e^-2≈0.1353. Степень blur задаёт tolerance training target, а не evaluation OKS напрямую.

<!-- /unit:15d67e55c3e28feae1 -->

<!-- unit:f9a711d7553f0c26a6 -->
<a id="u-f9a711d7553f0c26a6"></a>

Argmax выбирает пик resolution heatmap, вводя discretization. Refinement/soft-argmax могут дать subpixel coordinates, но soft-argmax усредняет multimodal peaks и требует своей temperature/normalization. Heatmap — не автоматически calibrated probability distribution, если objective и normalization этого не задают.

<!-- /unit:f9a711d7553f0c26a6 -->

<!-- unit:a14d955bb638cd6630 -->
<a id="u-a14d955bb638cd6630"></a>

Crop transform должен быть обратимым в согласованной геометрии. Если original crop начинается в (x0,y0), имеет width w,height h и resized to $W_{c}\times H_{c}$, простая непрерывная coordinate convention даёт $x_{\mathrm{orig}}=x_0+\frac{x_{\mathrm{crop}}w}{W_c}$, $y_{\mathrm{orig}}=y_0+\frac{y_{\mathrm{crop}}h}{H_c}$. Half-pixel/align details в реальной библиотеке проверяются отдельно. Letterbox добавляет offsets, их нельзя забывать.

<!-- /unit:a14d955bb638cd6630 -->

<!-- unit:076407e2b2cb25b174 -->
<a id="u-076407e2b2cb25b174"></a>

### Полный путь Gaussian target и его координатная сетка

<!-- /unit:076407e2b2cb25b174 -->

<!-- unit:6ad803af360dd1eafc -->
<a id="u-6ad803af360dd1eafc"></a>

Если input crop 256×192, а heatmap 64×48, масштаб по каждой оси равен четырём при простой сетке. GT point input=(80,120) переходит в heatmap=(20,30); Gaussian надо строить вокруг этой точки, а не вокруг (80,120) на маленькой карте. Неправильная единица sigma также меняет target: sigma=2 heatmap pixels соответствует примерно восьми input pixels, не двум.

<!-- /unit:6ad803af360dd1eafc -->

<!-- unit:c4c74bba5c80d5b5cf -->
<a id="u-c4c74bba5c80d5b5cf"></a>

Heatmap MSE часто сравнивает предсказанную карту с Gaussian значениями без нормировки суммы до единицы. Поэтому слово «распределение» здесь — интуиция, а не автоматическое вероятностное свойство. Для spatial cross-entropy или KL нужна выбранная нормировка и другая трактовка scores. Суммировать KL по сырым ненормированным, возможно отрицательным heatmap logits нельзя.

<!-- /unit:c4c74bba5c80d5b5cf -->

<!-- unit:580ade13d964a620f5 -->
<a id="u-580ade13d964a620f5"></a>

Argmax даёт дискретный cell. Soft-argmax сначала строит нормированные weights, затем ожидаемые координаты:

<!-- /unit:580ade13d964a620f5 -->

<!-- unit:7cee8dcb3a3138c4d5 -->
<a id="u-7cee8dcb3a3138c4d5"></a>

$$
\widehat x=\sum_{u,v}u\,p_{uv},\quad
\widehat y=\sum_{u,v}v\,p_{uv},\quad
p_{uv}=\frac{e^{z_{uv}/T}}{\sum_{a,b}e^{z_{ab}/T}}.
$$

<!-- /unit:7cee8dcb3a3138c4d5 -->

<!-- unit:f41519ecac637a39fc -->
<a id="u-f41519ecac637a39fc"></a>

Если есть два одинаково сильных peaks у x=10 и x=30, среднее даёт x=20, где peak может отсутствовать. Дифференцируемость не устраняет multimodality. Local refinement около выбранного peak имеет другую логику: уточняет одно решение, а не усредняет все возможные.

<!-- /unit:f41519ecac637a39fc -->

<!-- unit:80ceab93b48df6d0cb -->
<a id="u-80ceab93b48df6d0cb"></a>

SimCC представляет x и y отдельными одномерными classification distributions. Это экономит representation по сравнению с полной joint 2D grid: при размерах H_h,W_h нужно J(H_h+W_h) scores вместо JH_hW_h. Однако factorized marginals не хранят все зависимости между координатами. В разбираемом примере метод был назван универсальной частью YOLO-pose; такую привязку нельзя делать без проверки конкретной head: coordinate regression, heatmap и SimCC — самостоятельные варианты.

<!-- /unit:80ceab93b48df6d0cb -->

<!-- unit:98bc7840889c299f46 -->
<a id="u-98bc7840889c299f46"></a>

### Почему MSE на координатах может давать анатомически неверный компромисс

<!-- /unit:98bc7840889c299f46 -->

<!-- unit:e08072a6216e965e81 -->
<a id="u-e08072a6216e965e81"></a>

Если при закрытой кисти данные допускают позиции (10,0) и (−10,0) с равной частотой при одинаковых доступных признаках, оптимум среднего squared error — (0,0). Эта точка может быть физически неверна. Heatmap способна выразить два peaks, но argmax всё равно должен выбрать один, а soft-argmax вновь усреднит. Поэтому uncertainty и способ decoding не менее важны, чем форма output tensor.

<!-- /unit:e08072a6216e965e81 -->

<!-- unit:58358a9e0d1e54f3b9 -->
<a id="u-58358a9e0d1e54f3b9"></a>

В Mask R-CNN keypoint formulation исходной работы точка представлена one-hot location и loss — spatial softmax cross-entropy, а не обязательная Gaussian MSE из рассматриваемого примера. Другие heatmap pipelines действительно используют MSE. Нельзя переносить один loss всем моделям только потому, что их outputs рисуются heatmap. [Mask R-CNN](https://arxiv.org/abs/1703.06870).

<!-- /unit:58358a9e0d1e54f3b9 -->

<!-- unit:171e23be97020834e3 -->
<a id="u-171e23be97020834e3"></a>

## PCK и OKS

<!-- /unit:171e23be97020834e3 -->

<!-- unit:e85dfd7ebceea5fd94 -->
<a id="u-e85dfd7ebceea5fd94"></a>

PCK считает долю размеченных points с $\frac{d_i}{d_{\mathrm{norm}}}<\alpha$. Norm может быть head size, torso size или другое benchmark definition. «PCKh всегда расстояние между ушами» слишком узко: берётся определение конкретного dataset. При четырёх distances $\left(2,\,4,\,6,\,12\right)$, norm20, alpha0.25 threshold5 → PCK2/4=0.5.

<!-- /unit:e85dfd7ebceea5fd94 -->

<!-- unit:e04a4aa0ab74b3009e -->
<a id="u-e04a4aa0ab74b3009e"></a>

OKS усредняет scale-normalized Gaussian similarity:

<!-- /unit:e04a4aa0ab74b3009e -->

<!-- unit:9040767fc2d2f54df3 -->
<a id="u-9040767fc2d2f54df3"></a>

$$
\mathrm{OKS}=\frac{\sum_i\exp\!\left(-\frac{d_i^2}{2s^2k_i^2}\right)\mathbf{1}[v_i>0]}{\sum_i\mathbf{1}[v_i>0]}.
$$

<!-- /unit:9040767fc2d2f54df3 -->

<!-- unit:5d7b7571bd4a55eb42 -->
<a id="u-5d7b7571bd4a55eb42"></a>

s² соответствует object area по protocol, k_i — benchmark-specific tolerance. В COCO implementation используются stored sigmas и дополнительный множитель: при этой записи k_i соответствует $2\sigma_i$. Подставлять список sigma из исходника как k без проверки — получить другой threshold.

<!-- /unit:5d7b7571bd4a55eb42 -->

<!-- unit:c4417bd74d3576ce17 -->
<a id="u-c4417bd74d3576ce17"></a>

Иллюстрация с двумя labeled points, $s=100$,$k=0.1$: d1=5 даёт exp(-0.125)≈0.8825; d2=10 — exp(-0.5)≈0.6065; mean≈0.7445. На объекте вдвое меньшего linear scale те же pixel errors будут штрафоваться сильнее. Если третий point $v=0$, он не входит в это среднее; occluded $v=1$ входит.

<!-- /unit:c4417bd74d3576ce17 -->

<!-- unit:e49b0eb8c601b5fbd9 -->
<a id="u-e49b0eb8c601b5fbd9"></a>

Pose AP сопоставляет person predictions/GT по OKS и score, после чего строит PR/AP по benchmark. Это метрика instances, а PCK агрегирует keypoints иначе. Нельзя заменить pose AP средней confidence точек.

<!-- /unit:e49b0eb8c601b5fbd9 -->

<!-- unit:abb76dadf50dc6077c -->
<a id="u-abb76dadf50dc6077c"></a>

Для 3D MPJPE — средняя Euclidean joint error в заявленных units после определённого root/alignment protocol. P-MPJPE использует Procrustes alignment по protocol, N-MPJPE обычно scale alignment; фраза исходника «N-MPJPE после масштаба и поворота» не должна смешивать его с Procrustes. Сравнение возможно только при одинаковом protocol.

<!-- /unit:abb76dadf50dc6077c -->

<!-- unit:bdf5899696cbdcb01a -->
<a id="u-bdf5899696cbdcb01a"></a>

### Разворачиваем числовой OKS и связь со stored sigmas

<!-- /unit:bdf5899696cbdcb01a -->

<!-- unit:26f92b4eaf2617c749 -->
<a id="u-26f92b4eaf2617c749"></a>

При $s^2=10000$, $k=0.1$ знаменатель $2s^2k^2=200$. Ошибка d=5 даёт $e^{-25/200}=0.882497$, d=10 — $e^{-100/200}=0.606531$, их среднее 0.744514. Уменьшим linear scale до s=50: знаменатель станет 50, scores — 0.606531 и 0.135335, среднее 0.370933. Те же pixel shifts стали существенно хуже относительно размера человека.

<!-- /unit:26f92b4eaf2617c749 -->

<!-- unit:68ccb3471f107a47b7 -->
<a id="u-68ccb3471f107a47b7"></a>

Если implementation хранит sigma=0.05 и использует $(2\sigma)^2$ в знаменателе, это ровно тот же k=0.1. Подставить sigma=0.05 вместо k в компактную формулу означало бы уменьшить tolerance вдвое и знаменатель вчетверо. Object area берётся по protocol; это не обязательно площадь произвольно расширенного crop. [Официальный COCO evaluator](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py).

<!-- /unit:68ccb3471f107a47b7 -->

<!-- unit:c320e7ca173e3f452a -->
<a id="u-c320e7ca173e3f452a"></a>

PCK превращает расстояние в жёсткий успех/провал на выбранном пороге; OKS оставляет плавный вклад каждой точки, а затем pose AP применяет threshold к instance similarity. Усреднённый PCK можно повысить на множестве простых видимых суставов, сохранив плохое качество закрытых кистей. Поэтому раздельные срезы по joints, occlusion и size помогают увидеть ошибку, скрытую общим score.

<!-- /unit:c320e7ca173e3f452a -->

<!-- unit:94abd27de1337c2858 -->
<a id="u-94abd27de1337c2858"></a>

### Wing и другие loss: восстанавливаем функцию, а не только название

<!-- /unit:94abd27de1337c2858 -->

<!-- unit:7548bf34a67b7aa9a0 -->
<a id="u-7548bf34a67b7aa9a0"></a>

Wing loss для скалярной ошибки e задаётся

<!-- /unit:7548bf34a67b7aa9a0 -->

<!-- unit:2dbd9b504538d6d747 -->
<a id="u-2dbd9b504538d6d747"></a>

$$
\mathcal L_{\rm wing}(e)=
\begin{cases}w\ln(1+|e|/\epsilon),&|e|<w,\\
|e|-C,&|e|\ge w,\end{cases}
\quad C=w-w\ln(1+w/\epsilon).
$$

<!-- /unit:2dbd9b504538d6d747 -->

<!-- unit:46c013ce8933e4e87b -->
<a id="u-46c013ce8933e4e87b"></a>

C обеспечивает непрерывность на |e|=w. При w=10,epsilon=2 обе ветви на границе дают $10\ln6\approx17.917595$. Без C линейная ветвь дала бы 10 и создала скачок. Метод меняет относительную чувствительность к небольшим и большим ошибкам; он не является гарантией робастности к любому шуму. [Wing Loss](https://arxiv.org/abs/1711.06753).

<!-- /unit:46c013ce8933e4e87b -->

<!-- unit:69178f78c89b96a21d -->
<a id="u-69178f78c89b96a21d"></a>

Adaptive Wing из рассматриваемого примера нельзя определять только заменой коэффициента в этой формуле: его чувствительность зависит от значения target heatmap и собственной конструкции. Смысл — различать важность foreground/background значений при heatmap regression. KL работает с нормированными distributions, OKS-based loss нормирует геометрию по scale; это разные surrogate objectives. Их смешивание требует согласованных единиц и normalization.

<!-- /unit:69178f78c89b96a21d -->

<!-- unit:7f0cb408b2f7ea152e -->
<a id="u-7f0cb408b2f7ea152e"></a>

### D, видео и архитектурные альтернативы

<!-- /unit:7f0cb408b2f7ea152e -->

<!-- unit:bcd335a3222ba16138 -->
<a id="u-bcd335a3222ba16138"></a>

Из одной 2D проекции не следует единственная 3D поза: разные глубины суставов могут давать похожие image coordinates. Lifting учит prior по 2D skeleton, direct 3D model — по pixels/features, multi-view использует несколько согласованных камер, RGB-D добавляет depth measurement. Эти источники информации нельзя сравнивать как один и тот же input protocol.

<!-- /unit:bcd335a3222ba16138 -->

<!-- unit:cf55d120b63665a32d -->
<a id="u-cf55d120b63665a32d"></a>

Root-relative MPJPE убирает положение выбранного root по convention; Procrustes-aligned score может дополнительно убрать поворот и масштаб и потому скрыть ошибки global orientation/scale. Нужно объявлять joints и units, а не только имя MPJPE. Temporal models используют соседние кадры; offline двусторонний доступ к будущим frames не равен causal real-time режиму.

<!-- /unit:cf55d120b63665a32d -->

<!-- unit:24378c18bdfa88bdfd -->
<a id="u-24378c18bdfa88bdfd"></a>

SimpleBaseline проверяет сильный backbone и upsampling без сложной схемы; HRNet сохраняет высокий spatial путь; HigherHRNet добавляет bottom-up machinery, ViTPose использует Transformer features, RTMPose — coordinate-classification design, MediaPipe/BlazePose ориентированы на свой набор landmarks и pipeline. Эти имена из разбираемого примера сохраняются как различия решений. Число joints, latency, пригодность для камеры и качество оцениваются у конкретного artifact.

<!-- /unit:24378c18bdfa88bdfd -->

<!-- unit:3908f9e11d10f08394 -->
<a id="u-3908f9e11d10f08394"></a>

Сглаживание trajectories уменьшает jitter, но может запаздывать при резких движениях и смешать людей после ID-switch. Его оценивают вместе с задержкой и quality, отдельно от single-frame localization. При flip TTA нужно inverse spatial mapping и left/right swap до усреднения; при multi-scale — привести все coordinates в оригинальный кадр.

<!-- /unit:3908f9e11d10f08394 -->

<!-- unit:d6390cff08c414de79 -->
<a id="u-d6390cff08c414de79"></a>

## Ошибки и самостоятельная проверка

<!-- /unit:d6390cff08c414de79 -->

<!-- unit:4e5b3cb104cb3b10b3 -->
<a id="u-4e5b3cb104cb3b10b3"></a>

Частые ошибки: перепутанные left/right, joints другого человека, missing ankle из-за crop, завышенная confidence occluded point, нарушенный inverse resize, другая skeleton topology. Смотреть predicted skeleton поверх original image и score по occlusion/scale, а не только среднее.

<!-- /unit:4e5b3cb104cb3b10b3 -->

<!-- unit:36ab9df9c4e771644c -->
<a id="u-36ab9df9c4e771644c"></a>

1. Входит ли $v=1$ в формулу с v>0? **Ответ:** да, это размеченная закрытая точка.
2. Чем one-stage pose отличается от top-down crop system? **Ответ:** outputs получаются совместной head без обязательного отдельного forward по каждому crop.
3. Почему OKS зависит от scale? **Ответ:** одинаковый pixel shift имеет разную относительную величину на большом и малом человеке.

<!-- /unit:36ab9df9c4e771644c -->

<!-- unit:fcc77590ac6ff0c934 -->
<a id="u-fcc77590ac6ff0c934"></a>

**Практика:** пересчитать PCK/OKS examples, выполнить flip четырёх left/right joints и обратный transform crop. Затем сравнить ошибки visible/occluded и small/large людей при фиксированном split по scenes/videos.

<!-- /unit:fcc77590ac6ff0c934 -->

<!-- unit:c3617bb01c6ed787f0 -->
<a id="u-c3617bb01c6ed787f0"></a>

**Схема курса:** image/person regions → keypoint heatmaps or coordinates → visibility/confidence filtering → pose grouping → coordinate-aware transforms → OKS evaluation

<!-- /unit:c3617bb01c6ed787f0 -->

<!-- unit:35853b846a42810ca7 -->
<a id="u-35853b846a42810ca7"></a>

<!-- latex-display-v1 -->

<!-- /unit:35853b846a42810ca7 -->

<!-- unit:e5ae5b2a53c1d94232 -->
<a id="u-e5ae5b2a53c1d94232"></a>

<!-- source-depth-S7-Д -->

<!-- /unit:e5ae5b2a53c1d94232 -->

## Архитектуры: подробные схемы

### Top-down pose

![Top-down pose](<../оформление/course/arch-pose-topdown.svg>)

Сначала локализуется человек, затем keypoint head работает в его crop. Предсказанные координаты возвращаются в исходное изображение. Ошибка detector может ограничивать pose quality ещё до keypoint network.

### Bottom-up pose

![Bottom-up pose](<../оформление/course/arch-pose-bottomup.svg>)

Ключевые точки сначала обнаруживаются на общей сетке, затем группируются по людям. Хорошая локализация отдельной точки ещё не гарантирует правильное присвоение экземпляру.

### One-stage pose head

![One-stage pose head](<../оформление/course/arch-pose-onestage.svg>)

Head одновременно задаёт boxes и keypoints без отдельного crop-model этапа. Coordinate representation и matching необходимо проверить у выбранного artifact; само имя YOLO-pose не определяет SimCC.

### SimpleBaseline pose

![SimpleBaseline pose](<../оформление/course/arch-simplebaseline.svg>)

Сильный backbone и upsampling создают heatmaps для стандартизированного crop. Обратный geometry transform необходим для исходных координат.

### HigherHRNet

![HigherHRNet](<../оформление/course/arch-higherhrnet.svg>)

Bottom-up вариант добавляет головы и grouping machinery к backbone. Сохранение spatial resolution и назначение точки человеку остаются разными задачами.

### ViTPose

![ViTPose](<../оформление/course/arch-vitpose.svg>)

Transformer features возвращаются на spatial grid перед pose head. Точность координат зависит от crop, resolution, decoding и training targets, а не только от названия backbone.

### RTMPose / coordinate classification

![RTMPose / coordinate classification](<../оформление/course/arch-rtmpose.svg>)

Раздельные распределения по x и y сокращают число scores относительно полной 2D сетки, но не хранят все joint зависимости координат. Конкретный design, bins и decoding входят в конфигурацию.

### BlazePose / landmark pipeline

![BlazePose / landmark pipeline](<../оформление/course/arch-blazepose.svg>)

ROI и landmarks образуют отдельные этапы; tracking/smoothing меняют temporal поведение. Набор landmarks и coordinate contract проверяются у конкретного artifact.

### HRNet

![HRNet](<../оформление/course/arch-hrnet.svg>)

HRNet сохраняет high-resolution ветвь и обменивает информацию между разрешениями. Конкретная task head задаёт heatmaps или классы; grouping людей не возникает автоматически из backbone.

## Результат главы и переход дальше

Ответьте на вопрос: **Как связать положение ключевых точек с конкретным объектом?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

[Маршрут модуля](<README.md>) · [Происхождение и точные места переноса](<../editorial/coverage.md>)
