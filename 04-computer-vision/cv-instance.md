# Instance segmentation: Mask R-CNN и prototype masks

**Уровень:** общее ядро.

**Предпосылки:** [Изображение, карты признаков и блоки модели](<cv-input.md>), [Классификация, CNN и residual learning](<cv-classification.md>), [Detection: множества объектов, IoU, AP и архитектуры](<cv-detection.md>), [Semantic segmentation: FCN, U-Net и DeepLab](<cv-semantic.md>)

**Используют:** Выбранная прикладная задача и итоговая проверка результата.

**Учебный вопрос:** Как отличить экземпляры одного класса и получить маску каждого?





## Карта вычисления и границы этапов

![Instance segmentation: Mask R-CNN и prototype masks: входы, преобразования и выходы](<../оформление/course/cv-instance.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Сравните формы и смысл каждого перехода с формулами главы. Выход преобразования и доказательство его качества — разные объекты: последняя проверка использует отдельный протокол оценки.

<!-- unit:0e5e047d2ea5cd463a -->
<a id="u-0e5e047d2ea5cd463a"></a>

## Отличие от semantic segmentation

<!-- /unit:0e5e047d2ea5cd463a -->

<!-- unit:a86b671ea41afaf667 -->
<a id="u-a86b671ea41afaf667"></a>

Output — набор `(class,score,mask)` по экземплярам, часто ещё boxes. Для двух людей нужны две отдельные masks, даже если они соприкасаются. Semantic class map объединила бы их в один class region. Перекрывающиеся instance masks допустимы как representation; правила visible/amodal annotation определяются dataset.

<!-- /unit:a86b671ea41afaf667 -->

<!-- unit:9a84df3ae24803b9b2 -->
<a id="u-9a84df3ae24803b9b2"></a>

GT обычно хранит polygon или raster/RLE mask на instance. Box может быть получен из mask, но box annotation не определяет точную маску: внутри прямоугольника часто много background. Для amodal task требуется отдельно обозначить невидимую часть; ordinary visible segmentation нельзя оценивать по другому протоколу.

<!-- /unit:9a84df3ae24803b9b2 -->

<!-- unit:21ef86b0254d550fe4 -->
<a id="u-21ef86b0254d550fe4"></a>

Mask IoU — отношение общего количества foreground pixels к union. Mask AP использует ranked instances и matching по **mask IoU** при заданных thresholds. Box AP использует rectangles. Хороший box может сочетаться с плохой mask, особенно на тонких объектах, отверстиях и соприкасающихся instances. Поэтому отчитываться отдельно.

<!-- /unit:21ef86b0254d550fe4 -->

<!-- unit:093e9da8d79e96108a -->
<a id="u-093e9da8d79e96108a"></a>

Пример: два объекта находятся в одном большом predicted mask. Semantic foreground overlap может быть высоким, но instance evaluation получит максимум одно matched GT и пропуск второго. Другая ошибка — разрезать один объект на несколько masks: duplicates не становятся несколькими TP.

<!-- /unit:093e9da8d79e96108a -->

<!-- unit:22922523e1b4c3822f -->
<a id="u-22922523e1b4c3822f"></a>

### Instance ID не class ID и не порядковый номер score

<!-- /unit:22922523e1b4c3822f -->

<!-- unit:11b72a823df9c37205 -->
<a id="u-11b72a823df9c37205"></a>

Три кошки имеют один semantic class, но три identities внутри кадра. Можно хранить три binary masks или одну карту instance IDs с отдельным соответствием «ID→class». Номер instance условен: перестановка 1 и 2 при согласованных масках не меняет физический ответ. Для tracking постоянство ID во времени — уже дополнительное требование, которого обычная одиночная instance segmentation не обещает.

<!-- /unit:11b72a823df9c37205 -->

<!-- unit:86ece782c6fa2df7c3 -->
<a id="u-86ece782c6fa2df7c3"></a>

Если две видимые части одной кошки разделены предметом, instance annotation может относить оба островка к одной mask. Обычное connected-components разбиение ошибочно сделало бы две кошки. Если невидимая середина должна дорисовываться, это amodal supervision: нельзя назвать ошибкой visible mask за то, что она не предсказала невидимую область.

<!-- /unit:86ece782c6fa2df7c3 -->

<!-- unit:3ca959d254723538a5 -->
<a id="u-3ca959d254723538a5"></a>

### Polygon и RLE: сохраняем примеры исходника с поправкой

<!-- /unit:3ca959d254723538a5 -->

<!-- unit:478bd52b3f2e9e2146 -->
<a id="u-478bd52b3f2e9e2146"></a>

Polygon задаёт координаты контура, которые затем растеризуются. Число точек меняется, нормированные x/y переводятся через ширину/высоту. Несколько polygons могут описывать несколько частей объекта в поддерживаемом формате. RLE непосредственно кодирует raster, поэтому удобен для сложных областей; выбор между ними не сводится к формуле «polygon только без перекрытий, RLE только при overlap». Разные instance masks могут перекрываться независимо от способа хранения.

<!-- /unit:478bd52b3f2e9e2146 -->

<!-- unit:11271574862af664ca -->
<a id="u-11271574862af664ca"></a>

Рассмотрим одномерный пример mask=[0,0,1,1,1,0,0,1,1]. В формате alternating run counts, начинающемся с нулей, правильная последовательность — [2,3,2,2]. Она декодируется как два нуля, три единицы, два нуля, две единицы. Альтернативный JSON counts=[0,2,3,2,2] кодирует иную последовательность и не является эквивалентной записью. Начальный 0 нужен, если первый элемент mask — единица.

<!-- /unit:11271574862af664ca -->

<!-- unit:f155a9b9867c244cc9 -->
<a id="u-f155a9b9867c244cc9"></a>

Для двумерного COCO RLE нужно дополнительно знать size и column-major порядок обхода; compressed counts имеют своё кодирование. Универсальные пары (start,length) из учебного примера — другой вариант RLE, их нельзя без конвертации передать как COCO counts. Сумма uncompressed runs должна равняться H·W. Это позволяет обнаружить разрушенную маску ещё до обучения. [Официальный COCO mask API](https://github.com/cocodataset/cocoapi/blob/master/common/maskApi.c).

<!-- /unit:f155a9b9867c244cc9 -->

<!-- unit:0a72e16087b8eb59b6 -->
<a id="u-0a72e16087b8eb59b6"></a>

### Почему высокий box AP и mask AP могут расходиться

<!-- /unit:0a72e16087b8eb59b6 -->

<!-- unit:721f1d69deeed4bfab -->
<a id="u-721f1d69deeed4bfab"></a>

Два box совпадают идеально, но один объект имеет форму кольца. Если предсказать весь прямоугольник foreground, hole заполнится и mask overlap ухудшится. Для тонкого инструмента большая часть box вообще background. И наоборот, небольшое несовпадение прямоугольников не задаёт однозначного изменения mask IoU: распределение foreground внутри них важно. Поэтому утверждение «mask metric всегда строже» не является универсальным порядком чисел; это разные области сравнения.

<!-- /unit:721f1d69deeed4bfab -->

<!-- unit:1c89a473c96182c04d -->
<a id="u-1c89a473c96182c04d"></a>

Score ранжирует instances при AP. Если class confidence высокий, а mask плохая, такой candidate может рано принести FP на строгом threshold. Отдельная оценка mask quality, как в Mask Scoring R-CNN, пытается согласовать ranking с качеством mask. Это изменение confidence modelling, а не автоматическое улучшение её пиксельной формы.

<!-- /unit:1c89a473c96182c04d -->

<!-- unit:0afd226d9f33221103 -->
<a id="u-0afd226d9f33221103"></a>

## Mask R-CNN и RoIAlign

<!-- /unit:0afd226d9f33221103 -->

<!-- unit:878eb60af4e99d4d57 -->
<a id="u-878eb60af4e99d4d57"></a>

Mask R-CNN расширяет detector branch предсказанием маски для region. Shared backbone/FPN → proposals → region features → classification/box/mask branches. В классическом варианте mask branch даёт K class-specific binary masks на RoI, а supervision считается для channel true class. Sigmoid по каждому pixel не является softmax-конкуренцией разных классов в каждой mask. Есть class-agnostic variants, их нельзя автоматически описывать исходной K-channel схемой. [Mask R-CNN](https://arxiv.org/abs/1703.06870).

<!-- /unit:878eb60af4e99d4d57 -->

<!-- unit:20a5559f9154607934 -->
<a id="u-20a5559f9154607934"></a>

RoIPool округляет coordinates/границы bins, что создаёт spatial quantization error. RoIAlign выбирает fractional positions и использует bilinear interpolation; затем агрегирует samples по определённой sampling policy. Выход — fixed spatial tensor, например $C\times 14\times 14$, из которого mask branch может получить higher-resolution output. Это не обязательно flatten vector.

<!-- /unit:20a5559f9154607934 -->

<!-- unit:beead07f71373d3bd8 -->
<a id="u-beead07f71373d3bd8"></a>

Для четырёх значений $f_{00}=1,\qquad f_{10}=3,\qquad f_{01}=5,\qquad f_{11}=7$ в точке $x=0.25$,$y=0.5$:

<!-- /unit:beead07f71373d3bd8 -->

<!-- unit:d8f7714ef1f3fe23db -->
<a id="u-d8f7714ef1f3fe23db"></a>

$$
\begin{aligned}f(x,y)&=(1-x)(1-y)f_{00}+x(1-y)f_{10}+(1-x)yf_{01}+xyf_{11}\\&=0.375\cdot1+0.125\cdot3+0.375\cdot5+0.125\cdot7=3.5.\end{aligned}
$$

<!-- /unit:d8f7714ef1f3fe23db -->

<!-- unit:f98dfd31716f60f058 -->
<a id="u-f98dfd31716f60f058"></a>

Это простой пример interpolation внутри одной клетки feature map. В реальном RoIAlign важны image→feature stride, coordinate alignment offset, sampling ratio и boundary handling. Различия half-pixel conventions способны сдвинуть mask без очевидной shape ошибки.

<!-- /unit:f98dfd31716f60f058 -->

<!-- unit:c2d1b3e88b02c338e2 -->
<a id="u-c2d1b3e88b02c338e2"></a>

Общий training objective можно записать $\mathcal{L}=\lambda_{\mathrm{cls}}\mathcal{L}_{\mathrm{cls}}+\lambda_{\mathrm{box}}\mathcal{L}_{\mathrm{box}}+\lambda_{\mathrm{mask}}\mathcal{L}_{\mathrm{mask}}$. Mask BCE на foreground RoI учит форму, classification определяет category, box loss — localization. Loss weights/normalization влияют на баланс; нельзя утверждать, что ветви «не мешают друг другу» только из-за разных heads: shared features получают сумму gradients.

<!-- /unit:c2d1b3e88b02c338e2 -->

<!-- unit:cf3967cd789959f1ed -->
<a id="u-cf3967cd789959f1ed"></a>

### Почему RoIAlign нужен именно для границ

<!-- /unit:cf3967cd789959f1ed -->

<!-- unit:bd610aa0ed96457919 -->
<a id="u-bd610aa0ed96457919"></a>

Пусть region начинается в координате 10.4 input image, а feature map имеет stride4. Его начало в feature coordinates — 2.6 до возможных alignment offsets. Округление до 2 или 3 меняет, какие признаки извлекаются; после обратного увеличения ошибка становится заметной на input mask. RoIAlign сохраняет fractional sampling и интерполирует соседние values, а не требует считать исходную картинку заново для каждого crop.

<!-- /unit:bd610aa0ed96457919 -->

<!-- unit:b0a4bf219ea51f7e5b -->
<a id="u-b0a4bf219ea51f7e5b"></a>

В одном bin можно брать несколько sample points и усреднять или агрегировать их по выбранной policy. Поэтому bilinear interpolation четырёх соседей — один этап, не всё определение RoIAlign. За ним остаются границы region, bins, sampling ratio и отображение координат. Два kernels с разными half-pixel conventions могут выдавать одинаковые shapes и сдвинутые детали.

<!-- /unit:b0a4bf219ea51f7e5b -->

<!-- unit:69e90be5040b683f82 -->
<a id="u-69e90be5040b683f82"></a>

В исходном Mask R-CNN mask head получала пространственные признаки около 14×14, применяла несколько convolutions и transposed conv для 28×28 output. Свёртки внутри head используют layout, поэтому заменить вход одним pooled вектором без другой архитектуры нельзя. Class branch решает, какой channel mask использовать; mask branch учит binary foreground внутри этого channel для positive RoI.

<!-- /unit:69e90be5040b683f82 -->

<!-- unit:2e6b1a4e8182723bce -->
<a id="u-2e6b1a4e8182723bce"></a>

### От binary loss одной RoI к общему обучению

<!-- /unit:2e6b1a4e8182723bce -->

<!-- unit:0dce0c5a197e15f03e -->
<a id="u-0dce0c5a197e15f03e"></a>

Пусть target class региона — «человек», а head выдаёт K каналов. В исходной class-specific схеме mask BCE считается для канала человека. Если вместо этого применять softmax по K classes на каждом mask pixel, возникнет другая задача: конкуренция категорий внутри региона. Binary mask может отмечать background внутри box; category supervision уже существует в class branch.

<!-- /unit:0dce0c5a197e15f03e -->

<!-- unit:23259bca76c72cac4c -->
<a id="u-23259bca76c72cac4c"></a>

Для одного pixel target=1 и p=0.8 BCE≈0.223144; для target=0 и той же p ошибка≈1.609438. Loss усредняется по предусмотренным mask pixels и positive RoIs. Если число positives в batch меняется, normalization влияет на относительный масштаб mask contribution. Shared backbone получает сумму производных нескольких ветвей: хорошо разделённые heads не делают gradients автоматически согласованными.

<!-- /unit:23259bca76c72cac4c -->

<!-- unit:ad5ccf8bd2b0e53c27 -->
<a id="u-ad5ccf8bd2b0e53c27"></a>

При training region может сопоставляться с GT и получать target crop. На inference box уже предсказан и может быть смещён. Этот сдвиг меняет вход mask head и её обратное размещение в картинке. Oracle GT boxes помогают понять, ограничивает ли качество localization или сама условная сегментация; такой oracle score не является обычным inference score.

<!-- /unit:ad5ccf8bd2b0e53c27 -->

<!-- unit:d5b59bcb9927501e40 -->
<a id="u-d5b59bcb9927501e40"></a>

## Prototype masks: YOLACT и родственная идея

<!-- /unit:d5b59bcb9927501e40 -->

<!-- unit:14e342150261893598 -->
<a id="u-14e342150261893598"></a>

В YOLACT одна ветвь генерирует shared prototype maps, другая — coefficients для каждой detection. Пусть R — число prototypes, N — число candidates, H_m/W_m — mask resolution:

<!-- /unit:14e342150261893598 -->

<!-- unit:8c14cd7147e0a51d17 -->
<a id="u-8c14cd7147e0a51d17"></a>

$$
\begin{gathered}P\in\mathbb{R}^{R\times H_m\times W_m},\quad C\in\mathbb{R}^{N\times R},\quad\widetilde P\in\mathbb{R}^{R\times(H_mW_m)},\\Z_{\mathrm{mask}}=C\widetilde P\in\mathbb{R}^{N\times(H_mW_m)},\qquad M=\sigma(Z_{\mathrm{mask}}).\end{gathered}
$$

<!-- /unit:8c14cd7147e0a51d17 -->

<!-- unit:b9222d734ac3378cfc -->
<a id="u-b9222d734ac3378cfc"></a>

Prototype — basis map, а не обязательная готовая mask отдельного объекта. Coefficients могут комбинировать несколько maps; отрицательные coefficients тоже могут быть осмысленны до sigmoid. [YOLACT](https://arxiv.org/abs/1904.02689).

<!-- /unit:b9222d734ac3378cfc -->

<!-- unit:de56fe01222d15e60c -->
<a id="u-de56fe01222d15e60c"></a>

Маленький пример с R=2 и двумя pixels: $P_{1}=\left(1,\,0\right)$, $P_{2}=\left(0,\,1\right)$. Coefficients объекта A $\left(2,\,-2\right)$ дают logits $\left(2,\,-2\right)$, probabilities≈(0.8808,0.1192). Объект B $\left(-2,\,2\right)$ — противоположную mask. Общие prototypes не мешают разным masks благодаря coefficients.

<!-- /unit:de56fe01222d15e60c -->

<!-- unit:532d311c8493bdf3db -->
<a id="u-532d311c8493bdf3db"></a>

Память scales как $R H_mW_m+NR$ до assembly вместо отдельной тяжёлой spatial head на каждую RoI. Но после assembly всё равно может потребоваться $N H_mW_m$ output/intermediate memory. При большом N mask postprocessing может стать bottleneck; лозунг «R coefficients, значит masks почти ничего не стоят» неполон.

<!-- /unit:532d311c8493bdf3db -->

<!-- unit:92cc4bdaf28c30e84c -->
<a id="u-92cc4bdaf28c30e84c"></a>

YOLO-seg использует task-specific structures, в которых может быть родственная prototype+coefficient механика. Надо проверять конкретную head, resolution, crop, upsampling и порядок NMS/mask construction. Нельзя считать каждую YOLO-seg точной копией YOLACT или путать её output с semantic `*-sem` model.

<!-- /unit:92cc4bdaf28c30e84c -->

<!-- unit:79bf61b33784b85741 -->
<a id="u-79bf61b33784b85741"></a>

### Полный путь prototype assembly и его производные

<!-- /unit:79bf61b33784b85741 -->

<!-- unit:f53d0c623a613f1c6d -->
<a id="u-f53d0c623a613f1c6d"></a>

Сначала protonet выдаёт R shared maps. Другой head для candidate i вычисляет coefficients c_i, class и часто box. После отбора остаются N объектов, их coefficients образуют матрицу C. Умножение C на flattened prototypes сразу строит N mask-logit maps; затем sigmoid, crop/resize и threshold по точной inference-схеме. Один prototype не обязан совпадать с одним человеком: комбинации могут выделять разные regions.

<!-- /unit:f53d0c623a613f1c6d -->

<!-- unit:1714c1087e0c7be42b -->
<a id="u-1714c1087e0c7be42b"></a>

Пусть G — производная loss по assembled logits формы N×(H_mW_m). Тогда

<!-- /unit:1714c1087e0c7be42b -->

<!-- unit:cbfefe65cf3963c6ec -->
<a id="u-cbfefe65cf3963c6ec"></a>

$$
\nabla_C\mathcal L=G\widetilde P^\top,\qquad
\nabla_{\widetilde P}\mathcal L=C^\top G.
$$

<!-- /unit:cbfefe65cf3963c6ec -->

<!-- unit:a6c9a9a7ac7b1217cd -->
<a id="u-a6c9a9a7ac7b1217cd"></a>

Все объекты возвращают вклад в общие maps, а coefficients получают собственный signal. Поэтому prototypes обучаются совместно для изображения, а не отдельно по одному шаблону на class. Если box crop отрезал нужный край, pixel вне crop может не иметь прежнего пути gradient; детали training crop и normalization важны.

<!-- /unit:a6c9a9a7ac7b1217cd -->

<!-- unit:261e1418ef2c4e0170 -->
<a id="u-261e1418ef2c4e0170"></a>

При R=32,H_m=W_m=160 shared maps содержат 819200 values. Сто объектов добавляют всего 3200 coefficients, но собранные masks уже содержат 2560000 values. При float32 это 3.2768 MB для prototypes, 0.0128 MB для coefficients и 10.24 MB для одного полного assembled tensor. Выгода в shared computation не означает отсутствие стоимости окончательных масок, их resize и копирования.

<!-- /unit:261e1418ef2c4e0170 -->

<!-- unit:58bc8b2cc935833ac6 -->
<a id="u-58bc8b2cc935833ac6"></a>

В YOLACT есть detection outputs, поэтому исходная формулировка «one-stage без явной detection, например YOLACT» неверна. One-stage означает отсутствие отдельной классической второй RoI-стадии, а не запрет boxes. Fast NMS YOLACT и Matrix NMS SOLOv2 также нельзя считать одним методом только из-за матричной параллельной реализации подавления.

<!-- /unit:58bc8b2cc935833ac6 -->

<!-- unit:32185b7bd5a7def9f1 -->
<a id="u-32185b7bd5a7def9f1"></a>

### SOLO, bottom-up и query-based подходы из разбираемого примера

<!-- /unit:32185b7bd5a7def9f1 -->

<!-- unit:fcfeb543264114d2e9 -->
<a id="u-fcfeb543264114d2e9"></a>

SOLO назначает predictions по пространственным locations и масштабам объектов. Category branch выбирает активные locations, mask branch задаёт соответствующие masks; центры дают способ организации множества outputs без обязательного box predictor. У SOLOv2 mask kernel branch предсказывает динамические коэффициенты kernels, применяемые к shared mask features. Matrix NMS — часть этого направления. Формулировка «у SOLO вообще нет NMS» слишком сильна: mask suppression остаётся важной частью опубликованных designs. [SOLO](https://arxiv.org/abs/1912.04488), [SOLOv2](https://arxiv.org/abs/2003.10152).

<!-- /unit:fcfeb543264114d2e9 -->

<!-- unit:29f23948a66e6e523e -->
<a id="u-29f23948a66e6e523e"></a>

Bottom-up embedding подход сначала назначает pixel features/центры/связи, затем группирует их в экземпляры. Он может столкнуться с ошибками grouping даже при хороших semantic classes. DeepMask из разбираемого примера — mask proposal по patches, а не канонический пример группировки per-pixel embeddings: исторические методы не стоит помещать в неверную категорию ради короткой схемы.

<!-- /unit:29f23948a66e6e523e -->

<!-- unit:5b3092568b0a8e6e93 -->
<a id="u-5b3092568b0a8e6e93"></a>

PointRend уточняет mask в выбранных points, что позволяет тратить вычисления на неопределённые boundaries вместо одинакового повышения resolution везде. Mask2Former использует masks и queries с masked attention для разных segmentation tasks; SAM отвечает на prompt и не присваивает автоматически фиксированные semantic labels каждому найденному instance. Эти альтернативы меняют механизм выбора region, refinement или conditioning, а не только backbone brand.

<!-- /unit:5b3092568b0a8e6e93 -->

<!-- unit:21a4e88560f5ee23f2 -->
<a id="u-21a4e88560f5ee23f2"></a>

## Соприкасающиеся объекты, boundaries и panoptic

<!-- /unit:21a4e88560f5ee23f2 -->

<!-- unit:e4a1fb535fcd660398 -->
<a id="u-e4a1fb535fcd660398"></a>

Для тесно стоящих объектов detector assignment, crop и resolution взаимосвязаны. Пропущенный box может означать отсутствие соответствующей mask. Низкое prototype resolution теряет тонкие детали; ошибочный crop отрежет нужную часть даже хороших logits. Ошибки нужно локализовать по этапам, иногда с oracle boxes.

<!-- /unit:e4a1fb535fcd660398 -->

<!-- unit:4e76d21c88a12d8a65 -->
<a id="u-4e76d21c88a12d8a65"></a>

При segmentation вещей и сплошных regions различают **things** (люди, машины с instance identities) и **stuff** (небо, дорога). Panoptic segmentation объединяет семантику и экземпляры в согласованную разметку. Для matched segments panoptic quality:

<!-- /unit:4e76d21c88a12d8a65 -->

<!-- unit:3768921400765e2557 -->
<a id="u-3768921400765e2557"></a>

$$
\begin{aligned}\mathrm{PQ}&=\frac{\sum_{(p,g)\in\mathrm{TP}}\operatorname{IoU}(p,g)}{|\mathrm{TP}|+\tfrac12|\mathrm{FP}|+\tfrac12|\mathrm{FN}|}=\mathrm{SQ}\cdot\mathrm{RQ},\\\mathrm{SQ}&=\frac{\sum_{(p,g)\in\mathrm{TP}}\operatorname{IoU}(p,g)}{|\mathrm{TP}|},\qquad\mathrm{RQ}=\frac{|\mathrm{TP}|}{|\mathrm{TP}|+\tfrac12|\mathrm{FP}|+\tfrac12|\mathrm{FN}|}.\end{aligned}
$$

<!-- /unit:3768921400765e2557 -->

<!-- unit:75a392848bc0099e30 -->
<a id="u-75a392848bc0099e30"></a>

Exact matching/void rules берутся из benchmark. При двух TP с IoU0.8 и0.6, одном FP и одном FN: SQ0.7, RQ2/3, PQ≈0.4667. Это дополнительное углубление плана, а не автоматическая замена mask AP.

<!-- /unit:75a392848bc0099e30 -->

<!-- unit:bc8e2c3cdade4c7836 -->
<a id="u-bc8e2c3cdade4c7836"></a>

### Crop, upsampling и threshold не переставляются произвольно

<!-- /unit:bc8e2c3cdade4c7836 -->

<!-- unit:f4e9fb5a96394284f8 -->
<a id="u-f4e9fb5a96394284f8"></a>

Внутри bbox может находиться фрагмент другого объекта. Crop уменьшает область кандидата, но не заменяет разделение foreground внутри неё. Если predicted box уже слишком узок, hard crop необратимо удалит часть хорошей mask. При низком resolution тонкая структура может занимать долю pixel; threshold до upsampling и threshold после interpolation дают разные outlines. В отчёте нужно указать, какая continuous map и в каких coordinates преобразуется в binary mask.

<!-- /unit:f4e9fb5a96394284f8 -->

<!-- unit:4794c15942aebb56ff -->
<a id="u-4794c15942aebb56ff"></a>

При оценке overlap часть ошибок связана с detection matching, часть — с boundary, часть — с merging/splitting identities. Эти виды полезно показывать отдельно. Две правильные semantic regions, объединённые одним instance ID, всё равно дают один matched объект вместо двух. Два практически одинаковых masks для одного объекта не превращаются в два TP.

<!-- /unit:4794c15942aebb56ff -->

<!-- unit:3f89973779b478d48b -->
<a id="u-3f89973779b478d48b"></a>

Loss по boundary gradients из рассматриваемого примера, например $\sum_i|\nabla M_i-\nabla G_i|$, — только один возможный суррогат с выбранным дискретным operator. Он может дополнить BCE/Dice, но не является универсальным стандартом под названием Boundary Loss. Нужно определить нормировку, smoothing, resolution и обработку ignored pixels. Усиление высокочастотных границ без контроля способно усилить шум разметки.

<!-- /unit:3f89973779b478d48b -->

<!-- unit:df2c0230781717008d -->
<a id="u-df2c0230781717008d"></a>

Для panoptic output каждому pixel обычно назначают согласованную class/segment identity по protocol, поэтому конфликтующие instance masks нужно разрешать. Things получают отдельные identities, stuff объединяется по правилам dataset. PQ разделяет matching и overlap: при SQ=0.7 и RQ=2/3 итог 0.466667. Улучшить только форму matched masks при прежних пропусках — не то же, что улучшить число найденных объектов; разложение помогает выбрать следующий эксперимент.

<!-- /unit:df2c0230781717008d -->

<!-- unit:463cc8a221eda6a216 -->
<a id="u-463cc8a221eda6a216"></a>

### Promptable concept segmentation как соседняя постановка

<!-- /unit:463cc8a221eda6a216 -->

<!-- unit:df136f8105a7bc026e -->
<a id="u-df136f8105a7bc026e"></a>

Closed-set instance segmentation получает фиксированный список классов и выдаёт instance masks по этому ontology. SAM 3 расширяет интерфейс: короткая noun-phrase или visual exemplar может задавать концепт, а модель возвращает masks и instance identities для всех соответствующих объектов. Meta оценивает эту постановку на SA-Co / Promptable Concept Segmentation; её score нельзя напрямую подменять COCO mask AP, потому что prompts, vocabulary и единица оценки различаются.

<!-- /unit:df136f8105a7bc026e -->

<!-- unit:d0b9edb7fba47109fc -->
<a id="u-d0b9edb7fba47109fc"></a>

Это пример того, почему тип подсказки входит в контракт задачи. Для phrase «красный зонт» нужны критерии совпадения concept и все instances; для points/boxes пользователь задаёт конкретный region; для фиксированных labels проверяется closed-set class list. Отдельно учитывайте omitted instances, false matches и качество prompt.

<!-- /unit:d0b9edb7fba47109fc -->

<!-- unit:ce825d2677d7988ce8 -->
<a id="u-ce825d2677d7988ce8"></a>

**Источник:** [SAM 3 research release](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/). Авторы сообщают benchmark gains на собственном SA-Co и интерактивных segmentation tasks; перенос на другую разметку требует отдельного протокола.

<!-- /unit:ce825d2677d7988ce8 -->

## Архитектуры: подробные схемы

### Mask R-CNN

![Mask R-CNN](<../оформление/course/arch-mask-rcnn.svg>)

RoIAlign интерполирует признаки в непрерывных координатах, уменьшая ошибку грубой квантизации region. Mask branch выдаёт локальную маску экземпляра; её нужно отобразить обратно в координаты изображения.

### YOLACT / prototype masks

![YOLACT / prototype masks](<../оформление/course/arch-yolact.svg>)

Для одного объекта коэффициенты комбинируют общие prototype maps. Это позволяет разделить общие пространственные признаки и экземплярные параметры. Prototype не обязан иметь интерпретацию отдельной семантической части.

### SOLO

![SOLO](<../оформление/course/arch-solo.svg>)

Объект организует предсказание через location и scale, а не обязательно через предварительную box. Category branch выбирает активные позиции; mask branch определяет пространственную форму.

### SOLOv2

![SOLOv2](<../оформление/course/arch-solov2.svg>)

Dynamic kernels создают разные masks из общих features. Matrix NMS является mask-selection механизмом; отсутствие классической box-RoI стадии не означает отсутствие suppression.

### DeepMask

![DeepMask](<../оформление/course/arch-deepmask.svg>)

DeepMask формирует mask proposal для patch и score наличия объекта. Это proposal design; его нельзя описывать как каноническое grouping всех per-pixel embeddings.

### PointRend

![PointRend](<../оформление/course/arch-pointrend.svg>)

PointRend тратит вычисления на выбранные точки высокой неопределённости, используя интерполяцию features. Он уточняет маску и требует исходного coarse prediction; сам по себе не заменяет всю систему локализации.

### SAM: promptable masks

![SAM: promptable masks](<../оформление/course/arch-sam.svg>)

Image и prompt получают отдельные representations, затем mask decoder связывает их. Promptable mask не является автоматически полной semantic map пользовательского набора классов.

### SAM 3: concept segmentation

![SAM 3: concept segmentation](<../оформление/course/arch-sam3.svg>)

Это функциональный интерфейс Promptable Concept Segmentation: концепт задаётся phrase либо exemplar, выход содержит matching instances и masks. Схема не приписывает checkpoint неуказанное число слоёв или универсальные COCO gains.

### Mask2Former

![Mask2Former](<../оформление/course/arch-mask2former.svg>)

Queries предсказывают mask и class; masked attention ограничивает доступ к spatial features по промежуточным masks. Финальная сборка выхода зависит от выбранной segmentation formulation.

<!-- unit:7a3c8ebe34f7be6bbc -->
<a id="u-7a3c8ebe34f7be6bbc"></a>

## Самопроверка и практика

<!-- /unit:7a3c8ebe34f7be6bbc -->

<!-- unit:204f93e7587b9f5b0d -->
<a id="u-204f93e7587b9f5b0d"></a>

1. Почему высокий box AP не доказывает mask AP? **Ответ:** локализация прямоугольника не проверяет точную shape.
2. Что исправляет RoIAlign? **Ответ:** часть coordinate quantization при извлечении region features; missing information backbone он не восстанавливает.
3. Должен ли prototype соответствовать одному объекту? **Ответ:** нет, это shared component linear combination.
4. Зачем oracle boxes? **Ответ:** помогают отделить ошибки detector от conditional mask prediction.

<!-- /unit:204f93e7587b9f5b0d -->

<!-- unit:c5e2e682c1f1e85a66 -->
<a id="u-c5e2e682c1f1e85a66"></a>

**Практика:** пересчитать bilinear example и mask coefficients. Нарисовать два соприкасающихся objects и сравнить outputs merged-instance, duplicate-instance и correct-instance. В эксперименте сохранить один split и менять только resolution/один mask-head factor, измеряя box AP, mask AP, boundary cases и полную latency.

<!-- /unit:c5e2e682c1f1e85a66 -->

<!-- unit:3ae1bbdc70b2720e2e -->
<a id="u-3ae1bbdc70b2720e2e"></a>

**Схема курса:** image → proposals/queries → per-instance class and box → RoI or prototype mask head → per-object masks/IDs → matching and mask AP

<!-- /unit:3ae1bbdc70b2720e2e -->

<!-- unit:ade567eddd394cf1a3 -->
<a id="u-ade567eddd394cf1a3"></a>

<!-- latex-display-v1 -->

<!-- /unit:ade567eddd394cf1a3 -->

<!-- unit:f33131339304e7d725 -->
<a id="u-f33131339304e7d725"></a>

<!-- source-depth-S7-Г -->

<!-- /unit:f33131339304e7d725 -->

## Результат главы и переход дальше

Ответьте на вопрос: **Как отличить экземпляры одного класса и получить маску каждого?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

[Маршрут модуля](<README.md>) · [Происхождение и точные места переноса](<../editorial/coverage.md>)
