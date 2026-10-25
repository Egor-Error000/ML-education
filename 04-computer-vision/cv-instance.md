# Instance segmentation: Mask R-CNN и prototype masks

**Уровень:** общее ядро.

**Предпосылки:** [Detection: множества объектов, IoU, AP и архитектуры](<cv-detection.md>), [Semantic segmentation: FCN, U-Net и DeepLab](<cv-semantic.md>)

**Используют:** [YOLACT: prototype masks, коэффициенты и crop](<cv-prototypes.md>), [Mask2Former: mask queries и panoptic segmentation](<cv-mask2former.md>)

**Учебный вопрос:** Как отличить экземпляры одного класса и получить маску каждого?

## Карта вычисления и границы этапов

![Instance segmentation: Mask R-CNN и prototype masks: входы, преобразования и выходы](<../оформление/course/cv-instance.svg>)

Маска экземпляра принадлежит конкретному объекту, поэтому два объекта одного класса должны иметь разные выходы. Эта привязка отличает instance segmentation от простой карты классов пикселей.

## Отличие от semantic segmentation

Output — набор `(class,score,mask)` по экземплярам, часто ещё boxes. Для двух людей нужны две отдельные masks, даже если они соприкасаются. Semantic class map объединила бы их в один class region. Перекрывающиеся instance masks допустимы как representation; правила visible/amodal annotation определяются dataset.

GT обычно хранит polygon или raster/RLE mask на instance. Box может быть получен из mask, но box annotation не определяет точную маску: внутри прямоугольника часто много background. Для amodal task требуется отдельно обозначить невидимую часть; ordinary visible segmentation нельзя оценивать по другому протоколу.

Mask IoU — отношение общего количества foreground pixels к union. Mask AP использует ranked instances и matching по **mask IoU** при заданных thresholds. Box AP использует rectangles. Хороший box может сочетаться с плохой mask, особенно на тонких объектах, отверстиях и соприкасающихся instances. Поэтому отчитываться отдельно.

Пример: два объекта находятся в одном большом predicted mask. Semantic foreground overlap может быть высоким, но instance evaluation получит максимум одно matched GT и пропуск второго. Другая ошибка — разрезать один объект на несколько masks: duplicates не становятся несколькими TP.

### Instance ID не class ID и не порядковый номер score (оценка)

Три кошки имеют один semantic class, но три identities внутри кадра. Можно хранить три binary masks или одну карту instance IDs с отдельным соответствием «ID→class». Номер instance условен: перестановка 1 и 2 при согласованных масках не меняет физический ответ. Для tracking постоянство ID во времени — уже дополнительное требование, которого обычная одиночная instance segmentation не обещает.

Если две видимые части одной кошки разделены предметом, instance annotation может относить оба островка к одной mask. Обычное connected-components разбиение ошибочно сделало бы две кошки. Если невидимая середина должна дорисовываться, это amodal supervision: нельзя назвать ошибкой visible mask за то, что она не предсказала невидимую область.

### Полигоны и RLE: два способа хранения маски

Polygon задаёт координаты контура, которые затем растеризуются. Число точек меняется, нормированные x/y переводятся через ширину/высоту. Несколько polygons могут описывать несколько частей объекта в поддерживаемом формате. RLE непосредственно кодирует raster, поэтому удобен для сложных областей; выбор между ними не сводится к формуле «polygon только без перекрытий, RLE только при overlap». Разные instance masks могут перекрываться независимо от способа хранения.

Рассмотрим одномерный пример mask=[0,0,1,1,1,0,0,1,1]. В формате alternating run counts, начинающемся с нулей, правильная последовательность — [2,3,2,2]. Она декодируется как два нуля, три единицы, два нуля, две единицы. Альтернативный JSON counts=[0,2,3,2,2] кодирует иную последовательность и не является эквивалентной записью. Начальный 0 нужен, если первый элемент mask — единица.

Для двумерного COCO RLE нужно дополнительно знать size и column-major порядок обхода; compressed counts имеют своё кодирование. Универсальные пары (start,length) из учебного примера — другой вариант RLE, их нельзя без конвертации передать как COCO counts. Сумма uncompressed runs должна равняться H·W. Это позволяет обнаружить разрушенную маску ещё до обучения. [Официальный COCO mask API](https://github.com/cocodataset/cocoapi/blob/master/common/maskApi.c).

### Почему высокий box AP и mask AP могут расходиться

Два box совпадают идеально, но один объект имеет форму кольца. Если предсказать весь прямоугольник foreground, hole заполнится и mask overlap ухудшится. Для тонкого инструмента большая часть box вообще background. И наоборот, небольшое несовпадение прямоугольников не задаёт однозначного изменения mask IoU: распределение foreground внутри них важно. Поэтому утверждение «mask metric всегда строже» не является универсальным порядком чисел; это разные области сравнения.

Score ранжирует instances при AP. Если class confidence (уверенность по принятой оценке) высокий, а mask плохая, такой candidate может рано принести FP на строгом threshold. Отдельная оценка mask quality, как в Mask Scoring R-CNN, пытается согласовать ranking с качеством mask. Это изменение confidence modelling, а не автоматическое улучшение её пиксельной формы.

## Mask R-CNN и RoIAlign

Mask R-CNN расширяет detector branch предсказанием маски для region. Shared backbone (основная сеть извлечения признаков)/FPN → proposals → region features (признаки) → classification/box/mask branches. В классическом варианте mask branch даёт K class-specific binary masks на RoI, а supervision считается для channel true class. Sigmoid по каждому pixel не является softmax-конкуренцией разных классов в каждой mask. Есть class-agnostic variants, их нельзя автоматически описывать исходной K-channel схемой. [Mask R-CNN](https://arxiv.org/abs/1703.06870).

RoIPool округляет coordinates/границы bins, что создаёт spatial quantization (квантование числового представления) error. RoIAlign выбирает fractional positions и использует bilinear interpolation; затем агрегирует samples по определённой sampling (случайный выбор по распределению) policy. Выход — fixed spatial tensor, например $C\times 14\times 14$, из которого mask branch может получить higher-resolution output. Это не обязательно flatten vector.

Для четырёх значений $f_{00}=1,\qquad f_{10}=3,\qquad f_{01}=5,\qquad f_{11}=7$ в точке $x=0.25$,$y=0.5$:

$$
\begin{aligned}f(x,y)&=(1-x)(1-y)f_{00}+x(1-y)f_{10}+(1-x)yf_{01}+xyf_{11}\\&=0.375\cdot1+0.125\cdot3+0.375\cdot5+0.125\cdot7=3.5.\end{aligned}
$$

Это простой пример interpolation внутри одной клетки feature map (пространственная карта признаков). В реальном RoIAlign важны image→feature (признак) stride (шаг пространственного перемещения операции), coordinate alignment offset, sampling ratio и boundary handling. Различия half-pixel conventions способны сдвинуть mask без очевидной shape ошибки.

Общий training objective (оптимизируемая целевая функция) можно записать $\mathcal{L}=\lambda_{\mathrm{cls}}\mathcal{L}_{\mathrm{cls}}+\lambda_{\mathrm{box}}\mathcal{L}_{\mathrm{box}}+\lambda_{\mathrm{mask}}\mathcal{L}_{\mathrm{mask}}$. Mask BCE на foreground RoI учит форму, classification определяет category, box loss (функция потерь) — localization. Loss weights/normalization (нормировка по выбранному правилу) влияют на баланс; нельзя утверждать, что ветви «не мешают друг другу» только из-за разных heads: shared features получают сумму gradients.

### Почему RoIAlign нужен именно для границ

Пусть region начинается в координате 10.4 input image, а feature map имеет stride4. Его начало в feature coordinates — 2.6 до возможных alignment offsets. Округление до 2 или 3 меняет, какие признаки извлекаются; после обратного увеличения ошибка становится заметной на input mask. RoIAlign сохраняет fractional sampling и интерполирует соседние values, а не требует считать исходную картинку заново для каждого crop.

В одном bin можно брать несколько sample points и усреднять или агрегировать их по выбранной policy. Поэтому bilinear interpolation четырёх соседей — один этап, не всё определение RoIAlign. За ним остаются границы region, bins, sampling ratio и отображение координат. Два kernels с разными half-pixel conventions могут выдавать одинаковые shapes и сдвинутые детали.

В исходном Mask R-CNN mask head получала пространственные признаки около 14×14, применяла несколько convolutions и transposed conv для 28×28 output. Свёртки внутри head используют layout, поэтому заменить вход одним pooled вектором без другой архитектуры нельзя. Class branch решает, какой channel mask использовать; mask branch учит binary foreground внутри этого channel для positive RoI.

### От binary loss одной RoI к общему обучению

Пусть target class региона — «человек», а head выдаёт K каналов. В исходной class-specific схеме mask BCE считается для канала человека. Если вместо этого применять softmax по K classes на каждом mask pixel, возникнет другая задача: конкуренция категорий внутри региона. Binary mask может отмечать background внутри box; category supervision уже существует в class branch.

Для одного pixel target=1 и p=0.8 BCE≈0.223144; для target=0 и той же p ошибка≈1.609438. Loss усредняется по предусмотренным mask pixels и positive RoIs. Если число positives в batch (пакет примеров) меняется, normalization влияет на относительный масштаб mask contribution. Shared backbone получает сумму производных нескольких ветвей: хорошо разделённые heads не делают gradients автоматически согласованными.

При training region может сопоставляться с GT и получать target crop. На inference (применение обученной модели) box уже предсказан и может быть смещён. Этот сдвиг меняет вход mask head и её обратное размещение в картинке. Oracle GT boxes помогают понять, ограничивает ли качество localization или сама условная сегментация; такой oracle score не является обычным inference score.

## Prototype masks: YOLACT и родственная идея

В YOLACT одна ветвь генерирует shared prototype maps, другая — coefficients для каждой detection. Пусть R — число prototypes, N — число candidates, H_m/W_m — mask resolution:

$$
\begin{gathered}P\in\mathbb{R}^{R\times H_m\times W_m},\quad C\in\mathbb{R}^{N\times R},\quad\widetilde P\in\mathbb{R}^{R\times(H_mW_m)},\\Z_{\mathrm{mask}}=C\widetilde P\in\mathbb{R}^{N\times(H_mW_m)},\qquad M=\sigma(Z_{\mathrm{mask}}).\end{gathered}
$$

Prototype — basis map, а не обязательная готовая mask отдельного объекта. Coefficients могут комбинировать несколько maps; отрицательные coefficients тоже могут быть осмысленны до sigmoid. [YOLACT](https://arxiv.org/abs/1904.02689).

Маленький пример с R=2 и двумя pixels: $P_{1}=\left(1,\,0\right)$, $P_{2}=\left(0,\,1\right)$. Coefficients объекта A $\left(2,\,-2\right)$ дают logits (логиты, оценки альтернатив до нормировки) $\left(2,\,-2\right)$, probabilities≈(0.8808,0.1192). Объект B $\left(-2,\,2\right)$ — противоположную mask. Общие prototypes не мешают разным masks благодаря coefficients.

Память scales как $R H_mW_m+NR$ до assembly вместо отдельной тяжёлой spatial head на каждую RoI. Но после assembly всё равно может потребоваться $N H_mW_m$ output/intermediate memory. При большом N mask postprocessing может стать bottleneck; лозунг «R coefficients, значит masks почти ничего не стоят» неполон.

YOLO-seg использует task-specific structures, в которых может быть родственная prototype+coefficient механика. Надо проверять конкретную head, resolution, crop, upsampling (повышение пространственного разрешения) и порядок NMS/mask construction. Нельзя считать каждую YOLO-seg точной копией YOLACT или путать её output с semantic `*-sem` model.

### Полный путь prototype assembly и его производные

Сначала protonet выдаёт R shared maps. Другой head для candidate i вычисляет coefficients c_i, class и часто box. После отбора остаются N объектов, их coefficients образуют матрицу C. Умножение C на flattened prototypes сразу строит N mask-logit maps; затем sigmoid, crop/resize и threshold по точной inference-схеме. Один prototype не обязан совпадать с одним человеком: комбинации могут выделять разные regions.

Пусть G — производная loss по assembled logits формы N×(H_mW_m). Тогда

$$
\nabla_C\mathcal L=G\widetilde P^\top,\qquad
\nabla_{\widetilde P}\mathcal L=C^\top G.
$$

Все объекты возвращают вклад в общие maps, а coefficients получают собственный signal. Поэтому prototypes обучаются совместно для изображения, а не отдельно по одному шаблону на class. Если box crop отрезал нужный край, pixel вне crop может не иметь прежнего пути gradient (градиент); детали training crop и normalization важны.

При R=32,H_m=W_m=160 shared maps содержат 819200 values. Сто объектов добавляют всего 3200 coefficients, но собранные masks уже содержат 2560000 values. При float32 это 3.2768 MB для prototypes, 0.0128 MB для coefficients и 10.24 MB для одного полного assembled tensor. Выгода в shared computation не означает отсутствие стоимости окончательных масок, их resize и копирования.

В YOLACT есть detection outputs, поэтому исходная формулировка «one-stage без явной detection, например YOLACT» неверна. One-stage означает отсутствие отдельной классической второй RoI-стадии, а не запрет boxes. Fast NMS YOLACT и Matrix NMS SOLOv2 также нельзя считать одним методом только из-за матричной параллельной реализации подавления.

### SOLO, bottom-up и query-based подходы из разбираемого примера

SOLO назначает predictions по пространственным locations и масштабам объектов. Category branch выбирает активные locations, mask branch задаёт соответствующие masks; центры дают способ организации множества outputs без обязательного box predictor. У SOLOv2 mask kernel branch предсказывает динамические коэффициенты kernels, применяемые к shared mask features. Matrix NMS — часть этого направления. Формулировка «у SOLO вообще нет NMS» слишком сильна: mask suppression остаётся важной частью опубликованных designs. [SOLO](https://arxiv.org/abs/1912.04488), [SOLOv2](https://arxiv.org/abs/2003.10152).

Bottom-up embedding (векторное представление) подход сначала назначает pixel features/центры/связи, затем группирует их в экземпляры. Он может столкнуться с ошибками grouping даже при хороших semantic classes. DeepMask из разбираемого примера — mask proposal по patches, а не канонический пример группировки per-pixel embeddings (векторные представления): исторические методы не стоит помещать в неверную категорию ради короткой схемы.

PointRend уточняет mask в выбранных points, что позволяет тратить вычисления на неопределённые boundaries вместо одинакового повышения resolution везде. Mask2Former использует masks и queries с masked attention (механизм внимания) для разных segmentation tasks; SAM отвечает на prompt (входной запрос или инструкция) и не присваивает автоматически фиксированные semantic labels (метки) каждому найденному instance. Эти альтернативы меняют механизм выбора region, refinement или conditioning, а не только backbone brand.

## Соприкасающиеся объекты, boundaries и panoptic

Для тесно стоящих объектов detector assignment, crop и resolution взаимосвязаны. Пропущенный box может означать отсутствие соответствующей mask. Низкое prototype resolution теряет тонкие детали; ошибочный crop отрежет нужную часть даже хороших logits. Ошибки нужно локализовать по этапам, иногда с oracle boxes.

При segmentation вещей и сплошных regions различают **things** (люди, машины с instance identities) и **stuff** (небо, дорога). Panoptic segmentation объединяет семантику и экземпляры в согласованную разметку. Для matched segments panoptic quality:

$$
\begin{aligned}\mathrm{PQ}&=\frac{\sum_{(p,g)\in\mathrm{TP}}\operatorname{IoU}(p,g)}{|\mathrm{TP}|+\tfrac12|\mathrm{FP}|+\tfrac12|\mathrm{FN}|}=\mathrm{SQ}\cdot\mathrm{RQ},\\\mathrm{SQ}&=\frac{\sum_{(p,g)\in\mathrm{TP}}\operatorname{IoU}(p,g)}{|\mathrm{TP}|},\qquad\mathrm{RQ}=\frac{|\mathrm{TP}|}{|\mathrm{TP}|+\tfrac12|\mathrm{FP}|+\tfrac12|\mathrm{FN}|}.\end{aligned}
$$

Exact matching/void rules берутся из benchmark. При двух TP с IoU0.8 и0.6, одном FP и одном FN: SQ0.7, RQ2/3, PQ≈0.4667. Это дополнительное углубление плана, а не автоматическая замена mask AP.

### Crop, upsampling и threshold не переставляются произвольно

Внутри bbox может находиться фрагмент другого объекта. Crop уменьшает область кандидата, но не заменяет разделение foreground внутри неё. Если predicted box уже слишком узок, hard crop необратимо удалит часть хорошей mask. При низком resolution тонкая структура может занимать долю pixel; threshold до upsampling и threshold после interpolation дают разные outlines. В отчёте нужно указать, какая continuous map и в каких coordinates преобразуется в binary mask.

При оценке overlap часть ошибок связана с detection matching, часть — с boundary, часть — с merging/splitting identities. Эти виды полезно показывать отдельно. Две правильные semantic regions, объединённые одним instance ID, всё равно дают один matched объект вместо двух. Два практически одинаковых masks для одного объекта не превращаются в два TP.

Loss по boundary gradients из рассматриваемого примера, например $\sum_i|\nabla M_i-\nabla G_i|$, — только один возможный суррогат с выбранным дискретным operator. Он может дополнить BCE/Dice, но не является универсальным стандартом под названием Boundary Loss. Нужно определить нормировку, smoothing, resolution и обработку ignored pixels. Усиление высокочастотных границ без контроля способно усилить шум разметки.

Для panoptic output каждому pixel обычно назначают согласованную class/segment identity по protocol, поэтому конфликтующие instance masks нужно разрешать. Things получают отдельные identities, stuff объединяется по правилам dataset. PQ разделяет matching и overlap: при SQ=0.7 и RQ=2/3 итог 0.466667. Улучшить только форму matched masks при прежних пропусках — не то же, что улучшить число найденных объектов; разложение помогает выбрать следующий эксперимент.

### Promptable concept segmentation как соседняя постановка

Closed-set instance segmentation получает фиксированный список классов и выдаёт instance masks по этому ontology. SAM 3 расширяет интерфейс: короткая noun-phrase или visual exemplar может задавать концепт, а модель возвращает masks и instance identities для всех соответствующих объектов. Meta оценивает эту постановку на SA-Co / Promptable Concept Segmentation; её score нельзя напрямую подменять COCO mask AP, потому что prompts, vocabulary и единица оценки различаются.

Это пример того, почему тип подсказки входит в контракт задачи. Для phrase «красный зонт» нужны критерии совпадения concept и все instances; для points/boxes пользователь задаёт конкретный region; для фиксированных labels проверяется closed-set class list. Отдельно учитывайте omitted instances, false matches и качество prompt.

**Источник:** [SAM 3 research release](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/). Авторы сообщают benchmark gains на собственном SA-Co и интерактивных segmentation tasks; перенос на другую разметку требует отдельного протокола.

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

DeepMask формирует mask proposal для patch (фрагмент изображения) и score наличия объекта. Это proposal design; его нельзя описывать как каноническое grouping всех per-pixel embeddings.

### PointRend

![PointRend](<../оформление/course/arch-pointrend.svg>)

PointRend тратит вычисления на выбранные точки высокой неопределённости, используя интерполяцию features. Он уточняет маску и требует исходного coarse prediction; сам по себе не заменяет всю систему локализации.

### SAM: promptable masks

![SAM: promptable masks](<../оформление/course/arch-sam.svg>)

Image и prompt получают отдельные representations, затем mask decoder (декодер) связывает их. Promptable mask не является автоматически полной semantic map пользовательского набора классов.

### SAM 3: concept segmentation

![SAM 3: concept segmentation](<../оформление/course/arch-sam3.svg>)

Это функциональный интерфейс Promptable Concept Segmentation: концепт задаётся phrase либо exemplar, выход содержит matching instances и masks. Схема не приписывает checkpoint (сохранённое состояние модели) неуказанное число слоёв или универсальные COCO gains.

### Mask2Former

![Mask2Former](<../оформление/course/arch-mask2former.svg>)

Queries предсказывают mask и class; masked attention ограничивает доступ к spatial features по промежуточным masks. Финальная сборка выхода зависит от выбранной segmentation formulation.

## Самопроверка и практика

1. Почему высокий box AP не доказывает mask AP? **Ответ:** локализация прямоугольника не проверяет точную shape.
2. Что исправляет RoIAlign? **Ответ:** часть coordinate quantization при извлечении region features; missing information backbone он не восстанавливает.
3. Должен ли prototype соответствовать одному объекту? **Ответ:** нет, это shared component linear combination.
4. Зачем oracle boxes? **Ответ:** помогают отделить ошибки detector от conditional mask prediction.

**Практика:** пересчитать bilinear example и mask coefficients. Нарисовать два соприкасающихся objects и сравнить outputs merged-instance, duplicate-instance и correct-instance. В эксперименте сохранить один split (разбиение данных) и менять только resolution/один mask-head factor, измеряя box AP, mask AP, boundary cases и полную latency (задержка отдельного запроса).

**Схема курса:** image → proposals/queries → per-instance class and box → RoI or prototype mask head → per-object masks/IDs → matching and mask AP





## Результат главы и переход дальше

[Маршрут модуля](<README.md>) · [Происхождение и точные места переноса](<../editorial/coverage.md>)

## Иллюстрации из первоисточников

![mask-rcnn: Figure 4 из авторской статьи](<../оформление/course/papers/mask-rcnn.png>)

**Источник:** [mask-rcnn, Figure 4](https://arxiv.org/abs/1703.06870), PDF, с. 4. Проследите вход, преобразования и выход по стрелкам. Сопоставьте обозначения статьи с формами и targets, определёнными в тексте главы.

## Как читать углубления этой ветки


Mask R-CNN связывает spatial mask с positive RoI; [YOLACT](<cv-prototypes.md>) собирает masks из общих prototypes и coefficients; [Mask2Former](<cv-mask2former.md>) использует learned queries. Общая target semantics — отдельный instance, но способы связывания различны. Для Mask R-CNN mask head обучается по соответствующему классу positive RoI, а не на всех background crops.

В классическом варианте mask logits имеют форму $[R,K,m,m]$. Binary mask target строится в координатах RoI; resize и paste обратно в image требуют согласованной геометрии. Class head и mask head обучают разные свойства: категория и spatial support.

## Разобранный пример для проверки контракта


При 100 RoIs, 80 classes и mask size $28^2$ полный mask output содержит $100\cdot80\cdot784=6272000$ scalars. После выбора одного class channel на instance остаётся 78400 mask values до paste. Ground truth instance masks могут перекрываться на уровне annotations; semantic projection теряет различение IDs.

## Ограничения переноса

Рисунки исходного Mask R-CNN показывают конкретные heads и масштабы; class-agnostic masks и поздние refinements являются отдельными вариантами.

## Самопроверка с ответами

**Практика:** объедините две masks одного класса в одну. Semantic foreground union может сохраниться, но instance matching сочтёт структуру объектов неверной.

**Почему box AP не равен mask AP?** Рамка может перекрывать GT, а predicted foreground внутри неё быть неточным.

**Нужно ли применять softmax по K mask channels в классическом Mask R-CNN?** Нет, class-specific binary masks обучаются независимым sigmoid loss выбранного класса.

## Схема вычислений

![Instance segmentation: Mask R-CNN и prototype masks: вычислительные этапы](<../оформление/course/expanded-cv-instance.svg>)

Mask R-CNN head: Positive RoI + shared features → RoIAlign → spatial tensor → Class / box / mask branches → Chosen class mask → image paste.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Источники и продолжение

- [Mask R-CNN — He et al., ICCV 2017](https://arxiv.org/abs/1703.06870)

[Следующие углубления: cv-prototypes, cv-mask2former](<README.md>) · [Сценарии подраздела](<04-segmentation/План.md>)
