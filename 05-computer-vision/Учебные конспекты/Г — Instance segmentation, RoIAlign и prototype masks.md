# Г. Instance segmentation: отдельная маска для каждого объекта

[Оглавление](README.md). Пункт 4 плана.

**Основа:** S7, строки 3081–3712. Сохранены Mask R-CNN, RoIAlign, YOLACT и YOLO-seg. **Исправления:** one-stage mask prediction может включать boxes; RoIAlign не выдаёт автоматически один vector; совместное обучение не гарантирует улучшение каждой ветви. **Углубление:** числовая bilinear interpolation, матричные формы prototypes, box/mask AP и panoptic quality.

## 1. Отличие от semantic segmentation

Output — набор `(class,score,mask)` по экземплярам, часто ещё boxes. Для двух людей нужны две отдельные masks, даже если они соприкасаются. Semantic class map объединила бы их в один class region. Перекрывающиеся instance masks допустимы как representation; правила visible/amodal annotation определяются dataset.

GT обычно хранит polygon или raster/RLE mask на instance. Box может быть получен из mask, но box annotation не определяет точную маску: внутри прямоугольника часто много background. Для amodal task требуется отдельно обозначить невидимую часть; ordinary visible segmentation нельзя оценивать по другому протоколу.

Mask IoU — отношение общего количества foreground pixels к union. Mask AP использует ranked instances и matching по **mask IoU** при заданных thresholds. Box AP использует rectangles. Хороший box может сочетаться с плохой mask, особенно на тонких объектах, отверстиях и соприкасающихся instances. Поэтому отчитываться отдельно.

Пример: два объекта находятся в одном большом predicted mask. Semantic foreground overlap может быть высоким, но instance evaluation получит максимум одно matched GT и пропуск второго. Другая ошибка — разрезать один объект на несколько masks: duplicates не становятся несколькими TP.

### Instance ID не class ID и не порядковый номер score

Три кошки имеют один semantic class, но три identities внутри кадра. Можно хранить три binary masks или одну карту instance IDs с отдельным соответствием «ID→class». Номер instance условен: перестановка 1 и 2 при согласованных масках не меняет физический ответ. Для tracking постоянство ID во времени — уже дополнительное требование, которого обычная одиночная instance segmentation не обещает.

Если две видимые части одной кошки разделены предметом, instance annotation может относить оба островка к одной mask. Обычное connected-components разбиение ошибочно сделало бы две кошки. Если невидимая середина должна дорисовываться, это amodal supervision: нельзя назвать ошибкой visible mask за то, что она не предсказала невидимую область.

### Polygon и RLE: сохраняем примеры исходника с поправкой

Polygon задаёт координаты контура, которые затем растеризуются. Число точек меняется, нормированные x/y переводятся через ширину/высоту. Несколько polygons могут описывать несколько частей объекта в поддерживаемом формате. RLE непосредственно кодирует raster, поэтому удобен для сложных областей; выбор между ними не сводится к формуле «polygon только без перекрытий, RLE только при overlap». Разные instance masks могут перекрываться независимо от способа хранения.

В S7 дан одномерный пример mask=[0,0,1,1,1,0,0,1,1]. В формате alternating run counts, начинающемся с нулей, правильная последовательность — [2,3,2,2]. Она декодируется как два нуля, три единицы, два нуля, две единицы. Вставленный в чате JSON counts=[0,2,3,2,2] кодирует иную последовательность и не является эквивалентной записью. Начальный 0 нужен, если первый элемент mask — единица.

Для двумерного COCO RLE нужно дополнительно знать size и column-major порядок обхода; compressed counts имеют своё кодирование. Универсальные пары (start,length) из учебного примера — другой вариант RLE, их нельзя без конвертации передать как COCO counts. Сумма uncompressed runs должна равняться H·W. Это позволяет обнаружить разрушенную маску ещё до обучения. [Официальный COCO mask API](https://github.com/cocodataset/cocoapi/blob/master/common/maskApi.c).

### Почему высокий box AP и mask AP могут расходиться

Два box совпадают идеально, но один объект имеет форму кольца. Если предсказать весь прямоугольник foreground, hole заполнится и mask overlap ухудшится. Для тонкого инструмента большая часть box вообще background. И наоборот, небольшое несовпадение прямоугольников не задаёт однозначного изменения mask IoU: распределение foreground внутри них важно. Поэтому утверждение S7 «mask metric всегда строже» не является универсальным порядком чисел; это разные области сравнения.

Score ранжирует instances при AP. Если class confidence высокий, а mask плохая, такой candidate может рано принести FP на строгом threshold. Отдельная оценка mask quality, как в Mask Scoring R-CNN, пытается согласовать ranking с качеством mask. Это изменение confidence modelling, а не автоматическое улучшение её пиксельной формы.

## 2. Mask R-CNN и RoIAlign

Mask R-CNN расширяет detector branch предсказанием маски для region. Shared backbone/FPN → proposals → region features → classification/box/mask branches. В классическом варианте mask branch даёт K class-specific binary masks на RoI, а supervision считается для channel true class. Sigmoid по каждому pixel не является softmax-конкуренцией разных классов в каждой mask. Есть class-agnostic variants, их нельзя автоматически описывать исходной K-channel схемой. [Mask R-CNN](https://arxiv.org/abs/1703.06870).

RoIPool округляет coordinates/границы bins, что создаёт spatial quantization error. RoIAlign выбирает fractional positions и использует bilinear interpolation; затем агрегирует samples по определённой sampling policy. Выход — fixed spatial tensor, например $C\times 14\times 14$, из которого mask branch может получить higher-resolution output. Это не обязательно flatten vector.

Для четырёх значений $f_{00}=1,\qquad f_{10}=3,\qquad f_{01}=5,\qquad f_{11}=7$ в точке $x=0.25$,$y=0.5$:

$$
\begin{aligned}f(x,y)&=(1-x)(1-y)f_{00}+x(1-y)f_{10}+(1-x)yf_{01}+xyf_{11}\\&=0.375\cdot1+0.125\cdot3+0.375\cdot5+0.125\cdot7=3.5.\end{aligned}
$$

Это простой пример interpolation внутри одной клетки feature map. В реальном RoIAlign важны image→feature stride, coordinate alignment offset, sampling ratio и boundary handling. Различия half-pixel conventions способны сдвинуть mask без очевидной shape ошибки.

Общий training objective можно записать $\mathcal{L}=\lambda_{\mathrm{cls}}\mathcal{L}_{\mathrm{cls}}+\lambda_{\mathrm{box}}\mathcal{L}_{\mathrm{box}}+\lambda_{\mathrm{mask}}\mathcal{L}_{\mathrm{mask}}$. Mask BCE на foreground RoI учит форму, classification определяет category, box loss — localization. Loss weights/normalization влияют на баланс; нельзя утверждать, что ветви «не мешают друг другу» только из-за разных heads: shared features получают сумму gradients.

### Почему RoIAlign нужен именно для границ

Пусть region начинается в координате 10.4 input image, а feature map имеет stride4. Его начало в feature coordinates — 2.6 до возможных alignment offsets. Округление до 2 или 3 меняет, какие признаки извлекаются; после обратного увеличения ошибка становится заметной на input mask. RoIAlign сохраняет fractional sampling и интерполирует соседние values, а не требует считать исходную картинку заново для каждого crop.

В одном bin можно брать несколько sample points и усреднять или агрегировать их по выбранной policy. Поэтому bilinear interpolation четырёх соседей — один этап, не всё определение RoIAlign. За ним остаются границы region, bins, sampling ratio и отображение координат. Два kernels с разными half-pixel conventions могут выдавать одинаковые shapes и сдвинутые детали.

В исходном Mask R-CNN mask head получала пространственные признаки около 14×14, применяла несколько convolutions и transposed conv для 28×28 output. Свёртки внутри head используют layout, поэтому заменить вход одним pooled вектором без другой архитектуры нельзя. Class branch решает, какой channel mask использовать; mask branch учит binary foreground внутри этого channel для positive RoI.

### От binary loss одной RoI к общему обучению

Пусть target class региона — «человек», а head выдаёт K каналов. В исходной class-specific схеме mask BCE считается для канала человека. Если вместо этого применять softmax по K classes на каждом mask pixel, возникнет другая задача: конкуренция категорий внутри региона. Binary mask может отмечать background внутри box; category supervision уже существует в class branch.

Для одного pixel target=1 и p=0.8 BCE≈0.223144; для target=0 и той же p ошибка≈1.609438. Loss усредняется по предусмотренным mask pixels и positive RoIs. Если число positives в batch меняется, normalization влияет на относительный масштаб mask contribution. Shared backbone получает сумму производных нескольких ветвей: хорошо разделённые heads не делают gradients автоматически согласованными.

При training region может сопоставляться с GT и получать target crop. На inference box уже предсказан и может быть смещён. Этот сдвиг меняет вход mask head и её обратное размещение в картинке. Oracle GT boxes помогают понять, ограничивает ли качество localization или сама условная сегментация; такой oracle score не является обычным inference score.

## 3. Prototype masks: YOLACT и родственная идея

В YOLACT одна ветвь генерирует shared prototype maps, другая — coefficients для каждой detection. Пусть R — число prototypes, N — число candidates, H_m/W_m — mask resolution:

$$
\begin{gathered}P\in\mathbb{R}^{R\times H_m\times W_m},\quad C\in\mathbb{R}^{N\times R},\quad\widetilde P\in\mathbb{R}^{R\times(H_mW_m)},\\Z_{\mathrm{mask}}=C\widetilde P\in\mathbb{R}^{N\times(H_mW_m)},\qquad M=\sigma(Z_{\mathrm{mask}}).\end{gathered}
$$

Prototype — basis map, а не обязательная готовая mask отдельного объекта. Coefficients могут комбинировать несколько maps; отрицательные coefficients тоже могут быть осмысленны до sigmoid. [YOLACT](https://arxiv.org/abs/1904.02689).

Маленький пример с R=2 и двумя pixels: $P_{1}=\left(1,\,0\right)$, $P_{2}=\left(0,\,1\right)$. Coefficients объекта A $\left(2,\,-2\right)$ дают logits $\left(2,\,-2\right)$, probabilities≈(0.8808,0.1192). Объект B $\left(-2,\,2\right)$ — противоположную mask. Общие prototypes не мешают разным masks благодаря coefficients.

Память scales как $R H_mW_m+NR$ до assembly вместо отдельной тяжёлой spatial head на каждую RoI. Но после assembly всё равно может потребоваться $N H_mW_m$ output/intermediate memory. При большом N mask postprocessing может стать bottleneck; лозунг «R coefficients, значит masks почти ничего не стоят» неполон.

YOLO-seg использует task-specific structures, в которых может быть родственная prototype+coefficient механика. Надо проверять конкретную head, resolution, crop, upsampling и порядок NMS/mask construction. Нельзя считать каждую YOLO-seg точной копией YOLACT или путать её output с semantic `*-sem` model.

### Полный путь prototype assembly и его производные

Сначала protonet выдаёт R shared maps. Другой head для candidate i вычисляет coefficients c_i, class и часто box. После отбора остаются N объектов, их coefficients образуют матрицу C. Умножение C на flattened prototypes сразу строит N mask-logit maps; затем sigmoid, crop/resize и threshold по точной inference-схеме. Один prototype не обязан совпадать с одним человеком: комбинации могут выделять разные regions.

Пусть G — производная loss по assembled logits формы N×(H_mW_m). Тогда

$$
\nabla_C\mathcal L=G\widetilde P^\top,\qquad
\nabla_{\widetilde P}\mathcal L=C^\top G.
$$

Все объекты возвращают вклад в общие maps, а coefficients получают собственный signal. Поэтому prototypes обучаются совместно для изображения, а не отдельно по одному шаблону на class. Если box crop отрезал нужный край, pixel вне crop может не иметь прежнего пути gradient; детали training crop и normalization важны.

При R=32,H_m=W_m=160 shared maps содержат 819200 values. Сто объектов добавляют всего 3200 coefficients, но собранные masks уже содержат 2560000 values. При float32 это 3.2768 MB для prototypes, 0.0128 MB для coefficients и 10.24 MB для одного полного assembled tensor. Выгода в shared computation не означает отсутствие стоимости окончательных масок, их resize и копирования.

В YOLACT есть detection outputs, поэтому исходная формулировка «one-stage без явной detection, например YOLACT» неверна. One-stage означает отсутствие отдельной классической второй RoI-стадии, а не запрет boxes. Fast NMS YOLACT и Matrix NMS SOLOv2 также нельзя считать одним методом только из-за матричной параллельной реализации подавления.

### SOLO, bottom-up и query-based подходы из чата

SOLO назначает predictions по пространственным locations и масштабам объектов. Category branch выбирает активные locations, mask branch задаёт соответствующие masks; центры дают способ организации множества outputs без обязательного box predictor. У SOLOv2 mask kernel branch предсказывает динамические коэффициенты kernels, применяемые к shared mask features. Matrix NMS — часть этого направления. Формулировка S7 «у SOLO вообще нет NMS» слишком сильна: mask suppression остаётся важной частью опубликованных designs. [SOLO](https://arxiv.org/abs/1912.04488), [SOLOv2](https://arxiv.org/abs/2003.10152).

Bottom-up embedding подход сначала назначает pixel features/центры/связи, затем группирует их в экземпляры. Он может столкнуться с ошибками grouping даже при хороших semantic classes. DeepMask из чата — mask proposal по patches, а не канонический пример группировки per-pixel embeddings: исторические методы не стоит помещать в неверную категорию ради короткой схемы.

PointRend уточняет mask в выбранных points, что позволяет тратить вычисления на неопределённые boundaries вместо одинакового повышения resolution везде. Mask2Former использует masks и queries с masked attention для разных segmentation tasks; SAM отвечает на prompt и не присваивает автоматически фиксированные semantic labels каждому найденному instance. Эти альтернативы меняют механизм выбора region, refinement или conditioning, а не только backbone brand.

## 4. Соприкасающиеся объекты, boundaries и panoptic

Для тесно стоящих объектов detector assignment, crop и resolution взаимосвязаны. Пропущенный box может означать отсутствие соответствующей mask. Низкое prototype resolution теряет тонкие детали; ошибочный crop отрежет нужную часть даже хороших logits. Ошибки нужно локализовать по этапам, иногда с oracle boxes.

При segmentation вещей и сплошных regions различают **things** (люди, машины с instance identities) и **stuff** (небо, дорога). Panoptic segmentation объединяет семантику и экземпляры в согласованную разметку. Для matched segments panoptic quality:

$$
\begin{aligned}\mathrm{PQ}&=\frac{\sum_{(p,g)\in\mathrm{TP}}\operatorname{IoU}(p,g)}{|\mathrm{TP}|+\tfrac12|\mathrm{FP}|+\tfrac12|\mathrm{FN}|}=\mathrm{SQ}\cdot\mathrm{RQ},\\\mathrm{SQ}&=\frac{\sum_{(p,g)\in\mathrm{TP}}\operatorname{IoU}(p,g)}{|\mathrm{TP}|},\qquad\mathrm{RQ}=\frac{|\mathrm{TP}|}{|\mathrm{TP}|+\tfrac12|\mathrm{FP}|+\tfrac12|\mathrm{FN}|}.\end{aligned}
$$

Exact matching/void rules берутся из benchmark. При двух TP с IoU0.8 и0.6, одном FP и одном FN: SQ0.7, RQ2/3, PQ≈0.4667. Это дополнительное углубление плана, а не автоматическая замена mask AP.

### Crop, upsampling и threshold не переставляются произвольно

Внутри bbox может находиться фрагмент другого объекта. Crop уменьшает область кандидата, но не заменяет разделение foreground внутри неё. Если predicted box уже слишком узок, hard crop необратимо удалит часть хорошей mask. При низком resolution тонкая структура может занимать долю pixel; threshold до upsampling и threshold после interpolation дают разные outlines. В отчёте нужно указать, какая continuous map и в каких coordinates преобразуется в binary mask.

При оценке overlap часть ошибок связана с detection matching, часть — с boundary, часть — с merging/splitting identities. Эти виды полезно показывать отдельно. Две правильные semantic regions, объединённые одним instance ID, всё равно дают один matched объект вместо двух. Два практически одинаковых masks для одного объекта не превращаются в два TP.

Loss по boundary gradients из S7, например $\sum_i|\nabla M_i-\nabla G_i|$, — только один возможный суррогат с выбранным дискретным operator. Он может дополнить BCE/Dice, но не является универсальным стандартом под названием Boundary Loss. Нужно определить нормировку, smoothing, resolution и обработку ignored pixels. Усиление высокочастотных границ без контроля способно усилить шум разметки.

Для panoptic output каждому pixel обычно назначают согласованную class/segment identity по protocol, поэтому конфликтующие instance masks нужно разрешать. Things получают отдельные identities, stuff объединяется по правилам dataset. PQ разделяет matching и overlap: при SQ=0.7 и RQ=2/3 итог 0.466667. Улучшить только форму matched masks при прежних пропусках — не то же, что улучшить число найденных объектов; разложение помогает выбрать следующий эксперимент.

## 5. Самопроверка и практика

1. Почему высокий box AP не доказывает mask AP? **Ответ:** локализация прямоугольника не проверяет точную shape.
2. Что исправляет RoIAlign? **Ответ:** часть coordinate quantization при извлечении region features; missing information backbone он не восстанавливает.
3. Должен ли prototype соответствовать одному объекту? **Ответ:** нет, это shared component linear combination.
4. Зачем oracle boxes? **Ответ:** помогают отделить ошибки detector от conditional mask prediction.

**Практика:** пересчитать bilinear example и mask coefficients. Нарисовать два соприкасающихся objects и сравнить outputs merged-instance, duplicate-instance и correct-instance. В эксперименте сохранить один split и менять только resolution/один mask-head factor, измеряя box AP, mask AP, boundary cases и полную latency.

<!-- latex-display-v1 -->

<!-- source-depth-S7-Г -->
