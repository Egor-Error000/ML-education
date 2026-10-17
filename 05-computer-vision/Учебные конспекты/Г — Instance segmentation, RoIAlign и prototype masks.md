# Г. Instance segmentation: отдельная маска для каждого объекта

[Оглавление](README.md). Пункт 4 плана.

**Основа:** S7, строки 3081–3712. Сохранены Mask R-CNN, RoIAlign, YOLACT и YOLO-seg. **Исправления:** one-stage mask prediction может включать boxes; RoIAlign не выдаёт автоматически один vector; совместное обучение не гарантирует улучшение каждой ветви. **Углубление:** числовая bilinear interpolation, матричные формы prototypes, box/mask AP и panoptic quality.

## 1. Отличие от semantic segmentation

Output — набор `(class,score,mask)` по экземплярам, часто ещё boxes. Для двух людей нужны две отдельные masks, даже если они соприкасаются. Semantic class map объединила бы их в один class region. Перекрывающиеся instance masks допустимы как representation; правила visible/amodal annotation определяются dataset.

GT обычно хранит polygon или raster/RLE mask на instance. Box может быть получен из mask, но box annotation не определяет точную маску: внутри прямоугольника часто много background. Для amodal task требуется отдельно обозначить невидимую часть; ordinary visible segmentation нельзя оценивать по другому протоколу.

Mask IoU — отношение общего количества foreground pixels к union. Mask AP использует ranked instances и matching по **mask IoU** при заданных thresholds. Box AP использует rectangles. Хороший box может сочетаться с плохой mask, особенно на тонких объектах, отверстиях и соприкасающихся instances. Поэтому отчитываться отдельно.

Пример: два объекта находятся в одном большом predicted mask. Semantic foreground overlap может быть высоким, но instance evaluation получит максимум одно matched GT и пропуск второго. Другая ошибка — разрезать один объект на несколько masks: duplicates не становятся несколькими TP.

## 2. Mask R-CNN и RoIAlign

Mask R-CNN расширяет detector branch предсказанием маски для region. Shared backbone/FPN → proposals → region features → classification/box/mask branches. В классическом варианте mask branch даёт K class-specific binary masks на RoI, а supervision считается для channel true class. Sigmoid по каждому pixel не является softmax-конкуренцией разных классов в каждой mask. Есть class-agnostic variants, их нельзя автоматически описывать исходной K-channel схемой. [Mask R-CNN](https://arxiv.org/abs/1703.06870).

RoIPool округляет coordinates/границы bins, что создаёт spatial quantization error. RoIAlign выбирает fractional positions и использует bilinear interpolation; затем агрегирует samples по определённой sampling policy. Выход — fixed spatial tensor, например $C\times 14\times 14$, из которого mask branch может получить higher-resolution output. Это не обязательно flatten vector.

Для четырёх значений $f_{00}=1,\qquad f_{10}=3,\qquad f_{01}=5,\qquad f_{11}=7$ в точке $x=0.25$,$y=0.5$:

$$
\begin{aligned}f(x,y)&=(1-x)(1-y)f_{00}+x(1-y)f_{10}+(1-x)yf_{01}+xyf_{11}\\&=0.375\cdot1+0.125\cdot3+0.375\cdot5+0.125\cdot7=3.5.\end{aligned}
$$

Это простой пример interpolation внутри одной клетки feature map. В реальном RoIAlign важны image→feature stride, coordinate alignment offset, sampling ratio и boundary handling. Различия half-pixel conventions способны сдвинуть mask без очевидной shape ошибки.

Общий training objective можно записать $\mathcal{L}=\lambda_{\mathrm{cls}}\mathcal{L}_{\mathrm{cls}}+\lambda_{\mathrm{box}}\mathcal{L}_{\mathrm{box}}+\lambda_{\mathrm{mask}}\mathcal{L}_{\mathrm{mask}}$. Mask BCE на foreground RoI учит форму, classification определяет category, box loss — localization. Loss weights/normalization влияют на баланс; нельзя утверждать, что ветви «не мешают друг другу» только из-за разных heads: shared features получают сумму gradients.

## 3. Prototype masks: YOLACT и родственная идея

В YOLACT одна ветвь генерирует shared prototype maps, другая — coefficients для каждой detection. Пусть R — число prototypes, N — число candidates, H_m/W_m — mask resolution:

$$
\begin{gathered}P\in\mathbb{R}^{R\times H_m\times W_m},\quad C\in\mathbb{R}^{N\times R},\quad\widetilde P\in\mathbb{R}^{R\times(H_mW_m)},\\Z_{\mathrm{mask}}=C\widetilde P\in\mathbb{R}^{N\times(H_mW_m)},\qquad M=\sigma(Z_{\mathrm{mask}}).\end{gathered}
$$

Prototype — basis map, а не обязательная готовая mask отдельного объекта. Coefficients могут комбинировать несколько maps; отрицательные coefficients тоже могут быть осмысленны до sigmoid. [YOLACT](https://arxiv.org/abs/1904.02689).

Маленький пример с R=2 и двумя pixels: $P_{1}=\left(1,\,0\right)$, $P_{2}=\left(0,\,1\right)$. Coefficients объекта A $\left(2,\,-2\right)$ дают logits $\left(2,\,-2\right)$, probabilities≈(0.8808,0.1192). Объект B $\left(-2,\,2\right)$ — противоположную mask. Общие prototypes не мешают разным masks благодаря coefficients.

Память scales как $R H_mW_m+NR$ до assembly вместо отдельной тяжёлой spatial head на каждую RoI. Но после assembly всё равно может потребоваться $N H_mW_m$ output/intermediate memory. При большом N mask postprocessing может стать bottleneck; лозунг «R coefficients, значит masks почти ничего не стоят» неполон.

YOLO-seg использует task-specific structures, в которых может быть родственная prototype+coefficient механика. Надо проверять конкретную head, resolution, crop, upsampling и порядок NMS/mask construction. Нельзя считать каждую YOLO-seg точной копией YOLACT или путать её output с semantic `*-sem` model.

## 4. Соприкасающиеся объекты, boundaries и panoptic

Для тесно стоящих объектов detector assignment, crop и resolution взаимосвязаны. Пропущенный box может означать отсутствие соответствующей mask. Низкое prototype resolution теряет тонкие детали; ошибочный crop отрежет нужную часть даже хороших logits. Ошибки нужно локализовать по этапам, иногда с oracle boxes.

При segmentation вещей и сплошных regions различают **things** (люди, машины с instance identities) и **stuff** (небо, дорога). Panoptic segmentation объединяет семантику и экземпляры в согласованную разметку. Для matched segments panoptic quality:

$$
\begin{aligned}\mathrm{PQ}&=\frac{\sum_{(p,g)\in\mathrm{TP}}\operatorname{IoU}(p,g)}{|\mathrm{TP}|+\tfrac12|\mathrm{FP}|+\tfrac12|\mathrm{FN}|}=\mathrm{SQ}\cdot\mathrm{RQ},\\\mathrm{SQ}&=\frac{\sum_{(p,g)\in\mathrm{TP}}\operatorname{IoU}(p,g)}{|\mathrm{TP}|},\qquad\mathrm{RQ}=\frac{|\mathrm{TP}|}{|\mathrm{TP}|+\tfrac12|\mathrm{FP}|+\tfrac12|\mathrm{FN}|}.\end{aligned}
$$

Exact matching/void rules берутся из benchmark. При двух TP с IoU0.8 и0.6, одном FP и одном FN: SQ0.7, RQ2/3, PQ≈0.4667. Это дополнительное углубление плана, а не автоматическая замена mask AP.

## 5. Самопроверка и практика

1. Почему высокий box AP не доказывает mask AP? **Ответ:** локализация прямоугольника не проверяет точную shape.
2. Что исправляет RoIAlign? **Ответ:** часть coordinate quantization при извлечении region features; missing information backbone он не восстанавливает.
3. Должен ли prototype соответствовать одному объекту? **Ответ:** нет, это shared component linear combination.
4. Зачем oracle boxes? **Ответ:** помогают отделить ошибки detector от conditional mask prediction.

**Практика:** пересчитать bilinear example и mask coefficients. Нарисовать два соприкасающихся objects и сравнить outputs merged-instance, duplicate-instance и correct-instance. В эксперименте сохранить один split и менять только resolution/один mask-head factor, измеряя box AP, mask AP, boundary cases и полную latency.

<!-- latex-display-v1 -->
