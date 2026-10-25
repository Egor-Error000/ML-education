# Detection: множества объектов, IoU, AP и архитектуры

**Уровень:** общее ядро.

**Предпосылки:** [Изображение, карты признаков и блоки модели](<cv-input.md>), [Классификация, CNN и residual learning](<cv-classification.md>)

**Используют:** [Faster R-CNN: RPN, RoI и обучение двух стадий](<cv-faster-rcnn.md>), [RetinaNet и FCOS: anchors и dense prediction](<cv-dense-detectors.md>), [YOLOv3 и YOLOv8: головы, assignment и декодирование](<cv-yolo.md>), [DETR: queries, matching и set prediction](<cv-detr.md>), [Ориентированные рамки и периодичность угла](<cv-obb.md>), [Instance segmentation: Mask R-CNN и prototype masks](<cv-instance.md>), [CLIP, открытый словарь и YOLO-World](<cv-open-vocabulary.md>), [Tracking: Kalman, association и идентичность](<cv-tracking.md>), [Voxels, PointPillars и BEV: представления для 3D detection](<cv-bev.md>), [CRAFT и DBNet: поиск текста на изображении](<cv-text-detection.md>), [Диагностика CV: ошибки данных, модели и post-processing](<cv-diagnosis.md>)

**Учебный вопрос:** Как получить множество рамок и оценить его с учётом matching?

## Карта вычисления и границы этапов

![Detection: множества объектов, IoU, AP и архитектуры: входы, преобразования и выходы](<../оформление/course/cv-detection.svg>)

Детектор выдаёт классы вместе с координатами. Обучающее сопоставление и удаление повторных рамок при применении — разные операции, которые нельзя смешивать при интерпретации схемы.

## Выход — множество, а не одна метка

Для изображения output состоит из объектов `(box,class,score)`. Число объектов заранее неизвестно, порядок списка обычно не имеет семантики. Annotation задаёт box каждого экземпляра, а не один label (метка) всего изображения. Пропущенные в разметке реальные объекты могут при обучении/оценке ошибочно считаться background.

В `xyxy` box задаёт левый верхний и правый нижний углы $(x_1,y_1,x_2,y_2)$. В `xywh` надо уточнить, x/y — центр или corner. Нормированные и pixel coordinates нельзя смешивать. Для непрерывной геометрии площадь $(x_2-x_1)(y_2-y_1)$; исторические inclusive pixel conventions с $+1$ — другой protocol.

IoU=$\frac{|A\cap B|}{|A\cup B|}$. У A=(0,0,2,2), B=(1,1,3,3) площади 4 и4, пересечение1, union7, IoU=1/7≈0.142857. У непересекающихся boxes IoU0. Если boxes вырожденные, protocol должен определить обработку.

Class confidence не тождественна IoU: модель может уверенно назвать класс и плохо локализовать объект. В разных heads есть отдельный objectness или совместный score (оценка) качества/класса. Нельзя универсально обещать ровно «objectness×class probability» для каждого YOLO checkpoint (сохранённое состояние модели).

### Почему переменное число объектов не запрещает фиксированный tensor

В разбираемом примере сказано, что нельзя выдать фиксированный вектор, поскольку число объектов M неизвестно. Точнее: нельзя свести задачу к одному глобальному class vector как в классификации. Реализация вполне может выдавать фиксированное число кандидатов или queries, затем отбирать действительные объекты. Например, сто slots содержат scores (оценки) и boxes; slots с no-object или низким score не включаются в итоговое множество. Нужно различать внутреннюю форму tensor и семантическое число найденных объектов.

Список detections не имеет естественного порядка: две одинаковые по смыслу выдачи могут различаться перестановкой строк. Для training это создаёт задачу assignment — кому из кандидатов назначить какой GT. На evaluation задача похожа только внешне: там по объявленному protocol решается, что засчитать TP. Переносить training assignment непосредственно в метрику нельзя.

### Форматы VOC, COCO и YOLO на одном объекте

Сохраним пример источника: box с углами (100,200) и (300,400). В COCO-подобной записи верхний левый угол и размер будут `[100,200,200,200]`. Для изображения W=640,H=480 центр (200,300), нормированные координаты YOLO — $(200/640,300/480,200/640,200/480)=(0.3125,0.625,0.3125,0.416667)$. Строка с class ID 0 начинается с `0`, затем идут эти четыре значения.

Обратное преобразование: $x_1=(x_c-w/2)W$, $x_2=(x_c+w/2)W$, аналогично y. Проверка round trip должна восстановить углы. Делить все четыре координаты на ширину изображения ошибочно для непрямоугольного aspect ratio. Путаница «xywh с центром» и «xywh с corner» тоже даст допустимые числа неправильной геометрии.

VOC исторически использует XML, COCO — JSON с image/category IDs, распространённый YOLO detection label — строку на объект. Число классов 20 или 80 относится к известным benchmark-наборам, а не к принципиальному ограничению XML и JSON. Пользовательский датасет этих форматов может иметь другое число классов; category IDs также не обязаны совпадать с непрерывными индексами model head.

### Вычисление IoU по координатам без скрытых шагов

У двух axis-aligned boxes левый верхний угол пересечения получается через max нижних границ, правый нижний — через min верхних. Поэтому

$$
w_\cap=\max(0,\min(x_2^A,x_2^B)-\max(x_1^A,x_1^B)),
$$

$$
h_\cap=\max(0,\min(y_2^A,y_2^B)-\max(y_1^A,y_1^B)).
$$

Площадь пересечения равна произведению этих длин; union — сумма площадей минус пересечение. Max с нулём принципиален: отрицательная длина при раздельных прямоугольниках не должна давать положительную площадь после умножения двух отрицательных чисел.

При общем масштабировании всех координат в a раз площади умножаются на a², отношение сохраняется. Это и означает масштабную инвариантность IoU в геометрическом смысле. Но ошибка в два пикселя для маленького объекта и для большого — не одно общее масштабирование пары boxes: маленький box может потерять большую долю перекрытия. Поэтому высокие IoU thresholds особенно требовательны к локализации малых объектов.

## Matching: когда prediction становится TP

Для одной class и заданного IoU threshold τ predictions сортируют по score. Каждая prediction может сопоставиться с ещё не занятым GT того же class, если IoU удовлетворяет protocol. Первое подходящее совпадение TP, duplicate того же GT — FP. Непокрытые GT остаются FN. Оценка должна учитывать ignore/crowd правила выбранного benchmark, а не произвольную новую реализацию.

Пример: два GT, три predictions по убывающему score. P1 правильно покрыла GT1; P2 — duplicate GT1; P3 — GT2. Получаем TP flags $\left(1,\,0,\,1\right)$:

$$
\begin{array}{c|cc|cc}\text{после}&\mathrm{TP}&\mathrm{FP}&\mathrm{Precision}&\mathrm{Recall}\\\hline P_1&1&0&1&0.5\\P_2&1&1&0.5&0.5\\P_3&2&1&\frac23&1\end{array}
$$

AP агрегирует precision-recall curve, но **точное правило интегрирования** существенно. Для continuous interpolated envelope здесь первая половина recall имеет precision1, вторая2/3, AP=5/6≈0.8333. Это учебный вариант, не численный результат официального COCO evaluator с его сеткой recall/area/maxDets. Различные AP conventions не смешивать в отчёте.

mAP усредняет AP по выбранным классам и, если protocol требует, IoU thresholds. AP@0.5 проще, чем COCO-style AP@[0.5:0.95]: второй усредняет thresholds 0.50,0.55,…,0.95. Рост AP50 при неизменной высокой-threshold AP может означать, что объекты находятся, но box localization ещё недостаточно точна. Small/medium/large AP и per-class errors дают полезные срезы.

Порог score, выбранный для приложения, задаёт одну operating point. Для AP нужен корректный набор ranked detections по protocol; слишком раннее отбрасывание низких scores способно искусственно ограничить attainable recall.

### Расчёт AP: пять эталонных объектов и восемь предсказаний

В разбираемом примере для одного класса были пять истинных объектов и восемь detections со scores 0.95,0.90,0.85,0.80,0.70,0.60,0.50,0.40. При выбранном IoU threshold flags равны [TP,TP,FP,TP,FP,TP,FP,TP]. После каждого шага precision равна [1,1,2/3,3/4,3/5,4/6,4/7,5/8], recall — [1/5,2/5,2/5,3/5,3/5,4/5,4/5,1].

После третьего detection precision упала до 2/3, а после четвёртого выросла до 3/4. Поэтому PR-кривая не обязана монотонно падать — монотонность относится к envelope после интерполяции. Для каждого r она берёт лучший precision, доступный при recall не меньше r:

$$
p_{\rm interp}(r)=\max_{\widetilde r\ge r}p(\widetilde r).
$$

На первых двух интервалах recall ширины 1/5 envelope равна 1; на следующих — 3/4,2/3,5/8. Continuous interpolated AP в этом примере:

$$
\operatorname{AP}=\frac15\left(1+1+\frac34+\frac23+\frac58\right)=\frac{97}{120}\approx0.808333.
$$

Получается «около 0.81», но теперь известно точное правило. Равенство простому среднему здесь связано с равными приростами recall по 1/5. Для неравных интервалов нужно умножать precision на их ширины. VOC2007 с 11 recall-точками и COCO с собственной сеткой дадут другой численный protocol; один учебный integral не подменяет официальный evaluator.

Если оставить только predictions со score≥0.65, будут обработаны первые пять: TP=3, FP=2, recall=3/5. Модель сохранила более уверенные candidates, но уже не может достигнуть recall=1 на этом списке. AP характеризует ranking по набору thresholds; фиксированный operating threshold приложения выбирается отдельно. Нельзя описать улучшение AP одним повышением confidence threshold без проверки изменения самого списка и protocol.

### Как ошибки разных типов входят в counts

Правильный class при IoU ниже threshold считается FP, а неудачно покрытый GT может одновременно остаться FN. Неправильный class тоже может создать FP для одного класса и FN для другого. Duplicate — FP даже при прекрасной геометрии, если соответствующий GT уже занят. Это объясняет, почему «модель нашла объект глазами» и «метрика засчитала TP» могут расходиться.

Порядок scores влияет на precision раньше по списку. Если duplicate предсказывается выше более точного box, он может первым занять GT, а следующий стать FP. На одном IoU threshold оба могли бы подходить, на другом первый уже не пройдёт. Отсюда полезность срезов AP50, AP75 и более строгих порогов: они разделяют грубое нахождение и точную локализацию.

## Зачем NMS и где её границы

Greedy (выбор максимальной оценки на каждом шаге) NMS для каждой class: выбрать максимальный score, сохранить box, подавить оставшиеся boxes с IoU выше threshold, повторить. Она удаляет дубликаты, но может убрать разные близкие/перекрывающиеся объекты. Class-agnostic NMS объединяет classes и имеет другие trade-offs. Soft-NMS уменьшает scores вместо жёсткого удаления.

Если A=(0,0,2,2), C=(0.2,0.2,2.2,2.2), intersection3.24, union4.76, IoU≈0.68067. При NMS threshold0.5 lower-score box будет подавлена в одной class. Правильно ли это, зависит от того, duplicate это или два сильно перекрытых объекта.

Нельзя ставить одинаковый IoU threshold для training assignment, NMS и evaluation только потому, что везде написано IoU. Они управляют разными решениями. NMS-free inference (применение обученной модели) достигается специальной схемой обучения/выходов, а не просто выключением NMS у головы, обученной выдавать много duplicates.

### Пошаговая NMS и смысл её порога

Пусть три boxes одного класса: A со score=0.95, C со score=0.90 и удалённый D со score=0.80. У A/C IoU≈0.68067. С threshold=0.5 сначала сохраняется A, затем C подавляется как сильно пересекающаяся с сохранённой; D остаётся и выбирается следующим. При threshold=0.7 C не будет подавлена. В отличие от score threshold, который регулирует минимальную оценку кандидата, этот порог регулирует допустимое перекрытие уже выбранных boxes.

Если A и C — два реальных человека в толпе, suppression ошибочно уменьшит recall. Если они описывают одного человека, suppression удалит duplicate и может улучшить precision. Поэтому один и тот же геометрический критерий не знает истинного числа экземпляров. Class-aware вариант применяет его отдельно, class-agnostic может подавить и boxes разных классов. У Soft-NMS score понижается вместо немедленного удаления, например $s_j\leftarrow s_j\exp(-\operatorname{IoU}^2/\sigma)$ в Gaussian-варианте; затем ранжирование обновляется.

В разбираемом примере жёсткая NMS названа причиной того, что «нельзя обучать end-to-end». Точное ограничение: дискретные выборы обычной greedy NMS не дают удобной обычной производной по решению keep/drop. Но многие детекторы обучают differentiable raw outputs без прохождения loss (функция потерь) через inference NMS. Следовательно, наличие NMS на inference не запрещает совместное обучение backbone (основная сеть извлечения признаков)/head. NMS-free set prediction меняет задачу дубликатов, а не просто «разрешает backprop, которого раньше вообще не было».

## Two-stage, one-stage и признаки разных масштабов

Исторический sliding window запускает classifier на множестве crops, что дорого и дублирует вычисления. R-CNN использует region proposals и обработку регионов. Fast R-CNN разделяет shared feature (признак) extraction и region-wise head; Faster R-CNN добавляет обучаемую RPN. В современных описаниях RoIAlign/другие изменения надо отличать от исходного RoIPool, не приписывая их всем версиям сразу.

One-stage detector предсказывает кандидатов на feature maps без отдельной классической proposal-classification стадии. SSD/YOLO — важные примеры, но «one-stage всегда быстрее, two-stage всегда точнее» не теорема: сравниваются реальные models, resolution и hardware.

FPN объединяет семантику глубоких низкоразрешённых maps с деталями более высокого resolution. Это важно для малых объектов: после сильного downsampling (уменьшение пространственного разрешения) объект может занимать меньше одной эффективной клетки. Повышение input resolution увеличивает compute, но иногда исправляет bottleneck данных/геометрии лучше, чем увеличение model size.

YOLOv1 — grid-based прямое предсказание boxes без anchor механизма последующих v2/v3. Anchors — заранее заданные reference shapes, offsets относительно которых учатся. Anchor-free methods убирают именно эти шаблоны, но всё равно могут использовать feature-grid reference points. Decoupled head разделяет prediction branches, например class/box; это ещё одна независимая идея.

Training assignment отвечает, какие predictions получают targets (целевые значения) какого GT. One-to-many даёт несколько positives на GT и обычно требует удаления duplicates. One-to-one способствует unique output, но нуждается в подходящем loss/assignment. В YOLOv10 и последующих designs могут сочетаться training branches с разными assignment. Для YOLO26 доступен отдельный one-to-one NMS-free inference path; конкретный default/path нужно задавать явно. [Официальное описание YOLO26](https://docs.ultralytics.com/models/yolo26/).

DETR формулирует set prediction с object queries и bipartite/Hungarian matching training outputs к GT. Дополнительные predictions обучают no-object class/эквиваленту protocol. Отсутствие обычной NMS связано с set objective (оптимизируемая целевая функция), а не только со словом Transformer. RT-DETR использует связанные идеи с акцентом на speed; details выбранного variant проверяют отдельно.

### От sliding windows к shared feature map (пространственная карта признаков)

Для окна K×K со stride (шаг пространственного перемещения операции) S без padding (дополнение до общей длины) по изображению H×W точное число locations равно $(\lfloor(H-K)/S\rfloor+1)(\lfloor(W-K)/S\rfloor+1)$. Оценка HW/S² из рассматриваемого примера лишь приближение, игнорирующее размер окна и границы. Кроме locations нужны разные масштабы и aspect ratios; большинство crops может быть фоном. Проблема не в том, что окна прямоугольные, а в стоимости перебора и слабом использовании общих вычислений.

R-CNN уменьшает перебор до region proposals, но извлекает CNN-признаки каждого region отдельно. Перекрывающиеся crops снова проходят похожие вычисления. Fast R-CNN сначала строит общую feature map всего изображения, затем RoI Pooling (агрегирование состояний) превращает каждый region в фиксированную grid-форму для head. Разбиение region на bins и pooling внутри bins позволяют head получать одинаковую форму при разных размерах boxes. Однако proposals ещё приходят извне.

Faster R-CNN добавляет RPN, которая на общей карте предсказывает objectness и offsets относительно anchor boxes. У anchor $a=(x_a,y_a,w_a,h_a)$ один распространённый decoding задаёт

$$
x=x_a+w_at_x,\quad y=y_a+h_at_y,\quad
w=w_ae^{t_w},\quad h=h_ae^{t_h}.
$$

Например, anchor=(50,40,20,10), offsets $(0.1,-0.2,\ln1.5,\ln2)$ дают center=(52,38), size=(30,20), то есть corners=(37,28,67,48). Экспонента обеспечивает положительный размер, а координаты центра пока не обязаны ограничиваться anchor cell. Эти offsets относятся к конкретной параметризации; у YOLOv2 ограничение центра через sigmoid другое.

### YOLOv1: что означали 7×7×30 чисел

Рассмотрим архитектурный пример: S=7, B=2 box predictors на cell, K=20 классов. Длина последней оси $B\cdot5+K=30$: десять чисел описывают два boxes и их confidence, двадцать — class scores ячейки. Всего 1470 чисел, включая 98 box candidates. Это фиксированный tensor, из которого получается переменное итоговое множество после отбора.

Original design разделяет class probabilities ячейки между её boxes, и такая организация ограничивает разделение нескольких близких экземпляров. B=2 не означает независимую полноценную per-box классификацию каждой из двух целей в одной cell. В историческом loss были координатные, confidence и class-слагаемые с отдельным весом background; square roots ширины/высоты меняли чувствительность к абсолютным ошибкам размера. Это механизм YOLOv1, который не следует переносить как описание современных heads. [Оригинальная работа YOLO](https://arxiv.org/abs/1506.02640).

### Anchors, multi-scale и anchor-free — отдельные изменения

YOLOv2 использует prior sizes, подобранные по training boxes, и offsets относительно cell. Для центра $b_x=\sigma(t_x)+c_x$, $b_y=\sigma(t_y)+c_y$, для размера $b_w=p_we^{t_w}$, $b_h=p_he^{t_h}$. Все величины должны быть в согласованных единицах grid; перевод в pixels требует stride. Sigmoid держит центр в соответствующей ячейке, в отличие от неограниченного offset RPN.

В разбираемом примере multi-scale training YOLOv2 описан как смена размера каждые десять эпох. В исходной статье это каждые **10 batches**. Смысл — учить одну fully convolutional модель работать на разных разрешениях, а не хранить отдельный detector для каждого размера. Это отличается от multi-scale feature maps внутри одного forward (прямой проход вычисления). У YOLOv3 prediction идёт на нескольких spatial scales, помогая распределять маленькие и крупные объекты; исходный YOLOv3 нельзя описывать CIoU-loss позднейших модификаций. [YOLOv2](https://arxiv.org/abs/1612.08242), [YOLOv3](https://arxiv.org/abs/1804.02767).

Варианты семейства связывают v4/v5 с CSP-структурами, pyramid aggregation, Mosaic и практическими training/export pipelines. Эти идеи относятся к разным частям системы: backbone, neck, аугментации, loss и инфраструктура. Одно семейное имя не доказывает неизменность каждой детали во всех releases. Сравнивать checkpoints нужно по фактическому graph/config; автоматическая лестница «новее → всегда лучше» из перечисления версий не следует.

Anchor-free distance-представление относительно reference point $(x,y)$ даёт box $(x-l,y-t,x+r,y+b)$. Например, point=(20,30), distances=(5,10,15,20) дают corners=(15,20,35,50). Reference point всё ещё существует; исчезли заранее выбранные шаблоны ширины/высоты, а не вся пространственная сетка. Decoupled head отдельно вычисляет class и box outputs; один shared backbone при этом продолжает получать сигналы обоих objectives.

### Как set prediction заменяет задачу удаления дубликатов

В DETR fixed queries дают фиксированное число candidate slots. Training находит взаимно однозначное matching между GT и частью slots с учётом класса и геометрии. Оставшиеся обучаются отсутствию объекта. Если два slots предскажут один GT, только один сможет сопоставиться с ним; второй получает иной training signal, что помогает учить уникальный набор. Object queries — обучаемые запросы к признакам изображения, а не заранее известные метки «первый человек» и «вторая машина». [DETR](https://arxiv.org/abs/2005.12872).

One-to-many assignment обеспечивает несколько положительных обучающих примеров на GT; one-to-one направлен на уникальный inference output. В dual-branch схемах они могут использовать shared features (признаки) и согласованные criteria. Из этого не следует буквальный backward (обратный проход вычисления производных) из output одной головы в output другой: градиентные связи определяет graph, включая detach. Фраза «one-to-one получает градиенты от one-to-many» без уточнения graph слишком сильна.

RT-DETR сохраняет set prediction и использует свой эффективный encoder (кодировщик входа)/query-selection design. Скорость в имени не гарантирует более низкую latency (задержка отдельного запроса) любой конфигурации на любом устройстве: full pipeline и settings нужно измерить. Для YOLO26 режимы и поддерживаемые задачи следует проверять по [официальному описанию](https://docs.ultralytics.com/models/yolo26/); неподтверждённая запись с датой и «depth head» не служит достаточным подтверждением этих возможностей.

## Losses: локализация и класс

Box losses могут использовать L1 или IoU-family. Простой $1-\operatorname{IoU}$ плохо информирует о направлении при непересекающихся boxes. GIoU добавляет penalty за пустую часть минимального enclosing box:

$$
\operatorname{GIoU}=\operatorname{IoU}-\frac{|C|-|A\cup B|}{|C|}.
$$

Для A/B выше C area9, union7, GIoU=$\frac17-\frac29\approx-0.079365$. Loss1-GIoU может быть больше1, это не автоматически ошибка.

DIoU добавляет squared distance centres, нормированную squared diagonal enclosing box; CIoU дополнительно учитывает aspect ratio по своей формуле. Они задают разные surrogate geometry, не гарантируя одинаковое улучшение AP. Distribution Focal Loss в некоторых heads обучает дискретизированные distance distributions; это не то же, что focal classification loss. Версии без DFL не надо описывать старым head из памяти.

Для дисбаланса foreground/background classification focal loss имеет вид $-\alpha_t(1-p_t)^\gamma\log p_t$; easy examples получают меньший вес. Assignment, class loss и box loss связаны, но это три разных компонента.

### Почему IoU-loss не всегда подсказывает, куда двигать box

Когда два boxes разделены зазором, IoU остаётся нулевой при небольшом перемещении, пока пересечение не появится. В этой области один $1-\operatorname{IoU}$ не различает некоторые полезные направления. GIoU учитывает размер enclosing rectangle, DIoU — расстояние центров, CIoU — дополнительный aspect-ratio term. Они добавляют разные сигналы, а не просто разные названия одного коэффициента.

Для $A=(0,0,1,1)$ и $B=(3,0,4,1)$ union=2, enclosing area=4, IoU=0 и GIoU=−0.5. Если B сдвинуть влево до (2,0,3,1), IoU всё ещё 0, но enclosing area=3 и GIoU=−1/3. Loss $1-\operatorname{GIoU}$ уменьшился с 1.5 до 4/3: полезное направление стало различимо ещё до overlap. Это иллюстрирует геометрическую мотивацию, не гарантируя глобально простую оптимизацию всех координат.

Classification focal loss уменьшает вклад лёгких background examples через $(1-p_t)^\gamma$. Distribution Focal Loss решает другую задачу: распределение координатной дистанции может кодировать промежуточное значение через соседние bins. Например, target=2.3 можно разложить на веса 0.7 для bin 2 и 0.3 для bin 3; ожидаемое значение этих двух bins равно 2.3. Слово focal не делает эти objectives взаимозаменяемыми, и конкретная head может использовать иной способ регрессии.

## Проверка и ошибки

Визуализировать предсказания после точного inverse resize/letterbox transform, иначе boxes выглядят неверными даже при корректных model outputs. Отдельно анализировать missed objects, duplicates, class confusion, localization errors и crowded/small cases. Full latency включает preprocessing, transfer, forward, postprocessing и queue.

1. Почему duplicate — FP даже при IoU1? **Ответ:** GT уже сопоставлен, один экземпляр нельзя засчитать дважды.
2. Почему AP50 и AP50:95 различаются? **Ответ:** набор localisation thresholds разный.
3. Что теряется при выключении NMS без смены training? **Ответ:** остаются duplicates, если output не обучен unique set prediction.

**Практика:** пересчитать boxes и AP пример; затем поставить P2 выше P1 по score и проверить matching. Для реальной модели на одном split (разбиение данных) изменять один фактор: resolution, score operating threshold или NMS threshold. Сохранять все error slices и различать benchmark AP и application precision/recall.





## Архитектуры: подробные схемы

### FPN: lateral и top-down связи

![FPN: lateral и top-down связи](<../оформление/course/arch-fpn.svg>)

Top-down путь передаёт глубокую семантику к более подробным сеткам. На каждом уровне сумма требует совпадения spatial shape и channels после lateral projection. Detector heads читают P-levels; matching и AP остаются отдельными этапами.

### Faster R-CNN

![Faster R-CNN](<../оформление/course/arch-faster-rcnn.svg>)

RPN формирует кандидатов; второй этап читает признаки региона и уточняет класс/рамку. Отбор proposals, task losses и evaluation matching — разные операции. Конкретный вариант использует свой RoI operator.

### SSD

![SSD](<../оформление/course/arch-ssd.svg>)

SSD предсказывает offsets и классы относительно default boxes на нескольких масштабах. Параметры anchors и matching training positives влияют на targets; NMS применяется к готовым кандидатам.

### RetinaNet

![RetinaNet](<../оформление/course/arch-retinanet.svg>)

Focal loss уменьшает относительный вклад лёгких примеров classification; она не отменяет anchor matching или box regression. FPN обеспечивает несколько масштабов, heads используют разные output channels.

### YOLO: семейство dense detectors

![YOLO: семейство dense detectors](<../оформление/course/arch-yolo.svg>)

Функциональная схема семейства отделяет backbone, neck и task heads. Anchor-based и anchor-free варианты различаются targets и decode; нельзя приписывать одной схеме конкретные loss и output layout всех версий.

### DETR: set prediction

![DETR: set prediction](<../оформление/course/arch-detr.svg>)

Запросы читают признаки изображения через decoder (декодер). Hungarian matching назначает предсказания targets для loss; это не NMS по попарному IoU. Число queries и no-object convention входят в конкретную реализацию.

### YOLOv1: grid prediction

![YOLOv1: grid prediction](<../оформление/course/arch-yolov1.svg>)

Это историческая grid-постановка YOLOv1. Координаты и square-root size terms из её loss не описывают автоматически последующие поколения YOLO.

### YOLOv2: anchor offsets

![YOLOv2: anchor offsets](<../оформление/course/arch-yolov2.svg>)

Центр вычисляется относительно cell, размеры — относительно prior. Multi-scale training меняет входной размер между batches; оно не тождественно нескольким feature scales внутри одного forward.

### YOLOv3: несколько масштабов

![YOLOv3: несколько масштабов](<../оформление/course/arch-yolov3.svg>)

Предсказания на разных сетках поддерживают разные размеры объектов. Сборка кандидатов не делает loss исходного YOLOv3 равным loss более поздних вариантов.

### RT-DETR

![RT-DETR](<../оформление/course/arch-rtdetr.svg>)

Схема отделяет кодирование многомасштабных features и выбор queries от decoder. Отсутствие обычной NMS связано с set prediction; фактическая latency зависит от configuration и hardware.

## Результат главы и переход дальше

Следующие темы: [Instance segmentation: Mask R-CNN и prototype masks](<cv-instance.md>), [Ориентированные рамки и периодичность угла](<cv-obb.md>), [Tracking: Kalman, association и идентичность](<cv-tracking.md>), [CLIP, открытый словарь и YOLO-World](<cv-open-vocabulary.md>).

[Маршрут модуля](<README.md>) · [Происхождение и точные места переноса](<../editorial/coverage.md>)

## Как читать углубления этой ветки


Эта глава остаётся входом в detection. От неё идут самостоятельные разборы [Faster R-CNN](<cv-faster-rcnn.md>), [RetinaNet/FCOS](<cv-dense-detectors.md>), [YOLO](<cv-yolo.md>), [DETR](<cv-detr.md>) и [ускоренных Transformer detectors](<cv-deformable.md>). Assignment при обучении создаёт targets; evaluation matching считает TP/FP; NMS фильтрует дубликаты при inference. Даже если везде используется IoU, это разные операции.

Пустой кадр с objects absent должен оставаться допустимым случаем: классификационная часть может обучаться фону, а regression не должна получать фиктивный box. Missing annotation, напротив, способна ошибочно превратить настоящий объект в background.

## Разобранный пример для проверки контракта


Два GT и три predictions, отсортированные по confidence, при заданном IoU threshold дают TP,FP,TP. Precision по рангу: 1,1/2,2/3; recall: 1/2,1/2,1. Неинтерполированная AP по двум relevant ranks равна $(1+2/3)/2=5/6$. COCO использует другой interpolation и набор thresholds, поэтому это учебная AP, а не воспроизведение COCO evaluator.

## Ограничения переноса

Сравнение AP требует одинаковых categories, max detections, crowd/ignore conventions и scale slices. Подмена evaluator меняет результат без изменения model weights.

## Самопроверка с ответами

**Практика:** добавьте второй prediction той же GT с высоким confidence. Он может стать FP после one-to-one matching, даже если его box геометрически точен.

**Удаляет ли training matching дубликаты на inference?** Не как обычная NMS: оно создаёт обучающий стимул, а результат зависит от модели.

**Почему AP50 и AP75 могут расходиться?** Более строгий IoU чувствительнее к localization error.

## Схема вычислений

![Detection: множества объектов, IoU, AP и архитектуры: вычислительные этапы](<../оформление/course/expanded-cv-detection.svg>)

Три разных операции: GT + candidates при training → Assignment → targets / loss → Inference predictions → NMS → Evaluation matching → AP.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Источники и продолжение

- [COCO evaluation implementation](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)

[Следующие углубления: cv-faster-rcnn, cv-dense-detectors, cv-yolo, cv-detr, cv-obb, cv-instance, cv-open-vocabulary, cv-tracking, cv-bev, cv-text-detection, cv-diagnosis](<README.md>) · [Сценарии подраздела](<03-detection/План.md>)
