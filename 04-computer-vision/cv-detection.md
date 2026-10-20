# Detection: множества объектов, IoU, AP и архитектуры

**Уровень:** общее ядро.

**Предпосылки:** [Изображение, карты признаков и блоки модели](<cv-input.md>), [Классификация, CNN и residual learning](<cv-classification.md>)

**Используют:** [Instance segmentation: Mask R-CNN и prototype masks](<cv-instance.md>), [Ориентированные рамки и периодичность угла](<cv-obb.md>), [Tracking: Kalman, association и идентичность](<cv-tracking.md>), [CLIP, открытый словарь и YOLO-World](<cv-open-vocabulary.md>)

**Учебный вопрос:** Как получить множество рамок и оценить его с учётом matching?





## Карта вычисления и границы этапов

![Detection: множества объектов, IoU, AP и архитектуры: входы, преобразования и выходы](<../оформление/course/cv-detection.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Сравните формы и смысл каждого перехода с формулами главы. Выход преобразования и доказательство его качества — разные объекты: последняя проверка использует отдельный протокол оценки.

<!-- unit:caade8ae384a953f02 -->
<a id="u-caade8ae384a953f02"></a>

## Выход — множество, а не одна метка

<!-- /unit:caade8ae384a953f02 -->

<!-- unit:906bb6331021823bf1 -->
<a id="u-906bb6331021823bf1"></a>

Для изображения output состоит из объектов `(box,class,score)`. Число объектов заранее неизвестно, порядок списка обычно не имеет семантики. Annotation задаёт box каждого экземпляра, а не один label всего изображения. Пропущенные в разметке реальные объекты могут при обучении/оценке ошибочно считаться background.

<!-- /unit:906bb6331021823bf1 -->

<!-- unit:f39192c519c9b27e34 -->
<a id="u-f39192c519c9b27e34"></a>

В `xyxy` box задаёт левый верхний и правый нижний углы $(x_1,y_1,x_2,y_2)$. В `xywh` надо уточнить, x/y — центр или corner. Нормированные и pixel coordinates нельзя смешивать. Для непрерывной геометрии площадь $(x_2-x_1)(y_2-y_1)$; исторические inclusive pixel conventions с $+1$ — другой protocol.

<!-- /unit:f39192c519c9b27e34 -->

<!-- unit:dbe27c2cb1d4b371a7 -->
<a id="u-dbe27c2cb1d4b371a7"></a>

IoU=$\frac{|A\cap B|}{|A\cup B|}$. У A=(0,0,2,2), B=(1,1,3,3) площади 4 и4, пересечение1, union7, IoU=1/7≈0.142857. У непересекающихся boxes IoU0. Если boxes вырожденные, protocol должен определить обработку.

<!-- /unit:dbe27c2cb1d4b371a7 -->

<!-- unit:605eee2d38aa8bfb88 -->
<a id="u-605eee2d38aa8bfb88"></a>

Class confidence не тождественна IoU: модель может уверенно назвать класс и плохо локализовать объект. В разных heads есть отдельный objectness или совместный score качества/класса. Нельзя универсально обещать ровно «objectness×class probability» для каждого YOLO checkpoint.

<!-- /unit:605eee2d38aa8bfb88 -->

<!-- unit:fd01c8c8105dfebe84 -->
<a id="u-fd01c8c8105dfebe84"></a>

### Почему переменное число объектов не запрещает фиксированный tensor

<!-- /unit:fd01c8c8105dfebe84 -->

<!-- unit:7b459ea6e221536971 -->
<a id="u-7b459ea6e221536971"></a>

В разбираемом примере сказано, что нельзя выдать фиксированный вектор, поскольку число объектов M неизвестно. Точнее: нельзя свести задачу к одному глобальному class vector как в классификации. Реализация вполне может выдавать фиксированное число кандидатов или queries, затем отбирать действительные объекты. Например, сто slots содержат scores и boxes; slots с no-object или низким score не включаются в итоговое множество. Нужно различать внутреннюю форму tensor и семантическое число найденных объектов.

<!-- /unit:7b459ea6e221536971 -->

<!-- unit:f479500a1b610af6d5 -->
<a id="u-f479500a1b610af6d5"></a>

Список detections не имеет естественного порядка: две одинаковые по смыслу выдачи могут различаться перестановкой строк. Для training это создаёт задачу assignment — кому из кандидатов назначить какой GT. На evaluation задача похожа только внешне: там по объявленному protocol решается, что засчитать TP. Переносить training assignment непосредственно в метрику нельзя.

<!-- /unit:f479500a1b610af6d5 -->

<!-- unit:ca1829f7cb3a023401 -->
<a id="u-ca1829f7cb3a023401"></a>

### Форматы VOC, COCO и YOLO на одном объекте

<!-- /unit:ca1829f7cb3a023401 -->

<!-- unit:e114b53c07cbf8e7a3 -->
<a id="u-e114b53c07cbf8e7a3"></a>

Сохраним пример источника: box с углами (100,200) и (300,400). В COCO-подобной записи верхний левый угол и размер будут `[100,200,200,200]`. Для изображения W=640,H=480 центр (200,300), нормированные координаты YOLO — $(200/640,300/480,200/640,200/480)=(0.3125,0.625,0.3125,0.416667)$. Строка с class ID 0 начинается с `0`, затем идут эти четыре значения.

<!-- /unit:e114b53c07cbf8e7a3 -->

<!-- unit:5363e611142e30c68f -->
<a id="u-5363e611142e30c68f"></a>

Обратное преобразование: $x_1=(x_c-w/2)W$, $x_2=(x_c+w/2)W$, аналогично y. Проверка round trip должна восстановить углы. Делить все четыре координаты на ширину изображения ошибочно для непрямоугольного aspect ratio. Путаница «xywh с центром» и «xywh с corner» тоже даст допустимые числа неправильной геометрии.

<!-- /unit:5363e611142e30c68f -->

<!-- unit:25029b74f6d570763f -->
<a id="u-25029b74f6d570763f"></a>

VOC исторически использует XML, COCO — JSON с image/category IDs, распространённый YOLO detection label — строку на объект. Число классов 20 или 80 относится к известным benchmark-наборам, а не к принципиальному ограничению XML и JSON. Пользовательский датасет этих форматов может иметь другое число классов; category IDs также не обязаны совпадать с непрерывными индексами model head.

<!-- /unit:25029b74f6d570763f -->

<!-- unit:f6cd7c76c51c696612 -->
<a id="u-f6cd7c76c51c696612"></a>

### Вычисление IoU по координатам без скрытых шагов

<!-- /unit:f6cd7c76c51c696612 -->

<!-- unit:9bed007ca157be99fa -->
<a id="u-9bed007ca157be99fa"></a>

У двух axis-aligned boxes левый верхний угол пересечения получается через max нижних границ, правый нижний — через min верхних. Поэтому

<!-- /unit:9bed007ca157be99fa -->

<!-- unit:a62de713bfb1a170f9 -->
<a id="u-a62de713bfb1a170f9"></a>

$$
w_\cap=\max(0,\min(x_2^A,x_2^B)-\max(x_1^A,x_1^B)),
$$

<!-- /unit:a62de713bfb1a170f9 -->

<!-- unit:fbfc830c1d4668982e -->
<a id="u-fbfc830c1d4668982e"></a>

$$
h_\cap=\max(0,\min(y_2^A,y_2^B)-\max(y_1^A,y_1^B)).
$$

<!-- /unit:fbfc830c1d4668982e -->

<!-- unit:6a78cc4afdcf028da7 -->
<a id="u-6a78cc4afdcf028da7"></a>

Площадь пересечения равна произведению этих длин; union — сумма площадей минус пересечение. Max с нулём принципиален: отрицательная длина при раздельных прямоугольниках не должна давать положительную площадь после умножения двух отрицательных чисел.

<!-- /unit:6a78cc4afdcf028da7 -->

<!-- unit:c63ff739498ee7aec5 -->
<a id="u-c63ff739498ee7aec5"></a>

При общем масштабировании всех координат в a раз площади умножаются на a², отношение сохраняется. Это и означает масштабную инвариантность IoU в геометрическом смысле. Но ошибка в два пикселя для маленького объекта и для большого — не одно общее масштабирование пары boxes: маленький box может потерять большую долю перекрытия. Поэтому высокие IoU thresholds особенно требовательны к локализации малых объектов.

<!-- /unit:c63ff739498ee7aec5 -->

<!-- unit:3fb1a0511e318c2dfa -->
<a id="u-3fb1a0511e318c2dfa"></a>

## Matching: когда prediction становится TP

<!-- /unit:3fb1a0511e318c2dfa -->

<!-- unit:e37383f3cce0ba1ad2 -->
<a id="u-e37383f3cce0ba1ad2"></a>

Для одной class и заданного IoU threshold τ predictions сортируют по score. Каждая prediction может сопоставиться с ещё не занятым GT того же class, если IoU удовлетворяет protocol. Первое подходящее совпадение TP, duplicate того же GT — FP. Непокрытые GT остаются FN. Оценка должна учитывать ignore/crowd правила выбранного benchmark, а не произвольную новую реализацию.

<!-- /unit:e37383f3cce0ba1ad2 -->

<!-- unit:6cb5a3520ab40a578d -->
<a id="u-6cb5a3520ab40a578d"></a>

Пример: два GT, три predictions по убывающему score. P1 правильно покрыла GT1; P2 — duplicate GT1; P3 — GT2. Получаем TP flags $\left(1,\,0,\,1\right)$:

<!-- /unit:6cb5a3520ab40a578d -->

<!-- unit:66618bb27a5686f2ff -->
<a id="u-66618bb27a5686f2ff"></a>

$$
\begin{array}{c|cc|cc}\text{после}&\mathrm{TP}&\mathrm{FP}&\mathrm{Precision}&\mathrm{Recall}\\\hline P_1&1&0&1&0.5\\P_2&1&1&0.5&0.5\\P_3&2&1&\frac23&1\end{array}
$$

<!-- /unit:66618bb27a5686f2ff -->

<!-- unit:f52093d341765e5c5d -->
<a id="u-f52093d341765e5c5d"></a>

AP агрегирует precision-recall curve, но **точное правило интегрирования** существенно. Для continuous interpolated envelope здесь первая половина recall имеет precision1, вторая2/3, AP=5/6≈0.8333. Это учебный вариант, не численный результат официального COCO evaluator с его сеткой recall/area/maxDets. Различные AP conventions не смешивать в отчёте.

<!-- /unit:f52093d341765e5c5d -->

<!-- unit:61887bb069ee822219 -->
<a id="u-61887bb069ee822219"></a>

mAP усредняет AP по выбранным классам и, если protocol требует, IoU thresholds. AP@0.5 проще, чем COCO-style AP@[0.5:0.95]: второй усредняет thresholds 0.50,0.55,…,0.95. Рост AP50 при неизменной высокой-threshold AP может означать, что объекты находятся, но box localization ещё недостаточно точна. Small/medium/large AP и per-class errors дают полезные срезы.

<!-- /unit:61887bb069ee822219 -->

<!-- unit:8688d802cb111c6f61 -->
<a id="u-8688d802cb111c6f61"></a>

Порог score, выбранный для приложения, задаёт одну operating point. Для AP нужен корректный набор ranked detections по protocol; слишком раннее отбрасывание низких scores способно искусственно ограничить attainable recall.

<!-- /unit:8688d802cb111c6f61 -->

<!-- unit:ada1f310a61358a98f -->
<a id="u-ada1f310a61358a98f"></a>

### Возвращаем исходный пример AP: пять GT и восемь predictions

<!-- /unit:ada1f310a61358a98f -->

<!-- unit:33d01084b9e344806d -->
<a id="u-33d01084b9e344806d"></a>

В разбираемом примере для одного класса были пять истинных объектов и восемь detections со scores 0.95,0.90,0.85,0.80,0.70,0.60,0.50,0.40. При выбранном IoU threshold flags равны [TP,TP,FP,TP,FP,TP,FP,TP]. После каждого шага precision равна [1,1,2/3,3/4,3/5,4/6,4/7,5/8], recall — [1/5,2/5,2/5,3/5,3/5,4/5,4/5,1].

<!-- /unit:33d01084b9e344806d -->

<!-- unit:406afdec38dc70a502 -->
<a id="u-406afdec38dc70a502"></a>

После третьего detection precision упала до 2/3, а после четвёртого выросла до 3/4. Поэтому PR-кривая не обязана монотонно падать — монотонность относится к envelope после интерполяции. Для каждого r она берёт лучший precision, доступный при recall не меньше r:

<!-- /unit:406afdec38dc70a502 -->

<!-- unit:9735bd70b050b6c6e8 -->
<a id="u-9735bd70b050b6c6e8"></a>

$$
p_{\rm interp}(r)=\max_{\widetilde r\ge r}p(\widetilde r).
$$

<!-- /unit:9735bd70b050b6c6e8 -->

<!-- unit:b60199bd5dbe61c736 -->
<a id="u-b60199bd5dbe61c736"></a>

На первых двух интервалах recall ширины 1/5 envelope равна 1; на следующих — 3/4,2/3,5/8. Continuous interpolated AP в этом примере:

<!-- /unit:b60199bd5dbe61c736 -->

<!-- unit:4a1bdd4011a5d66d06 -->
<a id="u-4a1bdd4011a5d66d06"></a>

$$
\operatorname{AP}=\frac15\left(1+1+\frac34+\frac23+\frac58\right)=\frac{97}{120}\approx0.808333.
$$

<!-- /unit:4a1bdd4011a5d66d06 -->

<!-- unit:259ba9f2bfbfe4ee05 -->
<a id="u-259ba9f2bfbfe4ee05"></a>

Получается «около 0.81», но теперь известно точное правило. Равенство простому среднему здесь связано с равными приростами recall по 1/5. Для неравных интервалов нужно умножать precision на их ширины. VOC2007 с 11 recall-точками и COCO с собственной сеткой дадут другой численный protocol; один учебный integral не подменяет официальный evaluator.

<!-- /unit:259ba9f2bfbfe4ee05 -->

<!-- unit:30dff782db83c73b2a -->
<a id="u-30dff782db83c73b2a"></a>

Если оставить только predictions со score≥0.65, будут обработаны первые пять: TP=3, FP=2, recall=3/5. Модель сохранила более уверенные candidates, но уже не может достигнуть recall=1 на этом списке. AP характеризует ranking по набору thresholds; фиксированный operating threshold приложения выбирается отдельно. Нельзя описать улучшение AP одним повышением confidence threshold без проверки изменения самого списка и protocol.

<!-- /unit:30dff782db83c73b2a -->

<!-- unit:f36d76ab693e896163 -->
<a id="u-f36d76ab693e896163"></a>

### Как ошибки разных типов входят в counts

<!-- /unit:f36d76ab693e896163 -->

<!-- unit:6b0b02bfe0715048e7 -->
<a id="u-6b0b02bfe0715048e7"></a>

Правильный class при IoU ниже threshold считается FP, а неудачно покрытый GT может одновременно остаться FN. Неправильный class тоже может создать FP для одного класса и FN для другого. Duplicate — FP даже при прекрасной геометрии, если соответствующий GT уже занят. Это объясняет, почему «модель нашла объект глазами» и «метрика засчитала TP» могут расходиться.

<!-- /unit:6b0b02bfe0715048e7 -->

<!-- unit:bb56db6907af5ca2f5 -->
<a id="u-bb56db6907af5ca2f5"></a>

Порядок scores влияет на precision раньше по списку. Если duplicate предсказывается выше более точного box, он может первым занять GT, а следующий стать FP. На одном IoU threshold оба могли бы подходить, на другом первый уже не пройдёт. Отсюда полезность срезов AP50, AP75 и более строгих порогов: они разделяют грубое нахождение и точную локализацию.

<!-- /unit:bb56db6907af5ca2f5 -->

<!-- unit:2198b1851abc9cbe7a -->
<a id="u-2198b1851abc9cbe7a"></a>

## Зачем NMS и где её границы

<!-- /unit:2198b1851abc9cbe7a -->

<!-- unit:b0579c7a028313eba4 -->
<a id="u-b0579c7a028313eba4"></a>

Greedy NMS для каждой class: выбрать максимальный score, сохранить box, подавить оставшиеся boxes с IoU выше threshold, повторить. Она удаляет дубликаты, но может убрать разные близкие/перекрывающиеся объекты. Class-agnostic NMS объединяет classes и имеет другие trade-offs. Soft-NMS уменьшает scores вместо жёсткого удаления.

<!-- /unit:b0579c7a028313eba4 -->

<!-- unit:20a9c56da40f4d5d04 -->
<a id="u-20a9c56da40f4d5d04"></a>

Если A=(0,0,2,2), C=(0.2,0.2,2.2,2.2), intersection3.24, union4.76, IoU≈0.68067. При NMS threshold0.5 lower-score box будет подавлена в одной class. Правильно ли это, зависит от того, duplicate это или два сильно перекрытых объекта.

<!-- /unit:20a9c56da40f4d5d04 -->

<!-- unit:cdff882a74cca7b2d4 -->
<a id="u-cdff882a74cca7b2d4"></a>

Нельзя ставить одинаковый IoU threshold для training assignment, NMS и evaluation только потому, что везде написано IoU. Они управляют разными решениями. NMS-free inference достигается специальной схемой обучения/выходов, а не просто выключением NMS у головы, обученной выдавать много duplicates.

<!-- /unit:cdff882a74cca7b2d4 -->

<!-- unit:f7e99606cac4872d48 -->
<a id="u-f7e99606cac4872d48"></a>

### Пошаговая NMS и смысл её порога

<!-- /unit:f7e99606cac4872d48 -->

<!-- unit:89b28a90f7a158c75b -->
<a id="u-89b28a90f7a158c75b"></a>

Пусть три boxes одного класса: A со score=0.95, C со score=0.90 и удалённый D со score=0.80. У A/C IoU≈0.68067. С threshold=0.5 сначала сохраняется A, затем C подавляется как сильно пересекающаяся с сохранённой; D остаётся и выбирается следующим. При threshold=0.7 C не будет подавлена. В отличие от score threshold, который регулирует минимальную оценку кандидата, этот порог регулирует допустимое перекрытие уже выбранных boxes.

<!-- /unit:89b28a90f7a158c75b -->

<!-- unit:d0380b2ce0bf95d2c5 -->
<a id="u-d0380b2ce0bf95d2c5"></a>

Если A и C — два реальных человека в толпе, suppression ошибочно уменьшит recall. Если они описывают одного человека, suppression удалит duplicate и может улучшить precision. Поэтому один и тот же геометрический критерий не знает истинного числа экземпляров. Class-aware вариант применяет его отдельно, class-agnostic может подавить и boxes разных классов. У Soft-NMS score понижается вместо немедленного удаления, например $s_j\leftarrow s_j\exp(-\operatorname{IoU}^2/\sigma)$ в Gaussian-варианте; затем ранжирование обновляется.

<!-- /unit:d0380b2ce0bf95d2c5 -->

<!-- unit:818ef075a40de43559 -->
<a id="u-818ef075a40de43559"></a>

В разбираемом примере жёсткая NMS названа причиной того, что «нельзя обучать end-to-end». Точное ограничение: дискретные выборы обычной greedy NMS не дают удобной обычной производной по решению keep/drop. Но многие детекторы обучают differentiable raw outputs без прохождения loss через inference NMS. Следовательно, наличие NMS на inference не запрещает совместное обучение backbone/head. NMS-free set prediction меняет задачу дубликатов, а не просто «разрешает backprop, которого раньше вообще не было».

<!-- /unit:818ef075a40de43559 -->

<!-- unit:d515a001c879d0f70f -->
<a id="u-d515a001c879d0f70f"></a>

## Two-stage, one-stage и признаки разных масштабов

<!-- /unit:d515a001c879d0f70f -->

<!-- unit:10ce3249a6fddb68ff -->
<a id="u-10ce3249a6fddb68ff"></a>

Исторический sliding window запускает classifier на множестве crops, что дорого и дублирует вычисления. R-CNN использует region proposals и обработку регионов. Fast R-CNN разделяет shared feature extraction и region-wise head; Faster R-CNN добавляет обучаемую RPN. В современных описаниях RoIAlign/другие изменения надо отличать от исходного RoIPool, не приписывая их всем версиям сразу.

<!-- /unit:10ce3249a6fddb68ff -->

<!-- unit:e20c0cfb1010e3da3b -->
<a id="u-e20c0cfb1010e3da3b"></a>

One-stage detector предсказывает кандидатов на feature maps без отдельной классической proposal-classification стадии. SSD/YOLO — важные примеры, но «one-stage всегда быстрее, two-stage всегда точнее» не теорема: сравниваются реальные models, resolution и hardware.

<!-- /unit:e20c0cfb1010e3da3b -->

<!-- unit:4ae0fb614457c3fd66 -->
<a id="u-4ae0fb614457c3fd66"></a>

FPN объединяет семантику глубоких низкоразрешённых maps с деталями более высокого resolution. Это важно для малых объектов: после сильного downsampling объект может занимать меньше одной эффективной клетки. Повышение input resolution увеличивает compute, но иногда исправляет bottleneck данных/геометрии лучше, чем увеличение model size.

<!-- /unit:4ae0fb614457c3fd66 -->

<!-- unit:74232d5971bb5e89b2 -->
<a id="u-74232d5971bb5e89b2"></a>

YOLOv1 — grid-based прямое предсказание boxes без anchor механизма последующих v2/v3. Anchors — заранее заданные reference shapes, offsets относительно которых учатся. Anchor-free methods убирают именно эти шаблоны, но всё равно могут использовать feature-grid reference points. Decoupled head разделяет prediction branches, например class/box; это ещё одна независимая идея.

<!-- /unit:74232d5971bb5e89b2 -->

<!-- unit:cc0011a6736dd849da -->
<a id="u-cc0011a6736dd849da"></a>

Training assignment отвечает, какие predictions получают targets какого GT. One-to-many даёт несколько positives на GT и обычно требует удаления duplicates. One-to-one способствует unique output, но нуждается в подходящем loss/assignment. В YOLOv10 и последующих designs могут сочетаться training branches с разными assignment. Для YOLO26 доступен отдельный one-to-one NMS-free inference path; конкретный default/path нужно задавать явно. [Официальное описание YOLO26](https://docs.ultralytics.com/models/yolo26/).

<!-- /unit:cc0011a6736dd849da -->

<!-- unit:cd1c4d75c976fbeb77 -->
<a id="u-cd1c4d75c976fbeb77"></a>

DETR формулирует set prediction с object queries и bipartite/Hungarian matching training outputs к GT. Дополнительные predictions обучают no-object class/эквиваленту protocol. Отсутствие обычной NMS связано с set objective, а не только со словом Transformer. RT-DETR использует связанные идеи с акцентом на speed; details выбранного variant проверяют отдельно.

<!-- /unit:cd1c4d75c976fbeb77 -->

<!-- unit:b771b76f390ad224a9 -->
<a id="u-b771b76f390ad224a9"></a>

### От sliding windows к shared feature map

<!-- /unit:b771b76f390ad224a9 -->

<!-- unit:e0d724b262d6c61d7d -->
<a id="u-e0d724b262d6c61d7d"></a>

Для окна K×K со stride S без padding по изображению H×W точное число locations равно $(\lfloor(H-K)/S\rfloor+1)(\lfloor(W-K)/S\rfloor+1)$. Оценка HW/S² из рассматриваемого примера лишь приближение, игнорирующее размер окна и границы. Кроме locations нужны разные масштабы и aspect ratios; большинство crops может быть фоном. Проблема не в том, что окна прямоугольные, а в стоимости перебора и слабом использовании общих вычислений.

<!-- /unit:e0d724b262d6c61d7d -->

<!-- unit:75f9c06d907534e18f -->
<a id="u-75f9c06d907534e18f"></a>

R-CNN уменьшает перебор до region proposals, но извлекает CNN-признаки каждого region отдельно. Перекрывающиеся crops снова проходят похожие вычисления. Fast R-CNN сначала строит общую feature map всего изображения, затем RoI Pooling превращает каждый region в фиксированную grid-форму для head. Разбиение region на bins и pooling внутри bins позволяют head получать одинаковую форму при разных размерах boxes. Однако proposals ещё приходят извне.

<!-- /unit:75f9c06d907534e18f -->

<!-- unit:a0c535e17d36445d0b -->
<a id="u-a0c535e17d36445d0b"></a>

Faster R-CNN добавляет RPN, которая на общей карте предсказывает objectness и offsets относительно anchor boxes. У anchor $a=(x_a,y_a,w_a,h_a)$ один распространённый decoding задаёт

<!-- /unit:a0c535e17d36445d0b -->

<!-- unit:e812580d72337449cc -->
<a id="u-e812580d72337449cc"></a>

$$
x=x_a+w_at_x,\quad y=y_a+h_at_y,\quad
w=w_ae^{t_w},\quad h=h_ae^{t_h}.
$$

<!-- /unit:e812580d72337449cc -->

<!-- unit:3c80c7dd0f0255eef6 -->
<a id="u-3c80c7dd0f0255eef6"></a>

Например, anchor=(50,40,20,10), offsets $(0.1,-0.2,\ln1.5,\ln2)$ дают center=(52,38), size=(30,20), то есть corners=(37,28,67,48). Экспонента обеспечивает положительный размер, а координаты центра пока не обязаны ограничиваться anchor cell. Эти offsets относятся к конкретной параметризации; у YOLOv2 ограничение центра через sigmoid другое.

<!-- /unit:3c80c7dd0f0255eef6 -->

<!-- unit:11c51c916a0ea62523 -->
<a id="u-11c51c916a0ea62523"></a>

### YOLOv1: что означали 7×7×30 чисел

<!-- /unit:11c51c916a0ea62523 -->

<!-- unit:49746e5b28e344b898 -->
<a id="u-49746e5b28e344b898"></a>

Рассмотрим архитектурный пример: S=7, B=2 box predictors на cell, K=20 классов. Длина последней оси $B\cdot5+K=30$: десять чисел описывают два boxes и их confidence, двадцать — class scores ячейки. Всего 1470 чисел, включая 98 box candidates. Это фиксированный tensor, из которого получается переменное итоговое множество после отбора.

<!-- /unit:49746e5b28e344b898 -->

<!-- unit:1cff7a25727296c14b -->
<a id="u-1cff7a25727296c14b"></a>

Original design разделяет class probabilities ячейки между её boxes, и такая организация ограничивает разделение нескольких близких экземпляров. B=2 не означает независимую полноценную per-box классификацию каждой из двух целей в одной cell. В историческом loss были координатные, confidence и class-слагаемые с отдельным весом background; square roots ширины/высоты меняли чувствительность к абсолютным ошибкам размера. Это механизм YOLOv1, который не следует переносить как описание современных heads. [Оригинальная работа YOLO](https://arxiv.org/abs/1506.02640).

<!-- /unit:1cff7a25727296c14b -->

<!-- unit:0bccf229845ccdebfc -->
<a id="u-0bccf229845ccdebfc"></a>

### Anchors, multi-scale и anchor-free — отдельные изменения

<!-- /unit:0bccf229845ccdebfc -->

<!-- unit:bb83bde0addec26d99 -->
<a id="u-bb83bde0addec26d99"></a>

YOLOv2 использует prior sizes, подобранные по training boxes, и offsets относительно cell. Для центра $b_x=\sigma(t_x)+c_x$, $b_y=\sigma(t_y)+c_y$, для размера $b_w=p_we^{t_w}$, $b_h=p_he^{t_h}$. Все величины должны быть в согласованных единицах grid; перевод в pixels требует stride. Sigmoid держит центр в соответствующей ячейке, в отличие от неограниченного offset RPN.

<!-- /unit:bb83bde0addec26d99 -->

<!-- unit:8feceed4f668428111 -->
<a id="u-8feceed4f668428111"></a>

В разбираемом примере multi-scale training YOLOv2 описан как смена размера каждые десять эпох. В исходной статье это каждые **10 batches**. Смысл — учить одну fully convolutional модель работать на разных разрешениях, а не хранить отдельный detector для каждого размера. Это отличается от multi-scale feature maps внутри одного forward. У YOLOv3 prediction идёт на нескольких spatial scales, помогая распределять маленькие и крупные объекты; исходный YOLOv3 нельзя описывать CIoU-loss позднейших модификаций. [YOLOv2](https://arxiv.org/abs/1612.08242), [YOLOv3](https://arxiv.org/abs/1804.02767).

<!-- /unit:8feceed4f668428111 -->

<!-- unit:e7602e1d6ed3e89ece -->
<a id="u-e7602e1d6ed3e89ece"></a>

Варианты семейства связывают v4/v5 с CSP-структурами, pyramid aggregation, Mosaic и практическими training/export pipelines. Эти идеи относятся к разным частям системы: backbone, neck, аугментации, loss и инфраструктура. Одно семейное имя не доказывает неизменность каждой детали во всех releases. Сравнивать checkpoints нужно по фактическому graph/config; автоматическая лестница «новее → всегда лучше» из перечисления версий не следует.

<!-- /unit:e7602e1d6ed3e89ece -->

<!-- unit:bea21f66f9ec8a44c1 -->
<a id="u-bea21f66f9ec8a44c1"></a>

Anchor-free distance-представление относительно reference point $(x,y)$ даёт box $(x-l,y-t,x+r,y+b)$. Например, point=(20,30), distances=(5,10,15,20) дают corners=(15,20,35,50). Reference point всё ещё существует; исчезли заранее выбранные шаблоны ширины/высоты, а не вся пространственная сетка. Decoupled head отдельно вычисляет class и box outputs; один shared backbone при этом продолжает получать сигналы обоих objectives.

<!-- /unit:bea21f66f9ec8a44c1 -->

<!-- unit:472871cbf009468fe3 -->
<a id="u-472871cbf009468fe3"></a>

### Как set prediction заменяет задачу удаления дубликатов

<!-- /unit:472871cbf009468fe3 -->

<!-- unit:402d292cd370ecaeae -->
<a id="u-402d292cd370ecaeae"></a>

В DETR fixed queries дают фиксированное число candidate slots. Training находит взаимно однозначное matching между GT и частью slots с учётом класса и геометрии. Оставшиеся обучаются отсутствию объекта. Если два slots предскажут один GT, только один сможет сопоставиться с ним; второй получает иной training signal, что помогает учить уникальный набор. Object queries — обучаемые запросы к признакам изображения, а не заранее известные метки «первый человек» и «вторая машина». [DETR](https://arxiv.org/abs/2005.12872).

<!-- /unit:402d292cd370ecaeae -->

<!-- unit:251b781ab8183d88aa -->
<a id="u-251b781ab8183d88aa"></a>

One-to-many assignment обеспечивает несколько положительных обучающих примеров на GT; one-to-one направлен на уникальный inference output. В dual-branch схемах они могут использовать shared features и согласованные criteria. Из этого не следует буквальный backward из output одной головы в output другой: градиентные связи определяет graph, включая detach. Фраза «one-to-one получает градиенты от one-to-many» без уточнения graph слишком сильна.

<!-- /unit:251b781ab8183d88aa -->

<!-- unit:3e691bb03d805c7a17 -->
<a id="u-3e691bb03d805c7a17"></a>

RT-DETR сохраняет set prediction и использует свой эффективный encoder/query-selection design. Скорость в имени не гарантирует более низкую latency любой конфигурации на любом устройстве: full pipeline и settings нужно измерить. Для YOLO26 режимы и поддерживаемые задачи следует проверять по [официальному описанию](https://docs.ultralytics.com/models/yolo26/); неподтверждённая запись с датой и «depth head» не служит достаточным подтверждением этих возможностей.

<!-- /unit:3e691bb03d805c7a17 -->

<!-- unit:11e2b5f03047824421 -->
<a id="u-11e2b5f03047824421"></a>

## Losses: локализация и класс

<!-- /unit:11e2b5f03047824421 -->

<!-- unit:c20fe7115e006ecd59 -->
<a id="u-c20fe7115e006ecd59"></a>

Box losses могут использовать L1 или IoU-family. Простой $1-\operatorname{IoU}$ плохо информирует о направлении при непересекающихся boxes. GIoU добавляет penalty за пустую часть минимального enclosing box:

<!-- /unit:c20fe7115e006ecd59 -->

<!-- unit:6ce6d479ce25409ba3 -->
<a id="u-6ce6d479ce25409ba3"></a>

$$
\operatorname{GIoU}=\operatorname{IoU}-\frac{|C|-|A\cup B|}{|C|}.
$$

<!-- /unit:6ce6d479ce25409ba3 -->

<!-- unit:1c142971e915f16e16 -->
<a id="u-1c142971e915f16e16"></a>

Для A/B выше C area9, union7, GIoU=$\frac17-\frac29\approx-0.079365$. Loss1-GIoU может быть больше1, это не автоматически ошибка.

<!-- /unit:1c142971e915f16e16 -->

<!-- unit:15a5e2614a5803a085 -->
<a id="u-15a5e2614a5803a085"></a>

DIoU добавляет squared distance centres, нормированную squared diagonal enclosing box; CIoU дополнительно учитывает aspect ratio по своей формуле. Они задают разные surrogate geometry, не гарантируя одинаковое улучшение AP. Distribution Focal Loss в некоторых heads обучает дискретизированные distance distributions; это не то же, что focal classification loss. Версии без DFL не надо описывать старым head из памяти.

<!-- /unit:15a5e2614a5803a085 -->

<!-- unit:f1e18bdd75a31ab272 -->
<a id="u-f1e18bdd75a31ab272"></a>

Для дисбаланса foreground/background classification focal loss имеет вид $-\alpha_t(1-p_t)^\gamma\log p_t$; easy examples получают меньший вес. Assignment, class loss и box loss связаны, но это три разных компонента.

<!-- /unit:f1e18bdd75a31ab272 -->

<!-- unit:d2a58c7a6be69fe95f -->
<a id="u-d2a58c7a6be69fe95f"></a>

### Почему IoU-loss не всегда подсказывает, куда двигать box

<!-- /unit:d2a58c7a6be69fe95f -->

<!-- unit:6e1ef5528f084e4ff5 -->
<a id="u-6e1ef5528f084e4ff5"></a>

Когда два boxes разделены зазором, IoU остаётся нулевой при небольшом перемещении, пока пересечение не появится. В этой области один $1-\operatorname{IoU}$ не различает некоторые полезные направления. GIoU учитывает размер enclosing rectangle, DIoU — расстояние центров, CIoU — дополнительный aspect-ratio term. Они добавляют разные сигналы, а не просто разные названия одного коэффициента.

<!-- /unit:6e1ef5528f084e4ff5 -->

<!-- unit:db31084ffde86049e9 -->
<a id="u-db31084ffde86049e9"></a>

Для $A=(0,0,1,1)$ и $B=(3,0,4,1)$ union=2, enclosing area=4, IoU=0 и GIoU=−0.5. Если B сдвинуть влево до (2,0,3,1), IoU всё ещё 0, но enclosing area=3 и GIoU=−1/3. Loss $1-\operatorname{GIoU}$ уменьшился с 1.5 до 4/3: полезное направление стало различимо ещё до overlap. Это иллюстрирует геометрическую мотивацию, не гарантируя глобально простую оптимизацию всех координат.

<!-- /unit:db31084ffde86049e9 -->

<!-- unit:8dd0483c10cbce6c7f -->
<a id="u-8dd0483c10cbce6c7f"></a>

Classification focal loss уменьшает вклад лёгких background examples через $(1-p_t)^\gamma$. Distribution Focal Loss решает другую задачу: распределение координатной дистанции может кодировать промежуточное значение через соседние bins. Например, target=2.3 можно разложить на веса 0.7 для bin 2 и 0.3 для bin 3; ожидаемое значение этих двух bins равно 2.3. Слово focal не делает эти objectives взаимозаменяемыми, и конкретная head может использовать иной способ регрессии.

<!-- /unit:8dd0483c10cbce6c7f -->

<!-- unit:4909dfe667d221214c -->
<a id="u-4909dfe667d221214c"></a>

## Проверка и ошибки

<!-- /unit:4909dfe667d221214c -->

<!-- unit:9e3894e5977df3f049 -->
<a id="u-9e3894e5977df3f049"></a>

Визуализировать предсказания после точного inverse resize/letterbox transform, иначе boxes выглядят неверными даже при корректных model outputs. Отдельно анализировать missed objects, duplicates, class confusion, localization errors и crowded/small cases. Full latency включает preprocessing, transfer, forward, postprocessing и queue.

<!-- /unit:9e3894e5977df3f049 -->

<!-- unit:50b0bb624ae30b3b0d -->
<a id="u-50b0bb624ae30b3b0d"></a>

1. Почему duplicate — FP даже при IoU1? **Ответ:** GT уже сопоставлен, один экземпляр нельзя засчитать дважды.
2. Почему AP50 и AP50:95 различаются? **Ответ:** набор localisation thresholds разный.
3. Что теряется при выключении NMS без смены training? **Ответ:** остаются duplicates, если output не обучен unique set prediction.

<!-- /unit:50b0bb624ae30b3b0d -->

<!-- unit:864f924547361b2bf5 -->
<a id="u-864f924547361b2bf5"></a>

**Практика:** пересчитать boxes и AP пример; затем поставить P2 выше P1 по score и проверить matching. Для реальной модели на одном split изменять один фактор: resolution, score operating threshold или NMS threshold. Сохранять все error slices и различать benchmark AP и application precision/recall.

<!-- /unit:864f924547361b2bf5 -->

<!-- unit:b629981d53cc6c0cfe -->
<a id="u-b629981d53cc6c0cfe"></a>

<!-- latex-display-v1 -->

<!-- /unit:b629981d53cc6c0cfe -->

<!-- unit:67de3b8e5f3a6ad884 -->
<a id="u-67de3b8e5f3a6ad884"></a>

<!-- source-depth-S7-B -->

<!-- /unit:67de3b8e5f3a6ad884 -->

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

Запросы читают признаки изображения через decoder. Hungarian matching назначает предсказания targets для loss; это не NMS по попарному IoU. Число queries и no-object convention входят в конкретную реализацию.

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

Ответьте на вопрос: **Как получить множество рамок и оценить его с учётом matching?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

Следующие темы: [Instance segmentation: Mask R-CNN и prototype masks](<cv-instance.md>), [Ориентированные рамки и периодичность угла](<cv-obb.md>), [Tracking: Kalman, association и идентичность](<cv-tracking.md>), [CLIP, открытый словарь и YOLO-World](<cv-open-vocabulary.md>).

[Маршрут модуля](<README.md>) · [Происхождение и точные места переноса](<../editorial/coverage.md>)
