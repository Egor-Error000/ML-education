# Б. Object detection: от boxes к AP и устройству детектора

[Оглавление](README.md). Пункт 2 плана.

**Основа:** S7, строки 1618–2467. Сохранён порядок: постановка → IoU/matching/AP → NMS → two-stage/one-stage → YOLO/DETR. **Исправления:** YOLOv1 не anchor-based как v2/v3; one-stage не всегда быстрее любой two-stage модели; наличие версии YOLO26 не означает обязательный NMS-free режим. **Углубление:** ручной AP с явной convention, assignment и box losses.

## 1. Выход — множество, а не одна метка

Для изображения output состоит из объектов `(box,class,score)`. Число объектов заранее неизвестно, порядок списка обычно не имеет семантики. Annotation задаёт box каждого экземпляра, а не один label всего изображения. Пропущенные в разметке реальные объекты могут при обучении/оценке ошибочно считаться background.

В `xyxy` box задаёт левый верхний и правый нижний углы $(x_1,y_1,x_2,y_2)$. В `xywh` надо уточнить, x/y — центр или corner. Нормированные и pixel coordinates нельзя смешивать. Для непрерывной геометрии площадь $(x_2-x_1)(y_2-y_1)$; исторические inclusive pixel conventions с $+1$ — другой protocol.

IoU=$\frac{|A\cap B|}{|A\cup B|}$. У A=(0,0,2,2), B=(1,1,3,3) площади 4 и4, пересечение1, union7, IoU=1/7≈0.142857. У непересекающихся boxes IoU0. Если boxes вырожденные, protocol должен определить обработку.

Class confidence не тождественна IoU: модель может уверенно назвать класс и плохо локализовать объект. В разных heads есть отдельный objectness или совместный score качества/класса. Нельзя универсально обещать ровно «objectness×class probability» для каждого YOLO checkpoint.

## 2. Matching: когда prediction становится TP

Для одной class и заданного IoU threshold τ predictions сортируют по score. Каждая prediction может сопоставиться с ещё не занятым GT того же class, если IoU удовлетворяет protocol. Первое подходящее совпадение TP, duplicate того же GT — FP. Непокрытые GT остаются FN. Оценка должна учитывать ignore/crowd правила выбранного benchmark, а не произвольную новую реализацию.

Пример: два GT, три predictions по убывающему score. P1 правильно покрыла GT1; P2 — duplicate GT1; P3 — GT2. Получаем TP flags $\left(1,\,0,\,1\right)$:

$$
\begin{array}{c|cc|cc}\text{после}&\mathrm{TP}&\mathrm{FP}&\mathrm{Precision}&\mathrm{Recall}\\\hline P_1&1&0&1&0.5\\P_2&1&1&0.5&0.5\\P_3&2&1&\frac23&1\end{array}
$$

AP агрегирует precision-recall curve, но **точное правило интегрирования** существенно. Для continuous interpolated envelope здесь первая половина recall имеет precision1, вторая2/3, AP=5/6≈0.8333. Это учебный вариант, не численный результат официального COCO evaluator с его сеткой recall/area/maxDets. Различные AP conventions не смешивать в отчёте.

mAP усредняет AP по выбранным классам и, если protocol требует, IoU thresholds. AP@0.5 проще, чем COCO-style AP@[0.5:0.95]: второй усредняет thresholds 0.50,0.55,…,0.95. Рост AP50 при неизменной высокой-threshold AP может означать, что объекты находятся, но box localization ещё недостаточно точна. Small/medium/large AP и per-class errors дают полезные срезы.

Порог score, выбранный для приложения, задаёт одну operating point. Для AP нужен корректный набор ranked detections по protocol; слишком раннее отбрасывание низких scores способно искусственно ограничить attainable recall.

## 3. Зачем NMS и где её границы

Greedy NMS для каждой class: выбрать максимальный score, сохранить box, подавить оставшиеся boxes с IoU выше threshold, повторить. Она удаляет дубликаты, но может убрать разные близкие/перекрывающиеся объекты. Class-agnostic NMS объединяет classes и имеет другие trade-offs. Soft-NMS уменьшает scores вместо жёсткого удаления.

Если A=(0,0,2,2), C=(0.2,0.2,2.2,2.2), intersection3.24, union4.76, IoU≈0.68067. При NMS threshold0.5 lower-score box будет подавлена в одной class. Правильно ли это, зависит от того, duplicate это или два сильно перекрытых объекта.

Нельзя ставить одинаковый IoU threshold для training assignment, NMS и evaluation только потому, что везде написано IoU. Они управляют разными решениями. NMS-free inference достигается специальной схемой обучения/выходов, а не просто выключением NMS у головы, обученной выдавать много duplicates.

## 4. Two-stage, one-stage и признаки разных масштабов

Исторический sliding window запускает classifier на множестве crops, что дорого и дублирует вычисления. R-CNN использует region proposals и обработку регионов. Fast R-CNN разделяет shared feature extraction и region-wise head; Faster R-CNN добавляет обучаемую RPN. В современных описаниях RoIAlign/другие изменения надо отличать от исходного RoIPool, не приписывая их всем версиям сразу.

One-stage detector предсказывает кандидатов на feature maps без отдельной классической proposal-classification стадии. SSD/YOLO — важные примеры, но «one-stage всегда быстрее, two-stage всегда точнее» не теорема: сравниваются реальные models, resolution и hardware.

FPN объединяет семантику глубоких низкоразрешённых maps с деталями более высокого resolution. Это важно для малых объектов: после сильного downsampling объект может занимать меньше одной эффективной клетки. Повышение input resolution увеличивает compute, но иногда исправляет bottleneck данных/геометрии лучше, чем увеличение model size.

YOLOv1 — grid-based прямое предсказание boxes без anchor механизма последующих v2/v3. Anchors — заранее заданные reference shapes, offsets относительно которых учатся. Anchor-free methods убирают именно эти шаблоны, но всё равно могут использовать feature-grid reference points. Decoupled head разделяет prediction branches, например class/box; это ещё одна независимая идея.

Training assignment отвечает, какие predictions получают targets какого GT. One-to-many даёт несколько positives на GT и обычно требует удаления duplicates. One-to-one способствует unique output, но нуждается в подходящем loss/assignment. В YOLOv10 и последующих designs могут сочетаться training branches с разными assignment. Для YOLO26 доступен отдельный one-to-one NMS-free inference path; конкретный default/path нужно задавать явно. [Официальное описание YOLO26](https://docs.ultralytics.com/models/yolo26/).

DETR формулирует set prediction с object queries и bipartite/Hungarian matching training outputs к GT. Дополнительные predictions обучают no-object class/эквиваленту protocol. Отсутствие обычной NMS связано с set objective, а не только со словом Transformer. RT-DETR использует связанные идеи с акцентом на speed; details выбранного variant проверяют отдельно.

## 5. Losses: локализация и класс

Box losses могут использовать L1 или IoU-family. Простой $1-\operatorname{IoU}$ плохо информирует о направлении при непересекающихся boxes. GIoU добавляет penalty за пустую часть минимального enclosing box:

$$
\operatorname{GIoU}=\operatorname{IoU}-\frac{|C|-|A\cup B|}{|C|}.
$$

Для A/B выше C area9, union7, GIoU=$\frac17-\frac29\approx-0.079365$. Loss1-GIoU может быть больше1, это не автоматически ошибка.

DIoU добавляет squared distance centres, нормированную squared diagonal enclosing box; CIoU дополнительно учитывает aspect ratio по своей формуле. Они задают разные surrogate geometry, не гарантируя одинаковое улучшение AP. Distribution Focal Loss в некоторых heads обучает дискретизированные distance distributions; это не то же, что focal classification loss. Версии без DFL не надо описывать старым head из памяти.

Для дисбаланса foreground/background classification focal loss имеет вид $-\alpha_t(1-p_t)^\gamma\log p_t$; easy examples получают меньший вес. Assignment, class loss и box loss связаны, но это три разных компонента.

## 6. Проверка и ошибки

Визуализировать предсказания после точного inverse resize/letterbox transform, иначе boxes выглядят неверными даже при корректных model outputs. Отдельно анализировать missed objects, duplicates, class confusion, localization errors и crowded/small cases. Full latency включает preprocessing, transfer, forward, postprocessing и queue.

1. Почему duplicate — FP даже при IoU1? **Ответ:** GT уже сопоставлен, один экземпляр нельзя засчитать дважды.
2. Почему AP50 и AP50:95 различаются? **Ответ:** набор localisation thresholds разный.
3. Что теряется при выключении NMS без смены training? **Ответ:** остаются duplicates, если output не обучен unique set prediction.

**Практика:** пересчитать boxes и AP пример; затем поставить P2 выше P1 по score и проверить matching. Для реальной модели на одном split изменять один фактор: resolution, score operating threshold или NMS threshold. Сохранять все error slices и различать benchmark AP и application precision/recall.

<!-- latex-display-v1 -->
