# Ж. Multi-object tracking: сохранить identity во времени

[Оглавление](README.md). Пункт 7 плана.

**Основа:** S7, строки 5277–6186. Сохранены detector→association, SORT/DeepSORT, ByteTrack, BoT-SORT и trajectory metrics. **Исправления:** MOTA учитывает IDS, IDF1 зависит и от detection errors; HOTA содержит квадратный корень и protocol aggregation; tracker не может гарантированно восстановить ID после любой occlusion. **Углубление:** ручной assignment, Kalman correction и разбор ID metrics.

## 1. Детекция кадра ещё не tracking

На каждом frame detector может отлично находить двух людей, но менять им IDs местами. Object detection quality останется высокой, а задачи подсчёта пересечений/траекторий сломаются. Track — история состояния и identity во времени, а detection — наблюдение одного кадра.

Tracking-by-detection система хранит active/lost/tentative tracks. На новом frame прогнозирует положение, сопоставляет detections с tracks, обновляет matched tracks, создаёт новые и завершает слишком давно потерянные. Эти decisions — алгоритм над video state, не обязательное дообучение detector.

Нужны согласованные timestamps/FPS, coordinates и camera motion handling. Если frames приходят нерегулярно, постоянный шаг движения без учёта delta_t ошибочен. Offline tracker может использовать будущие frames; online protocol — нет. Сравнивать их без указания доступа к будущему некорректно.

## 2. Kalman prediction/update

Линейная state-space иллюстрация:

$$
\begin{aligned}s_t&=Fs_{t-1}+\varepsilon_t,\qquad z_t=Hs_t+\nu_t,\\s^-&=Fs,\qquad P^-=FPF^\top+Q,\\K&=P^-H^\top(HP^-H^\top+R)^{-1},\\s^+&=s^-+K(z-Hs^-).\end{aligned}
$$

State может включать box centre/scale/velocity; details SORT/BoT variants различаются. P описывает uncertainty estimate, Q/R — process/measurement noise. Это модель и допущения, а не точная физика поведения человека.

Скалярный пример: predicted position — 10, variance — 4, measurement — 12 с noise variance 1. Тогда $K=\frac{4}{4+1}=0.8$, updated position — 11.6, variance — 0.8 по простой формуле $(1-K)P^-$. Наблюдение с меньшим noise получает больший вес. При occlusion measurement нет: prediction продолжает двигаться, uncertainty растёт, но подтверждённой identity это не добавляет.

## 3. Assignment: почему greedy nearest не всегда годится

Для predicted tracks i и detections j строится cost: например $1-\operatorname{IoU}$, Mahalanobis distance, appearance cosine distance или их смесь. Gating исключает физически/семантически неприемлемые пары до/после assignment по определённому protocol.

Матрица cost:

$$
C_{\mathrm{association}}=\begin{bmatrix}1&2\\2&100\end{bmatrix}\quad\text{(строки — tracks, столбцы — detections)}.
$$

Greedy берёт минимальную1 для track1→det1, оставляя100 track2→det2, total101. Global assignment track1→det2 и track2→det1 даёт4. Hungarian algorithm ищет minimum total для заданной matching постановки. Нужно также учесть возможность unmatched: нельзя принуждать любую detection к любому track только ради полного bijection.

Смешивание costs требует scales/weights. Если appearance distance [0,2], а raw coordinate distance сотни pixels, простая сумма без normalization почти игнорирует appearance. Нельзя называть такую схему «balanced motion+ReID» без проверки.

## 4. SORT, DeepSORT, ByteTrack и BoT-SORT

SORT сочетает Kalman dynamics и assignment по box geometry. Он прост и полезен как baseline; труднее различает похожие trajectories при occlusion/crossing. [SORT](https://arxiv.org/abs/1602.00763).

DeepSORT добавляет appearance representation/ReID distance и соответствующий matching pipeline. Это помогает при кратком исчезновении и похожей motion geometry, но одинаковая одежда, domain shift и poor crops могут давать ложную уверенность. ReID model тоже нуждается в подходящей training distribution.

ByteTrack сначала использует high-score detections, затем связывает оставшиеся tracks с подходящими low-score detections. Идея: occluded object может иметь низкую confidence, но coherent motion. Это не «принять все слабые detections как новые tracks»; false low-score background отсекается association policy. [ByteTrack](https://arxiv.org/abs/2110.06864).

BoT-SORT сочетает motion/association changes с camera motion compensation и, в некоторых configs, appearance. Нельзя считать ReID обязательным у любого выбранного tracker config. В интеграции с YOLO детектор и tracker остаются разными компонентами; model version не определяет автоматически matching thresholds/timeouts.

## 5. MOTA, IDF1 и HOTA

MOTA=$1-\frac{\mathrm{FN}+\mathrm{FP}+\mathrm{IDS}}{GT_{\mathrm{total}}}$, где counts агрегируются по protocol. Если GT100,FN10,FP5,IDS2, MOTA0.83. При суммарных ошибках больше GT она отрицательна. IDS входит явно, но common detection errors могут доминировать. Фраза «MOTA вообще не учитывает identity» неверна; точнее сказать, что её баланс не описывает устойчивость identities полностью.

IDF1=$\frac{2\mathrm{IDTP}}{2\mathrm{IDTP}+\mathrm{IDFP}+\mathrm{IDFN}}$. Identity counts требуют global identity matching protocol, а не простого подсчёта правильного числового ID на frame: predicted ID1 может соответствовать GT ID50. Пропущенные detections увеличивают IDFN, лишние — IDFP, поэтому IDF1 не независим от detection quality. Для IDTP80,IDFP10,IDFN20 score160/190≈0.8421.

HOTA на заданном localisation threshold балансирует detection и association: $\mathrm{HOTA}_\alpha=\sqrt{\mathrm{DetA}_\alpha\mathrm{AssA}_\alpha}$, далее агрегируется по thresholds согласно определению. AssA — ассоциационное согласие по matched detections, а не просто1-IDS/N. Нельзя сначала независимо усреднить DetA/AssA и объявить sqrt их произведения exact overall HOTA. [HOTA](https://arxiv.org/abs/2009.07736).

При DetA0.81,AssA0.49 на одном threshold HOTA0.63. В original export корень терялся при записи/извлечении; произведение0.3969 — другой показатель. Дополнительные IDS/fragmentations/coverage дают важную diagnosis; definition каждого берётся из benchmark evaluator.

## 6. Occlusion, split и end-to-end evaluation

Long occlusion делает identity неоднозначной, особенно если одинаковые объекты пересекаются вне кадра. Полезно сохранять lost tracks до timeout, но слишком долгий timeout повышает false reattachment. «Всегда вернуть старый ID» — неверная цель без достаточного evidence.

Split по videos/scenes, а не random frames: соседние frames почти дубликаты. Качество зависит от detector confidence threshold, frame rate, video resize и motion compensation; фиксировать всё. Latency включает detector, ReID crops, association и I/O, не только лёгкий Kalman update.

1. Почему нельзя сравнивать IDs как category numbers? **Ответ:** labels tracks произвольны, важна consistent correspondence, а не равенство чисел.
2. Что делает второй этап ByteTrack? **Ответ:** использует часть low-score observations для продолжения существующих tracks по association.
3. Помогает ли tracker при missed detection? **Ответ:** может прогнозировать/recover, но uncertainty и неоднозначность остаются.

**Практика:** решить assignment example, затем запретить пару track2→det1 gating и определить unmatched policy. Разметить пять кадров двух пересекающихся объектов, специально swap IDs и посмотреть отдельно detection и trajectory errors. Для реального baseline сравнить SORT и один appearance/low-score вариант с тем же detector.

<!-- latex-display-v1 -->
