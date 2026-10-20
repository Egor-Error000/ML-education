# Tracking: Kalman, association и идентичность

**Уровень:** основа ветки.

**Предпосылки:** [Изображение, карты признаков и блоки модели](<cv-input.md>), [Классификация, CNN и residual learning](<cv-classification.md>), [Detection: множества объектов, IoU, AP и архитектуры](<cv-detection.md>)

**Используют:** Выбранная прикладная задача и итоговая проверка результата.

**Учебный вопрос:** Как сохранить идентичность объекта между кадрами?





## Карта вычисления и границы этапов

![Tracking: Kalman, association и идентичность: входы, преобразования и выходы](<../оформление/course/cv-tracking.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Сравните формы и смысл каждого перехода с формулами главы. Выход преобразования и доказательство его качества — разные объекты: последняя проверка использует отдельный протокол оценки.

<!-- unit:8db4be54d4f5f95fe8 -->
<a id="u-8db4be54d4f5f95fe8"></a>

## Детекция кадра ещё не tracking

<!-- /unit:8db4be54d4f5f95fe8 -->

<!-- unit:8b9124643042091a9f -->
<a id="u-8b9124643042091a9f"></a>

На каждом frame detector может отлично находить двух людей, но менять им IDs местами. Object detection quality останется высокой, а задачи подсчёта пересечений/траекторий сломаются. Track — история состояния и identity во времени, а detection — наблюдение одного кадра.

<!-- /unit:8b9124643042091a9f -->

<!-- unit:4ec78f7dc5243fc9cf -->
<a id="u-4ec78f7dc5243fc9cf"></a>

Tracking-by-detection система хранит active/lost/tentative tracks. На новом frame прогнозирует положение, сопоставляет detections с tracks, обновляет matched tracks, создаёт новые и завершает слишком давно потерянные. Эти decisions — алгоритм над video state, не обязательное дообучение detector.

<!-- /unit:4ec78f7dc5243fc9cf -->

<!-- unit:4c9a5090baacfd5b2f -->
<a id="u-4c9a5090baacfd5b2f"></a>

Нужны согласованные timestamps/FPS, coordinates и camera motion handling. Если frames приходят нерегулярно, постоянный шаг движения без учёта delta_t ошибочен. Offline tracker может использовать будущие frames; online protocol — нет. Сравнивать их без указания доступа к будущему некорректно.

<!-- /unit:4c9a5090baacfd5b2f -->

<!-- unit:912664f07a42de23cb -->
<a id="u-912664f07a42de23cb"></a>

### Что хранится между кадрами и почему ID нельзя получить из class

<!-- /unit:912664f07a42de23cb -->

<!-- unit:e65e317d4688fde5b7 -->
<a id="u-e65e317d4688fde5b7"></a>

Для человека class может всё время оставаться `person`, но конкретных людей несколько. Track ID — условная постоянная метка одной истории. Если detector на каждом кадре сортирует людей по score и назначает IDs по номерам строк, identities будут менятьcя при изменении confidence, даже если boxes остаются правильными. Поэтому tracker сопоставляет наблюдения с прежними состояниями, а не просто дописывает номер к каждой detection.

<!-- /unit:e65e317d4688fde5b7 -->

<!-- unit:b32f3a343c9a9d2a13 -->
<a id="u-b32f3a343c9a9d2a13"></a>

Track management содержит отдельные решения. Новая detection может породить tentative track, который ещё не показывается как подтверждённый. Повторные согласованные наблюдения подтверждают его. При пропуске track становится lost и живёт ограниченное время; после timeout удаляется. В разных реализациях названия и переходы различаются. Predicted box lost-трека — гипотеза движения, а не новое наблюдение объекта.

<!-- /unit:b32f3a343c9a9d2a13 -->

<!-- unit:33083cb3dcb5eed4f8 -->
<a id="u-33083cb3dcb5eed4f8"></a>

В MOTChallenge-like result строка содержит frame,ID,box,score и предусмотренные дополнительные поля. Ground-truth формат может включать class/visibility/ignore semantics, поэтому source-объяснение «все оставшиеся поля всегда заглушки, confidence GT всегда 1» нельзя переносить без проверки конкретного файла. Сдвиг frame numbering на единицу испортит matching при визуально похожем видео. Координаты xywh с corner также надо отличать от center-формы фильтра.

<!-- /unit:33083cb3dcb5eed4f8 -->

<!-- unit:f815c7fa66995264d0 -->
<a id="u-f815c7fa66995264d0"></a>

Online tracker использует только доступное прошлое; near-online добавляет буфер и latency, offline может переопределять связи после просмотра будущих кадров. «Online» не гарантирует обработку в real time: тяжёлый ReID или большой input может быть медленнее FPS камеры. Наоборот, offline batched computation может быть быстрым по throughput, но всё равно не подходит для мгновенной реакции.

<!-- /unit:f815c7fa66995264d0 -->

<!-- unit:3dce949de732f6cf0a -->
<a id="u-3dce949de732f6cf0a"></a>

## Kalman prediction/update

<!-- /unit:3dce949de732f6cf0a -->

<!-- unit:0ba97fe3bf3f3a9d95 -->
<a id="u-0ba97fe3bf3f3a9d95"></a>

Линейная state-space иллюстрация:

<!-- /unit:0ba97fe3bf3f3a9d95 -->

<!-- unit:f489b890ce60f4a136 -->
<a id="u-f489b890ce60f4a136"></a>

$$
\begin{aligned}s_t&=Fs_{t-1}+\varepsilon_t,\qquad z_t=Hs_t+\nu_t,\\s^-&=Fs,\qquad P^-=FPF^\top+Q,\\K&=P^-H^\top(HP^-H^\top+R)^{-1},\\s^+&=s^-+K(z-Hs^-).\end{aligned}
$$

<!-- /unit:f489b890ce60f4a136 -->

<!-- unit:c24af696d4a6255036 -->
<a id="u-c24af696d4a6255036"></a>

State может включать box centre/scale/velocity; details SORT/BoT variants различаются. P описывает uncertainty estimate, Q/R — process/measurement noise. Это модель и допущения, а не точная физика поведения человека.

<!-- /unit:c24af696d4a6255036 -->

<!-- unit:ca67ff426ff9c160d4 -->
<a id="u-ca67ff426ff9c160d4"></a>

Скалярный пример: predicted position — 10, variance — 4, measurement — 12 с noise variance 1. Тогда $K=\frac{4}{4+1}=0.8$, updated position — 11.6, variance — 0.8 по простой формуле $(1-K)P^-$. Наблюдение с меньшим noise получает больший вес. При occlusion measurement нет: prediction продолжает двигаться, uncertainty растёт, но подтверждённой identity это не добавляет.

<!-- /unit:ca67ff426ff9c160d4 -->

<!-- unit:36df4a5767e1748957 -->
<a id="u-36df4a5767e1748957"></a>

### От состояния SORT к координатам box

<!-- /unit:36df4a5767e1748957 -->

<!-- unit:d1ab07bfb38424ffe4 -->
<a id="u-d1ab07bfb38424ffe4"></a>

В учебном SORT-примере state имеет семь координат $(u,v,a,r,\dot u,\dot v,\dot a)$: центр, площадь, aspect ratio и три скорости. Здесь a обозначает площадь, чтобы не смешивать её с полным state s. При r=w/h получаем $w=\sqrt{ar}$, $h=\sqrt{a/r}$ для положительных a,r. Скорости площади и линейных размеров — разные модели; DeepSORT/BoT variants не обязаны хранить точно те же семь величин.

<!-- /unit:d1ab07bfb38424ffe4 -->

<!-- unit:79236443e019db00e3 -->
<a id="u-79236443e019db00e3"></a>

Для позиции и скорости одна constant-velocity часть transition имеет вид

<!-- /unit:79236443e019db00e3 -->

<!-- unit:0d7bd9238e52b69a80 -->
<a id="u-0d7bd9238e52b69a80"></a>

$$
\begin{bmatrix}u_t^-\\\dot u_t^-\end{bmatrix}=
\begin{bmatrix}1&\Delta t\\0&1\end{bmatrix}
\begin{bmatrix}u_{t-1}^+\\\dot u_{t-1}^+\end{bmatrix}.
$$

<!-- /unit:0d7bd9238e52b69a80 -->

<!-- unit:ad9897ed2d3aa6d804 -->
<a id="u-ad9897ed2d3aa6d804"></a>

Если u=10,velocity=2 pixels/frame и прошло три frame intervals, forecast u=16. Использовать u=12 после пропуска двух кадров — значит ошибиться в time step. При измерениях в секундах velocity и Q тоже должны иметь согласованные единицы.

<!-- /unit:ad9897ed2d3aa6d804 -->

<!-- unit:322b80aa1067e278bb -->
<a id="u-322b80aa1067e278bb"></a>

Innovation $z-Hs^-$ — разность измерения и предсказания. Kalman gain зависит от их uncertainty. В скалярном примере главы K=0.8, поэтому к forecast10 добавляется 0.8·(12−10)=1.6. Если measurement variance поднять с 1 до 16 при прежней P−=4, K=0.2 и update даст 10.4. Фильтр меньше доверяет шумному наблюдению; один большой score detector не заменяет автоматически правильно выбранную R.

<!-- /unit:322b80aa1067e278bb -->

<!-- unit:f544448cb7731ac32a -->
<a id="u-f544448cb7731ac32a"></a>

После update covariance в простой записи $P^+=(I-KH)P^-$; при аккуратной численной реализации могут использовать Joseph form. Несколько прогнозов без measurements добавляют process uncertainty. Это допускает более широкую область будущих matching, но расширение повышает риск связать другого человека. Неопределённость — часть решения, а не обещание помнить identity бесконечно.

<!-- /unit:f544448cb7731ac32a -->

<!-- unit:2948c364edfd33ab9c -->
<a id="u-2948c364edfd33ab9c"></a>

## Assignment: почему greedy nearest не всегда годится

<!-- /unit:2948c364edfd33ab9c -->

<!-- unit:eb6fbb356f6dfa04e4 -->
<a id="u-eb6fbb356f6dfa04e4"></a>

Для predicted tracks i и detections j строится cost: например $1-\operatorname{IoU}$, Mahalanobis distance, appearance cosine distance или их смесь. Gating исключает физически/семантически неприемлемые пары до/после assignment по определённому protocol.

<!-- /unit:eb6fbb356f6dfa04e4 -->

<!-- unit:c03264d06bde10cc85 -->
<a id="u-c03264d06bde10cc85"></a>

Матрица cost:

<!-- /unit:c03264d06bde10cc85 -->

<!-- unit:91d7e3dc149a6e01ac -->
<a id="u-91d7e3dc149a6e01ac"></a>

$$
C_{\mathrm{association}}=\begin{bmatrix}1&2\\2&100\end{bmatrix}\quad\text{(строки — tracks, столбцы — detections)}.
$$

<!-- /unit:91d7e3dc149a6e01ac -->

<!-- unit:d574d340f8c70286d7 -->
<a id="u-d574d340f8c70286d7"></a>

Greedy берёт минимальную1 для track1→det1, оставляя100 track2→det2, total101. Global assignment track1→det2 и track2→det1 даёт4. Hungarian algorithm ищет minimum total для заданной matching постановки. Нужно также учесть возможность unmatched: нельзя принуждать любую detection к любому track только ради полного bijection.

<!-- /unit:d574d340f8c70286d7 -->

<!-- unit:429aa123014d30f892 -->
<a id="u-429aa123014d30f892"></a>

Смешивание costs требует scales/weights. Если appearance distance [0,2], а raw coordinate distance сотни pixels, простая сумма без normalization почти игнорирует appearance. Нельзя называть такую схему «balanced motion+ReID» без проверки.

<!-- /unit:429aa123014d30f892 -->

<!-- unit:7386234923eba3bc81 -->
<a id="u-7386234923eba3bc81"></a>

### Motion gating, appearance и возможность оставить без пары

<!-- /unit:7386234923eba3bc81 -->

<!-- unit:caa5883ad8aec8ebd7 -->
<a id="u-caa5883ad8aec8ebd7"></a>

Для residual r=z−Hs− covariance наблюдения $S=HP^-H^\top+R$ squared Mahalanobis distance равна $r^\top S^{-1}r$. Сдвиг на 10 pixels может быть неправдоподобным при дисперсии 1 и допустимым при большой uncertainty. Euclidean distance не учитывает эту разницу. Gating запрещает совсем неправдоподобные связи, а assignment выбирает среди допустимых.

<!-- /unit:caa5883ad8aec8ebd7 -->

<!-- unit:0dd66e64af45e453b2 -->
<a id="u-0dd66e64af45e453b2"></a>

Пусть два похожих человека пересеклись. Геометрический cost может предпочесть nearest box, а appearance — другую пару. Для нормированных embeddings e1,e2 cosine distance=1−e1ᵀe2. Нужны сопоставимые crops и trained representation; одинаковая одежда может сделать обе appearance distances близкими. Простая weighted sum не превращает отсутствие информации в надёжное решение.

<!-- /unit:0dd66e64af45e453b2 -->

<!-- unit:a0da3355a062265418 -->
<a id="u-a0da3355a062265418"></a>

В матрице главы greedy total101 и global total4 показывают отличие оптимизации суммы от локального выбора. Но если все доступные пары cost100, правильнее оставить unmatched, чем гарантированно передать чужой ID. Для этого постановка содержит thresholds/dummy options/последующее отбрасывание согласно реализации. «Hungarian нашёл минимум» доказывает оптимум заданного cost, а не истинность identities.

<!-- /unit:a0da3355a062265418 -->

<!-- unit:ad811f37fc88d2aaba -->
<a id="u-ad811f37fc88d2aaba"></a>

## SORT, DeepSORT, ByteTrack и BoT-SORT

<!-- /unit:ad811f37fc88d2aaba -->

<!-- unit:c1a9c4c044f83cb78f -->
<a id="u-c1a9c4c044f83cb78f"></a>

SORT сочетает Kalman dynamics и assignment по box geometry. Он прост и полезен как baseline; труднее различает похожие trajectories при occlusion/crossing. [SORT](https://arxiv.org/abs/1602.00763).

<!-- /unit:c1a9c4c044f83cb78f -->

<!-- unit:98b56e7749a6d8d479 -->
<a id="u-98b56e7749a6d8d479"></a>

DeepSORT добавляет appearance representation/ReID distance и соответствующий matching pipeline. Это помогает при кратком исчезновении и похожей motion geometry, но одинаковая одежда, domain shift и poor crops могут давать ложную уверенность. ReID model тоже нуждается в подходящей training distribution.

<!-- /unit:98b56e7749a6d8d479 -->

<!-- unit:c798cfbe1888ea6e41 -->
<a id="u-c798cfbe1888ea6e41"></a>

ByteTrack сначала использует high-score detections, затем связывает оставшиеся tracks с подходящими low-score detections. Идея: occluded object может иметь низкую confidence, но coherent motion. Это не «принять все слабые detections как новые tracks»; false low-score background отсекается association policy. [ByteTrack](https://arxiv.org/abs/2110.06864).

<!-- /unit:c798cfbe1888ea6e41 -->

<!-- unit:e08454d4dd37c2df9a -->
<a id="u-e08454d4dd37c2df9a"></a>

BoT-SORT сочетает motion/association changes с camera motion compensation и, в некоторых configs, appearance. Нельзя считать ReID обязательным у любого выбранного tracker config. В интеграции с YOLO детектор и tracker остаются разными компонентами; model version не определяет автоматически matching thresholds/timeouts.

<!-- /unit:e08454d4dd37c2df9a -->

<!-- unit:7aeb41bc9783ac1558 -->
<a id="u-7aeb41bc9783ac1558"></a>

### DeepSORT cascade: исправляем порядок из разбираемого примера

<!-- /unit:7aeb41bc9783ac1558 -->

<!-- unit:81a72720a5b835a333 -->
<a id="u-81a72720a5b835a333"></a>

Ошибочная формулировка cascade: «сначала высокий confidence, затем низкий». В original DeepSORT matching cascade приоритет связан с временем после последнего update: недавно наблюдавшиеся tracks обрабатываются раньше давно потерянных. Это уменьшает конкуренцию со старыми неопределёнными hypotheses. Отбор detector scores и cascade recency — разные механизмы. Проверить порядок можно в [исходном matching cascade](https://github.com/nwojke/deep_sort/blob/master/deep_sort/linear_assignment.py).

<!-- /unit:81a72720a5b835a333 -->

<!-- unit:667ffa3027566ab869 -->
<a id="u-667ffa3027566ab869"></a>

Appearance gallery хранит несколько embeddings истории track, чтобы matching учитывал изменение ракурса. Слишком старая или загрязнённая gallery может закрепить ошибку после неверного присоединения. ReID полезен для продолжения identities, но не делает long re-entry всегда разрешимым. Решение о новом ID иногда честнее, чем уверенное присоединение к старому без достаточного signal.

<!-- /unit:667ffa3027566ab869 -->

<!-- unit:9c96a2555b12bc425a -->
<a id="u-9c96a2555b12bc425a"></a>

### ByteTrack на трёх последовательных кадрах

<!-- /unit:9c96a2555b12bc425a -->

<!-- unit:a86667a14bf9fac047 -->
<a id="u-a86667a14bf9fac047"></a>

В кадре t человек имеет detection score0.9 и подтверждённый track. В t+1 он частично закрыт: score0.25, но box близок forecast. Если входной confidence filter заранее удаляет всё ниже0.5, association не получит это наблюдение. Если сохранить low-score pool до0.1, второй этап ByteTrack может связать score0.25 с unmatched track. В t+2 score снова0.85, и identity продолжится без разрыва.

<!-- /unit:a86667a14bf9fac047 -->

<!-- unit:b9b643d23d214265fc -->
<a id="u-b9b643d23d214265fc"></a>

Однако слабая detection далёкого background не должна автоматически породить новый track. Поэтому high/low pool, first matching, second matching и new-track threshold — отдельные decisions. Числа 0.5/0.1/0.6 из разбираемого примера иллюстрируют разные пороги, а не универсальные настройки. Изменить внешний detector conf на0.3 при low threshold0.1 может сделать часть low pool недоступной.

<!-- /unit:b9b643d23d214265fc -->

<!-- unit:0dafb75ee4d7df3fe8 -->
<a id="u-0dafb75ee4d7df3fe8"></a>

### Camera motion и совместные модели

<!-- /unit:0dafb75ee4d7df3fe8 -->

<!-- unit:6401c071b016cb587f -->
<a id="u-6401c071b016cb587f"></a>

При camera pan статичный объект сдвигается в image coordinates. Модель «всё движение связано с собственным velocity объекта» начинает ошибаться для всех tracks одновременно. Global motion compensation оценивает background transformation и согласованно преобразует track geometry. Она может использовать affine motion, optical flow или feature matching; не каждая конфигурация обязана оценивать полную homography. Moving foreground и parallax нарушают единый global transform. [BoT-SORT](https://arxiv.org/abs/2206.14651).

<!-- /unit:6401c071b016cb587f -->

<!-- unit:d6c5a198ddd4ca9d2b -->
<a id="u-d6c5a198ddd4ca9d2b"></a>

JDE/FairMOT извлекают detector outputs и ReID features совместно, экономя повторные visual passes; association и lifecycle при этом всё равно существуют. Tracktor использует box regression detector для продолжения регионов, но требует решений о новых tracks и завершении. MOTR/TrackFormer переносят identity через track queries: отсутствие отдельного Hungarian association на inference не означает отсутствие задачи временного соответствия. Поэтому «MOT всегда ровно две независимые программы» — полезное описание tracking-by-detection, но не полная классификация всех architectures.

<!-- /unit:d6c5a198ddd4ca9d2b -->

<!-- unit:38558a162e23a29a81 -->
<a id="u-38558a162e23a29a81"></a>

## MOTA, IDF1 и HOTA

<!-- /unit:38558a162e23a29a81 -->

<!-- unit:f02805df9adfa35061 -->
<a id="u-f02805df9adfa35061"></a>

MOTA=$1-\frac{\mathrm{FN}+\mathrm{FP}+\mathrm{IDS}}{GT_{\mathrm{total}}}$, где counts агрегируются по protocol. Если GT100,FN10,FP5,IDS2, MOTA0.83. При суммарных ошибках больше GT она отрицательна. IDS входит явно, но common detection errors могут доминировать. Фраза «MOTA вообще не учитывает identity» неверна; точнее сказать, что её баланс не описывает устойчивость identities полностью.

<!-- /unit:f02805df9adfa35061 -->

<!-- unit:5d60146864e74631bd -->
<a id="u-5d60146864e74631bd"></a>

IDF1=$\frac{2\mathrm{IDTP}}{2\mathrm{IDTP}+\mathrm{IDFP}+\mathrm{IDFN}}$. Identity counts требуют global identity matching protocol, а не простого подсчёта правильного числового ID на frame: predicted ID1 может соответствовать GT ID50. Пропущенные detections увеличивают IDFN, лишние — IDFP, поэтому IDF1 не независим от detection quality. Для IDTP80,IDFP10,IDFN20 score160/190≈0.8421.

<!-- /unit:5d60146864e74631bd -->

<!-- unit:0be636a5edd1890587 -->
<a id="u-0be636a5edd1890587"></a>

HOTA на заданном localisation threshold балансирует detection и association: $\mathrm{HOTA}_\alpha=\sqrt{\mathrm{DetA}_\alpha\mathrm{AssA}_\alpha}$, далее агрегируется по thresholds согласно определению. AssA — ассоциационное согласие по matched detections, а не просто1-IDS/N. Нельзя сначала независимо усреднить DetA/AssA и объявить sqrt их произведения exact overall HOTA. [HOTA](https://arxiv.org/abs/2009.07736).

<!-- /unit:0be636a5edd1890587 -->

<!-- unit:60935da68c4bbb3d45 -->
<a id="u-60935da68c4bbb3d45"></a>

При DetA0.81,AssA0.49 на одном threshold HOTA0.63. В original export корень терялся при записи/извлечении; произведение0.3969 — другой показатель. Дополнительные IDS/fragmentations/coverage дают важную diagnosis; definition каждого берётся из benchmark evaluator.

<!-- /unit:60935da68c4bbb3d45 -->

<!-- unit:09a8d8eed1a7e0adb7 -->
<a id="u-09a8d8eed1a7e0adb7"></a>

### Два разных случая, скрытых одинаковым MOTA

<!-- /unit:09a8d8eed1a7e0adb7 -->

<!-- unit:c0b7c9a8bdca8998ba -->
<a id="u-c0b7c9a8bdca8998ba"></a>

Есть один GT человек на десяти кадрах, boxes всегда верные. Предсказанный ID7 действует первые пять, ID8 — вторые пять. Если protocol засчитает одну смену, MOTA=1−1/10=0.9. При global one-to-one identity matching только одна из двух predicted trajectories может соответствовать полной GT identity: IDTP=5,IDFP=5,IDFN=5, IDF1=10/20=0.5. Так видно, почему один краткий IDS penalty не описывает всей фрагментации истории.

<!-- /unit:c0b7c9a8bdca8998ba -->

<!-- unit:bdd2b6f20bf229c66a -->
<a id="u-bdd2b6f20bf229c66a"></a>

Если вместо split человек найден с одним ID лишь на первых пяти frames, IDTP=5,IDFP=0,IDFN=5 и IDF1=10/15≈0.666667. Пропуски изменили identity score, хотя ни одного ID-switch нет. Это прямое опровержение фразы «IDF1 не учитывает detection».

<!-- /unit:bdd2b6f20bf229c66a -->

<!-- unit:65cd18c454d579242f -->
<a id="u-65cd18c454d579242f"></a>

MOTA может стать отрицательной: при GT=10,FN=8,FP=10,IDS=3 результат −1.1. Это допустимый показатель по формуле. Для HOTA при одном threshold нужно sqrt(DetA·AssA); если thresholds дают разные пары, усреднение корней не равно корню из отдельно усреднённых факторов. Например, пары (1,0.25) и (0.25,1) дают HOTA по0.5, среднее0.5, тогда как sqrt(0.625²)=0.625 — другое число.

<!-- /unit:65cd18c454d579242f -->

<!-- unit:22c2e6e36b40ba0434 -->
<a id="u-22c2e6e36b40ba0434"></a>

MT/ML характеризуют покрытие GT trajectories, fragmentations — разрывы, IDS — смены по определённому protocol. Их нельзя выводить из одного IDF1. MOTP может быть определён как distance или overlap в конкретной реализации, поэтому направление «больше лучше» надо сверять. Для 3D MOT появляются другие geometry, sensors и protocols, например AMOTA; 2D IoU thresholds туда не переносятся механически.

<!-- /unit:22c2e6e36b40ba0434 -->

<!-- unit:5831075ea2f654aff7 -->
<a id="u-5831075ea2f654aff7"></a>

### Практическая интеграция без скрытого изменения задачи

<!-- /unit:5831075ea2f654aff7 -->

<!-- unit:06cf085854c87dfc21 -->
<a id="u-06cf085854c87dfc21"></a>

У YOLO+tracker detector выдаёт boxes, а tracker получает отсортированный во времени поток и persistent state. Случайно сбрасывать state каждый frame — значит снова создавать identities. Но переносить state между двумя независимыми видео тоже ошибка: coordinates и identity history уже относятся к другой сцене. Frame drops, resize и timestamp должны входить в записанную конфигурацию.

<!-- /unit:06cf085854c87dfc21 -->

<!-- unit:9bc813314d19ac4ffe -->
<a id="u-9bc813314d19ac4ffe"></a>

`track_buffer`, `max_age`, `min_hits` в разных trackers не обязаны иметь одинаковые единицы/условия. Buffer30 при30FPS примерно одна секунда только при таком определении кадрового шага. При5FPS это уже шесть секунд. В реальном протоколе учитывают способ normalization по frame rate. [Документация tracking](https://docs.ultralytics.com/modes/track/).

<!-- /unit:9bc813314d19ac4ffe -->

<!-- unit:8a7e5181725937b87e -->
<a id="u-8a7e5181725937b87e"></a>

Числа FPS из рассматриваемого примера нельзя сравнивать как готовый рейтинг: одни относятся к лёгкому association, другие включают detector/ReID, input sizes и hardware различны. Сравнение SORT/ByteTrack/appearance-конфигурации сохраняет один detector и input stream, отдельно фиксирует detection quality, IDs/IDF1/HOTA, full latency и поведение на crossings/occlusions. Это позволяет понять, какой компонент реально исправил ошибку.

<!-- /unit:8a7e5181725937b87e -->

<!-- unit:48ea9949e9eeb6a380 -->
<a id="u-48ea9949e9eeb6a380"></a>

## Occlusion, split и end-to-end evaluation

<!-- /unit:48ea9949e9eeb6a380 -->

<!-- unit:b983a2de7b69da227b -->
<a id="u-b983a2de7b69da227b"></a>

Long occlusion делает identity неоднозначной, особенно если одинаковые объекты пересекаются вне кадра. Полезно сохранять lost tracks до timeout, но слишком долгий timeout повышает false reattachment. «Всегда вернуть старый ID» — неверная цель без достаточного evidence.

<!-- /unit:b983a2de7b69da227b -->

<!-- unit:e35924db7089eb52da -->
<a id="u-e35924db7089eb52da"></a>

Split по videos/scenes, а не random frames: соседние frames почти дубликаты. Качество зависит от detector confidence threshold, frame rate, video resize и motion compensation; фиксировать всё. Latency включает detector, ReID crops, association и I/O, не только лёгкий Kalman update.

<!-- /unit:e35924db7089eb52da -->

<!-- unit:4fa3de4903afbc3ecf -->
<a id="u-4fa3de4903afbc3ecf"></a>

### SAM 3.1: трекинг понятий и цена числа объектов

<!-- /unit:4fa3de4903afbc3ecf -->

<!-- unit:38abb18eae0c2d25e4 -->
<a id="u-38abb18eae0c2d25e4"></a>

SAM 3.1 сохраняет promptable concept segmentation интерфейс SAM 3 и меняет обработку нескольких video objects: object multiplexing допускает до 16 tracked objects в одном forward pass. В релизном сообщении Meta приводит рост 16→32 FPS на одной H100 для video с medium number of objects. Это пример ускорения конкретной реализации, а не гарантированная скорость вашего tracker.

<!-- /unit:38abb18eae0c2d25e4 -->

<!-- unit:e63f35573196430f77 -->
<a id="u-e63f35573196430f77"></a>

Для собственного сравнения фиксируйте число объектов, resolution, prompts, video length, batching, precision, hardware и измеряемую границу latency. Считайте одновременно throughput, per-frame latency, memory, пропуски/дубли identity и mask quality. Разделяйте object detection/segmentation score и association quality: быстрее обрабатывать кадры не означает автоматически меньше ID switches.

<!-- /unit:e63f35573196430f77 -->

<!-- unit:1fe7fd970a4431accb -->
<a id="u-1fe7fd970a4431accb"></a>

**Источник и границы:** [официальное сообщение Meta о SAM 3.1](https://ai.meta.com/blog/segment-anything-model-3/), 27 марта 2026. Число FPS и предел 16 объектов приведены самой Meta для названной аппаратуры и тестовой нагрузки; повторное измерение на целевой системе обязательно.

<!-- /unit:1fe7fd970a4431accb -->

<!-- unit:94c96e03fccd1d4c13 -->
<a id="u-94c96e03fccd1d4c13"></a>

1. Почему нельзя сравнивать IDs как category numbers? **Ответ:** labels tracks произвольны, важна consistent correspondence, а не равенство чисел.
2. Что делает второй этап ByteTrack? **Ответ:** использует часть low-score observations для продолжения существующих tracks по association.
3. Помогает ли tracker при missed detection? **Ответ:** может прогнозировать/recover, но uncertainty и неоднозначность остаются.

<!-- /unit:94c96e03fccd1d4c13 -->

<!-- unit:d0681feccf471abda7 -->
<a id="u-d0681feccf471abda7"></a>

**Практика:** решить assignment example, затем запретить пару track2→det1 gating и определить unmatched policy. Разметить пять кадров двух пересекающихся объектов, специально swap IDs и посмотреть отдельно detection и trajectory errors. Для реального baseline сравнить SORT и один appearance/low-score вариант с тем же detector.

<!-- /unit:d0681feccf471abda7 -->

<!-- unit:05665721d141cde268 -->
<a id="u-05665721d141cde268"></a>

**Схема курса:** ordered frames + detections → Kalman state prediction → gated assignment/ReID → track update and identity state → trajectory metrics over complete sequence

<!-- /unit:05665721d141cde268 -->

<!-- unit:710472ad62dc8e0063 -->
<a id="u-710472ad62dc8e0063"></a>

<!-- latex-display-v1 -->

<!-- /unit:710472ad62dc8e0063 -->

<!-- unit:d25bd3387a5b0b0397 -->
<a id="u-d25bd3387a5b0b0397"></a>

<!-- source-depth-S7-Ж -->

<!-- /unit:d25bd3387a5b0b0397 -->

## Архитектуры: подробные схемы

### SORT

![SORT](<../оформление/course/arch-sort.svg>)

Kalman prediction и observation update разделены assignment. Недетектированный объект не обязательно сразу удаляется: правила жизненного цикла tracks являются частью алгоритма и evaluation.

### DeepSORT

![DeepSORT](<../оформление/course/arch-deepsort.svg>)

Appearance описывает crop и помогает association после пропуска; motion gating ограничивает допустимые связи. Similarity embedding не заменяет алгоритм назначения или правила track lifetime.

### ByteTrack

![ByteTrack](<../оформление/course/arch-bytetrack.svg>)

Low-score detections используются на втором этапе для поддержки существующих tracks. Это не означает создавать новый track из любого слабого detection; thresholds и lifecycle требуют согласованного protocol.

### BoT-SORT

![BoT-SORT](<../оформление/course/arch-botsort.svg>)

Компенсация движения камеры изменяет motion alignment; appearance используется согласно конфигурации. Сравнение с ByteTrack требует одинакового detector и явного указания включённых компонентов.

### SAM 3.1: video concept tracking

![SAM 3.1: video concept tracking](<../оформление/course/arch-sam31.svg>)

Мультиплексирование меняет организацию обработки объектов; сохранение identity и masks требует temporal state. Сообщённые автором FPS относятся к его hardware, числу объектов и implementation.

## Результат главы и переход дальше

Ответьте на вопрос: **Как сохранить идентичность объекта между кадрами?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

[Маршрут модуля](<README.md>) · [Происхождение и точные места переноса](<../editorial/coverage.md>)
