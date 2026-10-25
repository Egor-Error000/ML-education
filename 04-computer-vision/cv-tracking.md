# Tracking: Kalman, association и идентичность

**Уровень:** основа ветки.

**Предпосылки:** [Detection: множества объектов, IoU, AP и архитектуры](<cv-detection.md>)

**Используют:** [SORT, DeepSORT, ByteTrack и BoT-SORT: разбор последовательности](<cv-tracking-models.md>)

**Учебный вопрос:** Как сохранить идентичность объекта между кадрами?

## Карта вычисления и границы этапов

![Tracking: Kalman, association и идентичность: входы, преобразования и выходы](<../оформление/course/cv-tracking.svg>)

Трекер связывает наблюдения во времени. Правильная рамка в каждом кадре ещё не гарантирует сохранение идентичности; ошибки детекции и ассоциации оцениваются отдельно.

## Детекция кадра ещё не tracking

На каждом frame detector может отлично находить двух людей, но менять им IDs местами. Object detection quality останется высокой, а задачи подсчёта пересечений/траекторий сломаются. Track — история состояния и identity во времени, а detection — наблюдение одного кадра.

Tracking-by-detection система хранит active/lost/tentative tracks. На новом frame прогнозирует положение, сопоставляет detections с tracks, обновляет matched tracks, создаёт новые и завершает слишком давно потерянные. Эти decisions — алгоритм над video state (состояние системы), не обязательное дообучение detector.

Нужны согласованные timestamps/FPS, coordinates и camera motion handling. Если frames приходят нерегулярно, постоянный шаг движения без учёта delta_t ошибочен. Offline tracker может использовать будущие frames; online protocol — нет. Сравнивать их без указания доступа к будущему некорректно.

### Что хранится между кадрами и почему ID нельзя получить из class

Для человека class может всё время оставаться `person`, но конкретных людей несколько. Track ID — условная постоянная метка одной истории. Если detector на каждом кадре сортирует людей по score (оценка) и назначает IDs по номерам строк, identities будут менятьcя при изменении confidence (уверенность по принятой оценке), даже если boxes остаются правильными. Поэтому tracker сопоставляет наблюдения с прежними состояниями, а не просто дописывает номер к каждой detection.

Track management содержит отдельные решения. Новая detection может породить tentative track, который ещё не показывается как подтверждённый. Повторные согласованные наблюдения подтверждают его. При пропуске track становится lost и живёт ограниченное время; после timeout удаляется. В разных реализациях названия и переходы различаются. Predicted box lost-трека — гипотеза движения, а не новое наблюдение объекта.

В MOTChallenge-like result строка содержит frame,ID,box,score и предусмотренные дополнительные поля. Ground-truth формат может включать class/visibility/ignore semantics, поэтому source-объяснение «все оставшиеся поля всегда заглушки, confidence GT всегда 1» нельзя переносить без проверки конкретного файла. Сдвиг frame numbering на единицу испортит matching при визуально похожем видео. Координаты xywh с corner также надо отличать от center-формы фильтра.

Online tracker использует только доступное прошлое; near-online добавляет буфер и latency (задержка отдельного запроса), offline может переопределять связи после просмотра будущих кадров. «Online» не гарантирует обработку в real time: тяжёлый ReID или большой input может быть медленнее FPS камеры. Наоборот, offline batched computation может быть быстрым по throughput (число обработанных единиц за время), но всё равно не подходит для мгновенной реакции.

## Kalman prediction/update

Линейная state-space иллюстрация:

$$
\begin{aligned}s_t&=Fs_{t-1}+\varepsilon_t,\qquad z_t=Hs_t+\nu_t,\\s^-&=Fs,\qquad P^-=FPF^\top+Q,\\K&=P^-H^\top(HP^-H^\top+R)^{-1},\\s^+&=s^-+K(z-Hs^-).\end{aligned}
$$

State может включать box centre/scale/velocity; details SORT/BoT variants различаются. P описывает uncertainty estimate, Q/R — process/measurement noise. Это модель и допущения, а не точная физика поведения человека.

Скалярный пример: predicted position — 10, variance — 4, measurement — 12 с noise variance 1. Тогда $K=\frac{4}{4+1}=0.8$, updated position — 11.6, variance — 0.8 по простой формуле $(1-K)P^-$. Наблюдение с меньшим noise получает больший вес. При occlusion measurement нет: prediction продолжает двигаться, uncertainty растёт, но подтверждённой identity это не добавляет.

### От состояния SORT к координатам box

В учебном SORT-примере state имеет семь координат $(u,v,a,r,\dot u,\dot v,\dot a)$: центр, площадь, aspect ratio и три скорости. Здесь a обозначает площадь, чтобы не смешивать её с полным state s. При r=w/h получаем $w=\sqrt{ar}$, $h=\sqrt{a/r}$ для положительных a,r. Скорости площади и линейных размеров — разные модели; DeepSORT/BoT variants не обязаны хранить точно те же семь величин.

Для позиции и скорости одна constant-velocity часть transition имеет вид

$$
\begin{bmatrix}u_t^-\\\dot u_t^-\end{bmatrix}=
\begin{bmatrix}1&\Delta t\\0&1\end{bmatrix}
\begin{bmatrix}u_{t-1}^+\\\dot u_{t-1}^+\end{bmatrix}.
$$

Если u=10,velocity=2 pixels/frame и прошло три frame intervals, forecast u=16. Использовать u=12 после пропуска двух кадров — значит ошибиться в time step. При измерениях в секундах velocity и Q тоже должны иметь согласованные единицы.

Innovation $z-Hs^-$ — разность измерения и предсказания. Kalman gain зависит от их uncertainty. В скалярном примере главы K=0.8, поэтому к forecast10 добавляется 0.8·(12−10)=1.6. Если measurement variance поднять с 1 до 16 при прежней P−=4, K=0.2 и update даст 10.4. Фильтр меньше доверяет шумному наблюдению; один большой score detector не заменяет автоматически правильно выбранную R.

После update covariance в простой записи $P^+=(I-KH)P^-$; при аккуратной численной реализации могут использовать Joseph form. Несколько прогнозов без measurements добавляют process uncertainty. Это допускает более широкую область будущих matching, но расширение повышает риск связать другого человека. Неопределённость — часть решения, а не обещание помнить identity бесконечно.

## Assignment: почему greedy (выбор максимальной оценки на каждом шаге) nearest не всегда годится

Для predicted tracks i и detections j строится cost: например $1-\operatorname{IoU}$, Mahalanobis distance, appearance cosine distance или их смесь. Gating исключает физически/семантически неприемлемые пары до/после assignment по определённому protocol.

Матрица cost:

$$
C_{\mathrm{association}}=\begin{bmatrix}1&2\\2&100\end{bmatrix}\quad\text{(строки — tracks, столбцы — detections)}.
$$

Greedy берёт минимальную1 для track1→det1, оставляя100 track2→det2, total101. Global assignment track1→det2 и track2→det1 даёт4. Hungarian algorithm ищет minimum total для заданной matching постановки. Нужно также учесть возможность unmatched: нельзя принуждать любую detection к любому track только ради полного bijection.

Смешивание costs требует scales/weights. Если appearance distance [0,2], а raw coordinate distance сотни pixels, простая сумма без normalization (нормировка по выбранному правилу) почти игнорирует appearance. Нельзя называть такую схему «balanced motion+ReID» без проверки.

### Motion gating, appearance и возможность оставить без пары

Для residual r=z−Hs− covariance наблюдения $S=HP^-H^\top+R$ squared Mahalanobis distance равна $r^\top S^{-1}r$. Сдвиг на 10 pixels может быть неправдоподобным при дисперсии 1 и допустимым при большой uncertainty. Euclidean distance не учитывает эту разницу. Gating запрещает совсем неправдоподобные связи, а assignment выбирает среди допустимых.

Пусть два похожих человека пересеклись. Геометрический cost может предпочесть nearest box, а appearance — другую пару. Для нормированных embeddings (векторные представления) e1,e2 cosine distance=1−e1ᵀe2. Нужны сопоставимые crops и trained representation; одинаковая одежда может сделать обе appearance distances близкими. Простая weighted sum не превращает отсутствие информации в надёжное решение.

В матрице главы greedy total101 и global total4 показывают отличие оптимизации суммы от локального выбора. Но если все доступные пары cost100, правильнее оставить unmatched, чем гарантированно передать чужой ID. Для этого постановка содержит thresholds/dummy options/последующее отбрасывание согласно реализации. «Hungarian нашёл минимум» доказывает оптимум заданного cost, а не истинность identities.

## SORT, DeepSORT, ByteTrack и BoT-SORT

SORT сочетает Kalman dynamics и assignment по box geometry. Он прост и полезен как baseline (базовый вариант сравнения); труднее различает похожие trajectories при occlusion/crossing. [SORT](https://arxiv.org/abs/1602.00763).

DeepSORT добавляет appearance representation/ReID distance и соответствующий matching pipeline. Это помогает при кратком исчезновении и похожей motion geometry, но одинаковая одежда, domain shift и poor crops могут давать ложную уверенность. ReID model тоже нуждается в подходящей training distribution.

ByteTrack сначала использует high-score detections, затем связывает оставшиеся tracks с подходящими low-score detections. Идея: occluded object может иметь низкую confidence, но coherent motion. Это не «принять все слабые detections как новые tracks»; false low-score background отсекается association policy. [ByteTrack](https://arxiv.org/abs/2110.06864).

BoT-SORT сочетает motion/association changes с camera motion compensation и, в некоторых configs, appearance. Нельзя считать ReID обязательным у любого выбранного tracker config. В интеграции с YOLO детектор и tracker остаются разными компонентами; model version не определяет автоматически matching thresholds/timeouts.

### DeepSORT cascade: исправляем порядок из разбираемого примера

Ошибочная формулировка cascade: «сначала высокий confidence, затем низкий». В original DeepSORT matching cascade приоритет связан с временем после последнего update: недавно наблюдавшиеся tracks обрабатываются раньше давно потерянных. Это уменьшает конкуренцию со старыми неопределёнными hypotheses. Отбор detector scores (оценки) и cascade recency — разные механизмы. Проверить порядок можно в [исходном matching cascade](https://github.com/nwojke/deep_sort/blob/master/deep_sort/linear_assignment.py).

Appearance gallery хранит несколько embeddings истории track, чтобы matching учитывал изменение ракурса. Слишком старая или загрязнённая gallery может закрепить ошибку после неверного присоединения. ReID полезен для продолжения identities, но не делает long re-entry всегда разрешимым. Решение о новом ID иногда честнее, чем уверенное присоединение к старому без достаточного signal.

### ByteTrack на трёх последовательных кадрах

В кадре t человек имеет detection score0.9 и подтверждённый track. В t+1 он частично закрыт: score0.25, но box близок forecast. Если входной confidence filter заранее удаляет всё ниже0.5, association не получит это наблюдение. Если сохранить low-score pool до0.1, второй этап ByteTrack может связать score0.25 с unmatched track. В t+2 score снова0.85, и identity продолжится без разрыва.

Однако слабая detection далёкого background не должна автоматически породить новый track. Поэтому high/low pool, first matching, second matching и new-track threshold — отдельные decisions. Числа 0.5/0.1/0.6 из разбираемого примера иллюстрируют разные пороги, а не универсальные настройки. Изменить внешний detector conf на0.3 при low threshold0.1 может сделать часть low pool недоступной.

### Camera motion и совместные модели

При camera pan статичный объект сдвигается в image coordinates. Модель «всё движение связано с собственным velocity объекта» начинает ошибаться для всех tracks одновременно. Global motion compensation оценивает background transformation и согласованно преобразует track geometry. Она может использовать affine motion, optical flow или feature (признак) matching; не каждая конфигурация обязана оценивать полную homography. Moving foreground и parallax нарушают единый global transform. [BoT-SORT](https://arxiv.org/abs/2206.14651).

JDE/FairMOT извлекают detector outputs и ReID features (признаки) совместно, экономя повторные visual passes; association и lifecycle при этом всё равно существуют. Tracktor использует box regression detector для продолжения регионов, но требует решений о новых tracks и завершении. MOTR/TrackFormer переносят identity через track queries: отсутствие отдельного Hungarian association на inference (применение обученной модели) не означает отсутствие задачи временного соответствия. Поэтому «MOT всегда ровно две независимые программы» — полезное описание tracking-by-detection, но не полная классификация всех architectures.

## MOTA, IDF1 и HOTA

MOTA=$1-\frac{\mathrm{FN}+\mathrm{FP}+\mathrm{IDS}}{GT_{\mathrm{total}}}$, где counts агрегируются по protocol. Если GT100,FN10,FP5,IDS2, MOTA0.83. При суммарных ошибках больше GT она отрицательна. IDS входит явно, но common detection errors могут доминировать. Фраза «MOTA вообще не учитывает identity» неверна; точнее сказать, что её баланс не описывает устойчивость identities полностью.

IDF1=$\frac{2\mathrm{IDTP}}{2\mathrm{IDTP}+\mathrm{IDFP}+\mathrm{IDFN}}$. Identity counts требуют global identity matching protocol, а не простого подсчёта правильного числового ID на frame: predicted ID1 может соответствовать GT ID50. Пропущенные detections увеличивают IDFN, лишние — IDFP, поэтому IDF1 не независим от detection quality. Для IDTP80,IDFP10,IDFN20 score160/190≈0.8421.

HOTA на заданном localisation threshold балансирует detection и association: $\mathrm{HOTA}_\alpha=\sqrt{\mathrm{DetA}_\alpha\mathrm{AssA}_\alpha}$, далее агрегируется по thresholds согласно определению. AssA — ассоциационное согласие по matched detections, а не просто1-IDS/N. Нельзя сначала независимо усреднить DetA/AssA и объявить sqrt их произведения exact overall HOTA. [HOTA](https://arxiv.org/abs/2009.07736).

При DetA0.81,AssA0.49 на одном threshold HOTA0.63. В original export корень терялся при записи/извлечении; произведение0.3969 — другой показатель. Дополнительные IDS/fragmentations/coverage дают важную diagnosis; definition каждого берётся из benchmark evaluator.

### Два разных случая, скрытых одинаковым MOTA

Есть один GT человек на десяти кадрах, boxes всегда верные. Предсказанный ID7 действует первые пять, ID8 — вторые пять. Если protocol засчитает одну смену, MOTA=1−1/10=0.9. При global one-to-one identity matching только одна из двух predicted trajectories может соответствовать полной GT identity: IDTP=5,IDFP=5,IDFN=5, IDF1=10/20=0.5. Так видно, почему один краткий IDS penalty не описывает всей фрагментации истории.

Если вместо split (разбиение данных) человек найден с одним ID лишь на первых пяти frames, IDTP=5,IDFP=0,IDFN=5 и IDF1=10/15≈0.666667. Пропуски изменили identity score, хотя ни одного ID-switch нет. Это прямое опровержение фразы «IDF1 не учитывает detection».

MOTA может стать отрицательной: при GT=10,FN=8,FP=10,IDS=3 результат −1.1. Это допустимый показатель по формуле. Для HOTA при одном threshold нужно sqrt(DetA·AssA); если thresholds дают разные пары, усреднение корней не равно корню из отдельно усреднённых факторов. Например, пары (1,0.25) и (0.25,1) дают HOTA по0.5, среднее0.5, тогда как sqrt(0.625²)=0.625 — другое число.

MT/ML характеризуют покрытие GT trajectories, fragmentations — разрывы, IDS — смены по определённому protocol. Их нельзя выводить из одного IDF1. MOTP может быть определён как distance или overlap в конкретной реализации, поэтому направление «больше лучше» надо сверять. Для 3D MOT появляются другие geometry, sensors и protocols, например AMOTA; 2D IoU thresholds туда не переносятся механически.

### Практическая интеграция без скрытого изменения задачи

У YOLO+tracker detector выдаёт boxes, а tracker получает отсортированный во времени поток и persistent state. Случайно сбрасывать state каждый frame — значит снова создавать identities. Но переносить state между двумя независимыми видео тоже ошибка: coordinates и identity history уже относятся к другой сцене. Frame drops, resize и timestamp должны входить в записанную конфигурацию.

`track_buffer`, `max_age`, `min_hits` в разных trackers не обязаны иметь одинаковые единицы/условия. Buffer30 при30FPS примерно одна секунда только при таком определении кадрового шага. При5FPS это уже шесть секунд. В реальном протоколе учитывают способ normalization по frame rate. [Документация tracking](https://docs.ultralytics.com/modes/track/).

Числа FPS из рассматриваемого примера нельзя сравнивать как готовый рейтинг: одни относятся к лёгкому association, другие включают detector/ReID, input sizes и hardware различны. Сравнение SORT/ByteTrack/appearance-конфигурации сохраняет один detector и input stream, отдельно фиксирует detection quality, IDs/IDF1/HOTA, full latency и поведение на crossings/occlusions. Это позволяет понять, какой компонент реально исправил ошибку.

## Occlusion, split и end-to-end evaluation

Long occlusion делает identity неоднозначной, особенно если одинаковые объекты пересекаются вне кадра. Полезно сохранять lost tracks до timeout, но слишком долгий timeout повышает false reattachment. «Всегда вернуть старый ID» — неверная цель без достаточного evidence.

Split по videos/scenes, а не random frames: соседние frames почти дубликаты. Качество зависит от detector confidence threshold, frame rate, video resize и motion compensation; фиксировать всё. Latency включает detector, ReID crops, association и I/O, не только лёгкий Kalman update.

### SAM 3.1: трекинг понятий и цена числа объектов

SAM 3.1 сохраняет promptable concept segmentation интерфейс SAM 3 и меняет обработку нескольких video objects: object multiplexing допускает до 16 tracked objects в одном forward (прямой проход вычисления) pass. В релизном сообщении Meta приводит рост 16→32 FPS на одной H100 для video с medium number of objects. Это пример ускорения конкретной реализации, а не гарантированная скорость вашего tracker.

Для собственного сравнения фиксируйте число объектов, resolution, prompts, video length, batching, precision, hardware и измеряемую границу latency. Считайте одновременно throughput, per-frame latency, memory, пропуски/дубли identity и mask quality. Разделяйте object detection/segmentation score и association quality: быстрее обрабатывать кадры не означает автоматически меньше ID switches.

**Источник и границы:** [официальное сообщение Meta о SAM 3.1](https://ai.meta.com/blog/segment-anything-model-3/), 27 марта 2026. Число FPS и предел 16 объектов приведены самой Meta для названной аппаратуры и тестовой нагрузки; повторное измерение на целевой системе обязательно.

1. Почему нельзя сравнивать IDs как category numbers? **Ответ:** labels (метки) tracks произвольны, важна consistent correspondence, а не равенство чисел.
2. Что делает второй этап ByteTrack? **Ответ:** использует часть low-score observations для продолжения существующих tracks по association.
3. Помогает ли tracker при missed detection? **Ответ:** может прогнозировать/recover, но uncertainty и неоднозначность остаются.

**Практика:** решить assignment example, затем запретить пару track2→det1 gating и определить unmatched policy. Разметить пять кадров двух пересекающихся объектов, специально swap IDs и посмотреть отдельно detection и trajectory (траектория действий и наблюдений) errors. Для реального baseline сравнить SORT и один appearance/low-score вариант с тем же detector.

**Схема курса:** ordered frames + detections → Kalman state prediction → gated assignment/ReID → track update and identity state → trajectory metrics over complete sequence





## Архитектуры: подробные схемы

### SORT

![SORT](<../оформление/course/arch-sort.svg>)

Kalman prediction и observation (доступное наблюдение среды) update разделены assignment. Недетектированный объект не обязательно сразу удаляется: правила жизненного цикла tracks являются частью алгоритма и evaluation.

### DeepSORT

![DeepSORT](<../оформление/course/arch-deepsort.svg>)

Appearance описывает crop и помогает association после пропуска; motion gating ограничивает допустимые связи. Similarity embedding (векторное представление) не заменяет алгоритм назначения или правила track lifetime.

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

[Маршрут модуля](<README.md>) · [Происхождение и точные места переноса](<../editorial/coverage.md>)

## Как читать углубления этой ветки


Эта глава даёт математический вход; [разбор tracking families](<cv-tracking-models.md>) показывает последовательность кадров. Kalman prediction переносит mean и covariance, measurement update объединяет прогноз с observation. Association выбирает, какая observation относится к track; сам Kalman filter не устанавливает identity из class label.

Для state $(x,v)$ constant-velocity transition на интервале $\Delta t$ использует matrix $F=\begin{pmatrix}1&\Delta t\\0&1\end{pmatrix}$. Process noise добавляет uncertainty; его величина отражает imperfect motion model.

## Разобранный пример для проверки контракта


Mean state $(10,2)$ и $\Delta t=0.5$ дают predicted x=11. Для scalar update predicted variance 4, measurement variance 1, observation 12 gain $K=4/(4+1)=0.8$. Updated x $11+0.8(12-11)=11.8$, variance $(1-0.8)4=0.8$. Более точной observation придан больший вес.

## Ограничения переноса

Модель равномерного движения приближённа. Camera motion, ускорения и длительные occlusions требуют дополнительных cues и проверки association errors.

## Самопроверка с ответами

**Практика:** увеличьте measurement variance до 16. Gain $4/20=0.2$, updated x=11.2: observation меньше влияет на state.

**Почему пропуск frame влияет на tracking?** Меняется elapsed time и uncertainty, а не только номер кадра.

**Можно ли обновлять track любой ближайшей box?** Нет, нужны gating, assignment и возможность оставить track без пары.

## Схема вычислений

![Tracking: Kalman, association и идентичность: вычислительные этапы](<../оформление/course/expanded-cv-tracking.svg>)

Kalman + association: Track mean / covariance → Predict с реальным Δt → Gate → assignment observations → Update / lost / lifecycle.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Источники и продолжение

- [SORT — Bewley et al., ICIP 2016](https://arxiv.org/abs/1602.00763)

[Следующие углубления: cv-tracking-models](<README.md>) · [Сценарии подраздела](<05-video/План.md>)
