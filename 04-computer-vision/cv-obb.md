# Ориентированные рамки и периодичность угла

**Уровень:** углубление.

**Предпосылки:** [Detection: множества объектов, IoU, AP и архитектуры](<cv-detection.md>)

**Используют:** [Медицинские и спутниковые изображения: данные, объём и масштаб](<cv-domains.md>)

**Учебный вопрос:** Как сравнивать геометрически одинаковые рамки с разными параметрами?

## Карта вычисления и границы этапов

![Ориентированные рамки и периодичность угла: входы, преобразования и выходы](<../оформление/course/cv-obb.svg>)

Поворот добавляет неоднозначность параметризации: разные углы и перестановка сторон могут задавать одну рамку. Функция потерь должна учитывать эту эквивалентность.

## Почему повёрнутый прямоугольник полезен

Для длинного корабля под углом axis-aligned enclosing box захватывает много background и может перекрывать соседний объект. OBB задаёт centre $(c_x,c_y)$, размеры w/h и θ. Это плотнее описывает ориентированный объект, но не заменяет точную segmentation mask.

Axis-aligned detection не становится математически «неприменимой»: это другой, иногда достаточный output. Польза OBB должна соответствовать задаче и разметке. Для почти круглого объекта угол может не иметь устойчивого смысла; для квадрата часть углов эквивалентна, и annotation noise возрастает.

Annotation может хранить четыре corners, а model head — внутренние $(c_x,c_y,w,h,\theta)$. При conversion нужны порядок corners, единицы угла, clockwise/counterclockwise convention и направление image y. Library angle ranges различаются; нельзя копировать старое «OpenCV всегда имеет диапазон…» без версии/конвенции.

### Сколько лишнего фона даёт горизонтальный envelope

В разбираемом примере площадь enclosing axis-aligned box была записана как wh(cosθ+sinθ), а увеличение на 45° — примерно в 1.41 раза. Это неверно: размеры envelope зависят и от w, и от h. При положительных размерах и любом θ:

$$
W_{\rm env}=|w\cos\theta|+|h\sin\theta|,\qquad
H_{\rm env}=|w\sin\theta|+|h\cos\theta|.
$$

Площадь — произведение этих двух длин. Для квадрата со стороной a на 45° envelope имеет стороны a√2 и площадь 2a²: лишнего относительно исходного ровно вдвое, не √2. Для прямоугольника w=10,h=2 на 45° обе envelope стороны равны 12/√2, площадь 72 при исходной 20; отношение 3.6. Если сравнить именно этот rectangle с его envelope по одной polygon convention, IoU=20/72≈0.277778.

Это не означает, что horizontal detector принципиально не может найти корабль. Если benchmark GT тоже горизонтальный envelope, правильный envelope может иметь IoU=1 относительно него. Низкое 0.277778 получилось из сравнения разных shapes. Формат разметки и нужная downstream геометрия определяют задачу; OBB полезна, когда ориентированная локализация действительно нужна.

### Разметка corners и физический смысл ориентации

OBB задаёт направление прямоугольной оси, но не обязательно направление движения или «нос» объекта. Корабль, повёрнутый на 180°, занимает тот же rectangle, хотя его нос смотрит в обратную сторону. Если нужно directed heading, период π уже недостаточен: нужны отдельная разметка или дополнительные признаки.

DOTA-like строка содержит четыре corners, class и optional difficulty по dataset convention; YOLO-OBB training labels (метки) могут хранить нормированные corners. «COCO-style bbox из пяти чисел» — пользовательское расширение, а не автоматически стандартный COCO four-number bbox. Перед конвертацией фиксируют порядок vertices, units и class mapping. Параметризация из разбираемого примера, названная шестипараметрической, но содержащая пять перечисленных чисел, сама противоречива и без точного определения не используется.

## Преобразование в corners

Для θ в радианах и локальных signs $(s,t)\in\{(-1,-1),(1,-1),(1,1),(-1,1)\}$:

$$
\begin{aligned}x&=c_x+\frac{sw}{2}\cos\theta-\frac{th}{2}\sin\theta,\\y&=c_y+\frac{sw}{2}\sin\theta+\frac{th}{2}\cos\theta.\end{aligned}
$$

Это стандартная rotation matrix в согласованной координатной системе. Если image y направлен вниз, визуально положительный поворот выглядит иначе, чем на математической плоскости с y вверх; численный protocol должен быть один.

Пример $centre=\left(0,\,0\right),\qquad w=4,\qquad h=2,\qquad \theta=\frac{\pi}{2}$. Corners до поворота $\left(\left(-2,\,-1\right),\,\left(2,\,-1\right),\,\left(2,\,1\right),\,\left(-2,\,1\right)\right)$, после — $\left(\left(1,\,-2\right),\,\left(1,\,2\right),\,\left(-1,\,2\right),\,\left(-1,\,-2\right)\right)$. Площадь остаётся8. Новый axis-aligned envelope имеет width2,height4.

Обратное восстановление centre как среднего corners и lengths соседних edges работает для корректного прямоугольника с известным cyclic order. Если GT polygon — произвольный quadrilateral, fitted minimum-area rectangle может изменить shape/area; это approximation, которую нужно явно учитывать.

### Обратное преобразование и проверка прямоугольности

Для циклических corners p1,p2,p3,p4 прямоугольника центр — их среднее. Векторы e1=p2−p1 и e2=p3−p2 дают соседние стороны, w=||e1||, h=||e2||, θ=atan2(e1_y,e1_x). Проверка $e_1\cdot e_2\approx0$ выявляет отсутствие прямого угла. Противоположные стороны должны быть параллельны и равны по длине. Если входные точки переставлены крест-накрест, формулы соседних сторон больше не описывают rectangle.

Для source-примера (100,200),(300,200),(300,400),(100,400) получаются center=(200,300),w=h=200,θ=0 в выбранном порядке. В квадрате θ=π/2 с соответствующей перестановкой сторон задаёт ту же область. Для general quadrilateral среднее вершин — удобный reference, но не обязательно area centroid, а lengths двух соседних edges не превращают shape в прямоугольник. Minimum-area fitting — отдельная approximation.

При геометрической аугментации безопасно сначала преобразовать corners, затем заново получить целевое представление согласно convention. Например, anisotropic resize по x/y может превратить повёрнутый rectangle в параллелограмм, который уже не является OBB точной формы. Если pipeline снова fit-ит rectangle, он изменяет target; это следует учитывать, а не считать любой resize просто изменением center,w,h при прежнем θ.

## Один rectangle, разные числа

Геометрически:

$$
\begin{aligned}(w,h,\theta)&\equiv(w,h,\theta+\pi),\\(w,h,\theta)&\equiv\left(h,w,\theta+\frac\pi2\right).\end{aligned}
$$

В первом случае w/h не меняются. В исходном объяснении словесные условия этих двух равенств путались. Для ordinary undirected rectangle θ=1° и179° отличаются по orientation только на2°, а raw MSE видит178°.

Для period π можно кодировать direction $(\cos2\theta,\sin2\theta)$ или задать cyclic distance. Один smooth loss (функция потерь):

$$
\mathcal{L}_{\mathrm{angle}}=1-\cos\!\left(2(\theta_{\mathrm{pred}}-\theta_{\mathrm{true}})\right).
$$

При true1°,pred179° разность178°, удвоенный угол356°, loss≈0.002436. При true1°,pred91° и **фиксированных** w/h разность90°, loss2. Но если w/h поменялись местами, rectangle тот же, и whole-box representation/loss должен учесть ещё эту equivalence. Одной doubled-angle формулы недостаточно для всех conventions.

Canonicalization, например w≥h, уменьшает ambiguity, но вблизи square w≈h малая perturbation может менять выбранную длинную сторону. Angle classification с circular smooth labels задаёт wrap-around соседство bins; ordinary one-hot на180 bins сохраняет жёсткий разрыв между крайними индексами.

### Круговое расстояние и doubled-angle encoding

Для undirected orientation можно взять $d_\pi(\theta_1,\theta_2)=\min_{k\in\mathbb Z}|\theta_1-\theta_2+k\pi|$. Результат лежит между 0 и π/2. В паре 1°/179° ближайшие представители отличаются на 2°, поэтому squared distance должна учитывать именно 2°, а не 178°. Простое ограничение обоих углов диапазоном не убирает разрыв у его границы.

Отображение e(θ)=(cos2θ,sin2θ) одинаково для θ и θ+π. При единичных encoded vectors $\lVert e(\theta_1)-e(\theta_2)\rVert^2=2[1-\cos2(\theta_1-\theta_2)]$. Это связь Euclidean loss по encoding с приведённым cyclic loss. Если network выдаёт произвольные два числа, их длина не обязана быть единицей; normalizing почти нулевой вектор требует epsilon и определённого decoding.

Для circular labels с 180 one-degree bins истинный угол 0° должен иметь соседями и 1°, и 179°. Например, Gaussian weights вычисляют через circular bin distance, затем нормируют. Обычная label (метка) smoothing, одинаково раздающая массу всем чужим bins, не кодирует такую геометрию. CSL меняет target neighborhood, DCL — способ кодирования angle classes; это не безусловное устранение всех неоднозначностей w/h.

Canonical rule w≥h выбирает длинную сторону, но при w=10.001,h=10 малое изменение размеров может поменять выбранную ось на 90°. Geometry изменилась мало, численный angle target — резко. Этот пример объясняет instability square-like объектов без необходимости утверждать, что их orientation физически важна для любой задачи.

## Rotated IoU и surrogate losses

Формула IoU прежняя, сложнее вычислить intersection polygon. Можно найти пересечение convex polygons, отсортировать vertices и использовать polygon area. Нужно проверять порядок, точность около касания и degenerate cases. Axis-aligned IoU enclosing boxes — другой metric.

Повёрнутый IoU не обязательно лишён производных при всех углах. Он piecewise geometry с изменяющимися intersection configurations, сложными/негладкими границами и возможными слабыми signals. Специализированные differentiable implementations и surrogate objectives существуют; exact behavior зависит от formula/kernel.

Gaussian representation связывает rectangle centre с mean, а dimensions/orientation — с covariance вида $R\operatorname{diag}\!\left(\frac{w^2}{12},\frac{h^2}{12}\right)R^\top$ для uniform-rectangle moment interpretation. KLD/Gaussian Wasserstein/ProbIoU-family losses сравнивают соответствующие distributions. Но Gaussian overlap не равен literal polygon IoU: substitute должен оцениваться по целевой benchmark metric.

При square w=h covariance становится isotropic, и направление естественно теряет различимость. Это полезная геометрическая intuition, но разные Gaussian losses имеют разные scales и numerical details. Нельзя объявлять любой «probabilistic IoU» одной универсальной формулой.

### Polygon intersection и проверка площади

Для двух convex rectangles можно clip-ить один polygon последовательно half-planes другого, сохраняя внутренние вершины и точки пересечения рёбер. Итоговый polygon может иметь больше четырёх vertices. Для циклического списка q площадь вычисляется shoelace formula:

$$
\mathcal A=\frac12\left|\sum_{i=1}^{n}(x_iy_{i+1}-y_ix_{i+1})\right|,\qquad q_{n+1}=q_1.
$$

Дальше IoU использует площади двух исходных rectangles и intersection. Nearly parallel edges, касания и почти нулевая площадь требуют numerical tolerance. Если поменять порядок vertices произвольно, можно получить self-intersection и неверную площадь. Отсутствие простой одной ветви вычисления не равно отсутствию математической функции или всех производных.

### Gaussian box: от размеров к covariance

У равномерной координаты на [−w/2,w/2] дисперсия w²/12. Для axis-aligned uniform rectangle covariance=diag(w²/12,h²/12); rotation преобразует её как RΣRᵀ. Mean равна center. Именно moment-matching интерпретация объясняет коэффициент 1/12. Формула с w²/4,h²/4 соответствует иному масштабу и не должна незаметно подменять этот mapping.

При w=4,h=2 и θ=0 eigenvalues равны 4/3 и 1/3. При θ=π/2 они меняются местами на координатных осях, а при θ=π covariance возвращается к исходной. Для w=h обе eigenvalues равны, поэтому covariance одинакова при любых θ. Она описывает симметрию rectangle как распределения, но не сохраняет directed heading.

### Чем ProbIoU отличается от расстояния Вассерштейна

В разбираемом примере squared Gaussian Wasserstein expression была названа ProbIoU, а similarity записана $1-e^{-d^2}$. При совпадении boxes эта величина равна нулю, при удалении растёт к единице: это поведение distance-like loss, а не similarity, которая должна быть максимальна при совпадении.

Для невырожденных Gaussian P,Q один путь к probabilistic similarity идёт через Bhattacharyya distance. С $\overline\Sigma=(\Sigma_P+\Sigma_Q)/2$ и Δμ=μP−μQ:

$$
D_B=\frac18\Delta\mu^\top\overline\Sigma^{-1}\Delta\mu+
\frac12\ln\frac{\det\overline\Sigma}{\sqrt{\det\Sigma_P\det\Sigma_Q}}.
$$

Из него $H=\sqrt{1-e^{-D_B}}$ — Hellinger distance в этой convention, similarity=1−H. При одинаковых Gaussians DB=0,H=0,similarity=1; при больших различиях она уменьшается. Epsilon/clamps и optional penalties влияют на точный code output. Это сохраняет идею исходного разбора, но разделяет разные distances. [ProbIoU](https://arxiv.org/abs/2106.06072), [реализация Ultralytics](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py).

Если covariance двух boxes одинакова и центры отличаются на Δμ, log-determinant term равен нулю, остаётся quadratic form. Вдоль длинной оси с большей variance один и тот же shift даёт меньший penalty, чем вдоль короткой. Поэтому distribution comparison учитывает shape и orientation, а не только расстояние центров. Но literal polygon IoU по-прежнему другая величина.

Gaussian Wasserstein distance использует иной matrix-square-root term; KL в общем случае асимметрична, поэтому KL(P||Q) не равно KL(Q||P). Назвать все эти функции «Gaussian IoU» и ожидать одинакового scale/gradient (градиент) нельзя. Выбранный loss сверяют с benchmark rotated IoU, а также с boundary/square-like cases.

### Архитектурные решения по конкретному ограничению

Rotated RPN добавляет orientation в proposal stage; это увеличивает пространство reference shapes, если используются angle anchors. RoI Transformer учит преобразование horizontal regions в oriented regions, чтобы извлекать согласованные признаки. Oriented R-CNN использует собственное proposal/representation design; его нельзя свести к «просто добавить число θ» без учёта assignment и decoding.

Gliding Vertex и corner-based methods изменяют output representation: вместо прямой angle regression предсказывают geometric offsets/vertices. Неоднозначность порядка углов остаётся. ReDet вводит rotation-equivariant representations, CSL/DCL меняют angular target encoding. Text detection методы вроде EAST или DB могут выдавать quadrilaterals или polygons и не обязаны решать только задачу прямоугольного OBB.

YOLO-OBB, MMRotate и rotated Detectron2-сценарии — конкретные системы/реализации, где нужно согласовать angle units, mapping и output interpretation. Обычный RT-DETR checkpoint (сохранённое состояние модели) автоматически не получает angle output только из-за наличия object queries; oriented variant должен быть определён отдельно. Обобщённые заявления о каждой версии и каждом runtime (среда исполнения модели) поэтому требуют конкретного artifact.

### Suppression и rotation TTA

Если используется NMS, критерий duplicate должен соответствовать oriented geometry или явно выбранной surrogate similarity. Замена polygon overlap на Gaussian similarity меняет решение: одинаковые thresholds не обязаны давать одинаковые keep lists. Даже differentiable similarity не делает жёсткую операцию keep/drop differentiable.

При rotation TTA predictions нужно вернуть inverse affine transform в координаты исходного изображения. Сырые углы 1° и 179° нельзя усреднять как 90° для undirected rectangle. Усреднение doubled-angle vectors даёт направление возле 0° при согласованных w/h; ещё нужно решить correspondence instances. NMS-free inference (применение обученной модели) path уменьшает потребность в suppression, но не устраняет error analysis duplicates/misses в реальном output.

## Архитектуры и проверка

Rotated RPN/RRPN добавляют proposals с ориентацией; RoI Transformer учит переход от horizontal к rotated regions; corner-based methods предсказывают vertices; dense OBB heads добавляют angle/геометрические outputs. YOLO-OBB — task-specific detector, где topology/head/loss нужно сверять по checkpoint, включая boundary fixes конкретной версии.

Postprocessing требует rotated duplicate handling, если inference path его использует. Угол в degrees, переданный kernel как radians, даёт корректные shapes и совершенно неверную геометрию. Лучше визуализировать corners **до** обучения, после transforms и после inverse mapping predictions.

Срезы: elongated vs square objects, near angle boundary, crowded small objects, flipped/mirrored inputs. Split (разбиение данных) по сценам/географии важен для aerial images; соседние overlapping tiles не независимый test.

1. Обязательно ли менять w/h при θ→θ+π? **Ответ:** нет; swap соответствует дополнительным π/2.
2. Почему θ одного square неустойчив? **Ответ:** rotation symmetries и близость dimensions делают canonical direction неоднозначным.
3. Можно ли заменить polygon IoU Gaussian similarity? **Ответ:** как training surrogate при проверке, но benchmark definition сама не меняется.

**Практика:** пересчитать rotation example, затем применить inverse matrix и получить original corners. Проверить equality $\left(4,\,2,\,0\right)$ и $\left(2,\,4,\,\frac{\pi}{2}\right)$. Для cyclic loss сравнить1°/179° и1°/91°, явно задав whether w/h swapped.

**Схема курса:** image → oriented box parameters (параметры) → angle canonicalization/periodic loss → rotated geometry and matching → rotated IoU/NMS → angle-aware evaluation





## Архитектуры: подробные схемы

### Ориентированный detector

![Ориентированный detector](<../оформление/course/arch-obb.svg>)

Голова задаёт форму рамки, decode преобразует её в corners. Периодичность θ и эквивалентность перестановки сторон учитываются при loss и сравнении; стандартный axis-aligned IoU здесь отвечает другой геометрической задаче.

## Результат главы и переход дальше

[Маршрут модуля](<README.md>) · [Происхождение и точные места переноса](<../editorial/coverage.md>)

## Как читать углубления этой ветки


Ориентированная рамка может задаваться center, width, height и angle либо четырьмя vertices. Представления $(w,h,\theta)$ и $(h,w,\theta+\pi/2)$ могут описывать один rectangle. Target canonicalization должна совпадать между annotation conversion, regression и evaluator. Поворот image требует пересчёта всей geometry, а не только добавления angle.

Удлинённые aerial objects подробно рассматриваются в [главе о domains](<cv-domains.md>). Для почти square boxes orientation хуже обусловлена: небольшое изменение vertices может резко менять canonical angle без существенного изменения фигуры.

## Разобранный пример для проверки контракта


Для angle, периодичного по $\pi$, prediction 89° и target -89° различаются по обычному subtraction на 178°, но минимальная physical difference равна 2°. Представление $(\cos2\theta,\sin2\theta)$ делает equivalent angles совпадающими. При восстановлении $\theta=\tfrac12\operatorname{atan2}(s,c)$ получают выбранную canonical branch, а не уникальный универсальный angle.

## Ограничения переноса

Конвенции clockwise/counterclockwise, degrees/radians и image y-axis direction часто различаются между libraries. Совместимость проверяют на нарисованных corner points.

## Самопроверка с ответами

**Практика:** поменяйте width и height местами и добавьте 90°. Перед вычислением regression loss проверьте, что representation canonicalized, иначе эквивалентная рамка будет оштрафована.

**Почему angle MSE сам по себе опасен?** Он не учитывает periodicity и equivalences representation.

**Достаточно ли axis-aligned IoU?** Нет, он теряет ориентацию и иначе оценивает перекрытие длинных boxes.

## Схема вычислений

![Ориентированные рамки и периодичность угла: вычислительные этапы](<../оформление/course/expanded-cv-obb.svg>)

Rotated detector: Image → multi-scale features → Center / size / angle outputs → Canonicalization + rotated loss → Rotated IoU / NMS / AP.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Источники и продолжение

- [DOTA — authors benchmark](https://captain-whu.github.io/DOTA/)

[Следующие углубления: cv-domains](<README.md>) · [Сценарии подраздела](<03-detection/План.md>)
