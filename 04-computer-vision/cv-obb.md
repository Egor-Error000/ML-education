# Ориентированные рамки и периодичность угла

**Уровень:** основа ветки.

**Предпосылки:** [Изображение, карты признаков и блоки модели](<cv-input.md>), [Классификация, CNN и residual learning](<cv-classification.md>), [Detection: множества объектов, IoU, AP и архитектуры](<cv-detection.md>)

**Используют:** Выбранная прикладная задача и итоговая проверка результата.

**Учебный вопрос:** Как сравнивать геометрически одинаковые рамки с разными параметрами?





## Карта вычисления и границы этапов

![Ориентированные рамки и периодичность угла: входы, преобразования и выходы](<../оформление/course/cv-obb.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Сравните формы и смысл каждого перехода с формулами главы. Выход преобразования и доказательство его качества — разные объекты: последняя проверка использует отдельный протокол оценки.

<!-- unit:66a2699179ab76e7ea -->
<a id="u-66a2699179ab76e7ea"></a>

## Почему повёрнутый прямоугольник полезен

<!-- /unit:66a2699179ab76e7ea -->

<!-- unit:44fd9bc19cff92422d -->
<a id="u-44fd9bc19cff92422d"></a>

Для длинного корабля под углом axis-aligned enclosing box захватывает много background и может перекрывать соседний объект. OBB задаёт centre $(c_x,c_y)$, размеры w/h и θ. Это плотнее описывает ориентированный объект, но не заменяет точную segmentation mask.

<!-- /unit:44fd9bc19cff92422d -->

<!-- unit:39f8c69d22180097cc -->
<a id="u-39f8c69d22180097cc"></a>

Axis-aligned detection не становится математически «неприменимой»: это другой, иногда достаточный output. Польза OBB должна соответствовать задаче и разметке. Для почти круглого объекта угол может не иметь устойчивого смысла; для квадрата часть углов эквивалентна, и annotation noise возрастает.

<!-- /unit:39f8c69d22180097cc -->

<!-- unit:12e01be6c4ba94d47e -->
<a id="u-12e01be6c4ba94d47e"></a>

Annotation может хранить четыре corners, а model head — внутренние $(c_x,c_y,w,h,\theta)$. При conversion нужны порядок corners, единицы угла, clockwise/counterclockwise convention и направление image y. Library angle ranges различаются; нельзя копировать старое «OpenCV всегда имеет диапазон…» без версии/конвенции.

<!-- /unit:12e01be6c4ba94d47e -->

<!-- unit:270fe3f710bf323e03 -->
<a id="u-270fe3f710bf323e03"></a>

### Сколько лишнего фона даёт горизонтальный envelope

<!-- /unit:270fe3f710bf323e03 -->

<!-- unit:a0c50df21696fd2ed3 -->
<a id="u-a0c50df21696fd2ed3"></a>

В разбираемом примере площадь enclosing axis-aligned box была записана как wh(cosθ+sinθ), а увеличение на 45° — примерно в 1.41 раза. Это неверно: размеры envelope зависят и от w, и от h. При положительных размерах и любом θ:

<!-- /unit:a0c50df21696fd2ed3 -->

<!-- unit:163a9af79f7c92a313 -->
<a id="u-163a9af79f7c92a313"></a>

$$
W_{\rm env}=|w\cos\theta|+|h\sin\theta|,\qquad
H_{\rm env}=|w\sin\theta|+|h\cos\theta|.
$$

<!-- /unit:163a9af79f7c92a313 -->

<!-- unit:52bcdcce1d720f7309 -->
<a id="u-52bcdcce1d720f7309"></a>

Площадь — произведение этих двух длин. Для квадрата со стороной a на 45° envelope имеет стороны a√2 и площадь 2a²: лишнего относительно исходного ровно вдвое, не √2. Для прямоугольника w=10,h=2 на 45° обе envelope стороны равны 12/√2, площадь 72 при исходной 20; отношение 3.6. Если сравнить именно этот rectangle с его envelope по одной polygon convention, IoU=20/72≈0.277778.

<!-- /unit:52bcdcce1d720f7309 -->

<!-- unit:8cf0cc0cd1c17865ab -->
<a id="u-8cf0cc0cd1c17865ab"></a>

Это не означает, что horizontal detector принципиально не может найти корабль. Если benchmark GT тоже горизонтальный envelope, правильный envelope может иметь IoU=1 относительно него. Низкое 0.277778 получилось из сравнения разных shapes. Формат разметки и нужная downstream геометрия определяют задачу; OBB полезна, когда ориентированная локализация действительно нужна.

<!-- /unit:8cf0cc0cd1c17865ab -->

<!-- unit:b329e1e334e1e3475c -->
<a id="u-b329e1e334e1e3475c"></a>

### Разметка corners и физический смысл ориентации

<!-- /unit:b329e1e334e1e3475c -->

<!-- unit:bc1ff42137ddf49099 -->
<a id="u-bc1ff42137ddf49099"></a>

OBB задаёт направление прямоугольной оси, но не обязательно направление движения или «нос» объекта. Корабль, повёрнутый на 180°, занимает тот же rectangle, хотя его нос смотрит в обратную сторону. Если нужно directed heading, период π уже недостаточен: нужны отдельная разметка или дополнительные признаки.

<!-- /unit:bc1ff42137ddf49099 -->

<!-- unit:983d91f1f9c1adbb93 -->
<a id="u-983d91f1f9c1adbb93"></a>

DOTA-like строка содержит четыре corners, class и optional difficulty по dataset convention; YOLO-OBB training labels могут хранить нормированные corners. «COCO-style bbox из пяти чисел» — пользовательское расширение, а не автоматически стандартный COCO four-number bbox. Перед конвертацией фиксируют порядок vertices, units и class mapping. Параметризация из разбираемого примера, названная шестипараметрической, но содержащая пять перечисленных чисел, сама противоречива и без точного определения не используется.

<!-- /unit:983d91f1f9c1adbb93 -->

<!-- unit:f998f4ed699c3d81c1 -->
<a id="u-f998f4ed699c3d81c1"></a>

## Преобразование в corners

<!-- /unit:f998f4ed699c3d81c1 -->

<!-- unit:e99cafc1839ade27c1 -->
<a id="u-e99cafc1839ade27c1"></a>

Для θ в радианах и локальных signs $(s,t)\in\{(-1,-1),(1,-1),(1,1),(-1,1)\}$:

<!-- /unit:e99cafc1839ade27c1 -->

<!-- unit:16fecfe94de4ed6edf -->
<a id="u-16fecfe94de4ed6edf"></a>

$$
\begin{aligned}x&=c_x+\frac{sw}{2}\cos\theta-\frac{th}{2}\sin\theta,\\y&=c_y+\frac{sw}{2}\sin\theta+\frac{th}{2}\cos\theta.\end{aligned}
$$

<!-- /unit:16fecfe94de4ed6edf -->

<!-- unit:c16a76ba4e818319a8 -->
<a id="u-c16a76ba4e818319a8"></a>

Это стандартная rotation matrix в согласованной координатной системе. Если image y направлен вниз, визуально положительный поворот выглядит иначе, чем на математической плоскости с y вверх; численный protocol должен быть один.

<!-- /unit:c16a76ba4e818319a8 -->

<!-- unit:67d620ab4b202bce8f -->
<a id="u-67d620ab4b202bce8f"></a>

Пример $centre=\left(0,\,0\right),\qquad w=4,\qquad h=2,\qquad \theta=\frac{\pi}{2}$. Corners до поворота $\left(\left(-2,\,-1\right),\,\left(2,\,-1\right),\,\left(2,\,1\right),\,\left(-2,\,1\right)\right)$, после — $\left(\left(1,\,-2\right),\,\left(1,\,2\right),\,\left(-1,\,2\right),\,\left(-1,\,-2\right)\right)$. Площадь остаётся8. Новый axis-aligned envelope имеет width2,height4.

<!-- /unit:67d620ab4b202bce8f -->

<!-- unit:e1f7b4f52e7c88b8ab -->
<a id="u-e1f7b4f52e7c88b8ab"></a>

Обратное восстановление centre как среднего corners и lengths соседних edges работает для корректного прямоугольника с известным cyclic order. Если GT polygon — произвольный quadrilateral, fitted minimum-area rectangle может изменить shape/area; это approximation, которую нужно явно учитывать.

<!-- /unit:e1f7b4f52e7c88b8ab -->

<!-- unit:e8d283b74ad29c72fc -->
<a id="u-e8d283b74ad29c72fc"></a>

### Обратное преобразование и проверка прямоугольности

<!-- /unit:e8d283b74ad29c72fc -->

<!-- unit:88388a8462c2d83e22 -->
<a id="u-88388a8462c2d83e22"></a>

Для циклических corners p1,p2,p3,p4 прямоугольника центр — их среднее. Векторы e1=p2−p1 и e2=p3−p2 дают соседние стороны, w=||e1||, h=||e2||, θ=atan2(e1_y,e1_x). Проверка $e_1\cdot e_2\approx0$ выявляет отсутствие прямого угла. Противоположные стороны должны быть параллельны и равны по длине. Если входные точки переставлены крест-накрест, формулы соседних сторон больше не описывают rectangle.

<!-- /unit:88388a8462c2d83e22 -->

<!-- unit:7cd4ae47b72e311dde -->
<a id="u-7cd4ae47b72e311dde"></a>

Для source-примера (100,200),(300,200),(300,400),(100,400) получаются center=(200,300),w=h=200,θ=0 в выбранном порядке. В квадрате θ=π/2 с соответствующей перестановкой сторон задаёт ту же область. Для general quadrilateral среднее вершин — удобный reference, но не обязательно area centroid, а lengths двух соседних edges не превращают shape в прямоугольник. Minimum-area fitting — отдельная approximation.

<!-- /unit:7cd4ae47b72e311dde -->

<!-- unit:fb6fa251fbdfba08f6 -->
<a id="u-fb6fa251fbdfba08f6"></a>

При геометрической аугментации безопасно сначала преобразовать corners, затем заново получить целевое представление согласно convention. Например, anisotropic resize по x/y может превратить повёрнутый rectangle в параллелограмм, который уже не является OBB точной формы. Если pipeline снова fit-ит rectangle, он изменяет target; это следует учитывать, а не считать любой resize просто изменением center,w,h при прежнем θ.

<!-- /unit:fb6fa251fbdfba08f6 -->

<!-- unit:f5041749fbb79b674e -->
<a id="u-f5041749fbb79b674e"></a>

## Один rectangle, разные числа

<!-- /unit:f5041749fbb79b674e -->

<!-- unit:7f59cb247d2eb13f08 -->
<a id="u-7f59cb247d2eb13f08"></a>

Геометрически:

<!-- /unit:7f59cb247d2eb13f08 -->

<!-- unit:65ba7b39372446c544 -->
<a id="u-65ba7b39372446c544"></a>

$$
\begin{aligned}(w,h,\theta)&\equiv(w,h,\theta+\pi),\\(w,h,\theta)&\equiv\left(h,w,\theta+\frac\pi2\right).\end{aligned}
$$

<!-- /unit:65ba7b39372446c544 -->

<!-- unit:b352be42ada11e1968 -->
<a id="u-b352be42ada11e1968"></a>

В первом случае w/h не меняются. В исходном объяснении словесные условия этих двух равенств путались. Для ordinary undirected rectangle θ=1° и179° отличаются по orientation только на2°, а raw MSE видит178°.

<!-- /unit:b352be42ada11e1968 -->

<!-- unit:90f125c1171181e5ff -->
<a id="u-90f125c1171181e5ff"></a>

Для period π можно кодировать direction $(\cos2\theta,\sin2\theta)$ или задать cyclic distance. Один smooth loss:

<!-- /unit:90f125c1171181e5ff -->

<!-- unit:46e74c35114a41729c -->
<a id="u-46e74c35114a41729c"></a>

$$
\mathcal{L}_{\mathrm{angle}}=1-\cos\!\left(2(\theta_{\mathrm{pred}}-\theta_{\mathrm{true}})\right).
$$

<!-- /unit:46e74c35114a41729c -->

<!-- unit:c96c25bd9b6bb1e04d -->
<a id="u-c96c25bd9b6bb1e04d"></a>

При true1°,pred179° разность178°, удвоенный угол356°, loss≈0.002436. При true1°,pred91° и **фиксированных** w/h разность90°, loss2. Но если w/h поменялись местами, rectangle тот же, и whole-box representation/loss должен учесть ещё эту equivalence. Одной doubled-angle формулы недостаточно для всех conventions.

<!-- /unit:c96c25bd9b6bb1e04d -->

<!-- unit:e519cf35fb8d1f6249 -->
<a id="u-e519cf35fb8d1f6249"></a>

Canonicalization, например w≥h, уменьшает ambiguity, но вблизи square w≈h малая perturbation может менять выбранную длинную сторону. Angle classification с circular smooth labels задаёт wrap-around соседство bins; ordinary one-hot на180 bins сохраняет жёсткий разрыв между крайними индексами.

<!-- /unit:e519cf35fb8d1f6249 -->

<!-- unit:dc9fe83ed3ca469b77 -->
<a id="u-dc9fe83ed3ca469b77"></a>

### Круговое расстояние и doubled-angle encoding

<!-- /unit:dc9fe83ed3ca469b77 -->

<!-- unit:31e953f4d2f6e3eaa4 -->
<a id="u-31e953f4d2f6e3eaa4"></a>

Для undirected orientation можно взять $d_\pi(\theta_1,\theta_2)=\min_{k\in\mathbb Z}|\theta_1-\theta_2+k\pi|$. Результат лежит между 0 и π/2. В паре 1°/179° ближайшие представители отличаются на 2°, поэтому squared distance должна учитывать именно 2°, а не 178°. Простое ограничение обоих углов диапазоном не убирает разрыв у его границы.

<!-- /unit:31e953f4d2f6e3eaa4 -->

<!-- unit:43eb026a733ada3646 -->
<a id="u-43eb026a733ada3646"></a>

Отображение e(θ)=(cos2θ,sin2θ) одинаково для θ и θ+π. При единичных encoded vectors $\lVert e(\theta_1)-e(\theta_2)\rVert^2=2[1-\cos2(\theta_1-\theta_2)]$. Это связь Euclidean loss по encoding с приведённым cyclic loss. Если network выдаёт произвольные два числа, их длина не обязана быть единицей; normalizing почти нулевой вектор требует epsilon и определённого decoding.

<!-- /unit:43eb026a733ada3646 -->

<!-- unit:2660619eebe749bd20 -->
<a id="u-2660619eebe749bd20"></a>

Для circular labels с 180 one-degree bins истинный угол 0° должен иметь соседями и 1°, и 179°. Например, Gaussian weights вычисляют через circular bin distance, затем нормируют. Обычная label smoothing, одинаково раздающая массу всем чужим bins, не кодирует такую геометрию. CSL меняет target neighborhood, DCL — способ кодирования angle classes; это не безусловное устранение всех неоднозначностей w/h.

<!-- /unit:2660619eebe749bd20 -->

<!-- unit:e1d8f96022312abb49 -->
<a id="u-e1d8f96022312abb49"></a>

Canonical rule w≥h выбирает длинную сторону, но при w=10.001,h=10 малое изменение размеров может поменять выбранную ось на 90°. Geometry изменилась мало, численный angle target — резко. Этот пример объясняет instability square-like объектов без необходимости утверждать, что их orientation физически важна для любой задачи.

<!-- /unit:e1d8f96022312abb49 -->

<!-- unit:ffc3b87b921164815d -->
<a id="u-ffc3b87b921164815d"></a>

## Rotated IoU и surrogate losses

<!-- /unit:ffc3b87b921164815d -->

<!-- unit:9b42f8d2c93c33c3bc -->
<a id="u-9b42f8d2c93c33c3bc"></a>

Формула IoU прежняя, сложнее вычислить intersection polygon. Можно найти пересечение convex polygons, отсортировать vertices и использовать polygon area. Нужно проверять порядок, точность около касания и degenerate cases. Axis-aligned IoU enclosing boxes — другой metric.

<!-- /unit:9b42f8d2c93c33c3bc -->

<!-- unit:d4c018404c45e951df -->
<a id="u-d4c018404c45e951df"></a>

В отличие от неудачной фразы чата, rotated IoU не обязательно лишён derivatives на всех углах. Он piecewise geometry с изменяющимися intersection configurations, сложными/негладкими границами и возможными слабыми signals. Специализированные differentiable implementations и surrogate objectives существуют; exact behavior зависит от formula/kernel.

<!-- /unit:d4c018404c45e951df -->

<!-- unit:c9ee29e03149bb2049 -->
<a id="u-c9ee29e03149bb2049"></a>

Gaussian representation связывает rectangle centre с mean, а dimensions/orientation — с covariance вида $R\operatorname{diag}\!\left(\frac{w^2}{12},\frac{h^2}{12}\right)R^\top$ для uniform-rectangle moment interpretation. KLD/Gaussian Wasserstein/ProbIoU-family losses сравнивают соответствующие distributions. Но Gaussian overlap не равен literal polygon IoU: substitute должен оцениваться по целевой benchmark metric.

<!-- /unit:c9ee29e03149bb2049 -->

<!-- unit:6b4aa57cb1b33a9255 -->
<a id="u-6b4aa57cb1b33a9255"></a>

При square w=h covariance становится isotropic, и направление естественно теряет различимость. Это полезная геометрическая intuition, но разные Gaussian losses имеют разные scales и numerical details. Нельзя объявлять любой «probabilistic IoU» одной универсальной формулой.

<!-- /unit:6b4aa57cb1b33a9255 -->

<!-- unit:f99152a89ef87b34cd -->
<a id="u-f99152a89ef87b34cd"></a>

### Polygon intersection и проверка площади

<!-- /unit:f99152a89ef87b34cd -->

<!-- unit:b04616751f9b697ce6 -->
<a id="u-b04616751f9b697ce6"></a>

Для двух convex rectangles можно clip-ить один polygon последовательно half-planes другого, сохраняя внутренние вершины и точки пересечения рёбер. Итоговый polygon может иметь больше четырёх vertices. Для циклического списка q площадь вычисляется shoelace formula:

<!-- /unit:b04616751f9b697ce6 -->

<!-- unit:dd220eb649aa008004 -->
<a id="u-dd220eb649aa008004"></a>

$$
\mathcal A=\frac12\left|\sum_{i=1}^{n}(x_iy_{i+1}-y_ix_{i+1})\right|,\qquad q_{n+1}=q_1.
$$

<!-- /unit:dd220eb649aa008004 -->

<!-- unit:c45e0377b9a4930b09 -->
<a id="u-c45e0377b9a4930b09"></a>

Дальше IoU использует площади двух исходных rectangles и intersection. Nearly parallel edges, касания и почти нулевая площадь требуют numerical tolerance. Если поменять порядок vertices произвольно, можно получить self-intersection и неверную площадь. Отсутствие простой одной ветви вычисления не равно отсутствию математической функции или всех производных.

<!-- /unit:c45e0377b9a4930b09 -->

<!-- unit:a7074c4ba3c7bd95d9 -->
<a id="u-a7074c4ba3c7bd95d9"></a>

### Gaussian box: от размеров к covariance

<!-- /unit:a7074c4ba3c7bd95d9 -->

<!-- unit:32038151f3709bcde5 -->
<a id="u-32038151f3709bcde5"></a>

У равномерной координаты на [−w/2,w/2] дисперсия w²/12. Для axis-aligned uniform rectangle covariance=diag(w²/12,h²/12); rotation преобразует её как RΣRᵀ. Mean равна center. Именно moment-matching интерпретация объясняет коэффициент 1/12. Формула с w²/4,h²/4 соответствует иному масштабу и не должна незаметно подменять этот mapping.

<!-- /unit:32038151f3709bcde5 -->

<!-- unit:97610285c479877f2f -->
<a id="u-97610285c479877f2f"></a>

При w=4,h=2 и θ=0 eigenvalues равны 4/3 и 1/3. При θ=π/2 они меняются местами на координатных осях, а при θ=π covariance возвращается к исходной. Для w=h обе eigenvalues равны, поэтому covariance одинакова при любых θ. Она описывает симметрию rectangle как распределения, но не сохраняет directed heading.

<!-- /unit:97610285c479877f2f -->

<!-- unit:1d3eb92d02b9712d87 -->
<a id="u-1d3eb92d02b9712d87"></a>

### Исправляем перепутанные ProbIoU и Wasserstein в исходнике

<!-- /unit:1d3eb92d02b9712d87 -->

<!-- unit:23aeac79c1022a1857 -->
<a id="u-23aeac79c1022a1857"></a>

В разбираемом примере squared Gaussian Wasserstein expression была названа ProbIoU, а similarity записана $1-e^{-d^2}$. При совпадении boxes эта величина равна нулю, при удалении растёт к единице: это поведение distance-like loss, а не similarity, которая должна быть максимальна при совпадении.

<!-- /unit:23aeac79c1022a1857 -->

<!-- unit:5c2a11c1fed3f1018d -->
<a id="u-5c2a11c1fed3f1018d"></a>

Для невырожденных Gaussian P,Q один путь к probabilistic similarity идёт через Bhattacharyya distance. С $\overline\Sigma=(\Sigma_P+\Sigma_Q)/2$ и Δμ=μP−μQ:

<!-- /unit:5c2a11c1fed3f1018d -->

<!-- unit:24e0231f0d0d78e554 -->
<a id="u-24e0231f0d0d78e554"></a>

$$
D_B=\frac18\Delta\mu^\top\overline\Sigma^{-1}\Delta\mu+
\frac12\ln\frac{\det\overline\Sigma}{\sqrt{\det\Sigma_P\det\Sigma_Q}}.
$$

<!-- /unit:24e0231f0d0d78e554 -->

<!-- unit:22c20a531130f8c2e9 -->
<a id="u-22c20a531130f8c2e9"></a>

Из него $H=\sqrt{1-e^{-D_B}}$ — Hellinger distance в этой convention, similarity=1−H. При одинаковых Gaussians DB=0,H=0,similarity=1; при больших различиях она уменьшается. Epsilon/clamps и optional penalties влияют на точный code output. Это сохраняет идею исходного разбора, но разделяет разные distances. [ProbIoU](https://arxiv.org/abs/2106.06072), [реализация Ultralytics](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py).

<!-- /unit:22c20a531130f8c2e9 -->

<!-- unit:2de0b1cca20da49b46 -->
<a id="u-2de0b1cca20da49b46"></a>

Если covariance двух boxes одинакова и центры отличаются на Δμ, log-determinant term равен нулю, остаётся quadratic form. Вдоль длинной оси с большей variance один и тот же shift даёт меньший penalty, чем вдоль короткой. Поэтому distribution comparison учитывает shape и orientation, а не только расстояние центров. Но literal polygon IoU по-прежнему другая величина.

<!-- /unit:2de0b1cca20da49b46 -->

<!-- unit:f0f3967e52ffb376b4 -->
<a id="u-f0f3967e52ffb376b4"></a>

Gaussian Wasserstein distance использует иной matrix-square-root term; KL в общем случае асимметрична, поэтому KL(P||Q) не равно KL(Q||P). Назвать все эти функции «Gaussian IoU» и ожидать одинакового scale/gradient нельзя. Выбранный loss сверяют с benchmark rotated IoU, а также с boundary/square-like cases.

<!-- /unit:f0f3967e52ffb376b4 -->

<!-- unit:07d8ce7ab6189e3c2a -->
<a id="u-07d8ce7ab6189e3c2a"></a>

### Архитектурные решения по конкретному ограничению

<!-- /unit:07d8ce7ab6189e3c2a -->

<!-- unit:b08f35a8c7fb05ae5b -->
<a id="u-b08f35a8c7fb05ae5b"></a>

Rotated RPN добавляет orientation в proposal stage; это увеличивает пространство reference shapes, если используются angle anchors. RoI Transformer учит преобразование horizontal regions в oriented regions, чтобы извлекать согласованные признаки. Oriented R-CNN использует собственное proposal/representation design; его нельзя свести к «просто добавить число θ» без учёта assignment и decoding.

<!-- /unit:b08f35a8c7fb05ae5b -->

<!-- unit:032c4c45415167474a -->
<a id="u-032c4c45415167474a"></a>

Gliding Vertex и corner-based methods изменяют output representation: вместо прямой angle regression предсказывают geometric offsets/vertices. Неоднозначность порядка углов остаётся. ReDet вводит rotation-equivariant representations, CSL/DCL меняют angular target encoding. Text detection методы вроде EAST или DB могут выдавать quadrilaterals или polygons и не обязаны решать только задачу прямоугольного OBB.

<!-- /unit:032c4c45415167474a -->

<!-- unit:6ddb765894b126466b -->
<a id="u-6ddb765894b126466b"></a>

YOLO-OBB, MMRotate и rotated Detectron2-сценарии — конкретные системы/реализации, где нужно согласовать angle units, mapping и output interpretation. Обычный RT-DETR checkpoint автоматически не получает angle output только из-за наличия object queries; oriented variant должен быть определён отдельно. Обобщённые заявления о каждой версии и каждом runtime поэтому требуют конкретного artifact.

<!-- /unit:6ddb765894b126466b -->

<!-- unit:57d290e496b4e06c9c -->
<a id="u-57d290e496b4e06c9c"></a>

### Suppression и rotation TTA

<!-- /unit:57d290e496b4e06c9c -->

<!-- unit:250428bf6fb7225011 -->
<a id="u-250428bf6fb7225011"></a>

Если используется NMS, критерий duplicate должен соответствовать oriented geometry или явно выбранной surrogate similarity. Замена polygon overlap на Gaussian similarity меняет решение: одинаковые thresholds не обязаны давать одинаковые keep lists. Даже differentiable similarity не делает жёсткую операцию keep/drop differentiable.

<!-- /unit:250428bf6fb7225011 -->

<!-- unit:ff24e8d200115e6421 -->
<a id="u-ff24e8d200115e6421"></a>

При rotation TTA predictions нужно вернуть inverse affine transform в координаты исходного изображения. Сырые углы 1° и 179° нельзя усреднять как 90° для undirected rectangle. Усреднение doubled-angle vectors даёт направление возле 0° при согласованных w/h; ещё нужно решить correspondence instances. NMS-free inference path уменьшает потребность в suppression, но не устраняет error analysis duplicates/misses в реальном output.

<!-- /unit:ff24e8d200115e6421 -->

<!-- unit:8451df12b17adde9c4 -->
<a id="u-8451df12b17adde9c4"></a>

## Архитектуры и проверка

<!-- /unit:8451df12b17adde9c4 -->

<!-- unit:8e508df71b90a67985 -->
<a id="u-8e508df71b90a67985"></a>

Rotated RPN/RRPN добавляют proposals с ориентацией; RoI Transformer учит переход от horizontal к rotated regions; corner-based methods предсказывают vertices; dense OBB heads добавляют angle/геометрические outputs. YOLO-OBB — task-specific detector, где topology/head/loss нужно сверять по checkpoint, включая boundary fixes конкретной версии.

<!-- /unit:8e508df71b90a67985 -->

<!-- unit:6aa89883625ec7dd51 -->
<a id="u-6aa89883625ec7dd51"></a>

Postprocessing требует rotated duplicate handling, если inference path его использует. Угол в degrees, переданный kernel как radians, даёт корректные shapes и совершенно неверную геометрию. Лучше визуализировать corners **до** обучения, после transforms и после inverse mapping predictions.

<!-- /unit:6aa89883625ec7dd51 -->

<!-- unit:684c8464f36820cce5 -->
<a id="u-684c8464f36820cce5"></a>

Срезы: elongated vs square objects, near angle boundary, crowded small objects, flipped/mirrored inputs. Split по сценам/географии важен для aerial images; соседние overlapping tiles не независимый test.

<!-- /unit:684c8464f36820cce5 -->

<!-- unit:1348c38fb42fe0ae4c -->
<a id="u-1348c38fb42fe0ae4c"></a>

1. Обязательно ли менять w/h при θ→θ+π? **Ответ:** нет; swap соответствует дополнительным π/2.
2. Почему θ одного square неустойчив? **Ответ:** rotation symmetries и близость dimensions делают canonical direction неоднозначным.
3. Можно ли заменить polygon IoU Gaussian similarity? **Ответ:** как training surrogate при проверке, но benchmark definition сама не меняется.

<!-- /unit:1348c38fb42fe0ae4c -->

<!-- unit:cd4ccbda131daaca0e -->
<a id="u-cd4ccbda131daaca0e"></a>

**Практика:** пересчитать rotation example, затем применить inverse matrix и получить original corners. Проверить equality $\left(4,\,2,\,0\right)$ и $\left(2,\,4,\,\frac{\pi}{2}\right)$. Для cyclic loss сравнить1°/179° и1°/91°, явно задав whether w/h swapped.

<!-- /unit:cd4ccbda131daaca0e -->

<!-- unit:f678bb7f1fd3da318f -->
<a id="u-f678bb7f1fd3da318f"></a>

**Схема курса:** image → oriented box parameters → angle canonicalization/periodic loss → rotated geometry and matching → rotated IoU/NMS → angle-aware evaluation

<!-- /unit:f678bb7f1fd3da318f -->

<!-- unit:159df4f6d08c6a7486 -->
<a id="u-159df4f6d08c6a7486"></a>

<!-- latex-display-v1 -->

<!-- /unit:159df4f6d08c6a7486 -->

<!-- unit:b967a8c34e914a397a -->
<a id="u-b967a8c34e914a397a"></a>

<!-- source-depth-S7-Е -->

<!-- /unit:b967a8c34e914a397a -->

## Архитектуры: подробные схемы

### Ориентированный detector

![Ориентированный detector](<../оформление/course/arch-obb.svg>)

Голова задаёт форму рамки, decode преобразует её в corners. Периодичность θ и эквивалентность перестановки сторон учитываются при loss и сравнении; стандартный axis-aligned IoU здесь отвечает другой геометрической задаче.

## Результат главы и переход дальше

Ответьте на вопрос: **Как сравнивать геометрически одинаковые рамки с разными параметрами?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

[Маршрут модуля](<README.md>) · [Происхождение и точные места переноса](<../editorial/coverage.md>)
