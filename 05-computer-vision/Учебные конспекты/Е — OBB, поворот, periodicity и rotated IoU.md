# Е. Oriented boxes: геометрия, угол и неоднозначность

[Оглавление](README.md). Пункт 6 плана.

**Основа:** S7, строки 4524–5276. Сохранены representations OBB, corners, rotated IoU, periodicity, Gaussian losses и YOLO-OBB. **Исправления:** axis-aligned box применим, но может быть неэффективен; rotated IoU не «недифференцируем по углу везде»; произвольный четырёхугольник нельзя без потерь заменить пятью параметрами прямоугольника. **Углубление:** ручной поворот и cyclic loss.

## 1. Почему повёрнутый прямоугольник полезен

Для длинного корабля под углом axis-aligned enclosing box захватывает много background и может перекрывать соседний объект. OBB задаёт centre $(c_x,c_y)$, размеры w/h и θ. Это плотнее описывает ориентированный объект, но не заменяет точную segmentation mask.

Axis-aligned detection не становится математически «неприменимой»: это другой, иногда достаточный output. Польза OBB должна соответствовать задаче и разметке. Для почти круглого объекта угол может не иметь устойчивого смысла; для квадрата часть углов эквивалентна, и annotation noise возрастает.

Annotation может хранить четыре corners, а model head — внутренние $(c_x,c_y,w,h,\theta)$. При conversion нужны порядок corners, единицы угла, clockwise/counterclockwise convention и направление image y. Library angle ranges различаются; нельзя копировать старое «OpenCV всегда имеет диапазон…» без версии/конвенции.

## 2. Преобразование в corners

Для θ в радианах и локальных signs $(s,t)\in\{(-1,-1),(1,-1),(1,1),(-1,1)\}$:

$$
\begin{aligned}x&=c_x+\frac{sw}{2}\cos\theta-\frac{th}{2}\sin\theta,\\y&=c_y+\frac{sw}{2}\sin\theta+\frac{th}{2}\cos\theta.\end{aligned}
$$

Это стандартная rotation matrix в согласованной координатной системе. Если image y направлен вниз, визуально положительный поворот выглядит иначе, чем на математической плоскости с y вверх; численный protocol должен быть один.

Пример $centre=\left(0,\,0\right),\qquad w=4,\qquad h=2,\qquad \theta=\frac{\pi}{2}$. Corners до поворота $\left(\left(-2,\,-1\right),\,\left(2,\,-1\right),\,\left(2,\,1\right),\,\left(-2,\,1\right)\right)$, после — $\left(\left(1,\,-2\right),\,\left(1,\,2\right),\,\left(-1,\,2\right),\,\left(-1,\,-2\right)\right)$. Площадь остаётся8. Новый axis-aligned envelope имеет width2,height4.

Обратное восстановление centre как среднего corners и lengths соседних edges работает для корректного прямоугольника с известным cyclic order. Если GT polygon — произвольный quadrilateral, fitted minimum-area rectangle может изменить shape/area; это approximation, которую нужно явно учитывать.

## 3. Один rectangle, разные числа

Геометрически:

$$
\begin{aligned}(w,h,\theta)&\equiv(w,h,\theta+\pi),\\(w,h,\theta)&\equiv\left(h,w,\theta+\frac\pi2\right).\end{aligned}
$$

В первом случае w/h не меняются. В исходном объяснении словесные условия этих двух равенств путались. Для ordinary undirected rectangle θ=1° и179° отличаются по orientation только на2°, а raw MSE видит178°.

Для period π можно кодировать direction $(\cos2\theta,\sin2\theta)$ или задать cyclic distance. Один smooth loss:

$$
\mathcal{L}_{\mathrm{angle}}=1-\cos\!\left(2(\theta_{\mathrm{pred}}-\theta_{\mathrm{true}})\right).
$$

При true1°,pred179° разность178°, удвоенный угол356°, loss≈0.002436. При true1°,pred91° и **фиксированных** w/h разность90°, loss2. Но если w/h поменялись местами, rectangle тот же, и whole-box representation/loss должен учесть ещё эту equivalence. Одной doubled-angle формулы недостаточно для всех conventions.

Canonicalization, например w≥h, уменьшает ambiguity, но вблизи square w≈h малая perturbation может менять выбранную длинную сторону. Angle classification с circular smooth labels задаёт wrap-around соседство bins; ordinary one-hot на180 bins сохраняет жёсткий разрыв между крайними индексами.

## 4. Rotated IoU и surrogate losses

Формула IoU прежняя, сложнее вычислить intersection polygon. Можно найти пересечение convex polygons, отсортировать vertices и использовать polygon area. Нужно проверять порядок, точность около касания и degenerate cases. Axis-aligned IoU enclosing boxes — другой metric.

В отличие от неудачной фразы чата, rotated IoU не обязательно лишён derivatives на всех углах. Он piecewise geometry с изменяющимися intersection configurations, сложными/негладкими границами и возможными слабыми signals. Специализированные differentiable implementations и surrogate objectives существуют; exact behavior зависит от formula/kernel.

Gaussian representation связывает rectangle centre с mean, а dimensions/orientation — с covariance вида $R\operatorname{diag}\!\left(\frac{w^2}{12},\frac{h^2}{12}\right)R^\top$ для uniform-rectangle moment interpretation. KLD/Gaussian Wasserstein/ProbIoU-family losses сравнивают соответствующие distributions. Но Gaussian overlap не равен literal polygon IoU: substitute должен оцениваться по целевой benchmark metric.

При square w=h covariance становится isotropic, и направление естественно теряет различимость. Это полезная геометрическая intuition, но разные Gaussian losses имеют разные scales и numerical details. Нельзя объявлять любой «probabilistic IoU» одной универсальной формулой.

## 5. Архитектуры и проверка

Rotated RPN/RRPN добавляют proposals с ориентацией; RoI Transformer учит переход от horizontal к rotated regions; corner-based methods предсказывают vertices; dense OBB heads добавляют angle/геометрические outputs. YOLO-OBB — task-specific detector, где topology/head/loss нужно сверять по checkpoint, включая boundary fixes конкретной версии.

Postprocessing требует rotated duplicate handling, если inference path его использует. Угол в degrees, переданный kernel как radians, даёт корректные shapes и совершенно неверную геометрию. Лучше визуализировать corners **до** обучения, после transforms и после inverse mapping predictions.

Срезы: elongated vs square objects, near angle boundary, crowded small objects, flipped/mirrored inputs. Split по сценам/географии важен для aerial images; соседние overlapping tiles не независимый test.

1. Обязательно ли менять w/h при θ→θ+π? **Ответ:** нет; swap соответствует дополнительным π/2.
2. Почему θ одного square неустойчив? **Ответ:** rotation symmetries и близость dimensions делают canonical direction неоднозначным.
3. Можно ли заменить polygon IoU Gaussian similarity? **Ответ:** как training surrogate при проверке, но benchmark definition сама не меняется.

**Практика:** пересчитать rotation example, затем применить inverse matrix и получить original corners. Проверить equality $\left(4,\,2,\,0\right)$ и $\left(2,\,4,\,\frac{\pi}{2}\right)$. Для cyclic loss сравнить1°/179° и1°/91°, явно задав whether w/h swapped.

<!-- latex-display-v1 -->
