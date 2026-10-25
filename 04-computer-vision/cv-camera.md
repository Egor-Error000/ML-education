# Камера и геометрия: проекция, координаты и калибровка

**Уровень:** основа ветки.

**Предпосылки:** [Изображение, карты признаков и блоки модели](<cv-input.md>)

**Используют:** [Стереозрение: disparity, cost volume и погрешность глубины](<cv-stereo.md>), [Depth: геометрия, масштаб и DPT](<cv-depth.md>), [Feature matching, SfM и SLAM: восстановление сцены и движения](<cv-sfm.md>), [PointNet и PointNet++: обучение на множествах точек](<cv-pointnet.md>), [NeRF: лучи, radiance field и volume rendering](<cv-nerf.md>), [Медицинские и спутниковые изображения: данные, объём и масштаб](<cv-domains.md>)

**Учебный вопрос:** Как связать пиксели, координаты камеры и точки сцены?

## Схема вычислений

![Камера и геометрия: проекция, координаты и калибровка: вычислительные этапы](<../оформление/course/expanded-cv-camera.svg>)

Projection: World point Xw: metres → Extrinsics: Xc = R Xw + t → Perspective divide X/Z,Y/Z → Intrinsics K → pixel u,v. Backprojection: Pixel + axial depth Z → K inverse → camera ray → Умножение на depth Z → Inverse extrinsics → world.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Три системы координат

Точка сцены $X_w$ задана в world coordinates. Extrinsics переводят её в camera coordinates: $X_c=RX_w+t$, где $R$ — rotation, $t$ — translation в выбранных единицах длины. Intrinsics задают проекцию: $u=f_xX_c/Z_c+c_x$, $v=f_yY_c/Z_c+c_y$. Здесь $f_x,f_y$ измеряются в pixels, $(c_x,c_y)$ — principal point. Это pinhole-модель без distortion; реальные lenses часто требуют дополнительных параметров.

## Матрица и обратное преобразование

В homogeneous coordinates $s[u,v,1]^T=K[R|t][X_w,Y_w,Z_w,1]^T$. Масштаб $s$ связан с глубиной, а не произвольной class confidence. Camera center в world coordinates равен $-R^Tt$, а не $t$. Для обратной проекции нужен depth: $X_c=Z_cK^{-1}[u,v,1]^T$, затем $X_w=R^T(X_c-t)$. Один pixel без глубины задаёт луч, а не единственную 3D-точку.

Calibration оценивает intrinsics и distortion по известной геометрии и observations, например шахматной доске. Reprojection error измеряет согласование projected points и observed pixels. Малое среднее не гарантирует точности на краях кадра, если они плохо покрыты калибровочными видами. Модель и процедуру описывает [документация OpenCV](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html).

## Resize, crop и глубина

При resize с коэффициентами $s_x,s_y$ меняются $f_x,c_x$ и $f_y,c_y$ соответственно; pixel-center convention может добавить сдвиг. Crop с левым верхним углом $(a,b)$ даёт $c'_x=c_x-a$, $c'_y=c_y-b$. Extrinsics от обычного crop не меняются. Axial depth $Z_c$ и range $\|X_c\|$ отличаются вне optical axis. Ориентации осей, handedness и порядок transforms фиксируют до объединения sensors.

## Разобранный пример: projection и backprojection

Пусть $f_x=f_y=800$, $(c_x,c_y)=(320,240)$, point в camera coordinates $(1,0.5,4)$ metres. Тогда $u=800/4+320=520$, $v=800\cdot0.5/4+240=340$. Обратный путь: $X=(520-320)\cdot4/800=1$, $Y=(340-240)\cdot4/800=0.5$.

Range равен $\sqrt{1^2+0.5^2+4^2}=\sqrt{17.25}\approx4.15331$ m, хотя depth $Z=4$ m. После resize вдвое intrinsics становятся $(400,400,160,120)$, point проецируется в $(260,170)$. Если сохранить прежнюю $K$, восстановленная геометрия будет неверной.

## Ошибки применения

Translation в миллиметрах вместе с depth в метрах создаёт тысячекратную ошибку. Неправильное направление world-to-camera меняет позу, даже если matrix shape верна. Distortion следует исправлять согласованно с новой $K$. Calibration по test scenes и использование будущих camera poses при online evaluation могут нарушать постановку.

## Самопроверка с ответами

**Практика:** crop исходного примера начинается в $(100,50)$. Новый principal point $(220,190)$, projected pixel $(420,290)$; camera coordinates point прежние.

**Почему нельзя получить метры из одного pixel?** Все точки луча проектируются в одну позицию без дополнительной глубины или ограничений сцены.

**Является ли $t$ положением камеры?** При $X_c=RX_w+t$ положение камеры равно $-R^Tt$.

## Источники и продолжение

- [OpenCV camera calibration reference](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)

[Следующие углубления: cv-stereo, cv-depth, cv-sfm, cv-pointnet, cv-nerf, cv-domains](<README.md>) · [Сценарии подраздела](<06-geometry/План.md>)
