# Feature matching, SfM и SLAM: восстановление сцены и движения

**Уровень:** углубление.

**Предпосылки:** [Камера и геометрия: проекция, координаты и калибровка](<cv-camera.md>), [Классическое CV: фильтры, границы, морфология, HOG и SIFT](<cv-classical.md>)

**Используют:** Самостоятельное применение выбранной ветки.

**Учебный вопрос:** Как из нескольких изображений восстановить камеры и разреженную сцену?

## Схема вычислений

![Feature matching, SfM и SLAM: восстановление сцены и движения: вычислительные этапы](<../оформление/course/expanded-cv-sfm.svg>)

Incremental SfM: Images → keypoints / matching → RANSAC → relative geometry → Triangulate → register cameras → Bundle adjustment → sparse map. SLAM: Новый frame / sensor data → Tracking → current pose → Local map + keyframes → Loop closure → map correction.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Matching не равен reconstruction

Structure-from-Motion, SfM, восстанавливает camera poses и 3D points из наблюдений одной сцены. Начинают с keypoints и descriptors, затем ищут и геометрически проверяют пары. Одна неверная связь между похожими окнами здания способна испортить последующую reconstruction. [COLMAP](https://colmap.github.io/tutorial.html) разделяет feature extraction, matching, sparse reconstruction и dense multi-view stereo.

## От двух видов к общей сцене

Для normalized camera rays $x_1,x_2$ epipolar constraint имеет вид $x_2^TEx_1=0$, где essential matrix $E=[t]_\times R$. Для pixel coordinates используется fundamental matrix, связанная с intrinsics. Robust estimation исключает outliers. Из relative pose выбирают физически допустимую конфигурацию, проверяя положительную глубину points.

Triangulation пересекает наблюдаемые rays в 3D с учётом noise. Малый угол между rays делает глубину плохо обусловленной. Затем incremental SfM добавляет cameras, оценивает pose по уже восстановленным points и расширяет scene. Bundle adjustment, BA, совместно уточняет poses и points, минимизируя reprojection residuals:

$$
L=\sum_{i,j\in\mathcal V_i}\rho(\|\pi(K_i,R_i,t_i,X_j)-u_{ij}\|^2).
$$

Здесь $u_{ij}$ — observed pixel, $\pi$ — projection, $\rho$ — robust penalty. Входные measurements фиксированы; оптимизируются неизвестные параметры сцены и камер. Без фиксирования gauge остаются эквивалентные transformations всей reconstruction.

## SfM и SLAM

SLAM решает localization и mapping последовательно по sensor stream. Tracking оценивает текущую pose, local mapping уточняет структуру, loop closure сопоставляет повторное посещение места и уменьшает drift. Offline SfM может использовать все изображения сразу; online SLAM ограничен уже полученной информацией. Dynamic objects нарушают предположение неподвижной сцены. Monocular reconstruction без дополнительного масштаба имеет ambiguity глобального scale.

## Разобранный пример: reprojection и scale

В camera frame point $(1,0,5)$ при $f_x=500,c_x=320$ проецируется в $u=420$. Observation равна 422, residual prediction-minus-observation $-2$ pixels, squared contribution 4. Для второй observation residual 3 даёт суммарно 13 до robust weighting. Это ошибка изображения; она не задаёт однозначную ошибку в metres.

Если все camera translations и scene points увеличить вдвое, projected ratios $X/Z$ и $Y/Z$ сохраняются. Поэтому обычные monocular images не отличают модель комнаты в metres от модели в удвоенных единицах. Известная baseline, размер объекта или calibrated inertial constraints могут добавить scale information, но должны присутствовать в постановке.

## Диагностика

Отслеживают inlier count, triangulation angles, reprojection errors и coverage, а не только число points. Сцена из одной плоскости или почти чистое вращение затрудняет восстановление глубины. Плохая calibration может компенсироваться ошибочной геометрией. Reconstruction train views не означает корректных unseen views.

## Самопроверка с ответами

**Практика:** оставьте только поворот камеры вокруг её центра. Rays наблюдают сцену без полезной translation baseline; надёжная triangulation глубины невозможна в обычной схеме.

**Почему низкий BA loss не доказывает правильный metric scale?** Projection инвариантна к общему scale в monocular setup.

**Что добавляет loop closure?** Связи между удалёнными во времени участками trajectory, позволяющие скорректировать накопленный drift.

## Источники и продолжение

- [COLMAP tutorial — authors documentation](https://colmap.github.io/tutorial.html)

[Выбрать следующую ветку](<README.md>) · [Сценарии подраздела](<06-geometry/План.md>)
