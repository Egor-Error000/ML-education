# Стереозрение: disparity, cost volume и погрешность глубины

**Уровень:** основа ветки.

**Предпосылки:** [Камера и геометрия: проекция, координаты и калибровка](<cv-camera.md>)

**Используют:** Самостоятельное применение выбранной ветки.

**Учебный вопрос:** Как disparity превращается в глубину и почему дальние точки менее точны?

## Схема вычислений

![Стереозрение: disparity, cost volume и погрешность глубины: вычислительные этапы](<../оформление/course/expanded-cv-stereo.svg>)

Stereo geometry: Calibrated left / right → Rectification → common rows → Match → disparity d pixels → Z = f B / d metres. PSMNet: Shared image features + SPP → Disparity cost volume → 3D CNN regularization → Soft argmin → disparity.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Геометрия rectified пары

У rectified stereo cameras соответствующие точки лежат на одной horizontal scanline. Для параллельной модели с одинаковым focal length $f$ и baseline $B$ disparity $d=u_L-u_R=fB/Z$. Следовательно, $Z=fB/d$. $f$ и $d$ измеряются в pixels, $B$ задаёт единицы глубины. Формула предполагает согласованные principal points; для общего случая требуется учитывать calibration и rectification.

## От matching cost к cost volume

Для каждой позиции левого изображения проверяют возможные disparities. Cost volume хранит качество соответствия $C(d,u,v)$. Классические методы используют photometric или descriptor distance и regularization. В learning pipeline shared encoder извлекает features; concat или correlation собирает volume, а сеть агрегирует spatial и disparity context. [PSMNet](https://arxiv.org/abs/1803.08669) использует spatial pyramid pooling и 3D CNN aggregation.

Выход по disparities можно преобразовать softmax от отрицательных costs, затем soft argmin $\hat d=\sum_dd\,p(d)$. Это среднее distribution, а не обязательно один достоверный correspondence. Два peaks на разных поверхностях могут дать disparity, не соответствующую ни одной. Для supervised training нужны valid disparity masks и согласованные единицы; maximum disparity ограничивает представимый диапазон.

## Разобранный пример: глубина и неопределённость

Пусть $f=800$ pixels, $B=0.12$ m. При $d=40$ pixels глубина $96/40=2.4$ m. Ошибка $\Delta d=1$ pixel даёт локальную оценку $|\Delta Z|\approx fB\,|\Delta d|/d^2=96/1600=0.06$ m. При $d=4$ получаем $Z=24$ m и линейную оценку 6 m. Точное значение при $d=5$ равно 19.2 m: для большой относительной ошибки linear approximation уже неточна.

Distribution по disparities 2,3,4 имеет probabilities 0.2,0.5,0.3. Soft argmin даёт $0.4+1.5+1.2=3.1$ pixels. При $fB=96$ глубина 30.96774 m. Среднее depth по трём hypotheses было бы $0.2\cdot48+0.5\cdot32+0.3\cdot24=32.8$ m. Из-за нелинейности inverse disparity эти усреднения различаются.

Для volume $D=192,H=128,W=256$ нужно 6291456 scalar costs, 24 MiB float32. Feature concatenation с дополнительным channel dimension увеличивает память ещё сильнее; 3D CNN добавляет activations и gradient buffers.

## Сложные случаи

Однотонные области, повторяющийся узор, отражения и occlusion дают неоднозначные или отсутствующие correspondences. Left-right consistency помогает находить ошибки, но не доказывает истинность всех оставшихся depth values. Resize требует изменения $f$ и disparity scale. Слишком большой baseline улучшает triangulation дальних точек, но увеличивает различия видов и occlusions.

## Самопроверка с ответами

**Практика:** уменьшите обе картинки вдвое. $f$ и disparity уменьшатся вдвое, $B$ прежний, правильная глубина сохранится.

**Почему disparity 0 нельзя просто заменить глубиной 0?** В идеальной модели это бесконечная дальность, а в данных часто invalid correspondence.

**Создаёт ли 3D CNN метрическую шкалу сама?** Нет. Шкалу задают calibration и baseline; сеть оценивает correspondence.

## Иллюстрации из первоисточников

![stereo: Figure 1 из авторской статьи](<../оформление/course/papers/stereo.png>)

**Источник:** [stereo, Figure 1](https://arxiv.org/abs/1803.08669), PDF, с. 4. Проследите вход, преобразования и выход по стрелкам. Сопоставьте обозначения статьи с формами и targets, определёнными в тексте главы.

## Источники и продолжение

- [PSMNet — Chang, Chen, CVPR 2018](https://arxiv.org/abs/1803.08669)
- [OpenCV calibration and stereo geometry](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)

[Выбрать следующую ветку](<README.md>) · [Сценарии подраздела](<06-geometry/План.md>)
