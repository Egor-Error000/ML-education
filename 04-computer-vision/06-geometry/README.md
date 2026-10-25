# Геометрия и 3D

[Модуль «Компьютерное зрение»](<../README.md>) · [Сценарии глав](<План.md>)

- [Камера и геометрия: проекция, координаты и калибровка](<../cv-camera.md>) — основа ветки. Как связать пиксели, координаты камеры и точки сцены?
- [Стереозрение: disparity, cost volume и погрешность глубины](<../cv-stereo.md>) — основа ветки. Как disparity превращается в глубину и почему дальние точки менее точны?
- [Depth: геометрия, масштаб и DPT](<../cv-depth.md>) — основа ветки. Как связать output depth model с геометрией и корректной оценкой масштаба?
- [Feature matching, SfM и SLAM: восстановление сцены и движения](<../cv-sfm.md>) — углубление. Как из нескольких изображений восстановить камеры и разреженную сцену?
- [PointNet и PointNet++: обучение на множествах точек](<../cv-pointnet.md>) — углубление. Как сеть обрабатывает облако точек без заданного порядка?
- [Voxels, PointPillars и BEV: представления для 3D detection](<../cv-bev.md>) — углубление. Как преобразовать 3D measurements в удобное представление для detection?
- [NeRF: лучи, radiance field и volume rendering](<../cv-nerf.md>) — углубление. Как обучить представление сцены по фотографиям и отрисовать новый вид?
- [3D Gaussian Splatting: проекция, compositing и densification](<../cv-gaussians.md>) — углубление. Как явные 3D Gaussian primitives заменяют многократные запросы neural field?

**CV: геометрия и 3D:** [Постановка задачи и паспорт данных](<../../01-data-features/01-reliability/df-task.md>) → [Split, cross-validation и утечки](<../../01-data-features/01-reliability/df-split.md>) → [Dataset, batching и воспроизводимость](<../../01-data-features/01-reliability/df-loading.md>) → [Изображения, аннотации и синхронные преобразования](<../../01-data-features/03-modalities/df-images.md>) → [Изображение, карты признаков и блоки модели](<../cv-input.md>) → [Камера и геометрия: проекция, координаты и калибровка](<../cv-camera.md>) → [Стереозрение: disparity, cost volume и погрешность глубины](<../cv-stereo.md>) → [Классификация, CNN и residual learning](<../cv-classification.md>) → [Depth: геометрия, масштаб и DPT](<../cv-depth.md>) → [Классическое CV: фильтры, границы, морфология, HOG и SIFT](<../cv-classical.md>) → [Feature matching, SfM и SLAM: восстановление сцены и движения](<../cv-sfm.md>) → [PointNet и PointNet++: обучение на множествах точек](<../cv-pointnet.md>) → [Detection: множества объектов, IoU, AP и архитектуры](<../cv-detection.md>) → [Voxels, PointPillars и BEV: представления для 3D detection](<../cv-bev.md>) → [NeRF: лучи, radiance field и volume rendering](<../cv-nerf.md>) → [3D Gaussian Splatting: проекция, compositing и densification](<../cv-gaussians.md>)
