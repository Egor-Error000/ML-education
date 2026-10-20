# Компьютерное зрение

[Все модули](<../README.md>) · [Открыть в читателе](<../Читать конспекты.html#module-cv>) · [План и сценарии](<План.md>)

Общее ядро образует маршрут подраздела. Основа ветки нужна для выбранной задачи; углубление имеет отдельный вход и возврат к продолжению. Предпосылки каждой главы ведут к конкретным материалам.

## Основы и задачи

- [Изображение, карты признаков и блоки модели](<cv-input.md>) — общее ядро. Как меняются форма, разрешение и рецептивное поле по пути изображения?
- [Классификация, CNN и residual learning](<cv-classification.md>) — общее ядро. Как изображение превращается в logits класса и что меняет backbone?
- [Vision Transformer: patches, attention и обучение](<cv-vit.md>) — основа ветки. Как patch projection, позиции и attention преобразуют изображение?
- [Углубление: Swin и иерархические представления](<cv-hierarchy.md>) — углубление. Как окна и patch merging создают пространственную иерархию?
- [Углубление: SSL, MAE, DeiT и DINOv3](<cv-pretraining.md>) — углубление. Как получить и проверить представление без полного набора task labels?
- [Detection: множества объектов, IoU, AP и архитектуры](<cv-detection.md>) — общее ядро. Как получить множество рамок и оценить его с учётом matching?
- [Semantic segmentation: FCN, U-Net и DeepLab](<cv-semantic.md>) — общее ядро. Как предсказывать класс пикселя и восстанавливать пространственную сетку?
- [Instance segmentation: Mask R-CNN и prototype masks](<cv-instance.md>) — общее ядро. Как отличить экземпляры одного класса и получить маску каждого?
- [Pose: keypoints, heatmaps и группировка](<cv-pose.md>) — основа ветки. Как связать положение ключевых точек с конкретным объектом?
- [Ориентированные рамки и периодичность угла](<cv-obb.md>) — основа ветки. Как сравнивать геометрически одинаковые рамки с разными параметрами?
- [Tracking: Kalman, association и идентичность](<cv-tracking.md>) — основа ветки. Как сохранить идентичность объекта между кадрами?
- [Depth: геометрия, масштаб и DPT](<cv-depth.md>) — основа ветки. Что можно восстановить из одного изображения и какие допущения нужны для масштаба?
- [CLIP, открытый словарь и YOLO-World](<cv-open-vocabulary.md>) — основа ветки. Как связать текстовое название с локализованным объектом?

## Маршруты

**CV без LLM:** [Постановка задачи и паспорт данных](<../01-data-features/01-reliability/df-task.md>) → [Split, cross-validation и утечки](<../01-data-features/01-reliability/df-split.md>) → [Dataset, batching и воспроизводимость](<../01-data-features/01-reliability/df-loading.md>) → [Изображения, аннотации и синхронные преобразования](<../01-data-features/03-modalities/df-images.md>) → [Изображение, карты признаков и блоки модели](<cv-input.md>) → [Классификация, CNN и residual learning](<cv-classification.md>) → [Detection: множества объектов, IoU, AP и архитектуры](<cv-detection.md>) → [Semantic segmentation: FCN, U-Net и DeepLab](<cv-semantic.md>) → [Instance segmentation: Mask R-CNN и prototype masks](<cv-instance.md>)
