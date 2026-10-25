# Компьютерное зрение

[Все модули](<../README.md>) · [Открыть в читателе](<../Читать конспекты.html#module-cv>) · [План и сценарии](<План.md>)

Общее ядро образует маршрут подраздела. Основа ветки нужна для выбранной задачи; углубление имеет отдельный вход и возврат к продолжению. Предпосылки каждой главы ведут к конкретным материалам.

## Основы и архитектуры

[Оглавление подраздела](<01-foundations/README.md>)

- [Изображение, карты признаков и блоки модели](<cv-input.md>) — общее ядро. Как меняются форма, разрешение и рецептивное поле по пути изображения?
- [Классическое CV: фильтры, границы, морфология, HOG и SIFT](<cv-classical.md>) — углубление. Когда фиксированные признаки полезнее обучения большой сети?
- [Классификация, CNN и residual learning](<cv-classification.md>) — общее ядро. Как изображение превращается в logits класса и что меняет backbone?
- [ResNet: residual-блоки, projection и градиенты](<cv-resnet.md>) — углубление. Как residual-связь меняет вычисление и прохождение градиента?
- [MobileNet и EfficientNet: вычисления при ограниченном бюджете](<cv-efficient.md>) — углубление. Откуда берётся экономия MobileNet и как масштабируется EfficientNet?
- [ConvNeXt: современная свёрточная архитектура](<cv-convnext.md>) — углубление. Какие изменения позволяют CNN использовать идеи дизайна Transformer?
- [Vision Transformer: patches, attention и обучение](<cv-vit.md>) — основа ветки. Как patch projection, позиции и attention преобразуют изображение?
- [Углубление: Swin и иерархические представления](<cv-hierarchy.md>) — углубление. Как окна и patch merging создают пространственную иерархию?

## Обучение и представления

[Оглавление подраздела](<02-representations/README.md>)

- [Обучение и перенос: frozen features, fine-tuning и аугментации](<cv-transfer.md>) — основа ветки. Как выбрать между frozen features и полным fine-tuning?
- [Loss, дисбаланс, пороги и калибровка в CV](<cv-losses.md>) — основа ветки. Как связать функцию потерь, дисбаланс и порог принятия решения?
- [FPN и PAN: многомасштабные представления](<cv-pyramids.md>) — основа ветки. Как передать голове одновременно детали и контекст разных масштабов?
- [SimCLR и MoCo: контрастивное обучение изображений](<cv-contrastive.md>) — углубление. Как обучить encoder на разных видах одного изображения без class labels?
- [Углубление: SSL, MAE, DeiT и DINOv3](<cv-pretraining.md>) — углубление. Как различаются восстановление masked patches и distillation?
- [DINO, DINOv2 и DINOv3: self-distillation и dense features](<cv-dino.md>) — углубление. Как self-distillation учит глобальные и пространственные признаки?
- [Visual retrieval: metric learning и поиск изображений](<cv-retrieval.md>) — основа ветки. Как обучить и оценить поиск похожих изображений?

## Детекция

[Оглавление подраздела](<03-detection/README.md>)

- [Detection: множества объектов, IoU, AP и архитектуры](<cv-detection.md>) — общее ядро. Как отделить ошибки candidates, assignment и post-processing?
- [Faster R-CNN: RPN, RoI и обучение двух стадий](<cv-faster-rcnn.md>) — основа ветки. Как RPN и RoI-head совместно превращают карту признаков в объекты?
- [RetinaNet и FCOS: anchors и dense prediction](<cv-dense-detectors.md>) — основа ветки. Что меняется при переходе от anchors RetinaNet к расстояниям FCOS?
- [YOLOv3 и YOLOv8: головы, assignment и декодирование](<cv-yolo.md>) — основа ветки. Почему YOLOv3 и YOLOv8 нельзя объяснять одной формулой головы?
- [DETR: queries, matching и set prediction](<cv-detr.md>) — основа ветки. Как обучение множества предсказаний устраняет необходимость обычной NMS?
- [Deformable DETR и RT-DETR: внимание и скорость](<cv-deformable.md>) — углубление. Как сократить внимание к изображению и ускорить set prediction?
- [Ориентированные рамки и периодичность угла](<cv-obb.md>) — углубление. Как избежать противоречивых целей для угла ориентированной рамки?

## Сегментация и открытый словарь

[Оглавление подраздела](<04-segmentation/README.md>)

- [Semantic segmentation: FCN, U-Net и DeepLab](<cv-semantic.md>) — общее ядро. Как проверить pixel prediction отдельно от восстановления границ?
- [U-Net и nnU-Net: пространственный decoder и настройка pipeline](<cv-unet.md>) — основа ветки. Как U-Net восстанавливает детали и что автоматизирует nnU-Net?
- [DeepLabv3+ и SegFormer: контекст и декодирование](<cv-segformers.md>) — основа ветки. Как DeepLabv3+ и SegFormer объединяют контекст с пространственными деталями?
- [Instance segmentation: Mask R-CNN и prototype masks](<cv-instance.md>) — общее ядро. Как проверить связь mask prediction с конкретным экземпляром?
- [YOLACT: prototype masks, коэффициенты и crop](<cv-prototypes.md>) — основа ветки. Как получить маску экземпляра без отдельного пространственного decoder на каждый объект?
- [Mask2Former: mask queries и panoptic segmentation](<cv-mask2former.md>) — углубление. Как один набор mask queries решает разные виды segmentation?
- [CLIP, открытый словарь и YOLO-World](<cv-open-vocabulary.md>) — основа ветки. Как разделить text-image alignment, grounding и prompting масок?

## Поза, движение и видео

[Оглавление подраздела](<05-video/README.md>)

- [Pose: keypoints, heatmaps и группировка](<cv-pose.md>) — основа ветки. Как согласовать targets, visibility и координаты keypoints?
- [HRNet, ViTPose и RTMPose: архитектуры pose estimation](<cv-pose-models.md>) — углубление. Чем различаются HRNet, ViTPose и координатная классификация RTMPose?
- [Tracking: Kalman, association и идентичность](<cv-tracking.md>) — основа ветки. Как motion uncertainty и assignment определяют сохранение ID?
- [SORT, DeepSORT, ByteTrack и BoT-SORT: разбор последовательности](<cv-tracking-models.md>) — углубление. Как сохранить ID при падении confidence и пересечении траекторий?
- [Optical flow и RAFT: плотное движение](<cv-flow.md>) — углубление. Как RAFT находит перемещение каждого пикселя между двумя кадрами?
- [Video understanding: 3D CNN, SlowFast, TimeSformer и VideoMAE](<cv-video.md>) — углубление. Как модель использует время и почему выбор кадров входит в постановку?

## Геометрия и 3D

[Оглавление подраздела](<06-geometry/README.md>)

- [Камера и геометрия: проекция, координаты и калибровка](<cv-camera.md>) — основа ветки. Как связать пиксели, координаты камеры и точки сцены?
- [Стереозрение: disparity, cost volume и погрешность глубины](<cv-stereo.md>) — основа ветки. Как disparity превращается в глубину и почему дальние точки менее точны?
- [Depth: геометрия, масштаб и DPT](<cv-depth.md>) — основа ветки. Как связать output depth model с геометрией и корректной оценкой масштаба?
- [Feature matching, SfM и SLAM: восстановление сцены и движения](<cv-sfm.md>) — углубление. Как из нескольких изображений восстановить камеры и разреженную сцену?
- [PointNet и PointNet++: обучение на множествах точек](<cv-pointnet.md>) — углубление. Как сеть обрабатывает облако точек без заданного порядка?
- [Voxels, PointPillars и BEV: представления для 3D detection](<cv-bev.md>) — углубление. Как преобразовать 3D measurements в удобное представление для detection?
- [NeRF: лучи, radiance field и volume rendering](<cv-nerf.md>) — углубление. Как обучить представление сцены по фотографиям и отрисовать новый вид?
- [3D Gaussian Splatting: проекция, compositing и densification](<cv-gaussians.md>) — углубление. Как явные 3D Gaussian primitives заменяют многократные запросы neural field?

## Специализированные задачи

[Оглавление подраздела](<07-specialized/README.md>)

- [CRAFT и DBNet: поиск текста на изображении](<cv-text-detection.md>) — основа ветки. Как обнаруживать текст разной ориентации и формы?
- [CRNN, CTC и TrOCR: распознавание текста](<cv-text-recognition.md>) — основа ветки. Как читать строку без отдельной рамки каждого символа?
- [Document vision: layout, reading order и таблицы](<cv-documents.md>) — основа ветки. Как связать геометрию страницы, OCR и структуру документа?
- [Anomaly detection: reconstruction, PaDiM и PatchCore](<cv-anomaly.md>) — основа ветки. Как искать дефекты, если при обучении есть преимущественно нормальные изделия?
- [Восстановление изображений: denoising, deblurring и super-resolution](<cv-restoration.md>) — основа ветки. Как восстанавливать изображение, не принимая правдоподобные детали за измеренные?
- [Медицинские и спутниковые изображения: данные, объём и масштаб](<cv-domains.md>) — углубление. Какие свойства медицинских и спутниковых данных меняют обычный CV pipeline?

## Генерация изображений

[Оглавление подраздела](<08-generation/README.md>)

- [Autoencoder, VAE и GAN: вероятностные основы генерации](<cv-generative.md>) — основа ветки. Чем различаются восстановление входа и обучение распределения новых изображений?
- [DDPM: прямой процесс, noise prediction и обучение](<cv-ddpm.md>) — основа ветки. Как обучение предсказанию шума превращается в генерацию изображения?
- [DDIM, расписания шума и classifier-free guidance](<cv-sampling.md>) — основа ветки. Как менять число шагов и силу условия, не смешивая sampler с обучением?
- [Latent diffusion: autoencoder, denoiser и conditioning](<cv-latent-diffusion.md>) — основа ветки. Что меняется, когда diffusion работает в пространстве сжатых признаков?
- [DiT и flow matching: Transformer и цели обучения](<cv-dit-flow.md>) — углубление. Почему Transformer-backbone и flow matching являются разными выборами?
- [ControlNet, inpainting, image-to-image и LoRA](<cv-control.md>) — углубление. Как направлять генерацию геометрией, исходным изображением и небольшими обновлениями весов?
- [Оценка генерации: fidelity, diversity, условия и FID](<cv-generation-eval.md>) — основа ветки. Как проверить качество генератора без одного вводящего в заблуждение числа?

## Выбор и эксплуатация решения

[Оглавление подраздела](<09-systems/README.md>)

- [Диагностика CV: ошибки данных, модели и post-processing](<cv-diagnosis.md>) — основа ветки. Как локализовать причину плохого результата до замены архитектуры?
- [Robustness, domain shift, OOD и отказ от ответа](<cv-robustness.md>) — основа ветки. Как обнаружить потерю качества при новой камере или неизвестных объектах?
- [Deployment: экспорт, quantization, distillation и производительность](<cv-deployment.md>) — основа ветки. Как сохранить смысл предсказаний после экспорта и измерить полную стоимость?
- [Сквозные разборы: дефекты, видеопоток и документы](<cv-cases.md>) — основа ветки. Как выбрать и проверить весь CV pipeline под конкретную цель?

## Маршруты

**CV без LLM:** [Постановка задачи и паспорт данных](<../01-data-features/01-reliability/df-task.md>) → [Split, cross-validation и утечки](<../01-data-features/01-reliability/df-split.md>) → [Dataset, batching и воспроизводимость](<../01-data-features/01-reliability/df-loading.md>) → [Изображения, аннотации и синхронные преобразования](<../01-data-features/03-modalities/df-images.md>) → [Изображение, карты признаков и блоки модели](<cv-input.md>) → [Классификация, CNN и residual learning](<cv-classification.md>) → [Detection: множества объектов, IoU, AP и архитектуры](<cv-detection.md>) → [Semantic segmentation: FCN, U-Net и DeepLab](<cv-semantic.md>) → [Instance segmentation: Mask R-CNN и prototype masks](<cv-instance.md>)

**CV: backbone и представления:** [Постановка задачи и паспорт данных](<../01-data-features/01-reliability/df-task.md>) → [Split, cross-validation и утечки](<../01-data-features/01-reliability/df-split.md>) → [Dataset, batching и воспроизводимость](<../01-data-features/01-reliability/df-loading.md>) → [Изображения, аннотации и синхронные преобразования](<../01-data-features/03-modalities/df-images.md>) → [Изображение, карты признаков и блоки модели](<cv-input.md>) → [Классическое CV: фильтры, границы, морфология, HOG и SIFT](<cv-classical.md>) → [Классификация, CNN и residual learning](<cv-classification.md>) → [ResNet: residual-блоки, projection и градиенты](<cv-resnet.md>) → [MobileNet и EfficientNet: вычисления при ограниченном бюджете](<cv-efficient.md>) → [ConvNeXt: современная свёрточная архитектура](<cv-convnext.md>) → [Vision Transformer: patches, attention и обучение](<cv-vit.md>) → [Углубление: Swin и иерархические представления](<cv-hierarchy.md>) → [Обучение и перенос: frozen features, fine-tuning и аугментации](<cv-transfer.md>) → [Loss, дисбаланс, пороги и калибровка в CV](<cv-losses.md>) → [SimCLR и MoCo: контрастивное обучение изображений](<cv-contrastive.md>) → [Углубление: SSL, MAE, DeiT и DINOv3](<cv-pretraining.md>) → [DINO, DINOv2 и DINOv3: self-distillation и dense features](<cv-dino.md>) → [Visual retrieval: metric learning и поиск изображений](<cv-retrieval.md>)

**CV: детекция и архитектуры:** [Постановка задачи и паспорт данных](<../01-data-features/01-reliability/df-task.md>) → [Split, cross-validation и утечки](<../01-data-features/01-reliability/df-split.md>) → [Dataset, batching и воспроизводимость](<../01-data-features/01-reliability/df-loading.md>) → [Изображения, аннотации и синхронные преобразования](<../01-data-features/03-modalities/df-images.md>) → [Изображение, карты признаков и блоки модели](<cv-input.md>) → [Классификация, CNN и residual learning](<cv-classification.md>) → [Detection: множества объектов, IoU, AP и архитектуры](<cv-detection.md>) → [FPN и PAN: многомасштабные представления](<cv-pyramids.md>) → [Faster R-CNN: RPN, RoI и обучение двух стадий](<cv-faster-rcnn.md>) → [RetinaNet и FCOS: anchors и dense prediction](<cv-dense-detectors.md>) → [YOLOv3 и YOLOv8: головы, assignment и декодирование](<cv-yolo.md>) → [Vision Transformer: patches, attention и обучение](<cv-vit.md>) → [DETR: queries, matching и set prediction](<cv-detr.md>) → [Deformable DETR и RT-DETR: внимание и скорость](<cv-deformable.md>) → [Ориентированные рамки и периодичность угла](<cv-obb.md>)

**CV: сегментация и prompting:** [Постановка задачи и паспорт данных](<../01-data-features/01-reliability/df-task.md>) → [Split, cross-validation и утечки](<../01-data-features/01-reliability/df-split.md>) → [Dataset, batching и воспроизводимость](<../01-data-features/01-reliability/df-loading.md>) → [Изображения, аннотации и синхронные преобразования](<../01-data-features/03-modalities/df-images.md>) → [Изображение, карты признаков и блоки модели](<cv-input.md>) → [Классификация, CNN и residual learning](<cv-classification.md>) → [Semantic segmentation: FCN, U-Net и DeepLab](<cv-semantic.md>) → [U-Net и nnU-Net: пространственный decoder и настройка pipeline](<cv-unet.md>) → [Vision Transformer: patches, attention и обучение](<cv-vit.md>) → [DeepLabv3+ и SegFormer: контекст и декодирование](<cv-segformers.md>) → [Detection: множества объектов, IoU, AP и архитектуры](<cv-detection.md>) → [Instance segmentation: Mask R-CNN и prototype masks](<cv-instance.md>) → [YOLACT: prototype masks, коэффициенты и crop](<cv-prototypes.md>) → [DETR: queries, matching и set prediction](<cv-detr.md>) → [Mask2Former: mask queries и panoptic segmentation](<cv-mask2former.md>) → [CLIP, открытый словарь и YOLO-World](<cv-open-vocabulary.md>)

**CV: поза, движение и видео:** [Постановка задачи и паспорт данных](<../01-data-features/01-reliability/df-task.md>) → [Split, cross-validation и утечки](<../01-data-features/01-reliability/df-split.md>) → [Dataset, batching и воспроизводимость](<../01-data-features/01-reliability/df-loading.md>) → [Изображения, аннотации и синхронные преобразования](<../01-data-features/03-modalities/df-images.md>) → [Изображение, карты признаков и блоки модели](<cv-input.md>) → [Классификация, CNN и residual learning](<cv-classification.md>) → [Pose: keypoints, heatmaps и группировка](<cv-pose.md>) → [Vision Transformer: patches, attention и обучение](<cv-vit.md>) → [HRNet, ViTPose и RTMPose: архитектуры pose estimation](<cv-pose-models.md>) → [Detection: множества объектов, IoU, AP и архитектуры](<cv-detection.md>) → [Tracking: Kalman, association и идентичность](<cv-tracking.md>) → [SORT, DeepSORT, ByteTrack и BoT-SORT: разбор последовательности](<cv-tracking-models.md>) → [Optical flow и RAFT: плотное движение](<cv-flow.md>) → [Углубление: SSL, MAE, DeiT и DINOv3](<cv-pretraining.md>) → [Video understanding: 3D CNN, SlowFast, TimeSformer и VideoMAE](<cv-video.md>)

**CV: геометрия и 3D:** [Постановка задачи и паспорт данных](<../01-data-features/01-reliability/df-task.md>) → [Split, cross-validation и утечки](<../01-data-features/01-reliability/df-split.md>) → [Dataset, batching и воспроизводимость](<../01-data-features/01-reliability/df-loading.md>) → [Изображения, аннотации и синхронные преобразования](<../01-data-features/03-modalities/df-images.md>) → [Изображение, карты признаков и блоки модели](<cv-input.md>) → [Камера и геометрия: проекция, координаты и калибровка](<cv-camera.md>) → [Стереозрение: disparity, cost volume и погрешность глубины](<cv-stereo.md>) → [Классификация, CNN и residual learning](<cv-classification.md>) → [Depth: геометрия, масштаб и DPT](<cv-depth.md>) → [Классическое CV: фильтры, границы, морфология, HOG и SIFT](<cv-classical.md>) → [Feature matching, SfM и SLAM: восстановление сцены и движения](<cv-sfm.md>) → [PointNet и PointNet++: обучение на множествах точек](<cv-pointnet.md>) → [Detection: множества объектов, IoU, AP и архитектуры](<cv-detection.md>) → [Voxels, PointPillars и BEV: представления для 3D detection](<cv-bev.md>) → [NeRF: лучи, radiance field и volume rendering](<cv-nerf.md>) → [3D Gaussian Splatting: проекция, compositing и densification](<cv-gaussians.md>)

**CV: OCR и документы:** [Постановка задачи и паспорт данных](<../01-data-features/01-reliability/df-task.md>) → [Split, cross-validation и утечки](<../01-data-features/01-reliability/df-split.md>) → [Dataset, batching и воспроизводимость](<../01-data-features/01-reliability/df-loading.md>) → [Изображения, аннотации и синхронные преобразования](<../01-data-features/03-modalities/df-images.md>) → [Изображение, карты признаков и блоки модели](<cv-input.md>) → [Классификация, CNN и residual learning](<cv-classification.md>) → [Detection: множества объектов, IoU, AP и архитектуры](<cv-detection.md>) → [Semantic segmentation: FCN, U-Net и DeepLab](<cv-semantic.md>) → [CRAFT и DBNet: поиск текста на изображении](<cv-text-detection.md>) → [Vision Transformer: patches, attention и обучение](<cv-vit.md>) → [CRNN, CTC и TrOCR: распознавание текста](<cv-text-recognition.md>) → [Document vision: layout, reading order и таблицы](<cv-documents.md>)

**CV: восстановление и генерация:** [Постановка задачи и паспорт данных](<../01-data-features/01-reliability/df-task.md>) → [Split, cross-validation и утечки](<../01-data-features/01-reliability/df-split.md>) → [Dataset, batching и воспроизводимость](<../01-data-features/01-reliability/df-loading.md>) → [Изображения, аннотации и синхронные преобразования](<../01-data-features/03-modalities/df-images.md>) → [Изображение, карты признаков и блоки модели](<cv-input.md>) → [Классификация, CNN и residual learning](<cv-classification.md>) → [Восстановление изображений: denoising, deblurring и super-resolution](<cv-restoration.md>) → [Autoencoder, VAE и GAN: вероятностные основы генерации](<cv-generative.md>) → [DDPM: прямой процесс, noise prediction и обучение](<cv-ddpm.md>) → [DDIM, расписания шума и classifier-free guidance](<cv-sampling.md>) → [Vision Transformer: patches, attention и обучение](<cv-vit.md>) → [Latent diffusion: autoencoder, denoiser и conditioning](<cv-latent-diffusion.md>) → [DiT и flow matching: Transformer и цели обучения](<cv-dit-flow.md>) → [ControlNet, inpainting, image-to-image и LoRA](<cv-control.md>) → [Оценка генерации: fidelity, diversity, условия и FID](<cv-generation-eval.md>)

**CV: от baseline до применения:** [Постановка задачи и паспорт данных](<../01-data-features/01-reliability/df-task.md>) → [Split, cross-validation и утечки](<../01-data-features/01-reliability/df-split.md>) → [Dataset, batching и воспроизводимость](<../01-data-features/01-reliability/df-loading.md>) → [Изображения, аннотации и синхронные преобразования](<../01-data-features/03-modalities/df-images.md>) → [Изображение, карты признаков и блоки модели](<cv-input.md>) → [Классификация, CNN и residual learning](<cv-classification.md>) → [Обучение и перенос: frozen features, fine-tuning и аугментации](<cv-transfer.md>) → [Anomaly detection: reconstruction, PaDiM и PatchCore](<cv-anomaly.md>) → [Detection: множества объектов, IoU, AP и архитектуры](<cv-detection.md>) → [Tracking: Kalman, association и идентичность](<cv-tracking.md>) → [SORT, DeepSORT, ByteTrack и BoT-SORT: разбор последовательности](<cv-tracking-models.md>) → [Semantic segmentation: FCN, U-Net и DeepLab](<cv-semantic.md>) → [CRAFT и DBNet: поиск текста на изображении](<cv-text-detection.md>) → [Vision Transformer: patches, attention и обучение](<cv-vit.md>) → [CRNN, CTC и TrOCR: распознавание текста](<cv-text-recognition.md>) → [Document vision: layout, reading order и таблицы](<cv-documents.md>) → [Loss, дисбаланс, пороги и калибровка в CV](<cv-losses.md>) → [Диагностика CV: ошибки данных, модели и post-processing](<cv-diagnosis.md>) → [Deployment: экспорт, quantization, distillation и производительность](<cv-deployment.md>) → [Сквозные разборы: дефекты, видеопоток и документы](<cv-cases.md>) → [Robustness, domain shift, OOD и отказ от ответа](<cv-robustness.md>)
