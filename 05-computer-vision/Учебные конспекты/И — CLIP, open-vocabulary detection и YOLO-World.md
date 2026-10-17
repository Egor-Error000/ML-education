# И. Open-vocabulary detection: текст задаёт, что искать

[Оглавление](README.md). Пункт 9 плана.

**Основа:** S7, строки 6970–7816: shared image-text space, CLIP, OVD metrics, region methods, GLIP, Grounding DINO и YOLO-World. **Исправления:** open vocabulary не гарантирует любой произвольный объект/язык; global CLIP не локализует автоматически; название «YOLO26-OV» не заменяет проверку конкретной модели. **Углубление:** symmetric contrastive loss, prompt sets, base/novel protocol и threshold diagnosis.

## 1. Что меняется по сравнению с обычным detector

Closed-vocabulary head выдаёт scores для фиксированного списка K classes. В text-conditioned detection классы/phrases подаются на inference, а модель пытается связать regions с их text representations. Output всё ещё содержит localization и confidence, но class meaning задаётся запросом вроде «красный грузовик».

Это не обещание распознать любой термин: нужны выученные visual/text concepts и способность их совместить. Редкий медицинский label, новый инструмент или spatial relation могут быть вне capabilities. Отсутствие обучения **на box annotations novel class** не доказывает, что модель никогда не встречала этот concept в image-text pretraining.

Open-set recognition, OVD и phrase grounding пересекаются, но задачи различны. Open-set может требовать reject unknown; OVD — задавать labels текстом; grounding — найти region для referential phrase. «Красная чашка слева от тарелки» требует relations и выбор среди нескольких instances, что сложнее отдельного class name.

## 2. CLIP: два encoder и общее пространство

Image encoder f_I и text encoder f_T отображают разные inputs в согласованную D-мерную space. После L2 normalization similarity $s_{ij}=v_i^\top t_j$ — cosine. Для batch из N matched image-caption pairs logits=$\frac{s}{\tau}$ с temperature/learned scale.

Image-to-text loss:

$$
\mathcal{L}_I=-\frac1N\sum_{i=1}^{N}\log\frac{\exp(s_{ii}/\tau)}{\sum_{j=1}^{N}\exp(s_{ij}/\tau)}.
$$

Text-to-image аналогично нормируется по images; symmetric CLIP objective усредняет два направления. В чате была приведена одна сторона; для полного классического objective нужна и вторая. [CLIP](https://arxiv.org/abs/2103.00020).

Пример одной строки similarity $\left(0.8,\,0.2\right)$, tau0.1 даёт logits $\left(8,\,2\right)$ и вероятность matched caption≈0.9975. Это probability **среди данных candidates** при данном scale, а не calibrated вероятность, что объект существует. Добавление новых prompt classes меняет denominator и scores. Поэтому threshold нельзя переносить без проверки prompt set/model revision.

Для zero-shot classification encode templates «a photo of a cat», «a photo of a dog» и сравнить image vector с ними. Ensembles templates могут уменьшать чувствительность к формулировке, но требуют осознанного aggregation. Русский/английский synonym не обязан дать одинаковый vector, если text encoder multilingual coverage ограничена.

## 3. Почему CLIP-vector всего изображения недостаточен для boxes

Global score может показать, что image похожа на «собака», но не выдать координаты трёх собак. Для detection нужны region proposals/dense features/queries и localised text-image alignment. Варианты:

- region-based methods вроде ViLD/RegionCLIP используют candidate regions и согласуют их features с text space/teacher;
- GLIP связывает detection и grounding через deeper visual-language fusion;
- Grounding DINO использует text-aware feature enhancement/query selection и cross-modal decoder;
- YOLO-World строит быстрый detector с text-conditioned representations, включая RepVL-PAN и prompt-then-detect подход.

Это механизмы для сравнения, а не вечный ranking моделей по скорости. Точный runtime включает text encoding, image backbone, detection/postprocessing и возможное caching prompts. [YOLO-World](https://arxiv.org/abs/2401.17270).

Если список классов фиксирован на много images, text embeddings иногда можно предвычислить. Это уменьшает repeated text work, но не устраняет необходимость compatible encoders/reparameterization. Смена prompts/encoder может потребовать обновить precomputed state.

## 4. Novel AP и честное значение zero-shot

Разделить **base classes**, по которым доступно объявленное supervised training, и **novel classes**, не использованные в определённом виде training. Измерять AP_base,AP_novel и при необходимости generalized setting со всеми classes вместе. В отчёте указать, какие annotations/text corpora доступны: «unseen» всегда имеет scope.

Detection matching/AP такие же по геометрическому смыслу, как в главе Б, но label mapping задаётся protocol. Синонимы не должны случайно считаться разными mutually exclusive objects; несколько prompt variants одного class могут породить duplicate boxes. Нужно объявить synonym aggregation и postprocessing.

Prompt «vehicle» шире «car». Если GT содержит only cars, detector может правильно найти bicycle по broad prompt и быть засчитан FP в узком protocol. Это несовпадение ontology, а не обязательно visual failure. Для compositional phrases нужны отдельные тесты attributes/relations, а не только class AP.

Проверять blank/unrelated prompts, absent classes, rare/small objects, text language и negative cases. Candidate list может исключить правильную class; обычный softmax всё равно выберет лучшую из неправильных. Поэтому reject/abstention policy требует calibration по задаче.

## 5. Что делать с названиями «YOLO26-OV» из чата

Такое название в источнике не следует автоматически превращать в общеупотребимый model ID с придуманными weights/head. В официальной экосистеме есть YOLO-World и YOLOE с конкретными open-vocabulary возможностей и variants. Точную поддержку YOLO26-based variant следует устанавливать по released checkpoint/config и output contract. [YOLOE](https://docs.ultralytics.com/models/yoloe/).

Это позволяет сохранить учебный смысл исходной темы — локализация по тексту — без выдуманного универсального checkpoint. В плане архитектурное имя трактуется как проверяемый implementation example, а не обязательная новая отдельная теория.

## 6. Пример диагностики и самостоятельные вопросы

Есть image с двумя чашками, красной и синей. На «cup» detector находит обе; на «red cup» тоже обе. Box localisation уже работает, но attribute grounding не различает цвет. Увеличить NMS threshold здесь не исправит text-image alignment. Если красная найдена в region proposals, но rejected classifier, проблема может быть в alignment/threshold; если её нет в proposals — в localisation/scale.

1. Почему одинаковая D image/text vectors не достаточна? **Ответ:** пространства должны быть совместно согласованы objective; произвольные encoders не становятся compatible по размеру.
2. Почему high CLIP probability не existence probability? **Ответ:** зависит от набора candidates и logit scale, нужна separate calibration.
3. Что значит novel class? **Ответ:** отсутствие определённого вида supervision по protocol, а не гарантированное отсутствие concept во всём pretraining.

**Практика:** выбрать пять images с несколькими objects, фиксированный base/novel class list и по три перефразировки. Измерить geometry AP/recall и prompt sensitivity отдельно; добавить absent-object negatives. Для одного failure локализовать этап: proposal, alignment, threshold, ontology или relation understanding. Сохранить exact prompts и checkpoint revision.

<!-- latex-display-v1 -->
