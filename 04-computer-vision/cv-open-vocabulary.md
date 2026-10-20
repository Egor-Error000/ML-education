# CLIP, открытый словарь и YOLO-World

**Уровень:** основа ветки.

**Предпосылки:** [Изображение, карты признаков и блоки модели](<cv-input.md>), [Классификация, CNN и residual learning](<cv-classification.md>), [Detection: множества объектов, IoU, AP и архитектуры](<cv-detection.md>)

**Используют:** Выбранная прикладная задача и итоговая проверка результата.

**Учебный вопрос:** Как связать текстовое название с локализованным объектом?





## Карта вычисления и границы этапов

![CLIP, открытый словарь и YOLO-World: входы, преобразования и выходы](<../оформление/course/cv-open-vocabulary.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Сравните формы и смысл каждого перехода с формулами главы. Выход преобразования и доказательство его качества — разные объекты: последняя проверка использует отдельный протокол оценки.

<!-- unit:143f80b25dad4bf034 -->
<a id="u-143f80b25dad4bf034"></a>

**Схема курса:** image → visual encoder/region features; text prompt → text encoder; aligned scores → grounded box candidates → threshold/association → optional mask model; evaluation keeps classification, localization, mask and prompt sensitivity separate.

<!-- /unit:143f80b25dad4bf034 -->

<!-- unit:cccf9ccb7f61e03807 -->
<a id="u-cccf9ccb7f61e03807"></a>

## Что меняется по сравнению с обычным detector

<!-- /unit:cccf9ccb7f61e03807 -->

<!-- unit:1843e8bc63b46216d1 -->
<a id="u-1843e8bc63b46216d1"></a>

Closed-vocabulary head выдаёт scores для фиксированного списка K classes. В text-conditioned detection классы/phrases подаются на inference, а модель пытается связать regions с их text representations. Output всё ещё содержит localization и confidence, но class meaning задаётся запросом вроде «красный грузовик».

<!-- /unit:1843e8bc63b46216d1 -->

<!-- unit:b53082bacc6a5d7fde -->
<a id="u-b53082bacc6a5d7fde"></a>

Это не обещание распознать любой термин: нужны выученные visual/text concepts и способность их совместить. Редкий медицинский label, новый инструмент или spatial relation могут быть вне capabilities. Отсутствие обучения **на box annotations novel class** не доказывает, что модель никогда не встречала этот concept в image-text pretraining.

<!-- /unit:b53082bacc6a5d7fde -->

<!-- unit:f2a3df8c6555f31742 -->
<a id="u-f2a3df8c6555f31742"></a>

Open-set recognition, OVD и phrase grounding пересекаются, но задачи различны. Open-set может требовать reject unknown; OVD — задавать labels текстом; grounding — найти region для referential phrase. «Красная чашка слева от тарелки» требует relations и выбор среди нескольких instances, что сложнее отдельного class name.

<!-- /unit:f2a3df8c6555f31742 -->

<!-- unit:829ada52b4a92ee154 -->
<a id="u-829ada52b4a92ee154"></a>

### Что действительно открывается при смене текстового словаря

<!-- /unit:829ada52b4a92ee154 -->

<!-- unit:b20acf633e4ae47f45 -->
<a id="u-b20acf633e4ae47f45"></a>

Обычный classifier detector имеет K выходных class channels с обученными значениями. Добавить строку «редкий инструмент» в пользовательский интерфейс не создаёт новый channel с нужными весами. В text-conditioned detector class representation можно вычислить text encoder, поэтому словарь inputs меняется без обязательного переобучения всей head. Однако визуальные признаки и соответствие тексту должны уже поддерживать концепт; новый текст не создаёт отсутствующее знание объекта.

<!-- /unit:b20acf633e4ae47f45 -->

<!-- unit:ecd5787d9230494369 -->
<a id="u-ecd5787d9230494369"></a>

В примере «обычный COCO detector не найдёт красную чашку» нужно разделить две части. Он может найти cup, если этот класс поддержан; требование отличить именно красную от синей — дополнительное attribute grounding. Текстовый интерфейс удобнее выражает этот фильтр, но модель всё равно может проигнорировать adjective. Поэтому наличие natural-language input не подтверждает compositional understanding.

<!-- /unit:ecd5787d9230494369 -->

<!-- unit:9d91a9c886ddbf6b2a -->
<a id="u-9d91a9c886ddbf6b2a"></a>

Novel определяется видом отсутствующей supervision. Концепт мог встречаться в captions, image-level labels или pretrained teacher, хотя box annotations novel class не использовались на этапе detector training. Называть это «никогда не видела объект» нельзя. Cross-dataset transfer, class-held-out zero-shot и few-shot с несколькими novel examples — разные эксперименты, у каждого нужен собственный train/test contract.

<!-- /unit:9d91a9c886ddbf6b2a -->

<!-- unit:1cf6625b360f5c12cf -->
<a id="u-1cf6625b360f5c12cf"></a>

В робототехнике найденная box ещё не задаёт grasp pose, в автоматической разметке — не становится проверенной GT, в поиске по видео — не обеспечивает persistent ID. OVD решает локализацию по запросу как компонент более широкой системы. Для каждого применения выбирают geometry, rejection и последующий этап, сохраняя неопределённость text-region matching.

<!-- /unit:1cf6625b360f5c12cf -->

<!-- unit:0e608626eb8ea75fc4 -->
<a id="u-0e608626eb8ea75fc4"></a>

## CLIP: два encoder и общее пространство

<!-- /unit:0e608626eb8ea75fc4 -->

<!-- unit:039ec27948594abc8b -->
<a id="u-039ec27948594abc8b"></a>

Image encoder f_I и text encoder f_T отображают разные inputs в согласованную D-мерную space. После L2 normalization similarity $s_{ij}=v_i^\top t_j$ — cosine. Для batch из N matched image-caption pairs logits=$\frac{s}{\tau}$ с temperature/learned scale.

<!-- /unit:039ec27948594abc8b -->

<!-- unit:516a2838274b78faa7 -->
<a id="u-516a2838274b78faa7"></a>

Image-to-text loss:

<!-- /unit:516a2838274b78faa7 -->

<!-- unit:9b97c3f903a7c6c950 -->
<a id="u-9b97c3f903a7c6c950"></a>

$$
\mathcal{L}_I=-\frac1N\sum_{i=1}^{N}\log\frac{\exp(s_{ii}/\tau)}{\sum_{j=1}^{N}\exp(s_{ij}/\tau)}.
$$

<!-- /unit:9b97c3f903a7c6c950 -->

<!-- unit:54ba43478540dbb65f -->
<a id="u-54ba43478540dbb65f"></a>

Text-to-image аналогично нормируется по images; symmetric CLIP objective усредняет два направления. В разбираемом примере была приведена одна сторона; для полного классического objective нужна и вторая. [CLIP](https://arxiv.org/abs/2103.00020).

<!-- /unit:54ba43478540dbb65f -->

<!-- unit:d175986a443ad640a5 -->
<a id="u-d175986a443ad640a5"></a>

Пример одной строки similarity $\left(0.8,\,0.2\right)$, tau0.1 даёт logits $\left(8,\,2\right)$ и вероятность matched caption≈0.9975. Это probability **среди данных candidates** при данном scale, а не calibrated вероятность, что объект существует. Добавление новых prompt classes меняет denominator и scores. Поэтому threshold нельзя переносить без проверки prompt set/model revision.

<!-- /unit:d175986a443ad640a5 -->

<!-- unit:9ac0b0257108fad2b7 -->
<a id="u-9ac0b0257108fad2b7"></a>

Для zero-shot classification encode templates «a photo of a cat», «a photo of a dog» и сравнить image vector с ними. Ensembles templates могут уменьшать чувствительность к формулировке, но требуют осознанного aggregation. Русский/английский synonym не обязан дать одинаковый vector, если text encoder multilingual coverage ограничена.

<!-- /unit:9ac0b0257108fad2b7 -->

<!-- unit:a027943a83336a294f -->
<a id="u-a027943a83336a294f"></a>

### Полная матрица image-text matching и оба направления loss

<!-- /unit:a027943a83336a294f -->

<!-- unit:41a69b6323956dadbb -->
<a id="u-41a69b6323956dadbb"></a>

Возьмём две matched пары, cosine matrix $S=[[0.8,0.2],[0.1,0.7]]$ и temperature0.1. Logits равны [[8,2],[1,7]]. Image-to-text normalizes каждую строку: обе правильные probabilities равны $1/(1+e^{-6})\approx0.997527$, loss≈0.002476. Text-to-image normalizes столбцы: для первого text правильная probability $1/(1+e^{-7})\approx0.999089$, для второго — $1/(1+e^{-5})\approx0.993307$, средняя CE≈0.003813. Эти направления близки, но не равны.

<!-- /unit:41a69b6323956dadbb -->

<!-- unit:a9921ecfc70da8f6cb -->
<a id="u-a9921ecfc70da8f6cb"></a>

Полная симметричная формула:

<!-- /unit:a9921ecfc70da8f6cb -->

<!-- unit:4d2fa9e93442417620 -->
<a id="u-4d2fa9e93442417620"></a>

$$
\mathcal L_{\rm CLIP}=-\frac1{2N}\left[
\sum_i\log\frac{e^{S_{ii}/\tau}}{\sum_j e^{S_{ij}/\tau}}+
\sum_i\log\frac{e^{S_{ii}/\tau}}{\sum_j e^{S_{ji}/\tau}}
\right].
$$

<!-- /unit:4d2fa9e93442417620 -->

<!-- unit:c5266a6f0330db0fce -->
<a id="u-c5266a6f0330db0fce"></a>

В этом примере общий loss≈0.003145. Строковая и столбцовая competition учат согласовывать обе модальности, но среди batch negatives могут быть семантически подходящие captions других images. Диагональный objective не означает логического доказательства, что каждый недиагональный текст ложен для картинки.

<!-- /unit:c5266a6f0330db0fce -->

<!-- unit:657b019b5faefe0b17 -->
<a id="u-657b019b5faefe0b17"></a>

L2 normalization отделяет direction от исходной нормы, а temperature/learned logit scale регулирует резкость distribution. Удалить scale из zero-shot classification formula — изменить probabilities и loss, хотя argmax при positive scale сохранится. При единичном vector нулевой norm невозможен по определению, но сырой encoder output перед нормировкой требует определённой numerical policy.

<!-- /unit:657b019b5faefe0b17 -->

<!-- unit:2cb7dd4ee68479bfd8 -->
<a id="u-2cb7dd4ee68479bfd8"></a>

### Почему softmax score зависит от списка предложенных классов

<!-- /unit:2cb7dd4ee68479bfd8 -->

<!-- unit:d2be0e70f58efa0604 -->
<a id="u-d2be0e70f58efa0604"></a>

При logits [8,2] лидер получает0.997527. Добавим третью альтернативу с logit8 — теперь обе сильные получают $1/(2+e^{-6})\approx0.499381$. Картинка, первый prompt и их cosine не менялись. Изменилось множество конкурентов. Это объясняет, почему prompt variants одного class нельзя добавлять как независимые mutually-exclusive classes и сравнивать confidence с прежним threshold без aggregation.

<!-- /unit:d2be0e70f58efa0604 -->

<!-- unit:6929c27bd747c535e9 -->
<a id="u-6929c27bd747c535e9"></a>

Конкретные detector heads могут использовать independent sigmoid-like scores или другие outputs, а не один class softmax. Тогда добавление label не обязано менять scores через знаменатель, но может менять fusion, ranking, duplicates или postprocessing. Механизм нужно назвать по model contract. Пример с softmax доказывает свойство именно такой нормировки, а не универсальное поведение каждого OVD.

<!-- /unit:6929c27bd747c535e9 -->

<!-- unit:482b80ffc6b572b455 -->
<a id="u-482b80ffc6b572b455"></a>

CLIP pretrained на image-caption pairs, а не на обязательной разметке каждого отдельного объекта. Global image vector может быть сильно связан с общим scene content. Даже хорошее классификационное соответствие не сообщает, где лежит box маленького предмета и сколько его instances. Дополнительный region alignment нужен именно для этого разрыва между global objective и localized output. [CLIP](https://arxiv.org/abs/2103.00020).

<!-- /unit:482b80ffc6b572b455 -->

<!-- unit:5617a74a2d14f6f4d3 -->
<a id="u-5617a74a2d14f6f4d3"></a>

## Почему CLIP-vector всего изображения недостаточен для boxes

<!-- /unit:5617a74a2d14f6f4d3 -->

<!-- unit:14a9dab455f52c52e9 -->
<a id="u-14a9dab455f52c52e9"></a>

Global score может показать, что image похожа на «собака», но не выдать координаты трёх собак. Для detection нужны region proposals/dense features/queries и localised text-image alignment. Варианты:

<!-- /unit:14a9dab455f52c52e9 -->

<!-- unit:406f785eb03945a08d -->
<a id="u-406f785eb03945a08d"></a>

- region-based methods вроде ViLD/RegionCLIP используют candidate regions и согласуют их features с text space/teacher;
- GLIP связывает detection и grounding через deeper visual-language fusion;
- Grounding DINO использует text-aware feature enhancement/query selection и cross-modal decoder;
- YOLO-World строит быстрый detector с text-conditioned representations, включая RepVL-PAN и prompt-then-detect подход.

<!-- /unit:406f785eb03945a08d -->

<!-- unit:31f093785a5008b554 -->
<a id="u-31f093785a5008b554"></a>

Это механизмы для сравнения, а не вечный ranking моделей по скорости. Точный runtime включает text encoding, image backbone, detection/postprocessing и возможное caching prompts. [YOLO-World](https://arxiv.org/abs/2401.17270).

<!-- /unit:31f093785a5008b554 -->

<!-- unit:4d19d8791fc4e94a2b -->
<a id="u-4d19d8791fc4e94a2b"></a>

Если список классов фиксирован на много images, text embeddings иногда можно предвычислить. Это уменьшает repeated text work, но не устраняет необходимость compatible encoders/reparameterization. Смена prompts/encoder может потребовать обновить precomputed state.

<!-- /unit:4d19d8791fc4e94a2b -->

<!-- unit:227c74d33cbad75079 -->
<a id="u-227c74d33cbad75079"></a>

### Region teacher, region-text pretraining и глубокое fusion

<!-- /unit:227c74d33cbad75079 -->

<!-- unit:783a8c707e37c18643 -->
<a id="u-783a8c707e37c18643"></a>

В ViLD-подобной схеме proposals дают crops, teacher image encoder получает region representation, student detector учится согласовывать свои features с этим пространством, одновременно используя detection supervision. Так expensive teacher knowledge переносится в localized student outputs. Если proposals не покрывают объект, прекрасный text classifier не создаст отсутствующий candidate. С другой стороны, crop может удалить context, который нужен для интерпретации предмета.

<!-- /unit:783a8c707e37c18643 -->

<!-- unit:afad6eba35311ae3e8 -->
<a id="u-afad6eba35311ae3e8"></a>

RegionCLIP подчёркивает region-text alignment при pretraining: global caption representation не всегда подходит локальной части. Автоматическое сопоставление regions и phrases даёт дополнительную supervision, но наследует ошибки teacher/proposals и неполные описания. «Автоматически полученный текст региона» не равен гарантированно точной аннотации всех visual attributes.

<!-- /unit:afad6eba35311ae3e8 -->

<!-- unit:cf93274ecbbb8924ea -->
<a id="u-cf93274ecbbb8924ea"></a>

GLIP связывает detection с grounding и взаимодействием text/visual features; Grounding DINO использует feature enhancement, language-guided query selection и cross-modal decoding. Text здесь не только последний набор classifier vectors: он может влиять на то, какие candidates и признаки будут обработаны. Поэтому caching только final text embeddings не всегда позволяет полностью убрать все cross-modal вычисления image pass. [Grounding DINO](https://arxiv.org/abs/2303.05499).

<!-- /unit:cf93274ecbbb8924ea -->

<!-- unit:d6845f8d6ad3e4d131 -->
<a id="u-d6845f8d6ad3e4d131"></a>

### YOLO-World: prompt-then-detect и границы caching

<!-- /unit:d6845f8d6ad3e4d131 -->

<!-- unit:6647ee91c3b6c30965 -->
<a id="u-6647ee91c3b6c30965"></a>

В YOLO-World learned visual-language aggregation связывает image features с text representations, а region-text contrastive learning учит localized matching. При фиксированном vocabulary prompts подготавливают заранее и переиспользуют, сокращая repeated text work. Такой режим отличается от нового произвольного длинного запроса на каждом кадре. Быстрая precomputed-vocabulary inference не доказывает ту же latency для произвольного text-conditioning workflow. [YOLO-World](https://arxiv.org/abs/2401.17270).

<!-- /unit:6647ee91c3b6c30965 -->

<!-- unit:295edb1150d63b0498 -->
<a id="u-295edb1150d63b0498"></a>

Ключ cache должен включать text encoder/checkpoint, токенизацию, preprocessing и точные prompts. Если усреднялись templates, правило normalization/averaging тоже часть cached artifact. Смена text model при той же D не сохраняет совместимость со старым region space. Если система reparameterizes части сети для фиксированного vocabulary, его смена требует соответствующего обновления, а не простой замены подписи результата.

<!-- /unit:295edb1150d63b0498 -->

<!-- unit:a66ebf28e4e21206df -->
<a id="u-a66ebf28e4e21206df"></a>

### Другие способы связывать регионы и категории

<!-- /unit:a66ebf28e4e21206df -->

<!-- unit:5c760d8d2a6cfecce7 -->
<a id="u-5c760d8d2a6cfecce7"></a>

OWL-ViT/OWLv2 связывают Transformer visual features с text-conditioned detection и self-training-направлением. Detic расширяет detector supervision image-level labels; корректная работа называется «Detecting Twenty-thousand Classes using Image-level Supervision», а не «Detect Everything with Text» из разбираемого примера. Объект может иметь image label без точного box, поэтому используется другой механизм распространения supervision. [Detic](https://arxiv.org/abs/2201.02605).

<!-- /unit:5c760d8d2a6cfecce7 -->

<!-- unit:c293fa653ebfebfb5e -->
<a id="u-c293fa653ebfebfb5e"></a>

DetCLIP, FIBER, CORA, MM-Grounding-DINO и APE были перечислены в источнике как дополнительные варианты. Их полезно изучать по конкретному вопросу: как строится concept dictionary, где происходит fusion, что обучается на regions и какие types supervision доступны. Одно название в сравнительной таблице не устанавливает одинаковый backbone, arbitrary-language support или неизменный runtime. Для выбора нужны released model card, evaluation setup и точный output contract.

<!-- /unit:c293fa653ebfebfb5e -->

<!-- unit:2e331da7d7cd29becd -->
<a id="u-2e331da7d7cd29becd"></a>

## Novel AP и честное значение zero-shot

<!-- /unit:2e331da7d7cd29becd -->

<!-- unit:79b9180c1413fd9144 -->
<a id="u-79b9180c1413fd9144"></a>

Разделить **base classes**, по которым доступно объявленное supervised training, и **novel classes**, не использованные в определённом виде training. Измерять AP_base,AP_novel и при необходимости generalized setting со всеми classes вместе. В отчёте указать, какие annotations/text corpora доступны: «unseen» всегда имеет scope.

<!-- /unit:79b9180c1413fd9144 -->

<!-- unit:d54bca8c10589d4abe -->
<a id="u-d54bca8c10589d4abe"></a>

Detection matching/AP такие же по геометрическому смыслу, как в главе [«Detection: множества объектов, IoU, AP и архитектуры»](<cv-detection.md>), но label mapping задаётся protocol. Синонимы не должны случайно считаться разными mutually exclusive objects; несколько prompt variants одного class могут породить duplicate boxes. Нужно объявить synonym aggregation и postprocessing.

<!-- /unit:d54bca8c10589d4abe -->

<!-- unit:ea8e6cc30de1a6a7b7 -->
<a id="u-ea8e6cc30de1a6a7b7"></a>

Prompt «vehicle» шире «car». Если GT содержит only cars, detector может правильно найти bicycle по broad prompt и быть засчитан FP в узком protocol. Это несовпадение ontology, а не обязательно visual failure. Для compositional phrases нужны отдельные тесты attributes/relations, а не только class AP.

<!-- /unit:ea8e6cc30de1a6a7b7 -->

<!-- unit:582a59337fa8d62f70 -->
<a id="u-582a59337fa8d62f70"></a>

Проверять blank/unrelated prompts, absent classes, rare/small objects, text language и negative cases. Candidate list может исключить правильную class; обычный softmax всё равно выберет лучшую из неправильных. Поэтому reject/abstention policy требует calibration по задаче.

<!-- /unit:582a59337fa8d62f70 -->

<!-- unit:a2057b3342f271d481 -->
<a id="u-a2057b3342f271d481"></a>

### Base/novel, generalized и cross-dataset protocol

<!-- /unit:a2057b3342f271d481 -->

<!-- unit:3425ab98d7a31a2404 -->
<a id="u-3425ab98d7a31a2404"></a>

OV-COCO/OV-LVIS — не просто замена названия dataset: нужен конкретный class split и перечень разрешённых источников обучения. Long-tail LVIS подчёркивает редкие классы, Objects365/OpenImages дают другие labels и distributions, grounding datasets вроде Flickr30k Entities/Visual Genome — localized phrases своего вида. Смешивание этих источников может улучшить обучение, но меняет смысл «novel» при проверке пересечения classes и annotations.

<!-- /unit:3425ab98d7a31a2404 -->

<!-- unit:6799f668240f2880f9 -->
<a id="u-6799f668240f2880f9"></a>

Novel-only evaluation может скрывать конфликт base/novel scores. В generalized evaluation обе группы конкурируют в одной выдаче, поэтому calibration и mapping важны. Например, обученный class «animal» и novel «red fox» могут описывать одну region на разном ontology level. Считать их двумя независимыми физическими объектами ошибочно, но benchmark задаёт собственные допустимые labels.

<!-- /unit:6799f668240f2880f9 -->

<!-- unit:34f67e4880353bf06f -->
<a id="u-34f67e4880353bf06f"></a>

Корректный отчёт фиксирует prompts и synonyms, class mapping, region/box thresholds, NMS policy, input resolution и data provenance. Нельзя подобрать формулировки на test examples, затем представить финальный novel AP как результат без tuning. Если одновременно меняются backbone, box data и vocabulary, сравнение с baseline не изолирует эффект open-vocabulary alignment.

<!-- /unit:34f67e4880353bf06f -->

<!-- unit:cc69ce75523a5a9238 -->
<a id="u-cc69ce75523a5a9238"></a>

Псевдоразметка редкого класса может повысить coverage и закрепить teacher mistakes. Precision/recall teacher по предполагаемому novel concept проверяют отдельно, особенно при fine-grained distinctions. Общий detection loss с alignment term содержит компромисс: лучшее global contrastive matching не гарантирует более точные boxes маленьких объектов.

<!-- /unit:cc69ce75523a5a9238 -->

<!-- unit:89c983282394fd18fd -->
<a id="u-89c983282394fd18fd"></a>

## Что делать с названиями «YOLO26-OV» из разбираемого примера

<!-- /unit:89c983282394fd18fd -->

<!-- unit:9058aca18476f3694c -->
<a id="u-9058aca18476f3694c"></a>

Такое название в источнике не следует автоматически превращать в общеупотребимый model ID с придуманными weights/head. В официальной экосистеме есть YOLO-World и YOLOE с конкретными open-vocabulary возможностей и variants. Точную поддержку YOLO26-based variant следует устанавливать по released checkpoint/config и output contract. [YOLOE](https://docs.ultralytics.com/models/yoloe/).

<!-- /unit:9058aca18476f3694c -->

<!-- unit:7c261dde4b8a6aa09b -->
<a id="u-7c261dde4b8a6aa09b"></a>

Это позволяет сохранить учебный смысл исходной темы — локализация по тексту — без выдуманного универсального checkpoint. В плане архитектурное имя трактуется как проверяемый implementation example, а не обязательная новая отдельная теория.

<!-- /unit:7c261dde4b8a6aa09b -->

<!-- unit:daac2917d7926a1942 -->
<a id="u-daac2917d7926a1942"></a>

### Templates, ensembles и отрицание

<!-- /unit:daac2917d7926a1942 -->

<!-- unit:4e99f883e799de6d1a -->
<a id="u-4e99f883e799de6d1a"></a>

Шаблон «a photo of a dog» задаёт контекст, который может быть полезен для natural-image pretraining. Для diagram или медицинского снимка такой текст может быть неуместным. В template ensemble сначала получают нормированные vectors нескольких описаний, затем выбирают и документируют aggregation: средний vector с повторной нормировкой или среднее scores — разные правила. Несколько templates не гарантируют улучшение каждого класса.

<!-- /unit:4e99f883e799de6d1a -->

<!-- unit:2138be607984da08b1 -->
<a id="u-2138be607984da08b1"></a>

Фраза «dog, not cat» из рассматриваемого примера не является универсальным negative prompt operator. Обычный text encoder может плохо учитывать negation, а interface может трактовать запятую как разделитель labels. Если нужно исключить cats, возможны отдельные scores/правила и проверенные negative examples; нельзя предполагать логический NOT только по английскому слову not. Language coverage и синонимы также проверяются на фактических prompts.

<!-- /unit:2138be607984da08b1 -->

<!-- unit:b679c4a0b53ee3153b -->
<a id="u-b679c4a0b53ee3153b"></a>

### Разложение сложного запроса требует проверки relations

<!-- /unit:b679c4a0b53ee3153b -->

<!-- unit:d85871b619a7aab57d -->
<a id="u-d85871b619a7aab57d"></a>

Для «красная чашка на столе» можно найти candidates чашек, оценить цвет, найти стол и проверить spatial relation. Но разложение через LLM на две strings не обеспечивает правильного связывания. Если красная чашка на полке, а синяя на столе, простое наличие обеих категорий не удовлетворяет запрос. Нужен entity binding: adjective относится к той же чашке, которая находится на столе.

<!-- /unit:d85871b619a7aab57d -->

<!-- unit:6756c1fd2de852045d -->
<a id="u-6756c1fd2de852045d"></a>

Image-plane правило «box чашки выше box стола» — только эвристика, не доказательство физического контакта. Occlusion, camera angle и perspective могут его нарушить. Для проверки relations полезны минимальные пары, где состав objects одинаков, но нужный attribute или relation поменялся; class AP без таких примеров почти не измеряет compositional capability.

<!-- /unit:6756c1fd2de852045d -->

<!-- unit:8ebd22f17394e07808 -->
<a id="u-8ebd22f17394e07808"></a>

### Диагностика от localization до rejection

<!-- /unit:8ebd22f17394e07808 -->

<!-- unit:c156848af6bfec74fc -->
<a id="u-c156848af6bfec74fc"></a>

Сначала проверяют, есть ли region нужного объекта среди candidates при relaxed scoring. Если нет — ограничение proposals/resolution/visual features. Если есть, но score чужого phrase выше — проблема alignment, language или ontology. Если matching хорош, но множество синонимов создаёт duplicates — проблема aggregation/suppression. Отсутствующий object prompt должен давать возможность отказа, а не обязательный «лучший из плохих» box.

<!-- /unit:c156848af6bfec74fc -->

<!-- unit:121010e99bc834f4ed -->
<a id="u-121010e99bc834f4ed"></a>

Нулевой результат тоже неоднозначен: объект отсутствует, слишком мал, вне знания модели или отрезан threshold. Для каждого failure сохраняют image, exact prompts, raw candidates и принятое правило. Видео-поиск дополнительно требует track association, а region→SAM mask pipeline — mask prompting и независимую mask evaluation. Так использование OVD остаётся проверяемой частью системы, сохраняя различия задач из всех предыдущих глав.

<!-- /unit:121010e99bc834f4ed -->

<!-- unit:5e496a3c70d5ef6672 -->
<a id="u-5e496a3c70d5ef6672"></a>

### Open-vocabulary detector, exemplar prompt и SAM 3

<!-- /unit:5e496a3c70d5ef6672 -->

<!-- unit:a9953e9fabdb0dfa60 -->
<a id="u-a9953e9fabdb0dfa60"></a>

CLIP-style image-text similarity сравнивает image regions или global embedding с текстовым списком; detector превращает предложение в локализованные candidates. SAM 3 задаёт соседнюю, но иную задачу: text phrase либо image exemplar задаёт концепт, а выход содержит masks и identities matching instances. Эти outputs полезно связать в pipeline, но сам факт наличия text prompt не делает две модели одним типом detector.

<!-- /unit:a9953e9fabdb0dfa60 -->

<!-- unit:e3968c8ad83c1bbeae -->
<a id="u-e3968c8ad83c1bbeae"></a>

SAM 3 вводит Promptable Concept Segmentation и benchmark SA-Co. Meta сообщает о сравнении на SA-Co отдельно от traditional fixed-vocabulary benchmarks. Для evaluation укажите prompt modality, ontology/vocabulary, exhaustive instance protocol, localization rule и false-positive handling. Если используете SAM 3 как mask stage после внешнего detector, измеряйте proposal recall и mask quality условно на proposals, а затем отдельно end-to-end result.

<!-- /unit:e3968c8ad83c1bbeae -->

<!-- unit:6755b3675cf699f941 -->
<a id="u-6755b3675cf699f941"></a>

Также ограничьте язык вывода: release report описывает короткие noun phrases как основной prompt mode и отмечает сложности с длинными relational phrases без внешней multimodal reasoning stage. Это различие помогает отделить open vocabulary от свободного понимания произвольной фразы.

<!-- /unit:6755b3675cf699f941 -->

<!-- unit:01bd98444016d4d8f9 -->
<a id="u-01bd98444016d4d8f9"></a>

**Источники:** [SAM 3 research release](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/); [Meta SAM 3.1 update](https://ai.meta.com/blog/segment-anything-model-3/). Benchmark gains, prompt behavior и скорости — результаты авторов для собственных protocols.

<!-- /unit:01bd98444016d4d8f9 -->

<!-- unit:ff95ba5f95a07781d8 -->
<a id="u-ff95ba5f95a07781d8"></a>

## Пример диагностики и самостоятельные вопросы

<!-- /unit:ff95ba5f95a07781d8 -->

<!-- unit:5ecfeba303dff83f48 -->
<a id="u-5ecfeba303dff83f48"></a>

Есть image с двумя чашками, красной и синей. На «cup» detector находит обе; на «red cup» тоже обе. Box localisation уже работает, но attribute grounding не различает цвет. Увеличить NMS threshold здесь не исправит text-image alignment. Если красная найдена в region proposals, но rejected classifier, проблема может быть в alignment/threshold; если её нет в proposals — в localisation/scale.

<!-- /unit:5ecfeba303dff83f48 -->

<!-- unit:71da65fcd56d882007 -->
<a id="u-71da65fcd56d882007"></a>

1. Почему одинаковая D image/text vectors не достаточна? **Ответ:** пространства должны быть совместно согласованы objective; произвольные encoders не становятся compatible по размеру.
2. Почему high CLIP probability не existence probability? **Ответ:** зависит от набора candidates и logit scale, нужна separate calibration.
3. Что значит novel class? **Ответ:** отсутствие определённого вида supervision по protocol, а не гарантированное отсутствие concept во всём pretraining.

<!-- /unit:71da65fcd56d882007 -->

<!-- unit:d5bcd843d6ab4fc20d -->
<a id="u-d5bcd843d6ab4fc20d"></a>

**Практика:** выбрать пять images с несколькими objects, фиксированный base/novel class list и по три перефразировки. Измерить geometry AP/recall и prompt sensitivity отдельно; добавить absent-object negatives. Для одного failure локализовать этап: proposal, alignment, threshold, ontology или relation understanding. Сохранить exact prompts и checkpoint revision.

<!-- /unit:d5bcd843d6ab4fc20d -->

<!-- unit:550c826748528e979b -->
<a id="u-550c826748528e979b"></a>

<!-- latex-display-v1 -->

<!-- /unit:550c826748528e979b -->

<!-- unit:3e83f1f98de4e81bf9 -->
<a id="u-3e83f1f98de4e81bf9"></a>

<!-- source-depth-S7-И -->

<!-- /unit:3e83f1f98de4e81bf9 -->

## Архитектуры: подробные схемы

### CLIP: image и text ветви

![CLIP: image и text ветви](<../оформление/course/arch-clip.svg>)

Два encoder независимо кодируют модальности. Pairwise scores сравнивают векторы одинаковой размерности; training objective выделяет правильные пары среди альтернатив batch. Этот граф не содержит region proposals или boxes, поэтому image-level CLIP не является готовым detector.

### YOLO-World

![YOLO-World](<../оформление/course/arch-yolo-world.svg>)

Text features задают сравниваемые категории, а spatial detector локализует регионы. Open vocabulary относится к интерфейсу категорий; качество на новых классах проверяется по отдельному label/protocol split.

### ViLD: region-level distillation

![ViLD: region-level distillation](<../оформление/course/arch-vild.svg>)

Teacher кодирует candidate crops, student detector учится согласовывать region features с teacher space вместе с detection supervision. Proposal recall ограничивает достижимый результат до классификации; crop также может потерять контекст.

### RegionCLIP

![RegionCLIP](<../оформление/course/arch-regionclip.svg>)

Region-level alignment учит сопоставлять локальные признаки и текстовые понятия. Автоматические пары наследуют ошибки proposals/teacher и не становятся гарантированно полной атрибутивной разметкой.

### GLIP: grounding и detection

![GLIP: grounding и detection](<../оформление/course/arch-glip.svg>)

Text участвует в построении признаков, а не только в финальной таблице classifier weights. Фраза и локализация связываются в общей постановке; вычислительная стоимость зависит от fusion и числа кандидатов.

### Grounding DINO

![Grounding DINO](<../оформление/course/arch-grounding-dino.svg>)

Языковой контекст влияет на feature enhancement, выбор queries и decoder. Поэтому кэширование final text vectors не обязано устранять все cross-modal операции image pass.

### OWL-ViT

![OWL-ViT](<../оформление/course/arch-owlvit.svg>)

Spatial visual tokens используются для detection, а text vectors задают сравниваемые категории. Image-level pooling classifier CLIP не выдаёт такую локализацию без box mechanism.

### OWLv2: self-training направление

![OWLv2: self-training направление](<../оформление/course/arch-owlv2.svg>)

Схема показывает роль автоматически полученной supervision в расширении detector training. Pseudo labels и vocabulary не гарантируют отсутствие ошибок teacher или переноса annotation coverage на любую новую область.

### Detic: image-level supervision

![Detic: image-level supervision](<../оформление/course/arch-detic.svg>)

Название работы — Detecting Twenty-thousand Classes using Image-level Supervision. Image label не задаёт точную box; механизм использования слабой разметки отличается от предположения о полной box annotation. Это не расшифровка «Detect Everything with Text».

### SAM 3: concept segmentation

![SAM 3: concept segmentation](<../оформление/course/arch-sam3.svg>)

Это функциональный интерфейс Promptable Concept Segmentation: концепт задаётся phrase либо exemplar, выход содержит matching instances и masks. Схема не приписывает checkpoint неуказанное число слоёв или универсальные COCO gains.

## Результат главы и переход дальше

Ответьте на вопрос: **Как связать текстовое название с локализованным объектом?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

[Маршрут модуля](<README.md>) · [Происхождение и точные места переноса](<../editorial/coverage.md>)
