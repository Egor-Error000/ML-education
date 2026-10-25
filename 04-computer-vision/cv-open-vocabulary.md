# CLIP, открытый словарь и YOLO-World

**Уровень:** основа ветки.

**Предпосылки:** [Detection: множества объектов, IoU, AP и архитектуры](<cv-detection.md>), [Vision Transformer: patches, attention и обучение](<cv-vit.md>)

**Используют:** Самостоятельное применение выбранной ветки.

**Учебный вопрос:** Как связать текстовое название с локализованным объектом?

## Карта вычисления и границы этапов

![CLIP, открытый словарь и YOLO-World: входы, преобразования и выходы](<../оформление/course/cv-open-vocabulary.svg>)

Текст задаёт кандидатов классов, но не гарантирует обнаружение любого объекта. Формулировки и отрицательные примеры входят в протокол проверки открытого словаря.

**Схема курса:** image → visual encoder (кодировщик входа)/region features (признаки); text prompt (входной запрос или инструкция) → text encoder; aligned scores (оценки) → grounded box candidates → threshold/association → optional mask model; evaluation keeps classification, localization, mask and prompt sensitivity separate.

## Что меняется по сравнению с обычным detector

Closed-vocabulary head выдаёт scores для фиксированного списка K classes. В text-conditioned detection классы/phrases подаются на inference (применение обученной модели), а модель пытается связать regions с их text representations. Output всё ещё содержит localization и confidence (уверенность по принятой оценке), но class meaning задаётся запросом вроде «красный грузовик».

Это не обещание распознать любой термин: нужны выученные visual/text concepts и способность их совместить. Редкий медицинский label (метка), новый инструмент или spatial relation могут быть вне capabilities. Отсутствие обучения **на box annotations novel class** не доказывает, что модель никогда не встречала этот concept в image-text pretraining (предобучение).

Open-set recognition, OVD и phrase grounding пересекаются, но задачи различны. Open-set может требовать reject unknown; OVD — задавать labels (метки) текстом; grounding — найти region для referential phrase. «Красная чашка слева от тарелки» требует relations и выбор среди нескольких instances, что сложнее отдельного class name.

### Что действительно открывается при смене текстового словаря

Обычный classifier detector имеет K выходных class channels с обученными значениями. Добавить строку «редкий инструмент» в пользовательский интерфейс не создаёт новый channel с нужными весами. В text-conditioned detector class representation можно вычислить text encoder, поэтому словарь inputs меняется без обязательного переобучения всей head. Однако визуальные признаки и соответствие тексту должны уже поддерживать концепт; новый текст не создаёт отсутствующее знание объекта.

В примере «обычный COCO detector не найдёт красную чашку» нужно разделить две части. Он может найти cup, если этот класс поддержан; требование отличить именно красную от синей — дополнительное attribute grounding. Текстовый интерфейс удобнее выражает этот фильтр, но модель всё равно может проигнорировать adjective. Поэтому наличие natural-language input не подтверждает compositional understanding.

Novel определяется видом отсутствующей supervision. Концепт мог встречаться в captions, image-level labels или pretrained teacher (модель-учитель), хотя box annotations novel class не использовались на этапе detector training. Называть это «никогда не видела объект» нельзя. Cross-dataset transfer, class-held-out zero-shot и few-shot с несколькими novel examples — разные эксперименты, у каждого нужен собственный train/test contract.

В робототехнике найденная box ещё не задаёт grasp pose, в автоматической разметке — не становится проверенной GT, в поиске по видео — не обеспечивает persistent ID. OVD решает локализацию по запросу как компонент более широкой системы. Для каждого применения выбирают geometry, rejection и последующий этап, сохраняя неопределённость text-region matching.

## CLIP: два encoder и общее пространство

Image encoder f_I и text encoder f_T отображают разные inputs в согласованную D-мерную space. После L2 normalization (нормировка по выбранному правилу) similarity $s_{ij}=v_i^\top t_j$ — cosine. Для batch (пакет примеров) из N matched image-caption pairs logits (логиты, оценки альтернатив до нормировки)=$\frac{s}{\tau}$ с temperature (температура распределения выбора)/learned scale.

Image-to-text loss (функция потерь):

$$
\mathcal{L}_I=-\frac1N\sum_{i=1}^{N}\log\frac{\exp(s_{ii}/\tau)}{\sum_{j=1}^{N}\exp(s_{ij}/\tau)}.
$$

Text-to-image аналогично нормируется по images; symmetric CLIP objective (оптимизируемая целевая функция) усредняет два направления. В разбираемом примере была приведена одна сторона; для полного классического objective нужна и вторая. [CLIP](https://arxiv.org/abs/2103.00020).

Пример одной строки similarity $\left(0.8,\,0.2\right)$, tau0.1 даёт logits $\left(8,\,2\right)$ и вероятность matched caption≈0.9975. Это probability **среди данных candidates** при данном scale, а не calibrated вероятность, что объект существует. Добавление новых prompt classes меняет denominator и scores. Поэтому threshold нельзя переносить без проверки prompt set/model revision.

Для zero-shot classification encode templates «a photo of a cat», «a photo of a dog» и сравнить image vector с ними. Ensembles templates могут уменьшать чувствительность к формулировке, но требуют осознанного aggregation. Русский/английский synonym не обязан дать одинаковый vector, если text encoder multilingual coverage ограничена.

### Полная матрица image-text matching и оба направления loss

Возьмём две matched пары, cosine matrix $S=[[0.8,0.2],[0.1,0.7]]$ и temperature0.1. Logits равны [[8,2],[1,7]]. Image-to-text normalizes каждую строку: обе правильные probabilities равны $1/(1+e^{-6})\approx0.997527$, loss≈0.002476. Text-to-image normalizes столбцы: для первого text правильная probability $1/(1+e^{-7})\approx0.999089$, для второго — $1/(1+e^{-5})\approx0.993307$, средняя CE≈0.003813. Эти направления близки, но не равны.

Полная симметричная формула:

$$
\mathcal L_{\rm CLIP}=-\frac1{2N}\left[
\sum_i\log\frac{e^{S_{ii}/\tau}}{\sum_j e^{S_{ij}/\tau}}+
\sum_i\log\frac{e^{S_{ii}/\tau}}{\sum_j e^{S_{ji}/\tau}}
\right].
$$

В этом примере общий loss≈0.003145. Строковая и столбцовая competition учат согласовывать обе модальности, но среди batch negatives могут быть семантически подходящие captions других images. Диагональный objective не означает логического доказательства, что каждый недиагональный текст ложен для картинки.

L2 normalization отделяет direction от исходной нормы, а temperature/learned logit scale регулирует резкость distribution. Удалить scale из zero-shot classification formula — изменить probabilities и loss, хотя argmax при positive scale сохранится. При единичном vector нулевой norm невозможен по определению, но сырой encoder output перед нормировкой требует определённой numerical policy.

### Почему softmax score (оценка) зависит от списка предложенных классов

При logits [8,2] лидер получает0.997527. Добавим третью альтернативу с logit8 — теперь обе сильные получают $1/(2+e^{-6})\approx0.499381$. Картинка, первый prompt и их cosine не менялись. Изменилось множество конкурентов. Это объясняет, почему prompt variants одного class нельзя добавлять как независимые mutually-exclusive classes и сравнивать confidence с прежним threshold без aggregation.

Конкретные detector heads могут использовать independent sigmoid-like scores или другие outputs, а не один class softmax. Тогда добавление label не обязано менять scores через знаменатель, но может менять fusion, ranking, duplicates или postprocessing. Механизм нужно назвать по model contract. Пример с softmax доказывает свойство именно такой нормировки, а не универсальное поведение каждого OVD.

CLIP pretrained на image-caption pairs, а не на обязательной разметке каждого отдельного объекта. Global image vector может быть сильно связан с общим scene content. Даже хорошее классификационное соответствие не сообщает, где лежит box маленького предмета и сколько его instances. Дополнительный region alignment нужен именно для этого разрыва между global objective и localized output. [CLIP](https://arxiv.org/abs/2103.00020).

## Почему CLIP-vector всего изображения недостаточен для boxes

Global score может показать, что image похожа на «собака», но не выдать координаты трёх собак. Для detection нужны region proposals/dense features/queries и localised text-image alignment. Варианты:

- region-based methods вроде ViLD/RegionCLIP используют candidate regions и согласуют их features с text space/teacher;
- GLIP связывает detection и grounding через deeper visual-language fusion;
- Grounding DINO использует text-aware feature (признак) enhancement/query (запрос) selection и cross-modal decoder (декодер);
- YOLO-World строит быстрый detector с text-conditioned representations, включая RepVL-PAN и prompt-then-detect подход.

Это механизмы для сравнения, а не вечный ranking моделей по скорости. Точный runtime (среда исполнения модели) включает text encoding, image backbone (основная сеть извлечения признаков), detection/postprocessing и возможное caching prompts. [YOLO-World](https://arxiv.org/abs/2401.17270).

Если список классов фиксирован на много images, text embeddings (векторные представления) иногда можно предвычислить. Это уменьшает repeated text work, но не устраняет необходимость compatible encoders/reparameterization. Смена prompts/encoder может потребовать обновить precomputed state (состояние системы).

### Region teacher, region-text pretraining и глубокое fusion

В ViLD-подобной схеме proposals дают crops, teacher image encoder получает region representation, student (модель-ученик) detector учится согласовывать свои features с этим пространством, одновременно используя detection supervision. Так expensive teacher knowledge переносится в localized student outputs. Если proposals не покрывают объект, прекрасный text classifier не создаст отсутствующий candidate. С другой стороны, crop может удалить context (доступный модели контекст), который нужен для интерпретации предмета.

RegionCLIP подчёркивает region-text alignment при pretraining: global caption representation не всегда подходит локальной части. Автоматическое сопоставление regions и phrases даёт дополнительную supervision, но наследует ошибки teacher/proposals и неполные описания. «Автоматически полученный текст региона» не равен гарантированно точной аннотации всех visual attributes.

GLIP связывает detection с grounding и взаимодействием text/visual features; Grounding DINO использует feature enhancement, language-guided query selection и cross-modal decoding. Text здесь не только последний набор classifier vectors: он может влиять на то, какие candidates и признаки будут обработаны. Поэтому caching только final text embeddings не всегда позволяет полностью убрать все cross-modal вычисления image pass. [Grounding DINO](https://arxiv.org/abs/2303.05499).

### YOLO-World: prompt-then-detect и границы caching

В YOLO-World learned visual-language aggregation связывает image features с text representations, а region-text contrastive learning учит localized matching. При фиксированном vocabulary prompts подготавливают заранее и переиспользуют, сокращая repeated text work. Такой режим отличается от нового произвольного длинного запроса на каждом кадре. Быстрая precomputed-vocabulary inference не доказывает ту же latency (задержка отдельного запроса) для произвольного text-conditioning workflow. [YOLO-World](https://arxiv.org/abs/2401.17270).

Ключ cache (кеш повторно используемых вычислений) должен включать text encoder/checkpoint (сохранённое состояние модели), токенизацию, preprocessing и точные prompts. Если усреднялись templates, правило normalization/averaging тоже часть cached artifact. Смена text model при той же D не сохраняет совместимость со старым region space. Если система reparameterizes части сети для фиксированного vocabulary, его смена требует соответствующего обновления, а не простой замены подписи результата.

### Другие способы связывать регионы и категории

OWL-ViT/OWLv2 связывают Transformer visual features с text-conditioned detection и self-training-направлением. Detic расширяет detector supervision image-level labels; корректная работа называется «Detecting Twenty-thousand Classes using Image-level Supervision», а не «Detect Everything with Text» из разбираемого примера. Объект может иметь image label без точного box, поэтому используется другой механизм распространения supervision. [Detic](https://arxiv.org/abs/2201.02605).

DetCLIP, FIBER, CORA, MM-Grounding-DINO и APE были перечислены в источнике как дополнительные варианты. Их полезно изучать по конкретному вопросу: как строится concept dictionary, где происходит fusion, что обучается на regions и какие types supervision доступны. Одно название в сравнительной таблице не устанавливает одинаковый backbone, arbitrary-language support или неизменный runtime. Для выбора нужны released model card, evaluation setup и точный output contract.

## Novel AP и честное значение zero-shot

Разделить **base classes**, по которым доступно объявленное supervised training, и **novel classes**, не использованные в определённом виде training. Измерять AP_base,AP_novel и при необходимости generalized setting со всеми classes вместе. В отчёте указать, какие annotations/text corpora доступны: «unseen» всегда имеет scope.

Detection matching/AP такие же по геометрическому смыслу, как в главе [«Detection: множества объектов, IoU, AP и архитектуры»](<cv-detection.md>), но label mapping задаётся protocol. Синонимы не должны случайно считаться разными mutually exclusive objects; несколько prompt variants одного class могут породить duplicate boxes. Нужно объявить synonym aggregation и postprocessing.

Prompt «vehicle» шире «car». Если GT содержит only cars, detector может правильно найти bicycle по broad prompt и быть засчитан FP в узком protocol. Это несовпадение ontology, а не обязательно visual failure. Для compositional phrases нужны отдельные тесты attributes/relations, а не только class AP.

Проверять blank/unrelated prompts, absent classes, rare/small objects, text language и negative cases. Candidate list может исключить правильную class; обычный softmax всё равно выберет лучшую из неправильных. Поэтому reject/abstention policy требует calibration по задаче.

### Base/novel, generalized и cross-dataset protocol

OV-COCO/OV-LVIS — не просто замена названия dataset: нужен конкретный class split (разбиение данных) и перечень разрешённых источников обучения. Long-tail LVIS подчёркивает редкие классы, Objects365/OpenImages дают другие labels и distributions, grounding datasets вроде Flickr30k Entities/Visual Genome — localized phrases своего вида. Смешивание этих источников может улучшить обучение, но меняет смысл «novel» при проверке пересечения classes и annotations.

Novel-only evaluation может скрывать конфликт base/novel scores. В generalized evaluation обе группы конкурируют в одной выдаче, поэтому calibration и mapping важны. Например, обученный class «animal» и novel «red fox» могут описывать одну region на разном ontology level. Считать их двумя независимыми физическими объектами ошибочно, но benchmark задаёт собственные допустимые labels.

Корректный отчёт фиксирует prompts и synonyms, class mapping, region/box thresholds, NMS policy, input resolution и data provenance. Нельзя подобрать формулировки на test examples, затем представить финальный novel AP как результат без tuning. Если одновременно меняются backbone, box data и vocabulary, сравнение с baseline (базовый вариант сравнения) не изолирует эффект open-vocabulary alignment.

Псевдоразметка редкого класса может повысить coverage и закрепить teacher mistakes. Precision/recall teacher по предполагаемому novel concept проверяют отдельно, особенно при fine-grained distinctions. Общий detection loss с alignment term содержит компромисс: лучшее global contrastive matching не гарантирует более точные boxes маленьких объектов.

## Что делать с названиями «YOLO26-OV» из разбираемого примера

Такое название в источнике не следует автоматически превращать в общеупотребимый model ID с придуманными weights/head. В официальной экосистеме есть YOLO-World и YOLOE с конкретными open-vocabulary возможностей и variants. Точную поддержку YOLO26-based variant следует устанавливать по released checkpoint/config и output contract. [YOLOE](https://docs.ultralytics.com/models/yoloe/).

Это позволяет сохранить учебный смысл исходной темы — локализация по тексту — без выдуманного универсального checkpoint. В плане архитектурное имя трактуется как проверяемый implementation example, а не обязательная новая отдельная теория.

### Templates, ensembles и отрицание

Шаблон «a photo of a dog» задаёт контекст, который может быть полезен для natural-image pretraining. Для diagram или медицинского снимка такой текст может быть неуместным. В template (шаблон сериализации входа) ensemble сначала получают нормированные vectors нескольких описаний, затем выбирают и документируют aggregation: средний vector с повторной нормировкой или среднее scores — разные правила. Несколько templates не гарантируют улучшение каждого класса.

Фраза «dog, not cat» из рассматриваемого примера не является универсальным negative prompt operator. Обычный text encoder может плохо учитывать negation, а interface может трактовать запятую как разделитель labels. Если нужно исключить cats, возможны отдельные scores/правила и проверенные negative examples; нельзя предполагать логический NOT только по английскому слову not. Language coverage и синонимы также проверяются на фактических prompts.

### Разложение сложного запроса требует проверки relations

Для «красная чашка на столе» можно найти candidates чашек, оценить цвет, найти стол и проверить spatial relation. Но разложение через LLM на две strings не обеспечивает правильного связывания. Если красная чашка на полке, а синяя на столе, простое наличие обеих категорий не удовлетворяет запрос. Нужен entity binding: adjective относится к той же чашке, которая находится на столе.

Image-plane правило «box чашки выше box стола» — только эвристика, не доказательство физического контакта. Occlusion, camera angle и perspective могут его нарушить. Для проверки relations полезны минимальные пары, где состав objects одинаков, но нужный attribute или relation поменялся; class AP без таких примеров почти не измеряет compositional capability.

### Диагностика от localization до rejection

Сначала проверяют, есть ли region нужного объекта среди candidates при relaxed scoring. Если нет — ограничение proposals/resolution/visual features. Если есть, но score чужого phrase выше — проблема alignment, language или ontology. Если matching хорош, но множество синонимов создаёт duplicates — проблема aggregation/suppression. Отсутствующий object prompt должен давать возможность отказа, а не обязательный «лучший из плохих» box.

Нулевой результат тоже неоднозначен: объект отсутствует, слишком мал, вне знания модели или отрезан threshold. Для каждого failure сохраняют image, exact prompts, raw candidates и принятое правило. Видео-поиск дополнительно требует track association, а region→SAM mask pipeline — mask prompting и независимую mask evaluation. Так использование OVD остаётся проверяемой частью системы, сохраняя различия задач из всех предыдущих глав.

### Open-vocabulary detector, exemplar prompt и SAM 3

CLIP-style image-text similarity сравнивает image regions или global embedding (векторное представление) с текстовым списком; detector превращает предложение в локализованные candidates. SAM 3 задаёт соседнюю, но иную задачу: text phrase либо image exemplar задаёт концепт, а выход содержит masks и identities matching instances. Эти outputs полезно связать в pipeline, но сам факт наличия text prompt не делает две модели одним типом detector.

SAM 3 вводит Promptable Concept Segmentation и benchmark SA-Co. Meta сообщает о сравнении на SA-Co отдельно от traditional fixed-vocabulary benchmarks. Для evaluation укажите prompt modality, ontology/vocabulary, exhaustive instance protocol, localization rule и false-positive handling. Если используете SAM 3 как mask stage после внешнего detector, измеряйте proposal recall и mask quality условно на proposals, а затем отдельно end-to-end result.

Также ограничьте язык вывода: release report описывает короткие noun phrases как основной prompt mode и отмечает сложности с длинными relational phrases без внешней multimodal reasoning stage. Это различие помогает отделить open vocabulary от свободного понимания произвольной фразы.

**Источники:** [SAM 3 research release](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/); [Meta SAM 3.1 update](https://ai.meta.com/blog/segment-anything-model-3/). Benchmark gains, prompt behavior и скорости — результаты авторов для собственных protocols.

## Пример диагностики и самостоятельные вопросы

Есть image с двумя чашками, красной и синей. На «cup» detector находит обе; на «red cup» тоже обе. Box localisation уже работает, но attribute grounding не различает цвет. Увеличить NMS threshold здесь не исправит text-image alignment. Если красная найдена в region proposals, но rejected classifier, проблема может быть в alignment/threshold; если её нет в proposals — в localisation/scale.

1. Почему одинаковая D image/text vectors не достаточна? **Ответ:** пространства должны быть совместно согласованы objective; произвольные encoders не становятся compatible по размеру.
2. Почему high CLIP probability не existence probability? **Ответ:** зависит от набора candidates и logit scale, нужна separate calibration.
3. Что значит novel class? **Ответ:** отсутствие определённого вида supervision по protocol, а не гарантированное отсутствие concept во всём pretraining.

**Практика:** выбрать пять images с несколькими objects, фиксированный base/novel class list и по три перефразировки. Измерить geometry AP/recall и prompt sensitivity отдельно; добавить absent-object negatives. Для одного failure локализовать этап: proposal, alignment, threshold, ontology или relation understanding. Сохранить exact prompts и checkpoint revision.





## Архитектуры: подробные схемы

### CLIP: image и text ветви

![CLIP: image и text ветви](<../оформление/course/arch-clip.svg>)

Два encoder независимо кодируют модальности. Pairwise scores сравнивают векторы одинаковой размерности; training objective выделяет правильные пары среди альтернатив batch. Этот граф не содержит region proposals или boxes, поэтому image-level CLIP не является готовым detector.

### YOLO-World

![YOLO-World](<../оформление/course/arch-yolo-world.svg>)

Text features задают сравниваемые категории, а spatial detector локализует регионы. Open vocabulary относится к интерфейсу категорий; качество на новых классах проверяется по отдельному label/protocol split.

### ViLD: region-level distillation (дистилляция, обучение по сигналу модели-учителя)

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

Spatial visual tokens (токены) используются для detection, а text vectors задают сравниваемые категории. Image-level pooling (агрегирование состояний) classifier CLIP не выдаёт такую локализацию без box mechanism.

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

[Маршрут модуля](<README.md>) · [Происхождение и точные места переноса](<../editorial/coverage.md>)

## Иллюстрации из первоисточников

![clip: Figure 1 из авторской статьи](<../оформление/course/papers/clip.png>)

**Источник:** [clip, Figure 1](https://arxiv.org/abs/2103.00020), PDF, с. 2. Проследите вход, преобразования и выход по стрелкам. Сопоставьте обозначения статьи с формами и targets, определёнными в тексте главы.

![sam: Figure 4 из авторской статьи](<../оформление/course/papers/sam.png>)

**Источник:** [sam, Figure 4](https://arxiv.org/abs/2304.02643), PDF, с. 5. Это исходный SAM со spatial prompts. Text/concept интерфейс поздних SAM 3 и video updates не следует приписывать этой схеме.

## Как читать углубления этой ветки


CLIP связывает global image и text embeddings; Grounding DINO локализует regions по тексту; исходный SAM возвращает masks по spatial prompts. Это разные interfaces. CLIP сходство whole-image vector не является box detector. Grounding может дать boxes для SAM, но pipeline требует согласованных coordinate transforms и отдельной оценки каждого этапа.

В [SAM](https://arxiv.org/abs/2304.02643) image encoder создаёт reusable image embedding. Prompt encoder переводит points, boxes и masks в sparse/dense embeddings. Mask decoder связывает prompt и image features и выдаёт несколько masks и оценки их качества. Несколько candidates представляют неоднозначность prompt, а не автоматически несколько разных semantic classes. Для повторных prompts по одному image image embedding можно кешировать при том же preprocessing.

SAM 3 с concept prompting и SAM 3.1 с video multiplexing — более поздние interfaces, которые не следует приписывать исходной схеме SAM. Их сведения сохранены в существующем тексте и повторно проверены по [репозиторию авторов](https://github.com/facebookresearch/sam3). Confidence и prompt semantics каждой версии оценивают по соответствующему protocol.

## Разобранный пример для проверки контракта


Image $800\times600$ приведён к $1024\times768$ с padding 256 rows снизу. Scale 1.28, point $(400,300)$ переходит в $(512,384)$; padding снизу не меняет origin. При симметричном padding 128 rows сверху тот же point имел бы y=512. Передать prompt без знания политики padding означает указать другое место.

## Ограничения переноса

Prompt templates, novel/base split, per-image vocabularies и pretraining exposure меняют смысл zero-shot evaluation. Одной gallery иллюстраций недостаточно для quantitative conclusions.

## Самопроверка с ответами

**Практика:** дайте SAM point внутри двух вложенных объектов, например рука и человек. Несколько допустимых masks объясняют неоднозначность гранулярности, а не ошибку decoder по определению.

**Является ли SAM исходной версии text classifier?** Нет, её spatial-prompt interface отличается от later concept models.

**Гарантирует ли смена vocabulary обнаружение новых objects?** Нет: localization, text understanding и domain transfer имеют собственные ограничения.

## Схема вычислений

![CLIP, открытый словарь и YOLO-World: вычислительные этапы](<../оформление/course/expanded-cv-open-vocabulary.svg>)

Grounding + SAM: Text + image → grounding boxes → Согласованные image coordinates → SAM image + prompt encoders → Mask decoder → candidates / scores.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Источники и продолжение

- [CLIP — Radford et al., ICML 2021](https://arxiv.org/abs/2103.00020)
- [Grounding DINO — Liu et al., ECCV 2024](https://arxiv.org/abs/2303.05499)
- [SAM — Kirillov et al., ICCV 2023](https://arxiv.org/abs/2304.02643)
- [SAM 3 — official repository](https://github.com/facebookresearch/sam3)

[Выбрать следующую ветку](<README.md>) · [Сценарии подраздела](<04-segmentation/План.md>)
