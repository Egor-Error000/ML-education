# Кластеризация изображений и superpixels

**Уровень:** основа ветки.

**Предпосылки:** [Кластеризация замороженных представлений](<cl-frozen.md>)

**Используют:** Выбранная прикладная задача и итоговая проверка результата.

**Учебный вопрос:** Как различаются группировка изображений и группировка пикселей?





## Карта вычисления и границы этапов

![Кластеризация изображений и superpixels: входы, преобразования и выходы](<../оформление/course/cl-images.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Сравните формы и смысл каждого перехода с формулами главы. Выход преобразования и доказательство его качества — разные объекты: последняя проверка использует отдельный протокол оценки.

<!-- unit:10c5b6432eb0a1e17a -->
<a id="u-10c5b6432eb0a1e17a"></a>

## Семантика не равна raw pixels

<!-- /unit:10c5b6432eb0a1e17a -->

<!-- unit:4a0f6bac6d76ec1f4c -->
<a id="u-4a0f6bac6d76ec1f4c"></a>

Два снимка одной собаки при разном фоне могут быть дальше по pixels, чем собака и стул на одинаковом фоне. Image encoder сокращает nuisance variation, но objective training определяет, что именно сохраняется. Supervised classifier features, self-supervised representations и generative latents могут давать разные clusters.

<!-- /unit:4a0f6bac6d76ec1f4c -->

<!-- unit:c98d52ad411df58839 -->
<a id="u-c98d52ad411df58839"></a>

Frozen CNN/ViT → pooling → normalization/PCA → K-Means/density method — разумный baseline. Для ViT нужно указать CLS/mean patch pooling и layer; для CNN — spatial pooling и preprocessing resolution. Нельзя произвольно сменить crop/normalization и считать encoder тем же representation protocol.

<!-- /unit:c98d52ad411df58839 -->

<!-- unit:46e30a5dc501577568 -->
<a id="u-46e30a5dc501577568"></a>

SimCLR, MoCo и DINO — примеры разных self-supervised training families, а не три гарантированно эквивалентных pretrained features. Contrastive learning использует positive views и negatives по своему objective; self-distillation variants имеют другие targets/collapse-control mechanisms. Для clustering важен measured перенос features, а не популярность названия.

<!-- /unit:46e30a5dc501577568 -->

<!-- unit:30e534f643fda2f419 -->
<a id="u-30e534f643fda2f419"></a>

## Аугментации задают группировку

<!-- /unit:30e534f643fda2f419 -->

<!-- unit:534e74f0143d2d588c -->
<a id="u-534e74f0143d2d588c"></a>

Если positive views получают aggressive crops, encoder обучается сближать views, которые training считает одним объектом. Если crop сохранил только фон, такая связь может быть ошибочной. Если цвет определяет target class, слишком сильное подавление color variation может уничтожить нужный distinction.

<!-- /unit:534e74f0143d2d588c -->

<!-- unit:5e2feac5d924e68e8f -->
<a id="u-5e2feac5d924e68e8f"></a>

Поэтому cluster audit включает labels факторов: object type, source/camera, illumination, background и duplicates. Не обязательно иметь полную human taxonomy; можно вручную проверить typical examples/nearest neighbours и выявить, что именно объединяет group. K-Means centre часто не является существующим image; medoid/nearest representative удобнее для inspection.

<!-- /unit:5e2feac5d924e68e8f -->

<!-- unit:3f576f5fa34053eb4a -->
<a id="u-3f576f5fa34053eb4a"></a>

## Pixel clustering

<!-- /unit:3f576f5fa34053eb4a -->

<!-- unit:a60fc18b196b3fcdbc -->
<a id="u-a60fc18b196b3fcdbc"></a>

Для простой unsupervised segmentation pixel representation может включать colour и coordinates:

<!-- /unit:a60fc18b196b3fcdbc -->

<!-- unit:c9a05a97cc628ce6dc -->
<a id="u-c9a05a97cc628ce6dc"></a>

$$
z_{ij}=(\lambda_c L_{ij},\lambda_c a_{ij},\lambda_c b_{ij},
\lambda_s i,\lambda_s j).
$$

<!-- /unit:c9a05a97cc628ce6dc -->

<!-- unit:48b183b5bb878ef8a9 -->
<a id="u-48b183b5bb878ef8a9"></a>

Weights контролируют trade-off colour similarity и spatial compactness. Если coordinates слишком слабы, далёкие одинаковые colours объединятся; если слишком сильны — получатся почти регулярные клетки. Эти regions не являются автоматически semantic objects. Один object может содержать разные colours, а разные objects — одинаковые.

<!-- /unit:48b183b5bb878ef8a9 -->

<!-- unit:2fa36ca89f30ce7354 -->
<a id="u-2fa36ca89f30ce7354"></a>

Raw RGB distance зависит от illumination и color space. Lab часто используют для perceptual colour difference, но сам выбор space не делает segmentation semantic. Масштаб coordinates зависит от resolution; resize без изменения weights меняет effective objective.

<!-- /unit:2fa36ca89f30ce7354 -->

<!-- unit:74d61ebb0c7b5c3715 -->
<a id="u-74d61ebb0c7b5c3715"></a>

## SLIC

<!-- /unit:74d61ebb0c7b5c3715 -->

<!-- unit:ac0b25c9e91030ab8e -->
<a id="u-ac0b25c9e91030ab8e"></a>

SLIC строит компактные superpixels через локальную clustering pixels в colour–spatial space. При приблизительном grid step $S$ и compactness parameter $m$ common distance имеет вид:

<!-- /unit:ac0b25c9e91030ab8e -->

<!-- unit:b730a2cb880e9fbd35 -->
<a id="u-b730a2cb880e9fbd35"></a>

$$
D=\sqrt{d_c^2+\left(\frac mS\right)^2d_s^2}.
$$

<!-- /unit:b730a2cb880e9fbd35 -->

<!-- unit:0a5fadb5c4b2226713 -->
<a id="u-0a5fadb5c4b2226713"></a>

Search ограничивается окрестностями centres, затем coordinates/colours centres обновляются; дополнительная connectivity обработка делает superpixels связными. Увеличение $m$ усиливает spatial regularity, уменьшение позволяет сильнее следовать colour boundaries. [Оригинальный SLIC](https://www.epfl.ch/labs/ivrl/research/slic-superpixels/).

<!-- /unit:0a5fadb5c4b2226713 -->

<!-- unit:be4fc0ec790df0442e -->
<a id="u-be4fc0ec790df0442e"></a>

Toy calculation: при $d_c=10,d_s=5,m=20,S=10$ получаем $D=\sqrt{200}\approx14.142$. При $m=5$ distance около 10.308. Это не сравнение качества, а влияние spatial term. Superpixel — удобная единица preprocessing, но не label «кошка» или «дорога». Избыточно крупные regions пересекают истинные boundaries.

<!-- /unit:be4fc0ec790df0442e -->

<!-- unit:cfee72f59c200a0d69 -->
<a id="u-cfee72f59c200a0d69"></a>

## Self-supervised segmentation и поток

<!-- /unit:cfee72f59c200a0d69 -->

<!-- unit:10d5397f4e9a72dbff -->
<a id="u-10d5397f4e9a72dbff"></a>

Кластеризация patch/pixel embeddings может выделять semantic-like regions, если features несут нужную информацию. Spatial upsampling, boundary quality и mapping clusters к labels требуют отдельной оценки. Semantic class IDs нельзя назначить по произвольным cluster numbers без protocol.

<!-- /unit:10d5397f4e9a72dbff -->

<!-- unit:a8add4062af7f1b385 -->
<a id="u-a8add4062af7f1b385"></a>

В stream новые categories и distribution drift меняют taxonomy. Incremental K-Means обновляет centres, но обычно сохраняет заданное K; это не автоматически dynamic discovery новых classes. Для создания/слияния groups нужны novelty thresholds, evidence counts и правила versioning. Threshold не выбирается только по текущей красивой visualization.

<!-- /unit:a8add4062af7f1b385 -->

<!-- unit:bafc1756b2d6ea03c7 -->
<a id="u-bafc1756b2d6ea03c7"></a>

Report включает out-of-sample assignment, unknown/noise fraction и stability cluster identities между versions. При centre drift старый ID может начать означать другое; downstream consumers должны знать version. Duplicate video frames нельзя разнести между fit/evaluation как independent objects.

<!-- /unit:bafc1756b2d6ea03c7 -->

<!-- unit:286b99e49d243ffae4 -->
<a id="u-286b99e49d243ffae4"></a>

## Контроль и практика

<!-- /unit:286b99e49d243ffae4 -->

<!-- unit:afa91cb640b8eccd24 -->
<a id="u-afa91cb640b8eccd24"></a>

1. Равен ли superpixel semantic object? **Ответ:** нет; он группирует локально сходные pixels и может быть частью объекта.
2. Почему frozen encoder не гарантирует нужную taxonomy? **Ответ:** representation обучалось своему objective и может сохранять другие factors.
3. Открывает ли incremental K-Means новые classes автоматически? **Ответ:** обычный fixed-K update этого не делает; нужен отдельный novelty/group management protocol.

<!-- /unit:afa91cb640b8eccd24 -->

<!-- unit:f60d941aa298dfbd5b -->
<a id="u-f60d941aa298dfbd5b"></a>

**Практика:** выберите 30 изображений с разными backgrounds и duplicates. Сравните raw pixels и frozen embeddings на одном algorithm, вручную опишите factors каждого cluster. Для одного image измените spatial/colour weights pixel clustering и объясните boundary trade-off; затем спроектируйте обработку unknown stream objects.

<!-- /unit:f60d941aa298dfbd5b -->

<!-- unit:950c7e81c4538d70cf -->
<a id="u-950c7e81c4538d70cf"></a>

**Схема курса:** image → feature map/superpixels → region representation → clustering → assign later stream item or abstain → inspect image-level and region-level errors

<!-- /unit:950c7e81c4538d70cf -->

## Результат главы и переход дальше

Ответьте на вопрос: **Как различаются группировка изображений и группировка пикселей?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

[Маршрут модуля](<README.md>) · [Происхождение и точные места переноса](<../editorial/coverage.md>)
