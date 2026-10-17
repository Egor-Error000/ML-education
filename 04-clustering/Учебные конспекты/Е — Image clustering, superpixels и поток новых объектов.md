# Е. Clustering изображений и pixels: разные уровни задачи

[Оглавление](README.md). Пункт 6 плана. **Основа:** S7:207–1617 — backbones/ViT, S7:2469–3080 — segmentation. Clustering/SSL/SLIC — дополнение. Упоминание «вашего SSL-проекта» в плане не означает подтверждённый здесь выполненный проект.

## 1. Семантика не равна raw pixels

Два снимка одной собаки при разном фоне могут быть дальше по pixels, чем собака и стул на одинаковом фоне. Image encoder сокращает nuisance variation, но objective training определяет, что именно сохраняется. Supervised classifier features, self-supervised representations и generative latents могут давать разные clusters.

Frozen CNN/ViT → pooling → normalization/PCA → K-Means/density method — разумный baseline. Для ViT нужно указать CLS/mean patch pooling и layer; для CNN — spatial pooling и preprocessing resolution. Нельзя произвольно сменить crop/normalization и считать encoder тем же representation protocol.

SimCLR, MoCo и DINO — примеры разных self-supervised training families, а не три гарантированно эквивалентных pretrained features. Contrastive learning использует positive views и negatives по своему objective; self-distillation variants имеют другие targets/collapse-control mechanisms. Для clustering важен measured перенос features, а не популярность названия.

## 2. Аугментации задают группировку

Если positive views получают aggressive crops, encoder обучается сближать views, которые training считает одним объектом. Если crop сохранил только фон, такая связь может быть ошибочной. Если цвет определяет target class, слишком сильное подавление color variation может уничтожить нужный distinction.

Поэтому cluster audit включает labels факторов: object type, source/camera, illumination, background и duplicates. Не обязательно иметь полную human taxonomy; можно вручную проверить typical examples/nearest neighbours и выявить, что именно объединяет group. K-Means centre часто не является существующим image; medoid/nearest representative удобнее для inspection.

## 3. Pixel clustering

Для простой unsupervised segmentation pixel representation может включать colour и coordinates:

$$
z_{ij}=(\lambda_c L_{ij},\lambda_c a_{ij},\lambda_c b_{ij},
\lambda_s i,\lambda_s j).
$$

Weights контролируют trade-off colour similarity и spatial compactness. Если coordinates слишком слабы, далёкие одинаковые colours объединятся; если слишком сильны — получатся почти регулярные клетки. Эти regions не являются автоматически semantic objects. Один object может содержать разные colours, а разные objects — одинаковые.

Raw RGB distance зависит от illumination и color space. Lab часто используют для perceptual colour difference, но сам выбор space не делает segmentation semantic. Масштаб coordinates зависит от resolution; resize без изменения weights меняет effective objective.

## 4. SLIC

SLIC строит компактные superpixels через локальную clustering pixels в colour–spatial space. При приблизительном grid step $S$ и compactness parameter $m$ common distance имеет вид:

$$
D=\sqrt{d_c^2+\left(\frac mS\right)^2d_s^2}.
$$

Search ограничивается окрестностями centres, затем coordinates/colours centres обновляются; дополнительная connectivity обработка делает superpixels связными. Увеличение $m$ усиливает spatial regularity, уменьшение позволяет сильнее следовать colour boundaries. [Оригинальный SLIC](https://www.epfl.ch/labs/ivrl/research/slic-superpixels/).

Toy calculation: при $d_c=10,d_s=5,m=20,S=10$ получаем $D=\sqrt{200}\approx14.142$. При $m=5$ distance около 10.308. Это не сравнение качества, а влияние spatial term. Superpixel — удобная единица preprocessing, но не label «кошка» или «дорога». Избыточно крупные regions пересекают истинные boundaries.

## 5. Self-supervised segmentation и поток

Кластеризация patch/pixel embeddings может выделять semantic-like regions, если features несут нужную информацию. Spatial upsampling, boundary quality и mapping clusters к labels требуют отдельной оценки. Semantic class IDs нельзя назначить по произвольным cluster numbers без protocol.

В stream новые categories и distribution drift меняют taxonomy. Incremental K-Means обновляет centres, но обычно сохраняет заданное K; это не автоматически dynamic discovery новых classes. Для создания/слияния groups нужны novelty thresholds, evidence counts и правила versioning. Threshold не выбирается только по текущей красивой visualization.

Report включает out-of-sample assignment, unknown/noise fraction и stability cluster identities между versions. При centre drift старый ID может начать означать другое; downstream consumers должны знать version. Duplicate video frames нельзя разнести между fit/evaluation как independent objects.

## 6. Контроль и практика

1. Равен ли superpixel semantic object? **Ответ:** нет; он группирует локально сходные pixels и может быть частью объекта.
2. Почему frozen encoder не гарантирует нужную taxonomy? **Ответ:** representation обучалось своему objective и может сохранять другие factors.
3. Открывает ли incremental K-Means новые classes автоматически? **Ответ:** обычный fixed-K update этого не делает; нужен отдельный novelty/group management protocol.

**Практика:** выберите 30 изображений с разными backgrounds и duplicates. Сравните raw pixels и frozen embeddings на одном algorithm, вручную опишите factors каждого cluster. Для одного image измените spatial/colour weights pixel clustering и объясните boundary trade-off; затем спроектируйте обработку unknown stream objects.
