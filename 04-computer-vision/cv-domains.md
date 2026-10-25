# Медицинские и спутниковые изображения: данные, объём и масштаб

**Уровень:** углубление.

**Предпосылки:** [U-Net и nnU-Net: пространственный decoder и настройка pipeline](<cv-unet.md>), [Ориентированные рамки и периодичность угла](<cv-obb.md>), [Камера и геометрия: проекция, координаты и калибровка](<cv-camera.md>)

**Используют:** Самостоятельное применение выбранной ветки.

**Учебный вопрос:** Какие свойства медицинских и спутниковых данных меняют обычный CV pipeline?

## Схема вычислений

![Медицинские и спутниковые изображения: данные, объём и масштаб: вычислительные этапы](<../оформление/course/expanded-cv-domains.svg>)

Medical pipeline: Volume + spacing / orientation → Resample / normalize / patches → 2D / 3D U-Net → logits → Original grid → physical metrics. Satellite pipeline: Scene + bands + GSD → Tiles с overlap + offsets → Detector / segmenter → local → Scene merge → geographic split.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Единица независимости и физический масштаб

В medical CV независимым объектом часто является patient, а не slice. В remote sensing — geographic region, acquisition date или orbit, а не случайный tile. Пространственно близкие tiles и slices одного volume содержат общий контекст. Random split может измерять узнавание знакомого объекта вместо переноса. Общие split правила даны в [главе о достоверности эксперимента](<../01-data-features/01-reliability/df-split.md>).

## Medical volumes

Volume содержит grid $[D,H,W]$, spacing в mm и orientation. Одинаковые voxel counts могут обозначать разный физический размер. 3D CNN учитывает соседние slices, но memory растёт с объёмом; anisotropic spacing может требовать разных kernels и pooling по осям. [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) согласует preprocessing, patch geometry и network configuration с данными.

Intensity normalization зависит от modality. CT values и MRI intensities имеют разные смыслы; windowing и normalization не следует применять одинаково без обоснования. Ignore labels, partial annotation и inter-rater disagreement входят в target contract. Dice крупного органа не показывает sensitivity маленьких lesions. Physical boundary errors измеряют в mm с корректным spacing.

## Remote sensing

Satellite images могут иметь multispectral bands, georeferencing и ground sampling distance, GSD. RGB-pretrained model не принимает произвольные дополнительные bands без изменения stem и normalization. Rotated boxes и polygons полезны для вытянутых объектов; axis orientation и units должны быть согласованы. Tiling позволяет обработать большие scenes, но создаёт border effects и дубликаты objects на overlap.

При tile inference predictions переводят в общие scene coordinates, объединяют overlap и только затем оценивают objects. Cloud, season, sensor и sun angle вызывают domain shift. [DOTA](https://captain-whu.github.io/DOTA/) — пример benchmark с oriented annotations; его task protocol нельзя переносить на любой satellite dataset автоматически.

## Разобранный пример: spacing и tiling

Medical volume $256\times256\times128$ при spacing $(0.7,0.7,3)$ mm, в порядке x,y,z, покрывает по соглашению voxel extent приблизительно $(179.2,179.2,384)$ mm. Кубический kernel $3^3$ samples имеет разные расстояния между centers по осям. Два voxels по z — 6 mm, два по x — 1.4 mm: одинаковая index error физически различна.

Satellite tile $512\times512$ при GSD 0.5 m/pixel покрывает 256 m на сторону. Объект длиной 10 m занимает 20 pixels, а после downsampling вчетверо — пять. Для изображения $2048^2$, tiles 512 и stride 384 требуется по $\lceil(2048-512)/384\rceil+1=5$ tiles на ось, всего 25. Граница в этом примере совпадает точно; для других размеров нужен explicit last-tile/padding policy.

## Самопроверка с ответами

**Практика:** разделите patient slices случайно. Test теперь содержит изображения того же пациента; оценка не отвечает переносу на нового пациента.

**Почему нужен split по geographic regions?** Соседние участки разделяют текстуру, инфраструктуру и условия съёмки, что создаёт spatial leakage.

**Можно ли усреднить overlap logits и считать проблему решённой?** Для semantic maps это допустимая политика после проверки; instance boxes требуют ещё сопоставления и удаления дубликатов в scene coordinates.

## Источники и продолжение

- [nnU-Net official repository](https://github.com/MIC-DKFZ/nnUNet)
- [DOTA — authors benchmark](https://captain-whu.github.io/DOTA/)

[Выбрать следующую ветку](<README.md>) · [Сценарии подраздела](<07-specialized/План.md>)
