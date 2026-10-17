# В. Semantic segmentation: классы пикселей и восстановление деталей

[Оглавление](README.md). Пункт 3 плана.

**Основа:** S7, строки 2469–3080. Сохранена последовательность pixel task → masks/metrics → FCN → U-Net → dilated convolution/DeepLab. **Углубление:** ручная confusion matrix, CE/Dice, ignore-index и разница semantic auxiliary loss/отдельной semantic head.

## 1. Что выдаётся на каждом пикселе

Модель возвращает logits $B\times K\times H_{\mathrm{out}}\times W_{\mathrm{out}}$; после согласования resolution каждый pixel получает class ID. Две соприкасающиеся собаки имеют один class «собака»: semantic output не различает их instance identities. Background — обычный class только если так определена разметка; void/ignore — отдельная политика, а не ещё один обучаемый класс автоматически.

Ground truth для mutually exclusive classes обычно integer mask $B\times H\times W$. Цветной preview может хранить palette, но colour RGB не обязательно равен class ID. Если исходные mask IDs 0,1,2, это не интенсивности изображения: нельзя применять к ним bilinear interpolation как к photo.

Аугментации геометрии синхронизируют image и mask. Для categorical mask обычно используют nearest interpolation и проверяют border fill/ignore. Для logits перед argmax допустимо другое resize; порядок «resize logits → argmax» и «argmax → resize IDs» даёт разные boundaries и должен соответствовать protocol.

## 2. Pixel accuracy, IoU и mIoU

Для класса k $\operatorname{IoU}_k=\frac{\mathrm{TP}_k}{\mathrm{TP}_k+\mathrm{FP}_k+\mathrm{FN}_k}$. Pixel accuracy=$\frac{\sum_k\mathrm{TP}_k}{N_{\mathrm{valid}}}$; mIoU усредняет выбранные class IoU. Ignore pixels исключаются из соответствующих counts. Classes absent в GT и prediction имеют нулевой denominator; evaluator должен явно определить их включение/исключение.

Учебная mask из 100 pixels содержит 10 foreground и90 background. Модель предсказала background везде. Pixel accuracy90%. Для foreground TP0,FP0,FN10 → IoU0. Для background TP90,FP10,FN0 → IoU0.9. Mean по двум classes=0.45. Это объясняет, почему высокая accuracy не достаточна для малого объекта.

Другой пример: истинных foreground10, predicted foreground12, intersection8. Тогда TP8,FP4,FN2, $\operatorname{IoU}=\frac{8}{14}\approx 0.57143$, $Dice=\frac{16}{22}\approx 0.72727$. На binary sets $\mathrm{Dice}=\frac{2\operatorname{IoU}}{1+\operatorname{IoU}}$. При усреднении классов и soft losses со smoothing/weights нельзя без проверки превращать одну общую метрику в другую той же формулой.

Boundary quality может быть плохой при неплохой area IoU: тонкие structures и несколько сдвинутых pixels существенно влияют на практическую пользу. Добавлять boundary-specific evaluation по задаче, а не заменять ею целиком segmentation quality.

## 3. Обучение CE и Dice

Pixelwise CE $-\log p_{\mathrm{true\ class}}$ усредняется по valid pixels с возможными class weights. При сильном class imbalance большие background regions доминируют. Class weights, focal loss, sampling crops с foreground и Dice-family objectives меняют signal разными способами.

Soft Dice для binary mask g и probabilities p можно задать:

$$
D_{\mathrm{soft}}=\frac{2\sum_i p_ig_i+\varepsilon}{\sum_i p_i+\sum_i g_i+\varepsilon},\qquad\mathcal{L}_{\mathrm{Dice}}=1-D_{\mathrm{soft}}.
$$

Есть variants с квадратами denominator, per-image/per-batch averaging и разной обработкой пустого foreground. Нужно объявить convention. Dice зависит от совокупности pixels, тогда как CE даёт локальный signal. Смесь CE+λDice часто рассматривают как баланс, но λ подбирается на validation, не по универсальной константе.

Ignore-index исключает pixel supervision. Его нужно согласовать в loss и metrics: иначе «неразмеченное» может случайно стать background. Empty mask тоже содержит информацию, если scene действительно без objects; это не повод автоматически исключить все такие изображения.

## 4. FCN и U-Net: пространственная структура

Обычный classification head с flatten/global pooling теряет точную layout для dense outputs. Fully convolutional network сохраняет spatial feature maps и выдаёт class scores по сетке, затем upsamples к нужному resolution. Замена FC на conv — историческая идея, но одного nearest upsample обычно недостаточно для точных boundaries: глубокие maps уже потеряли часть деталей.

U-Net сочетает encoder downsampling и decoder upsampling, передавая high-resolution features через skip concatenations. Decoder получает и semantic context глубокой ветви, и spatial detail ранних уровней. [U-Net](https://arxiv.org/abs/1505.04597).

Это отличается от residual sum ResNet: U-Net skip обычно связывает соответствующие resolutions encoder/decoder и часто concatenates channels. Если encoder feature $B\times 64\times 64\times 64$, decoder feature $B\times 128\times 64\times 64$, concat даёт $B\times 192\times 64\times 64$, не сумму несовместимых channels. Далее conv смешивает их. Cropping/padding/align conventions должны согласовать spatial coordinates.

Upsampling может быть interpolation+conv или transposed convolution. Последняя — не буквальное обращение ранее выполненной conv и не автоматически восстанавливает потерянные pixels. Checkerboard artifacts возможны в некоторых конфигурациях, поэтому качество проверяют визуально.

## 5. Dilated convolution и DeepLab

Dilated/atrous conv раздвигает sampling positions kernel. Для $k\times k$ и dilation d effective receptive field одной операции $k_{\mathrm{eff}}=k+(k-1)(d-1)$. При $k=3$,$d=2$ размер охвата5, но обучаемых spatial weights всё ещё9, а не25. Он не читает все25 positions: внутри есть промежутки.

Это расширяет context без обязательного нового downsampling. Но большой dilation может создавать gridding effects и терять локальные связи. ASPP объединяет branches с разными atrous rates и/или global context по конкретному DeepLab variant. Decoder уточняет boundaries; details DeepLab versions не одинаковы.

Output stride задаёт отношение input/output resolution глубокого feature map. Меньший stride сохраняет detail, увеличивая compute/memory. Нельзя считать output stride и dilation одним параметром: они связаны настройкой architecture, но описывают разное.

## 6. Semantic и instance heads в YOLO-семействе

В исходнике semantic head описывалась как общая «урезанная U-Net внутри YOLO». Такое тождество не следует из имени. Отдельная semantic model должна выдавать dense classes; instance segmentation head выдаёт маски объектов. Auxiliary semantic loss при обучении instance model ещё не доказывает наличие отдельного поддерживаемого semantic inference output.

На текущей проверке официальная YOLO26 документация перечисляет отдельные `*-sem` модели; это конкретная implementation, которую нужно отличать от `*-seg`. Здесь изучается задача и путь features, а для запуска сверяют shape outputs, checkpoints и data config. [Поддерживаемые задачи](https://docs.ultralytics.com/tasks/).

## 7. Самопроверка и практика

1. Почему две соседние собаки не разделятся в semantic mask? **Ответ:** задача кодирует class, а не instance ID.
2. Сколько весов у одного $3\times 3$ filter с dilation2? **Ответ:** девять на input channel, dilation не добавляет25 weights.
3. Почему bilinear resize IDs ошибочен? **Ответ:** создаёт несуществующие промежуточные labels и меняет смысл категорий.
4. Когда 90% pixel accuracy бесполезны? **Ответ:** например, при полном пропуске малого важного foreground.

**Практика:** нарисовать две binary masks $5\times 5$, вручную определить intersection/union/Dice и добавить ignored row. Затем проследить shapes маленькой U-Net, указав concat channel counts. Для реальных данных сравнить один CE baseline с одной Dice/weight intervention, сохраняя split, input resolution и valid-pixel protocol.

<!-- latex-display-v1 -->
