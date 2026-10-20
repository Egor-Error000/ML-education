# Semantic segmentation: FCN, U-Net и DeepLab

**Уровень:** общее ядро.

**Предпосылки:** [Изображение, карты признаков и блоки модели](<cv-input.md>), [Классификация, CNN и residual learning](<cv-classification.md>)

**Используют:** [Instance segmentation: Mask R-CNN и prototype masks](<cv-instance.md>)

**Учебный вопрос:** Как предсказывать класс пикселя и восстанавливать пространственную сетку?





## Карта вычисления и границы этапов

![Semantic segmentation: FCN, U-Net и DeepLab: входы, преобразования и выходы](<../оформление/course/cv-semantic.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Сравните формы и смысл каждого перехода с формулами главы. Выход преобразования и доказательство его качества — разные объекты: последняя проверка использует отдельный протокол оценки.

<!-- unit:ea973ad28af32525f7 -->
<a id="u-ea973ad28af32525f7"></a>

## Что выдаётся на каждом пикселе

<!-- /unit:ea973ad28af32525f7 -->

<!-- unit:05fc1e5f9aabd5ef68 -->
<a id="u-05fc1e5f9aabd5ef68"></a>

Модель возвращает logits $B\times K\times H_{\mathrm{out}}\times W_{\mathrm{out}}$; после согласования resolution каждый pixel получает class ID. Две соприкасающиеся собаки имеют один class «собака»: semantic output не различает их instance identities. Background — обычный class только если так определена разметка; void/ignore — отдельная политика, а не ещё один обучаемый класс автоматически.

<!-- /unit:05fc1e5f9aabd5ef68 -->

<!-- unit:f4c4979ca9eaee9e9f -->
<a id="u-f4c4979ca9eaee9e9f"></a>

Ground truth для mutually exclusive classes обычно integer mask $B\times H\times W$. Цветной preview может хранить palette, но colour RGB не обязательно равен class ID. Если исходные mask IDs 0,1,2, это не интенсивности изображения: нельзя применять к ним bilinear interpolation как к photo.

<!-- /unit:f4c4979ca9eaee9e9f -->

<!-- unit:af96064b061eba3046 -->
<a id="u-af96064b061eba3046"></a>

Аугментации геометрии синхронизируют image и mask. Для categorical mask обычно используют nearest interpolation и проверяют border fill/ignore. Для logits перед argmax допустимо другое resize; порядок «resize logits → argmax» и «argmax → resize IDs» даёт разные boundaries и должен соответствовать protocol.

<!-- /unit:af96064b061eba3046 -->

<!-- unit:6479a8b147e986d732 -->
<a id="u-6479a8b147e986d732"></a>

### Почему «предсказать каждый пиксель» не означает решить его независимо

<!-- /unit:6479a8b147e986d732 -->

<!-- unit:6e3f8f9ae84bda1340 -->
<a id="u-6e3f8f9ae84bda1340"></a>

Если пиксель имеет серый цвет, по одному RGB нельзя надёжно отличить дорогу от крыши. Модель получает изображение целиком или crop и использует окружение: горизонт, соседние текстуры, форму региона. Pixelwise относится к форме ответа и месту supervision, а не к требованию рассматривать каждый входной pixel изолированно. Точно так же CE по отдельным позициям не запрещает общие признаки и пространственный контекст.

<!-- /unit:6e3f8f9ae84bda1340 -->

<!-- unit:7c5bdd1230927f88ca -->
<a id="u-7c5bdd1230927f88ca"></a>

Ground-truth маска и probabilities имеют разный смысл. Integer mask хранит один ID на location. Для K классов модель хранит K logits на той же location, softmax нормируется по классовой оси, и argmax выбирает один ID. Число цветов palette PNG не является автоматически K: могут присутствовать ignored values, а последовательность ID может требовать remapping. Читать palette preview как обычную RGB-картинку и брать её среднюю яркость нельзя.

<!-- /unit:7c5bdd1230927f88ca -->

<!-- unit:56bf1f30ebdf21db6a -->
<a id="u-56bf1f30ebdf21db6a"></a>

Представим labels [0,2] на соседних пикселях. Bilinear interpolation в середине вернёт 1, хотя класса 1 в этой области вообще не было. Nearest выбирает существующий label. Для floating-point class scores интерполяция допустима как часть заданного decoder/postprocessing: интерполируют K вещественных карт, затем принимают решение. Это два разных объекта, несмотря на одинаковую spatial сетку.

<!-- /unit:56bf1f30ebdf21db6a -->

<!-- unit:93a4f1b838adbd8aa3 -->
<a id="u-93a4f1b838adbd8aa3"></a>

В разбираемом примере классы одного типа описывались как «слитые в одну маску». Это не обязательно один связный регион: semantic mask «кошка» может содержать три раздельных островка. Она не кодирует, каким из них принадлежит общий instance ID. Connected components иногда отделят раздельные островки, но соприкоснувшиеся кошки останутся одним компонентом, а одна кошка за забором может распасться на несколько. Геометрическая связность не заменяет разметку экземпляров.

<!-- /unit:93a4f1b838adbd8aa3 -->

<!-- unit:e9799eb85c7a10b338 -->
<a id="u-e9799eb85c7a10b338"></a>

## Pixel accuracy, IoU и mIoU

<!-- /unit:e9799eb85c7a10b338 -->

<!-- unit:99d5367ab2c242f6cc -->
<a id="u-99d5367ab2c242f6cc"></a>

Для класса k $\operatorname{IoU}_k=\frac{\mathrm{TP}_k}{\mathrm{TP}_k+\mathrm{FP}_k+\mathrm{FN}_k}$. Pixel accuracy=$\frac{\sum_k\mathrm{TP}_k}{N_{\mathrm{valid}}}$; mIoU усредняет выбранные class IoU. Ignore pixels исключаются из соответствующих counts. Classes absent в GT и prediction имеют нулевой denominator; evaluator должен явно определить их включение/исключение.

<!-- /unit:99d5367ab2c242f6cc -->

<!-- unit:0468672d28d050bf8a -->
<a id="u-0468672d28d050bf8a"></a>

Учебная mask из 100 pixels содержит 10 foreground и90 background. Модель предсказала background везде. Pixel accuracy90%. Для foreground TP0,FP0,FN10 → IoU0. Для background TP90,FP10,FN0 → IoU0.9. Mean по двум classes=0.45. Это объясняет, почему высокая accuracy не достаточна для малого объекта.

<!-- /unit:0468672d28d050bf8a -->

<!-- unit:c912602d5e71436f6c -->
<a id="u-c912602d5e71436f6c"></a>

Другой пример: истинных foreground10, predicted foreground12, intersection8. Тогда TP8,FP4,FN2, $\operatorname{IoU}=\frac{8}{14}\approx 0.57143$, $Dice=\frac{16}{22}\approx 0.72727$. На binary sets $\mathrm{Dice}=\frac{2\operatorname{IoU}}{1+\operatorname{IoU}}$. При усреднении классов и soft losses со smoothing/weights нельзя без проверки превращать одну общую метрику в другую той же формулой.

<!-- /unit:c912602d5e71436f6c -->

<!-- unit:5c8638ce43ecbb01b0 -->
<a id="u-5c8638ce43ecbb01b0"></a>

Boundary quality может быть плохой при неплохой area IoU: тонкие structures и несколько сдвинутых pixels существенно влияют на практическую пользу. Добавлять boundary-specific evaluation по задаче, а не заменять ею целиком segmentation quality.

<!-- /unit:5c8638ce43ecbb01b0 -->

<!-- unit:aecbe6b2a8da0fa0fb -->
<a id="u-aecbe6b2a8da0fa0fb"></a>

### Вывод метрик из одной confusion matrix

<!-- /unit:aecbe6b2a8da0fa0fb -->

<!-- unit:b9fc474fda3ce61571 -->
<a id="u-b9fc474fda3ce61571"></a>

Для трёх классов рассмотрим counts, где строки — истина, столбцы — предсказание:

<!-- /unit:b9fc474fda3ce61571 -->

<!-- unit:e5ec0fa38fe847d4bd -->
<a id="u-e5ec0fa38fe847d4bd"></a>

$$
C=\begin{bmatrix}50&5&0\\3&20&2\\0&4&16\end{bmatrix}.
$$

<!-- /unit:e5ec0fa38fe847d4bd -->

<!-- unit:e8d4bf498ead0aa910 -->
<a id="u-e8d4bf498ead0aa910"></a>

Всего 100 valid pixels, диагональ содержит 86 верных, pixel accuracy=0.86. Для класса 0 TP=50, FP=3, FN=5, поэтому IoU=50/58≈0.862069. Для класса 1 TP=20, FP=9, FN=5, IoU=20/34≈0.588235. Для класса 2 TP=16, FP=2, FN=4, IoU=16/22≈0.727273. mIoU≈0.725859. Этот результат заметно ниже accuracy, поскольку ошибки малых классов не растворились в числе правильных background pixels.

<!-- /unit:e8d4bf498ead0aa910 -->

<!-- unit:1cce945a592c40329e -->
<a id="u-1cce945a592c40329e"></a>

Frequency-weighted IoU из рассматриваемого примера использует доли GT классов: 0.55,0.25,0.20. Взвешенная сумма равна примерно 0.766651. Это другое усреднение, с большим влиянием частого класса. Ни mIoU, ни fwIoU не являются ошибкой вычисления только из-за расхождения с accuracy: они отвечают на разные вопросы.

<!-- /unit:1cce945a592c40329e -->

<!-- unit:990b8279a3ceaf9d4b -->
<a id="u-990b8279a3ceaf9d4b"></a>

Для каждого класса $TP_k=C_{kk}$, $FN_k=\sum_jC_{kj}-C_{kk}$, $FP_k=\sum_iC_{ik}-C_{kk}$. Если ignored row pixels не удалить до накопления counts, они исказят сразу несколько показателей. Также нельзя сначала посчитать per-image mIoU и объявить среднее тем же, что IoU по общей confusion matrix: отношение сумм и среднее отношений в общем случае различаются.

<!-- /unit:990b8279a3ceaf9d4b -->

<!-- unit:0640eb03da18e797f4 -->
<a id="u-0640eb03da18e797f4"></a>

## Обучение CE и Dice

<!-- /unit:0640eb03da18e797f4 -->

<!-- unit:051d0ed15f22688500 -->
<a id="u-051d0ed15f22688500"></a>

Pixelwise CE $-\log p_{\mathrm{true\ class}}$ усредняется по valid pixels с возможными class weights. При сильном class imbalance большие background regions доминируют. Class weights, focal loss, sampling crops с foreground и Dice-family objectives меняют signal разными способами.

<!-- /unit:051d0ed15f22688500 -->

<!-- unit:6bb0c29cdb1c63d479 -->
<a id="u-6bb0c29cdb1c63d479"></a>

Soft Dice для binary mask g и probabilities p можно задать:

<!-- /unit:6bb0c29cdb1c63d479 -->

<!-- unit:d15b92269980136996 -->
<a id="u-d15b92269980136996"></a>

$$
D_{\mathrm{soft}}=\frac{2\sum_i p_ig_i+\varepsilon}{\sum_i p_i+\sum_i g_i+\varepsilon},\qquad\mathcal{L}_{\mathrm{Dice}}=1-D_{\mathrm{soft}}.
$$

<!-- /unit:d15b92269980136996 -->

<!-- unit:846302c8974f47fb83 -->
<a id="u-846302c8974f47fb83"></a>

Есть variants с квадратами denominator, per-image/per-batch averaging и разной обработкой пустого foreground. Нужно объявить convention. Dice зависит от совокупности pixels, тогда как CE даёт локальный signal. Смесь CE+λDice часто рассматривают как баланс, но λ подбирается на validation, не по универсальной константе.

<!-- /unit:846302c8974f47fb83 -->

<!-- unit:d5f0159ea5a39e3a95 -->
<a id="u-d5f0159ea5a39e3a95"></a>

Ignore-index исключает pixel supervision. Его нужно согласовать в loss и metrics: иначе «неразмеченное» может случайно стать background. Empty mask тоже содержит информацию, если scene действительно без objects; это не повод автоматически исключить все такие изображения.

<!-- /unit:d5f0159ea5a39e3a95 -->

<!-- unit:09f779541ff3535f28 -->
<a id="u-09f779541ff3535f28"></a>

### Локальный CE и глобальная нормировка Dice

<!-- /unit:09f779541ff3535f28 -->

<!-- unit:d23206d4491167bbc2 -->
<a id="u-d23206d4491167bbc2"></a>

Для одного binary foreground logit z вероятность $p=\sigma(z)$, а BCE равна $-[g\ln p+(1-g)\ln(1-p)]$. Производная по z — p−g. Каждый valid pixel даёт свой прямой сигнал. Если 9900 background pixels уже немного ошибаются, их суммарный вклад всё ещё может превосходить вклад ста foreground pixels. Усреднение по всему изображению не делает классы равновесными.

<!-- /unit:d23206d4491167bbc2 -->

<!-- unit:0f95b03216ca340a6e -->
<a id="u-0f95b03216ca340a6e"></a>

Weighted CE меняет стоимость ошибок классов; sampling меняет частоту примеров, видимых optimizer; focal меняет вес в зависимости от текущей сложности. Они не взаимозаменяемы. Правило $w_k=1/f_k$ из разбираемого примера — возможная эвристика, а не автоматический безопасный выбор: очень редкий шумный класс способен получить огромный вес. Для weighted reduction нужно также определить знаменатель — число pixels либо сумму активных weights.

<!-- /unit:0f95b03216ca340a6e -->

<!-- unit:7824bdaf1b556224f8 -->
<a id="u-7824bdaf1b556224f8"></a>

Для soft Dice обозначим $A=2\sum_i p_ig_i+\varepsilon$ и $B=\sum_i p_i+\sum_i g_i+\varepsilon$. Тогда

<!-- /unit:7824bdaf1b556224f8 -->

<!-- unit:d8c3c7080c7ac1d066 -->
<a id="u-d8c3c7080c7ac1d066"></a>

$$
\frac{\partial(1-A/B)}{\partial p_j}=\frac{A-2g_jB}{B^2}.
$$

<!-- /unit:d8c3c7080c7ac1d066 -->

<!-- unit:502fbd2b53af8e47fe -->
<a id="u-502fbd2b53af8e47fe"></a>

Градиент одного pixel зависит от sums по всей маске. Поэтому Dice связывает вклад размера region и overlap; CE отдельно штрафует неверную вероятность каждой цели. При p=(0.8,0.2), g=(1,0), без smoothing получаем A=1.6,B=2,Dice=0.8. Производные loss по p равны (−0.6,0.4): foreground вероятность надо увеличивать, background уменьшать. Это не те же масштабы, что у BCE.

<!-- /unit:502fbd2b53af8e47fe -->

<!-- unit:bbf62860f10b704a93 -->
<a id="u-bbf62860f10b704a93"></a>

Для пустой GT маски без smoothing числитель A=0. При ненулевой prediction sum Dice=0, но производная выбранного варианта по всем p тоже нулевая: один этот objective не обучает подавлять пустой foreground. Epsilon и особые empty-mask правила меняют поведение; CE может давать нужный локальный сигнал. Поэтому обещание «Dice работает при дисбалансе» требует конкретной формулы и политики пустых targets.

<!-- /unit:bbf62860f10b704a93 -->

<!-- unit:a42b24eb3ce7a4b8d8 -->
<a id="u-a42b24eb3ce7a4b8d8"></a>

В разбираемом примере также вводился soft IoU: $I=\sum p_ig_i$, $U=\sum p_i+\sum g_i-I$, loss=1−I/U. При тех же p/g soft IoU=0.8/1.2=2/3, в отличие от Dice=0.8. Это непрерывные training surrogates; hard metric после threshold может быть равна 1, если бинарная маска уже верна. Улучшение loss и скачок hard score происходят не одинаково.

<!-- /unit:a42b24eb3ce7a4b8d8 -->

<!-- unit:6251f682c6b38fab12 -->
<a id="u-6251f682c6b38fab12"></a>

## FCN и U-Net: пространственная структура

<!-- /unit:6251f682c6b38fab12 -->

<!-- unit:1f1593fdee9a3775cd -->
<a id="u-1f1593fdee9a3775cd"></a>

### До FCN: pixels, superpixels и пространственные модели

<!-- /unit:1f1593fdee9a3775cd -->

<!-- unit:ca4211115c0695bd52 -->
<a id="u-ca4211115c0695bd52"></a>

В разбираемом примере перечислялись pixel features с SVM/Random Forest, superpixels, MRF/CRF и sliding windows. Эти подходы не следует объявлять «не работавшими вообще». Они вводили разные предположения и давали полезные решения, но ограничивали обучаемую иерархию, стоимость вычислений или точность границ. Например, classifier по цвету различает некоторые материалы, но соседние pixels разного смысла могут иметь одинаковый RGB.

<!-- /unit:ca4211115c0695bd52 -->

<!-- unit:07c6610a568641cafd -->
<a id="u-07c6610a568641cafd"></a>

Superpixel объединяет похожие nearby pixels в region, для которого считают признаки и class scores. Так объектов классификации становится меньше, однако если один superpixel пересёк настоящую границу, последующий единый label уже не разделит её без refinement. MRF/CRF вводит согласованность labels и appearance relationships; она дополняет local classifier, а не обязательно создаёт semantic knowledge из одного соседства.

<!-- /unit:07c6610a568641cafd -->

<!-- unit:f09dacd57ea6a861bb -->
<a id="u-f09dacd57ea6a861bb"></a>

Sliding-window CNN использует context вокруг каждого центрального pixel, но независимо обрабатывать множество перекрывающихся окон дорого. FCN переиспользует общие convolutional features для многих output locations. Основная идея перехода — эффективность и совместно обучаемое dense prediction, а не внезапное появление пространственных данных, которых до этого якобы нигде не было.

<!-- /unit:f09dacd57ea6a861bb -->

<!-- unit:279d2b472ea7bcd364 -->
<a id="u-279d2b472ea7bcd364"></a>

Обычный classification head с flatten/global pooling теряет точную layout для dense outputs. Fully convolutional network сохраняет spatial feature maps и выдаёт class scores по сетке, затем upsamples к нужному resolution. Замена FC на conv — историческая идея, но одного nearest upsample обычно недостаточно для точных boundaries: глубокие maps уже потеряли часть деталей.

<!-- /unit:279d2b472ea7bcd364 -->

<!-- unit:5753490ef5d67d8c46 -->
<a id="u-5753490ef5d67d8c46"></a>

U-Net сочетает encoder downsampling и decoder upsampling, передавая high-resolution features через skip concatenations. Decoder получает и semantic context глубокой ветви, и spatial detail ранних уровней. [U-Net](https://arxiv.org/abs/1505.04597).

<!-- /unit:5753490ef5d67d8c46 -->

<!-- unit:4152776b995767416d -->
<a id="u-4152776b995767416d"></a>

Это отличается от residual sum ResNet: U-Net skip обычно связывает соответствующие resolutions encoder/decoder и часто concatenates channels. Если encoder feature $B\times 64\times 64\times 64$, decoder feature $B\times 128\times 64\times 64$, concat даёт $B\times 192\times 64\times 64$, не сумму несовместимых channels. Далее conv смешивает их. Cropping/padding/align conventions должны согласовать spatial coordinates.

<!-- /unit:4152776b995767416d -->

<!-- unit:401dbbd84dfbbddf44 -->
<a id="u-401dbbd84dfbbddf44"></a>

Upsampling может быть interpolation+conv или transposed convolution. Последняя — не буквальное обращение ранее выполненной conv и не автоматически восстанавливает потерянные pixels. Checkerboard artifacts возможны в некоторых конфигурациях, поэтому качество проверяют визуально.

<!-- /unit:401dbbd84dfbbddf44 -->

<!-- unit:9d18a34005eab1faea -->
<a id="u-9d18a34005eab1faea"></a>

### Что меняет FCN и откуда берутся FCN-32s/16s/8s

<!-- /unit:9d18a34005eab1faea -->

<!-- unit:d4db2560a5f3fdac30 -->
<a id="u-d4db2560a5f3fdac30"></a>

Если classifier ожидает feature map 7×7×512 и flatten→4096, каждый hidden unit имеет 7·7·512 коэффициентов. Эту операцию можно представить filter размером 7×7 на 512 channels с 4096 outputs. На исходном размере результат будет 1×1×4096; на большей spatial map filter применится в нескольких местах. Следующий полностью связный переход по channel dimension соответствует 1×1 convolution. Так фиксированное глобальное решение превращается в dense sliding computation с общими весами.

<!-- /unit:d4db2560a5f3fdac30 -->

<!-- unit:a362c9d15c40765e4f -->
<a id="u-a362c9d15c40765e4f"></a>

После пяти downsampling операций с фактором 2 карта имеет output stride 32. Один coarse score отвечает области, а простое увеличение карты не создаёт заново потерянные границы. FCN-16s объединяет upsampled deep scores с более подробными признаками предыдущей стадии; FCN-8s добавляет ещё более подробный путь. Их смысл — использовать и семантику глубокой ветви, и пространственную детализацию ранней, а не просто сменить коэффициент последнего resize.

<!-- /unit:a362c9d15c40765e4f -->

<!-- unit:c528ca7796f989f1bf -->
<a id="u-c528ca7796f989f1bf"></a>

У U-Net эта идея организована симметрично по нескольким resolutions. Например, encoder сохраняет карты 64×64,32×32,16×16. Decoder upsamples глубокую 16×16 до 32×32, соединяет с соответствующей encoder map, затем повторяет для 64×64. Совпадения числа пикселей недостаточно: padding/cropping должны согласовать, какой input location соответствует каждому output cell. Исходная U-Net использовала valid convolutions и cropping; современные padded варианты могут иметь другие промежуточные формы.

<!-- /unit:c528ca7796f989f1bf -->

<!-- unit:6b9fd3f7719592b38c -->
<a id="u-6b9fd3f7719592b38c"></a>

### Взвешивание границ из исходного разбора U-Net

<!-- /unit:6b9fd3f7719592b38c -->

<!-- unit:368e24aa449969d945 -->
<a id="u-368e24aa449969d945"></a>

В разбираемом примере приведена формула, усиливающая область между близкими экземплярами:

<!-- /unit:368e24aa449969d945 -->

<!-- unit:14f51cdda80dcca589 -->
<a id="u-14f51cdda80dcca589"></a>

$$
w(x)=w_c(x)+w_0\exp\!\left[-\frac{(d_1(x)+d_2(x))^2}{2\sigma^2}\right].
$$

<!-- /unit:14f51cdda80dcca589 -->

<!-- unit:6bba5c0cf1662c0c5a -->
<a id="u-6bba5c0cf1662c0c5a"></a>

Здесь d1 и d2 относятся к расстояниям до границ двух ближайших объектов, а не просто двум произвольным контурам одной клетки. Если их сумма мала, экспоненциальный добавочный вес велик: loss сильнее штрафует ошибки на узкой перемычке между объектами. Если $w_0=10$, $\sigma=5$ и $d_1+d_2=2$, прибавка около 9.231; при сумме 10 — около 1.353. Это способ изменить training emphasis, а не отдельный algorithm instance assignment на inference. [U-Net](https://arxiv.org/abs/1505.04597).

<!-- /unit:6bba5c0cf1662c0c5a -->

<!-- unit:d09f0a634521ab299e -->
<a id="u-d09f0a634521ab299e"></a>

U-Net++ изменяет пути передачи encoder/decoder features, attention gates регулируют их использование, residual variants меняют блоки преобразования, 3D U-Net заменяет пространственные операции на объёмные. Эти изменения решают разные ограничения. Название «3D» означает работу с дополнительной spatial axis, а не простое stacking независимых 2D масок. Pretraining и объём supervision всё равно важны: U-Net не гарантирует хорошую оценку на любом малом наборе.

<!-- /unit:d09f0a634521ab299e -->

<!-- unit:6f7e083b3059863abc -->
<a id="u-6f7e083b3059863abc"></a>

## Dilated convolution и DeepLab

<!-- /unit:6f7e083b3059863abc -->

<!-- unit:da6b93407cb4900f69 -->
<a id="u-da6b93407cb4900f69"></a>

Dilated/atrous conv раздвигает sampling positions kernel. Для $k\times k$ и dilation d effective receptive field одной операции $k_{\mathrm{eff}}=k+(k-1)(d-1)$. При $k=3$,$d=2$ размер охвата5, но обучаемых spatial weights всё ещё9, а не25. Он не читает все25 positions: внутри есть промежутки.

<!-- /unit:da6b93407cb4900f69 -->

<!-- unit:6e8dfa9802b601b767 -->
<a id="u-6e8dfa9802b601b767"></a>

Это расширяет context без обязательного нового downsampling. Но большой dilation может создавать gridding effects и терять локальные связи. ASPP объединяет branches с разными atrous rates и/или global context по конкретному DeepLab variant. Decoder уточняет boundaries; details DeepLab versions не одинаковы.

<!-- /unit:6e8dfa9802b601b767 -->

<!-- unit:0e21f2b59f7a299071 -->
<a id="u-0e21f2b59f7a299071"></a>

Output stride задаёт отношение input/output resolution глубокого feature map. Меньший stride сохраняет detail, увеличивая compute/memory. Нельзя считать output stride и dilation одним параметром: они связаны настройкой architecture, но описывают разное.

<!-- /unit:0e21f2b59f7a299071 -->

<!-- unit:8fb7bbd35959bc0a14 -->
<a id="u-8fb7bbd35959bc0a14"></a>

### Индексы dilation и receptive field целого стека

<!-- /unit:8fb7bbd35959bc0a14 -->

<!-- unit:392c6e165bc6bc24b8 -->
<a id="u-392c6e165bc6bc24b8"></a>

Для одномерной записи $y_i=\sum_m w_mx_{i+dm}$ видно, что dilation меняет адрес чтения, а не добавляет обучаемые коэффициенты. При kernel 3 и d=4 используются offsets −4,0,+4: охват равен 9, но внутри остаются пропуски. Двумерный filter берёт декартово произведение таких offsets.

<!-- /unit:392c6e165bc6bc24b8 -->

<!-- unit:2736b34a3e62a6ba1d -->
<a id="u-2736b34a3e62a6ba1d"></a>

Формула $RF=1+\sum_l(k_l-1)d_l$ верна для последовательного стека со stride=1 в общей входной сетке. При наличии stride нужно учитывать шаг между соседними признаками предыдущей стадии. Обозначим его j, начав с j0=1,R0=1:

<!-- /unit:2736b34a3e62a6ba1d -->

<!-- unit:7e3787ed19bf484e4c -->
<a id="u-7e3787ed19bf484e4c"></a>

$$
R_l=R_{l-1}+(k_l-1)d_lj_{l-1},\qquad j_l=j_{l-1}s_l.
$$

<!-- /unit:7e3787ed19bf484e4c -->

<!-- unit:2bd704c2ab7dc8d53c -->
<a id="u-2bd704c2ab7dc8d53c"></a>

Для conv k=3,s=2,d=1 получаем R1=3,j1=2. Следующая conv k=3,s=1,d=2 даёт R2=3+2·2·2=11. Наивная сумма без strides дала бы 7. Поэтому «большое receptive field» следует вычислять относительно исходного изображения, если сравниваются разные стадии.

<!-- /unit:2bd704c2ab7dc8d53c -->

<!-- unit:4969c2e04c647f20ad -->
<a id="u-4969c2e04c647f20ad"></a>

### ASPP: объединение контекстов с разным охватом

<!-- /unit:4969c2e04c647f20ad -->

<!-- unit:b97ae345f877211f0f -->
<a id="u-b97ae345f877211f0f"></a>

Параллельные atrous branches читают одну карту с разными rates. В common ASPP design к ним добавляют 1×1 branch и image pooling, после pooling возвращают spatial размер интерполяцией и конкатенируют каналы. Вектор GAP формы B×C×1×1 нельзя непосредственно concatenatе с B×C×H×W без такого согласования. Последующая projection смешивает локальную и широкую информацию.

<!-- /unit:b97ae345f877211f0f -->

<!-- unit:9bfe5c58476feb130e -->
<a id="u-9bfe5c58476feb130e"></a>

Rates 6,12,18 из разбираемого примера — пример конфигурации, связанный с output stride и variant. При смене stride их нужно осмысленно пересмотреть: rate=6 на карте stride16 охватывает другие input pixels, чем на stride8. DeepLabv1/v2 сочетали atrous features и CRF-подобное уточнение; v3 развивал ASPP, v3+ добавил decoder для границ. Общая история не означает один неизменный graph под всеми именами. [DeepLab](https://arxiv.org/abs/1606.00915), [DeepLabv3+](https://arxiv.org/abs/1802.02611).

<!-- /unit:9bfe5c58476feb130e -->

<!-- unit:5a1f16c2175eb9d282 -->
<a id="u-5a1f16c2175eb9d282"></a>

### Альтернативы из разбираемого примера: сохраняем различия механизмов

<!-- /unit:5a1f16c2175eb9d282 -->

<!-- unit:dfc625963b32dabd8c -->
<a id="u-dfc625963b32dabd8c"></a>

SegNet переносит pooling indices в unpooling: decoder знает locations прежних максимумов, но не восстанавливает все отброшенные значения. PSPNet агрегирует контекст pooling по сеткам нескольких размеров; это не то же, что раздвигать sampling kernel dilation. HRNet поддерживает несколько ветвей разрешения и обменивается ими, сохраняя high-resolution path вместо единственного глубокого bottleneck.

<!-- /unit:dfc625963b32dabd8c -->

<!-- unit:1ce0771b6d1d34fa05 -->
<a id="u-1ce0771b6d1d34fa05"></a>

SegFormer сочетает иерархические Transformer features с лёгким decoder; Mask2Former задаёт masks через query-based prediction и masked attention и применяется к разным segmentation formulations. SAM получает prompt, например точки или box, и возвращает candidate masks; он сам по себе не решает задачу назначения всем pixels фиксированных пользовательских class IDs. Эти названия полезны как варианты по конкретному ограничению, а не рейтинг «самая новая всегда лучшая». Одно и то же backbone family допускает разные heads и обучающие задачи.

<!-- /unit:1ce0771b6d1d34fa05 -->

<!-- unit:070f1d79d0617cf9b1 -->
<a id="u-070f1d79d0617cf9b1"></a>

## Semantic и instance heads в YOLO-семействе

<!-- /unit:070f1d79d0617cf9b1 -->

<!-- unit:771f2cf45649e66cdc -->
<a id="u-771f2cf45649e66cdc"></a>

В исходнике semantic head описывалась как общая «урезанная U-Net внутри YOLO». Такое тождество не следует из имени. Отдельная semantic model должна выдавать dense classes; instance segmentation head выдаёт маски объектов. Auxiliary semantic loss при обучении instance model ещё не доказывает наличие отдельного поддерживаемого semantic inference output.

<!-- /unit:771f2cf45649e66cdc -->

<!-- unit:53e996a6ad026729c2 -->
<a id="u-53e996a6ad026729c2"></a>

На текущей проверке официальная YOLO26 документация перечисляет отдельные `*-sem` модели; это конкретная implementation, которую нужно отличать от `*-seg`. Здесь изучается задача и путь features, а для запуска сверяют shape outputs, checkpoints и data config. [Поддерживаемые задачи](https://docs.ultralytics.com/tasks/).

<!-- /unit:53e996a6ad026729c2 -->

<!-- unit:9d3841984432c780e5 -->
<a id="u-9d3841984432c780e5"></a>

### Что могут и чего не могут post-processing и TTA

<!-- /unit:9d3841984432c780e5 -->

<!-- unit:ea24d65e2d7a6c0a17 -->
<a id="u-ea24d65e2d7a6c0a17"></a>

Классический CRF из рассматриваемого примера сочетает unary cost от предсказаний сети и pairwise cost пространственной совместимости labels. Unary удерживает правдоподобие класса, pairwise может поощрять одинаковые labels у похожих nearby pixels и разрешать границы при сильном различии внешнего вида. Это модель предположений: визуально похожие соседние pixels могут принадлежать разным объектам, а нужная semantic граница не всегда совпадает с цветовой.

<!-- /unit:ea24d65e2d7a6c0a17 -->

<!-- unit:1a36802361c7489201 -->
<a id="u-1a36802361c7489201"></a>

Opening и closing тоже меняют целевую геометрию. Opening может убрать маленькие islands вместе с настоящими мелкими объектами; closing — закрыть отверстие, которое важно в разметке. Их structuring element выбирают с учётом размера и topology, оценивая итоговую mask. «Стало визуально гладко» не доказывает улучшение качества на тонких структурах.

<!-- /unit:1a36802361c7489201 -->

<!-- unit:76285e84ce18c081f5 -->
<a id="u-76285e84ce18c081f5"></a>

При test-time flip сначала выполняют inverse spatial transform карт scores, затем совмещают их с исходной сеткой и агрегируют. Усреднение несопоставленных flipped maps смешает левую и правую части изображения. Multi-scale TTA увеличивает вычисления и может менять ошибки разных размеров; его latency и настройку включают в протокол сравнения. Подбор постобработки на test masks недопустимо выдавать за независимую оценку.

<!-- /unit:76285e84ce18c081f5 -->

## Архитектуры: подробные схемы

### Fully Convolutional Network

![Fully Convolutional Network](<../оформление/course/arch-fcn.svg>)

FCN сохраняет пространственное размещение scores вместо глобального FC classifier. Upsampling возвращает сетку изображения; точность границ зависит от разрешения и способов слияния признаков.

### U-Net: skip concatenation

![U-Net: skip concatenation](<../оформление/course/arch-unet.svg>)

Сохранённая encoder map поступает в decoder в обход bottleneck. После согласования сеток выполняется concat, затем convolution/head. В отличие от residual-суммы concat увеличивает число channels; это должно учитываться следующим слоем.

### DeepLab и ASPP

![DeepLab и ASPP](<../оформление/course/arch-deeplab.svg>)

Dilated filters расширяют рецептивное поле без эквивалентного роста числа sampled coefficients. ASPP объединяет несколько масштабов контекста; image pooling и decoder зависят от версии DeepLab.

### SegNet

![SegNet](<../оформление/course/arch-segnet.svg>)

Pooling indices указывают места максимумов при unpooling. Они не сохраняют все отброшенные значения encoder и не равнозначны U-Net concatenation полных карт.

### PSPNet

![PSPNet](<../оформление/course/arch-pspnet.svg>)

Pyramid pooling агрегирует контекст на нескольких сетках. Upsampling согласует размеры перед concat; это отличается от параллельных dilated filters ASPP.

### HRNet

![HRNet](<../оформление/course/arch-hrnet.svg>)

HRNet сохраняет high-resolution ветвь и обменивает информацию между разрешениями. Конкретная task head задаёт heatmaps или классы; grouping людей не возникает автоматически из backbone.

### SegFormer

![SegFormer](<../оформление/course/arch-segformer.svg>)

Иерархический encoder даёт карты разных масштабов; лёгкий decoder согласует их и предсказывает классы. Формы stages и ширины относятся к конкретной конфигурации.

### Mask2Former

![Mask2Former](<../оформление/course/arch-mask2former.svg>)

Queries предсказывают mask и class; masked attention ограничивает доступ к spatial features по промежуточным masks. Финальная сборка выхода зависит от выбранной segmentation formulation.

<!-- unit:6906a2619f5c12683e -->
<a id="u-6906a2619f5c12683e"></a>

## Самопроверка и практика

<!-- /unit:6906a2619f5c12683e -->

<!-- unit:4732e409f281bc3f6b -->
<a id="u-4732e409f281bc3f6b"></a>

1. Почему две соседние собаки не разделятся в semantic mask? **Ответ:** задача кодирует class, а не instance ID.
2. Сколько весов у одного $3\times 3$ filter с dilation2? **Ответ:** девять на input channel, dilation не добавляет25 weights.
3. Почему bilinear resize IDs ошибочен? **Ответ:** создаёт несуществующие промежуточные labels и меняет смысл категорий.
4. Когда 90% pixel accuracy бесполезны? **Ответ:** например, при полном пропуске малого важного foreground.

<!-- /unit:4732e409f281bc3f6b -->

<!-- unit:50c4e0dcb8e1645553 -->
<a id="u-50c4e0dcb8e1645553"></a>

**Практика:** нарисовать две binary masks $5\times 5$, вручную определить intersection/union/Dice и добавить ignored row. Затем проследить shapes маленькой U-Net, указав concat channel counts. Для реальных данных сравнить один CE baseline с одной Dice/weight intervention, сохраняя split, input resolution и valid-pixel protocol.

<!-- /unit:50c4e0dcb8e1645553 -->

<!-- unit:fb62dc3eaf9483230f -->
<a id="u-fb62dc3eaf9483230f"></a>

**Схема курса:** image tensor → backbone feature scales → upsampling/skip/context head → per-pixel logits → ignore-mask loss → resize to original coordinates → IoU/Dice

<!-- /unit:fb62dc3eaf9483230f -->

<!-- unit:d9444c97d0101ed8c7 -->
<a id="u-d9444c97d0101ed8c7"></a>

<!-- latex-display-v1 -->

<!-- /unit:d9444c97d0101ed8c7 -->

<!-- unit:f17a18040ff7704fb0 -->
<a id="u-f17a18040ff7704fb0"></a>

<!-- source-depth-S7-В -->

<!-- /unit:f17a18040ff7704fb0 -->

## Результат главы и переход дальше

Ответьте на вопрос: **Как предсказывать класс пикселя и восстанавливать пространственную сетку?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

Следующие темы: [Instance segmentation: Mask R-CNN и prototype masks](<cv-instance.md>).

[Маршрут модуля](<README.md>) · [Происхождение и точные места переноса](<../editorial/coverage.md>)
