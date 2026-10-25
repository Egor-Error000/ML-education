# Semantic segmentation: FCN, U-Net и DeepLab

**Уровень:** общее ядро.

**Предпосылки:** [Изображение, карты признаков и блоки модели](<cv-input.md>), [Классификация, CNN и residual learning](<cv-classification.md>)

**Используют:** [U-Net и nnU-Net: пространственный decoder и настройка pipeline](<cv-unet.md>), [DeepLabv3+ и SegFormer: контекст и декодирование](<cv-segformers.md>), [Instance segmentation: Mask R-CNN и prototype masks](<cv-instance.md>), [CRAFT и DBNet: поиск текста на изображении](<cv-text-detection.md>), [Диагностика CV: ошибки данных, модели и post-processing](<cv-diagnosis.md>)

**Учебный вопрос:** Как предсказывать класс пикселя и восстанавливать пространственную сетку?

## Карта вычисления и границы этапов

![Semantic segmentation: FCN, U-Net и DeepLab: входы, преобразования и выходы](<../оформление/course/cv-semantic.svg>)

Сегментация сохраняет пространственную привязку ответа. Skip-связи возвращают детали высокого разрешения; восстановленный размер карты сам по себе не гарантирует точность границ.

## Что выдаётся на каждом пикселе

Модель возвращает logits (логиты, оценки альтернатив до нормировки) $B\times K\times H_{\mathrm{out}}\times W_{\mathrm{out}}$; после согласования resolution каждый pixel получает class ID. Две соприкасающиеся собаки имеют один class «собака»: semantic output не различает их instance identities. Background — обычный class только если так определена разметка; void/ignore — отдельная политика, а не ещё один обучаемый класс автоматически.

Ground truth (эталонная разметка) для mutually exclusive classes обычно integer mask $B\times H\times W$. Цветной preview может хранить palette, но colour RGB не обязательно равен class ID. Если исходные mask IDs 0,1,2, это не интенсивности изображения: нельзя применять к ним bilinear interpolation как к photo.

Аугментации геометрии синхронизируют image и mask. Для categorical mask обычно используют nearest interpolation и проверяют border fill/ignore. Для logits перед argmax допустимо другое resize; порядок «resize logits → argmax» и «argmax → resize IDs» даёт разные boundaries и должен соответствовать protocol.

### Почему «предсказать каждый пиксель» не означает решить его независимо

Если пиксель имеет серый цвет, по одному RGB нельзя надёжно отличить дорогу от крыши. Модель получает изображение целиком или crop и использует окружение: горизонт, соседние текстуры, форму региона. Pixelwise относится к форме ответа и месту supervision, а не к требованию рассматривать каждый входной pixel изолированно. Точно так же CE по отдельным позициям не запрещает общие признаки и пространственный контекст.

Ground-truth маска и probabilities имеют разный смысл. Integer mask хранит один ID на location. Для K классов модель хранит K logits на той же location, softmax нормируется по классовой оси, и argmax выбирает один ID. Число цветов palette PNG не является автоматически K: могут присутствовать ignored values, а последовательность ID может требовать remapping. Читать palette preview как обычную RGB-картинку и брать её среднюю яркость нельзя.

Представим labels (метки) [0,2] на соседних пикселях. Bilinear interpolation в середине вернёт 1, хотя класса 1 в этой области вообще не было. Nearest выбирает существующий label (метка). Для floating-point class scores (оценки) интерполяция допустима как часть заданного decoder (декодер)/postprocessing: интерполируют K вещественных карт, затем принимают решение. Это два разных объекта, несмотря на одинаковую spatial сетку.

В разбираемом примере классы одного типа описывались как «слитые в одну маску». Это не обязательно один связный регион: semantic mask «кошка» может содержать три раздельных островка. Она не кодирует, каким из них принадлежит общий instance ID. Connected components иногда отделят раздельные островки, но соприкоснувшиеся кошки останутся одним компонентом, а одна кошка за забором может распасться на несколько. Геометрическая связность не заменяет разметку экземпляров.

## Pixel accuracy, IoU и mIoU

Для класса k $\operatorname{IoU}_k=\frac{\mathrm{TP}_k}{\mathrm{TP}_k+\mathrm{FP}_k+\mathrm{FN}_k}$. Pixel accuracy=$\frac{\sum_k\mathrm{TP}_k}{N_{\mathrm{valid}}}$; mIoU усредняет выбранные class IoU. Ignore pixels исключаются из соответствующих counts. Classes absent в GT и prediction имеют нулевой denominator; evaluator должен явно определить их включение/исключение.

Учебная mask из 100 pixels содержит 10 foreground и90 background. Модель предсказала background везде. Pixel accuracy90%. Для foreground TP0,FP0,FN10 → IoU0. Для background TP90,FP10,FN0 → IoU0.9. Mean по двум classes=0.45. Это объясняет, почему высокая accuracy не достаточна для малого объекта.

Другой пример: истинных foreground10, predicted foreground12, intersection8. Тогда TP8,FP4,FN2, $\operatorname{IoU}=\frac{8}{14}\approx 0.57143$, $Dice=\frac{16}{22}\approx 0.72727$. На binary sets $\mathrm{Dice}=\frac{2\operatorname{IoU}}{1+\operatorname{IoU}}$. При усреднении классов и soft losses со smoothing/weights нельзя без проверки превращать одну общую метрику в другую той же формулой.

Boundary quality может быть плохой при неплохой area IoU: тонкие structures и несколько сдвинутых pixels существенно влияют на практическую пользу. Добавлять boundary-specific evaluation по задаче, а не заменять ею целиком segmentation quality.

### Вывод метрик из одной confusion matrix

Для трёх классов рассмотрим counts, где строки — истина, столбцы — предсказание:

$$
C=\begin{bmatrix}50&5&0\\3&20&2\\0&4&16\end{bmatrix}.
$$

Всего 100 valid pixels, диагональ содержит 86 верных, pixel accuracy=0.86. Для класса 0 TP=50, FP=3, FN=5, поэтому IoU=50/58≈0.862069. Для класса 1 TP=20, FP=9, FN=5, IoU=20/34≈0.588235. Для класса 2 TP=16, FP=2, FN=4, IoU=16/22≈0.727273. mIoU≈0.725859. Этот результат заметно ниже accuracy, поскольку ошибки малых классов не растворились в числе правильных background pixels.

Frequency-weighted IoU из рассматриваемого примера использует доли GT классов: 0.55,0.25,0.20. Взвешенная сумма равна примерно 0.766651. Это другое усреднение, с большим влиянием частого класса. Ни mIoU, ни fwIoU не являются ошибкой вычисления только из-за расхождения с accuracy: они отвечают на разные вопросы.

Для каждого класса $TP_k=C_{kk}$, $FN_k=\sum_jC_{kj}-C_{kk}$, $FP_k=\sum_iC_{ik}-C_{kk}$. Если ignored row pixels не удалить до накопления counts, они исказят сразу несколько показателей. Также нельзя сначала посчитать per-image mIoU и объявить среднее тем же, что IoU по общей confusion matrix: отношение сумм и среднее отношений в общем случае различаются.

## Обучение CE и Dice

Pixelwise CE $-\log p_{\mathrm{true\ class}}$ усредняется по valid pixels с возможными class weights. При сильном class imbalance большие background regions доминируют. Class weights, focal loss (функция потерь), sampling (случайный выбор по распределению) crops с foreground и Dice-family objectives меняют signal разными способами.

Soft Dice для binary mask g и probabilities p можно задать:

$$
D_{\mathrm{soft}}=\frac{2\sum_i p_ig_i+\varepsilon}{\sum_i p_i+\sum_i g_i+\varepsilon},\qquad\mathcal{L}_{\mathrm{Dice}}=1-D_{\mathrm{soft}}.
$$

Есть variants с квадратами denominator, per-image/per-batch averaging и разной обработкой пустого foreground. Нужно объявить convention. Dice зависит от совокупности pixels, тогда как CE даёт локальный signal. Смесь CE+λDice часто рассматривают как баланс, но λ подбирается на validation (проверочная часть для выбора решения), не по универсальной константе.

Ignore-index исключает pixel supervision. Его нужно согласовать в loss и metrics: иначе «неразмеченное» может случайно стать background. Empty mask тоже содержит информацию, если scene действительно без objects; это не повод автоматически исключить все такие изображения.

### Локальный CE и глобальная нормировка Dice

Для одного binary foreground logit z вероятность $p=\sigma(z)$, а BCE равна $-[g\ln p+(1-g)\ln(1-p)]$. Производная по z — p−g. Каждый valid pixel даёт свой прямой сигнал. Если 9900 background pixels уже немного ошибаются, их суммарный вклад всё ещё может превосходить вклад ста foreground pixels. Усреднение по всему изображению не делает классы равновесными.

Weighted CE меняет стоимость ошибок классов; sampling меняет частоту примеров, видимых optimizer (оптимизатор обновления параметров); focal меняет вес в зависимости от текущей сложности. Они не взаимозаменяемы. Правило $w_k=1/f_k$ из разбираемого примера — возможная эвристика, а не автоматический безопасный выбор: очень редкий шумный класс способен получить огромный вес. Для weighted reduction нужно также определить знаменатель — число pixels либо сумму активных weights.

Для soft Dice обозначим $A=2\sum_i p_ig_i+\varepsilon$ и $B=\sum_i p_i+\sum_i g_i+\varepsilon$. Тогда

$$
\frac{\partial(1-A/B)}{\partial p_j}=\frac{A-2g_jB}{B^2}.
$$

Градиент одного pixel зависит от sums по всей маске. Поэтому Dice связывает вклад размера region и overlap; CE отдельно штрафует неверную вероятность каждой цели. При p=(0.8,0.2), g=(1,0), без smoothing получаем A=1.6,B=2,Dice=0.8. Производные loss по p равны (−0.6,0.4): foreground вероятность надо увеличивать, background уменьшать. Это не те же масштабы, что у BCE.

Для пустой GT маски без smoothing числитель A=0. При ненулевой prediction sum Dice=0, но производная выбранного варианта по всем p тоже нулевая: один этот objective (оптимизируемая целевая функция) не обучает подавлять пустой foreground. Epsilon и особые empty-mask правила меняют поведение; CE может давать нужный локальный сигнал. Поэтому обещание «Dice работает при дисбалансе» требует конкретной формулы и политики пустых targets (целевые значения).

В разбираемом примере также вводился soft IoU: $I=\sum p_ig_i$, $U=\sum p_i+\sum g_i-I$, loss=1−I/U. При тех же p/g soft IoU=0.8/1.2=2/3, в отличие от Dice=0.8. Это непрерывные training surrogates; hard metric после threshold может быть равна 1, если бинарная маска уже верна. Улучшение loss и скачок hard score (оценка) происходят не одинаково.

## FCN и U-Net: пространственная структура

### До FCN: pixels, superpixels и пространственные модели

В разбираемом примере перечислялись pixel features (признаки) с SVM/Random Forest, superpixels, MRF/CRF и sliding windows. Эти подходы не следует объявлять «не работавшими вообще». Они вводили разные предположения и давали полезные решения, но ограничивали обучаемую иерархию, стоимость вычислений или точность границ. Например, classifier по цвету различает некоторые материалы, но соседние pixels разного смысла могут иметь одинаковый RGB.

Superpixel объединяет похожие nearby pixels в region, для которого считают признаки и class scores. Так объектов классификации становится меньше, однако если один superpixel пересёк настоящую границу, последующий единый label уже не разделит её без refinement. MRF/CRF вводит согласованность labels и appearance relationships; она дополняет local classifier, а не обязательно создаёт semantic knowledge из одного соседства.

Sliding-window CNN использует context (доступный модели контекст) вокруг каждого центрального pixel, но независимо обрабатывать множество перекрывающихся окон дорого. FCN переиспользует общие convolutional features для многих output locations. Основная идея перехода — эффективность и совместно обучаемое dense prediction, а не внезапное появление пространственных данных, которых до этого якобы нигде не было.

Обычный classification head с flatten/global pooling (агрегирование состояний) теряет точную layout для dense outputs. Fully convolutional network сохраняет spatial feature (признак) maps и выдаёт class scores по сетке, затем upsamples к нужному resolution. Замена FC на conv — историческая идея, но одного nearest upsample обычно недостаточно для точных boundaries: глубокие maps уже потеряли часть деталей.

U-Net сочетает encoder (кодировщик входа) downsampling (уменьшение пространственного разрешения) и decoder upsampling (повышение пространственного разрешения), передавая high-resolution features через skip concatenations. Decoder получает и semantic context глубокой ветви, и spatial detail ранних уровней. [U-Net](https://arxiv.org/abs/1505.04597).

Это отличается от residual sum ResNet: U-Net skip обычно связывает соответствующие resolutions encoder/decoder и часто concatenates channels. Если encoder feature $B\times 64\times 64\times 64$, decoder feature $B\times 128\times 64\times 64$, concat даёт $B\times 192\times 64\times 64$, не сумму несовместимых channels. Далее conv смешивает их. Cropping/padding (дополнение до общей длины)/align conventions должны согласовать spatial coordinates.

Upsampling может быть interpolation+conv или transposed convolution. Последняя — не буквальное обращение ранее выполненной conv и не автоматически восстанавливает потерянные pixels. Checkerboard artifacts возможны в некоторых конфигурациях, поэтому качество проверяют визуально.

### Что меняет FCN и откуда берутся FCN-32s/16s/8s

Если classifier ожидает feature map (пространственная карта признаков) 7×7×512 и flatten→4096, каждый hidden unit имеет 7·7·512 коэффициентов. Эту операцию можно представить filter размером 7×7 на 512 channels с 4096 outputs. На исходном размере результат будет 1×1×4096; на большей spatial map filter применится в нескольких местах. Следующий полностью связный переход по channel dimension соответствует 1×1 convolution. Так фиксированное глобальное решение превращается в dense sliding computation с общими весами.

После пяти downsampling операций с фактором 2 карта имеет output stride (шаг пространственного перемещения операции) 32. Один coarse score отвечает области, а простое увеличение карты не создаёт заново потерянные границы. FCN-16s объединяет upsampled deep scores с более подробными признаками предыдущей стадии; FCN-8s добавляет ещё более подробный путь. Их смысл — использовать и семантику глубокой ветви, и пространственную детализацию ранней, а не просто сменить коэффициент последнего resize.

У U-Net эта идея организована симметрично по нескольким resolutions. Например, encoder сохраняет карты 64×64,32×32,16×16. Decoder upsamples глубокую 16×16 до 32×32, соединяет с соответствующей encoder map, затем повторяет для 64×64. Совпадения числа пикселей недостаточно: padding/cropping должны согласовать, какой input location соответствует каждому output cell. Исходная U-Net использовала valid convolutions и cropping; современные padded варианты могут иметь другие промежуточные формы.

### Взвешивание границ в U-Net

В разбираемом примере приведена формула, усиливающая область между близкими экземплярами:

$$
w(x)=w_c(x)+w_0\exp\!\left[-\frac{(d_1(x)+d_2(x))^2}{2\sigma^2}\right].
$$

Здесь d1 и d2 относятся к расстояниям до границ двух ближайших объектов, а не просто двум произвольным контурам одной клетки. Если их сумма мала, экспоненциальный добавочный вес велик: loss сильнее штрафует ошибки на узкой перемычке между объектами. Если $w_0=10$, $\sigma=5$ и $d_1+d_2=2$, прибавка около 9.231; при сумме 10 — около 1.353. Это способ изменить training emphasis, а не отдельный algorithm instance assignment на inference (применение обученной модели). [U-Net](https://arxiv.org/abs/1505.04597).

U-Net++ изменяет пути передачи encoder/decoder features, attention (механизм внимания) gates регулируют их использование, residual variants меняют блоки преобразования, 3D U-Net заменяет пространственные операции на объёмные. Эти изменения решают разные ограничения. Название «3D» означает работу с дополнительной spatial axis, а не простое stacking независимых 2D масок. Pretraining (предобучение) и объём supervision всё равно важны: U-Net не гарантирует хорошую оценку на любом малом наборе.

## Dilated convolution и DeepLab

Dilated/atrous conv раздвигает sampling positions kernel. Для $k\times k$ и dilation d effective receptive field (область входа, влияющая на элемент карты) одной операции $k_{\mathrm{eff}}=k+(k-1)(d-1)$. При $k=3$,$d=2$ размер охвата5, но обучаемых spatial weights всё ещё9, а не25. Он не читает все25 positions: внутри есть промежутки.

Это расширяет context без обязательного нового downsampling. Но большой dilation может создавать gridding effects и терять локальные связи. ASPP объединяет branches с разными atrous rates и/или global context по конкретному DeepLab variant. Decoder уточняет boundaries; details DeepLab versions не одинаковы.

Output stride задаёт отношение input/output resolution глубокого feature map. Меньший stride сохраняет detail, увеличивая compute/memory. Нельзя считать output stride и dilation одним параметром: они связаны настройкой architecture, но описывают разное.

### Индексы dilation и receptive field целого стека

Для одномерной записи $y_i=\sum_m w_mx_{i+dm}$ видно, что dilation меняет адрес чтения, а не добавляет обучаемые коэффициенты. При kernel 3 и d=4 используются offsets −4,0,+4: охват равен 9, но внутри остаются пропуски. Двумерный filter берёт декартово произведение таких offsets.

Формула $RF=1+\sum_l(k_l-1)d_l$ верна для последовательного стека со stride=1 в общей входной сетке. При наличии stride нужно учитывать шаг между соседними признаками предыдущей стадии. Обозначим его j, начав с j0=1,R0=1:

$$
R_l=R_{l-1}+(k_l-1)d_lj_{l-1},\qquad j_l=j_{l-1}s_l.
$$

Для conv k=3,s=2,d=1 получаем R1=3,j1=2. Следующая conv k=3,s=1,d=2 даёт R2=3+2·2·2=11. Наивная сумма без strides дала бы 7. Поэтому «большое receptive field» следует вычислять относительно исходного изображения, если сравниваются разные стадии.

### ASPP: объединение контекстов с разным охватом

Параллельные atrous branches читают одну карту с разными rates. В common ASPP design к ним добавляют 1×1 branch и image pooling, после pooling возвращают spatial размер интерполяцией и конкатенируют каналы. Вектор GAP формы B×C×1×1 нельзя непосредственно concatenatе с B×C×H×W без такого согласования. Последующая projection смешивает локальную и широкую информацию.

Rates 6,12,18 из разбираемого примера — пример конфигурации, связанный с output stride и variant. При смене stride их нужно осмысленно пересмотреть: rate=6 на карте stride16 охватывает другие input pixels, чем на stride8. DeepLabv1/v2 сочетали atrous features и CRF-подобное уточнение; v3 развивал ASPP, v3+ добавил decoder для границ. Общая история не означает один неизменный graph под всеми именами. [DeepLab](https://arxiv.org/abs/1606.00915), [DeepLabv3+](https://arxiv.org/abs/1802.02611).

### Альтернативы из разбираемого примера: сохраняем различия механизмов

SegNet переносит pooling indices в unpooling: decoder знает locations прежних максимумов, но не восстанавливает все отброшенные значения. PSPNet агрегирует контекст pooling по сеткам нескольких размеров; это не то же, что раздвигать sampling kernel dilation. HRNet поддерживает несколько ветвей разрешения и обменивается ими, сохраняя high-resolution path вместо единственного глубокого bottleneck.

SegFormer сочетает иерархические Transformer features с лёгким decoder; Mask2Former задаёт masks через query-based prediction и masked attention и применяется к разным segmentation formulations. SAM получает prompt (входной запрос или инструкция), например точки или box, и возвращает candidate masks; он сам по себе не решает задачу назначения всем pixels фиксированных пользовательских class IDs. Эти названия полезны как варианты по конкретному ограничению, а не рейтинг «самая новая всегда лучшая». Одно и то же backbone (основная сеть извлечения признаков) family допускает разные heads и обучающие задачи.

## Semantic и instance heads в YOLO-семействе

В исходнике semantic head описывалась как общая «урезанная U-Net внутри YOLO». Такое тождество не следует из имени. Отдельная semantic model должна выдавать dense classes; instance segmentation head выдаёт маски объектов. Auxiliary semantic loss при обучении instance model ещё не доказывает наличие отдельного поддерживаемого semantic inference output.

На текущей проверке официальная YOLO26 документация перечисляет отдельные `*-sem` модели; это конкретная implementation, которую нужно отличать от `*-seg`. Здесь изучается задача и путь features, а для запуска сверяют shape outputs, checkpoints и data config. [Поддерживаемые задачи](https://docs.ultralytics.com/tasks/).

### Что могут и чего не могут post-processing и TTA

Классический CRF из рассматриваемого примера сочетает unary cost от предсказаний сети и pairwise cost пространственной совместимости labels. Unary удерживает правдоподобие класса, pairwise может поощрять одинаковые labels у похожих nearby pixels и разрешать границы при сильном различии внешнего вида. Это модель предположений: визуально похожие соседние pixels могут принадлежать разным объектам, а нужная semantic граница не всегда совпадает с цветовой.

Opening и closing тоже меняют целевую геометрию. Opening может убрать маленькие islands вместе с настоящими мелкими объектами; closing — закрыть отверстие, которое важно в разметке. Их structuring element выбирают с учётом размера и topology, оценивая итоговую mask. «Стало визуально гладко» не доказывает улучшение качества на тонких структурах.

При test-time flip сначала выполняют inverse spatial transform карт scores, затем совмещают их с исходной сеткой и агрегируют. Усреднение несопоставленных flipped maps смешает левую и правую части изображения. Multi-scale TTA увеличивает вычисления и может менять ошибки разных размеров; его latency (задержка отдельного запроса) и настройку включают в протокол сравнения. Подбор постобработки на test masks недопустимо выдавать за независимую оценку.

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

## Самопроверка и практика

1. Почему две соседние собаки не разделятся в semantic mask? **Ответ:** задача кодирует class, а не instance ID.
2. Сколько весов у одного $3\times 3$ filter с dilation2? **Ответ:** девять на input channel, dilation не добавляет25 weights.
3. Почему bilinear resize IDs ошибочен? **Ответ:** создаёт несуществующие промежуточные labels и меняет смысл категорий.
4. Когда 90% pixel accuracy бесполезны? **Ответ:** например, при полном пропуске малого важного foreground.

**Практика:** нарисовать две binary masks $5\times 5$, вручную определить intersection/union/Dice и добавить ignored row. Затем проследить shapes маленькой U-Net, указав concat channel counts. Для реальных данных сравнить один CE baseline (базовый вариант сравнения) с одной Dice/weight intervention, сохраняя split (разбиение данных), input resolution и valid-pixel protocol.

**Схема курса:** image tensor → backbone feature scales → upsampling/skip/context head → per-pixel logits → ignore-mask loss → resize to original coordinates → IoU/Dice





## Результат главы и переход дальше

Следующие темы: [Instance segmentation: Mask R-CNN и prototype masks](<cv-instance.md>).

[Маршрут модуля](<README.md>) · [Происхождение и точные места переноса](<../editorial/coverage.md>)

## Как читать углубления этой ветки


Semantic mask задаёт class каждого valid pixel. Подробнее [U-Net/nnU-Net](<cv-unet.md>) раскрывает spatial decoder и pipeline, [DeepLabv3+/SegFormer](<cv-segformers.md>) — способы контекста, [Mask2Former](<cv-mask2former.md>) — mask classification. Для dataset preparation используйте [главу изображений](<../01-data-features/03-modalities/df-images.md>).

Ignore region — область без ordinary target, а не дополнительный background. Loss и confusion counts должны исключать её согласованно. Binary foreground, mutually exclusive multiclass и overlapping multi-label masks задают разные output heads.

## Разобранный пример для проверки контракта


Для foreground GT $(1,1,0,0)$ и hard prediction $(1,0,1,0)$ имеем TP=1,FP=1,FN=1,TN=1. IoU foreground $1/3$, Dice $2/(2+1+1)=1/2$, pixel accuracy $2/4=1/2$. Если последний pixel ignore, valid accuracy станет $1/3$; считать ignored правильным background неверно.

## Ограничения переноса

IoU больших областей может скрывать ошибку тонких boundaries и маленьких classes. Метрики дополнительно разбивают по class и scale.

## Самопроверка с ответами

**Практика:** исключите тот же ignored pixel также из loss. Denominator valid pixels изменится; так loss и evaluation относятся к одному множеству targets.

**Всегда ли высокий pixel accuracy полезен?** При большом background classifier «всё фон» может иметь высокий score и нулевой recall объекта.

**Почему resize mask bilinear опасен?** Он смешивает class IDs; categorical mask обычно требует nearest-neighbour либо специального soft-target контракта.

## Схема вычислений

![Semantic segmentation: FCN, U-Net и DeepLab: вычислительные этапы](<../оформление/course/expanded-cv-semantic.svg>)

Semantic contract: Image + class mask + ignore → Encoder → decoder features → B×K×H×W logits → Valid loss → confusion / mIoU.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Источники и продолжение

- [DeepLabv3+ — Chen et al., ECCV 2018](https://arxiv.org/abs/1802.02611)

[Следующие углубления: cv-unet, cv-segformers, cv-instance, cv-text-detection, cv-diagnosis](<README.md>) · [Сценарии подраздела](<04-segmentation/План.md>)
