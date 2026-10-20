# Depth: геометрия, масштаб и DPT

**Уровень:** основа ветки.

**Предпосылки:** [Изображение, карты признаков и блоки модели](<cv-input.md>), [Классификация, CNN и residual learning](<cv-classification.md>)

**Используют:** Выбранная прикладная задача и итоговая проверка результата.

**Учебный вопрос:** Что можно восстановить из одного изображения и какие допущения нужны для масштаба?





## Карта вычисления и границы этапов

![Depth: геометрия, масштаб и DPT: входы, преобразования и выходы](<../оформление/course/cv-depth.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Сравните формы и смысл каждого перехода с формулами главы. Выход преобразования и доказательство его качества — разные объекты: последняя проверка использует отдельный протокол оценки.

<!-- unit:5de1f7e0c2d5dd1914 -->
<a id="u-5de1f7e0c2d5dd1914"></a>

## Одно изображение не определяет единственную 3D сцену

<!-- /unit:5de1f7e0c2d5dd1914 -->

<!-- unit:208067691801af8516 -->
<a id="u-208067691801af8516"></a>

В pinhole модели $u=f_x\frac XZ+c_x$, $v=f_y\frac YZ+c_y$. Если одновременно умножить X,Y,Z сцены на a, image projection не меняется. Без дополнительной информации absolute scale не идентифицируется из одной такой проекции. Модель использует training priors: типичные размеры объектов, perspective, occlusion, texture, semantic scene structure.

<!-- /unit:208067691801af8516 -->

<!-- unit:0a1853db1ad1f5d914 -->
<a id="u-0a1853db1ad1f5d914"></a>

Это позволяет полезные predictions, но не гарантирует физическую истинность новой необычной сцены. Игрушечная машинка рядом и большая машина далеко могут занимать похожую image area. Defocus зависит от focus distance/aperture и может возрастать по обе стороны focus plane; правило «дальше всегда более размыто» не универсально.

<!-- /unit:0a1853db1ad1f5d914 -->

<!-- unit:cca92f22ee9886ee0e -->
<a id="u-cca92f22ee9886ee0e"></a>

Metric depth задаёт величину в units, например metres, по определённому camera/protocol. Relative output может отражать ordering и иметь неизвестные scale/shift; некоторые models предсказывают inverse depth-like quantity. Прежде чем переводить colour preview в metres, надо узнать точный output contract checkpoint.

<!-- /unit:cca92f22ee9886ee0e -->

<!-- unit:af1c64e23ff045a18f -->
<a id="u-af1c64e23ff045a18f"></a>

### Что именно называется расстоянием на depth map

<!-- /unit:af1c64e23ff045a18f -->

<!-- unit:9e3673a3aa7d35a029 -->
<a id="u-9e3673a3aa7d35a029"></a>

Pinhole depth Z — координата вдоль optical axis, а Euclidean range до camera centre равен $\sqrt{X^2+Y^2+Z^2}$. На центральном луче они совпадают, у края кадра — нет. Sensor/model contract должен уточнять величину. Цветная visualization с нормировкой на min/max изображения вообще не задаёт единиц: одинаковый цвет на двух frames может соответствовать разным distances.

<!-- /unit:9e3673a3aa7d35a029 -->

<!-- unit:b1ed29a5339374d8b9 -->
<a id="u-b1ed29a5339374d8b9"></a>

Монокулярная модель использует perspective, relative size, texture, occlusion, shading и semantic priors. Их полезно разбирать отдельно. Overlap может подсказать ordering двух непрозрачных поверхностей; известный размер машины помогает scale prior; сходящиеся линии — layout. Но машина на рекламном щите тоже похожа на машину, а небо не обязано иметь конечную надёжно измеренную depth. Статистические cues дают вероятностное предположение, а не инверсию утраченной геометрии без дополнительных условий.

<!-- /unit:b1ed29a5339374d8b9 -->

<!-- unit:12741632c92535657d -->
<a id="u-12741632c92535657d"></a>

Фраза чата «если машина на изображении меньше человека, она дальше» требует реальных размеров, позы и intrinsics. Для плоского объекта высоты H_o image height≈fH_o/Z. При одинаковой видимой высоте10px объект0.2m с f=500px находится примерно на10m, объект2m — на100m. Один и тот же image extent не определяет, какой сценарий истинный. При необычных масштабах learned prior может дать правдоподобную, но неверную сцену.

<!-- /unit:12741632c92535657d -->

<!-- unit:7dd786ab8953a5b965 -->
<a id="u-7dd786ab8953a5b965"></a>

Shading зависит от normals, lighting и reflectance; blur — от фокусировки и оптики. Изменить освещение или напечатать texture означает поменять visual cue без нужного изменения depth. Поэтому проверка unusual scenes, зеркал, мелких моделей и прозрачных surfaces важна для понимания границ метода.

<!-- /unit:7dd786ab8953a5b965 -->

<!-- unit:c8f6b4a2c5b2b503cc -->
<a id="u-c8f6b4a2c5b2b503cc"></a>

## Depth, disparity и intrinsics

<!-- /unit:c8f6b4a2c5b2b503cc -->

<!-- unit:6a63285fe2e6fa66db -->
<a id="u-6a63285fe2e6fa66db"></a>

Для rectified stereo disparity d связана с depth Z формулой $d=\frac{fB}{Z}$, где f — focal length в pixels, B — baseline в distance units. При f=500px,B=0.2m,d=20px глубина5m. Это даёт scale, которого не хватает single-image projection при неизвестной geometry.

<!-- /unit:6a63285fe2e6fa66db -->

<!-- unit:01d6e946d7692443e5 -->
<a id="u-01d6e946d7692443e5"></a>

«Inverse depth» 1/Z и stereo disparity пропорциональны только при фиксированном fB; interchange без units ошибочен. Если image resize меняет focal length в pixel coordinates, нужно согласованно менять intrinsics. Crop также сдвигает principal point.

<!-- /unit:01d6e946d7692443e5 -->

<!-- unit:15a5e6bbd684d3c664 -->
<a id="u-15a5e6bbd684d3c664"></a>

Даже RGB-D GT не идеален: LiDAR sparse, отражающие/прозрачные поверхности дают holes/noise, sensors могут иметь registration offset между RGB и depth. Zero может означать invalid, а не объект у camera centre. Validity mask и crop/range filters входят в evaluation definition.

<!-- /unit:15a5e6bbd684d3c664 -->

<!-- unit:26e6a2ede4215dc578 -->
<a id="u-26e6a2ede4215dc578"></a>

### Стерео и backprojection: от pixels к 3D

<!-- /unit:26e6a2ede4215dc578 -->

<!-- unit:df1e32ce32d4003b95 -->
<a id="u-df1e32ce32d4003b95"></a>

Для known intrinsics и depth Z точка pixel=(u,v) поднимается в camera coordinates:

<!-- /unit:df1e32ce32d4003b95 -->

<!-- unit:e7055844d8f1959420 -->
<a id="u-e7055844d8f1959420"></a>

$$
X=(u-c_x)Z/f_x,\qquad Y=(v-c_y)Z/f_y,\qquad Z=Z.
$$

<!-- /unit:e7055844d8f1959420 -->

<!-- unit:b56f87c67388256d08 -->
<a id="u-b56f87c67388256d08"></a>

При fx=fy=500,c=(320,240),pixel=(420,240),Z=5m получаем (1,0,5)m. Euclidean range=√26≈5.099m. Если resized image вдвое меньше, fx и cx также уменьшаются вдвое; pixel переходит в(210,120), и восстановленная точка должна остаться той же. Подставить старое fx с новыми pixels — получить другую geometry.

<!-- /unit:b56f87c67388256d08 -->

<!-- unit:0ca870ee050dd15ffe -->
<a id="u-0ca870ee050dd15ffe"></a>

В rectified stereo координаты связаны baseline B, и disparity d=fB/Z. При увеличении depth вдвое disparity уменьшается вдвое. Небольшая ошибка d на дальнем объекте может дать большую ошибку Z: производная $\partial Z/\partial d=-fB/d^2$. При fB=100 px·m, d=2px глубина50m; ошибка до3px меняет оценку на33.33m. Поэтому fixed-pixel matching precision имеет разные последствия на разных дальностях.

<!-- /unit:0ca870ee050dd15ffe -->

<!-- unit:fd8c345573da1d7d0c -->
<a id="u-fd8c345573da1d7d0c"></a>

### Depth targets, sparse masks и единицы хранения

<!-- /unit:fd8c345573da1d7d0c -->

<!-- unit:41f210a21867d32246 -->
<a id="u-41f210a21867d32246"></a>

RGB может быть dense, а depth sensor — sparse. Loss считают только по valid measurements; заполнять holes нулём и считать их физической depth0 — неверная задача. Даже 16-bit PNG требует scale factor: stored value1000 может означать1m, а может другую величину согласно dataset. Missing value, clipping maximum и invalid flags должны обрабатываться до логарифмов и относительных делений.

<!-- /unit:41f210a21867d32246 -->

<!-- unit:ecbc8dc82a6ee36afa -->
<a id="u-ecbc8dc82a6ee36afa"></a>

KITTI-like LiDAR и indoor RGB-D имеют разную sparsity, range и failure modes. NYU/ScanNet ориентированы на indoor scenes, другие наборы — outdoor/stereo; использование синтетики даёт плотную геометрическую supervision, но render materials и camera distribution могут отличаться от реальности. Число сцен или название формата из рассматриваемого примера не заменяет protocol и scene-level split. Нельзя класть frames одной комнаты в train/test как независимые картинки.

<!-- /unit:ecbc8dc82a6ee36afa -->

<!-- unit:918efc62fd030e141c -->
<a id="u-918efc62fd030e141c"></a>

## Architectures и training objectives

<!-- /unit:918efc62fd030e141c -->

<!-- unit:4412d65d8ffcc74cc4 -->
<a id="u-4412d65d8ffcc74cc4"></a>

Dense prediction backbone извлекает features, decoder восстанавливает spatial depth. CNN coarse-to-fine модели объединяют global layout и локальные details. MiDaS — family с mixed-data training; не все её variants обязаны иметь Transformer. DPT использует Transformer representations нескольких stages и dense decoder; это architecture для dense tasks, не название одного единственного output dtype. [DPT](https://arxiv.org/abs/2103.13413).

<!-- /unit:4412d65d8ffcc74cc4 -->

<!-- unit:7c34b2a9073a5e3284 -->
<a id="u-7c34b2a9073a5e3284"></a>

Depth Anything подчёркивает масштаб data/pseudo-labeling и pretrained representations; relative model и metric fine-tuned variant имеют разные contracts. Нельзя только по family name считать любой checkpoint calibrated metre estimator. [Depth Anything](https://arxiv.org/abs/2401.10891).

<!-- /unit:7c34b2a9073a5e3284 -->

<!-- unit:9a3a8bfe9bc8acadf1 -->
<a id="u-9a3a8bfe9bc8acadf1"></a>

В supervised setting L1/L2 сравнивают valid depths; log-depth objective уменьшает относительный вес дальних absolute errors. Для e_i=$\log\left(pred_{i}\right)-\log\left(gt_{i}\right)$ scale-invariant quadratic family:

<!-- /unit:9a3a8bfe9bc8acadf1 -->

<!-- unit:d7b82af1b89f9e30e6 -->
<a id="u-d7b82af1b89f9e30e6"></a>

$$
\mathcal{L}=\frac1N\sum_i e_i^2-\lambda\left(\frac1N\sum_i e_i\right)^2.
$$

<!-- /unit:d7b82af1b89f9e30e6 -->

<!-- unit:5ba97af8c531818fc2 -->
<a id="u-5ba97af8c531818fc2"></a>

При lambda1 и глобальном умножении pred на a все e_i сдвигаются на log a, а variance сохраняется. При lambda<1 полная scale invariance теряется. Некоторые definitions берут sqrt и дополнительный scale factor: SI loss/SILog нельзя сравнивать по одному имени без formula.

<!-- /unit:5ba97af8c531818fc2 -->

<!-- unit:ea27ee15bc6d92b296 -->
<a id="u-ea27ee15bc6d92b296"></a>

Scale-and-shift invariant methods могут решать alignment $\min_{a,b}\sum_i(a\,\widehat D_i+b-D_i)^2$ в определённом output domain, например inverse depth. Это **не то же самое**, что variance log-errors: addition b в depth не соответствует constant shift log-depth. В исходном чате эти варианты были смешаны.

<!-- /unit:ea27ee15bc6d92b296 -->

<!-- unit:ebd1ca411f2d53eb70 -->
<a id="u-ebd1ca411f2d53eb70"></a>

Self-supervised geometric training может использовать соседние frames, predicted pose и reprojection: depth переводит pixel в3D, pose — в другую camera, затем photo consistency. Dynamic objects, occlusion, textureless regions и illumination нарушают assumptions; masks/robust objectives помогают, но не гарантируют правильную metric scale.

<!-- /unit:ebd1ca411f2d53eb70 -->

<!-- unit:aeb72468ec003bbedc -->
<a id="u-aeb72468ec003bbedc"></a>

### От coarse-to-fine CNN к DPT и обучению на смеси данных

<!-- /unit:aeb72468ec003bbedc -->

<!-- unit:af15a92005c5b8509d -->
<a id="u-af15a92005c5b8509d"></a>

В coarse-to-fine подходе глобальная ветвь оценивает layout, а refinement добавляет локальные детали. Fully convolutional decoder делает плотный output, использует skip features и upsampling. DPT извлекает tokens разных глубин Transformer, reassembles их в image-like feature maps и объединяет dense decoder. Tokens разных слоёв не обязательно имеют разные исходные spatial resolutions; нужные scales формируются reassembly. Это точнее схемы «Transformer автоматически выдаёт pyramid как CNN». [DPT](https://arxiv.org/abs/2103.13413).

<!-- /unit:af15a92005c5b8509d -->

<!-- unit:45f5028120ef3c9da4 -->
<a id="u-45f5028120ef3c9da4"></a>

MiDaS связывает разные источники supervision, где scale/range/representation могут различаться. В разбираемом примере это ошибочно сведено к variance log-errors, которая убирает multiplicative scale, но не произвольный additive shift в inverse depth. Для affine alignment в chosen domain решается least squares по a,b или используется другое явно определённое нормирование. Смешанные datasets и objective invariance — отдельная идея от выбора CNN/ViT backbone. [MiDaS](https://arxiv.org/abs/1907.01341).

<!-- /unit:45f5028120ef3c9da4 -->

<!-- unit:e1546495ad2d750009 -->
<a id="u-e1546495ad2d750009"></a>

Depth Anything использует масштаб unlabeled data, pseudo-labels и pretrained visual representations; V2 подчёркивает высококачественную synthetic teacher supervision и перенос на real images. Нельзя описать оба поколения одним «увеличили число фотографий» или приписать любой relative checkpoint metric units. [Depth Anything](https://arxiv.org/abs/2401.10891), [Depth Anything V2](https://arxiv.org/abs/2406.09414).

<!-- /unit:e1546495ad2d750009 -->

<!-- unit:5936554e91983e80b2 -->
<a id="u-5936554e91983e80b2"></a>

Ordinal methods вроде DORN превращают range в упорядоченные thresholds/classes, adaptive bins вроде AdaBins учат удобные интервалы, planar guidance добавляет geometric structure. ZoeDepth/Metric3D/UniDepth используют свои способы metric prediction и camera-related conditioning, Marigold исследует diffusion prior, Depth Pro — своё sharp metric estimation design. Эти направления из рассматриваемого примера сохранены по типу решения; общий ярлык SOTA и произвольные year/version списки не являются универсальным сравнением качества.

<!-- /unit:5936554e91983e80b2 -->

<!-- unit:672a7e9c6341bd5d5e -->
<a id="u-672a7e9c6341bd5d5e"></a>

### BerHu, gradients и surface normals

<!-- /unit:672a7e9c6341bd5d5e -->

<!-- unit:c95fafdeecefa31fac -->
<a id="u-c95fafdeecefa31fac"></a>

Reverse Huber из разбираемого примера при c>0:

<!-- /unit:c95fafdeecefa31fac -->

<!-- unit:ec558c88695ad05ab6 -->
<a id="u-ec558c88695ad05ab6"></a>

$$
\mathcal L_{\rm berHu}(e)=\begin{cases}|e|,&|e|\le c,\\
(e^2+c^2)/(2c),&|e|>c.\end{cases}
$$

<!-- /unit:ec558c88695ad05ab6 -->

<!-- unit:c84e747aad5e635dd8 -->
<a id="u-c84e747aad5e635dd8"></a>

На границе обе ветви дают c и одинаковый slope по |e|=1. При c=2 ошибки1,2,4 дают losses1,2,5. Это линейная малых и quadratic больших ошибок — противоположное расположение областей обычного Huber. Если c зависит от batch errors, правило его определения также часть objective.

<!-- /unit:c84e747aad5e635dd8 -->

<!-- unit:4c091f5a5dc14be573 -->
<a id="u-4c091f5a5dc14be573"></a>

Gradient loss сравнивает пространственные differences, например горизонтальные/вертикальные first differences predicted и GT maps. Это поощряет boundaries, но неверная sensor registration может обучать сдвинутые edges. Для normals нужно сначала backproject depth с intrinsics и оценить касательные в 3D; произвольная image gradient не является физической surface normal без camera geometry. Normal loss 1−dot unit normals требует согласованной orientation.

<!-- /unit:4c091f5a5dc14be573 -->

<!-- unit:cb5ef80ffd92eae550 -->
<a id="u-cb5ef80ffd92eae550"></a>

### Self-supervised reprojection как вычислительная цепочка

<!-- /unit:cb5ef80ffd92eae550 -->

<!-- unit:95bdab1796c9c89f5b -->
<a id="u-95bdab1796c9c89f5b"></a>

Pixel p target-frame поднимается в 3D по predicted depth и K inverse, затем transformed camera pose R,t переносит точку в source-frame, K проецирует её назад на image plane. Differentiable sampling source image создаёт реконструкцию target, которую сравнивают photometric objective. Теперь depth влияет на то, откуда читается пиксель, поэтому ошибка изображения может обучать depth без прямой depth label.

<!-- /unit:95bdab1796c9c89f5b -->

<!-- unit:d38cbc48a54a10907d -->
<a id="u-d38cbc48a54a10907d"></a>

При independently moving объекте одна camera pose не описывает его motion; при occlusion соответствующая surface вообще может отсутствовать на source frame; при textureless region много disparity вариантов дают близкий цвет. Minimum reprojection и masking пытаются уменьшить такие нарушения. Низкий photometric loss сам по себе не доказывает правильную 3D сцену. Для monocular video масштаб depth и translation может совместно меняться; known stereo baseline добавляет metric constraint. [Monodepth2](https://arxiv.org/abs/1806.01260).

<!-- /unit:d38cbc48a54a10907d -->

<!-- unit:e9c6402543f500416f -->
<a id="u-e9c6402543f500416f"></a>

## Метрики и ручной пример

<!-- /unit:e9c6402543f500416f -->

<!-- unit:16c0c7092255699e47 -->
<a id="u-16c0c7092255699e47"></a>

Для valid pixels с positive GT D_i и positive predictions P_i:

<!-- /unit:16c0c7092255699e47 -->

<!-- unit:5ed0a8a097d49cf7db -->
<a id="u-5ed0a8a097d49cf7db"></a>

$$
\begin{aligned}\mathrm{AbsRel}&=\frac1N\sum_i\frac{|P_i-D_i|}{D_i},\\\mathrm{SqRel}&=\frac1N\sum_i\frac{(P_i-D_i)^2}{D_i},\\\mathrm{RMSE}&=\sqrt{\frac1N\sum_i(P_i-D_i)^2},\\\mathrm{RMSE}_{\log}&=\sqrt{\frac1N\sum_i(\log P_i-\log D_i)^2},\\\delta_k&=\frac1N\sum_i\mathbf{1}\!\left[\max\!\left(\frac{P_i}{D_i},\frac{D_i}{P_i}\right)<1.25^k\right].\end{aligned}
$$

<!-- /unit:5ed0a8a097d49cf7db -->

<!-- unit:8d163d71ec5046dd80 -->
<a id="u-8d163d71ec5046dd80"></a>

Log base, strict/non-strict threshold, range clipping, crop и alignment задаются protocol. δ2 означает ratio threshold1.5625, δ3 —1.953125; их нельзя объяснять как «25 в квадрате процентов». Даже δ1 задаёт multiplicative ratio interval, не симметричный additive±25% вокруг GT.

<!-- /unit:8d163d71ec5046dd80 -->

<!-- unit:00a9614476f3d4f35b -->
<a id="u-00a9614476f3d4f35b"></a>

Пример GT=(1,2,4)m, pred=(2,4,8)m. Output идеально сохраняет relative proportions, но имеет doubled scale:

<!-- /unit:00a9614476f3d4f35b -->

<!-- unit:65736aa5aa25501dbe -->
<a id="u-65736aa5aa25501dbe"></a>

$$
\mathrm{AbsRel}=1,\qquad\mathrm{RMSE}=\sqrt{\frac{1+4+16}{3}}=\sqrt7\approx2.64575\;\mathrm{m},\qquad\delta_1=\delta_2=\delta_3=0.
$$

<!-- /unit:65736aa5aa25501dbe -->

<!-- unit:6b2abcdb89b277125e -->
<a id="u-6b2abcdb89b277125e"></a>

Если protocol разрешает scale alignment a=0.5, все ошибки становятся нулевыми. Это не улучшение model output: изменился вопрос оценки. Raw metric performance плохое, relative geometry в этом примере хорошая. Нельзя сравнивать aligned score одной model с raw score другой и объявлять победителя.

<!-- /unit:6b2abcdb89b277125e -->

<!-- unit:c490c57973598db448 -->
<a id="u-c490c57973598db448"></a>

Для $\widehat D=(1.1,1.8,4.8)$ получаем $\mathrm{AbsRel}=\frac{0.1+0.1+0.2}{3}\approx0.13333$ и $\mathrm{RMSE}=\sqrt{0.23}\approx0.47958$. Эти числа полезны для проверки реализации masks/units. Масштаб в millimetres вместо metres умножит RMSE на 1000, а AbsRel не изменится при согласованном переводе обеих величин.

<!-- /unit:c490c57973598db448 -->

<!-- unit:8d527a55432d865096 -->
<a id="u-8d527a55432d865096"></a>

### Раскрываем инвариантности на конкретных ошибках

<!-- /unit:8d527a55432d865096 -->

<!-- unit:d52e07ace8aaaf30b8 -->
<a id="u-d52e07ace8aaaf30b8"></a>

Пусть GT=(1,2,4),prediction=(2,4,8). Все log-errors равны ln2, поэтому их variance равна0. При λ=1 указанный loss не штрафует этот global factor; при λ=0.5 он равен $0.5(\ln2)^2\approx0.240227$. Следовательно, называть любой λ в этой формуле «полностью scale-invariant» неверно.

<!-- /unit:d52e07ace8aaaf30b8 -->

<!-- unit:5a5a43a5d8a32e9e36 -->
<a id="u-5a5a43a5d8a32e9e36"></a>

Добавление depth shift b не является таким же преобразованием: prediction=(2,3,5) для GT=(1,2,4) имеет разные ratios2,1.5,1.25 и разные log-errors. Affine alignment с a=1,b=−1 сделает её точной, хотя одного multiplicative scale недостаточно. В inverse-depth domain всё иначе: affine relation между inverse quantities превращается в nonlinear преобразование исходных depths. Именно domain надо указывать рядом со словами scale/shift.

<!-- /unit:5a5a43a5d8a32e9e36 -->

<!-- unit:7ee219bca026d80f44 -->
<a id="u-7ee219bca026d80f44"></a>

Для least-squares alignment prediction x к target y на valid pixels при ненулевой variance x:

<!-- /unit:7ee219bca026d80f44 -->

<!-- unit:cf47a37e96f018c758 -->
<a id="u-cf47a37e96f018c758"></a>

$$
a=\frac{\sum_i(x_i-\bar x)(y_i-\bar y)}{\sum_i(x_i-\bar x)^2},\qquad
b=\bar y-a\bar x.
$$

<!-- /unit:cf47a37e96f018c758 -->

<!-- unit:eeb5d40a7a49a28825 -->
<a id="u-eeb5d40a7a49a28825"></a>

Если prediction константна, знаменатель нулевой: полезную geometry из неё alignment не создаст. Если a отрицателен, он меняет ordering; допустимость этого решения определяется контрактом relative output/protocol. Оптимизировать a,b по test GT можно только как явно разрешённую операцию оценки, а не как якобы самостоятельную metric inference model.

<!-- /unit:eeb5d40a7a49a28825 -->

<!-- unit:7bc897cc26ea75a21c -->
<a id="u-7bc897cc26ea75a21c"></a>

### Почему δ1 не равно симметричному ±25%

<!-- /unit:7bc897cc26ea75a21c -->

<!-- unit:fb7372a31eb2686dab -->
<a id="u-fb7372a31eb2686dab"></a>

Условие max(P/D,D/P)<1.25 эквивалентно 0.8<P/D<1.25. Нижний предел — −20% от GT, верхний — +25%, потому что расстояние multiplicative. На границе1.25 строгий threshold не засчитывает pixel. Для k=2 и3 пределы соответственно1.5625 и1.953125, а не квадраты/кубы процентов. Если evaluations используют разное clipping или нестрогое сравнение, граничные случаи будут различаться.

<!-- /unit:fb7372a31eb2686dab -->

<!-- unit:ccd53408ff4826e408 -->
<a id="u-ccd53408ff4826e408"></a>

AbsRel не имеет единиц, RMSE имеет units depth, SqRel в приведённом определении тоже имеет units depth: в знаменателе GT, а не GT². RMSE_log зависит от base логарифма, inverse-depth errors усиливают влияние близких областей. Назначение «AbsRel только indoor, RMSE только outdoor» из рассматриваемого примера слишком жёсткое: набор metrics определяется benchmark и применением. Порог δ1>0.95 сам по себе не доказывает пригодность для нового диапазона или camera.

<!-- /unit:ccd53408ff4826e408 -->

<!-- unit:e59e6349685f2ab974 -->
<a id="u-e59e6349685f2ab974"></a>

### Smoothing не восстанавливает metric scale

<!-- /unit:e59e6349685f2ab974 -->

<!-- unit:86a6bb3dff23a043e8 -->
<a id="u-86a6bb3dff23a043e8"></a>

Median scaling использует отношение median GT к median prediction: это oracle scale по eval GT. Edge-aware upsampling и bilateral filtering могут улучшить appearance границ, но перенести texture на depth или удалить тонкий предмет. Для TTA predictions сначала приводят к общей geometry и chosen output domain. Среднее depth и inverse-depth с последующим обращением в общем случае даёт разные значения.

<!-- /unit:86a6bb3dff23a043e8 -->

<!-- unit:87f0788f9416b7c543 -->
<a id="u-87f0788f9416b7c543"></a>

Прозрачное стекло и отражение не означают, что физическая depth «не существует». Сложность в том, какую surface отражает observation и что измерил sensor. Задача должна указать first surface, background behind glass или иной target. AR-окклюзия может обходиться relative ordering, navigation/grasping требует другой проверенной metric accuracy. Название семейства model не заменяет этот выбор.

<!-- /unit:87f0788f9416b7c543 -->

<!-- unit:110d42d2a8bffacaff -->
<a id="u-110d42d2a8bffacaff"></a>

## Task-specific YOLO depth и диагностика

<!-- /unit:110d42d2a8bffacaff -->

<!-- unit:1c82a57dc60f9ea520 -->
<a id="u-1c82a57dc60f9ea520"></a>

В [официальном описании YOLO26](https://docs.ultralytics.com/models/yolo26/) перечислены отдельные `*-depth` variants. Для depth нужны соответствующая task head, обученные веса и документированный контракт масштаба/единиц. Наличие общего backbone не превращает обычную detection head в depth estimator. Конкретный checkpoint оценивают depth-метриками, а не mAP; поддерживаемые входы и units проверяют по его model card.

<!-- /unit:1c82a57dc60f9ea520 -->

<!-- unit:d8e9abdf3d25e51e0e -->
<a id="u-d8e9abdf3d25e51e0e"></a>

Проверять: metric/relative mode, output orientation ближе/дальше, invalid pixels, maximum distance, indoor/outdoor domain и camera. Визуально красивые smooth maps могут скрывать неверный scale и тонкие boundaries. Отдельные slices для зеркал, неба, тонких poles и unusual sizes помогают увидеть priors/ошибки.

<!-- /unit:d8e9abdf3d25e51e0e -->

<!-- unit:ca0d88d002fa4c3822 -->
<a id="u-ca0d88d002fa4c3822"></a>

1. Почему картинка не определяет global scale? **Ответ:** одинаковое масштабирование 3D coordinates сохраняет X/Z,Y/Z.
2. Чем d отличается от1/Z? **Ответ:** stereo disparity включает fB и pixel units.
3. Почему alignment нельзя добавлять молча? **Ответ:** он может убрать существенную metric-scale ошибку.

<!-- /unit:ca0d88d002fa4c3822 -->

<!-- unit:8421022e377064596a -->
<a id="u-8421022e377064596a"></a>

**Практика:** пересчитать два numerical examples; заменить один GT pixel на invalid и исключить из denominator. Для выбранной модели зафиксировать camera/data split, range/crop/alignment protocol и сравнить raw/aligned metrics раздельно. Не выдавать relative colour map за измеренные metres.

<!-- /unit:8421022e377064596a -->

<!-- unit:a25115e30604a7be34 -->
<a id="u-a25115e30604a7be34"></a>

**Схема курса:** image → dense relative/metric depth prediction → align scale only when protocol allows → mask valid pixels → compute raw/aligned metrics → report scale assumptions

<!-- /unit:a25115e30604a7be34 -->

<!-- unit:697b4ac9b724269ae3 -->
<a id="u-697b4ac9b724269ae3"></a>

<!-- latex-display-v1 -->

<!-- /unit:697b4ac9b724269ae3 -->

<!-- unit:170d216777279a9af4 -->
<a id="u-170d216777279a9af4"></a>

<!-- source-depth-S7-З -->

<!-- /unit:170d216777279a9af4 -->

## Архитектуры: подробные схемы

### DPT: dense prediction

![DPT: dense prediction](<../оформление/course/arch-dpt.svg>)

Reassemble возвращает token features на пространственные уровни, fusion объединяет масштабы. Выход интерпретируется согласно training objective: архитектура сама не гарантирует метрические метры или известный абсолютный масштаб.

### MiDaS: обучение на смеси depth datasets

![MiDaS: обучение на смеси depth datasets](<../оформление/course/arch-midas.svg>)

MiDaS здесь показывает идею mixed-data supervision. Инвариантность objective относится к выбранному domain глубины; variance log-error не убирает произвольный additive shift inverse depth.

### Depth Anything / V2

![Depth Anything / V2](<../оформление/course/arch-depth-anything.svg>)

Teacher-generated supervision и масштаб данных относятся к обучению; decoder выдаёт depth по своему contract. Metric fine-tuned variant требует отдельной проверки units и данных калибровки.

## Результат главы и переход дальше

Ответьте на вопрос: **Что можно восстановить из одного изображения и какие допущения нужны для масштаба?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

[Маршрут модуля](<README.md>) · [Происхождение и точные места переноса](<../editorial/coverage.md>)
