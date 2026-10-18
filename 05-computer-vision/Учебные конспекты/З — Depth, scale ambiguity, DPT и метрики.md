# З. Depth estimation: глубина, масштаб и протокол оценки

[Оглавление](README.md). Пункт 8 плана.

**Основа:** S7, строки 6187–6969. Сохранены monocular cues, ambiguity, relative/metric output, MiDaS/DPT, Depth Anything и regression metrics. **Исправления:** disparity не буквально 1/Z без scale, δ2/δ3 не «25²%/25³%», log-scale invariant loss не тождественна произвольной affine alignment в depth. **Углубление:** camera geometry, ручной пример raw/aligned metrics и validity protocol.

## 1. Одно изображение не определяет единственную 3D сцену

В pinhole модели $u=f_x\frac XZ+c_x$, $v=f_y\frac YZ+c_y$. Если одновременно умножить X,Y,Z сцены на a, image projection не меняется. Без дополнительной информации absolute scale не идентифицируется из одной такой проекции. Модель использует training priors: типичные размеры объектов, perspective, occlusion, texture, semantic scene structure.

Это позволяет полезные predictions, но не гарантирует физическую истинность новой необычной сцены. Игрушечная машинка рядом и большая машина далеко могут занимать похожую image area. Defocus зависит от focus distance/aperture и может возрастать по обе стороны focus plane; правило «дальше всегда более размыто» не универсально.

Metric depth задаёт величину в units, например metres, по определённому camera/protocol. Relative output может отражать ordering и иметь неизвестные scale/shift; некоторые models предсказывают inverse depth-like quantity. Прежде чем переводить colour preview в metres, надо узнать точный output contract checkpoint.

### Что именно называется расстоянием на depth map

Pinhole depth Z — координата вдоль optical axis, а Euclidean range до camera centre равен $\sqrt{X^2+Y^2+Z^2}$. На центральном луче они совпадают, у края кадра — нет. Sensor/model contract должен уточнять величину. Цветная visualization с нормировкой на min/max изображения вообще не задаёт единиц: одинаковый цвет на двух frames может соответствовать разным distances.

Модель из S7 использует perspective, relative size, texture, occlusion, shading и semantic priors. Их полезно разбирать отдельно. Overlap может подсказать ordering двух непрозрачных поверхностей; известный размер машины помогает scale prior; сходящиеся линии — layout. Но машина на рекламном щите тоже похожа на машину, а небо не обязано иметь конечную надёжно измеренную depth. Статистические cues дают вероятностное предположение, а не инверсию утраченной геометрии без дополнительных условий.

Фраза чата «если машина на изображении меньше человека, она дальше» требует реальных размеров, позы и intrinsics. Для плоского объекта высоты H_o image height≈fH_o/Z. При одинаковой видимой высоте10px объект0.2m с f=500px находится примерно на10m, объект2m — на100m. Один и тот же image extent не определяет, какой сценарий истинный. При необычных масштабах learned prior может дать правдоподобную, но неверную сцену.

Shading зависит от normals, lighting и reflectance; blur — от фокусировки и оптики. Изменить освещение или напечатать texture означает поменять visual cue без нужного изменения depth. Поэтому проверка unusual scenes, зеркал, мелких моделей и прозрачных surfaces важна для понимания границ метода.

## 2. Depth, disparity и intrinsics

Для rectified stereo disparity d связана с depth Z формулой $d=\frac{fB}{Z}$, где f — focal length в pixels, B — baseline в distance units. При f=500px,B=0.2m,d=20px глубина5m. Это даёт scale, которого не хватает single-image projection при неизвестной geometry.

«Inverse depth» 1/Z и stereo disparity пропорциональны только при фиксированном fB; interchange без units ошибочен. Если image resize меняет focal length в pixel coordinates, нужно согласованно менять intrinsics. Crop также сдвигает principal point.

Даже RGB-D GT не идеален: LiDAR sparse, отражающие/прозрачные поверхности дают holes/noise, sensors могут иметь registration offset между RGB и depth. Zero может означать invalid, а не объект у camera centre. Validity mask и crop/range filters входят в evaluation definition.

### Стерео и backprojection: от pixels к 3D

Для known intrinsics и depth Z точка pixel=(u,v) поднимается в camera coordinates:

$$
X=(u-c_x)Z/f_x,\qquad Y=(v-c_y)Z/f_y,\qquad Z=Z.
$$

При fx=fy=500,c=(320,240),pixel=(420,240),Z=5m получаем (1,0,5)m. Euclidean range=√26≈5.099m. Если resized image вдвое меньше, fx и cx также уменьшаются вдвое; pixel переходит в(210,120), и восстановленная точка должна остаться той же. Подставить старое fx с новыми pixels — получить другую geometry.

В rectified stereo координаты связаны baseline B, и disparity d=fB/Z. При увеличении depth вдвое disparity уменьшается вдвое. Небольшая ошибка d на дальнем объекте может дать большую ошибку Z: производная $\partial Z/\partial d=-fB/d^2$. При fB=100 px·m, d=2px глубина50m; ошибка до3px меняет оценку на33.33m. Поэтому fixed-pixel matching precision имеет разные последствия на разных дальностях.

### Depth targets, sparse masks и единицы хранения

RGB может быть dense, а depth sensor — sparse. Loss считают только по valid measurements; заполнять holes нулём и считать их физической depth0 — неверная задача. Даже 16-bit PNG требует scale factor: stored value1000 может означать1m, а может другую величину согласно dataset. Missing value, clipping maximum и invalid flags должны обрабатываться до логарифмов и относительных делений.

KITTI-like LiDAR и indoor RGB-D имеют разную sparsity, range и failure modes. NYU/ScanNet ориентированы на indoor scenes, другие наборы — outdoor/stereo; использование синтетики даёт плотную геометрическую supervision, но render materials и camera distribution могут отличаться от реальности. Число сцен или название формата из S7 не заменяет protocol и scene-level split. Нельзя класть frames одной комнаты в train/test как независимые картинки.

## 3. Architectures и training objectives

Dense prediction backbone извлекает features, decoder восстанавливает spatial depth. CNN coarse-to-fine модели объединяют global layout и локальные details. MiDaS — family с mixed-data training; не все её variants обязаны иметь Transformer. DPT использует Transformer representations нескольких stages и dense decoder; это architecture для dense tasks, не название одного единственного output dtype. [DPT](https://arxiv.org/abs/2103.13413).

Depth Anything подчёркивает масштаб data/pseudo-labeling и pretrained representations; relative model и metric fine-tuned variant имеют разные contracts. Нельзя только по family name считать любой checkpoint calibrated metre estimator. [Depth Anything](https://arxiv.org/abs/2401.10891).

В supervised setting L1/L2 сравнивают valid depths; log-depth objective уменьшает относительный вес дальних absolute errors. Для e_i=$\log\left(pred_{i}\right)-\log\left(gt_{i}\right)$ scale-invariant quadratic family:

$$
\mathcal{L}=\frac1N\sum_i e_i^2-\lambda\left(\frac1N\sum_i e_i\right)^2.
$$

При lambda1 и глобальном умножении pred на a все e_i сдвигаются на log a, а variance сохраняется. При lambda<1 полная scale invariance теряется. Некоторые definitions берут sqrt и дополнительный scale factor: SI loss/SILog нельзя сравнивать по одному имени без formula.

Scale-and-shift invariant methods могут решать alignment $\min_{a,b}\sum_i(a\,\widehat D_i+b-D_i)^2$ в определённом output domain, например inverse depth. Это **не то же самое**, что variance log-errors: addition b в depth не соответствует constant shift log-depth. В исходном чате эти варианты были смешаны.

Self-supervised geometric training может использовать соседние frames, predicted pose и reprojection: depth переводит pixel в3D, pose — в другую camera, затем photo consistency. Dynamic objects, occlusion, textureless regions и illumination нарушают assumptions; masks/robust objectives помогают, но не гарантируют правильную metric scale.

### От coarse-to-fine CNN к DPT и обучению на смеси данных

В coarse-to-fine подходе глобальная ветвь оценивает layout, а refinement добавляет локальные детали. Fully convolutional decoder делает плотный output, использует skip features и upsampling. DPT извлекает tokens разных глубин Transformer, reassembles их в image-like feature maps и объединяет dense decoder. Tokens разных слоёв не обязательно имеют разные исходные spatial resolutions; нужные scales формируются reassembly. Это точнее схемы «Transformer автоматически выдаёт pyramid как CNN». [DPT](https://arxiv.org/abs/2103.13413).

MiDaS связывает разные источники supervision, где scale/range/representation могут различаться. В S7 это ошибочно сведено к variance log-errors, которая убирает multiplicative scale, но не произвольный additive shift в inverse depth. Для affine alignment в chosen domain решается least squares по a,b или используется другое явно определённое нормирование. Смешанные datasets и objective invariance — отдельная идея от выбора CNN/ViT backbone. [MiDaS](https://arxiv.org/abs/1907.01341).

Depth Anything использует масштаб unlabeled data, pseudo-labels и pretrained visual representations; V2 подчёркивает высококачественную synthetic teacher supervision и перенос на real images. Нельзя описать оба поколения одним «увеличили число фотографий» или приписать любой relative checkpoint metric units. [Depth Anything](https://arxiv.org/abs/2401.10891), [Depth Anything V2](https://arxiv.org/abs/2406.09414).

Ordinal methods вроде DORN превращают range в упорядоченные thresholds/classes, adaptive bins вроде AdaBins учат удобные интервалы, planar guidance добавляет geometric structure. ZoeDepth/Metric3D/UniDepth используют свои способы metric prediction и camera-related conditioning, Marigold исследует diffusion prior, Depth Pro — своё sharp metric estimation design. Эти направления из S7 сохранены по типу решения; общий ярлык SOTA и произвольные year/version списки не являются универсальным сравнением качества.

### BerHu, gradients и surface normals

Reverse Huber из чата при c>0:

$$
\mathcal L_{\rm berHu}(e)=\begin{cases}|e|,&|e|\le c,\\
(e^2+c^2)/(2c),&|e|>c.\end{cases}
$$

На границе обе ветви дают c и одинаковый slope по |e|=1. При c=2 ошибки1,2,4 дают losses1,2,5. Это линейная малых и quadratic больших ошибок — противоположное расположение областей обычного Huber. Если c зависит от batch errors, правило его определения также часть objective.

Gradient loss сравнивает пространственные differences, например горизонтальные/вертикальные first differences predicted и GT maps. Это поощряет boundaries, но неверная sensor registration может обучать сдвинутые edges. Для normals нужно сначала backproject depth с intrinsics и оценить касательные в 3D; произвольная image gradient не является физической surface normal без camera geometry. Normal loss 1−dot unit normals требует согласованной orientation.

### Self-supervised reprojection как вычислительная цепочка

Pixel p target-frame поднимается в 3D по predicted depth и K inverse, затем transformed camera pose R,t переносит точку в source-frame, K проецирует её назад на image plane. Differentiable sampling source image создаёт реконструкцию target, которую сравнивают photometric objective. Теперь depth влияет на то, откуда читается пиксель, поэтому ошибка изображения может обучать depth без прямой depth label.

При independently moving объекте одна camera pose не описывает его motion; при occlusion соответствующая surface вообще может отсутствовать на source frame; при textureless region много disparity вариантов дают близкий цвет. Minimum reprojection и masking пытаются уменьшить такие нарушения. Низкий photometric loss сам по себе не доказывает правильную 3D сцену. Для monocular video масштаб depth и translation может совместно меняться; known stereo baseline добавляет metric constraint. [Monodepth2](https://arxiv.org/abs/1806.01260).

## 4. Метрики и ручной пример

Для valid pixels с positive GT D_i и positive predictions P_i:

$$
\begin{aligned}\mathrm{AbsRel}&=\frac1N\sum_i\frac{|P_i-D_i|}{D_i},\\\mathrm{SqRel}&=\frac1N\sum_i\frac{(P_i-D_i)^2}{D_i},\\\mathrm{RMSE}&=\sqrt{\frac1N\sum_i(P_i-D_i)^2},\\\mathrm{RMSE}_{\log}&=\sqrt{\frac1N\sum_i(\log P_i-\log D_i)^2},\\\delta_k&=\frac1N\sum_i\mathbf{1}\!\left[\max\!\left(\frac{P_i}{D_i},\frac{D_i}{P_i}\right)<1.25^k\right].\end{aligned}
$$

Log base, strict/non-strict threshold, range clipping, crop и alignment задаются protocol. δ2 означает ratio threshold1.5625, δ3 —1.953125; их нельзя объяснять как «25 в квадрате процентов». Даже δ1 задаёт multiplicative ratio interval, не симметричный additive±25% вокруг GT.

Пример GT=(1,2,4)m, pred=(2,4,8)m. Output идеально сохраняет relative proportions, но имеет doubled scale:

$$
\mathrm{AbsRel}=1,\qquad\mathrm{RMSE}=\sqrt{\frac{1+4+16}{3}}=\sqrt7\approx2.64575\;\mathrm{m},\qquad\delta_1=\delta_2=\delta_3=0.
$$

Если protocol разрешает scale alignment a=0.5, все ошибки становятся нулевыми. Это не улучшение model output: изменился вопрос оценки. Raw metric performance плохое, relative geometry в этом примере хорошая. Нельзя сравнивать aligned score одной model с raw score другой и объявлять победителя.

Для $\widehat D=(1.1,1.8,4.8)$ получаем $\mathrm{AbsRel}=\frac{0.1+0.1+0.2}{3}\approx0.13333$ и $\mathrm{RMSE}=\sqrt{0.23}\approx0.47958$. Эти числа полезны для проверки реализации masks/units. Масштаб в millimetres вместо metres умножит RMSE на 1000, а AbsRel не изменится при согласованном переводе обеих величин.

### Раскрываем инвариантности на конкретных ошибках

Пусть GT=(1,2,4),prediction=(2,4,8). Все log-errors равны ln2, поэтому их variance равна0. При λ=1 указанный loss не штрафует этот global factor; при λ=0.5 он равен $0.5(\ln2)^2\approx0.240227$. Следовательно, называть любой λ в этой формуле «полностью scale-invariant» неверно.

Добавление depth shift b не является таким же преобразованием: prediction=(2,3,5) для GT=(1,2,4) имеет разные ratios2,1.5,1.25 и разные log-errors. Affine alignment с a=1,b=−1 сделает её точной, хотя одного multiplicative scale недостаточно. В inverse-depth domain всё иначе: affine relation между inverse quantities превращается в nonlinear преобразование исходных depths. Именно domain надо указывать рядом со словами scale/shift.

Для least-squares alignment prediction x к target y на valid pixels при ненулевой variance x:

$$
a=\frac{\sum_i(x_i-\bar x)(y_i-\bar y)}{\sum_i(x_i-\bar x)^2},\qquad
b=\bar y-a\bar x.
$$

Если prediction константна, знаменатель нулевой: полезную geometry из неё alignment не создаст. Если a отрицателен, он меняет ordering; допустимость этого решения определяется контрактом relative output/protocol. Оптимизировать a,b по test GT можно только как явно разрешённую операцию оценки, а не как якобы самостоятельную metric inference model.

### Почему δ1 не равно симметричному ±25%

Условие max(P/D,D/P)<1.25 эквивалентно 0.8<P/D<1.25. Нижний предел — −20% от GT, верхний — +25%, потому что расстояние multiplicative. На границе1.25 строгий threshold не засчитывает pixel. Для k=2 и3 пределы соответственно1.5625 и1.953125, а не квадраты/кубы процентов. Если evaluations используют разное clipping или нестрогое сравнение, граничные случаи будут различаться.

AbsRel не имеет единиц, RMSE имеет units depth, SqRel в приведённом определении тоже имеет units depth: в знаменателе GT, а не GT². RMSE_log зависит от base логарифма, inverse-depth errors усиливают влияние близких областей. Назначение «AbsRel только indoor, RMSE только outdoor» из S7 слишком жёсткое: набор metrics определяется benchmark и применением. Порог δ1>0.95 сам по себе не доказывает пригодность для нового диапазона или camera.

### Smoothing не восстанавливает metric scale

Median scaling использует отношение median GT к median prediction: это oracle scale по eval GT. Edge-aware upsampling и bilateral filtering могут улучшить appearance границ, но перенести texture на depth или удалить тонкий предмет. Для TTA predictions сначала приводят к общей geometry и chosen output domain. Среднее depth и inverse-depth с последующим обращением в общем случае даёт разные значения.

Прозрачное стекло и отражение не означают, что физическая depth «не существует». Сложность в том, какую surface отражает observation и что измерил sensor. Задача должна указать first surface, background behind glass или иной target. AR-окклюзия может обходиться relative ordering, navigation/grasping требует другой проверенной metric accuracy. Название семейства model не заменяет этот выбор.

## 5. Task-specific YOLO depth и диагностика

В новом чате YOLO26 depth обсуждается как ещё одна head. По текущей официальной документации отдельные `*-depth` variants перечислены; их support, pretrained data и units нужно подтверждать конкретным checkpoint/model card. Слово «unified» не делает detection head готовым depth estimator и не позволяет оценивать depth через mAP.

Проверять: metric/relative mode, output orientation ближе/дальше, invalid pixels, maximum distance, indoor/outdoor domain и camera. Визуально красивые smooth maps могут скрывать неверный scale и тонкие boundaries. Отдельные slices для зеркал, неба, тонких poles и unusual sizes помогают увидеть priors/ошибки.

1. Почему картинка не определяет global scale? **Ответ:** одинаковое масштабирование 3D coordinates сохраняет X/Z,Y/Z.
2. Чем d отличается от1/Z? **Ответ:** stereo disparity включает fB и pixel units.
3. Почему alignment нельзя добавлять молча? **Ответ:** он может убрать существенную metric-scale ошибку.

**Практика:** пересчитать два numerical examples; заменить один GT pixel на invalid и исключить из denominator. Для выбранной модели зафиксировать camera/data split, range/crop/alignment protocol и сравнить raw/aligned metrics раздельно. Не выдавать relative colour map за измеренные metres.

<!-- latex-display-v1 -->

<!-- source-depth-S7-З -->
