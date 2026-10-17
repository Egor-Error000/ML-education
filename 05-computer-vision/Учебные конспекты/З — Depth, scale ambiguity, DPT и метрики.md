# З. Depth estimation: глубина, масштаб и протокол оценки

[Оглавление](README.md). Пункт 8 плана.

**Основа:** S7, строки 6187–6969. Сохранены monocular cues, ambiguity, relative/metric output, MiDaS/DPT, Depth Anything и regression metrics. **Исправления:** disparity не буквально 1/Z без scale, δ2/δ3 не «25²%/25³%», log-scale invariant loss не тождественна произвольной affine alignment в depth. **Углубление:** camera geometry, ручной пример raw/aligned metrics и validity protocol.

## 1. Одно изображение не определяет единственную 3D сцену

В pinhole модели $u=f_x\frac XZ+c_x$, $v=f_y\frac YZ+c_y$. Если одновременно умножить X,Y,Z сцены на a, image projection не меняется. Без дополнительной информации absolute scale не идентифицируется из одной такой проекции. Модель использует training priors: типичные размеры объектов, perspective, occlusion, texture, semantic scene structure.

Это позволяет полезные predictions, но не гарантирует физическую истинность новой необычной сцены. Игрушечная машинка рядом и большая машина далеко могут занимать похожую image area. Defocus зависит от focus distance/aperture и может возрастать по обе стороны focus plane; правило «дальше всегда более размыто» не универсально.

Metric depth задаёт величину в units, например metres, по определённому camera/protocol. Relative output может отражать ordering и иметь неизвестные scale/shift; некоторые models предсказывают inverse depth-like quantity. Прежде чем переводить colour preview в metres, надо узнать точный output contract checkpoint.

## 2. Depth, disparity и intrinsics

Для rectified stereo disparity d связана с depth Z формулой $d=\frac{fB}{Z}$, где f — focal length в pixels, B — baseline в distance units. При f=500px,B=0.2m,d=20px глубина5m. Это даёт scale, которого не хватает single-image projection при неизвестной geometry.

«Inverse depth» 1/Z и stereo disparity пропорциональны только при фиксированном fB; interchange без units ошибочен. Если image resize меняет focal length в pixel coordinates, нужно согласованно менять intrinsics. Crop также сдвигает principal point.

Даже RGB-D GT не идеален: LiDAR sparse, отражающие/прозрачные поверхности дают holes/noise, sensors могут иметь registration offset между RGB и depth. Zero может означать invalid, а не объект у camera centre. Validity mask и crop/range filters входят в evaluation definition.

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

## 5. Task-specific YOLO depth и диагностика

В новом чате YOLO26 depth обсуждается как ещё одна head. По текущей официальной документации отдельные `*-depth` variants перечислены; их support, pretrained data и units нужно подтверждать конкретным checkpoint/model card. Слово «unified» не делает detection head готовым depth estimator и не позволяет оценивать depth через mAP.

Проверять: metric/relative mode, output orientation ближе/дальше, invalid pixels, maximum distance, indoor/outdoor domain и camera. Визуально красивые smooth maps могут скрывать неверный scale и тонкие boundaries. Отдельные slices для зеркал, неба, тонких poles и unusual sizes помогают увидеть priors/ошибки.

1. Почему картинка не определяет global scale? **Ответ:** одинаковое масштабирование 3D coordinates сохраняет X/Z,Y/Z.
2. Чем d отличается от1/Z? **Ответ:** stereo disparity включает fB и pixel units.
3. Почему alignment нельзя добавлять молча? **Ответ:** он может убрать существенную metric-scale ошибку.

**Практика:** пересчитать два numerical examples; заменить один GT pixel на invalid и исключить из denominator. Для выбранной модели зафиксировать camera/data split, range/crop/alignment protocol и сравнить raw/aligned metrics раздельно. Не выдавать relative colour map за измеренные metres.

<!-- latex-display-v1 -->
