# Д. Pose estimation: ключевые точки и принадлежность человеку

[Оглавление](README.md). Пункт 5 плана.

**Основа:** S7, строки 3713–4523. **Исправления:** YOLO-pose one-stage не равен «detector → отдельный crop model»; COCO v>0 включает размеченные occluded points, а не только видимые; OKS sigma conventions не смешиваются. **Углубление:** ручной Gaussian target, PCK/OKS и inverse crop transform.

## 1. Что должна предсказать модель

Для каждого человека задан фиксированный словарь joints: например, плечи, локти, запястья. Output содержит координаты, confidence/visibility и принадлежность человеку. Это не arbitrary object mask и не само по себе 3D reconstruction.

Аннотация должна различать отсутствие разметки и occlusion. В COCO-подобной convention $v=0$ — не размечено, $v=1$ — размечено, но закрыто, $v=2$ — видно. Метрики с indicator(v>0) учитывают и $v=1$, и v=2. В исходном тексте это называлось «только видимые», что теряет важное различие.

Левая/правая сторона обычно определяются по телу объекта, а не по viewer. Horizontal flip требует не только $x\mapsto W-1-x$ при pixel-centre convention, но и перестановки labels left/right. Иначе image выглядит корректно, а targets становятся противоположными.

## 2. Top-down, bottom-up и one-stage

Top-down pipeline сначала находит человека, вырезает region, затем отдельная pose model обрабатывает crop. Это делает scale более стандартизированным, но missed detection теряет pose; compute обычно зависит от количества crops. HRNet-подобный estimator может использоваться в такой системе.

Bottom-up находит все joints и grouping cues, потом связывает точки по людям. OpenPose использует part affinity fields — vector fields вдоль связей тела. Candidate limbs оцениваются по согласованности направления сегмента с field, затем собираются skeletons. [OpenPose](https://arxiv.org/abs/1611.08050).

One-stage detector с keypoint head совместно выдаёт boxes и points. YOLO-pose относится к такому типу task-specific prediction; наличие boxes не превращает его автоматически в двухэтапный crop pipeline. Вычислительный граф и assignment нужно смотреть у конкретной модели.

HRNet сохраняет high-resolution branch и обменивает information между resolutions. Это помогает точной localisation, но width/resolution стоят памяти. Хорошая local coordinate accuracy не решает grouping нескольких людей сама по себе.

## 3. Heatmaps и coordinate regression

Direct regression возвращает x/y, например нормированные внутри crop. Loss может быть L1/L2, robust или probabilistic variant; visibility weights определяют supervision. При двух допустимых positions MSE может усреднить их в несуществующее место, поэтому uncertainty/distribution важны.

Heatmap head создаёт map $B\times J\times H_{h}\times W_{h}$. Для GT point $(u,v)$ target можно задать Gaussian:

$$
G(x,y)=\exp\!\left(-\frac{(x-u)^2+(y-v)^2}{2\sigma^2}\right).
$$

При sigma1 значение на centre1, на расстоянии1 — e^-0.5≈0.6065, на расстоянии2 — e^-2≈0.1353. Степень blur задаёт tolerance training target, а не evaluation OKS напрямую.

Argmax выбирает пик resolution heatmap, вводя discretization. Refinement/soft-argmax могут дать subpixel coordinates, но soft-argmax усредняет multimodal peaks и требует своей temperature/normalization. Heatmap — не автоматически calibrated probability distribution, если objective и normalization этого не задают.

Crop transform должен быть обратимым в согласованной геометрии. Если original crop начинается в (x0,y0), имеет width w,height h и resized to $W_{c}\times H_{c}$, простая непрерывная coordinate convention даёт $x_{\mathrm{orig}}=x_0+\frac{x_{\mathrm{crop}}w}{W_c}$, $y_{\mathrm{orig}}=y_0+\frac{y_{\mathrm{crop}}h}{H_c}$. Half-pixel/align details в реальной библиотеке проверяются отдельно. Letterbox добавляет offsets, их нельзя забывать.

## 4. PCK и OKS

PCK считает долю размеченных points с $\frac{d_i}{d_{\mathrm{norm}}}<\alpha$. Norm может быть head size, torso size или другое benchmark definition. «PCKh всегда расстояние между ушами» слишком узко: берётся определение конкретного dataset. При четырёх distances $\left(2,\,4,\,6,\,12\right)$, norm20, alpha0.25 threshold5 → PCK2/4=0.5.

OKS усредняет scale-normalized Gaussian similarity:

$$
\mathrm{OKS}=\frac{\sum_i\exp\!\left(-\frac{d_i^2}{2s^2k_i^2}\right)\mathbf{1}[v_i>0]}{\sum_i\mathbf{1}[v_i>0]}.
$$

s² соответствует object area по protocol, k_i — benchmark-specific tolerance. В COCO implementation используются stored sigmas и дополнительный множитель: при этой записи k_i соответствует $2\sigma_i$. Подставлять список sigma из исходника как k без проверки — получить другой threshold.

Иллюстрация с двумя labeled points, $s=100$,$k=0.1$: d1=5 даёт exp(-0.125)≈0.8825; d2=10 — exp(-0.5)≈0.6065; mean≈0.7445. На объекте вдвое меньшего linear scale те же pixel errors будут штрафоваться сильнее. Если третий point $v=0$, он не входит в это среднее; occluded $v=1$ входит.

Pose AP сопоставляет person predictions/GT по OKS и score, после чего строит PR/AP по benchmark. Это метрика instances, а PCK агрегирует keypoints иначе. Нельзя заменить pose AP средней confidence точек.

Для 3D MPJPE — средняя Euclidean joint error в заявленных units после определённого root/alignment protocol. P-MPJPE использует Procrustes alignment по protocol, N-MPJPE обычно scale alignment; фраза исходника «N-MPJPE после масштаба и поворота» не должна смешивать его с Procrustes. Сравнение возможно только при одинаковом protocol.

## 5. Ошибки и самостоятельная проверка

Частые ошибки: перепутанные left/right, joints другого человека, missing ankle из-за crop, завышенная confidence occluded point, нарушенный inverse resize, другая skeleton topology. Смотреть predicted skeleton поверх original image и score по occlusion/scale, а не только среднее.

1. Входит ли $v=1$ в формулу с v>0? **Ответ:** да, это размеченная закрытая точка.
2. Чем one-stage pose отличается от top-down crop system? **Ответ:** outputs получаются совместной head без обязательного отдельного forward по каждому crop.
3. Почему OKS зависит от scale? **Ответ:** одинаковый pixel shift имеет разную относительную величину на большом и малом человеке.

**Практика:** пересчитать PCK/OKS examples, выполнить flip четырёх left/right joints и обратный transform crop. Затем сравнить ошибки visible/occluded и small/large людей при фиксированном split по scenes/videos.

<!-- latex-display-v1 -->
