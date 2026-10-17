# А. Классификация изображений, backbone и ViT

[Оглавление](README.md). Пункт 1 плана; отдельное углубление из чата о patch embedding.

**Основа:** S7, строки 207–1617: классификация, вопрос «что такое backbone», просьба подробно разобрать шаг проекции патчей с математикой. Из экспорта исключены английские служебные вставки и дубли повреждённых формул. **Углубление/исправления:** единая NCHW-нотация, точная эквивалентность Conv2d, equivariance вместо ошибочного «permutation-invariant patch embedding», ограничения интуиции ResNet и ViT.

## 1. Задача до выбора архитектуры

В single-label classification изображению $X\in\mathbb{R}^{C\times H\times W}$ соответствует один класс $y\in\{1,\ldots,K\}$. Модель выдаёт K logits, softmax переводит их в probabilities. Predicted class — argmax. Выход не локализует объект: на изображении с тремя собаками может быть одна метка «собака». Если нужно несколько независимых labels, это multi-label задача с sigmoid/BCE или другим подходящим objective, а не тот же mutually exclusive softmax.

Для примера с правильным классом c cross-entropy $\mathcal{L}=-\log p_c$. При $p=\left(0.1,\,0.7,\,0.2\right)$ и c=2 loss≈0.3567. Gradient по logits $p-\operatorname{one\_hot}(c)$; он передаёт сигнал head и backbone, если их параметры обучаемы. Большая вероятность не гарантирует хорошую calibration: это проверяется на независимых данных.

Accuracy — доля правильных argmax. Top-k accuracy считает пример верным, если правильный класс входит в k наибольших scores. Top-5 при $K=3$ бессмысленна как различающая метрика: все классы и так входят в ответ. Confusion matrix зададим строками истинных классов и столбцами predicted.

Для one-vs-rest класса: $\mathrm{Precision}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}$, $\mathrm{Recall}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}$, $F_1=\frac{2\mathrm{TP}}{2\mathrm{TP}+\mathrm{FP}+\mathrm{FN}}$. Macro усредняет классы равноправно, weighted учитывает support; micro суммирует counts до вычисления. Для single-label multiclass без исключения классов micro-F1 совпадает с accuracy. Нулевые denominators обрабатываются по объявленной convention.

Пример: 90 обычных кадров и 10 с дефектом; модель всегда говорит «обычный». Accuracy 90%, recall дефекта 0. Модель не решает задачу поиска дефекта, хотя общий score выглядит высоким. Поэтому нужны class slices, confusion matrix, порог под цену ошибок и calibration.

## 2. Почему начинать с простого baseline

kNN по сырым pixels сравнивает развёрнутые изображения. Небольшой сдвиг объекта сильно меняет distance, фон и освещение могут доминировать. MLP на flattened $224\times 224\times 3$ входе получает 150 528 признаков. Уже hidden layer 1000 требует 150 528 000 весов до biases и следующего слоя. Пространственная близость пикселей явно не используется.

Классические HOG/SIFT-представления с classifier вводят ручную структуру; kNN или linear probe по pretrained embeddings используют выученную. Такие baselines полезны: если representations уже хорошо разделяют классы, full fine-tuning может быть не нужен. При frozen encoder embeddings можно предвычислить, но fit scaler/PCA и подбор k выполняются только по train/validation.

**Backbone** — основной feature extractor, **head** превращает его признаки в output конкретной задачи. Классификатор агрегирует spatial features в один class vector, detector сохраняет локализацию и выдаёт множество объектов, segmenter восстанавливает плотную карту. У некоторых систем есть **neck**, который объединяет features разных resolutions. Эти роли — способ описать computation, а не требование, чтобы у каждой модели были три отдельно названных Python-класса.

## 3. CNN и residual learning

Свёртка использует локальное окно и общие weights для разных позиций. Для kernel $k\times k$ количество весов $C_{\mathrm{out}} C_{\mathrm{in}} k^2$ без biases не зависит от H/W. Слой выдаёт feature maps; нелинейности и следующие слои создают более сложные функции. Pooling/stride уменьшают resolution. Translation equivariance приблизительно сохраняется при подходящих условиях, но boundaries, stride и pooling её ограничивают; полной инвариантности к любому сдвигу нет.

Простой pipeline: conv → activation → pooling → несколько blocks → global average pooling → linear head. LeNet — исторический пример компактной CNN, AlexNet — более глубокого классификатора с ReLU/dropout и большим training scale. Не нужно копировать историческую архитектуру как обязательный лучший baseline сегодня.

ResNet задаёт $y=x+F(x)$, при разных shapes используется projection shortcut. Jacobian $\frac{\partial y}{\partial x}=I+\frac{\partial F}{\partial x}$ создаёт прямой путь gradient. Это помогает optimization, но не гарантирует, что любой очень глубокий стек не имеет exploding/vanishing effects. Исходная residual идея также отвечает проблеме degradation training качества при наращивании обычной глубины, а не только слову «vanishing». [ResNet](https://arxiv.org/abs/1512.03385).

YOLO classification model использует backbone и global classification head. Его не следует путать с class prediction на каждой detection candidate. Модель `*-cls` выдаёт global labels, а `*-detect` ещё и boxes. Формат task-specific head и preprocessing проверяются у конкретного checkpoint; имя YOLO не означает одну общую head для всех задач.

## 4. ViT: изображение становится последовательностью

ViT делит изображение на patches, проецирует каждый в D-мерный vector, добавляет position information и обрабатывает Transformer encoder. Для обычного classification attention bidirectional по доступным image tokens. В отличие от decoder-only LLM, causal mask здесь обычно не нужна.

Пусть B — batch, C — channels, H/W — размеры, P — сторона неперекрывающегося patch. Если размеры делятся на P:

$$
\begin{gathered}N_h=\frac HP,\quad N_w=\frac WP,\quad N=N_hN_w,\quad K_{\mathrm{patch}}=CP^2,\\X\in\mathbb{R}^{B\times C\times H\times W},\quad X_{\mathrm{patches}}\in\mathbb{R}^{B\times N\times K_{\mathrm{patch}}},\\E\in\mathbb{R}^{K_{\mathrm{patch}}\times D},\quad b\in\mathbb{R}^{D},\qquad Z=X_{\mathrm{patches}}E+b\in\mathbb{R}^{B\times N\times D}.\end{gathered}
$$

Embedding width D задаётся архитектурой и не обязана равняться CP². Совпадение 768 для RGB patches $16\times 16$ и ViT-Base $D=768$ — частный выбор.

Для H=W=224,$P=16$,$C=3$ получаем сетку $14\times 14$, $N=196$, K_patch=768. Матрица проекции содержит $768\cdot 768+768=590 592$ параметра с bias. Добавление одного learned CLS token даёт длину 197; его итоговое state используют для head. Другой вариант — pooling patch states, так что CLS не обязательная математическая часть всех ViT.

Далее, для одного encoder head при row tokens:

$$
Q=ZW_Q,\quad K=ZW_K,\quad V=ZW_V,\qquad S=\operatorname{softmax}_{\mathrm{rows}}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right),\quad O=SV.
$$

Multi-head outputs объединяются, проецируются, добавляются residual/norm и tokenwise MLP. Векторы patches получают глобальный контекст через attention. Подробный backward этого механизма уже разобран в [основах LLM](<../../06-llm/01-foundations/Учебные конспекты/Г — QKV, multi-head attention и точный backward.md>): смена modality не меняет chain rule.

## 5. Полная индексация patch embedding

Зафиксируем порядок flattened patch: сначала c, затем локальные p,q. Для $i\in\{0,\ldots,N_h-1\}$, $j\in\{0,\ldots,N_w-1\}$, $p,q\in\{0,\ldots,P-1\}$:

$$
\begin{aligned}n&=iN_w+j,\qquad k=cP^2+pP+q,\\(X_{\mathrm{patches}})_{b,n,k}&=X_{b,c,iP+p,jP+q},\\Z_{b,n,d}&=\sum_k(X_{\mathrm{patches}})_{b,n,k}E_{k,d}+b_d.\end{aligned}
$$

В original export сначала объявлялся HWC, а flatten объяснялся как для CHW. Само отображение может быть задано как угодно, но соответствие memory axes надо фиксировать. Здесь всё согласовано с NCHW.

Линейная проекция всех patches эквивалентна Conv2d с `in_channels=C,out_channels=D,kernel_size=P,stride=P,padding=0`, если:

$$
(W_{\mathrm{conv}})_{d,c,p,q}=E_{cP^2+pP+q,\,d}.
$$

Действительно:

$$
\begin{aligned}Y_{b,d,i,j}&=\sum_{c,p,q}(W_{\mathrm{conv}})_{d,c,p,q}X_{b,c,iP+p,jP+q}+b_d\\&=Z_{b,\,iN_w+j,\,d}.\end{aligned}
$$

После conv output $B\times D\times N_{h}\times N_{w}$ нужно flatten spatial и transpose в $B\times N\times D$. В PyTorch `Linear.weight` имеет $D\times K_{\mathrm{patch}}$, то есть хранит Eᵀ нашей формулы. Переставить веса без этого transpose/reshape — ошибка, которая не обязательно обнаружится только по конечным shapes.

### Минимальный ручной пример

Одноканальное $4\times 4$ изображение содержит числа 1…16 по строкам. При $P=2$ первый patch $\left(1,\,2,\,5,\,6\right)$, второй $\left(3,\,4,\,7,\,8\right)$, третий $\left(9,\,10,\,13,\,14\right)$, четвёртый $\left(11,\,12,\,15,\,16\right)$. Возьмём $D=2$, bias=0 и columns E:

$$
E_{:,1}=\begin{bmatrix}1\\1\\1\\1\end{bmatrix},\qquad E_{:,2}=\begin{bmatrix}1\\0\\0\\-1\end{bmatrix}.
$$

Получим $Z=\begin{bmatrix}14 & -5\\22 & -5\\46 & -5\\54 & -5\end{bmatrix}$. Два conv filters $\begin{bmatrix}1 & 1\\1 & 1\end{bmatrix}$ и $\begin{bmatrix}1 & 0\\0 & -1\end{bmatrix}$ со stride2 дадут те же четыре vectors. Первая координата суммирует patch; вторая сравнивает диагональные pixels. Это пример learned linear filters, а не обязательное значение реальных trained columns.

Gradient также прозрачен. При $G=\frac{\partial\mathcal{L}}{\partial Z}$ матрично $\frac{\partial\mathcal{L}}{\partial E}=X_{\mathrm{patches}}^\top G$, $\frac{\partial\mathcal{L}}{\partial X_{\mathrm{patches}}}=GE^\top$, а bias gradient суммирует B/N. Входной gradient раскладывается обратно по местам patches. При overlap вклады в одни pixels складываются.

## 6. Зачем patches, positions и какие ограничения

Если сделать token из каждого pixel $224\times 224$, $N=50$ 176, матрица attention имеет около 2.52 млрд элементов на head. Для $N=196$ — 38 416. Отношение **числа pairwise entries** равно $256^2=65 536$ без CLS. Это не 65 536-кратный выигрыш полного training: остаются projections/MLP, другая ширина и kernels.

Уменьшение P увеличивает N и сохраняет более мелкую spatial структуру, но повышает attention cost. При doubling обеих сторон изображения N растёт вчетверо, pairwise entries — в 16 раз. Поэтому resolution влияет иначе, чем только число parameters.

Одинаковая linear projection применяется к каждому patch. Если переставить patches, выходные vectors переставятся так же: это **equivariance**, не invariance. Последовательность сама помнит порядок индексов в массиве, но без position information стандартный Transformer encoder не получает привилегированной геометрии этой сетки. Learned absolute/relative position schemes добавляют spatial relations; при новом resolution может требоваться корректная interpolation/иная схема.

Из linear embedding не следует, что semantic patches автоматически близки. Это свойство зависит от обучения всех последующих layers/objective. CNN имеет сильные locality/weight sharing priors; ViT получает часть locality через patches, но глобальное смешивание attention устроено иначе. Больше данных/pretraining часто помогает ViT, однако «ViT всегда требует гигантский dataset» слишком категорично: data-efficient training и transfer меняют условия. [ViT](https://arxiv.org/abs/2010.11929).

## 7. SSL и проверка представлений

В связи с SSL-проектом из чата полезно разделять objective и evaluation. Contrastive InfoNCE для query q, positive k+ и negatives k_j:

$$
\mathcal{L}_{\mathrm{InfoNCE}}=-\log\frac{\exp(\operatorname{sim}(q,k^+)/\tau)}{\sum_j\exp(\operatorname{sim}(q,k_j)/\tau)}.
$$

В denominator включается positive и выбранные negatives. Аугментации определяют, какие изменения считаются сохраняющими identity. Для похожих классов false negatives могут давать нежелательный сигнал. Non-contrastive methods предотвращают representation collapse другими механизмами, поэтому отсутствие явных negatives само по себе не ошибка.

Linear probe обучает только head на frozen features; full/partial fine-tuning меняет backbone. kNN — ещё один контроль геометрии. Сравнивать их на одном split, не считать высокий train score качеством representations. Collapse проявляется почти одинаковыми vectors для разных изображений; проверять variance/rank и downstream utility, а не только одну красивую проекцию.

## 8. Самопроверка и практика

1. Почему D не обязательно CP²? **Ответ:** E — произвольная обучаемая rectangular projection.
2. Сколько tokens с CLS у изображения $224\times224$ при размере patch $16\times16$? **Ответ:** 197.
3. Почему patch embedding не permutation-invariant? **Ответ:** перестановка входных patches переставляет outputs, а не оставляет всю последовательность прежней.
4. Что оптимизировать при замороженном backbone? **Ответ:** только выбранную head/прочие явно разрешённые parameters; features не меняются.

**Практика:** пересчитать $4\times 4$ пример и conv correspondence. Затем при $B=2$,H=W=32,$C=3$,$P=8$,$D=24$ выписать $patches:2\times 16\times 192$, $E:192\times 24$, $Z:2\times 16\times 24$, budget $192\cdot 24+24=4632$. Для реальных данных сравнить frozen+kNN, frozen+linear и одну partial-tuning конфигурацию; анализировать false positives/negatives по источникам, масштабу и backgrounds.

<!-- latex-display-v1 -->
