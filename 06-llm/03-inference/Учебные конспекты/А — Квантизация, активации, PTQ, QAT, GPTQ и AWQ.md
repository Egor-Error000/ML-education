# А. Квантизация: что хранится, что вычисляется и где появляется ошибка

[Оглавление](README.md). Объединены inference A.1–A.3 и практический протокол 2.

**Основа:** S4, строки 4–2649; S3, 674–715. Сохранены вопросы про activations, обозначения GGUF в локальном интерфейсе, обучение сразу в INT4, Hessian/Jacobian и «online quantization». **Исправления C05/C10:** алгоритм, representation, container и runtime разделены; нет правила «INT4 обязательно ломает модель», «1% весов делает 99% работы» или универсального dtype для JSON. **Углубление:** ручной affine quantizer, ошибка выхода слоя и связь Hessian с calibration inputs.

## 1. Веса и активации — разные массивы

Вес W хранится между запросами и задаёт функцию. Активации X/Y — текущие входы, hidden states и другие промежуточные результаты. Для $Y=XW$ новые тексты меняют X, а не автоматически W. Веса можно подготовить offline, диапазоны activations зависят от данных и положения в сети.

Меньшая разрядность уменьшает storage/bandwidth, но вводит ошибку. FP32, FP16, BF16 имеют exponent/mantissa; FP16 и BF16 оба 16-bit, но у BF16 шире exponent range и меньше точность mantissa. INT8/INT4 кодируют небольшое число integer levels с scale/zero-point; это не «обычные float с меньшим числом знаков».

Четыре независимых уровня:

- **Алгоритм:** GPTQ/AWQ выбирают, как подготовить quantized weights.
- **Представление:** например, INT4 или NF4 задаёт значения/кодирование.
- **Файл:** GGUF — container с tensors/metadata, а не один quantization algorithm.
- **Runtime/kernel:** выполняет доступные операции на конкретном hardware.

Название вроде `Q4_K_*` относится к конкретной схеме quantization/упаковки в соответствующей экосистеме. Это не прямое имя architecture и не доказательство, что activations и KV тоже 4-bit. GGUF не означает CPU-only; backend/offload определяет размещение. Конкретный suffix и поддержку нужно читать в metadata/документации выбранной версии, а не угадывать по «4».

## 2. Простой affine quantizer

Определим scale s>0, integer zero-point z и диапазон $[q_{\min},q_{\max}]$:

$$
q=\operatorname{clip}\!\left(\operatorname{round}\!\left(\frac xs\right)+z,q_{\min},q_{\max}\right),\qquad\widehat x=s(q-z).
$$

Для symmetric signed примера с уровнями -7…7, z=0, max|x|=1.4 берём s=0.2. Вектор $\left(-1.4,\,-0.3,\,0.2,\,1.3\right)$ приблизится к $\left(-1.4,\,-0.4,\,0.2,\,1.2\right)$, если в этом примере half-ties округлять от нуля. Реальные kernels могут иметь round-to-even и иной signed range: в учебном расчёте convention задана явно.

Ошибки здесь $\left(0,\,-0.1,\,0,\,-0.1\right)$, MSE=0.005. Без clipping максимальное rounding error около s/2. При clipping выбросов ошибка может быть гораздо больше, зато меньшее s повышает точность большинства значений. Calibration выбирает trade-off на representative inputs.

Asymmetric quantization позволяет ненулевой z. Например, для x∈[0,3] и unsigned 4-bit 0…15 удобно $s=0.2$,z=0. Для сдвинутого диапазона scale/zero-point выбираются совместно; точная достижимость min/max зависит от integer rounding и clipping. Это полезно для несимметричных activations, но добавляет детали kernels/metadata.

Per-tensor scale общий для всего массива; per-channel — отдельный для каналов; per-group — для небольших блоков. Если в одном блоке есть очень крупный outlier, общий s становится крупным и мелкие значения схлопываются. Более мелкие группы уменьшают конфликт диапазонов, но увеличивают metadata и иногда overhead. Выбор channel axis зависит от ориентации W и matmul.

## 3. Почему activation outliers важны даже при weight-only quantization

Пусть для столбцов $y=Wx$, после quantization $\widehat W=W+\Delta W$. Тогда ошибка выхода $\Delta y=\Delta W\,x$. Одинаковая ошибка веса может быть безразличной при x_j≈0 и существенной при большом x_j. Поэтому minimizing только $\lVert W-\widehat W\rVert_F^2$ не всегда сохраняет outputs хорошо.

Пример: один выход $y=w_1x_1+w_2x_2$, $x=\left(100,\,1\right)$. Ошибка веса 0.01 в первой координате даёт output error 1; та же ошибка во второй — 0.01. Это показывает dependence on activations без лозунга о «VIP 1%».

Calibration corpus должен отражать реальные languages, lengths и task mix. Если статистика собрана только на коротком английском тексте, качество русских длинных tool prompts не доказано. Метки обычно не обязательны для такой calibration, но inputs должны быть независимы от final test выбора гиперпараметров.

## 4. PTQ и QAT

PTQ готовит quantized representation после основного training. Простая calibration/rounding и продвинутые error-compensation methods отличаются качеством/ценой. PTQ не означает «без вычислений»: могут потребоваться forward passes, statistics и оптимизация reconstruction.

QAT включает имитацию округления/clipping в training forward, чтобы параметры адаптировались к будущей ошибке. Fake quantization может вычислять dequantized values в float и не сжимает training memory автоматически до integer bitwidth. Поскольку round имеет нулевую производную почти везде и скачки на границах, часто используют straight-through estimator: surrogate backward, а не истинную derivative round.

На вопрос «почему сразу не обучать огромную модель в INT4» ответ: discretization ограничивает small updates, gradients/optimizer states нуждаются в иной точности/representation, а kernels и training stability зависят от метода. Low-bit training исследуется; из неудачи наивного integer SGD не следует невозможность всех специальных подходов. QAT также не гарантирует выигрыш при любом dataset/budget.

## 5. GPTQ, AWQ и Hessian

GPTQ использует approximate second-order information для compensation при последовательной quantization весов слоя. В упрощённой layer reconstruction задаче с row inputs X:

$$
\mathcal{E}=\lVert XW-X\widehat W\rVert_F^2.
$$

Для одной output column Hessian квадратичной ошибки по weight vector пропорционален $X^\top X$ (точный коэффициент зависит от 1/2 и нормировки). Это не обязательно полный Hessian всей language-model CE по миллиардам параметров. По этой структуре оценивается, как компенсировать rounding error другими ещё не зафиксированными весами. [GPTQ](https://arxiv.org/abs/2210.17323).

**Jacobian** содержит первые derivatives векторной функции: для $f:\mathbb{R}^{d}\to\mathbb{R}^{m}$ форма $m\times d$. **Hessian** — вторые derivatives scalar function: $d\times d$. Для $f(u,v)=u^2+3uv+2v^2$ gradient $(2u+3v,3u+4v)$, Hessian $\begin{bmatrix}2 & 3\\3 & 4\end{bmatrix}$. Wronskian, о котором вспомнили в чате, связан с derivatives набора функций и их линейной независимостью; это другое понятие.

AWQ выбирает activation-aware channel scales, чтобы снизить чувствительную quantization error без обязательного хранения «важных весов» в отдельном high dtype. Для столбцов можно записать тождество $Wx=(WD)(D^{-1}x)$ с обратимой diagonal D. До quantization функция та же; после quantization выбор D меняет rounding/clipping. Дальше важно, можно ли scales эффективно слить в соседние операции. [AWQ](https://arxiv.org/abs/2306.00978).

Нельзя из статьи вывести вечное «AWQ быстрее GPTQ на всех GPU» или одинаковую runtime поддержку. Сравниваются concrete configs, hardware и quality constraint.

## 6. Что означает «динамическая/online» квантизация

Dynamic activation quantization вычисляет scales по текущим inputs, например на вызове linear layer; это не обязательно изменение weight representation после каждого пользовательского токена. KV-cache quantization — ещё один независимый выбор. Adaptive bitwidth switching — специальная runtime policy, требующая собственных kernels/quality tests.

Если runtime при каждом запросе заново переквантует всю большую базу, overhead может превысить выгоду. Если динамически меняется только activation scale, overhead и смысл другие. На вопрос из чата сначала нужно уточнить **какой массив** и **когда** меняется.

## 7. Проверка и упражнение

Оценка weight storage $\frac{Pb}{8}$ — нижняя грубая часть бюджета: для 7B, 4-bit это 3.5 GB до scales, unquantized tensors, padding и runtime. Speedup не следует из четырёхкратного уменьшения weight bytes: нужны подходящие kernels, а workload может ограничиваться prefill compute/очередью/KV.

Сравнить float baseline и одну quantized конфигурацию на одинаковых prompts: correctness, schema/tool correctness, peak memory, TTFT, tokens/s, p95 при concurrency. Нельзя назначать «для JSON минимум Q5/Q6» без конкретного эксперимента.

1. Почему маленький MSE весов не гарантирует маленький output error? **Ответ:** ошибка умножается на inputs; важны направления/статистика X.
2. Чем Hessian отличается от Jacobian? **Ответ:** вторые derivatives scalar objective против первых derivatives векторной функции.
3. Что меняет QAT backward через round? **Ответ:** обычно surrogate estimator; это не обычная точная derivative.

**Практика:** пересчитать affine example, затем добавить outlier 14 вместо 1.4 и взять один общий scale. Показать, как ухудшилась различимость малых чисел; сравнить две группы с отдельными scales. Учесть metadata, а не только MSE.

<!-- latex-display-v1 -->
