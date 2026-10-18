# Б. Pruning: удалить веса не всегда означает ускорить модель

[Оглавление](README.md). Пункты A.4–A.5. Дистилляция подробно раскрыта в [адаптации](<../../02-adaptation/Учебные конспекты/К — Дистилляция, teacher annealing и обучение на своих траекториях.md>).

**Основа:** S4, строки 2653–3441 и 3445–4767. Сохранены вопросы «почему сначала создают избыточные головы, а потом удаляют?» и предположение о деградации закрытых сервисов. **Исправления C10/C13/C17:** удаление параметров не гарантирует runtime speedup; жалоба на ухудшение не доказывает pruning. **Углубление:** параметры, FLOPs, sparse storage и причинная проверка.

## 1. Что удаляется при разных видах pruning

Unstructured pruning зануляет отдельные weights. Например, из $4\times 4$ матрицы можно оставить восемь элементов в произвольных местах. Dense matmul обычно всё ещё выполняет операции над 16 позициями: нули сами не создают sparse kernel. Sparse storage добавляет indices и может быть неэффективным для умеренной sparsity/маленьких blocks.

Structured pruning удаляет целые каналы, neurons, heads или blocks. Если линейный слой $D\to F$ теряет часть output channels, следующая операция должна потерять соответствующие input channels. Только согласованное изменение shapes даёт меньший dense graph. В residual branches внешние shapes должны продолжать совпадать.

Semi-structured sparsity использует ограниченные patterns, которые поддерживает hardware/kernel, например определённое число nonzero в группе. Arbitrary mask с тем же общим процентом нулей может не поддерживаться. Поэтому в отчёте нужны pattern, dtype, runtime и фактически измеренные характеристики.

Пример: $W_{\mathrm{up}}:F\times D$, $W_{\mathrm{down}}:D\times F$, $D=4$,$F=10$, без biases. Две матрицы имеют 80 weights. Удалим 4 промежуточных neurons и соответствующие строки up/столбцы down: остаётся $F=6$, 48 weights. Для такой ветви matmul arithmetic уменьшается, но весь Transformer содержит другие операции, и общий latency не обязан уменьшиться на 40%.

### Нули, отсутствующие данные и уменьшенные shapes

Вернёмся к матрице из S4: строки(0.1,0.5,0,0.3), (0,0.2,0.8,0), (0.4,0,0.1,0.6), (0,0.7,0,0.2). В ней уже 6zeros из 16positions. Удаление двух weights0.1 доводит число zeros до 8, то есть **50% final sparsity**. Это не удаление половины исходных nonzero weights: их было 10, удалены 2. Denominator важен при сравнении методов и schedules.

Для x=(1,2,3,4) исходный Wx=(2.3,2.8,3.1,2.2), после такого pruning —(2.2,2.8,2.8,2.2). Две маленькие weights дали разные output errors−0.1 и−0.3, потому что соответствующие inputs различались. Ноль в сохранённом dense tensor всё ещё занимает два bytes при FP16 и остаётся operand dense matmul. Чтобы не хранить его, нужен другой representation; чтобы не вычислять его contribution, нужен соответствующий execution path.

Для этой 4×4 матрицы FP16 dense storage=32bytes. Наивный CSR для 8nonzeros с 16bit values,32bit column indices и 5row pointers потребует 8·(2+4)+5·4=68bytes. Sparse encoding оказался больше. Для большой 1000×1000 матрицы те же правила дают 2MB dense против 6k+4004bytes CSR при k nonzeros. Выигрыш storage возникает примерно ниже 33.27% density, то есть выше 66.73% sparsity. Это пример конкретного encoding, а не универсальный порог: bitmasks, shorter indices и block formats дают другие числа.

CSR/CSC позволяют пропускать нулевые contributions на обычных CPU/GPU при наличии подходящих sparse kernels. Поэтому утверждение S4 «обычное hardware вообще не умеет sparse multiplication» слишком сильное. Практические препятствия — irregular memory access, индексы, слабое переиспользование, размер batch и уровень sparsity. Даже уменьшение числа multiply-adds не гарантирует меньшего времени, если усложнились reads и scheduling. Иногда полезен только меньший disk artifact; иногда и storage не уменьшится.

### Что именно отрезается в MLP и attention

Для gated MLP, в отличие от двухматричного примера выше, есть up, gate и down. При column inputs up/gate имеют F×D, down —D×F. Выбираем одно множество retained neurons S: оставляем **одни и те же** строки up/gate и соответствующие столбцы down. Иначе поэлементное произведение gate и up перестанет соединять нужные coordinates. Для D=4,F=10 всего 120weights; при F'=6 остаётся 72. Residual width D=4 не меняется, значит MLP-only pruning сам по себе не уменьшает KV-cache.

Удаление heads требует определить, что сохраняется: residual width D или head dimension d_h. Если D фиксирован, d_h фиксирован, H heads дают inner width H·d_h. Для обычного MHA Q/K/V имеют D×(H·d_h) в row convention, output projection —(H·d_h)×D. Удаление половины heads с согласованной output projection уменьшает эти четыре matrices вдвое. Фраза S4 «50%heads дают 25%параметров attention» не следует из этой схемы. Доля **всего Transformer** другая, потому что MLP и embeddings остались.

Если же уменьшить H, но увеличить d_h так, чтобы H·d_h=D сохранилось, число projection parameters не уменьшится только от нового числа heads. При GQA удаление query heads и KV groups —ещё разные операции. Нужны допустимый mapping, divisibility и поддерживаемые новые shapes. Оставить половину logits heads в mask —не то же самое, что экспортировать physically smaller projections.

Удаление 4blocks из 32 делает stack28-layer, но 12.5% reduction относится к одинаковым blocks и их cache. Общий parameter count также включает embeddings/output head. Уменьшение residual width потребует согласовать всю сеть: projections, normalization, residual additions, embeddings и final head. Удаление vocabulary entries дополнительно меняет token IDs/tokenizer, coverage текста и длину tokenization. Это не локальный безопасный способ «стереть редкие знания» без остальных изменений.

## 2. Как выбрать, что удалять

Magnitude-based pruning берёт малые |w|. Это простой heuristic, но значение веса само по себе зависит от масштабов соседних слоёв. Малый weight, умноженный на огромную activation, может быть важен. Для структурного элемента смотрят норму группы, outputs или sensitivity loss.

Taylor-интуиция: небольшое возмущение Δw меняет loss примерно как $g^\top \Delta w+0.5\Delta w^\top H\Delta w$. При удалении $\Delta w=-w$, но такой большой шаг не обязательно находится в области хорошего Taylor approximation. Поэтому importance score — приближение, а не сертификат безвредности.

Для head можно занулить её output и измерить изменение на независимой validation subset. Но одна голова может быть важна только для редкого среза. Удаление нескольких поодиночке «неважных» heads может дать общий сильный эффект из-за redundancy/interactions.

Обычный цикл: baseline → небольшой pruning step → recovery fine-tuning → evaluation → следующий step при выполнении quality constraint. Fine-tuning восстанавливает часть поведения, но добавляет training cost и риск forgetting. Сравнивать нужно с прямым обучением/адаптацией меньшей модели при сопоставимом общем бюджете.

### Малый gradient не означает ненужный weight

В S4 gradient-based importance объяснялась тем, что мало обновляющиеся weights якобы менее важны. Для L(w)=(w−10)² при w=10 gradient0, но pruning w→0 увеличивает loss до 100. Точное попадание в minimum не означает, что параметр можно удалить. В first-order Taylor score величина |w·g| оценивает конкретное perturbation−w; это local heuristic, который может провалиться при больших changes и близком к stationary point gradient.

Нужно различать |mean(g)|, mean(|g|) и mean(|w·g|). На разных examples signs могут компенсироваться: средний gradient почти 0, хотя отдельные task slices чувствительны. Аналогично большой activation не доказывает большой contribution, если его downstream weight мал или contributions взаимно компенсируются. Нормы, ablations и loss sensitivity дают разные observations, которые полезно сравнить.

Magnitude score также зависит от representation. Для ReLU network channel можно умножить на положительное c и разделить его downstream weights на c без изменения функции. Тогда часть weights стала меньше, но функция прежняя. Это контрпример универсальному смыслу «самые маленькие weights всегда самые ненужные». Он не доказывает, что magnitude pruning бесполезен: это дешёвый baseline с известными ограничениями.

### Wanda и SparseGPT: два разных использования activations

[Wanda](https://arxiv.org/abs/2306.11695) для W формы C_out×C_in использует score $S_{ij}=|W_{ij}|\|X_{:,j}\|_2$, где X объединяет calibration tokens по строкам. Сравнение ведётся внутри output row при заданной sparsity. Это конкретнее, чем mean(abs(activation)) из S4. Если weights(0.01,0.2), а input norms(100,1), scores=(1,0.2): маленький первый weight оказывается важнее второго. Метод не требует recovery weight update в своей базовой процедуре, однако это не гарантия нулевой потери при любой sparsity.

[SparseGPT](https://arxiv.org/abs/2301.00774) решает approximate layer reconstruction с компенсацией errors, используя second-order structure inputs. Связь с GPTQ понятна: там фиксируем weight на quantization level, здесь —на 0. Для local positive-definite Hessian минимальная цена удаления coordinate i с компенсацией остальных имеет вид $w_i^2/(2[H^{-1}]_{ii})$. Это условная quadratic задача, а не «теоретически оптимальный pruning всей LLM», как можно было понять из S4. Approximation, damping, calibration и уже удалённые coordinates ограничивают точность вывода.

### Coupled structures и слой как единица удаления

[LLM-Pruner](https://arxiv.org/abs/2305.11627) учитывает связанные structures и использует importance/recovery procedure. Dependency здесь означает: удаление одной dimension вынуждает изменить другие tensors, чтобы computation оставалось определённым. Успешное согласование shapes доказывает, что graph исполняется, но не сохранение его прежней функции. Recovery LoRA —один вариант подстройки survivors, со своим training budget.

[SliceGPT](https://arxiv.org/abs/2401.15024) использует invariance к согласованным transformations и principal components **signals/activations**, затем срезает dimensions transformed matrices. Формулировка S4 «PCA весов и все размеры меньше» упускает главное: до slicing transformation должна сохранять вычисление всей сети. Само truncation уже приближённое. Произвольный PCA одной weight matrix без изменения neighbours этого не воспроизводит. Числа quality retention из paper привязаны к моделям и задачам; они не постоянные 95% для любого checkpoint.

[ShortGPT](https://arxiv.org/abs/2403.03853), работа 2024 года, оценивает Block Influence по изменению направления hidden states: $\mathrm{BI}_i=1-\mathbb E[\cos(x_i,x_{i+1})]$. Small BI предлагает кандидата на удаление, но большой cosine не означает тождество функции: могут меняться magnitude и редкие examples. Результат «32→26 и 98%качества» из S4 не используем как универсальную характеристику метода. Это alternative к итеративному expensive recovery, а не доказательство, что любой single-shot или iterative method всегда лучше другого.

### Итерации: десять процентов от чего

Удаляя 10% **оставшихся** weights каждый round, через k rounds получаем retained fraction0.9ᵏ. После 6rounds это 0.531441, после 7 —0.4782969, то есть sparsity52.17031%, а не ровно 50%. Для exact target schedule задают cumulative sparsity и число удаляемых entries с учётом округления. При обычном monotone pruning mask только ужесточается; если allowed regrowth, это отдельный algorithm.

Между rounds recovery может перераспределить contributions, поэтому scores пересчитывают. Но повторение recovery epochs стоит compute, а progressive selection на одном test переобучает и recipe. Выбирают validation, фиксируют final test и сравнивают total budget с one-shot pruning и compact-model baseline. Если independent ablation каждой из двух redundant heads дала малый effect, совместное удаление обеих всё ещё может оказаться большим: оставшийся путь, который компенсировал первую потерю, теперь тоже исчез.

## 3. Почему не спроектировать сразу оптимальный размер

Вопрос чата содержит разумную альтернативу. Нет общего закона, что всегда выгодно сначала учить максимальную модель. Overparameterization иногда облегчает optimization и создаёт redundancy, которая помогает восстановлению после удаления. Но заранее неизвестно, какие representations важны для конкретной задачи и distribution.

Поиск архитектуры тоже стоит вычислений. Маленькая модель может быть труднее в optimization при тех же данных, а большая — удобным teacher. Однако конкретное сравнение «большая+pruning» против «маленькая с нуля» зависит от данных, compute и target hardware. Полезный baseline — компактная architecture без pruning, а не вера в единственный путь.

Также training optimum и serving optimum различаются: shape, удобная для GPU tensor cores, может быть быстрее чуть меньшего, но неудобного shape. Удаление одной головы не обязательно сократит kernel time, если остаются прежние блоки выравнивания/коммуникаций.

### Более полный ответ на ваш вопрос об избыточных heads

«Сначала больше, потом меньше» —один из способов искать функцию, но не необходимый этап жизни каждой сети. Сначала выбирают model family и budget, затем weights учатся на data distribution. Полезность head выявляется в этом совместном процессе. Из возможности убрать head **после** обучения не следует, что та же модель обучилась бы так же, если **до** обучения лишить её этой degree of freedom. Сравниваются разные optimization trajectories.

Избыточность может позволить нескольким representations конкурировать и частично дублировать друг друга. Однако красивые роли «одна голова только для синтаксиса, другая только для сущностей» из S4 —интуитивные примеры, не обязательные выделенные modules. Функции могут распределяться между heads и layers; редкий механизм почти не заметен на усреднённом calibration corpus. Поэтому pruning под один domain уменьшает требования к сохранению других behaviors и не доказывает универсальную ненужность удалённых units.

[Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635) ставит вопрос о subnetworks и их подходящей initialization/training. Найти sparse mask, которая работает с обученными weights, и доказать, что эта topology обучается с нуля из любого random initialization, —разные результаты. Наличие winning tickets в изученной постановке не превращает всякую pruned LLM в изначально найденную оптимальную architecture.

Оптимальность требует criterion: minimal peak memory при quality≥q; минимальная p95 latency при заданном concurrency; меньше энергии на успешный ответ; либо больший quality при fixed training compute. Один checkpoint может быть хорошим teacher для множества downstream artifacts, и стоимость общего pretraining распределяется между ними. Но если задача узкая и компактная модель обучается достаточно хорошо, тренировать огромную специально ради её pruning может быть невыгодно. Гипотезу о выгоде нужно сравнить с этим baseline.

NAS ищет в пространстве architectures, а данные, training schedule и proxy evaluations определяют стоимость поиска. Это не обязательно тысячи полных pretraining runs, но и не бесплатный oracle. MobileNet/EfficientNet из S4 иллюстрируют hardware-aware compact design; DistilBERT/TinyBERT —также knowledge transfer. У них разные причины компактности, поэтому нельзя объявить все примеры одним доказательством «маленькая с нуля всегда хуже».

## 4. Где граница с distillation

Pruning сохраняет существенную часть исходных весов/структуры и удаляет выбранные элементы. Distillation учит student на signals teacher и может выбрать новую architecture. Quantization сохраняет topology, но меняет представление чисел. На практике возможен student, который затем pruning/quantized, но эффекты нужно измерять раздельно.

Для выбора serving artifact сравнить: исходная модель, pruned+recovered, distilled student и quantized baseline. Метрики — task quality, per-slice failures, peak memory, time/throughput на одинаковом workload, total preparation cost. Меньшее число параметров без нового benchmark — ещё не практическая победа.

**Числовой пример ограниченного ускорения:** доля времени исходной системы на оптимизируемой ветви 40%; её удалось ускорить вдвое. По закону Амдала общий speedup $\frac{1}{0.6+\frac{0.4}{2}}=1.25$, а не 2. Если оставшийся bottleneck — tokenization/queue/tool latency, эффект сжатия linear layers ещё меньше.

### Комбинация pruning и quantization требует новой проверки

После structural pruning меняются activations survivors, а иногда и shapes. Quantization scales, calibration statistics и group alignment от исходной модели могут больше не подходить. Поэтому последовательность pruning→recovery→calibration→quantization —разумный проверяемый experiment, а не гарантированно лучший порядок. При reverse order либо joint optimization задача другая: ошибки quantization и удаления могут взаимодействовать.

Если удалено 30% параметров и оставшиеся вместо 16bits хранятся в 4bits, идеальный weight payload равен 0.7·4/16=0.175 исходного, то есть reduction82.5% или≈5.71× по этой компоненте. Это не та же степень end-to-end compression, поскольку cache, float tensors и metadata сохранились. Для decoder block MLP pruning не меняет Hkv и layers, а head/group/layer pruning может менять cache по разным множителям.

Новый artifact должен содержать корректные config shapes, weights, tokenizer/template и поддерживаемый runtime graph. «Structured значит работает везде, где оригинал» из S4 неверно: runtime может предполагать одинаковую ширину каждого layer или фиксированную схему heads. Сначала проверяют load/forward, затем quality и только затем runtime benefit. Hardware-friendly rounding размеров иногда оставляет немного больше parameters ради более быстрого kernel; минимум parameter count и минимум latency не обязаны совпасть.

## 5. Можно ли объяснить ухудшение сервиса pruning

В чате обсуждалась такая гипотеза о закрытых моделях. Наблюдаемое качество зависит также от model revision, routing, system instructions, decoding, доступных tools, context truncation и выбранных задач. Без опубликованного свидетельства/контролируемого сравнения нельзя установить скрытый training/deployment recipe. Это не запрет обсуждать pruning; это разграничение механизма и атрибуции.

Практический разбор ухудшения: сохранить один и тот же набор inputs/config, повторить запросы, выделить тип ошибки, сравнить известные versions, проверить sampling variation. Дальше формулировать гипотезы с границами evidence. Не подменять гипотезу утверждением «они точно удалили головы».

### Возвращаем гипотезу о деградации закрытых сервисов

Вы предположили, что жалобы на ухудшение моделей могут объясняться циклом pruning+fine-tuning. Это допустимая гипотеза, но ответ S4 превратил её в неподтверждённый рассказ о конкретных компаниях: исходных «десятках триллионов параметров», скрытой замене teacher на student, обязательном FP8 и редкости production pruning. По предоставленным материалам такие внутренние устройства не установлены. Наблюдаемое ухудшение не позволяет выбрать одну причину среди нескольких скрытых изменений.

Разберём возможные механизмы без выдуманной атрибуции. Compression может изменить rare-task behavior, но не обязана ухудшать каждую задачу. Preference training может увеличить false refusals или сдвинуть стиль; это возможный trade-off, а не закон «RLHF убивает интеллект». Continual fine-tuning может ухудшить старую задачу, но для forgetting нужно сравнить один и тот же old-task evaluation до и после update. Routing и доступные tools меняют поведение целой системы даже без изменений weights конкретного checkpoint.

MoE также не тождествен динамическому pruning. Router выбирает предусмотренные обученной architecture experts для текущего input. Это sparse execution, но из него не следует, что сначала dense layers вырезали из той же модели. Короткий ответ сам по себе не доказывает сокращения внутреннего reasoning budget, а высокий/низкий output entropy не является универсальной шкалой рассуждения. Наконец, Goodhart-like оптимизация benchmark может расходиться с вашим application distribution и без доказанного намеренного запоминания тестов.

В проверяемом сравнении фиксируют точные prompts, system/template, доступный context/evidence, tools, decoding и объявленную version. Повторяют стохастические запросы, сохраняют ошибки по срезам и проверяют, что старые и новые inputs действительно совпадали. Такой опыт может подтвердить падение качества на выбранном наборе. Чтобы установить скрытый pruning recipe, нужны дополнительные свидетельства о самой реализации; одна и та же ошибка JSON может возникнуть от многих причин. Так ваш вопрос сохраняется в конспекте вместе с механизмами и границами вывода.

1. Почему 50% нулевых weights может дать нулевой speedup? **Ответ:** dense kernel всё равно обрабатывает исходный shape; sparse overhead/поддержка не обеспечены.
2. Что нужно согласовать при удалении MLP neurons? **Ответ:** соответствующие output/input dimensions соседних проекций и invariants блока.
3. Чем измерять пользу pruning? **Ответ:** quality constraints и end-to-end runtime/memory, плюс recovery cost.

**Практика:** нарисовать masks для unstructured 50% sparsity и удаления четырёх MLP neurons в примере 4→10→4. Посчитать shapes/weights и объяснить, где может потребоваться новый kernel или graph export.

<!-- source-depth-inference-Б -->
