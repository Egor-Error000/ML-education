# А. Квантизация: что хранится, что вычисляется и где появляется ошибка

[Оглавление](README.md). Объединены inference A.1–A.3 и практический протокол 2.

**Основа:** S4, строки 4–2649; S3, 674–715. Сохранены вопросы про activations, обозначения GGUF в локальном интерфейсе, обучение сразу в INT4, Hessian/Jacobian и «online quantization». **Исправления C05/C10/C16:** алгоритм, representation, container и runtime разделены; нет правила «INT4 обязательно ломает модель», «1% весов делает 99% работы» или универсального dtype для JSON. **Углубление второй редакции:** путь activations через Transformer, ручные quantizers, metadata и память, calibration, LLM.int8()/SmoothQuant, QAT/STE, вывод одного шага GPTQ, численный AWQ и ограничения online requantization.

## 1. Веса и активации — разные массивы

Вес W хранится между запросами и задаёт функцию. Активации X/Y — текущие входы, hidden states и другие промежуточные результаты. Для $Y=XW$ новые тексты меняют X, а не автоматически W. Веса можно подготовить offline, диапазоны activations зависят от данных и положения в сети.

Меньшая разрядность уменьшает storage/bandwidth, но вводит ошибку. FP32, FP16, BF16 имеют exponent/mantissa; FP16 и BF16 оба 16-bit, но у BF16 шире exponent range и меньше точность mantissa. INT8/INT4 кодируют небольшое число integer levels с scale/zero-point; это не «обычные float с меньшим числом знаков».

Четыре независимых уровня:

- **Алгоритм:** GPTQ/AWQ выбирают, как подготовить quantized weights.
- **Представление:** например, INT4 или NF4 задаёт значения/кодирование.
- **Файл:** GGUF — container с tensors/metadata, а не один quantization algorithm.
- **Runtime/kernel:** выполняет доступные операции на конкретном hardware.

Название вроде `Q4_K_*` относится к конкретной схеме quantization/упаковки в соответствующей экосистеме. Это не прямое имя architecture и не доказательство, что activations и KV тоже 4-bit. GGUF не означает CPU-only; backend/offload определяет размещение. Конкретный suffix и поддержку нужно читать в metadata/документации выбранной версии, а не угадывать по «4».

### Ваш вопрос «что такое активации» — путь одного тензора

В S4 вы отдельно попросили подробнее объяснить активации, поэтому определения «это промежуточные значения» недостаточно. Пусть вход содержит B последовательностей по L токенов, ширина model representation D. Таблица embeddings E размера V×D — обучаемые параметры. Результат выбора строк этой таблицы по token IDs — уже активации X размера B×L×D. Одна и та же строка E может использоваться в тысячах запросов, но X принадлежит конкретному forward. Embedding weights и выбранные embedding vectors играют разные роли, хотя числа в момент lookup могут совпадать.

Для простого pre-norm Transformer блок можно читать так: $H=\operatorname{Norm}(X)$; затем $Q=HW_Q$, $K=HW_K$, $V=HW_V$; scores строятся из QKᵀ, после mask и softmax получается A; AV проходит output projection; результат складывается с X. Затем второй normalization и MLP снова создают промежуточные массивы и residual sum. В gated MLP отдельно вычисляются up и gate, применяется нелинейность и поэлементное произведение, затем down projection возвращает ширину D. Q/K/V, scores, softmax probabilities, MLP intermediates и logits — разные активации с разными shapes, распределениями и сроками жизни.

Слово «активации» здесь шире, чем «результат activation function». Даже output обычного linear layer без GELU — активация в этом смысле. Attention probabilities после softmax не имеют такого же распределения, как residual stream: они неотрицательны и нормированы по соответствующей оси. Нельзя автоматически назначить всем промежуточным массивам одну симметричную сетку и одну статистику.

Каждый Transformer layer формирует **собственный** K/V cache. Формулировка схемы S4 «слой 2 использует KV-cache слоя 1» неверна для обычного decoder: layer2 получает hidden states предыдущего слоя, но хранит K/V, вычисленные своими projections. Query текущей позиции обычно не нужно сохранять для следующих decode steps. K/V остаются дольше большинства прочих intermediates именно потому, что следующие запросы attention обращаются к истории.

В inference память многих временных activations переиспользуется после выполнения операции. В training часть intermediates сохраняется для backward либо пересчитывается при checkpointing. Поэтому нельзя умножить сумму всех перечисленных tensors на число layers и назвать результат обязательным inference peak. Fused attention может вообще не материализовать полную scores matrix в global memory. Логические tensors в формуле и одновременно выделенные buffers runtime — разные уровни описания.

### Восстанавливаем и исправляем числа памяти из чата

Для B=1,L=2048,D=4096 и MHA с 32heads в двухбайтовом dtype один hidden tensor занимает 16MiB. Q/K/V вместе —48MiB, если каждый имеет полную ширину D. Наивно сохранённые scores имеют 32×2048×2048 элементов и занимают **256MiB**, а не 128MB из S4. MLP tensor ширины 11008 занимает 43MiB. Два gated intermediates и их произведение нельзя без анализа lifetime считать одним таким tensor, но нельзя и всегда считать тремя одновременно хранимыми.

При GQA ширина K/V уменьшается до Hkv·dh, поэтому исходное «3×L×D» тоже перестаёт быть точным. Для 80layers,Hkv=8,dh=128,L=8192,B=1 двухбайтовый cache занимает 2.5GiB: множитель 2 учитывает K и V. Подстановка всей hidden width8192 вместо KV width1024 завышает его до 20GiB. Более того,20GiB всё равно меньше примерно 70GB однобайтовых весов 70B модели: обратное сравнение в S4 было арифметически неверно. При достаточно большой суммарной длине sequences cache действительно может стать больше weights, но это нужно посчитать, а не объявлять по имени модели.

### Форматы: диапазон, точность и единица хранения

У FP32 есть sign1bit, exponent8bits и stored fraction23bits; у FP16 —1/5/10; у BF16 —1/8/7. Для normal binary floats ведущая единица significand хранится неявно. Это объясняет, почему одинаковое общее число bits не означает одинаковую сетку. Около 1 шаг FP16 равен 2⁻¹⁰≈0.00097656, BF16 —2⁻⁷=0.0078125. Например,1.001 округляется в FP16 примерно до 1.00097656, а в BF16 при round-to-nearest —до 1.0. При большом magnitude абсолютный шаг плавающей точки увеличивается.

Максимальное конечное FP16 число 65504, у BF16 —порядка 3.39×10³⁸. Более широкий exponent range уменьшает вероятность overflow при тех же значениях, но не «гасит loss spikes» и не устраняет ошибку алгоритма. Значения вне диапазона и слишком мелкие значения могут переполниться или потеряться; реальное поведение subnormal зависит также от hardware mode. Низкая точность — не только число знаков после запятой.

INT4 содержит 16 кодов, но их real values определяются scale и zero-point. При s=10 код 6 представляет 60: утверждение S4 «INT4 физически не может создать activation+60, потому что максимум+7» смешивало integer code и decoded value. Сложность в том, что при большом s рядом с 60 трудно различить малые значения. NF4 использует другую, неравномерную кодовую сетку и тоже не тождествен обычному равномерному INT4. Формат хранения, dtype multiplication и dtype accumulator нужно указывать отдельно.

### Почему вы видели Q4_K_M, а не BF16

Ваше наблюдение о списке файлов в LM Studio — вопрос об artifact, который показал каталог, а не доказательство невозможности float weights в локальном runtime. Quantized variants легче разместить в памяти, поэтому они часто встречаются среди распространяемых файлов. В каталоге могли быть только такие conversions; точное название модели из реплики не используем для догадки о её architecture или поддержке. Нужны metadata конкретного файла и version runtime.

Обозначения Q8_0, Q4_K_S, Q4_K_M, Q5_K_M, Q6_K, Q3_K_M, Q2_K и IQ2_XXS из чата относятся к разным recipes/types в экосистеме GGUF. Число в имени не равно точной file-average bitwidth, а S/M не является шкалой независимой от модели гарантированной точности. K-quants используют блочные схемы с собственными scales/mins; интерпретация K как обязательного «K-means clustering» в S4 не обоснована. IQ family тоже нельзя описать просто как тот же affine INT4 с меньшим числом bits.

Проверяемый пример из [структур ggml](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h): Q4_K block для 256 значений содержит 128bytes codes,12bytes упакованных scales/mins и 4bytes двух half super-scales. Итого 144bytes, или 4.5bits на значение. Q4_K_M — recipe смешения tensor types, поэтому средний bitwidth всего файла может отличаться от 4.5. Шестибитные **scales** и шестибитные **weight codes** — разные вещи; фраза «6 бит или 6 битные масштабы» не позволяет считать storage как будто это один формат.

GGUF содержит tensors и metadata, в том числе сведения, нужные для tokenization/architecture, но файл не является самозапускающимся executable с произвольным готовым compute graph. Runtime должен знать architecture и поддерживать tensor types/backend. CPU/GPU offload выбирает исполнение; из GGUF не следует ни CPU-only, ни автоматическая поддержка любого GPU. Аналогично GPTQ/AWQ — методы подготовки, которым соответствуют конкретные layouts и kernels; они не задают вечного ranking скоростей против GGUF.

### imatrix и EXL2: два уточнения к локальному запуску

В S4 упоминался imatrix, но его роль не сводится к «запрету трогать важные веса». В [llama-imatrix](https://github.com/ggml-org/llama.cpp/tree/master/tools/imatrix) на выбранном тексте собирают статистику входных activations для последующей квантизации. Связь с математикой главы видна из output error: для одной weight row она зависит от input Gram matrix. Если использовать только diagonal approximation, разные координаты получают вес, связанный с суммой квадратов их входных активаций. Это полезная оценка чувствительности, а не semantic knowledge map и не полный Hessian всей LLM. Результат зависит от calibration corpus, coverage tensors и поддерживаемого quantizer.

Последовательность действий: выбрать точный исходный checkpoint и representative texts; получить importance statistics; передать их поддерживаемой процедуре conversion; оценить полученный artifact на независимых задачах. Присутствие imatrix в имени файла само по себе не подтверждает ни происхождение calibration data, ни выигрыш на вашем домене. Один и тот же final tensor type может быть получен с разной подготовкой.

В [ExLlamaV2](https://github.com/turboderp-org/exllamav2) различаются поддержка GPTQ checkpoints и собственный EXL2 format. EXL2 использует связанные идеи оптимизации и допускает смешение bitwidth с целевой средней bitrate. Поэтому реплика S4 «GPTQ, то есть формат EXL2» неточна. Например, среднее4bits/value может получаться из неодинаковой precision разных частей модели; это не обязательно uniform INT4 каждой weight matrix. Bitrate budget и точность подбираются по calibration, а runtime должен поддержать итоговый layout. Обещание «это самый быстрый вариант для любой3090/4090» из одного названия не следует: benchmark включает model, context, batch, kernels и прочий workload.

## 2. Простой affine quantizer

Определим scale s>0, integer zero-point z и диапазон $[q_{\min},q_{\max}]$:

$$
q=\operatorname{clip}\!\left(\operatorname{round}\!\left(\frac xs\right)+z,q_{\min},q_{\max}\right),\qquad\widehat x=s(q-z).
$$

Для symmetric signed примера с уровнями -7…7, z=0, max|x|=1.4 берём s=0.2. Вектор $\left(-1.4,\,-0.3,\,0.2,\,1.3\right)$ приблизится к $\left(-1.4,\,-0.4,\,0.2,\,1.2\right)$, если в этом примере half-ties округлять от нуля. Реальные kernels могут иметь round-to-even и иной signed range: в учебном расчёте convention задана явно.

Ошибки здесь $\left(0,\,-0.1,\,0,\,-0.1\right)$, MSE=0.005. Без clipping максимальное rounding error около s/2. При clipping выбросов ошибка может быть гораздо больше, зато меньшее s повышает точность большинства значений. Calibration выбирает trade-off на representative inputs.

Asymmetric quantization позволяет ненулевой z. Например, для x∈[0,3] и unsigned 4-bit 0…15 удобно $s=0.2$,z=0. Для сдвинутого диапазона scale/zero-point выбираются совместно; точная достижимость min/max зависит от integer rounding и clipping. Это полезно для несимметричных activations, но добавляет детали kernels/metadata.

Per-tensor scale общий для всего массива; per-channel — отдельный для каналов; per-group — для небольших блоков. Если в одном блоке есть очень крупный outlier, общий s становится крупным и мелкие значения схлопываются. Более мелкие группы уменьшают конфликт диапазонов, но увеличивают metadata и иногда overhead. Выбор channel axis зависит от ориентации W и matmul.

### Доводим исходный пример до integer codes

В S4 веса(−0.8,−0.2,0.3,0.9) были приведены без самого расчёта. Для symmetric INT8 с codes−127…127 берём s=0.9/127,z=0. После nearest rounding codes равны(−113,−28,42,127), decoded weights примерно(−0.800787,−0.198425,0.297638,0.9). Низкий code не является потерянным весом: при matmul используется соответствующий scale. Выбор симметричных 255 уровней здесь намеренный; полное signed INT8 множество содержит 256 кодов, и treatment−128 зависит от схемы.

Для того же вектора min-max affine quantizer с unsigned codes0…255 даёт s=(0.9+0.8)/255=1/150,z=120. Codes(0,90,165,255) восстанавливают четыре числа точно в арифметике real numbers. Это удачное совпадение сетки с данным маленьким набором, а не доказательство, что asymmetric scheme всегда безошибочна. Zero-point нужен в том числе для точного представления реального нуля, хотя endpoints после округления z не всегда оба достигаются точно.

Скалярное произведение двух affine-quantized векторов имеет вид:

$$
\widehat x^\top\widehat w=s_xs_w\left(\sum_i q_{x,i}q_{w,i}-z_w\sum_iq_{x,i}-z_x\sum_iq_{w,i}+n z_xz_w\right).
$$

Поэтому ненулевые zero-points могут требовать correction terms или их algebraic folding. Даже если inputs INT8, accumulator часто имеет большую разрядность: произведений много, и сумма должна вместить их диапазон. Термин W8A8 не означает, что абсолютно всё, включая sums, normalization и softmax, выполняется в восьмибитной целочисленной арифметике.

### Ошибка округления и clipping — разные потери

Внутри неперегруженного интервала при nearest rounding absolute error ограничена примерно s/2. Это bound на отдельное значение, а не на итоговый language-model loss. Если clipping range ограничен±1, input100 восстанавливается около 1 и ошибка около 99: bound s/2 для такого outlier уже неприменим. Меньший range повышает разрешение остальных значений, но эта выигранная точность оплачена обрезанием tail.

Для symmetric INT8 с absmax1 шаг 1/127≈0.007874. При absmax100 шаг 100/127≈0.7874. В S4 scale местами назывался обратным коэффициентом 127/absmax, тогда как в этой главе s — **real-value step**, absmax/127. Обе parameterizations возможны, но подставлять обратный коэффициент в формулу деквантизации нельзя.

Пара 0.1 и 0.5 из S4 при s≈0.7874 не обязана схлопываться: nearest codes равны 0 и 1. Зато 0.1 и 0.3 оба превращаются в 0. Проблема range pollution реальна, но конкретный пример нужно проверять. Из него также не следует «информация полностью потеряна во всём tensor»: часть значений различается, а downstream impact зависит от использования этих directions.

### Гранулярность задаёт, кто разделяет сетку

Для X формы N×D per-token scale выделяет сетку каждой строке, per-channel — каждому feature столбцу. Это разные ограничения. Если один большой feature присутствует во всех tokens, per-token scale всё ещё вынужден в каждой строке учитывать этот feature. Если один token целиком имеет большой magnitude, per-channel scale по всем tokens может передать его влияние остальным строкам. Структуру outliers нужно измерить по обеим осям.

Для W формы D_in×D_out слова input/output channel нужно связывать с указанной ориентацией, а не запоминать «столбцы всегда такие». Group size32/64/128 означает число значений, обслуживаемых одним набором metadata по выбранному layout. Per-group не находится автоматически выше per-channel по точности: разбиения могут идти по разным осям, а heuristics выбора scales отличаются. При вложенном разбиении и свободном точном поиске старые scales остаются допустимым решением, но реальный quantizer не обязан находить этот optimum.

Если на g четырёхбитных weights хранится один 16bit scale, effective bits/value=4+16/g. При g=128 получаем 4.125, при 32 —4.5; если добавить 16bit zero-point, соответственно 4.25 и 5. Если g=1, один scale уже дороже исходного четырёхбитного code. Это объясняет metadata trade-off без обещания, что smallest groups дают лучший runtime. В настоящем формате scales сами могут быть quantized или shared между sub-blocks, как в примере Q4_K.

## 3. Почему activation outliers важны даже при weight-only quantization

Пусть для столбцов $y=Wx$, после quantization $\widehat W=W+\Delta W$. Тогда ошибка выхода $\Delta y=\Delta W\,x$. Одинаковая ошибка веса может быть безразличной при x_j≈0 и существенной при большом x_j. Поэтому minimizing только $\lVert W-\widehat W\rVert_F^2$ не всегда сохраняет outputs хорошо.

Пример: один выход $y=w_1x_1+w_2x_2$, $x=\left(100,\,1\right)$. Ошибка веса 0.01 в первой координате даёт output error 1; та же ошибка во второй — 0.01. Это показывает dependence on activations без лозунга о «VIP 1%».

Calibration corpus должен отражать реальные languages, lengths и task mix. Если статистика собрана только на коротком английском тексте, качество русских длинных tool prompts не доказано. Метки обычно не обязательны для такой calibration, но inputs должны быть независимы от final test выбора гиперпараметров.

### Из чего состоит calibration, если у нас ещё нет low-bit модели

В простом static pipeline сначала фиксируют float checkpoint и набор input sequences. На выбранных узлах forward observers собирают min/max, histogram, quantiles или channel statistics; затем выбирают quantizer parameters и конвертируют graph/tensors. Обычно target labels для такой статистики не нужны, а сам исходный pretraining corpus не обязателен. Это не означает «данные совсем не нужны»: representative inputs определяют то, какую часть распределения мы сохраним.

Min-max сохраняет крайние calibration values, но один редкий большой элемент растягивает сетку. Percentile исключает заданную tail долю и намеренно допускает clipping. MSE search выбирает range с минимальной reconstruction error на sample. Histogram/KL criteria сравнивают распределения при заданном binning и treatment пустых bins. Это разные objectives: минимальный tensor MSE не тождествен максимальной task correctness, а красивое совпадение гистограмм не доказывает сохранение редкого tool argument.

В коротком calibration report достаточно указать source mix, tokenization/template, число tokens/sequence lengths, какие узлы наблюдались, axis/group size и выбранный критерий. Числа 128–512 или 128–1024 examples из S4 — примеры experimental budgets, не установленный минимум для любой модели. Сто коротких строк и сто длинных документов дают разное количество activation samples; одинаковый sample count не доказывает одинаковое coverage.

Статистику активаций **можно оценивать заранее**, хотя нельзя заранее перечислить точные X всех будущих запросов. Поэтому формулировка S4 «активации нельзя проанализировать offline» чрезмерна. При shift на другой язык или длинные context windows static scales могут устареть. Dynamic scale адаптируется к текущему magnitude, но не решает все ошибки quantization и добавляет runtime reduction/metadata work.

Диапазоны вроде «MLP до±1000», «логиты±10» и сюжет о росте значений в средних слоях в S4 не являются контрактом Transformer. Распределение зависит от architecture, trained weights, узла измерения и inputs. Оно не обязано быть Gaussian. LayerNorm контролирует агрегированную статистику по выбранной оси, но не делает каждый feature одинаковым и не запрещает отдельным компонентам быть заметно больше остальных. Аналогия «weights — память, activations — мысли» помогает различать сроки жизни, но сама не объясняет причину outliers. Предложенные в чате причины про специализированные features и routing оставляем как интуиции, а не доказанный универсальный механизм.

### Три стратегии работы с outliers из исходного обсуждения

В [LLM.int8()](https://arxiv.org/abs/2208.07339) важна декомпозиция по feature dimensions: сумма вкладов обычных dimensions считается low-bit, а выделенных outlier dimensions — в higher precision. Алгебраически $XW=X_{:,R}W_{R,:}+X_{:,O}W_{O,:}$. Каждое слагаемое относится к соответствующим **входным** dimensions слоя. Это не две полные копии matmul и не схема «маленькую сумму просто прибавили к большим весам».

Какие dimensions попадают в O, зависит от правила detection/threshold и текущих activations; фраза S4 «определить однажды статически» не является обязательным свойством метода. Наблюдение статьи о систематических outliers объясняет пользу такого разделения, но не задаёт для всех architectures порог размера 6.7B, ровно 0.1% channels или одинаковые номера features. Также доля values в INT8 не равна гарантированно такой же доле wall-clock time: сбор outlier path и launches тоже стоят времени.

[SmoothQuant](https://arxiv.org/abs/2211.10438) использует точное reparameterization для row inputs: $XW=(XS^{-1})(SW)$, где S — обратимая diagonal matrix по input channels. До rounding output не меняется. Если activation channel большой, выбранный s_j уменьшает его magnitude, одновременно увеличивая соответствующую строку W. Это переносит часть quantization difficulty между двумя операндами, а не удаляет outlier knowledge.

В примере X=(100,1),W=(0.01,1)ᵀ output=2. При $S=\operatorname{diag}(10,1)$ получаем X'=(10,1),W'=(0.1,1)ᵀ и тот же output2. Теперь activation range менее перекошен. Но если ещё сильнее нарастить S, труднее станет квантовать weights. Выбор strength балансирует обе стороны; division by zero для пустых channels требует отдельной обработки. Folding scales в соседний normalization/linear должен сохранять все использующие тот же tensor branches: нельзя произвольно изменить одну ветку residual graph и забыть остальные.

Третья стратегия — изменить quantization granularity: per-token, per-channel или blocks. Она часто сочетается с предыдущими. «Per-token+per-channel» не обязательно означает отдельный float scale для каждой scalar activation: возможны factored scales или two-dimensional groups. Если же действительно хранить scale на каждый элемент, metadata могут уничтожить экономию. Стратегию описывают точным layout, а не только словом 2D.

## 4. PTQ и QAT

PTQ готовит quantized representation после основного training. Простая calibration/rounding и продвинутые error-compensation methods отличаются качеством/ценой. PTQ не означает «без вычислений»: могут потребоваться forward passes, statistics и оптимизация reconstruction.

QAT включает имитацию округления/clipping в training forward, чтобы параметры адаптировались к будущей ошибке. Fake quantization может вычислять dequantized values в float и не сжимает training memory автоматически до integer bitwidth. Поскольку round имеет нулевую производную почти везде и скачки на границах, часто используют straight-through estimator: surrogate backward, а не истинную derivative round.

На вопрос «почему сразу не обучать огромную модель в INT4» ответ: discretization ограничивает small updates, gradients/optimizer states нуждаются в иной точности/representation, а kernels и training stability зависят от метода. Low-bit training исследуется; из неудачи наивного integer SGD не следует невозможность всех специальных подходов. QAT также не гарантирует выигрыш при любом dataset/budget.

### Ваш вопрос «почему сразу не обучать в INT4»

Наивное обновление quantized weight и хранение float master weight с quantized forward — разные algorithms. Пусть grid step0.125, в памяти хранится только code для 0.125. Если каждое обновление пытается прибавить 0.005 и тут же снова округляет, результат остаётся 0.125. Десять таких попыток не накапливаются. Если же хранить float master w, десять increments меняют его на 0.05, и quantized forward в какой-то момент перескочит на следующий code. QAT может использовать именно такую возможность. Это поясняет исчезновение микрообновлений, не объявляя low-bit learning невозможным.

В S4 шаг для 16 равномерных уровней на обоих endpoints[−1,1] считался как 2/16. Если оба endpoint входят в сетку, между 16 уровнями 15 интервалов и шаг 2/15. Сетка с step0.125 тоже допустима, например 16 codes для[−1,0.875], но это уже другой range. Для symmetric−7…7 имеем 15levels и step1/7 при absmax1. Сначала определяют кодовую сетку, затем обсуждают, какие updates она различает.

Точная derivative round почти везде 0; STE заменяет её surrogate, часто 1 внутри выбранного range и 0 вне него. Это управляемое расхождение forward/backward, а не доказательство точной производной. Float master updates могут долго оставаться в одной quantization cell, а затем изменить code. Optimizer state, gradient precision, accumulator precision и activation dtype остаются отдельными решениями; QAT training memory не равна размеру финального INT4 artifact.

Adam second moment — среднее **квадратов** gradient, не автоматически центрированная variance. Утверждение S4 «оба states обязаны быть FP32, иначе мгновенный NaN» неверно как общий запрет: существуют специальные quantized optimizer schemes. Равным образом BF16 широкий exponent range не гарантирует стабильности всех calculations. Нужен конкретный training recipe. Для frozen QLoRA base вычисление производной по input требует dequantized W, но не обязательно STE через rounding самой базы: её codes не являются обучаемыми параметрами.

Низкобитное обучение изучается, однако результат нельзя перенести на произвольный pretrained checkpoint простым cast. [BitNet b1.58](https://arxiv.org/abs/2402.17764) использует ternary weights в quantized linear computations;1.58≈log₂3 описывает число состояний, а не обещает такой же размер всех training states. Работа сохраняет LLaMA-подобные RMSNorm, SwiGLU и rotary embeddings. Заявление S4, будто ради этого «убрали сложные нелинейности», не соответствует описанию. Ternary weight multiplication допускает специальные вычислительные приёмы, но остальной Transformer и training не становятся целиком одними сложениями.

При выборе PTQ/QAT полезнее сравнить нужное качество и полный budget, чем следовать таблице «PTQ=одна GPU/часы, QAT=полный dataset/недели». QAT fine-tuning может быть коротким, PTQ reconstruction — дорогим. Нет универсальной гарантии идеального INT8, полного разрушения INT4 или неизбежного превосходства QAT. Исследовательские названия Atom/QuIP из S4 нельзя использовать как доказательство W4A4 pretraining: конкретную работу проверяют по её objective и процедуре.

В исходной [QLoRA](https://arxiv.org/abs/2305.14314) сообщается о fine-tuning65B на одной 48GB GPU; реплика S4 о 65B на 24GB consumer GPU не переносится как доказанный результат. [Atom](https://arxiv.org/abs/2310.19102) посвящён low-bit serving и W4A4 evaluation, что само по себе не является обучением Transformer с нуля в INT4. В обоих случаях нужно читать, какая часть pipeline реально низкобитная.

## 5. GPTQ, AWQ и Hessian

GPTQ использует approximate second-order information для compensation при последовательной quantization весов слоя. В упрощённой layer reconstruction задаче с row inputs X:

$$
\mathcal{E}=\lVert XW-X\widehat W\rVert_F^2.
$$

Для одной output column Hessian квадратичной ошибки по weight vector пропорционален $X^\top X$ (точный коэффициент зависит от 1/2 и нормировки). Это не обязательно полный Hessian всей language-model CE по миллиардам параметров. По этой структуре оценивается, как компенсировать rounding error другими ещё не зафиксированными весами. [GPTQ](https://arxiv.org/abs/2210.17323).

**Jacobian** содержит первые derivatives векторной функции: для $f:\mathbb{R}^{d}\to\mathbb{R}^{m}$ форма $m\times d$. **Hessian** — вторые derivatives scalar function: $d\times d$. Для $f(u,v)=u^2+3uv+2v^2$ gradient $(2u+3v,3u+4v)$, Hessian $\begin{bmatrix}2 & 3\\3 & 4\end{bmatrix}$. Wronskian, о котором вспомнили в чате, связан с derivatives набора функций и их линейной независимостью; это другое понятие.

AWQ выбирает activation-aware channel scales, чтобы снизить чувствительную quantization error без обязательного хранения «важных весов» в отдельном high dtype. Для столбцов можно записать тождество $Wx=(WD)(D^{-1}x)$ с обратимой diagonal D. До quantization функция та же; после quantization выбор D меняет rounding/clipping. Дальше важно, можно ли scales эффективно слить в соседние операции. [AWQ](https://arxiv.org/abs/2306.00978).

Нельзя из статьи вывести вечное «AWQ быстрее GPTQ на всех GPU» или одинаковую runtime поддержку. Сравниваются concrete configs, hardware и quality constraint.

### Сначала различаем curvature всей модели и reconstruction слоя

Для scalar f local Taylor expansion имеет линейный term ∇fᵀδ и quadratic term ½δᵀHδ. Большая curvature вдоль direction означает сильное **изменение** local objective, но не всегда рост: Hessian может иметь negative eigenvalues. У приведённого выше f(u,v) determinant H равен−1, поэтому он indefinite. Одна большая diagonal entry также не описывает cross terms между coordinate changes.

В GPTQ удобнее начать с собственного layer objective, а не предполагать, что pretrained model находится в точном минимуме language-model CE на любом новом corpus. Пусть row-input matrix X имеет N×d, один weight column w — d×1. Для error $e=\widehat w-w$ имеем $\mathcal E=\tfrac12\|Xe\|^2=\tfrac12e^\top(X^\top X)e$. Поэтому $H=X^\top X$ — d×d и positive semidefinite: это covariance/Gram structure **calibration inputs**. В точке ŵ=w reconstruction error и её gradient действительно 0 по построению. Это гораздо точнее, чем ссылка на неизвестный global minimum всей модели.

### Один шаг компенсации GPTQ вручную

Предположим $H=\begin{bmatrix}2&1\\1&2\end{bmatrix}$, первый вес округляется с 0.4 до 0, значит δ1=−0.4. Если второй вес пока доступен для коррекции, минимум quadratic cost по нему удовлетворяет $H_{21}\delta_1+H_{22}\delta_2=0$. Отсюда δ2=+0.2. Cost без компенсации 0.16; с δ=(−0.4,0.2) cost0.12. Второй вес не просто получил весь «долг 0.4»: нужная компенсация определяется корреляциями inputs и curvature.

Для положительно определённого H и фиксированной coordinate i условная задача даёт:

$$
\delta^*=\frac{q(w_i)-w_i}{[H^{-1}]_{ii}}H^{-1}e_i,\qquad
\mathcal E(\delta^*)=\frac{(q(w_i)-w_i)^2}{2[H^{-1}]_{ii}}.
$$

Здесь e_i — unit vector, не reconstruction error. Для примера $H^{-1}=\begin{bmatrix}2/3&-1/3\\-1/3&2/3\end{bmatrix}$, первая column точно даёт(−0.4,0.2). После quantization следующего веса появится новая ошибка, а frozen coordinates уже не свободны. Practical GPTQ использует эффективную blocked/sequential организацию такой компенсации; это не построение dense inverse Hessian по всем параметрам LLM сразу. Сингулярные и плохо обусловленные Gram matrices требуют damping/численной обработки, а слабое calibration coverage оставляет неувиденные directions.

Аналогия S4 с суммой 1.4+2.2 полезна как первый мотив: обычное rounding даёт 1+2=3 вместо 3.6, перенос 0.4 приводит ко второму rounding2.6→3 и сумме 4, ближе к 3.6. Но для общего matmul важна не сумма весов, а Xδ. Пример с H показывает, как точное условие заменяет аналогию. Малый reconstruction error на calibration — свидетельство локальной сохранности, а не доказательство отсутствия downstream regressions.

### AWQ: масштабирование вместо «не трогать 1% весов»

Для column-input convention AWQ reparameterization Wx=(WD)(D⁻¹x) увеличивает выбранные weight columns и уменьшает соответствующие inputs. При row-input записи XW порядок обратный: (XS⁻¹)(SW). В S4 формула(X·s)(W/s) сопровождалась словесным описанием уменьшения X; знак scale direction нельзя угадать по названию метода. Сначала проверяют algebraic identity и ось, затем влияние rounding.

Возьмём один output, w=(0.13,0.77),x=(10,1). Exact y=2.07. Symmetric grid step0.77/7=0.11 даёт ŵ=(0.11,0.77),ŷ=1.87,error−0.20. Защитим первую direction scaling factor4: w'=(0.52,0.77),x'=(2.5,1). После quantization w'≈(0.55,0.77),ŷ=2.145,error+0.075. Те же nominal4bits, но другая сетка в effective coordinates дала меньшую output error. Не потребовалось хранить первый вес отдельно в FP16.

Scale может также ухудшить другие values в общей group, поэтому его выбирают с учётом reconstruction, а не бесконечно увеличивают каждый «важный» channel. AWQ salient directions связаны с measured activations; это не theorem «1% весов выполняют 99% работы». И нет гарантии, что одна и та же доля/scale пригодна для любого domain. GPTQ и AWQ сравнивают с одинаковыми model revision, calibration, group size, dtype остальных tensors и конечными quality/runtime constraints.

### Ваше уточнение о Jacobian, Hessian и Вронском

Для $F(u,v)=(u^2+v,uv)$ Jacobian $J=\begin{bmatrix}2u&1\\v&u\end{bmatrix}$. Его строки соответствуют двум outputs, столбцы —двум inputs; при (1,2) получаем$\begin{bmatrix}2&1\\2&1\end{bmatrix}$. У scalar loss f gradient —column первых производных, а Hessian равен Jacobian этого gradient при согласованной convention. В chain rule backward использует vector-Jacobian products, поэтому библиотеке обычно не нужно строить весь Jacobian явно.

Матрица Вронского для y1(t),y2(t) содержит строки(y1,y2) и(y1',y2'), а Wronskian — её determinant. Например, для exp(t),exp(2t) determinant=exp(3t)≠0, что показывает независимость этих функций. Нулевой Wronskian для произвольных smooth functions не всегда доказывает зависимость без дополнительных условий; стандартные теоремы для решений одного линейного ODE используют свою постановку. Так сохраняется связь с вашим вопросом, но Вронскиан не подменяет curvature matrix quantization objective.

## 6. Что означает «динамическая/online» квантизация

Dynamic activation quantization вычисляет scales по текущим inputs, например на вызове linear layer; это не обязательно изменение weight representation после каждого пользовательского токена. KV-cache quantization — ещё один независимый выбор. Adaptive bitwidth switching — специальная runtime policy, требующая собственных kernels/quality tests.

Если runtime при каждом запросе заново переквантует всю большую базу, overhead может превысить выгоду. Если динамически меняется только activation scale, overhead и смысл другие. На вопрос из чата сначала нужно уточнить **какой массив** и **когда** меняется.

### Какие именно параметры пересчитываются online

Static quantization использует scales, выбранные раньше по calibration. Dynamic activation quantization вычисляет их на текущем tensor/строке; можно хранить weights заранее low-bit и каждый вызов получать новые activation codes. Для двух inputs с ranges±1 и±50 динамические s=1/127 и 50/127 разумно различаются. Но static scale тоже может работать в выбранном workload: S4 утверждал, что без online W8A8 невозможен, хотя в SmoothQuant есть dynamic O1/O2 и static O3 variants. Fused implementation возможна, но latency «микросекунды всегда» не математическое свойство метода.

Новая activation scale не означает нового knowledge или дообучения weights. Requantization cache — отдельная операция над уже сохранёнными K/V. У неё есть стоимость чтения, преобразования и записи, новые scales и distortion; старые токены могут влиять на любое следующее attention. [KIVI](https://arxiv.org/abs/2402.02750) различает полезные axes quantization keys и values. Это конкретный метод, а не доказательство, что любой server автоматически переквантует старые INT8 entries в INT4 при нехватке памяти.

Adaptive bitwidth по запросу можно реализовать хранением нескольких artifacts, вложенным code representation или quantization из более точной базы. Но из одного INT4 файла нельзя восстановить утраченные исходные FP16 weights простым увеличением bitwidth: dequantize выдаёт приближение, записанное более широким dtype. Изменение scales тех же codes меняет саму функцию и требует либо компенсирующей transformation, либо новой проверки качества. Router по «сложности токена» — отдельная policy, а не стандартное следствие слова dynamic.

В S4 OmniQuant/QuIP# упоминались как runtime rescaling weights, BitNet —как adaptive bitwidth, StreamingLLM —как cache requantization. Такие attribution не подтверждаются одним названием. Например, [OmniQuant](https://arxiv.org/abs/2308.13137) учит параметры преобразований/clipping в post-training calibration; из этого не следует пересмотр scales weights после каждого пользовательского token. Eviction entries и снижение их precision также разные операции: первое удаляет историю, второе оставляет приближённую. Поддержку конкретной runtime функции проверяют отдельно от paper о related method.

### Память, скорость и качество: доводим расчёт до результата

Для P billion weights чистые decimal GB равны P·b/8. Возвращаем исходные примеры:7B требует 28/14/7/3.5GB при 32/16/8/4bits;13B —26/13/6.5GB при 16/8/4bits;70B —140/70/35GB. Для 9B в 16bits это 18GB, в 4bits —4.5GB до metadata. GB=10⁹bytes,GiB=2³⁰bytes:3.5GB≈3.26GiB. Различие единиц нельзя выдавать за сжатие или ошибку памяти runtime.

Дальше прибавляют scales/zero-points, оставленные float tensors, alignment, cache, activations и workspaces. Model weights могут помещаться в VRAM, а запуск с длинным context —уже нет. CPU offload дополнительно меняет transfers и latency. Выбор между большей quantized и меньшей float моделью требует одинаковой task evaluation: parameter count не определяет победителя при любом языке, instruction tuning и latency budget.

Weight-only inference может ускоряться даже при float matmul, если лимитирует чтение weights и unpacking fused с вычислением. И наоборот, W8A8 может проиграть на малом batch из-за conversions и kernel overhead. Для простого memory-bound decode условная нижняя оценка времени чтения weights равна bytes/bandwidth. При 14GB и 1TB/s это 14ms, при 3.5GB —3.5ms **только данного компонента**, без overhead/KV/compute. Если прочая работа занимает 10ms, idealized total меняется 24→13.5ms, то есть примерно 1.78×, а не 4×. Это иллюстрация ограниченного speedup, не benchmark реального GPU.

Для качества сохраняют float baseline, quantized variant и одинаковый evaluation protocol. В S3 приводились JSON success75%/95%, минимумQ5–Q6 и порог 95% для 100prompts без dataset и условий; эти числа не считаем измеренным результатом или универсальной нормой. Schema validity отделяют от правильности полей и tool success. Снижение logits MSE или небольшая PPL разница полезны для диагностики, но не заменяют эти проверки. Малое изменение top logits может поменять greedy token при почти равных alternatives и затем весь continuation.

Упражнение этой главы теперь можно выполнить полностью: вывести оба quantizer variants для исходных четырёх weights; пересчитать step и codes при activation outlier100; проверить metadata для g=32/128; восстановить один compensation step по H; затем сравнить two-coordinate AWQ example. Для практического запуска сначала выбирают один supported artifact и измеряют качество, peak memory, TTFT и decode throughput, явно задавая context/concurrency. Готового подтверждённого запуска пользователя в исходниках нет: численные примеры здесь учебные.

## 7. Проверка и упражнение

Оценка weight storage $\frac{Pb}{8}$ — нижняя грубая часть бюджета: для 7B, 4-bit это 3.5 GB до scales, unquantized tensors, padding и runtime. Speedup не следует из четырёхкратного уменьшения weight bytes: нужны подходящие kernels, а workload может ограничиваться prefill compute/очередью/KV.

Сравнить float baseline и одну quantized конфигурацию на одинаковых prompts: correctness, schema/tool correctness, peak memory, TTFT, tokens/s, p95 при concurrency. Нельзя назначать «для JSON минимум Q5/Q6» без конкретного эксперимента.

1. Почему маленький MSE весов не гарантирует маленький output error? **Ответ:** ошибка умножается на inputs; важны направления/статистика X.
2. Чем Hessian отличается от Jacobian? **Ответ:** вторые derivatives scalar objective против первых derivatives векторной функции.
3. Что меняет QAT backward через round? **Ответ:** обычно surrogate estimator; это не обычная точная derivative.

**Практика:** пересчитать affine example, затем добавить outlier 14 вместо 1.4 и взять один общий scale. Показать, как ухудшилась различимость малых чисел; сравнить две группы с отдельными scales. Учесть metadata, а не только MSE.

<!-- latex-display-v1 -->

<!-- source-depth-inference-А -->
