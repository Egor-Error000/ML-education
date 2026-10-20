# Углубление: SSL, MAE, DeiT и DINOv3

**Уровень:** углубление.

**Предпосылки:** [Vision Transformer: patches, attention и обучение](<cv-vit.md>)

**Используют:** Выбранная прикладная задача и итоговая проверка результата.

**Учебный вопрос:** Как получить и проверить представление без полного набора task labels?

Сравниваются цели получения признаков: frozen-feature baseline, masked reconstruction и teacher/student supervision. Предобучение и downstream оценка отделены; ни один из этих методов не является обязательной предпосылкой обычного detector.



## Карта вычисления и границы этапов

![Углубление: SSL, MAE, DeiT и DINOv3: входы, преобразования и выходы](<../оформление/course/cv-pretraining.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Сравните формы и смысл каждого перехода с формулами главы. Выход преобразования и доказательство его качества — разные объекты: последняя проверка использует отдельный протокол оценки.

<!-- unit:3c120c39500e69f01a -->
<a id="u-3c120c39500e69f01a"></a>

## SSL и проверка представлений

<!-- /unit:3c120c39500e69f01a -->

<!-- unit:b1bf255dc6ae9ad673 -->
<a id="u-b1bf255dc6ae9ad673"></a>

В связи с SSL-проектом из разбираемого примера полезно разделять objective и evaluation. Contrastive InfoNCE для query q, positive k+ и negatives k_j:

<!-- /unit:b1bf255dc6ae9ad673 -->

<!-- unit:a36dbd0bd54f0c2c94 -->
<a id="u-a36dbd0bd54f0c2c94"></a>

$$
\mathcal{L}_{\mathrm{InfoNCE}}=-\log\frac{\exp(\operatorname{sim}(q,k^+)/\tau)}{\sum_j\exp(\operatorname{sim}(q,k_j)/\tau)}.
$$

<!-- /unit:a36dbd0bd54f0c2c94 -->

<!-- unit:0b413aa4de3882beb6 -->
<a id="u-0b413aa4de3882beb6"></a>

В denominator включается positive и выбранные negatives. Аугментации определяют, какие изменения считаются сохраняющими identity. Для похожих классов false negatives могут давать нежелательный сигнал. Non-contrastive methods предотвращают representation collapse другими механизмами, поэтому отсутствие явных negatives само по себе не ошибка.

<!-- /unit:0b413aa4de3882beb6 -->

<!-- unit:4c36901fd1fe71ce5a -->
<a id="u-4c36901fd1fe71ce5a"></a>

Linear probe обучает только head на frozen features; full/partial fine-tuning меняет backbone. kNN — ещё один контроль геометрии. Сравнивать их на одном split, не считать высокий train score качеством representations. Collapse проявляется почти одинаковыми vectors для разных изображений; проверять variance/rank и downstream utility, а не только одну красивую проекцию.

<!-- /unit:4c36901fd1fe71ce5a -->

<!-- unit:1f51fcb304f16622f7 -->
<a id="u-1f51fcb304f16622f7"></a>

<!-- transformer-priority-cv-training-v1 -->

<!-- /unit:1f51fcb304f16622f7 -->

<!-- unit:fd6e6d6dcc843a30ab -->
<a id="u-fd6e6d6dcc843a30ab"></a>

## MAE: чему Transformer учится без классов изображений

<!-- /unit:fd6e6d6dcc843a30ab -->

<!-- unit:8d8ceb2108907c9c04 -->
<a id="u-8d8ceb2108907c9c04"></a>

В разбираемом примере MAE приведён как пример self-supervised pretraining. Его задача отличается и от классификации, и от авторегрессионного продолжения текста. Обучающая цель берётся из самого изображения: скрываем часть patches и восстанавливаем их пиксели. В базовом MAE тяжёлый encoder обрабатывает только видимые patches; более лёгкий decoder получает их представления и mask tokens на скрытых местах. Это асимметрия архитектуры [Masked Autoencoders](https://arxiv.org/abs/2111.06377).

<!-- /unit:8d8ceb2108907c9c04 -->

<!-- unit:fa0ae5f4cb3d90ffae -->
<a id="u-fa0ae5f4cb3d90ffae"></a>

![MAE: видимые patches идут в encoder, скрытые восстанавливает decoder](../archive/2026-10-20-before-rebuild/оформление/transformer/mae-visible-hidden.svg)

<!-- /unit:fa0ae5f4cb3d90ffae -->

<!-- unit:f85f837d2c087c0153 -->
<a id="u-f85f837d2c087c0153"></a>

### Маленький пример с точным восстановлением индексов

<!-- /unit:f85f837d2c087c0153 -->

<!-- unit:e7d362cf8c2aa4cb8d -->
<a id="u-e7d362cf8c2aa4cb8d"></a>

Пусть имеется 16 patches с индексами 0–15. Скрываем 75%, то есть 12, и оставляем четыре: 1, 4, 10, 15. На рисунке номера на единицу больше, чтобы привычно читать клетки 1–16. Encoder получает четыре вектора с информацией об их исходных позициях; он не должен считать, что после удаления они стали четырьмя соседними patches верхней строки.

<!-- /unit:e7d362cf8c2aa4cb8d -->

<!-- unit:f41b40ea231b1106c3 -->
<a id="u-f41b40ea231b1106c3"></a>

Упаковка может хранить видимые в порядке [10,1,15,4]. Этот порядок сам по себе допустим при согласованных позиционных признаках и восстановлении индексов. Decoder должен поставить выход для patch 10 именно на его место, для patch 1 — на его, а остальные 12 мест заполнить mask tokens. Простая операция «добавить 12 одинаковых токенов в конец» без обратной перестановки потеряла бы соответствие исходной сетке. Одинаковый начальный mask token различается по местам после добавления decoder position embeddings.

<!-- /unit:f41b40ea231b1106c3 -->

<!-- unit:5bf1e47f3efc00eebc -->
<a id="u-5bf1e47f3efc00eebc"></a>

Для изображения 224×224 и patch 16 имеется 196 patches. При 75% маскирования видимы 49; с CLS encoder обрабатывает 50 токенов вместо 197. Число attention pairs на голову — 2500 вместо 38 809, отношение около 15.52. Но decoder всё равно работает с восстановленной полной последовательностью, а patch projection и другие операции имеют свои затраты. Поэтому это не обещание ускорения полного обучения в 15.52 раза.

<!-- /unit:5bf1e47f3efc00eebc -->

<!-- unit:f37c0457b2bbdc724d -->
<a id="u-f37c0457b2bbdc724d"></a>

### Loss вычисляется по скрытым целям

<!-- /unit:f37c0457b2bbdc724d -->

<!-- unit:340068bf8b255b3710 -->
<a id="u-340068bf8b255b3710"></a>

Пусть $\mathcal M$ — множество скрытых patch indices, а $K=CP^2$ — число пиксельных компонент patch. Простой reconstruction loss:

<!-- /unit:340068bf8b255b3710 -->

<!-- unit:27fb535b43df9db0c9 -->
<a id="u-27fb535b43df9db0c9"></a>

$$
\mathcal L_{\rm MAE}=\frac{1}{|\mathcal M|K}\sum_{n\in\mathcal M}\sum_{c=1}^{K}(\hat x_{nc}-x_{nc})^2.
$$

<!-- /unit:27fb535b43df9db0c9 -->

<!-- unit:21543c80c1887236cf -->
<a id="u-21543c80c1887236cf"></a>

Здесь c обозначает компоненту распрямлённого patch, а не только RGB-канал. Возможна нормализация целевых пикселей внутри patch; тогда цель в формуле меняется соответственно. В базовой процедуре средняя ошибка учитывает скрытые patches. Если начислять основную ошибку за копирование видимых patches, можно облегчить задачу без нужного восстановления отсутствующей информации. Формы gather, обратной перестановки и маскированного MSE показаны в [коде авторов MAE](https://github.com/facebookresearch/mae/blob/main/models_mae.py).

<!-- /unit:21543c80c1887236cf -->

<!-- unit:1adea48818ab467f29 -->
<a id="u-1adea48818ab467f29"></a>

Рассмотрим учебный случай двух скрытых patches по два числа. Цели $(1,3)$ и $(2,4)$, предсказания $(0,5)$ и $(3,3)$. Квадраты ошибок равны $(1,4)$ и $(1,1)$, среднее по четырём компонентам — 7/4=1.75. Градиенты по четырём предсказаниям равны $2(\hat x-x)/4$, то есть $(-0.5,1,0.5,-0.5)$. Ошибка на видимом patch не входит в это среднее. Но его **представление** получает градиент через decoder: видимые токены используются, чтобы восстановить скрытые.

<!-- /unit:1adea48818ab467f29 -->

<!-- unit:114e7b6a8c8c221f83 -->
<a id="u-114e7b6a8c8c221f83"></a>

Модель может предлагать правдоподобную, но не совпадающую с исходником деталь: одних видимых областей не всегда достаточно для однозначного восстановления. Reconstruction MSE — не вероятность класса объекта и не доказательство истинности дорисованной части. Предобучение оценивают по полезности переноса представлений, например через fine-tuning или linear probe на отложенной задаче, а не только по визуальной привлекательности реконструкций.

<!-- /unit:114e7b6a8c8c221f83 -->

<!-- unit:42e3da7233f9d9f7ce -->
<a id="u-42e3da7233f9d9f7ce"></a>

### Почему это не то же самое, что BERT MLM

<!-- /unit:42e3da7233f9d9f7ce -->

<!-- unit:eb569fd9c481b83859 -->
<a id="u-eb569fd9c481b83859"></a>

В обоих подходах задача строится из скрытия частей входа, но единицы и выходы различаются. В текстовом MLM предсказываются дискретные ID словаря; здесь — пиксельные компоненты patches. У базового MAE скрытые patches удалены из encoder-последовательности, а decoder возвращает места. Типичный BERT сохраняет последовательность с изменёнными входными токенами. Нельзя автоматически перенести BERT-правило подготовки MLM-входа в MAE и ожидать ту же архитектуру и цену.

<!-- /unit:eb569fd9c481b83859 -->

<!-- unit:f672fad427dc45e40e -->
<a id="u-f672fad427dc45e40e"></a>

Для downstream классификации reconstruction decoder обычно отбрасывают и используют encoder с новой головой. Все patches нового изображения становятся доступны encoder; случайное удаление 75% было обучающей задачей предобучения, а не обязательным способом обычной классификации. Наличие decoder в MAE не делает его авторегрессионным языковым decoder: здесь название обозначает восстановление выходов из представления, без обязательной causal маски или генерации patches по одному.

<!-- /unit:f672fad427dc45e40e -->

<!-- unit:9f80f148430f5d95e9 -->
<a id="u-9f80f148430f5d95e9"></a>

<!-- transformer-deit-training-v1 -->

<!-- /unit:9f80f148430f5d95e9 -->

<!-- unit:04bb7ad539245a150b -->
<a id="u-04bb7ad539245a150b"></a>

## DeiT: как teacher помогает обучать Vision Transformer

<!-- /unit:04bb7ad539245a150b -->

<!-- unit:a7cc96367cf64a8689 -->
<a id="u-a7cc96367cf64a8689"></a>

В разбираемом примере DeiT упомянут одной строкой как distillation для обучения ViT на ImageNet-1k. Здесь разберём механизм. Само семейство DeiT включает и модели без дополнительного distillation token; ниже рассматривается **distilled** вариант. Главная идея: student получает обычную метку изображения и дополнительный сигнал teacher через отдельную обучаемую позицию. В исходной работе teacher мог быть CNN; одинаковое внутреннее устройство teacher и student не требуется. Отправная точка — [DeiT](https://arxiv.org/abs/2012.12877).

<!-- /unit:a7cc96367cf64a8689 -->

<!-- unit:97554e3cfa1f6f84fe -->
<a id="u-97554e3cfa1f6f84fe"></a>

![Teacher target и истинная метка обучают разные головы student; их gradients встречаются в общих блоках.](../archive/2026-10-20-before-rebuild/оформление/transformer/deit-teacher-student.svg)

<!-- /unit:97554e3cfa1f6f84fe -->

<!-- unit:866ae86a595ad98b21 -->
<a id="u-866ae86a595ad98b21"></a>

Teacher target и истинная метка обучают разные головы student; их gradients встречаются в общих блоках.

<!-- /unit:866ae86a595ad98b21 -->

<!-- unit:5ec6f7d598b28c64b8 -->
<a id="u-5ec6f7d598b28c64b8"></a>

### Что добавляется к последовательности patches

<!-- /unit:5ec6f7d598b28c64b8 -->

<!-- unit:8a3475127be68a21d7 -->
<a id="u-8a3475127be68a21d7"></a>

Для картинки 224×224 с patch 16 имеется 196 patch tokens. Обычная CLS-схема даёт 197 позиций, distilled схема — 198: CLS, DIST и patches. DIST начинается как общий обучаемый вектор и получает собственный positional embedding. После прохождения encoder его состояние зависит от картинки через self-attention. CLS и DIST могут взаимодействовать и друг с другом, и с patches; треугольная causal mask для обычной классификации не нужна.

<!-- /unit:8a3475127be68a21d7 -->

<!-- unit:f835497a687cef282a -->
<a id="u-f835497a687cef282a"></a>

У итогового CLS своя classification head, у DIST — отдельная distillation head. Обе выдают C logits по одному и тому же набору классов. Совпадение label order обязательно: если teacher class ID 7 означает другое, чем student ID 7, supervision будет неверным при совершенно правильных shapes. Внутренние feature widths, число слоёв и тип backbone teacher могут отличаться, поскольку сравниваются ответы по классам.

<!-- /unit:f835497a687cef282a -->

<!-- unit:2f10edf7bc18f98a0c -->
<a id="u-2f10edf7bc18f98a0c"></a>

В этой учебной конфигурации число attention pairs на голову меняется с $197^2=38809$ на $198^2=39204$, прирост 395, около 1.02%. Это цена дополнительной позиции в одном attention, а не полный overhead distillation training: teacher тоже делает forward. При D=768 и C=1000 отдельные DIST-вектор, дополнительный position-вектор и head с bias дают $2D+DC+C=770536$ новых параметров относительно выбранной CLS-схемы. Не все эти веса используются одинаково в другой конфигурации checkpoint.

<!-- /unit:2f10edf7bc18f98a0c -->

<!-- unit:cf414899787f39a2dd -->
<a id="u-cf414899787f39a2dd"></a>

### Hard distillation: две цели и две головы

<!-- /unit:cf414899787f39a2dd -->

<!-- unit:1739a6a92caa582578 -->
<a id="u-1739a6a92caa582578"></a>

Пусть y — правильная метка картинки, teacher logits — t, а его выбранный класс $y_T=\arg\max_c t_c$. Упрощённый loss для одного примера:

<!-- /unit:1739a6a92caa582578 -->

<!-- unit:069d6b746f95fb0073 -->
<a id="u-069d6b746f95fb0073"></a>

$$
\mathcal L=(1-\alpha)\operatorname{CE}(z_{\rm CLS},y)+\alpha\operatorname{CE}(z_{\rm DIST},y_T).
$$

<!-- /unit:069d6b746f95fb0073 -->

<!-- unit:52fa4a2ca4a84b6d9c -->
<a id="u-52fa4a2ca4a84b6d9c"></a>

В base criterion могут входить soft labels после mixup или label smoothing; здесь для ясности используются обычные hard labels. Teacher output рассматривается как фиксированная цель текущего шага, без backward в teacher. Это не самообучение, где обе сети совместно свободно изменяются от одной ошибки. Детали выбора hard/soft loss есть в [коде DeiT](https://github.com/facebookresearch/deit/blob/main/losses.py).

<!-- /unit:52fa4a2ca4a84b6d9c -->

<!-- unit:c2454135594eff22d5 -->
<a id="u-c2454135594eff22d5"></a>

Учебный пример: правильный первый класс, teacher выбрал второй. CLS probabilities student равны (0.6,0.3,0.1), DIST probabilities — (0.2,0.7,0.1), α=0.5. Тогда loss равна $(-\log0.6-\log0.7)/2\approx0.433750$. Градиент CLS logits равен $0.5(-0.4,0.3,0.1)$, DIST logits — $0.5(0.2,-0.3,0.1)$. Головы получают разные задачи; их градиенты далее проходят в общий encoder и складываются на общих параметрах.

<!-- /unit:c2454135594eff22d5 -->

<!-- unit:aa66ed50ea83b5ec32 -->
<a id="u-aa66ed50ea83b5ec32"></a>

Teacher может ошибаться относительно y. Наличие отдельной головы позволяет представить оба сигнала, но не гарантирует, что конфликт всегда улучшит student. Важно качество teacher, совпадение классов и соответствие данных. Distillation передаёт поведение teacher, включая его ошибки, а не предоставляет дополнительную безошибочную разметку.

<!-- /unit:aa66ed50ea83b5ec32 -->

<!-- unit:a5deebfb792a3a9a18 -->
<a id="u-a5deebfb792a3a9a18"></a>

### Soft distillation: передаём распределение, а не один argmax

<!-- /unit:a5deebfb792a3a9a18 -->

<!-- unit:cdeda789163349a807 -->
<a id="u-cdeda789163349a807"></a>

Вместо hard класса можно сравнивать распределения. Пусть $q_T=\operatorname{softmax}(t/\tau)$ и $p_T=\operatorname{softmax}(z_{\rm DIST}/\tau)$. Один распространённый вариант:

<!-- /unit:cdeda789163349a807 -->

<!-- unit:c127c7afba1e1930f5 -->
<a id="u-c127c7afba1e1930f5"></a>

$$
\mathcal L_{\rm KD}=\tau^2\sum_c q_{T,c}\log\frac{q_{T,c}}{p_{T,c}}.
$$

<!-- /unit:c127c7afba1e1930f5 -->

<!-- unit:5d46e14763c86f6938 -->
<a id="u-5d46e14763c86f6938"></a>

Это KL teacher→student, то есть target distribution фиксирована, student подгоняется под неё. Для одного примера derivative по student logits равна $\tau(p_T-q_T)$ до дополнительных коэффициентов усреднения и α. Множитель τ² компенсирует часть изменения масштаба производных при softened logits; он не делает каждый training recipe полностью инвариантным к temperature.

<!-- /unit:5d46e14763c86f6938 -->

<!-- unit:b5dc850a328b9f5f28 -->
<a id="u-b5dc850a328b9f5f28"></a>

При τ=1, q=(0.6,0.3,0.1), p=(0.4,0.4,0.2) gradient равен (−0.2,0.1,0.1). В отличие от hard target второго класса, soft target сообщает также относительную вероятность остальных классов. Это полезный вид supervision, если teacher probabilities несут нужную информацию; высокая temperature сама по себе не создаёт её.

<!-- /unit:b5dc850a328b9f5f28 -->

<!-- unit:f0799632ad309f087f -->
<a id="u-f0799632ad309f087f"></a>

Нормировка должна быть явной. В формуле выше для batch обычно отдельно усредняют loss по примерам. В опубликованном `losses.py` soft branch использует сумму KL, умноженную на τ² и делённую на число элементов student logits B×C; это добавляет фактор 1/C относительно среднего по batch. Поэтому нельзя переносить α из одного соглашения в другое, считая scale loss прежним. Hard branch в том же коде использует CE по teacher argmax.

<!-- /unit:f0799632ad309f087f -->

<!-- unit:32f4a3b82fb723d76c -->
<a id="u-32f4a3b82fb723d76c"></a>

### Как loss на DIST достигает patches

<!-- /unit:32f4a3b82fb723d76c -->

<!-- unit:d18b9f94a856d8acbc -->
<a id="u-d18b9f94a856d8acbc"></a>

DIST участвует в тех же encoder-блоках, что и patches. Его итоговое состояние содержит mixtures V, полученные по Q/K attention. Ошибка distillation head меняет эти weights и values, а через предыдущие слои — patch projection, positions и общие MLP. Не нужно помечать каждый patch teacher class отдельно. Как и при CLS loss, supervision одной позиции может обучать источники её контекста.

<!-- /unit:d18b9f94a856d8acbc -->

<!-- unit:64b68771f4dd4a3ad9 -->
<a id="u-64b68771f4dd4a3ad9"></a>

Teacher CNN не копирует свои convolution filters в attention student. Передаваемый в этом варианте сигнал — class logits или argmax. Student должен научиться получать подходящий ответ своими patch embeddings, attention и MLP. Отдельная feature-level distillation была бы другой процедурой, требующей согласования внутренних пространств.

<!-- /unit:64b68771f4dd4a3ad9 -->

<!-- unit:4ac1ab614751350c5c -->
<a id="u-4ac1ab614751350c5c"></a>

Учебный input teacher должен соответствовать тому же изображению и преобразованиям, на которых считается student target. Различные требования к нормировке teacher/student можно согласовать preprocessing wrappers, сохранив смысл входа. Кроме отключения autograd важно задать нужный eval-режим teacher: иначе dropout или обновление BatchNorm statistics способны менять targets. Заморозить gradients и переключить train/eval — разные действия.

<!-- /unit:4ac1ab614751350c5c -->

<!-- unit:e614ce182bcfe5f54a -->
<a id="u-e614ce182bcfe5f54a"></a>

### Что происходит при inference

<!-- /unit:e614ce182bcfe5f54a -->

<!-- unit:8a77ce199a2c061cb3 -->
<a id="u-8a77ce199a2c061cb3"></a>

Teacher нужен для обучения, но distilled student способен работать самостоятельно. В опубликованном [forward модели](https://github.com/facebookresearch/deit/blob/main/models.py) training возвращает logits двух heads, а eval — их среднее. Это **среднее logits**, хотя комментарий использует общее слово predictions. Конкретную реализацию checkpoint нужно читать точно.

<!-- /unit:8a77ce199a2c061cb3 -->

<!-- unit:4d472cca9eacbb6a03 -->
<a id="u-4d472cca9eacbb6a03"></a>

Среднее logits и среднее probabilities различаются. Для двух бинарных heads с probabilities (0.9,0.1) и (0.6,0.4) арифметическое среднее probabilities равно (0.75,0.25). Если взять logits как log probabilities, усреднить их и затем применить softmax, получим нормированные $\bigl(\sqrt{0.9\cdot0.6},\sqrt{0.1\cdot0.4}\bigr)$ — другое распределение. Аддитивные общие сдвиги logits каждой головы на этот итоговый softmax не влияют.

<!-- /unit:4d472cca9eacbb6a03 -->

<!-- unit:cb0ab0ea971822d10d -->
<a id="u-cb0ab0ea971822d10d"></a>

### Как отделить эффект distillation от остального обучения

<!-- /unit:cb0ab0ea971822d10d -->

<!-- unit:da3a0c126697419de1 -->
<a id="u-da3a0c126697419de1"></a>

Сравнение должно фиксировать student architecture, данные, число training updates, аугментации и evaluation. Иначе выигрыш может происходить от нового training recipe, большего compute или другого preprocessing. Дополнительно учитывается offline/training стоимость teacher; отсутствие teacher при inference не означает, что его обучение было бесплатным.

<!-- /unit:da3a0c126697419de1 -->

<!-- unit:3d4ab2b44b959e1a87 -->
<a id="u-3d4ab2b44b959e1a87"></a>

Утверждения «ViT на маленьких данных проигрывает CNN» и «на больших всегда лучше масштабируется» следует читать как мотивацию сравнения inductive biases и recipes, а не универсальные законы. Предобученный ViT на маленькой downstream задаче, ViT с нуля и CNN с сильным предобучением — разные условия. DeiT показывает конкретный способ сделать обучение Transformer более эффективным по данным; он не отменяет измерения на вашей задаче.

<!-- /unit:3d4ab2b44b959e1a87 -->

<!-- unit:f20519bb68370b7f62 -->
<a id="u-f20519bb68370b7f62"></a>

## Современный case: frozen dense features DINOv3

<!-- /unit:f20519bb68370b7f62 -->

<!-- unit:ccf0987030590adc92 -->
<a id="u-ccf0987030590adc92"></a>

Результат классификации через global pooled vector не полностью описывает качество spatial features. DINOv3 — self-supervised family, чья техническая работа сообщает об улучшении dense features и downstream результатов без fine-tuning на выбранных benchmarks. Для своей задачи это гипотеза о подходящем frozen backbone, а не готовый вывод о превосходстве.

<!-- /unit:ccf0987030590adc92 -->

<!-- unit:59ea771b09ef38450b -->
<a id="u-59ea771b09ef38450b"></a>

Сравните на одном split: (1) frozen features + linear probe, (2) spatial features с небольшим task head, (3) fine-tuning при сопоставимом compute budget. Для retrieval дополните top-k nearest-neighbor audit; для segmentation — spatial task metric; затем оцените отдельный domain/time slice. Проверьте preprocessing, разрешение, выходной слой/feature map и checkpoint revision: одинаковая надпись «DINOv3» не гарантирует одинаковый интерфейс.

<!-- /unit:59ea771b09ef38450b -->

<!-- unit:9ad88c32082786e3de -->
<a id="u-9ad88c32082786e3de"></a>

**Источник и границы:** [DINOv3 technical report](https://arxiv.org/abs/2508.10104) от 13 августа 2025 описывает scaling, data preparation и Gram anchoring для сохранения качества dense features. Benchmark claims принадлежат авторам и относятся к указанным task/dataset/protocol. В этой главе они служат примером выбора baseline, а не доказательством, что pretrained feature всегда лучше task-specific training.

<!-- /unit:9ad88c32082786e3de -->

## Архитектуры: подробные схемы

### MAE: видимый вход и скрытая цель

![MAE: видимый вход и скрытая цель](<../оформление/course/arch-mae.svg>)

Только visible tokens проходят encoder. Decoder получает latent и placeholders с восстановленными позициями. Loss сопоставляет выбранные outputs с исходными masked pixels; исходные targets не подаются encoder как видимые входы.

### DeiT: label и teacher supervision

![DeiT: label и teacher supervision](<../оформление/course/arch-deit.svg>)

Обе головы читают отдельные token states одного encoder. Ground-truth class и teacher signal задают разные слагаемые loss. Teacher prediction не заменяет истинную разметку; правило применения голов на inference фиксируется recipe.

### DINOv3: применение frozen features

![DINOv3: применение frozen features](<../оформление/course/arch-dinov3.svg>)

Схема показывает downstream использование DINOv3: backbone фиксирован, readout и данные задачи учитываются отдельно. Dense features не являются готовыми boxes или masks; преобразование пространственной сетки и голова задачи остаются необходимыми.

<!-- unit:8067a16d1433a6a806 -->
<a id="u-8067a16d1433a6a806"></a>

5. Зачем Swin сдвигает границы окон? **Ответ:** чтобы следующие локальные группы пересекали прежние и создавали пути обмена между ними. MLP сам по себе таких путей не создаёт.
6. Равнозначны ли patch merging и уменьшение числа attention heads? **Ответ:** нет. Первое уменьшает пространственную сетку и преобразует ширину; второе меняет организацию attention при своей ширине головы.
7. Почему видимые patches MAE получают градиент, хотя loss считается по скрытым? **Ответ:** их представления используются decoder для восстановления скрытых выходов, поэтому chain rule возвращает ошибку через эти зависимости.
8. Нужно ли при классификации после MAE всегда скрывать 75% patches? **Ответ:** нет. Это задача предобучения; обычный downstream encoder может получать все patches, а reconstruction decoder отбрасывается.

<!-- /unit:8067a16d1433a6a806 -->

<!-- unit:4329faa02c86adb034 -->
<a id="u-4329faa02c86adb034"></a>

9. Копирует ли DeiT convolution filters CNN teacher в ViT student? **Ответ:** в разобранной схеме нет. Сигнал teacher — class logits/argmax; student получает его через свою distillation head и общие attention-блоки.
10. Почему teacher не нужен при inference student? **Ответ:** после обучения student сам выдаёт logits двух heads; teacher был источником targets во время training.
11. Равно ли среднее logits среднему probabilities? **Ответ:** нет. Softmax нелинеен; опубликованный distilled forward усредняет logits, что нужно сохранить при воспроизведении.
12. Почему α зависит от нормировки KD loss? **Ответ:** деление KL на B и на B×C меняет её масштаб относительно base loss, хотя название objective остаётся тем же.

<!-- /unit:4329faa02c86adb034 -->

## Самостоятельная практика

**Практика:** Сравните frozen representation + одинаковый readout для двух encoder. Для MAE отдельно выпишите visible indices, masked targets и loss mask; для distillation — teacher signal и task labels. Объясните, почему увеличение linear-probe score не доказывает улучшение всех dense tasks.

## Результат главы и переход дальше

Ответьте на вопрос: **Как получить и проверить представление без полного набора task labels?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

[Маршрут модуля](<README.md>) · [Происхождение и точные места переноса](<../editorial/coverage.md>)
