# Label smoothing, MixUp, CutMix и аугментации

**Уровень:** общее ядро.

**Предпосылки:** [Обобщение, диагностика и границы теории](<reg-diagnosis.md>)

**Используют:** Завершение выбранной ветки: самостоятельный эксперимент по практике этой главы.

**Учебный вопрос:** Как изменения inputs и targets задают предположения о задаче?

## Карта вычисления и границы этапов

![Label smoothing, MixUp, CutMix и аугментации: входы, преобразования и выходы](<../оформление/course/reg-augmentation.svg>)

Аугментация кодирует предположение о допустимом изменении объекта. Если преобразование меняет смысл метки, обучение закрепляет ошибочную инвариантность.

## Предположения аугментаций

Обучение с transforms минимизирует риск по выбранному распределению $T$:

$$
\mathcal L(\theta)=
\mathbb E_{(x,y),T}[\ell(f_\theta(Tx),T_y(y))].
$$

Для классификации $T_y$ иногда identity, для detection/segmentation — согласованное геометрическое преобразование. Label-preserving transformations должны быть обоснованы задачей. Усиление augmentation меняет objective (оптимизируемая целевая функция) и difficulty, поэтому при сравнении фиксируются steps, sampling (случайный выбор по распределению) и остальные настройки.

## Label (метка) smoothing и soft targets (мягкие целевые распределения)

Одна convention смешивает hard label с uniform по всем $K$ classes:

$$
q_k=(1-\epsilon)\mathbf1[k=y]+\frac\epsilon K,
\qquad
L=-\sum_{k=1}^K q_k\log p_k.
$$

При $K=3,\epsilon=0.1$ target для class 1 равен $(0.9333\ldots,0.0333\ldots,0.0333\ldots)$. Другая convention оставляет true class $1-\epsilon$ и делит $\epsilon$ только между остальными: $(0.9,0.05,0.05)$. Одинаковое epsilon не делает их одинаковыми.

Для softmax logits (логиты, оценки альтернатив до нормировки) gradient (градиент) равен $\partial L/\partial z_k=p_k-q_k$. Поэтому smoothing ослабляет стимул бесконечно увеличивать true logit относительно остальных. Это может уменьшить overconfidence, но не гарантирует лучшую calibration или ranking в каждой задаче. Неправильно подобранный smoothing способен мешать learning редких/сложных classes.

Uniform prior тоже не всегда соответствует task structure: соседние classes и заведомо невозможные alternatives различаются. Более сложные soft targets должны иметь собственное обоснование и independent validation (проверочная часть для выбора решения).

### Вывод gradient cross-entropy

Для logits z и soft target q с суммой 1 запишем $L=-\sum_k q_kz_k+\log\sum_j e^{z_j}$. Первая часть дифференцируется в −q, вторая — в softmax p, поэтому gradient равен p−q. Это относится и к hard one-hot, и к MixUp, и к smoothed labels (метки). Если target не нормирован, коэффициент перед log-sum-exp меняется; проверка суммы q является частью data contract.

Пусть K=3, истинный класс первый, p=(0.98,0.01,0.01). Для hard target gradient равен (−0.02,0.01,0.01): gradient descent продолжает увеличивать true logit. При smoothing epsilon=0.1 с uniform по всем классам q=(28/30,1/30,1/30), gradient становится примерно (0.04667,−0.02333,−0.02333). Теперь true logit уменьшается: модель уже увереннее, чем требует изменённый target.

В идеальной свободно настраиваемой модели минимум CE для одного soft target достигается при p=q. Для указанного target отношение true probability к каждой другой равно 28, поэтому нужная разность logits — $\log28\approx3.3322$. Для альтернативного соглашения q=(0.9,0.05,0.05) это $\log18\approx2.8904$. Одинаковое название smoothing и одинаковое epsilon задают разные оптимумы. При hard target без других ограничений улучшение CE может продолжаться с ростом margin к бесконечности.

Минимальная CE soft target равна его entropy и обычно положительна. Поэтому ненулевой train loss (функция потерь) не обязательно означает, что модель не смогла выучить такие данные. Сравнивать raw train CE экспериментов с разными targets (целевые значения) без оговорки тоже нельзя. Мотивация label smoothing представлена в [Rethinking the Inception Architecture](https://arxiv.org/abs/1512.00567).

### Что означает uniform-компонента

Обозначим uniform distribution u. Тогда smoothing loss равен $(1-\epsilon)CE(y,p)+\epsilon CE(u,p)$, а $CE(u,p)=H(u)+KL(u\Vert p)$. Добавка предпочитает не делать альтернативные probabilities слишком малыми. Это не тот же objective, что entropy bonus модели: $-H(p)$ связан с $KL(p\Vert u)$, где направление KL противоположно.

Если истинная conditional probability равна p*, средний smoothed target в идеальном случае равен $(1-\epsilon)p^*+\epsilon u$. Именно его и выучивает свободная модель при таком objective. Например, для бинарного p*=0.9 и epsilon=0.1 получаем 0.86. Значит, label smoothing намеренно меняет вероятностную цель. Оно может компенсировать переуверенность конкретной конечной модели, но не гарантирует калибровку для исходной population probability.

Uniform по vocabulary, uniform по допустимым actions и uniform по классам отдельной задачи — разные distributions. Для NLP надо решить, включены ли special tokens (токены) и как обрабатываются padding (дополнение до общей длины)/ignored positions. Для классов с известной близостью можно рассматривать другой prior, но его правильность и coefficient требуют независимой проверки. «Мягкая» метка не становится достовернее только из-за отсутствия точных нулей.

## MixUp в inputs и representations

MixUp строит:

$$
\widetilde x=\lambda x_i+(1-\lambda)x_j,
\qquad
\widetilde y=\lambda y_i+(1-\lambda)y_j,
\qquad
\lambda\sim\operatorname{Beta}(\alpha,\alpha).
$$

При classes A/B и $\lambda=0.3$ target содержит 0.3 для A и 0.7 для B. Cross-entropy с таким target равен weighted sum двух class losses. Смешивать class IDs численно нельзя: label 2 и label 6 не дают semantic class 4. Смешиваются one-hot/distributions.

Linear interpolation задаёт prior о поведении между examples. Не все mixed inputs естественны; польза метода эмпирическая и task-dependent. Для unrelated tasks или incompatible modalities механическая смесь может не иметь смысла.

Manifold Mixup смешивает hidden representations на выбранном layer, затем продолжает forward (прямой проход вычисления) с mixed target. Простое смешивание двух decoded слов или integer tokens не эквивалентно этому. При frozen encoder (кодировщик входа) нужно понять, на каком уровне trainable model может воспользоваться смешанным representation. Batch (пакет примеров) pairing, distribution lambda и выбранные layers являются частью experiment config.

### Один forward и две составляющие loss

Учебные входы $x_i=(0,1)$ и $x_j=(1,0)$ при lambda=0.3 дают mixed input (0.7,0.3). Если их классы A и B, target для трёх классов равен (0.3,0.7,0). Пусть prediction на mixed input p=(0.4,0.5,0.1). Тогда

$$
L=-0.3\log0.4-0.7\log0.5\approx0.76009,
\qquad \nabla_zL=(0.1,-0.2,0.1).
$$

Это равно $0.3CE(A,p)+0.7CE(B,p)$ для одного и того же p на mixed input. Оно не равно автоматически weighted losses двух отдельных forwards на исходных xi и xj. Функция сети нелинейна, поэтому её response на смесь является дополнительным требованием. MixUp обучает предпочтению относительно простого поведения между выбранными примерами, как сформулировано в [исходной работе](https://arxiv.org/abs/1710.09412).

Численно усреднять nominal class IDs нельзя. Пример IDs 2 и 6 не создаёт «истинный класс 4»: арифметика относится к vectors distributions. Входы также должны иметь совместимые shape и representation. Интерполяция двух пиксельных изображений не обязательно является фотографией реального объекта; метод использует обучающий prior, а не доказанную физическую модель композиции.

### Как alpha управляет силой смешивания

Для $\lambda\sim Beta(\alpha,\alpha)$ среднее равно 0.5, variance равна $1/[4(2\alpha+1)]$. При alpha<1 плотность больше у краёв, поэтому часто доминирует один исходник; при alpha=1 распределение равномерно; при большом alpha коэффициенты концентрируются около 0.5. Например, variance для alpha=1 равна 1/12, а для alpha=10 — 1/84.

Сохранение одного и того же среднего lambda не означает одинаковой силы регуляризации. В одной реализации lambda выбирают на весь batch, в другой на каждую пару; pairing может допускать совпадение с самим собой или ограничиваться группами. Эти решения задают разное распределение обучающих задач. При class imbalance случайное pairing и sampling также меняют, какие редкие признаки чаще смешиваются с большинством.

MixUp и fixed uniform label smoothing являются affine-преобразованиями targets и коммутируют при одинаковом epsilon. Для q=(0.3,0.7,0), epsilon=0.1 получаем (0.30333,0.66333,0.03333) независимо от того, сгладили исходные labels до смеси или смешали сначала. Это алгебра targets; совместное влияние методов на обучение не обязано быть полезным. Слишком сильное размягчение может убрать нужный сигнал.

### Manifold Mixup и frozen representation

Представим сеть как $f=g\circ h$. Manifold Mixup создаёт $\widetilde h=\lambda h(x_i)+(1-\lambda)h(x_j)$ и продолжает вычисление через g с mixed target. Не надо декодировать mixed hidden vector в «новое слово» или изображение: требование относится к последующему предсказателю.

Если h обучаем, gradients могут менять геометрию representation вместе с g. Если h заморожен, можно обучать g на смеси его фиксированных outputs, но h не научится новым инвариантностям этим путём. Совместимость hidden states, layer choice, paired examples и task labels остаётся предпосылкой. Для sequence tensors нужны ещё выравнивание позиций и padding masks; одинаковый shape не означает одинаковое смысловое место. Механизм исследуется в [Manifold Mixup](https://arxiv.org/abs/1806.05236).

## CutMix и удаление областей

CutMix заменяет region image A region из B, а weights targets определяются фактической заменённой площадью:

$$
\lambda_{\mathrm{eff}}=1-\frac{A_{\mathrm{patch}}}{A_{\mathrm{image}}}.
$$

Если в image $100\times100$ вставлен patch (фрагмент изображения) $20\times40$, исходному class остаётся weight 0.92, вставленному — 0.08. После clipping patch по границам нужно пересчитать actual area, а не сохранять исходную sampled lambda. При detection/segmentation simple class mixture недостаточна: обновляются spatial labels, видимость и правила для перекрытых objects.

Cutout/Random Erasing удаляют/заменяют region без добавления label другого объекта. Они учат устойчивости к occlusion только пока оставшееся evidence достаточно. Если entire small object исчез, сохранение label может стать noise. DropBlock из главы [«Dropout, DropConnect, DropBlock и случайные ветви»](<reg-stochastic.md>) удаляет activations внутри network, а не входные pixels; эффекты и настройки различаются.

AutoAugment ищет политику transformations в заданном search space, RandAugment упрощает управление выбором и strength. Это способы организовать augmentation policy, а не гарантированно безопасные для любой annotation presets. Подбор policy по test недопустим.

### Реальная площадь после clipping

Пусть изображение 100×100, nominal patch 40×40 с центром (5,10). Его half-open границы по x равны [−15,25), по y [−10,30). После пересечения с изображением остаётся область [0,25)×[0,30), площадь 750, а не 1600. При area-weighted CutMix target исходного класса имеет weight 0.925, донорского — 0.075. Использование nominal площади дало бы ошибочный weight 0.84.

Геометрические границы зависят от округления, индексации и способа выбора patch. Coefficient пересчитывают по фактически заменённым pixels. Если область пустая, input и target должны согласованно остаться исходными. Само правило площади является prior классификационного [CutMix](https://arxiv.org/abs/1905.04899), а не измерением доли содержательного evidence каждого класса: 7.5% картинки могут содержать весь маленький объект или только фон.

### Почему segmentation и detection нельзя свести к одной смеси классов

Для semantic segmentation label в области donor обычно берётся из donor mask, вне области — из recipient mask, с сохранением ignore policy. Это пространственная композиция targets, а не один global vector 0.925/0.075. Для instance segmentation сохраняют identity и видимую часть каждого объекта, не смешивая instance IDs как непрерывные числа.

Для detection donor objects могут быть обрезаны patch, а recipient objects — частично закрыты. Нужно выбрать, описывает target видимую часть или полный объект, и как учитывается occlusion. Иногда надо обновить bbox, mask и visibility либо исключить объект по явному правилу. Если удалить label, оставив хорошо видимый объект как фон, получится новый источник противоречивого supervision. Связанные поля фильтруют одним object mapping.

Cutout и Random Erasing не добавляют другой класс: область удаляется либо заполняется по заданному правилу, а target сохраняется только при допустимой постановке. Их параметры включают долю площади, aspect ratio, число областей и распределение заполнения. DropBlock применяется к feature map (пространственная карта признаков) внутри сети, поэтому похожая геометрическая картинка маски не делает эти методы одним оператором.

## Политики и total variation

Для реконструкции image prior гладкости может задаваться anisotropic TV:

$$
\operatorname{TV}(u)=
\sum_{i,j}\bigl(|u_{i+1,j}-u_{i,j}|+|u_{i,j+1}-u_{i,j}|\bigr),
$$

где границы сумм ограничены существующими neighbours. Isotropic variant использует square root суммы квадратов directional differences. TV предпочитает кусочно-гладкие images, но может стирать texture и мелкие детали. Это prior на восстанавливаемое поле, а не универсальный penalty каждого CV classifier.

### Поиск политики тоже использует данные

AutoAugment задаёт пространство subpolicies: упорядоченные операции, вероятность применения каждой и magnitude. Policy search оценивает качество обученных кандидатов на development-данных. Поиск может быть дорогим; если его стоимость исключить из сравнения, итоговый бюджет будет неполным. Зафиксированная policy затем применяется при training, но её переносимость на новые источники и annotations требует проверки. Устройство поиска описано в [AutoAugment](https://arxiv.org/abs/1805.09501).

RandAugment уменьшает пространство управления до небольшого числа параметров, обычно количества выбираемых операций N и общего magnitude M. Один M преобразуется в разные физические единицы для rotation, contrast и других операций; M не является универсальным числом градусов или процентов. Конкретный банк операций, sampling и диапазоны принадлежат реализации. Более простая настройка в [RandAugment](https://arxiv.org/abs/1909.13719) не устраняет необходимость аудита меток.

Политика, хорошая для предметной классификации, может испортить OCR, pose или задачу различения диагностического цвета. Проверяют редкие срезы и композиции transforms. Validation, использованная для выбора policy, не является независимым доказательством её финального выигрыша; test сохраняют для оценки выбранного pipeline.

### Total variation — prior на поле

Для 2×2 изображения со строками (0,1) и (1,1) сумма модулей всех горизонтальных и вертикальных соседних различий равна 2. Если isotropic TV считать по forward differences, задавая отсутствующие на границе разности нулём, единственный ненулевой совместный gradient имеет компоненты (1,1), поэтому TV равна $\sqrt2$. Разные conventions дают разные значения и gradients.

TV поощряет небольшую общую величину пространственных изменений и может сохранять резкие границы между кусочно-постоянными областями. Но мелкая текстура или узкая структура тоже увеличивает TV; сильный коэффициент способен их стереть и создавать ступенчатые участки. Для реконструкции penalty сопоставляют с data fidelity, например squared error наблюдений. Для обычного classifier нет универсального смысла штрафовать TV любого внутреннего tensor без отдельной гипотезы.

При изменении resolution меняются число соседних пар и физический шаг сетки. Сумма и среднее TV — разные масштабы objective; их lambda нельзя переносить без пересчёта. Восстанавливаемая intensity map и дискретная class mask также требуют разных интерпретаций гладкости.

## Контроль и практика

1. Почему label smoothing conventions дают разные targets? **Ответ:** epsilon распределяется либо по всем classes, либо только по incorrect classes.
2. Почему CutMix использует actual area? **Ответ:** crop/clipping меняют количество реально заменённых pixels.
3. Можно ли MixUp считать линейной смесью class IDs? **Ответ:** нет; IDs nominal, смешиваются target distributions.

**Практика:** вручную вычислите оба smoothed target при четырёх classes. Сделайте MixUp и CutMix двух examples, проверьте сумму target weights. Для small-object image найдите strength, при которой label уже не оправдан; включите этот случай в ручной аудит до training comparison.

**Схема курса:** example + label → smoothing distribution or paired MixUp/CutMix transform → aligned soft target → loss → metric under declared augmentation policy



1. **Почему smoothing может уменьшить вероятность правильного класса при корректном обучении?** Оптимум изменённого target отличается от исходной conditional probability.
2. **Где вычисляются две CE составляющие MixUp?** На одном prediction для mixed input, а не обязательно на двух исходных images.
3. **Почему площадь CutMix не равна доле semantic evidence?** Содержательная информация распределена по pixels неравномерно.
4. **Почему uniform smoothing не тождественно entropy bonus?** Они соответствуют разным направлениям KL и разным целям.

Практика: вывести gradient soft CE, рассчитать оба smoothing conventions и один MixUp forward. Для CutMix записать nominal и clipped границы, обновить targets выбранной CV-задачи. Отдельно сравнить одиночные изменения и их сочетание при понятном бюджете. Все inputs, probabilities и площади главы учебные; результаты реального обучения и policy search здесь не заявляются.

## Результат главы и переход дальше

[Маршрут модуля](<README.md>) · [Происхождение и точные места переноса](<../editorial/coverage.md>)
