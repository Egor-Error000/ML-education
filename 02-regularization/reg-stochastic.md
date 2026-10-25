# Dropout, DropConnect, DropBlock и случайные ветви

**Уровень:** общее ядро.

**Предпосылки:** [Обобщение, диагностика и границы теории](<reg-diagnosis.md>)

**Используют:** [SAM, Lookahead и итоговый выбор метода](<reg-selection.md>)

**Учебный вопрос:** Как структура случайной маски меняет вычисление и objective?

## Карта вычисления и границы этапов

![Dropout, DropConnect, DropBlock и случайные ветви: входы, преобразования и выходы](<../оформление/course/reg-stochastic.svg>)

Dropout меняет случайную обучающую функцию. Масштабирование сохраняет ожидаемый сигнал в принятой схеме; режим применения отключает шум и должен проверяться отдельно от режима обучения.

## Dropout: случайный оператор и objective (оптимизируемая целевая функция)

Пусть activation $h_j$ оставляется с вероятностью $q=1-p$. Для mask $m_j\sim\operatorname{Bernoulli}(q)$:

$$
\widetilde h_j=\frac{m_j}{q}h_j,
\qquad
\mathbb E[\widetilde h_j\mid h_j]=h_j,
\qquad
\operatorname{Var}(\widetilde h_j\mid h_j)=\frac{p}{1-p}h_j^2.
$$

Это inverted scaling: training activations усиливаются на $1/q$, а на eval dropout становится identity. При $p=0.5,h=2$ выход равен 0 или 4 с равной вероятностью, mean 2 и variance 4. Вероятность $p=1$ недопустима для этой формулы; нулевой keep probability требует отдельного поведения реализации.

Сохраняется expected activation, а не обязательно expected output всей nonlinear network. Поэтому выражение «точно усредняем все подсети» — слишком сильное. Dropout напоминает shared-weight ensemble, но eval identity не является точным average predictions всех возможных masks.

Train/eval mode важен. Если забыть eval, predictions становятся стохастическими; если забыть train, regularizing noise пропадает. MC-dropout намеренно оставляет stochastic passes для оценки variation, но не превращает её автоматически в калиброванную полную Bayesian uncertainty.

### Вывод среднего и дисперсии

При keep probability q и независимой от текущего activation маске m имеем $E[m]=E[m^2]=q$. Поэтому $E[mh/q\mid h]=h$, а второй момент равен $h^2/q$. Вычитая квадрат среднего, получаем variance $h^2(1/q-1)=h^2p/(1-p)$. Чем ближе p к 1, тем больше масштабы редких сохранённых activations и их шум.

Это условное утверждение для фиксированного h. В ходе обучения h тоже меняется и зависит от других параметров и масок. Сохранение среднего в одном месте сети не даёт равенства outputs или gradients всей детерминированной eval-модели средним по всем stochastic networks. Исходная мотивация метода изложена в [статье Dropout](https://jmlr.org/papers/v15/srivastava14a.html); выбранная здесь inverted convention соответствует распространённой реализации с identity на eval.

При фиксированной реализованной маске derivative $\widetilde h$ по h равна m/q. Сохранённые элементы получают соответствующий scaled gradient (градиент), выключенные — нулевой через этот путь. Маску forward (прямой проход вычисления) используют и для обычного backward (обратный проход вычисления производных) этого же вычисления; повторно случайно нарисовать её на backward — уже другой алгоритм. Выключение на одном шаге не означает постоянного удаления параметра из модели.

### Все четыре маски двух activations

Пусть h=(1,2), линейный выход $s=h_1-h_2$, q=0.5, маски независимы. Возможны четыре равновероятных состояния: `(0,0)` даёт s=0; `(0,1)` — s=−4; `(1,0)` — s=2; `(1,1)` — s=−2. Средний выход −1 совпадает с eval-значением 1−2. Дисперсия равна $(1+9+9+1)/4=5$.

Но после ReLU outputs равны 0, 0, 2 и 0, их среднее 0.5. Детерминированный eval даёт $\operatorname{ReLU}(-1)=0$. Так можно непосредственно увидеть ограничение ансамблевой интерпретации: ожидание не проходит через произвольную нелинейность. То же относится к softmax и решениям по threshold.

### Когда dropout даёт квадратичный штраф

Для линейного предсказания $s=w^\top h$, независимых coordinate masks с общим q и squared loss (функция потерь) можно разложить ожидание:

$$
E_m\left[\frac12(w^\top\widetilde h-y)^2\right]
=\frac12(w^\top h-y)^2+
\frac{p}{2q}\sum_jw_j^2h_j^2.
$$

Случайная ошибка вокруг среднего имеет нулевое ожидание, а независимость masks убирает cross-covariances. Остаётся дополнительная variance линейного выхода. После усреднения по dataset вес квадратичного штрафа каждой координаты зависит от среднего $h_j^2$. Даже в этом простом случае это не обязательно одинаковый isotropic L2.

Для h и w из предыдущего примера при y=0 детерминированный half-squared loss равен 0.5, добавочный term=2.5, итог=3. Среднее прямых потерь четырёх выходов даёт то же число. Для nonlinear сети или другого loss равенство надо выводить заново: dropout не следует объявлять универсальным другим способом записать AdamW.

## Структура маски и зависимости

**DropConnect** маскирует weights: $\widetilde W=(M\odot W)/q$, затем применяет слой. Dropout маскирует outputs/inputs units, так что несколько связей получают общую mask. Это разные структуры noise, даже если marginal keep probability одинакова.

В feature (признак) maps соседние activations коррелируют; независимые одиночные нули иногда легко компенсируются соседями. **DropBlock** удаляет пространственные блоки feature map. Размер и вероятность seed-blocks выбирают с учётом desired drop fraction и границ; поставить dropout probability напрямую как block-centre probability обычно неверно.

Channel/spatial dropout может выключать весь channel, сохраняя одну mask по пространственным координатам. Это ещё один granularity choice, а не синоним произвольного DropBlock. Метод выбирают по структуре representation, не по общему названию «обнуление».

### Какие оси разделяют случайность

Elementwise dropout на тензоре `(B,T,D)` обычно может иметь отдельное решение для каждого элемента. Locked mask вида `(B,1,D)` сохраняет одно решение по feature во всех временных позициях объекта. Для CV channel mask `(B,C,1,1)` выключает канал целиком по H/W. Маска `(B,1,1,1)` на residual branch задаёт одно решение на весь пример. Broadcasting экономит запись, но меняет корреляцию шума.

Нельзя говорить «один и тот же p, значит одинаковый regularizer». При общей маске несколько activations выключаются вместе; при независимых масках один путь способен компенсировать другой. Для recurrent и последовательностных моделей временная согласованность шума является отдельным выбором. API конкретного Dropout/Dropout2d или custom module надо читать с учётом формы входа и версии.

### DropConnect и общий input dropout

Возьмём один input x=1 и два выходных элемента с весами 1. Input dropout с q=0.5 использует общую маску для x, поэтому outputs одновременно равны `(0,0)` или `(2,2)`. Каждый имеет mean 1 и variance 1; covariance outputs равна 1.

DropConnect может независимо маскировать две связи. Тогда outputs имеют те же отдельные распределения, но возможны также `(0,2)` и `(2,0)`, covariance равна нулю. Различие находится в совместной структуре шума, хотя marginal probability каждой связи одинакова. Маскирование outputs другого слоя опять задаёт иной набор совместно выключаемых путей.

### Пространственные блоки и границы

DropBlock удаляет связную пространственную область feature map, затрудняя восстановление информации из непосредственных соседей. Это отличается и от channel dropout, и от Cutout исходного изображения. Удаляется representation внутри модели, а не геометрически меняется target. Размер блока задаётся в координатах конкретной feature map: блок 3×3 на карте 7×7 и на карте 56×56 покрывает разные доли.

Для карты H×W, блока b×b и допустимых полностью помещающихся размещений приближённая seed (начальное состояние генератора случайных чисел) probability часто записывается как

$$
\gamma\approx\frac{pHW}{b^2(H-b+1)(W-b+1)}.
$$

При H=W=7, b=3, desired p=0.1 получаем gamma≈0.02178. Есть 25 допустимых размещений; ожидаемое число выбранных seed-blocks около 0.5444. Если игнорировать overlap, ожидаемое число удалённых элементов 0.5444×9=4.9, то есть 10% из 49. Но пересекающиеся блоки удаляют общие элементы лишь один раз, а покрытие у границ отличается. Поэтому это калибровка приближённого механизма, а не гарантия точной доли p на каждом примере.

Реализации могут нормировать surviving activations по фактически оставшемуся числу элементов. Для постоянной карты это сохраняет суммарный масштаб, но не является доказательством одинакового unbiased expectation каждого элемента при неравномерном покрытии. Случай, когда ничего не осталось, требует отдельной обработки. Полные правила и schedule надо брать из выбранного [DropBlock-метода](https://arxiv.org/abs/1810.12890) и реализации.

## Стохастические residual-ветви

Для residual block $x+F(x)$ training может использовать:

$$
y=x+\frac bqF(x),\qquad b\sim\operatorname{Bernoulli}(q).
$$

На eval используется $x+F(x)$ в данной inverted convention. Исходные/конкретные implementations могут применять другую scaling convention. Identity path сохраняет передачу сигнала, а случайная residual branch меняет effective depth. Часто keep probability зависит от глубины; это hyperparameter schedule.

Нельзя удалить arbitrary слой с несовместимыми input/output shapes и считать его identity. Projection/downsampling (уменьшение пространственного разрешения) blocks требуют отдельной обработки. Если implementation всё равно считает $F(x)$ и лишь затем умножает на ноль, экономии compute на этом step может не быть, хотя stochastic regularization сохраняется.

В двух residual branches Shake-Shake использует случайную смесь:

$$
y=x+\alpha F_1(x)+(1-\alpha)F_2(x).
$$

В training коэффициент обратного распространения может быть другим, $\beta$. Поэтому заданный backward является специальным surrogate rule, а не обычной производной forward с fixed $\alpha$. Важно не выводить градиент как будто coefficients совпадают. На eval применяется deterministic combination по выбранной convention. См. [Shake-Shake](https://arxiv.org/abs/1705.07485).

ShakeDrop сочетает Bernoulli gate и случайное масштабирование residual contribution, включая разные forward/backward coefficients, чтобы распространить идею на другие residual architectures. Это не просто другое название stochastic depth. Конкретные coefficient ranges и layer schedules берутся из [описания метода](https://arxiv.org/abs/1802.02375), а не произвольно переносятся между architectures.

### Что сохраняет identity path

В inverted stochastic depth условный mean блока $x+bF(x)/q$ равен $x+F(x)$ при фиксированном x и независимом gate. Для x=1, F=3, q=0.5 возможны outputs 1 и 7; mean=4, variance=9. Сам x передаётся при обоих исходах. Это позволяет временно убрать residual contribution, сохранив совместимый обходной путь.

Однако projection skip может менять размерность, а branch может включать normalization (нормировка по выбранному правилу) и другие состояния. Пропуск должен соответствовать архитектуре. Если F сначала полностью вычислили, а затем умножили на ноль, FLOPs этой ветви уже потрачены. Экономия работы возможна только при подходящем условном выполнении; per-example masks внутри общего batch (пакет примеров) не обязательно дают её автоматически.

Для трёх branches с keep probabilities 1, 0.75 и 0.5 ожидаемое число активных residual contributions равно 2.25. Это счётчик gates, а не точное равенство времени исполнения глубине 2.25 блока. Исходный [Stochastic Depth](https://arxiv.org/abs/1603.09382) и современные DropPath реализации могут использовать разные train/eval scaling conventions; их нельзя смешивать в одной формуле без пересчёта.

### Shake-Shake меняет и forward, и правило backward

В двух residual branches forward использует alpha, а специальный backward — beta. Для общего параметра theta пусть $F_1=2\theta$, $F_2=-\theta$, skip=0. При alpha=0.25 и theta=1 forward равен −0.25. Обычная производная такого forward по theta тоже равна −0.25. Если backward использует beta=0.75, он передаёт коэффициент $0.75\cdot2+0.25\cdot(-1)=1.25$ при единичном upstream gradient.

Следовательно, обычный autograd по записанному forward не реализует этот surrogate rule сам собой. Нужен явно предусмотренный механизм разных coefficients. В варианте с независимыми uniform alpha/beta на [0,1] eval использует средние coefficients 0.5. Выбор per-image или per-batch и длительность обучения входят в рецепт [Shake-Shake](https://arxiv.org/abs/1705.07485), а не выводятся из одного имени метода.

### ShakeDrop: gate и случайный residual scale

Механизм можно записать через forward scale $s_f=b+(1-b)\alpha$ и backward scale $s_b=b+(1-b)\beta$. При b=1 ветвь работает с coefficient 1 в обоих направлениях. При b=0 forward использует alpha, backward — beta. Это не просто Bernoulli-удаление: отрицательный alpha способен поменять знак residual contribution, а beta задаёт отдельную величину gradient.

Например, при b=0, alpha=−0.5 и beta=0.25 forward contribution равен −0.5F, а multiplier backward — 0.25. В исходном исследовании рассматриваются, в зависимости от архитектуры, alpha=0 либо uniform alpha на [−1,1], beta на [0,1]. При независимости b и alpha и нулевом среднем alpha ожидаемый forward scale равен keep probability q. Поэтому в этой convention eval-масштаб q, а не 1 из inverted stochastic depth. Перенос scaling без проверки изменяет модель. Условия выбора distributions и стабилизации приведены в [ShakeDrop](https://arxiv.org/html/1802.02375v3).

## Расположение в Transformer

### Attention (механизм внимания) probabilities и маска доступности — разные операции

После $A=\operatorname{softmax}(QK^\top/\sqrt d+M)$ можно применить dropout к A, затем умножить полученные weights на V. До dropout каждая допустимая строка A суммируется в 1. После независимого inverted dropout это не обязано выполняться для конкретной маски.

При A=(0.25,0.75), q=0.5 и маске `(1,0)` получаются weights `(0.5,0)` с суммой 0.5. Маска `(0,1)` даёт `(0,1.5)` с суммой 1.5; `(0,0)` — нулевую строку. При независимой маске ожидаемая строка остаётся исходной A, но отдельный forward не является выпуклой комбинацией V с единичной суммой. Дополнительная renormalization создала бы другой оператор.

Causal/padding (дополнение до общей длины) mask M запрещает доступ к определённым позициям по структуре задачи; dropout случайно ослабляет разрешённые связи в обучении. Это не взаимозаменяемые маски. Dropout logits (логиты, оценки альтернатив до нормировки) до softmax также имеет другой эффект. Место применения должно совпадать с architecture/config: наличие attention в названии не означает один обязательный dropout-рецепт всех Transformer-моделей.

### Residual, MLP, embeddings (векторные представления) и LoRA

Dropout может применяться к embeddings, hidden activations MLP, outputs attention перед residual addition или к отдельной ветви. DropPath выключает residual contribution более крупным gate. Если после них стоит LayerNorm, статистика текущих activations дополнительно меняется; равенство средних до LayerNorm не переносится автоматически за неё.

Рассмотрим настройку `lora_dropout=0.05`. В типичной линейной LoRA-ветви forward при обучении можно записать как

$$
y=W_0x+\frac{\alpha}{r}BA\operatorname{Dropout}(x).
$$

Шум действует на input адаптера, а frozen base branch продолжает получать исходный x. Это не то же самое, что dropout всего слоя или случайное обнуление rank-компонент. При eval dropout отключён, и для совместимой линейной постановки можно рассматривать merged update $W_0+(\alpha/r)BA$. Во время stochastic training одна фиксированная merged matrix не воспроизводит все маски. Фактический порядок операций проверяют в adapter implementation.

### Eval mode и MC-dropout

Обычный eval переводит dropout в identity для выбранной inverted convention. Но вызов без gradient recording сам по себе не обязан менять train/eval mode. Поэтому «вычисляли без backward» не доказывает отключения шума. Контракт стандартного слоя описан в [PyTorch Dropout](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html).

Для MC-dropout намеренно получают несколько stochastic predictions одной модели. Если включить train mode целиком, BatchNorm тоже может перейти на batch statistics и обновлять buffers. Обычно хотят отдельно управлять dropout и остальными слоями, сохраняя нужный режим оценки. Разброс outputs отражает выбранный механизм и обученную модель; он не является автоматически полной uncertainty о новых доменах, неверных labels (метки) или неизвестных событиях. Качество и calibration такого показателя проверяют отдельно.

## Проверка и практика

Нужен baseline (базовый вариант сравнения) с $p=0$, несколько вероятностей, один data split (разбиение данных) и comparable steps/LR. Слишком сильный noise затрудняет fit и может ухудшить generalization. Несколько noises одновременно затрудняют диагностику. Не переносите «обычно полезный dropout» на каждый pretrained backbone (основная сеть извлечения признаков) без validation (проверочная часть для выбора решения): training regime и already existing regularization важны.

1. Зачем division на keep probability? **Ответ:** сохраняется conditional mean activation при training.
2. Точен ли ensemble interpretation для всей nonlinear network? **Ответ:** нет; равенство mean activation не переносится через любую нелинейность.
3. Чем DropConnect отличается от dropout? **Ответ:** mask относится к отдельным weights, а не к outputs units.

**Практика:** вычислите все masks для двух activations при $p=0.5$ и их expected linear output. Затем добавьте nonlinear activation и проверьте, совпадает ли expected result с eval. Нарисуйте masks Dropout/Channel/DropBlock для одной feature map и объясните различия.

**Схема курса:** activation/weight/region/branch → sampled mask at train time → scaled or conditional forward path → deterministic or averaged evaluation path



Начинают с p=0 и фиксируют расположение каждой маски, её форму, sharing по осям, scaling и train/eval поведение. Затем сравнивают ограниченный набор p при одинаковом data protocol и понятном budget. Сильный dropout может повысить train loss и одновременно ухудшить validation: шум затрудняет извлечение нужного сигнала, особенно при сочетании с другими сильными ограничениями. Его полезность не определяется только шириной train–validation gap.

1. **Почему сохранившееся среднее activation не гарантирует средний output?** Нелинейные преобразования не коммутируют с ожиданием.
2. **Почему input dropout и DropConnect могут иметь одинаковые marginals, но разный эффект?** Они задают разную совместную зависимость выключаемых связей.
3. **Почему DropBlock seed probability меньше desired drop fraction?** Один seed удаляет несколько spatial elements; overlap и границы дополнительно меняют долю.
4. **Почему обычный backward недостаточен для Shake-Shake?** Он использует производную с forward coefficient, тогда как метод может задавать другой backward coefficient.

Практика: перечислить четыре маски двух activations, проверить mean, variance и nonlinear output. Для одного Transformer-block выписать, где применяются causal mask, attention dropout, residual dropout и adapter dropout. Отдельно проверить eval mode и сделать несколько повторных forwards без заявления о Bayesian guarantee. Числа главы учебные; реальный прирост качества, ускорение stochastic depth и калиброванная uncertainty пользовательской модели здесь не заявляются.

## Результат главы и переход дальше

Следующие темы: [SAM, Lookahead и итоговый выбор метода](<reg-selection.md>).

[Маршрут модуля](<README.md>) · [Происхождение и точные места переноса](<../editorial/coverage.md>)
