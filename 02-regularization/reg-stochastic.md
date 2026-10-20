# Dropout, DropConnect, DropBlock и случайные ветви

**Уровень:** общее ядро.

**Предпосылки:** [Обобщение, диагностика и границы теории](<reg-diagnosis.md>)

**Используют:** [SAM, Lookahead и итоговый выбор метода](<reg-selection.md>)

**Учебный вопрос:** Как структура случайной маски меняет вычисление и objective?





## Карта вычисления и границы этапов

![Dropout, DropConnect, DropBlock и случайные ветви: входы, преобразования и выходы](<../оформление/course/reg-stochastic.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Сравните формы и смысл каждого перехода с формулами главы. Выход преобразования и доказательство его качества — разные объекты: последняя проверка использует отдельный протокол оценки.

<!-- unit:2935674eb359b85f25 -->
<a id="u-2935674eb359b85f25"></a>

## Dropout: случайный оператор и objective

<!-- /unit:2935674eb359b85f25 -->

<!-- unit:4dc36d1459066d5992 -->
<a id="u-4dc36d1459066d5992"></a>

Пусть activation $h_j$ оставляется с вероятностью $q=1-p$. Для mask $m_j\sim\operatorname{Bernoulli}(q)$:

<!-- /unit:4dc36d1459066d5992 -->

<!-- unit:80cc015a99105eeb89 -->
<a id="u-80cc015a99105eeb89"></a>

$$
\widetilde h_j=\frac{m_j}{q}h_j,
\qquad
\mathbb E[\widetilde h_j\mid h_j]=h_j,
\qquad
\operatorname{Var}(\widetilde h_j\mid h_j)=\frac{p}{1-p}h_j^2.
$$

<!-- /unit:80cc015a99105eeb89 -->

<!-- unit:81ce16397dc03b2065 -->
<a id="u-81ce16397dc03b2065"></a>

Это inverted scaling: training activations усиливаются на $1/q$, а на eval dropout становится identity. При $p=0.5,h=2$ выход равен 0 или 4 с равной вероятностью, mean 2 и variance 4. Вероятность $p=1$ недопустима для этой формулы; нулевой keep probability требует отдельного поведения реализации.

<!-- /unit:81ce16397dc03b2065 -->

<!-- unit:2d079f51af7ea3a57e -->
<a id="u-2d079f51af7ea3a57e"></a>

Сохраняется expected activation, а не обязательно expected output всей nonlinear network. Поэтому выражение «точно усредняем все подсети» — слишком сильное. Dropout напоминает shared-weight ensemble, но eval identity не является точным average predictions всех возможных masks.

<!-- /unit:2d079f51af7ea3a57e -->

<!-- unit:d4fa64918c2593778a -->
<a id="u-d4fa64918c2593778a"></a>

Train/eval mode важен. Если забыть eval, predictions становятся стохастическими; если забыть train, regularizing noise пропадает. MC-dropout намеренно оставляет stochastic passes для оценки variation, но не превращает её автоматически в калиброванную полную Bayesian uncertainty.

<!-- /unit:d4fa64918c2593778a -->

<!-- unit:44f68429c0804c474b -->
<a id="u-44f68429c0804c474b"></a>

### Вывод среднего и дисперсии

<!-- /unit:44f68429c0804c474b -->

<!-- unit:b4b8d7fc0d4efbf465 -->
<a id="u-b4b8d7fc0d4efbf465"></a>

При keep probability q и независимой от текущего activation маске m имеем $E[m]=E[m^2]=q$. Поэтому $E[mh/q\mid h]=h$, а второй момент равен $h^2/q$. Вычитая квадрат среднего, получаем variance $h^2(1/q-1)=h^2p/(1-p)$. Чем ближе p к 1, тем больше масштабы редких сохранённых activations и их шум.

<!-- /unit:b4b8d7fc0d4efbf465 -->

<!-- unit:7c9a45d60cc2b8680e -->
<a id="u-7c9a45d60cc2b8680e"></a>

Это условное утверждение для фиксированного h. В ходе обучения h тоже меняется и зависит от других параметров и масок. Сохранение среднего в одном месте сети не даёт равенства outputs или gradients всей детерминированной eval-модели средним по всем stochastic networks. Исходная мотивация метода изложена в [статье Dropout](https://jmlr.org/papers/v15/srivastava14a.html); выбранная здесь inverted convention соответствует распространённой реализации с identity на eval.

<!-- /unit:7c9a45d60cc2b8680e -->

<!-- unit:b6d9e89c035c973bd6 -->
<a id="u-b6d9e89c035c973bd6"></a>

При фиксированной реализованной маске derivative $\widetilde h$ по h равна m/q. Сохранённые элементы получают соответствующий scaled gradient, выключенные — нулевой через этот путь. Маску forward используют и для обычного backward этого же вычисления; повторно случайно нарисовать её на backward — уже другой алгоритм. Выключение на одном шаге не означает постоянного удаления параметра из модели.

<!-- /unit:b6d9e89c035c973bd6 -->

<!-- unit:baaace39e176dce2db -->
<a id="u-baaace39e176dce2db"></a>

### Все четыре маски двух activations

<!-- /unit:baaace39e176dce2db -->

<!-- unit:034477bbebb5a22e47 -->
<a id="u-034477bbebb5a22e47"></a>

Пусть h=(1,2), линейный выход $s=h_1-h_2$, q=0.5, маски независимы. Возможны четыре равновероятных состояния: `(0,0)` даёт s=0; `(0,1)` — s=−4; `(1,0)` — s=2; `(1,1)` — s=−2. Средний выход −1 совпадает с eval-значением 1−2. Дисперсия равна $(1+9+9+1)/4=5$.

<!-- /unit:034477bbebb5a22e47 -->

<!-- unit:bc635cd924b5c304fa -->
<a id="u-bc635cd924b5c304fa"></a>

Но после ReLU outputs равны 0, 0, 2 и 0, их среднее 0.5. Детерминированный eval даёт $\operatorname{ReLU}(-1)=0$. Так можно непосредственно увидеть ограничение ансамблевой интерпретации: ожидание не проходит через произвольную нелинейность. То же относится к softmax и решениям по threshold.

<!-- /unit:bc635cd924b5c304fa -->

<!-- unit:5203bbe8725a56e92c -->
<a id="u-5203bbe8725a56e92c"></a>

### Когда dropout даёт квадратичный штраф

<!-- /unit:5203bbe8725a56e92c -->

<!-- unit:480df040bb29f7a940 -->
<a id="u-480df040bb29f7a940"></a>

Для линейного предсказания $s=w^\top h$, независимых coordinate masks с общим q и squared loss можно разложить ожидание:

<!-- /unit:480df040bb29f7a940 -->

<!-- unit:67481c59048894d83b -->
<a id="u-67481c59048894d83b"></a>

$$
E_m\left[\frac12(w^\top\widetilde h-y)^2\right]
=\frac12(w^\top h-y)^2+
\frac{p}{2q}\sum_jw_j^2h_j^2.
$$

<!-- /unit:67481c59048894d83b -->

<!-- unit:fc2005a652016f9414 -->
<a id="u-fc2005a652016f9414"></a>

Случайная ошибка вокруг среднего имеет нулевое ожидание, а независимость masks убирает cross-covariances. Остаётся дополнительная variance линейного выхода. После усреднения по dataset вес квадратичного штрафа каждой координаты зависит от среднего $h_j^2$. Даже в этом простом случае это не обязательно одинаковый isotropic L2.

<!-- /unit:fc2005a652016f9414 -->

<!-- unit:876b25b6238709261a -->
<a id="u-876b25b6238709261a"></a>

Для h и w из предыдущего примера при y=0 детерминированный half-squared loss равен 0.5, добавочный term=2.5, итог=3. Среднее прямых потерь четырёх выходов даёт то же число. Для nonlinear сети или другого loss равенство надо выводить заново: dropout не следует объявлять универсальным другим способом записать AdamW.

<!-- /unit:876b25b6238709261a -->

<!-- unit:20a8433c57795fd1eb -->
<a id="u-20a8433c57795fd1eb"></a>

## Структура маски и зависимости

<!-- /unit:20a8433c57795fd1eb -->

<!-- unit:2aefc430265a132350 -->
<a id="u-2aefc430265a132350"></a>

**DropConnect** маскирует weights: $\widetilde W=(M\odot W)/q$, затем применяет слой. Dropout маскирует outputs/inputs units, так что несколько связей получают общую mask. Это разные структуры noise, даже если marginal keep probability одинакова.

<!-- /unit:2aefc430265a132350 -->

<!-- unit:69b499bd97b71cb7a3 -->
<a id="u-69b499bd97b71cb7a3"></a>

В feature maps соседние activations коррелируют; независимые одиночные нули иногда легко компенсируются соседями. **DropBlock** удаляет пространственные блоки feature map. Размер и вероятность seed-blocks выбирают с учётом desired drop fraction и границ; поставить dropout probability напрямую как block-centre probability обычно неверно.

<!-- /unit:69b499bd97b71cb7a3 -->

<!-- unit:efe080ec42e2ea1e2b -->
<a id="u-efe080ec42e2ea1e2b"></a>

Channel/spatial dropout может выключать весь channel, сохраняя одну mask по пространственным координатам. Это ещё один granularity choice, а не синоним произвольного DropBlock. Метод выбирают по структуре representation, не по общему названию «обнуление».

<!-- /unit:efe080ec42e2ea1e2b -->

<!-- unit:4ff68a0db4cada57fe -->
<a id="u-4ff68a0db4cada57fe"></a>

### Какие оси разделяют случайность

<!-- /unit:4ff68a0db4cada57fe -->

<!-- unit:291c554f783491671b -->
<a id="u-291c554f783491671b"></a>

Elementwise dropout на тензоре `(B,T,D)` обычно может иметь отдельное решение для каждого элемента. Locked mask вида `(B,1,D)` сохраняет одно решение по feature во всех временных позициях объекта. Для CV channel mask `(B,C,1,1)` выключает канал целиком по H/W. Маска `(B,1,1,1)` на residual branch задаёт одно решение на весь пример. Broadcasting экономит запись, но меняет корреляцию шума.

<!-- /unit:291c554f783491671b -->

<!-- unit:2457d13d1d1dd3ccb7 -->
<a id="u-2457d13d1d1dd3ccb7"></a>

Нельзя говорить «один и тот же p, значит одинаковый regularizer». При общей маске несколько activations выключаются вместе; при независимых масках один путь способен компенсировать другой. Для recurrent и последовательностных моделей временная согласованность шума является отдельным выбором. API конкретного Dropout/Dropout2d или custom module надо читать с учётом формы входа и версии.

<!-- /unit:2457d13d1d1dd3ccb7 -->

<!-- unit:5c81d6e557f578b6df -->
<a id="u-5c81d6e557f578b6df"></a>

### DropConnect и общий input dropout

<!-- /unit:5c81d6e557f578b6df -->

<!-- unit:dac38752a02556c330 -->
<a id="u-dac38752a02556c330"></a>

Возьмём один input x=1 и два выходных элемента с весами 1. Input dropout с q=0.5 использует общую маску для x, поэтому outputs одновременно равны `(0,0)` или `(2,2)`. Каждый имеет mean 1 и variance 1; covariance outputs равна 1.

<!-- /unit:dac38752a02556c330 -->

<!-- unit:cbb1c7e2358cfd4972 -->
<a id="u-cbb1c7e2358cfd4972"></a>

DropConnect может независимо маскировать две связи. Тогда outputs имеют те же отдельные распределения, но возможны также `(0,2)` и `(2,0)`, covariance равна нулю. Различие находится в совместной структуре шума, хотя marginal probability каждой связи одинакова. Маскирование outputs другого слоя опять задаёт иной набор совместно выключаемых путей.

<!-- /unit:cbb1c7e2358cfd4972 -->

<!-- unit:1bac0e91a494d900af -->
<a id="u-1bac0e91a494d900af"></a>

### Пространственные блоки и границы

<!-- /unit:1bac0e91a494d900af -->

<!-- unit:f1f9fdb5a60a7cc9c6 -->
<a id="u-f1f9fdb5a60a7cc9c6"></a>

DropBlock удаляет связную пространственную область feature map, затрудняя восстановление информации из непосредственных соседей. Это отличается и от channel dropout, и от Cutout исходного изображения. Удаляется representation внутри модели, а не геометрически меняется target. Размер блока задаётся в координатах конкретной feature map: блок 3×3 на карте 7×7 и на карте 56×56 покрывает разные доли.

<!-- /unit:f1f9fdb5a60a7cc9c6 -->

<!-- unit:5fe46acdd33de46af4 -->
<a id="u-5fe46acdd33de46af4"></a>

Для карты H×W, блока b×b и допустимых полностью помещающихся размещений приближённая seed probability часто записывается как

<!-- /unit:5fe46acdd33de46af4 -->

<!-- unit:a0e8986372afaa80ad -->
<a id="u-a0e8986372afaa80ad"></a>

$$
\gamma\approx\frac{pHW}{b^2(H-b+1)(W-b+1)}.
$$

<!-- /unit:a0e8986372afaa80ad -->

<!-- unit:f7f2bec1fbe83382ec -->
<a id="u-f7f2bec1fbe83382ec"></a>

При H=W=7, b=3, desired p=0.1 получаем gamma≈0.02178. Есть 25 допустимых размещений; ожидаемое число выбранных seed-blocks около 0.5444. Если игнорировать overlap, ожидаемое число удалённых элементов 0.5444×9=4.9, то есть 10% из 49. Но пересекающиеся блоки удаляют общие элементы лишь один раз, а покрытие у границ отличается. Поэтому это калибровка приближённого механизма, а не гарантия точной доли p на каждом примере.

<!-- /unit:f7f2bec1fbe83382ec -->

<!-- unit:4dd921434c576933d2 -->
<a id="u-4dd921434c576933d2"></a>

Реализации могут нормировать surviving activations по фактически оставшемуся числу элементов. Для постоянной карты это сохраняет суммарный масштаб, но не является доказательством одинакового unbiased expectation каждого элемента при неравномерном покрытии. Случай, когда ничего не осталось, требует отдельной обработки. Полные правила и schedule надо брать из выбранного [DropBlock-метода](https://arxiv.org/abs/1810.12890) и реализации.

<!-- /unit:4dd921434c576933d2 -->

<!-- unit:783edacfa69541922b -->
<a id="u-783edacfa69541922b"></a>

## Стохастические residual-ветви

<!-- /unit:783edacfa69541922b -->

<!-- unit:743aad54ae35132c71 -->
<a id="u-743aad54ae35132c71"></a>

Для residual block $x+F(x)$ training может использовать:

<!-- /unit:743aad54ae35132c71 -->

<!-- unit:617e7ec0539c54eef6 -->
<a id="u-617e7ec0539c54eef6"></a>

$$
y=x+\frac bqF(x),\qquad b\sim\operatorname{Bernoulli}(q).
$$

<!-- /unit:617e7ec0539c54eef6 -->

<!-- unit:675da7cd39da4a00a8 -->
<a id="u-675da7cd39da4a00a8"></a>

На eval используется $x+F(x)$ в данной inverted convention. Исходные/конкретные implementations могут применять другую scaling convention. Identity path сохраняет передачу сигнала, а случайная residual branch меняет effective depth. Часто keep probability зависит от глубины; это hyperparameter schedule.

<!-- /unit:675da7cd39da4a00a8 -->

<!-- unit:f9cc5c9993ad1ef9f7 -->
<a id="u-f9cc5c9993ad1ef9f7"></a>

Нельзя удалить arbitrary слой с несовместимыми input/output shapes и считать его identity. Projection/downsampling blocks требуют отдельной обработки. Если implementation всё равно считает $F(x)$ и лишь затем умножает на ноль, экономии compute на этом step может не быть, хотя stochastic regularization сохраняется.

<!-- /unit:f9cc5c9993ad1ef9f7 -->

<!-- unit:1ed7daf35e5e924897 -->
<a id="u-1ed7daf35e5e924897"></a>

В двух residual branches Shake-Shake использует случайную смесь:

<!-- /unit:1ed7daf35e5e924897 -->

<!-- unit:feec3c6a9fc6015210 -->
<a id="u-feec3c6a9fc6015210"></a>

$$
y=x+\alpha F_1(x)+(1-\alpha)F_2(x).
$$

<!-- /unit:feec3c6a9fc6015210 -->

<!-- unit:c95523ff1fa1e2dd7e -->
<a id="u-c95523ff1fa1e2dd7e"></a>

В training коэффициент обратного распространения может быть другим, $\beta$. Поэтому заданный backward является специальным surrogate rule, а не обычной производной forward с fixed $\alpha$. Важно не выводить градиент как будто coefficients совпадают. На eval применяется deterministic combination по выбранной convention. См. [Shake-Shake](https://arxiv.org/abs/1705.07485).

<!-- /unit:c95523ff1fa1e2dd7e -->

<!-- unit:e5dd8607a740433c67 -->
<a id="u-e5dd8607a740433c67"></a>

ShakeDrop сочетает Bernoulli gate и случайное масштабирование residual contribution, включая разные forward/backward coefficients, чтобы распространить идею на другие residual architectures. Это не просто другое название stochastic depth. Конкретные coefficient ranges и layer schedules берутся из [описания метода](https://arxiv.org/abs/1802.02375), а не произвольно переносятся между architectures.

<!-- /unit:e5dd8607a740433c67 -->

<!-- unit:ae694aed1cee53a5f8 -->
<a id="u-ae694aed1cee53a5f8"></a>

### Что сохраняет identity path

<!-- /unit:ae694aed1cee53a5f8 -->

<!-- unit:9f1371bf34bbd412dd -->
<a id="u-9f1371bf34bbd412dd"></a>

В inverted stochastic depth условный mean блока $x+bF(x)/q$ равен $x+F(x)$ при фиксированном x и независимом gate. Для x=1, F=3, q=0.5 возможны outputs 1 и 7; mean=4, variance=9. Сам x передаётся при обоих исходах. Это позволяет временно убрать residual contribution, сохранив совместимый обходной путь.

<!-- /unit:9f1371bf34bbd412dd -->

<!-- unit:3441f711a9ea9b550b -->
<a id="u-3441f711a9ea9b550b"></a>

Однако projection skip может менять размерность, а branch может включать normalization и другие состояния. Пропуск должен соответствовать архитектуре. Если F сначала полностью вычислили, а затем умножили на ноль, FLOPs этой ветви уже потрачены. Экономия работы возможна только при подходящем условном выполнении; per-example masks внутри общего batch не обязательно дают её автоматически.

<!-- /unit:3441f711a9ea9b550b -->

<!-- unit:975288b2a697d99dd3 -->
<a id="u-975288b2a697d99dd3"></a>

Для трёх branches с keep probabilities 1, 0.75 и 0.5 ожидаемое число активных residual contributions равно 2.25. Это счётчик gates, а не точное равенство времени исполнения глубине 2.25 блока. Исходный [Stochastic Depth](https://arxiv.org/abs/1603.09382) и современные DropPath реализации могут использовать разные train/eval scaling conventions; их нельзя смешивать в одной формуле без пересчёта.

<!-- /unit:975288b2a697d99dd3 -->

<!-- unit:8b4ce7771eb4f7365a -->
<a id="u-8b4ce7771eb4f7365a"></a>

### Shake-Shake меняет и forward, и правило backward

<!-- /unit:8b4ce7771eb4f7365a -->

<!-- unit:475f96f2882fa3d829 -->
<a id="u-475f96f2882fa3d829"></a>

В двух residual branches forward использует alpha, а специальный backward — beta. Для общего параметра theta пусть $F_1=2\theta$, $F_2=-\theta$, skip=0. При alpha=0.25 и theta=1 forward равен −0.25. Обычная производная такого forward по theta тоже равна −0.25. Если backward использует beta=0.75, он передаёт коэффициент $0.75\cdot2+0.25\cdot(-1)=1.25$ при единичном upstream gradient.

<!-- /unit:475f96f2882fa3d829 -->

<!-- unit:733a12a5b33ec9e072 -->
<a id="u-733a12a5b33ec9e072"></a>

Следовательно, обычный autograd по записанному forward не реализует этот surrogate rule сам собой. Нужен явно предусмотренный механизм разных coefficients. В варианте с независимыми uniform alpha/beta на [0,1] eval использует средние coefficients 0.5. Выбор per-image или per-batch и длительность обучения входят в рецепт [Shake-Shake](https://arxiv.org/abs/1705.07485), а не выводятся из одного имени метода.

<!-- /unit:733a12a5b33ec9e072 -->

<!-- unit:6314502bcaf060d1c4 -->
<a id="u-6314502bcaf060d1c4"></a>

### ShakeDrop: gate и случайный residual scale

<!-- /unit:6314502bcaf060d1c4 -->

<!-- unit:b1307da4f70598ee85 -->
<a id="u-b1307da4f70598ee85"></a>

Механизм можно записать через forward scale $s_f=b+(1-b)\alpha$ и backward scale $s_b=b+(1-b)\beta$. При b=1 ветвь работает с coefficient 1 в обоих направлениях. При b=0 forward использует alpha, backward — beta. Это не просто Bernoulli-удаление: отрицательный alpha способен поменять знак residual contribution, а beta задаёт отдельную величину gradient.

<!-- /unit:b1307da4f70598ee85 -->

<!-- unit:882dac5d49e1eb97c7 -->
<a id="u-882dac5d49e1eb97c7"></a>

Например, при b=0, alpha=−0.5 и beta=0.25 forward contribution равен −0.5F, а multiplier backward — 0.25. В исходном исследовании рассматриваются, в зависимости от архитектуры, alpha=0 либо uniform alpha на [−1,1], beta на [0,1]. При независимости b и alpha и нулевом среднем alpha ожидаемый forward scale равен keep probability q. Поэтому в этой convention eval-масштаб q, а не 1 из inverted stochastic depth. Перенос scaling без проверки изменяет модель. Условия выбора distributions и стабилизации приведены в [ShakeDrop](https://arxiv.org/html/1802.02375v3).

<!-- /unit:882dac5d49e1eb97c7 -->

<!-- unit:45a1d2d2659f65e955 -->
<a id="u-45a1d2d2659f65e955"></a>

## Расположение в Transformer

<!-- /unit:45a1d2d2659f65e955 -->

<!-- unit:9e2d5a18245e5fc4bb -->
<a id="u-9e2d5a18245e5fc4bb"></a>

### Attention probabilities и маска доступности — разные операции

<!-- /unit:9e2d5a18245e5fc4bb -->

<!-- unit:6a53c13be2757d7268 -->
<a id="u-6a53c13be2757d7268"></a>

После $A=\operatorname{softmax}(QK^\top/\sqrt d+M)$ можно применить dropout к A, затем умножить полученные weights на V. До dropout каждая допустимая строка A суммируется в 1. После независимого inverted dropout это не обязано выполняться для конкретной маски.

<!-- /unit:6a53c13be2757d7268 -->

<!-- unit:5a10a2f162050b9349 -->
<a id="u-5a10a2f162050b9349"></a>

При A=(0.25,0.75), q=0.5 и маске `(1,0)` получаются weights `(0.5,0)` с суммой 0.5. Маска `(0,1)` даёт `(0,1.5)` с суммой 1.5; `(0,0)` — нулевую строку. При независимой маске ожидаемая строка остаётся исходной A, но отдельный forward не является выпуклой комбинацией V с единичной суммой. Дополнительная renormalization создала бы другой оператор.

<!-- /unit:5a10a2f162050b9349 -->

<!-- unit:eaee0af3acfe707593 -->
<a id="u-eaee0af3acfe707593"></a>

Causal/padding mask M запрещает доступ к определённым позициям по структуре задачи; dropout случайно ослабляет разрешённые связи в обучении. Это не взаимозаменяемые маски. Dropout logits до softmax также имеет другой эффект. Место применения должно совпадать с architecture/config: наличие attention в названии не означает один обязательный dropout-рецепт всех Transformer-моделей.

<!-- /unit:eaee0af3acfe707593 -->

<!-- unit:5fa543cbb150662b26 -->
<a id="u-5fa543cbb150662b26"></a>

### Residual, MLP, embeddings и LoRA

<!-- /unit:5fa543cbb150662b26 -->

<!-- unit:0f6927ce00f79d409f -->
<a id="u-0f6927ce00f79d409f"></a>

Dropout может применяться к embeddings, hidden activations MLP, outputs attention перед residual addition или к отдельной ветви. DropPath выключает residual contribution более крупным gate. Если после них стоит LayerNorm, статистика текущих activations дополнительно меняется; равенство средних до LayerNorm не переносится автоматически за неё.

<!-- /unit:0f6927ce00f79d409f -->

<!-- unit:0cfe56524f1ff37b94 -->
<a id="u-0cfe56524f1ff37b94"></a>

В S2 есть пример `lora_dropout=0.05`. В типичной линейной LoRA-ветви forward при обучении можно записать как

<!-- /unit:0cfe56524f1ff37b94 -->

<!-- unit:53fe79c238a24e71eb -->
<a id="u-53fe79c238a24e71eb"></a>

$$
y=W_0x+\frac{\alpha}{r}BA\operatorname{Dropout}(x).
$$

<!-- /unit:53fe79c238a24e71eb -->

<!-- unit:35f55b40c825b3b7fd -->
<a id="u-35f55b40c825b3b7fd"></a>

Шум действует на input адаптера, а frozen base branch продолжает получать исходный x. Это не то же самое, что dropout всего слоя или случайное обнуление rank-компонент. При eval dropout отключён, и для совместимой линейной постановки можно рассматривать merged update $W_0+(\alpha/r)BA$. Во время stochastic training одна фиксированная merged matrix не воспроизводит все маски. Фактический порядок операций проверяют в adapter implementation.

<!-- /unit:35f55b40c825b3b7fd -->

<!-- unit:e01682a97addb71671 -->
<a id="u-e01682a97addb71671"></a>

### Eval mode и MC-dropout

<!-- /unit:e01682a97addb71671 -->

<!-- unit:7566db410b1f7752ee -->
<a id="u-7566db410b1f7752ee"></a>

Обычный eval переводит dropout в identity для выбранной inverted convention. Но вызов без gradient recording сам по себе не обязан менять train/eval mode. Поэтому «вычисляли без backward» не доказывает отключения шума. Контракт стандартного слоя описан в [PyTorch Dropout](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html).

<!-- /unit:7566db410b1f7752ee -->

<!-- unit:9318f92bcf2797213a -->
<a id="u-9318f92bcf2797213a"></a>

Для MC-dropout намеренно получают несколько stochastic predictions одной модели. Если включить train mode целиком, BatchNorm тоже может перейти на batch statistics и обновлять buffers. Обычно хотят отдельно управлять dropout и остальными слоями, сохраняя нужный режим оценки. Разброс outputs отражает выбранный механизм и обученную модель; он не является автоматически полной uncertainty о новых доменах, неверных labels или неизвестных событиях. Качество и calibration такого показателя проверяют отдельно.

<!-- /unit:9318f92bcf2797213a -->

<!-- unit:d7fd215b2a933e37a3 -->
<a id="u-d7fd215b2a933e37a3"></a>

## Проверка и практика

<!-- /unit:d7fd215b2a933e37a3 -->

<!-- unit:081413da2fd3e78b1d -->
<a id="u-081413da2fd3e78b1d"></a>

Нужен baseline с $p=0$, несколько вероятностей, один data split и comparable steps/LR. Слишком сильный noise затрудняет fit и может ухудшить generalization. Несколько noises одновременно затрудняют диагностику. Не переносите «обычно полезный dropout» на каждый pretrained backbone без validation: training regime и already existing regularization важны.

<!-- /unit:081413da2fd3e78b1d -->

<!-- unit:0ba75685e2efdc76fe -->
<a id="u-0ba75685e2efdc76fe"></a>

1. Зачем division на keep probability? **Ответ:** сохраняется conditional mean activation при training.
2. Точен ли ensemble interpretation для всей nonlinear network? **Ответ:** нет; равенство mean activation не переносится через любую нелинейность.
3. Чем DropConnect отличается от dropout? **Ответ:** mask относится к отдельным weights, а не к outputs units.

<!-- /unit:0ba75685e2efdc76fe -->

<!-- unit:3c7404febbd06eddea -->
<a id="u-3c7404febbd06eddea"></a>

**Практика:** вычислите все masks для двух activations при $p=0.5$ и их expected linear output. Затем добавьте nonlinear activation и проверьте, совпадает ли expected result с eval. Нарисуйте masks Dropout/Channel/DropBlock для одной feature map и объясните различия.

<!-- /unit:3c7404febbd06eddea -->

<!-- unit:b8513c6056e8719941 -->
<a id="u-b8513c6056e8719941"></a>

**Схема курса:** activation/weight/region/branch → sampled mask at train time → scaled or conditional forward path → deterministic or averaged evaluation path

<!-- /unit:b8513c6056e8719941 -->

<!-- unit:15cf4a17a2e4e3d7e8 -->
<a id="u-15cf4a17a2e4e3d7e8"></a>

<!-- source-depth-regularization-Г -->

<!-- /unit:15cf4a17a2e4e3d7e8 -->

<!-- unit:6025b136c42c8d8a0e -->
<a id="u-6025b136c42c8d8a0e"></a>

Начинают с p=0 и фиксируют расположение каждой маски, её форму, sharing по осям, scaling и train/eval поведение. Затем сравнивают ограниченный набор p при одинаковом data protocol и понятном budget. Сильный dropout может повысить train loss и одновременно ухудшить validation: шум затрудняет извлечение нужного сигнала, особенно при сочетании с другими сильными ограничениями. Его полезность не определяется только шириной train–validation gap.

<!-- /unit:6025b136c42c8d8a0e -->

<!-- unit:5a1d7ae3af04b660b8 -->
<a id="u-5a1d7ae3af04b660b8"></a>

1. **Почему сохранившееся среднее activation не гарантирует средний output?** Нелинейные преобразования не коммутируют с ожиданием.
2. **Почему input dropout и DropConnect могут иметь одинаковые marginals, но разный эффект?** Они задают разную совместную зависимость выключаемых связей.
3. **Почему DropBlock seed probability меньше desired drop fraction?** Один seed удаляет несколько spatial elements; overlap и границы дополнительно меняют долю.
4. **Почему обычный backward недостаточен для Shake-Shake?** Он использует производную с forward coefficient, тогда как метод может задавать другой backward coefficient.

<!-- /unit:5a1d7ae3af04b660b8 -->

<!-- unit:a1641fa6df106b1797 -->
<a id="u-a1641fa6df106b1797"></a>

Практика: перечислить четыре маски двух activations, проверить mean, variance и nonlinear output. Для одного Transformer-block выписать, где применяются causal mask, attention dropout, residual dropout и adapter dropout. Отдельно проверить eval mode и сделать несколько повторных forwards без заявления о Bayesian guarantee. Числа главы учебные; реальный прирост качества, ускорение stochastic depth и калиброванная uncertainty пользовательской модели здесь не заявляются.

<!-- /unit:a1641fa6df106b1797 -->

## Результат главы и переход дальше

Ответьте на вопрос: **Как структура случайной маски меняет вычисление и objective?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

Следующие темы: [SAM, Lookahead и итоговый выбор метода](<reg-selection.md>).

[Маршрут модуля](<README.md>) · [Происхождение и точные места переноса](<../editorial/coverage.md>)
