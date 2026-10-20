# Штрафы, ограничения, L1/L2 и AdamW

**Уровень:** общее ядро.

**Предпосылки:** [Обобщение, диагностика и границы теории](<reg-diagnosis.md>), [Числовые преобразования и масштабирование](<../01-data-features/02-features/feat-numeric.md>)

**Используют:** [SAM, Lookahead и итоговый выбор метода](<reg-selection.md>)

**Учебный вопрос:** Как штраф изменяет оптимальное решение и зависит от масштаба признаков?





## Карта вычисления и границы этапов

![Штрафы, ограничения, L1/L2 и AdamW: входы, преобразования и выходы](<../оформление/course/reg-penalties.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Сравните формы и смысл каждого перехода с формулами главы. Выход преобразования и доказательство его качества — разные объекты: последняя проверка использует отдельный протокол оценки.

<!-- unit:d7c9316655ba6b13b2 -->
<a id="u-d7c9316655ba6b13b2"></a>

## Ridge и геометрия штрафа

<!-- /unit:d7c9316655ba6b13b2 -->

<!-- unit:726978e9d8b55c6768 -->
<a id="u-726978e9d8b55c6768"></a>

Для data loss $L(w)$ распространены:

<!-- /unit:726978e9d8b55c6768 -->

<!-- unit:d103dd9462f69eeac8 -->
<a id="u-d103dd9462f69eeac8"></a>

$$
L_{\mathrm{ridge}}=L(w)+\frac\lambda2\|w\|_2^2,
\qquad
L_{\mathrm{lasso}}=L(w)+\lambda\|w\|_1.
$$

<!-- /unit:d103dd9462f69eeac8 -->

<!-- unit:1f7661d5a02b87e941 -->
<a id="u-1f7661d5a02b87e941"></a>

Ridge предпочитает небольшую общую norm, lasso — в том числе sparse solutions. Геометрически level sets data loss касаются L2-ball или L1-ball; у L1 есть углы на координатных осях. Это объясняет появление точных нулей при соответствующей оптимизации. Обычный finite-step gradient descent на nonsmooth L1 не обязан сам давать идеальные нули без proximal/coordinate procedure.

<!-- /unit:1f7661d5a02b87e941 -->

<!-- unit:72375b3fd0a11b83f8 -->
<a id="u-72375b3fd0a11b83f8"></a>

Для ridge gradient penalty равен $\lambda w$. Если $L=\tfrac12\|Xw-y\|_2^2$, normal equations:

<!-- /unit:72375b3fd0a11b83f8 -->

<!-- unit:3497ad8d3ed2fb1b4f -->
<a id="u-3497ad8d3ed2fb1b4f"></a>

$$
(X^\top X+\lambda I)\widehat w=X^\top y.
$$

<!-- /unit:3497ad8d3ed2fb1b4f -->

<!-- unit:89d96c595d0ff9d0aa -->
<a id="u-89d96c595d0ff9d0aa"></a>

При $\lambda>0$ penalized coefficients получают устойчивое решение даже при singular $X^\top X$, если все рассматриваемые направления штрафуются. Intercept обычно не penalize; тогда matrix отличается соответствующим diagonal element. Численно решают систему, а не обязательно явно вычисляют inverse.

<!-- /unit:89d96c595d0ff9d0aa -->

<!-- unit:5268931f590d16451c -->
<a id="u-5268931f590d16451c"></a>

### Вывести решение, не теряя соглашение о масштабе

<!-- /unit:5268931f590d16451c -->

<!-- unit:7f70ebe2cee58d44a6 -->
<a id="u-7f70ebe2cee58d44a6"></a>

Зафиксируем objective $J(w)=\tfrac12\|Xw-y\|^2+\tfrac\lambda2\|w\|^2$. Производная data term равна $X^\top(Xw-y)$, penalty — $\lambda w$. Приравнивая сумму нулю, получаем normal equations раздела 1. Если data loss разделён на n, матрица уравнения становится $X^\top X+n\lambda I$ при той же численной lambda. Поэтому сравнение параметров разных реализаций начинается с полной формулы objective.

<!-- /unit:7f70ebe2cee58d44a6 -->

<!-- unit:ed932c239a7b27bb19 -->
<a id="u-ed932c239a7b27bb19"></a>

При $\lambda>0$ и штрафе всех рассматриваемых координат для любого ненулевого v имеем

<!-- /unit:ed932c239a7b27bb19 -->

<!-- unit:15f45b775e93f2f9fa -->
<a id="u-15f45b775e93f2f9fa"></a>

$$
v^\top(X^\top X+\lambda I)v=\|Xv\|^2+\lambda\|v\|^2>0.
$$

<!-- /unit:15f45b775e93f2f9fa -->

<!-- unit:29138b191bae0958a2 -->
<a id="u-29138b191bae0958a2"></a>

Матрица положительно определена, решение единственно даже при линейно зависимых columns X. Это свойство выпуклой квадратичной задачи, а не утверждение о единственном оптимуме произвольной нейросети с L2. Если intercept не штрафуется, его учитывают отдельно; центрирование X и y на train часто позволяет вывести решение без явного penalized intercept.

<!-- /unit:29138b191bae0958a2 -->

<!-- unit:49737f6fa376c00271 -->
<a id="u-49737f6fa376c00271"></a>

### Один полностью вычисленный пример

<!-- /unit:49737f6fa376c00271 -->

<!-- unit:9dbf35d92dec58981d -->
<a id="u-9dbf35d92dec58981d"></a>

Пусть $X=\operatorname{diag}(3,1)$ и $y=(3,1)^\top$. OLS даёт w=(1,1). При lambda=1 ridge решает диагональную систему с элементами 10 и 2 и правой частью (9,1), поэтому $w=(0.9,0.5)$. Предсказания равны (2.7,0.5). Data loss — $(0.3^2+0.5^2)/2=0.17$, penalty — $(0.9^2+0.5^2)/2=0.53$, objective=0.70.

<!-- /unit:9dbf35d92dec58981d -->

<!-- unit:bf2cea3bb9eeb758b7 -->
<a id="u-bf2cea3bb9eeb758b7"></a>

У OLS в этом же penalized objective data loss=0, penalty=1, objective=1. Значит, ridge намеренно допускает ненулевую training error, чтобы уменьшить выбранный общий objective. Это ещё не доказательство меньшего test risk: для него нужны предположения о данных или независимая оценка. Численное значение objective нельзя напрямую сравнивать с другим loss, у которого другая нормировка или другой коэффициент.

<!-- /unit:bf2cea3bb9eeb758b7 -->

<!-- unit:34e9e5d763b52a3e1a -->
<a id="u-34e9e5d763b52a3e1a"></a>

### Почему малые singular directions особенно чувствительны

<!-- /unit:34e9e5d763b52a3e1a -->

<!-- unit:1074d808b3849bd058 -->
<a id="u-1074d808b3849bd058"></a>

Для SVD $X=U\operatorname{diag}(d_j)V^\top$ ridge coefficients в соответствующих направлениях имеют множители $d_j/(d_j^2+\lambda)$ при проекциях $u_j^\top y$. OLS использует $1/d_j$ для ненулевых singular values. Малый d усиливает наблюдательный шум в OLS; ridge ограничивает это усиление.

<!-- /unit:1074d808b3849bd058 -->

<!-- unit:13b990f2ce373035e1 -->
<a id="u-13b990f2ce373035e1"></a>

В fitted predictions shrinkage множитель равен $d_j^2/(d_j^2+\lambda)$. В примере d=3 и d=1 дают 0.9 и 0.5. Значит, ridge не просто умножает все коэффициенты на одну константу: сила сжатия зависит от геометрии X. Для linear Gaussian model с noise variance $\sigma^2$ дисперсия коэффициента в направлении j равна $\sigma^2d_j^2/(d_j^2+\lambda)^2$, а сжатие создаёт bias относительно ненулевого истинного коэффициента.

<!-- /unit:13b990f2ce373035e1 -->

<!-- unit:ae0121766732f41e6a -->
<a id="u-ae0121766732f41e6a"></a>

След матрицы сглаживания $X(X^\top X+\lambda I)^{-1}X^\top$ равен сумме этих prediction-множителей и используется как effective degrees of freedom в данном линейном контексте. В примере получаем 1.4 вместо 2 у полного OLS. Это конкретная характеристика linear smoother, а не универсальное число обучаемых параметров нейросети. Связанные постановки линейных моделей описаны в [документации scikit-learn](https://scikit-learn.org/stable/modules/linear_model.html).

<!-- /unit:ae0121766732f41e6a -->

<!-- unit:a86fa0806a6224e9d6 -->
<a id="u-a86fa0806a6224e9d6"></a>

## L1 и proximal update

<!-- /unit:a86fa0806a6224e9d6 -->

<!-- unit:a6e31ecd8440bc6262 -->
<a id="u-a6e31ecd8440bc6262"></a>

Proximal gradient сначала делает обычный шаг $z=w-\eta\nabla L(w)$, затем решает отдельную задачу штрафа. Для L1 результат — soft threshold:

<!-- /unit:a6e31ecd8440bc6262 -->

<!-- unit:053f4b1109b6885074 -->
<a id="u-053f4b1109b6885074"></a>

$$
w_j^+=\operatorname{sign}(z_j)\max(|z_j|-\eta\lambda,0).
$$

<!-- /unit:053f4b1109b6885074 -->

<!-- unit:e3bd27139d01c91a4e -->
<a id="u-e3bd27139d01c91a4e"></a>

При $z=(0.30,-0.04,0.01)$ и $\eta\lambda=0.05$ получаем $(0.25,0,0)$. В отличие от L2, малые coordinates могут стать строго нулевыми. При сильной корреляции признаков lasso может нестабильно выбрать один из похожих; выбранный zero/nonzero не доказывает causal importance.

<!-- /unit:e3bd27139d01c91a4e -->

<!-- unit:d24d459ebd3bb576ae -->
<a id="u-d24d459ebd3bb576ae"></a>

Elastic Net сочетает $\lambda_1\|w\|_1+\tfrac{\lambda_2}{2}\|w\|_2^2$. L2 часть делает распределение коэффициентов между correlated features устойчивее, L1 сохраняет sparsity. Group lasso использует:

<!-- /unit:d24d459ebd3bb576ae -->

<!-- unit:41bba0bc0b8a843bd4 -->
<a id="u-41bba0bc0b8a843bd4"></a>

$$
\Omega(w)=\lambda\sum_g\alpha_g\|w_g\|_2.
$$

<!-- /unit:41bba0bc0b8a843bd4 -->

<!-- unit:9594477f28b3caea28 -->
<a id="u-9594477f28b3caea28"></a>

Внутри группы L2 norm, между группами сумма norms; это может выключать группу целиком. Для категориального признака группой могут быть все его one-hot columns. Веса $\alpha_g$ учитывают, например, размер группы; без этого большие группы получают иной effective penalty. Перекрывающиеся groups требуют отдельной постановки.

<!-- /unit:9594477f28b3caea28 -->

<!-- unit:84e1383dc882d787ab -->
<a id="u-84e1383dc882d787ab"></a>

### Subgradient и условие выключения координаты

<!-- /unit:84e1383dc882d787ab -->

<!-- unit:2ebad5911106b78406 -->
<a id="u-2ebad5911106b78406"></a>

У абсолютной величины derivative равна +1 справа от нуля и −1 слева. В нуле subgradient — весь отрезок [−1,1]. Для гладкого выпуклого data loss условие оптимальности lasso записывается как $0\in\nabla L(w)+\lambda\partial\|w\|_1$. Если координата равна нулю, соответствующий data gradient может лежать в [−lambda,+lambda], и subgradient штрафа его компенсирует.

<!-- /unit:2ebad5911106b78406 -->

<!-- unit:3a959aec3f6d6598e2 -->
<a id="u-3a959aec3f6d6598e2"></a>

Именно это создаёт диапазон условий, при которых ноль является оптимумом. У L2 derivative в нуле равна нулю: ненулевой data gradient там нечем компенсировать. Это объясняет различие sparsity без обещания, что любой L1-код автоматически получит нули. Обычный gradient/subgradient step может перескакивать через ноль; алгоритм решения важен.

<!-- /unit:3a959aec3f6d6598e2 -->

<!-- unit:2c911466f584d4d111 -->
<a id="u-2c911466f584d4d111"></a>

Для $L=\tfrac12\|Xw-y\|^2$ без intercept или после корректного центрирования весь w=0 оптимален, если $\|X^\top y\|_\infty\le\lambda$. Это полезная граница масштаба lasso-path для объявленного objective. Если loss делят на n, порог соответственно меняется. Использовать test y для вычисления пути выбора нельзя.

<!-- /unit:2c911466f584d4d111 -->

<!-- unit:a62e61cfc975e87921 -->
<a id="u-a62e61cfc975e87921"></a>

### Вывод soft threshold

<!-- /unit:a62e61cfc975e87921 -->

<!-- unit:385586dd49bc677b69 -->
<a id="u-385586dd49bc677b69"></a>

После data step $z=w-\eta\nabla L(w)$ proximal update решает по каждой координате

<!-- /unit:385586dd49bc677b69 -->

<!-- unit:53df9823d64cd61a01 -->
<a id="u-53df9823d64cd61a01"></a>

$$
\min_v\ \frac{(v-z)^2}{2\eta}+\lambda|v|.
$$

<!-- /unit:53df9823d64cd61a01 -->

<!-- unit:0e878629e3a4b0f220 -->
<a id="u-0e878629e3a4b0f220"></a>

При v>0 стационарность даёт $v=z-\eta\lambda$; такое решение допустимо лишь при $z>\eta\lambda$. При v<0 получаем $v=z+\eta\lambda$ и требование $z<-\eta\lambda$. Между этими порогами оптимум v=0. Сведение трёх случаев даёт soft-threshold формулу раздела 2.

<!-- /unit:0e878629e3a4b0f220 -->

<!-- unit:fe3e207db26cc42180 -->
<a id="u-fe3e207db26cc42180"></a>

Для z=(0.30,−0.04,0.01) и threshold 0.05 первая координата становится 0.25, остальные нулевыми. В отличие от hard threshold, большие значения тоже уменьшаются на порог. Для сравнения proximal L2 при штрафе $\lambda v^2/2$ даёт $v=z/(1+\eta\lambda)$, а явный gradient step по одному penalty даёт $(1-\eta\lambda)z$. Это разные дискретные шаги, хотя при малом произведении близки; нельзя подменять одну реализацию другой при конечном шаге.

<!-- /unit:fe3e207db26cc42180 -->

<!-- unit:0c9ccd99506c9e6bcf -->
<a id="u-0c9ccd99506c9e6bcf"></a>

### Корреляция признаков и смысл выбранного нуля

<!-- /unit:0c9ccd99506c9e6bcf -->

<!-- unit:62d3efb96c94af05c4 -->
<a id="u-62d3efb96c94af05c4"></a>

Рассмотрим один учебный пример с prediction $w_1+w_2$ и target 2. Data loss равен $\tfrac12(w_1+w_2-2)^2$: признаки полностью дублируются. Для lasso с lambda=0.5 оптимальная сумма равна 1.5, и любое неотрицательное распределение этой суммы между w1 и w2 даёт тот же objective 0.875. Решения (1.5,0), (0,1.5) и (0.75,0.75) эквивалентны по задаче.

<!-- /unit:62d3efb96c94af05c4 -->

<!-- unit:ea31c5b4bc8ccb3d58 -->
<a id="u-ea31c5b4bc8ccb3d58"></a>

Поэтому выбранный алгоритмом ноль не доказывает, что соответствующий признак не содержит информации. Малое изменение sample или порядка вычислений может выбрать другого представителя коррелированной группы. Для ridge с lambda=1 в том же data loss решение единственно: w1=w2=2/3. Добавление L2 к L1 в Elastic Net также предпочитает распределение между одинаковыми координатами. При lambda1=0.5 и lambda2=1 получаем w1=w2=0.5. Это демонстрация механизма, не сравнение качества разных objectives.

<!-- /unit:ea31c5b4bc8ccb3d58 -->

<!-- unit:3c99fd2f1f626053e6 -->
<a id="u-3c99fd2f1f626053e6"></a>

## Группы, масштаб и кодирование

<!-- /unit:3c99fd2f1f626053e6 -->

<!-- unit:b75abc88840af9f322 -->
<a id="u-b75abc88840af9f322"></a>

Признак в метрах и тот же признак в миллиметрах требуют разных coefficients для одинакового prediction. Один и тот же L1/L2 penalty поэтому задаёт разные предпочтения. Standardization fit только на train и применяется к validation/test. Для sparse matrices centering может разрушить sparsity; preprocessing выбирается с учётом representation.

<!-- /unit:b75abc88840af9f322 -->

<!-- unit:6b24765b3e033228a6 -->
<a id="u-6b24765b3e033228a6"></a>

Feature selection тоже находится внутри CV. Выбрать признаки на всех labels и затем оценить ridge на split означает leakage даже при корректно написанном penalty.

<!-- /unit:6b24765b3e033228a6 -->

<!-- unit:9873ad459b6c844b4b -->
<a id="u-9873ad459b6c844b4b"></a>

### Group soft threshold

<!-- /unit:9873ad459b6c844b4b -->

<!-- unit:3b91cc3917d5337147 -->
<a id="u-3b91cc3917d5337147"></a>

Для неперекрывающихся групп proximal update штрафа $\lambda\sum_g\alpha_g\|w_g\|_2$ имеет вид

<!-- /unit:3b91cc3917d5337147 -->

<!-- unit:ed745ceb817dda4854 -->
<a id="u-ed745ceb817dda4854"></a>

$$
w_g^+=\left(1-\frac{\eta\lambda\alpha_g}{\|z_g\|_2}\right)_+z_g,
$$

<!-- /unit:ed745ceb817dda4854 -->

<!-- unit:a3dd5393211518d2e1 -->
<a id="u-a3dd5393211518d2e1"></a>

а при нулевой norm группы результат определяют как ноль. В отличие от coordinate L1 здесь порог сравнивается с norm всей группы, а при сохранении группы её направление не меняется. Для z=(3,4), norm=5 и group threshold=2 получаем множитель 0.6 и результат (1.8,2.4). Для z=(0.3,0.4) с norm=0.5 та же группа выключается целиком. Coordinate L1 с threshold 2 на (3,4) дал бы (1,2), то есть другой механизм.

<!-- /unit:a3dd5393211518d2e1 -->

<!-- unit:9b14ca2106b05c1e82 -->
<a id="u-9b14ca2106b05c1e82"></a>

В one-hot группе можно объединить все columns одного категориального признака. Тогда решение может исключить признак целиком, а не отдельную категорию. Выбор alpha, например пропорционально квадратному корню числа координат, меняет относительное давление на группы; это соглашение, а не автоматическая гарантия справедливого сравнения всех групп. Перекрывающиеся группы требуют отдельной оптимизации: простое независимое применение приведённой формулы уже не решает общую задачу.

<!-- /unit:9b14ca2106b05c1e82 -->

<!-- unit:9e6aa99879824de519 -->
<a id="u-9e6aa99879824de519"></a>

### Единицы измерения меняют penalty

<!-- /unit:9e6aa99879824de519 -->

<!-- unit:132b3a24c4f617e02f -->
<a id="u-132b3a24c4f617e02f"></a>

Пусть x в метрах и prediction равен wx. После замены $x_{mm}=1000x$ тот же prediction требует $w_{mm}=w/1000$. Его L2 penalty в миллион раз меньше при той же численной lambda, хотя функция физически не изменилась. Значит, без согласованного scaling штраф предпочитает некоторые единицы измерения.

<!-- /unit:132b3a24c4f617e02f -->

<!-- unit:367c42664e1c495e1b -->
<a id="u-367c42664e1c495e1b"></a>

При стандартизации $z=(x-\mu)/s$ коэффициент и intercept переходят как $w_z=ws$, $b_z=b+w\mu$. После обучения обратное преобразование нужно для интерпретации. Среднее и scale fit только на train; константные и пропущенные признаки требуют объявленного правила. Для sparse one-hot matrix центрирование может сделать матрицу плотной и резко увеличить память.

<!-- /unit:367c42664e1c495e1b -->

<!-- unit:d7fa8f7e8299f3f778 -->
<a id="u-d7fa8f7e8299f3f778"></a>

Штраф зависит также от кодирования категорий. Reference coding и полный one-hot с intercept представляют похожие функции разными параметрами; naive norm penalty может предпочитать их по-разному. Смысл «выбрать категориальный признак» надо связать с группами и identifiability, а не просто считать число ненулевых столбцов. Полное объяснение representations находится в курсе категориальных признаков.

<!-- /unit:d7fa8f7e8299f3f778 -->

<!-- unit:f5fc72818fe5cead98 -->
<a id="u-f5fc72818fe5cead98"></a>

### Штраф вокруг нуля и штраф вокруг pretrained-параметров

<!-- /unit:f5fc72818fe5cead98 -->

<!-- unit:c9c9ee8f928f02b720 -->
<a id="u-c9c9ee8f928f02b720"></a>

L2 на полном W предпочитает близость к нулю. Штраф $\|W-W_0\|^2$ предпочитает близость к pretrained reference W0; это другая цель. AdamW decay на обучаемом W сам по себе не равен ограничению расстояния до W0. При замороженном W0 и обучаемом update надо уточнить, штрафуется ли update, его факторы или итоговая матрица.

<!-- /unit:c9c9ee8f928f02b720 -->

<!-- unit:3bb99abf20bf80eb12 -->
<a id="u-3bb99abf20bf80eb12"></a>

Для LoRA $\Delta W=BA$ низкий rank не ограничивает norm update. Штраф $\|A\|^2+\|B\|^2$ также не тождественен $\|BA\|^2$: замена A на cA и B на B/c сохраняет произведение, но меняет сумму norms факторов. Поэтому в Transformer fine-tuning важно записать, к каким параметрам применён decay, их scaling и способ инициализации. Одна фраза «добавили регуляризацию LoRA» не определяет objective.

<!-- /unit:3bb99abf20bf80eb12 -->

<!-- unit:d79ed3b6ba95adb53d -->
<a id="u-d79ed3b6ba95adb53d"></a>

## L2 и decoupled weight decay

<!-- /unit:d79ed3b6ba95adb53d -->

<!-- unit:57656fe6cae41a1236 -->
<a id="u-57656fe6cae41a1236"></a>

При обычном SGD без momentum:

<!-- /unit:57656fe6cae41a1236 -->

<!-- unit:bb2b66280355aa9e76 -->
<a id="u-bb2b66280355aa9e76"></a>

$$
w^+=w-\eta(\nabla L(w)+\lambda w)
=(1-\eta\lambda)w-\eta\nabla L(w).
$$

<!-- /unit:bb2b66280355aa9e76 -->

<!-- unit:04770514a7c1cca852 -->
<a id="u-04770514a7c1cca852"></a>

Так L2 эквивалентен согласованному multiplicative decay. В adaptive optimizer gradient масштабируется по coordinates и накопленным moments. Если включить $\lambda w$ в gradient, он также попадёт в эти moments. Decoupled AdamW применяет decay отдельно от preconditioned data update:

<!-- /unit:04770514a7c1cca852 -->

<!-- unit:d8e89dc993d95b9e78 -->
<a id="u-d8e89dc993d95b9e78"></a>

$$
w^+=(1-\eta\lambda)w-
\eta\frac{\widehat m}{\sqrt{\widehat v}+\epsilon}.
$$

<!-- /unit:d8e89dc993d95b9e78 -->

<!-- unit:73741feea00e2e1df8 -->
<a id="u-73741feea00e2e1df8"></a>

При первом Adam step с нулевыми moments и без epsilon normalised gradient имеет величину около 1 по ненулевой coordinate. Возьмём $w=2,g=0.1,\eta=0.01,\lambda=0.1$. Adam с L2 использует $g+\lambda w=0.3$ и даёт примерно $1.99$. AdamW использует data gradient и отдельный decay: $2-0.01-0.002=1.988$. Уже первый шаг различается; далее различаются и moments. Это учебный scalar example, не сравнение качества методов.

<!-- /unit:73741feea00e2e1df8 -->

<!-- unit:60b860d7dff6048199 -->
<a id="u-60b860d7dff6048199"></a>

Применять ли decay к bias и normalization parameters — отдельная config decision. Нельзя механически переносить lambda между SGD, Adam-L2 и AdamW; LR schedule и число steps тоже меняют cumulative shrinkage.

<!-- /unit:60b860d7dff6048199 -->

<!-- unit:5c84cfe1721a01a61e -->
<a id="u-5c84cfe1721a01a61e"></a>

### Почему эквивалентность SGD ограничена

<!-- /unit:5c84cfe1721a01a61e -->

<!-- unit:2cd070c37a45e4e5b3 -->
<a id="u-2cd070c37a45e4e5b3"></a>

В обычном SGD без momentum с согласованным коэффициентом включение lambda w в gradient и multiplicative decay дают одинаковую формулу раздела 4. Если шаг использует preconditioner D, coupled вариант меняет параметры как $w-\eta D(g+\lambda w)$, а отдельный decay — как $(1-\eta\lambda)w-\eta Dg$. Различие равно применению D к penalty в первом варианте.

<!-- /unit:2cd070c37a45e4e5b3 -->

<!-- unit:bf531f2265795dc9c4 -->
<a id="u-bf531f2265795dc9c4"></a>

Учебный локальный пример: w=(2,2), g=0, $D=\operatorname{diag}(1,0.1)$, eta=0.1, lambda=0.1. Coupled shrink даёт (1.98,1.998), decoupled — (1.98,1.98). Здесь D фиксирована специально для объяснения геометрии. В настоящем Adam она зависит от истории gradients; включённый L2 дополнительно меняет и саму историю.

<!-- /unit:bf531f2265795dc9c4 -->

<!-- unit:86dc4c96648a235e8f -->
<a id="u-86dc4c96648a235e8f"></a>

Momentum также требует точного определения: если penalty накапливается в velocity, это не то же самое, что вынести его за velocity без изменения остальных формул. Поэтому утверждение об эквивалентности следует всегда привязывать к указанному update, а не к общему слову «SGD».

<!-- /unit:86dc4c96648a235e8f -->

<!-- unit:7f767f288305d4ad78 -->
<a id="u-7f767f288305d4ad78"></a>

### Какие moments использует Adam

<!-- /unit:7f767f288305d4ad78 -->

<!-- unit:1f33abe14a6b008f52 -->
<a id="u-1f33abe14a6b008f52"></a>

Для data gradient g Adam обновляет

<!-- /unit:1f33abe14a6b008f52 -->

<!-- unit:8920aed2278160b771 -->
<a id="u-8920aed2278160b771"></a>

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
$$

<!-- /unit:8920aed2278160b771 -->

<!-- unit:738bfe5f0031512fb8 -->
<a id="u-738bfe5f0031512fb8"></a>

затем использует bias-corrected $\widehat m_t=m_t/(1-\beta_1^t)$ и $\widehat v_t=v_t/(1-\beta_2^t)$. Coupled L2 подставляет вместо g величину $g+\lambda w$ в обе формулы. В AdamW moments относятся к data gradient, а shrink выполняется отдельно. Если epsilon=0, moments изначально нулевые и первый gradient ненулевой, normalized update первого шага имеет величину 1 и знак gradient.

<!-- /unit:738bfe5f0031512fb8 -->

<!-- unit:261160028894e4e73f -->
<a id="u-261160028894e4e73f"></a>

В scalar примере раздела 4 g=0.1, w=2, lambda=0.1 дают coupled gradient 0.3. При eta=0.01 оба нормированных gradients на первом шаге равны +1. Coupled Adam получает 1.99; AdamW дополнительно уменьшает старый w на 0.002 и получает 1.988. Различие не исчезает из-за одинакового знака: moments уже накопили разные величины и дальше будут давать другую траекторию. Epsilon и иные варианты реализации меняют точные числа. Механизм отделения decay от адаптивного шага предложен в [статье AdamW](https://arxiv.org/abs/1711.05101).

<!-- /unit:261160028894e4e73f -->

<!-- unit:4f5f09e83f21d499ee -->
<a id="u-4f5f09e83f21d499ee"></a>

### Decoupled не означает независимый от schedule

<!-- /unit:4f5f09e83f21d499ee -->

<!-- unit:e0a189c67c71c3222e -->
<a id="u-e0a189c67c71c3222e"></a>

Если data gradient нулевой, а decay действительно применяется на каждом шаге, последовательность имеет вид

<!-- /unit:e0a189c67c71c3222e -->

<!-- unit:c84567aae2315abce6 -->
<a id="u-c84567aae2315abce6"></a>

$$
w_T=\left[\prod_{t=1}^T(1-\eta_t\lambda_t)\right]w_0.
$$

<!-- /unit:c84567aae2315abce6 -->

<!-- unit:42c805decfa83b021b -->
<a id="u-42c805decfa83b021b"></a>

При eta=0.01, lambda=0.1 и T=100 множитель равен $0.999^{100}\approx0.9048$. Даже без data update начальный w=2 уменьшится примерно до 1.8096. Для малых положительных $\eta_t\lambda_t$ произведение приближённо равно $\exp(-\sum_t\eta_t\lambda_t)$. Другой LR schedule или число optimizer steps меняют итоговое shrinkage. При ненулевых gradients полная траектория дополнительно зависит от порядка data updates.

<!-- /unit:42c805decfa83b021b -->

<!-- unit:29dd9ad9bd4ad32b8a -->
<a id="u-29dd9ad9bd4ad32b8a"></a>

Нулевой gradient tensor и отсутствие gradient у параметра могут обрабатываться optimizer по-разному; frozen параметры и пропущенные шаги тоже требуют проверки. Следует читать API своей версии, особенно при parameter groups и mixed precision. Практическая политика decay для bias, normalization scales, embeddings и adapters задаётся явно. Исключение некоторых групп — настройка метода, а не математический закон, что эти параметры никогда нельзя штрафовать.

<!-- /unit:29dd9ad9bd4ad32b8a -->

<!-- unit:aa1a656f2f9b0f6bf8 -->
<a id="u-aa1a656f2f9b0f6bf8"></a>

## Контроль и практика

<!-- /unit:aa1a656f2f9b0f6bf8 -->

<!-- unit:f4eef162676e8b0976 -->
<a id="u-f4eef162676e8b0976"></a>

1. Почему L2 обычно не обнуляет coefficient точно? **Ответ:** smooth shrinkage не создаёт L1 threshold, кроме специальных data/solutions.
2. Что группирует Group Lasso? **Ответ:** объявленные subsets parameters; группа выключается целиком благодаря norm penalty.
3. Почему Adam-L2 отличается от AdamW? **Ответ:** в первом случае penalty проходит adaptive moments/scaling, во втором decay отделён от data gradient.

<!-- /unit:f4eef162676e8b0976 -->

<!-- unit:befc9cb53ee72f43fe -->
<a id="u-befc9cb53ee72f43fe"></a>

**Практика:** выведите gradient ridge и soft-threshold для трёх coordinates. Сравните два nearly identical features при L1/L2/Elastic Net. Вручную повторите scalar Adam пример; при эксперименте подбирайте lambda на validation при одинаковом budget и сохраняйте scale preprocessing.

<!-- /unit:befc9cb53ee72f43fe -->

<!-- unit:1b6de99ba8e3beda65 -->
<a id="u-1b6de99ba8e3beda65"></a>

**Схема курса:** data loss + parameter groups → L1/L2/group penalty in objective or decoupled AdamW decay → gradient/update → validation behavior and parameter structure

<!-- /unit:1b6de99ba8e3beda65 -->

<!-- unit:1b105ee0f049eb7761 -->
<a id="u-1b105ee0f049eb7761"></a>

<!-- source-depth-regularization-Б -->

<!-- /unit:1b105ee0f049eb7761 -->

<!-- unit:b796e8077116f10691 -->
<a id="u-b796e8077116f10691"></a>

Фиксируют target, split, feature scaling и нормировку data loss. Строят baseline, затем sweep lambda с логарифмическим шагом; для Elastic Net подбирают обе составляющие или их объявленное сочетание. В каждой CV-итерации preprocessing и feature selection fit внутри train. Сохраняют не только validation metric, но и coefficients/norms, sparsity, число выбранных групп и устойчивость по допустимым повторениям.

<!-- /unit:b796e8077116f10691 -->

<!-- unit:9f7ff47521e44aec78 -->
<a id="u-9f7ff47521e44aec78"></a>

У нейросети отдельно записывают optimizer, parameter groups, decay, learning-rate schedule и число обновлений. Нельзя объяснить улучшение именно decay, если одновременно поменялись batch size, budget и аугментации. После выбора проверяют итоговую процедуру на независимом test и анализируют вероятность, calibration и нужные срезы задачи. Большая sparsity полезна лишь если соответствует цели; нулевой коэффициент не является доказательством причинной неважности признака.

<!-- /unit:9f7ff47521e44aec78 -->

<!-- unit:2b02ef75170d0223cd -->
<a id="u-2b02ef75170d0223cd"></a>

1. **Почему ridge не сжимает все singular directions одинаково?** Data curvature зависит от singular values X, а penalty добавляет одинаковую lambda к их квадратам.
2. **Почему lasso может выбрать произвольный дубликат признака?** Несколько распределений коэффициентов имеют одинаковый prediction и L1 penalty.
3. **Чем group threshold отличается от coordinate threshold?** Он сравнивает norm всей группы с порогом и при сохранении группы сжимает её целиком.
4. **Почему AdamW lambda нельзя переносить между schedules без проверки?** Cumulative decay зависит от произведений learning rate и decay на фактически выполненных шагах.

<!-- /unit:2b02ef75170d0223cd -->

<!-- unit:ba11418e63f99f6cc5 -->
<a id="u-ba11418e63f99f6cc5"></a>

Практика: воспроизвести diagonal ridge, lasso subgradient и group proximal примеры. Вручную сравнить coupled и decoupled шаги, затем проверить полную конфигурацию своего optimizer. Все расчёты здесь учебные; обучение пользовательской модели и measured выигрыш AdamW не заявляются.

<!-- /unit:ba11418e63f99f6cc5 -->

## Результат главы и переход дальше

Ответьте на вопрос: **Как штраф изменяет оптимальное решение и зависит от масштаба признаков?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

Следующие темы: [SAM, Lookahead и итоговый выбор метода](<reg-selection.md>).

[Маршрут модуля](<README.md>) · [Происхождение и точные места переноса](<../editorial/coverage.md>)
