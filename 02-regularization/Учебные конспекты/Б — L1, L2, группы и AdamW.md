# Б. Штрафы на параметры: от геометрии до шага optimizer

[Оглавление](README.md). Пункт 2 и практика AdamW из плана. **Происхождение:** дополнительный разбор; в чатах PEFT нет полного вывода ridge/lasso. AdamW сверён с [первоисточником](https://arxiv.org/abs/1711.05101).

## 1. Один objective, разные предпочтения

Для data loss $L(w)$ распространены:

$$
L_{\mathrm{ridge}}=L(w)+\frac\lambda2\|w\|_2^2,
\qquad
L_{\mathrm{lasso}}=L(w)+\lambda\|w\|_1.
$$

Ridge предпочитает небольшую общую norm, lasso — в том числе sparse solutions. Геометрически level sets data loss касаются L2-ball или L1-ball; у L1 есть углы на координатных осях. Это объясняет появление точных нулей при соответствующей оптимизации. Обычный finite-step gradient descent на nonsmooth L1 не обязан сам давать идеальные нули без proximal/coordinate procedure.

Для ridge gradient penalty равен $\lambda w$. Если $L=\tfrac12\|Xw-y\|_2^2$, normal equations:

$$
(X^\top X+\lambda I)\widehat w=X^\top y.
$$

При $\lambda>0$ penalized coefficients получают устойчивое решение даже при singular $X^\top X$, если все рассматриваемые направления штрафуются. Intercept обычно не penalize; тогда matrix отличается соответствующим diagonal element. Численно решают систему, а не обязательно явно вычисляют inverse.

## 2. L1 и proximal step

Proximal gradient сначала делает обычный шаг $z=w-\eta\nabla L(w)$, затем решает отдельную задачу штрафа. Для L1 результат — soft threshold:

$$
w_j^+=\operatorname{sign}(z_j)\max(|z_j|-\eta\lambda,0).
$$

При $z=(0.30,-0.04,0.01)$ и $\eta\lambda=0.05$ получаем $(0.25,0,0)$. В отличие от L2, малые coordinates могут стать строго нулевыми. При сильной корреляции признаков lasso может нестабильно выбрать один из похожих; выбранный zero/nonzero не доказывает causal importance.

Elastic Net сочетает $\lambda_1\|w\|_1+\tfrac{\lambda_2}{2}\|w\|_2^2$. L2 часть делает распределение коэффициентов между correlated features устойчивее, L1 сохраняет sparsity. Group lasso использует:

$$
\Omega(w)=\lambda\sum_g\alpha_g\|w_g\|_2.
$$

Внутри группы L2 norm, между группами сумма norms; это может выключать группу целиком. Для категориального признака группой могут быть все его one-hot columns. Веса $\alpha_g$ учитывают, например, размер группы; без этого большие группы получают иной effective penalty. Перекрывающиеся groups требуют отдельной постановки.

## 3. Почему scaling важен

Признак в метрах и тот же признак в миллиметрах требуют разных coefficients для одинакового prediction. Один и тот же L1/L2 penalty поэтому задаёт разные предпочтения. Standardization fit только на train и применяется к validation/test. Для sparse matrices centering может разрушить sparsity; preprocessing выбирается с учётом representation.

Feature selection тоже находится внутри CV. Выбрать признаки на всех labels и затем оценить ridge на split означает leakage даже при корректно написанном penalty.

## 4. L2 и decoupled weight decay

При обычном SGD без momentum:

$$
w^+=w-\eta(\nabla L(w)+\lambda w)
=(1-\eta\lambda)w-\eta\nabla L(w).
$$

Так L2 эквивалентен согласованному multiplicative decay. В adaptive optimizer gradient масштабируется по coordinates и накопленным moments. Если включить $\lambda w$ в gradient, он также попадёт в эти moments. Decoupled AdamW применяет decay отдельно от preconditioned data update:

$$
w^+=(1-\eta\lambda)w-
\eta\frac{\widehat m}{\sqrt{\widehat v}+\epsilon}.
$$

При первом Adam step с нулевыми moments и без epsilon normalised gradient имеет величину около 1 по ненулевой coordinate. Возьмём $w=2,g=0.1,\eta=0.01,\lambda=0.1$. Adam с L2 использует $g+\lambda w=0.3$ и даёт примерно $1.99$. AdamW использует data gradient и отдельный decay: $2-0.01-0.002=1.988$. Уже первый шаг различается; далее различаются и moments. Это учебный scalar example, не сравнение качества методов.

Применять ли decay к bias и normalization parameters — отдельная config decision. Нельзя механически переносить lambda между SGD, Adam-L2 и AdamW; LR schedule и число steps тоже меняют cumulative shrinkage.

## 5. Контроль и практика

1. Почему L2 обычно не обнуляет coefficient точно? **Ответ:** smooth shrinkage не создаёт L1 threshold, кроме специальных data/solutions.
2. Что группирует Group Lasso? **Ответ:** объявленные subsets parameters; группа выключается целиком благодаря norm penalty.
3. Почему Adam-L2 отличается от AdamW? **Ответ:** в первом случае penalty проходит adaptive moments/scaling, во втором decay отделён от data gradient.

**Практика:** выведите gradient ridge и soft-threshold для трёх coordinates. Сравните два nearly identical features при L1/L2/Elastic Net. Вручную повторите scalar Adam пример; при эксперименте подбирайте lambda на validation при одинаковом budget и сохраняйте scale preprocessing.
