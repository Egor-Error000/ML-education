# Временные признаки и доступность метки

**Уровень:** основа ветки.

**Предпосылки:** [Split, cross-validation и утечки](<../01-reliability/df-split.md>), [Типы признаков и контракты представлений](<feat-types.md>)

**Используют:** [Автоматическая генерация признаков](<feat-autofe.md>)

**Учебный вопрос:** Как вычислить признак на исторический момент с учётом задержки данных?





## Карта вычисления и границы этапов

![Временные признаки и доступность метки: входы, преобразования и выходы](<../../оформление/course/feat-time.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Обучаемые статистики вычисляются на train; validation и новые объекты используют сохранённое состояние. Геометрическая формула с фиксированными константами не превращается в fit только из-за наличия блока на схеме.

<!-- unit:67d98b546491ca7a9b -->
<a id="u-67d98b546491ca7a9b"></a>

## Два времени одного примера

<!-- /unit:67d98b546491ca7a9b -->

<!-- unit:e41cdc46e147451a37 -->
<a id="u-e41cdc46e147451a37"></a>

У строки $j$ есть event time $t_j$ и label availability time $a_j$. На prediction time $t_i$ разрешены только labels, уже доступные к этому моменту. Признак «средний возврат по товару» не может включать ещё не произошедший возврат старого заказа, даже если сам заказ уже находится в истории.

<!-- /unit:e41cdc46e147451a37 -->

<!-- unit:ebae69083d56bb6108 -->
<a id="u-ebae69083d56bb6108"></a>

Допустимый набор для category $c$:

<!-- /unit:ebae69083d56bb6108 -->

<!-- unit:1ab8e20f627963edb9 -->
<a id="u-1ab8e20f627963edb9"></a>

$$
\mathcal H_i(c)=\{j:c_j=c,\ t_j<t_i,\ a_j<t_i\}.
$$

<!-- /unit:1ab8e20f627963edb9 -->

<!-- unit:3d305f617d896d128d -->
<a id="u-3d305f617d896d128d"></a>

Здесь использовано строгое неравенство; при совпадающих timestamps нужно определить порядок событий или cutoff. Если label пришла до inference в тот же timestamp по reliable event ordering, contract может учитывать это иначе. Главное — реальная availability, а не произвольный sorting строк.

<!-- /unit:3d305f617d896d128d -->

<!-- unit:d5680ed8e60dfe8c71 -->
<a id="u-d5680ed8e60dfe8c71"></a>

## Expanding и rolling statistics

<!-- /unit:d5680ed8e60dfe8c71 -->

<!-- unit:6ed8a98208a1487060 -->
<a id="u-6ed8a98208a1487060"></a>

Expanding encoding использует всё допустимое прошлое:

<!-- /unit:6ed8a98208a1487060 -->

<!-- unit:a097bdd61155ce3418 -->
<a id="u-a097bdd61155ce3418"></a>

$$
\operatorname{TE}_i(c)=
\frac{\sum_{j\in\mathcal H_i(c)}y_j+\alpha\mu_i}
{|\mathcal H_i(c)|+\alpha}.
$$

<!-- /unit:a097bdd61155ce3418 -->

<!-- unit:a80d879938d7356b77 -->
<a id="u-a80d879938d7356b77"></a>

Global prior $\mu_i$ тоже вычисляется только по доступным прошлым labels. При отсутствии истории применяется заранее объявленный initial prior. Вычисление «глобального среднего» по всем датам уже внесёт future information, даже если category sum causal.

<!-- /unit:a80d879938d7356b77 -->

<!-- unit:c6e528fa215882a378 -->
<a id="u-c6e528fa215882a378"></a>

Rolling window ограничивает набор, например $t_i-W\le t_j<t_i$, плюс requirement availability. Это позволяет быстрее реагировать на drift, но увеличивает variance на редких categories. Window по event time и window по label arrival time отвечают на разные вопросы; следует выбрать подходящий.

<!-- /unit:c6e528fa215882a378 -->

<!-- unit:376b9f3ce09f29b0bc -->
<a id="u-376b9f3ce09f29b0bc"></a>

Count всех предыдущих events можно знать раньше, чем их labels. Нельзя делить сумму известных positives на count всех events и трактовать неизвестные outcomes как negatives. Нужны counts completed labels и, при необходимости, отдельный feature count pending outcomes.

<!-- /unit:376b9f3ce09f29b0bc -->

<!-- unit:670910353876f551fc -->
<a id="u-670910353876f551fc"></a>

## Числовой пример задержки

<!-- /unit:670910353876f551fc -->

<!-- unit:6b4c0b16a953551d3e -->
<a id="u-6b4c0b16a953551d3e"></a>

Заказ 1 товара A создан в день 1, label 1 приходит перед prediction дня 4. Заказ 2 создан в день 2, label 0 приходит перед prediction дня 5. На день 3 ни одна label не доступна: encoding использует initial prior, а не mean $(1+0)/2$. После первого feedback в день 4 при prior 0.5 и $\alpha=2$ получаем $(1+2\cdot0.5)/(1+2)=2/3$. После двух feedback в день 5 получаем $(1+1)/(2+2)=0.5$.

<!-- /unit:6b4c0b16a953551d3e -->

<!-- unit:47b0f59633a29bec6d -->
<a id="u-47b0f59633a29bec6d"></a>

Нулевая label второго заказа до дня 5 неизвестна. Выгрузка, сделанная позже, содержит оба outcomes, но retrospective feature generation должна восстановить знания каждого historical момента. Это и есть point-in-time correctness.

<!-- /unit:47b0f59633a29bec6d -->

<!-- unit:dbc6cf478ebe7a1fe5 -->
<a id="u-dbc6cf478ebe7a1fe5"></a>

## Split и online update

<!-- /unit:dbc6cf478ebe7a1fe5 -->

<!-- unit:02413fa2b0da5b6a75 -->
<a id="u-02413fa2b0da5b6a75"></a>

В time holdout training model fit по прошлому, validation simulates future. Допустимо ли обновлять TE state во время validation? Зависит от deployment protocol. Если online system получает feedback и обновляет statistics без gradient retraining, evaluation может повторять этот процесс, используя labels только после их arrival. Если deployment mappings frozen, validation тоже должна их freeze.

<!-- /unit:02413fa2b0da5b6a75 -->

<!-- unit:200a2e31fb0ac72be5 -->
<a id="u-200a2e31fb0ac72be5"></a>

Нельзя использовать все validation labels сразу для encoding ранних validation rows. Walk-forward с proper delays может быть честным, но это другой protocol, чем frozen batch prediction. Его нужно описать и применить одинаково ко всем models.

<!-- /unit:200a2e31fb0ac72be5 -->

<!-- unit:522edf890d6a2a8e51 -->
<a id="u-522edf890d6a2a8e51"></a>

Для OOF training в chronological задаче ранний block может не иметь достаточного прошлого; его features строятся по prior, либо warm-up block используется только как history по объявленному правилу. Обычный random KFold использует future labels и недопустим для causal target statistic. Перестановки CatBoost решают другую algorithmic проблему и не отменяют этот temporal boundary.

<!-- /unit:522edf890d6a2a8e51 -->

<!-- unit:35c1540ca99c637887 -->
<a id="u-35c1540ca99c637887"></a>

## Drift и reproducibility

<!-- /unit:35c1540ca99c637887 -->

<!-- unit:c423d2eca4c33a0865 -->
<a id="u-c423d2eca4c33a0865"></a>

Popularity categories меняется; unknown/rare rates и counts monitors помогают понять coverage. Rolling statistics адаптируются быстрее, но могут шуметь. Window/smoothing выбираются на historical validation periods, а не по последнему закрытому test.

<!-- /unit:c423d2eca4c33a0865 -->

<!-- unit:380ca68b2bf1bfd4a1 -->
<a id="u-380ca68b2bf1bfd4a1"></a>

Сохраняются state snapshot, event cutoff, label delay rules, vocabulary и code version. При replay одинакового source stream должны получаться одинаковые features, если config и ordering fixed. Late corrections labels требуют policy: пересчитать прошлые snapshots для анализа либо только обновить будущее состояние; нельзя незаметно смешать оба режима.

<!-- /unit:380ca68b2bf1bfd4a1 -->

<!-- unit:3e9d0163bc49c22ced -->
<a id="u-3e9d0163bc49c22ced"></a>

## Контроль и итоговая практика

<!-- /unit:3e9d0163bc49c22ced -->

<!-- unit:5ace6e5a186f76abf3 -->
<a id="u-5ace6e5a186f76abf3"></a>

1. Достаточно ли $t_j<t_i$? **Ответ:** нет; target старого события мог ещё не прийти, требуется availability cutoff.
2. Можно ли unknown outcomes считать negatives в denominator? **Ответ:** нет; это смешивает отсутствие feedback с отрицательной меткой.
3. Всегда ли запрещён update statistics на validation? **Ответ:** он допустим в заранее объявленном online protocol при causal feedback и одинаковых условиях сравнения; мгновенное использование всех validation labels недопустимо.

<!-- /unit:5ace6e5a186f76abf3 -->

<!-- unit:beb4f1653cb8b1173f -->
<a id="u-beb4f1653cb8b1173f"></a>

**Практика:** постройте timeline пяти событий с delayed labels и вручную посчитайте expanding/rolling features. Затем соберите итоговый pipeline: schema/vocabulary, group/time split, manual baseline, OOF target encoding или native model, round-trip и rare/unknown slices. Итоговые метрики и threshold берите из курса данных, а не из training score.

<!-- /unit:beb4f1653cb8b1173f -->

<!-- unit:f282e851cae3ac4a04 -->
<a id="u-f282e851cae3ac4a04"></a>

**Схема курса:** event timestamps + feature available_at + label available_at → as-of cutoff → past-only aggregate/target statistic → point-in-time row → leakage audit

<!-- /unit:f282e851cae3ac4a04 -->

## Результат главы и переход дальше

Ответьте на вопрос: **Как вычислить признак на исторический момент с учётом задержки данных?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

Следующие темы: [Автоматическая генерация признаков](<feat-autofe.md>).

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
