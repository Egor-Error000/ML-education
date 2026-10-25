# Временные признаки и доступность метки

**Уровень:** основа ветки.

**Предпосылки:** [Split, cross-validation и утечки](<../01-reliability/df-split.md>), [Типы признаков и контракты представлений](<feat-types.md>)

**Используют:** [Автоматическая генерация признаков](<feat-autofe.md>)

**Учебный вопрос:** Как вычислить признак на исторический момент с учётом задержки данных?

## Карта вычисления и границы этапов

![Временные признаки и доступность метки: входы, преобразования и выходы](<../../оформление/course/feat-time.svg>)

Окно заканчивается в момент, доступный предсказанию. Периодические координаты сохраняют близость соседних фаз, но не разрешают использовать будущие события или поздно поступившие метки.

## Два времени одного примера

У строки $j$ есть event time $t_j$ и label (метка) availability time $a_j$. На prediction time $t_i$ разрешены только labels (метки), уже доступные к этому моменту. Признак «средний возврат по товару» не может включать ещё не произошедший возврат старого заказа, даже если сам заказ уже находится в истории.

Допустимый набор для category $c$:

$$
\mathcal H_i(c)=\{j:c_j=c,\ t_j<t_i,\ a_j<t_i\}.
$$

Здесь использовано строгое неравенство; при совпадающих timestamps нужно определить порядок событий или cutoff. Если label пришла до inference (применение обученной модели) в тот же timestamp по reliable event ordering, contract может учитывать это иначе. Главное — реальная availability, а не произвольный sorting строк.

## Expanding и rolling statistics

Expanding encoding использует всё допустимое прошлое:

$$
\operatorname{TE}_i(c)=
\frac{\sum_{j\in\mathcal H_i(c)}y_j+\alpha\mu_i}
{|\mathcal H_i(c)|+\alpha}.
$$

Global prior $\mu_i$ тоже вычисляется только по доступным прошлым labels. При отсутствии истории применяется заранее объявленный initial prior. Вычисление «глобального среднего» по всем датам уже внесёт future information, даже если category sum causal.

Rolling window ограничивает набор, например $t_i-W\le t_j<t_i$, плюс requirement availability. Это позволяет быстрее реагировать на drift (изменение распределения или зависимости со временем), но увеличивает variance на редких categories. Window по event time и window по label arrival time отвечают на разные вопросы; следует выбрать подходящий.

Count всех предыдущих events можно знать раньше, чем их labels. Нельзя делить сумму известных positives на count всех events и трактовать неизвестные outcomes как negatives. Нужны counts completed labels и, при необходимости, отдельный feature (признак) count pending outcomes.

## Числовой пример задержки

Заказ 1 товара A создан в день 1, label 1 приходит перед prediction дня 4. Заказ 2 создан в день 2, label 0 приходит перед prediction дня 5. На день 3 ни одна label не доступна: encoding использует initial prior, а не mean $(1+0)/2$. После первого feedback в день 4 при prior 0.5 и $\alpha=2$ получаем $(1+2\cdot0.5)/(1+2)=2/3$. После двух feedback в день 5 получаем $(1+1)/(2+2)=0.5$.

Нулевая label второго заказа до дня 5 неизвестна. Выгрузка, сделанная позже, содержит оба outcomes, но retrospective feature generation должна восстановить знания каждого historical момента. Это и есть point-in-time correctness.

## Split (разбиение данных) и online update

В time holdout (отложенная часть данных) training model fit по прошлому, validation (проверочная часть для выбора решения) simulates future. Допустимо ли обновлять TE state (состояние системы) во время validation? Зависит от deployment protocol. Если online system получает feedback и обновляет statistics без gradient (градиент) retraining, evaluation может повторять этот процесс, используя labels только после их arrival. Если deployment mappings frozen, validation тоже должна их freeze.

Нельзя использовать все validation labels сразу для encoding ранних validation rows. Walk-forward с proper delays может быть честным, но это другой protocol, чем frozen batch (пакет примеров) prediction. Его нужно описать и применить одинаково ко всем models.

Для OOF training в chronological задаче ранний block может не иметь достаточного прошлого; его features (признаки) строятся по prior, либо warm-up block используется только как history по объявленному правилу. Обычный random KFold использует future labels и недопустим для causal target statistic. Перестановки CatBoost решают другую algorithmic проблему и не отменяют этот temporal boundary.

## Drift и reproducibility

Popularity categories меняется; unknown/rare rates и counts monitors помогают понять coverage. Rolling statistics адаптируются быстрее, но могут шуметь. Window/smoothing выбираются на historical validation periods, а не по последнему закрытому test.

Сохраняются state snapshot, event cutoff, label delay rules, vocabulary и code version. При replay одинакового source stream должны получаться одинаковые features, если config и ordering fixed. Late corrections labels требуют policy: пересчитать прошлые snapshots для анализа либо только обновить будущее состояние; нельзя незаметно смешать оба режима.

## Контроль и итоговая практика

1. Достаточно ли $t_j<t_i$? **Ответ:** нет; target старого события мог ещё не прийти, требуется availability cutoff.
2. Можно ли unknown outcomes считать negatives в denominator? **Ответ:** нет; это смешивает отсутствие feedback с отрицательной меткой.
3. Всегда ли запрещён update statistics на validation? **Ответ:** он допустим в заранее объявленном online protocol при causal feedback и одинаковых условиях сравнения; мгновенное использование всех validation labels недопустимо.

**Практика:** постройте timeline пяти событий с delayed labels и вручную посчитайте expanding/rolling features. Затем соберите итоговый pipeline: schema (схема допустимых данных)/vocabulary, group/time split, manual baseline (базовый вариант сравнения), OOF target encoding или native model, round-trip и rare/unknown slices. Итоговые метрики и threshold берите из курса данных, а не из training score (оценка).

**Схема курса:** event timestamps + feature available_at + label available_at → as-of cutoff → past-only aggregate/target statistic → point-in-time row → leakage (утечка недоступной информации) audit

## Результат главы и переход дальше

Следующие темы: [Автоматическая генерация признаков](<feat-autofe.md>).

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
