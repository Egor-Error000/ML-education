# Д. Как дерево делит категории на группы

[Оглавление](README.md). Пункт 5 плана. **Происхождение:** дополнение по плану. API-границы сверены с официальными страницами 17 октября 2026; при реализации нужно фиксировать installed version и device/mode.

## 1. Numeric threshold и categorical subset

Numeric split имеет вид $x<t$. Categorical split может иметь вид $c\in S$, где $S$ — subset categories. Пусть A/B/C/D имеют эффекты $(1,-1,1,-1)$, а произвольные codes — 1/2/3/4. Один numeric threshold не отделит A/C от B/D, но subset split сделает это сразу. Поэтому передать codes как обычное число и объявить field categorical — разные операции.

Полный перебор бинарных partitions $K$ categories требует $2^{K-1}-1$ вариантов без учёта симметрии сторон. Для scalar leaf objectives можно использовать ordering categories по статистике leaf response и проверять contiguous partitions этого порядка. Это порядок, выученный в текущем node по objective, а не порядок исходных IDs.

При second-order boosting node агрегирует gradients $G$ и Hessians $H$. Типичный scalar leaf weight имеет форму $-G/(H+\lambda)$. При подходящих assumptions categories с близкими optimal responses можно группировать. Для vector outputs универсального scalar ordering нет; алгоритм/approximation зависит от implementation. Нельзя переносить proof scalar optimal partitioning на все objectives без оговорок.

## 2. LightGBM

LightGBM поддерживает categorical fields и partitioning. В Python dataframe metadata помогает согласовать categories; при raw integer arrays нужно самостоятельно сохранить codes. Отрицательные category codes рассматриваются как missing, поэтому применять generic unknown=-1 означает выбрать именно missing route, а не выучить новый category effect.

Сглаживание и минимальные размеры category groups ограничивают overfit; это hyperparameters, которые подбираются внутри validation. При high cardinality стоит сравнить native mode с manual representations, а не предполагать universal winner. [Официальные categorical/missing conventions](https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html).

Пример ошибки contract: train mapping A=0,B=1,C=2, inference самостоятельно строит A=0,C=1. Теперь C будет прочитан как B, если metadata не восстановлена и передаются только codes. Совпадение dtype/shape не обнаружит semantic corruption.

## 3. CatBoost

CatBoost использует разные numerical statistics категорий и их combinations, а небольшие vocabularies может обрабатывать one-hot в зависимости от mode/config. Ordered target statistics опираются на precedents permutation; это не просто full-data mean. Ordered boosting отдельно связан с уменьшением prediction shift в самом gradient boosting. [Описание алгоритма](https://arxiv.org/abs/1706.09516).

Не нужно предварительно one-hot кодировать все fields, если цель — использовать native mechanism CatBoost: так он потеряет исходные category identities/комбинации. Threshold one-hot и доступные statistics зависят от loss, CPU/GPU и config. [Документация категориальных features](https://catboost.ai/docs/en/features/categorical-features).

Missing marker в string categorical field канонизируется осознанно: строки `"1"` и `"1.0"` не обязаны быть одним значением. Формат train/inference должен совпадать; null handling нельзя механически переносить с numeric features. Автоматические combinations тоже ограничиваются настройками и стоимостью, а не перебирают все возможные interactions без ограничений.

## 4. XGBoost

XGBoost поддерживает one-hot-подобные categorical splits и partitioning; выбор зависит, в частности, от threshold `max_cat_to_onehot`. Передаются categorical dtype/feature types и включается соответствующая поддержка. Для сохранения model используются formats, сохраняющие categorical metadata, например JSON/UBJSON.

Python interface начиная с 3.1 умеет auto-recoding для поддержанных dataframe inputs. Это решает согласование codes известных categories, но не делает unseen category статистически известной модели; documentation отдельно указывает ограничения input types и unknown handling. В других interfaces/старых версиях mapping может оставаться ответственностью preprocessing. [Categorical Data](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html).

Не следует утверждать исторически, что XGBoost «не умеет native categories», и не следует делать противоположное слишком сильное утверждение «теперь все unknown работают автоматически везде». Round-trip test должен проверять конкретный installed stack.

## 5. Как сравнивать библиотеки

Одинаковый split и исходные разрешённые features, comparable tuning budget, одинаковый outcome metric и дополнительные reports memory/latency. Native handling сравнивается с понятным one-hot/frequency/OOF baseline. Одинаковое число trees не всегда означает одинаковый compute или capacity.

Обязательно сохраняем pipeline → загружаем → сравниваем predictions на известных values, reordered vocabulary, missing, rare и unseen. Тестируйте также отсутствие category, присутствовавшей в train: это не то же, что появление новой. Статус dictionary и policy fallback фиксируются в artifact.

## 6. Контроль и практика

1. Почему codes не равны native handling? **Ответ:** numeric thresholds и subset splits задают разные partitions.
2. Делает ли auto-recoding unseen category выученной? **Ответ:** нет; согласование IDs не создаёт observations её effect.
3. Заменяет ли ordered permutation time split? **Ответ:** нет; chronological availability задаётся отдельно.

**Практика:** для A/B/C/D найдите subset с одинаковыми effects и покажите неудачу одного numeric threshold. Составьте serialization cases для трёх libraries с exact versions; выполненные results и project design записывайте раздельно.
