# Е. Как доставить правильные данные модели и восстановить эксперимент

[Оглавление](README.md). Пункт 6 плана. **Происхождение:** дополнительный учебный материал; чат о serving описывает другие очереди, а здесь рассматривается обучающий data pipeline. Термины API сверены с [PyTorch DataLoader](https://docs.pytorch.org/docs/stable/data.html).

## 1. От sample к batch

Dataset описывает получение example и его target/metadata. Map-style dataset позволяет обращаться по index; iterable-style выдаёт поток examples. Sampler выбирает индексы/порядок, collate объединяет examples в batch, transforms готовят входы. Разделение обязанностей помогает найти ошибки: неправильный label в Dataset отличается от испорченного padding в collate.

Для variable-length текста collate создаёт padding и attention mask. Пусть длины трёх последовательностей равны $(5,20,40)$. Padding до максимума даёт $3\cdot40=120$ позиций при 65 настоящих tokens; padding overhead равен $55/120\approx45.8\%$. Bucketing группирует близкие длины и снижает waste, но чрезмерно детерминированный порядок может изменить training dynamics. Для нескольких batches нужен shuffle между/внутри допустимых buckets.

Loss mask и attention mask имеют разный смысл. Attention mask исключает недоступные позиции из внимания; loss mask определяет, по каким targets вычисляется training loss. Нельзя считать, что padding автоматически исключён из всех objectives только потому, что одна mask существует.

## 2. Найти bottleneck измерением

Один batch проходит чтение, decode, CPU transforms, collation, transfer и compute. В последовательной схеме времена складываются. В устойчивом конвейере с достаточным перекрытием throughput ограничивает медленная стадия:

$$
T_{\mathrm{batch,seq}}=\sum_j t_j,
\qquad
T_{\mathrm{batch,pipe}}\gtrsim\max_j t_j.
$$

Это идеализированная нижняя граница, без queue overhead и resource contention. Например, чтение 12 ms, transform 18 ms, transfer 4 ms и compute 30 ms дают 64 ms последовательно и теоретический предел около 30 ms в установившемся pipeline. Простое удвоение workers не ускоряет GPU compute и может перегрузить диск.

Измерять нужно warm-up отдельно, время ожидания batch, compute, память, samples/s и полноту эпохи. Быстрый loader, который пропускает половину примеров, не является улучшением. Cache делает первый и последующие проходы разными; это указывается в benchmark.

## 3. Workers, prefetch и pinned memory

`num_workers` задаёт процессы загрузки; при нуле работа идёт в основном процессе. `prefetch_factor` и persistent workers влияют на подготовку/повторное использование workers. `pin_memory` может помогать CPU→CUDA transfer, но эффект зависит от устройства и полного pipeline. Параметры выбираются по measured throughput и памяти, а не по числу CPU cores автоматически.

На Windows multiprocessing требует корректной точки входа: запуск main-кода защищают `if __name__ == "__main__":`, а определения Dataset/collate/worker functions делают доступными для сериализации. Локальные lambda/closures могут мешать spawn. Сначала полезен корректный запуск с нулём workers, затем отдельная проверка multiprocessing. Точный API и поддержка параметров проверяются в установленной версии библиотеки.

Для iterable dataset каждый worker получает свою копию потока. Без sharding workers могут читать одни и те же examples. Аналогично несколько distributed ranks должны разделять samples; иначе «ускорение» незаметно меняет effective dataset. Проверяйте IDs, количество и duplicates за эпоху. `drop_last` сознательно исключает неполный batch и потому меняет полноту прохода.

## 4. Seeds — только часть воспроизводимости

Случайность живёт в sampler, transforms, worker processes и model training. Нужны согласованные generators/seeds и отдельное внимание к внешним библиотекам, которые используются в transforms. Но одинаковый seed не гарантирует побайтового совпадения на другой платформе/версии или при недетерминированных kernels.

Чтобы восстановить experiment, сохраняют исходный manifest, hashes/versions, split IDs, schema, preprocessing config, tokenizer/model versions, sampling/augmentation settings, code/config и evaluation protocol. DVC — один из способов реализации, но сами требования не зависят от выбранного инструмента.

Checkpoint модели без dataset version недостаточен: тот же filename может указывать на уже изменённые labels. Для streaming нужны позиция/shard state и правила shuffle/replay; при restart следует понимать, какие examples повторятся или будут пропущены. Точное продолжение посередине эпохи сложнее, чем запуск с начала эпохи с тем же seed.

## 5. Контроль и практика

1. Ускорит ли больше workers любую задачу? **Ответ:** нет; ограничивать могут compute, storage, сериализация или contention.
2. Почему iterable dataset проверяют на duplicates? **Ответ:** worker/rank replicas без правильного partition могут читать одинаковый поток.
3. Достаточно ли сохранить seed? **Ответ:** нет; нужны сами версии данных, split, transforms, код и конфигурация.

**Практика:** составьте таблицу измерений нескольких worker/prefetch settings в своём журнале: samples/s, wait time, RAM и число уникальных IDs. Для каждой эпохи проверьте ожидаемую полноту с учётом `drop_last`. Сохраните manifest и попробуйте восстановить split и первые batches в новом запуске.
