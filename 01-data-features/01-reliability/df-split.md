# Split, cross-validation и утечки

**Уровень:** общее ядро.

**Предпосылки:** [Постановка задачи и паспорт данных](<df-task.md>)

**Используют:** [Качество данных, пропуски, выбросы и разметка](<df-quality.md>), [Дисбаланс, метрики, порог и калибровка](<df-metrics.md>), [Дубликаты, аугментации и синтетика](<df-augmentation.md>), [Dataset, batching и воспроизводимость](<df-loading.md>), [Drift и надёжность оценки](<df-drift.md>), [Типы признаков и контракты представлений](<../02-features/feat-types.md>), [Числовые преобразования и масштабирование](<../02-features/feat-numeric.md>), [Кардинальность, частотные кодировки и hashing](<../02-features/feat-hashing.md>), [Target statistics, smoothing, cross-fitting и WoE](<../02-features/feat-target.md>), [Обучаемые embeddings и перенос представлений](<../02-features/feat-embeddings.md>), [Временные признаки и доступность метки](<../02-features/feat-time.md>), [Географические признаки и пространственная валидация](<../02-features/feat-geography.md>), [Отбор признаков и устойчивость выбора](<../02-features/feat-selection.md>), [Сокращение размерности: PCA и SVD](<../02-features/feat-reduction.md>), [Единый pipeline и применение к новым объектам](<../02-features/feat-pipeline.md>), [Изображения, аннотации и синхронные преобразования](<../03-modalities/df-images.md>), [Текст, токенизация и разбиение документов](<../03-modalities/df-text.md>), [Обобщение, диагностика и границы теории](<../../02-regularization/reg-diagnosis.md>), [Похожесть, расстояния и постановка кластеризации](<../../03-clustering/cl-geometry.md>), [Encoder, tokenizer, attention и pooling](<../../05-language-models/02-bert-encoders/bert-interface.md>), [Задача адаптации, baseline и выбор вмешательства](<../../05-language-models/03-adaptation/adapt-choice.md>), [RAG: BM25, dense, hybrid и multi-vector retrieval](<../../05-language-models/05-applications/app-rag.md>)

**Учебный вопрос:** Как разделить данные так, чтобы оценить именно будущий сценарий применения?





## Карта вычисления и границы этапов

![Split, cross-validation и утечки: входы, преобразования и выходы](<../../оформление/course/df-split.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Обучаемые статистики вычисляются на train; validation и новые объекты используют сохранённое состояние. Геометрическая формула с фиксированными константами не превращается в fit только из-за наличия блока на схеме.

<!-- unit:ceb1aef3ce3ce5daee -->
<a id="u-ceb1aef3ce3ce5daee"></a>

## Что означает test

<!-- /unit:ceb1aef3ce3ce5daee -->

<!-- unit:cb1d324d9476f12f5d -->
<a id="u-cb1d324d9476f12f5d"></a>

Training set нужен для fit параметров, validation — для выбора преобразований, hyperparameters и порога, test — для итоговой независимой оценки принятого решения. Если многократно смотреть test и менять модель под его ошибки, он становится частью разработки. Название файла `test` не сохраняет независимость автоматически.

<!-- /unit:cb1d324d9476f12f5d -->

<!-- unit:e8f67ba3069edc6060 -->
<a id="u-e8f67ba3069edc6060"></a>

Split должен имитировать будущий вопрос. При независимых объектах подходит random split; stratification сохраняет пропорции labels и уменьшает риск совсем пустого редкого класса. Для новых пользователей разделяют группы пользователей. Для будущего периода обучение должно предшествовать проверке. Одновременно новые пользователи и будущие даты могут требовать комбинированной схемы, а не выбора одного удобного splitter.

<!-- /unit:e8f67ba3069edc6060 -->

<!-- unit:068bba1ea963ef8229 -->
<a id="u-068bba1ea963ef8229"></a>

Stratification не устраняет совпадение пользователей или соседних видеокадров между частями. GroupKFold разделяет объявленные groups; если group ID выбран неправильно, механизм не исправит постановку. Разные снимки одного пациента или chunks одного документа следует связывать общей группой. См. [официальный разбор cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html).

<!-- /unit:068bba1ea963ef8229 -->

<!-- unit:3c92c58ea9762626f1 -->
<a id="u-3c92c58ea9762626f1"></a>

<!-- source-depth-data-Б -->

<!-- /unit:3c92c58ea9762626f1 -->

<!-- unit:25fcd738ac07dd8abe -->
<a id="u-25fcd738ac07dd8abe"></a>

### Split проверяет определённый перенос

<!-- /unit:25fcd738ac07dd8abe -->

<!-- unit:4604c8e020f88822f4 -->
<a id="u-4604c8e020f88822f4"></a>

Разделение не существует независимо от задачи. Если serving получает будущие события известных пользователей, важен порядок времени и доступная история. Если новые users, важна граница groups. Если новая камера или документ, проверяют соответствующий источник. Одна и та же случайная доля 20% может отвечать разным вопросам в зависимости от identities внутри неё.

<!-- /unit:4604c8e020f88822f4 -->

<!-- unit:978a0f67364f134ac5 -->
<a id="u-978a0f67364f134ac5"></a>

Train используется для fit; validation — для выбора метода, hyperparameters, threshold и других решений разработки. Test нужен для независимой оценки уже выбранной процедуры. Эти роли определяются использованием данных, а не названием файла. Если после каждого test результата менять модель, будущий score относится к процедуре, уже адаптированной к этому набору.

<!-- /unit:978a0f67364f134ac5 -->

<!-- unit:fb31c6d1dc32c83bb9 -->
<a id="u-fb31c6d1dc32c83bb9"></a>

Для known corpus retrieval новые queries могут искать по тем же documents, которые будут доступны приложению. Это другой protocol, чем обучение encoder с оценкой на невиданных documents. Не следует механически объявлять всякое повторение source leakage или, наоборот, любое исключение labels достаточным. Сначала фиксируют intended information/transfer contract.

<!-- /unit:fb31c6d1dc32c83bb9 -->

<!-- unit:6f8adb23298f0bfc09 -->
<a id="u-6f8adb23298f0bfc09"></a>

### Разделяем двенадцать строк тремя способами

<!-- /unit:6f8adb23298f0bfc09 -->

<!-- unit:e0a1010795251910ea -->
<a id="u-e0a1010795251910ea"></a>

В toy таблице users A,B,C имеют по записи в dates1,2,3,4. Всего 12 строк. Для простоты labels доступны сразу; delayed outcomes разбираются ниже.

<!-- /unit:e0a1010795251910ea -->

<!-- unit:a9979b1656f43ba245 -->
<a id="u-a9979b1656f43ba245"></a>

* **Новые users в том же условном периоде:** train — все 8 строк A/B, test —4 строки C. Group identity не пересекается, но это не утверждение о переносе на будущую дату.
* **Будущее известных users:** train —6 строк A/B/C за даты 1,2, test —6 строк за даты 3,4. Users повторяются намеренно, потому что такова задача.
* **Новые users в будущем:** train —4 строки A/B за даты 1,2, test —2 строки C за даты 3,4. Остальные 6 строк не обязаны использоваться в этом конкретном experiment.

<!-- /unit:a9979b1656f43ba245 -->

<!-- unit:5d87b587cdd92ac619 -->
<a id="u-5d87b587cdd92ac619"></a>

Последняя схема требует одновременно двух условий. Не нужно насильно включать каждую строку в train/test, нарушая выбранный вопрос. При cold-start отдельно задают, может ли inference использовать доступную прошлую history нового для model user; это не то же самое, что обучаться на его test labels.

<!-- /unit:5d87b587cdd92ac619 -->

<!-- unit:82eb0163072e499e00 -->
<a id="u-82eb0163072e499e00"></a>

### Stratification и groups не заменяют друг друга

<!-- /unit:82eb0163072e499e00 -->

<!-- unit:c76ee2bf502a87ff21 -->
<a id="u-c76ee2bf502a87ff21"></a>

Stratification сохраняет выбранную label composition приблизительно по возможности. Group split препятствует пересечению объявленной identity. Если rare class встречается только у одной group, невозможно одновременно распределить его по всем folds и сохранить эту group целиком. Splitter не создаёт новых независимых редких объектов.

<!-- /unit:c76ee2bf502a87ff21 -->

<!-- unit:b4dda92e7029f67695 -->
<a id="u-b4dda92e7029f67695"></a>

Идентичности могут быть вложенными:frames→video→camera,orders→user→organization,chunks→document→revision family. Если цель —новая камера, группировка только по video может дать завышенный перенос. Если требуется новый video известной камеры, запрет на повторение camera проверит уже более сложный вопрос. Group key выбирается по цели, а не по первой удобной колонке.

<!-- /unit:b4dda92e7029f67695 -->

<!-- unit:f0ef34a63b2d8f8ac7 -->
<a id="u-f0ef34a63b2d8f8ac7"></a>

Exact/near duplicates проверяют на всех declared границах. Удалить ID из input не значит убрать информацию об объекте:фон изображения, текстовый шаблон или соседние события могут сохранять identity. Полезны manifests split IDs и отчёт пересечений, а не только итоговые размеры частей.

<!-- /unit:f0ef34a63b2d8f8ac7 -->

<!-- unit:f8600fc86b7a780319 -->
<a id="u-f8600fc86b7a780319"></a>

## Leakage до модели

<!-- /unit:f8600fc86b7a780319 -->

<!-- unit:e14fadb8a4f0477d24 -->
<a id="u-e14fadb8a4f0477d24"></a>

Обучаемые преобразования — часть pipeline. Среднее для imputation, mean/std scaler, vocabulary, PCA и feature selection должны fit только на training части каждого fold. Даже без использования labels преобразование по всем данным раскрывает информацию о распределении validation/test и нарушает выбранный inductive protocol.

<!-- /unit:e14fadb8a4f0477d24 -->

<!-- unit:d9eac78a5085f42b47 -->
<a id="u-d9eac78a5085f42b47"></a>

Пример: train-значения $(1,3)$, test-значение $100$. Среднее train равно 2, среднее всех равно $104/3\approx34.667$. Заполнение train-пропуска вторым числом использует будущий объект. С label-based feature selection утечка ещё сильнее: среди тысяч случайных признаков можно выбрать те, которые случайно коррелируют с test labels.

<!-- /unit:d9eac78a5085f42b47 -->

<!-- unit:94e12102cba1a75666 -->
<a id="u-94e12102cba1a75666"></a>

Для $K$ folds корректная схема:

<!-- /unit:94e12102cba1a75666 -->

<!-- unit:e347b1f015228581ed -->
<a id="u-e347b1f015228581ed"></a>

$$
T_k=\operatorname{fit\_transform}(D_{-k}),
\quad
f_k=\operatorname{fit}(T_k),
\quad
\widehat y_i=f_k(\operatorname{transform}_k(x_i)),\quad i\in D_k.
$$

<!-- /unit:e347b1f015228581ed -->

<!-- unit:c9a09313e3fd78dab8 -->
<a id="u-c9a09313e3fd78dab8"></a>

Преобразование для validation не refit. Итоговый pipeline после выбора конфигурации можно fit на всём development set, а затем один раз применить к test. [Pipeline и типичные утечки](https://scikit-learn.org/stable/common_pitfalls) описаны в документации sklearn.

<!-- /unit:c9a09313e3fd78dab8 -->

<!-- unit:264af333e8e18d9644 -->
<a id="u-264af333e8e18d9644"></a>

Oversampling/undersampling выполняются внутри train fold. Если сначала размножить positive examples и затем разделить строки, копии попадут по обе стороны. Target encoding требует ещё и out-of-fold values для train: категории рассмотрены в отдельном курсе.

<!-- /unit:264af333e8e18d9644 -->

<!-- unit:809df3bcdf3c40ee4a -->
<a id="u-809df3bcdf3c40ee4a"></a>

### Все learned stages входят в процедуру fit

<!-- /unit:809df3bcdf3c40ee4a -->

<!-- unit:f93193c18b0b3ca883 -->
<a id="u-f93193c18b0b3ca883"></a>

В каждом fold train получает новый imputer/scaler/vocabulary/PCA/feature selector и model. Validation проходит только transform соответствующей fitted pipeline. Если feature selection использовала все labels до CV, последующее честное разделение строк уже не отменяет раскрытую информацию. Если resampling создал synthetic/duplicate samples до split, их родственные records могут оказаться с обеих сторон.

<!-- /unit:f93193c18b0b3ca883 -->

<!-- unit:51e3d30a17dd1087f0 -->
<a id="u-51e3d30a17dd1087f0"></a>

Даже unsupervised fit по validation/test не соответствует обычному inductive protocol. Но transductive setup может явно разрешать доступ к unlabeled target inputs; тогда задача и claim другие, а labels всё равно закрыты. Нельзя молча использовать такую информацию и сравнивать результат с baseline, которому она недоступна. Frozen pretrained encoder, заранее полученный независимо по объявленному protocol, не обязательно обучается заново в каждом fold.

<!-- /unit:51e3d30a17dd1087f0 -->

<!-- unit:55a6ba3818c523236b -->
<a id="u-55a6ba3818c523236b"></a>

Pipeline автоматизирует правильный порядок только для включённых stages. Если перед ним уже заполнены пропуски по полной таблице, learned transformer не находится внутри CV в содержательном смысле. Проверяют data lineage каждой derived field и actual места fit/transform, а не наличие слова Pipeline в коде.

<!-- /unit:55a6ba3818c523236b -->

<!-- unit:9c756b431323257ad1 -->
<a id="u-9c756b431323257ad1"></a>

### Как выглядит OOF prediction

<!-- /unit:9c756b431323257ad1 -->

<!-- unit:66e0da9f30eab27350 -->
<a id="u-66e0da9f30eab27350"></a>

В K-fold development set делится на K validation parts. Для каждой row её out-of-fold prediction создаёт pipeline, не обучавшаяся на этой row и её запрещённых dependencies. Groups/time restrictions при этом сохраняются. Из OOF predictions можно строить diagnostics и выбирать threshold по development protocol, но score после такого выбора уже участвовал в tuning.

<!-- /unit:66e0da9f30eab27350 -->

<!-- unit:02d1c55b91102c40d6 -->
<a id="u-02d1c55b91102c40d6"></a>

Train-only target encoding требует дополнительного cross-fitting внутри training portion:иначе собственный y попадёт в encoding этой же row. Внутренняя логика encoding и внешняя CV —разные уровни. Подробности находятся в categorical-features курсе; здесь важно не считать один внешний split автоматической защитой всех внутренних transformations.

<!-- /unit:02d1c55b91102c40d6 -->

<!-- unit:f77caba0e3eb02f723 -->
<a id="u-f77caba0e3eb02f723"></a>

После выбора configuration final pipeline fit на разрешённом full development set и применяют к закрытому test. OOF models и final model обучены на разном объёме, поэтому OOF score не обещает точное качество именно последнего artifact. Model/template/data/preprocessing revisions сохраняют вместе.

<!-- /unit:f77caba0e3eb02f723 -->

<!-- unit:9e0f5545dd4a63aec4 -->
<a id="u-9e0f5545dd4a63aec4"></a>

### Минимальный audit протокола

<!-- /unit:9e0f5545dd4a63aec4 -->

<!-- unit:b47f3fd4ff35252413 -->
<a id="u-b47f3fd4ff35252413"></a>

Для каждого этапа записывают:inputs,нужны ли labels, какие parameters обучаются и какие IDs участвуют. У validation/test не должно быть вызова fit для learned stages выбранного inductive experiment. Fit статистик по всему train до внутреннего hyperparameter CV тоже может раскрывать inner validation, даже если внешний test остался закрыт.

<!-- /unit:b47f3fd4ff35252413 -->

<!-- unit:93b0bd7558a8d8ff10 -->
<a id="u-93b0bd7558a8d8ff10"></a>

Проверки ID intersections, duplicates и timestamps дополняются inspection transformed examples. Одинаковая shape tensor не доказывает отсутствие будущих данных. Удаление leakage часто снижает score; это улучшение достоверности, а не обязательная деградация настоящего future качества.

<!-- /unit:93b0bd7558a8d8ff10 -->

<!-- unit:becc0bc9716a3ee3b2 -->
<a id="u-becc0bc9716a3ee3b2"></a>

## Время, окна и доступность target

<!-- /unit:becc0bc9716a3ee3b2 -->

<!-- unit:d5f8952fef1c2015bb -->
<a id="u-d5f8952fef1c2015bb"></a>

Walk-forward последовательно обучает на прошлом и проверяет следующий период. Expanding window накапливает прошлое; rolling window ограничивает его давность. Выбор зависит от стабильности процесса и допустимой стоимости переобучения.

<!-- /unit:d5f8952fef1c2015bb -->

<!-- unit:2140e0489d7c0c6b5e -->
<a id="u-2140e0489d7c0c6b5e"></a>

Пусть target использует outcome следующих семи дней. Строка с prediction time 30 сентября имеет label window до 7 октября. Если validation начинается 1 октября, такой train target пересекает его период. Нужен gap/purging по реальной доступности outcome. Размер gap нельзя выбрать только по длине feature window: важны labels, перекрытие объектов и доступность данных. `TimeSeriesSplit(gap=...)` задаёт число исключённых samples; это не автоматически семь календарных дней при нерегулярных записях. [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).

<!-- /unit:2140e0489d7c0c6b5e -->

<!-- unit:5535c0c4e710c1bfea -->
<a id="u-5535c0c4e710c1bfea"></a>

Embargo — дополнительное исключение близкого периода в определённых схемах проверки, например при перекрывающихся финансовых событиях. Это средство контроля конкретной зависимости, а не магическое обязательное число процентов. В паспорте объясняется, какую утечку устраняет выбранный разрыв.

<!-- /unit:5535c0c4e710c1bfea -->

<!-- unit:bc09ac6cdc2c1952a2 -->
<a id="u-bc09ac6cdc2c1952a2"></a>

### Когда метка действительно была доступна

<!-- /unit:bc09ac6cdc2c1952a2 -->

<!-- unit:88afdf2b81d11b41b4 -->
<a id="u-88afdf2b81d11b41b4"></a>

Предположим, обучение выполняется в момент 20. Target использует семь дней после prediction и ещё два дня обработки. При строгом правиле available_at < 20 запись с prediction time 10 готова в момент 19 и разрешена. Запись 11 готова в момент 20 и уже исключена; запись 12 готова в момент 21 и тоже исключена. Равенство границ задают явно: другая временная convention изменит крайнюю допустимую запись.

<!-- /unit:88afdf2b81d11b41b4 -->

<!-- unit:758928e558845a9b22 -->
<a id="u-758928e558845a9b22"></a>

Сам факт, что prediction date прошлая, не означает готовую метку. Прогноз 30 сентября с семидневным outcome до 7 октября нельзя считать полностью размеченным при обучении на 1 октября. Отдельно учитывают задержку самих features, как в главе [«Постановка задачи и паспорт данных»](<df-task.md>). Gap и purging выбирают для реального механизма доступности и пересечения label windows.

<!-- /unit:758928e558845a9b22 -->

<!-- unit:362b9880044bb81eff -->
<a id="u-362b9880044bb81eff"></a>

Перекрытие прошлых feature windows само по себе не всегда leakage. Два predictions могут законно использовать одни и те же прежние события, если они уже известны в обоих моментах. Такая зависимость важна для uncertainty, но не доказывает подсматривание недоступных labels. Не назначают произвольное правило «gap всегда равен длине feature window» без связи с задачей.

<!-- /unit:362b9880044bb81eff -->

<!-- unit:cbf1b6f3e4c70c6112 -->
<a id="u-cbf1b6f3e4c70c6112"></a>

### Expanding, rolling и частота обновления

<!-- /unit:cbf1b6f3e4c70c6112 -->

<!-- unit:4a6fb65c4e3a9ca394 -->
<a id="u-4a6fb65c4e3a9ca394"></a>

Expanding window сохраняет всё больше истории; rolling window использует объявленный последний период. Старые данные могут помогать или устаревать, а ограничение давности меняет training population. Этот выбор проверяют по development protocol, не подгоняя последний holdout.

<!-- /unit:4a6fb65c4e3a9ca394 -->

<!-- unit:b77696da42901e1f7d -->
<a id="u-b77696da42901e1f7d"></a>

Если система обновляется раз в неделю, backtest с retrain перед каждой row даёт другое информационное и вычислительное преимущество. Walk-forward должен воспроизводить допустимое расписание. Validation period можно оценивать одним frozen checkpoint или с объявленными updates, но эти режимы различают в отчёте.

<!-- /unit:b77696da42901e1f7d -->

<!-- unit:7a6114a774e2ef1850 -->
<a id="u-7a6114a774e2ef1850"></a>

Параметр gap в TimeSeriesSplit измеряется в samples, а не в календарных днях. При двух равномерных samples в сутки семь исключённых строк соответствуют 3.5 суткам шага, а не семи. При нерегулярных данных фиксированного перевода в дни вообще нет. Для groups и window labels может требоваться собственный фильтр по ID/времени с проверяемыми invariants.

<!-- /unit:7a6114a774e2ef1850 -->

<!-- unit:e42c699c009896448f -->
<a id="u-e42c699c009896448f"></a>

## Настройка и nested CV

<!-- /unit:e42c699c009896448f -->

<!-- unit:b2fb6b9b8839c0e9b1 -->
<a id="u-b2fb6b9b8839c0e9b1"></a>

При подборе многих конфигураций лучший validation score оптимистичен: часть выигрыша — случайный шум. Nested CV отделяет внутренний подбор от внешней оценки всей процедуры выбора. Во внешнем fold внутренние folds используют только его train. Если вычисления ограничены, разумный development split плюс закрытый test тоже может быть честным; главное — не объявлять score подбора окончательной оценкой.

<!-- /unit:b2fb6b9b8839c0e9b1 -->

<!-- unit:7af7694c86a1654585 -->
<a id="u-7af7694c86a1654585"></a>

Проверяются пересечения IDs, exact/near duplicates, timestamps и target availability. Сам по себе высокий score не доказывает leakage, но неожиданно сильный признак требует расследования. Удаление утечки часто снижает метрику и одновременно повышает достоверность эксперимента.

<!-- /unit:7af7694c86a1654585 -->

<!-- unit:d08e9a9cb6e7e41eaf -->
<a id="u-d08e9a9cb6e7e41eaf"></a>

### Nested CV оценивает процедуру выбора

<!-- /unit:d08e9a9cb6e7e41eaf -->

<!-- unit:de93c57f2179897476 -->
<a id="u-de93c57f2179897476"></a>

Внутри outer train подбирают configuration только по inner folds. Затем выбранную pipeline переобучают на outer train и оценивают на outer test. Каждый outer test не участвует в выборе своей модели. Итоговые outer scores оценивают процедуру training/tuning, а не configuration, которую незаметно выбрали по лучшему outer fold.

<!-- /unit:de93c57f2179897476 -->

<!-- unit:f06fa324576e01a56b -->
<a id="u-f06fa324576e01a56b"></a>

Для трёх outer folds, двух inner folds и четырёх configurations получится 24 inner fits и три outer refits: всего 27. Если затем тот же подбор 2×4 провести на full development и обучить final pipeline, добавятся девять fits, итого 36. Это число обучений для заданного простого grid, а не прогноз wall time. Early stopping, caching и parallel execution имеют свои условия.

<!-- /unit:f06fa324576e01a56b -->

<!-- unit:856f372d7d38d6e156 -->
<a id="u-856f372d7d38d6e156"></a>

Outer training sets пересекаются. Поэтому standard deviation трёх scores не обязательно даёт независимую standard error как sd/√3. Для содержательного вывода учитывают tasks, groups, время и variation. Один удачный fold не объявляют итоговой quality всех будущих популяций.

<!-- /unit:856f372d7d38d6e156 -->

<!-- unit:7f58f9634f4df760fc -->
<a id="u-7f58f9634f4df760fc"></a>

### Threshold, calibration и закрытый test

<!-- /unit:7f58f9634f4df760fc -->

<!-- unit:f1ebab36b8e54a685b -->
<a id="u-f1ebab36b8e54a685b"></a>

Выбор threshold под recall/cost, calibrator, числа features и model family — решения по development data. В nested procedure они входят во внутренний выбор; outer test оценивает уже выбранную процедуру. Нельзя подобрать threshold на outer test labels и назвать полученный там же score независимой проверкой.

<!-- /unit:f1ebab36b8e54a685b -->

<!-- unit:424c53b8e76f184f3a -->
<a id="u-424c53b8e76f184f3a"></a>

Вычислительно доступная альтернатива — разумный development protocol и действительно закрытый final test. Nested CV не обязателен для каждой задачи, но разделение ролей tuning/evaluation необходимо. Результат формулируют по transfer question и описывают все использованные stages, включая preprocessing до модели.

<!-- /unit:424c53b8e76f184f3a -->

<!-- unit:c60e270d2302718146 -->
<a id="u-c60e270d2302718146"></a>

## Контроль и практика

<!-- /unit:c60e270d2302718146 -->

<!-- unit:3fe123033083d20818 -->
<a id="u-3fe123033083d20818"></a>

1. Устранит ли StratifiedKFold совпадение пользователей? **Ответ:** нет; он контролирует labels, а не groups.
2. Почему scaler fit внутри каждого fold? **Ответ:** validation должен преобразовываться статистиками доступной training части.
3. Когда seven-sample gap не равен семи дням? **Ответ:** при нерегулярном времени или нескольких записях за день.

<!-- /unit:3fe123033083d20818 -->

<!-- unit:9609e406476e81e57b -->
<a id="u-9609e406476e81e57b"></a>

4. Проверяет ли random split одновременно новые users и будущее? **Ответ:** не как гарантию; обе границы нужно обеспечить явно.
5. Нужно ли каждую row включать в train/test? **Ответ:** нет, целевой protocol может оставлять неиспользованные records.
6. Спасает ли Pipeline от preprocessing, уже fitted на всех data? **Ответ:** нет, stages нужно поместить в правильную границу fit.
7. Доказывает ли overlap прошлых feature windows утечку? **Ответ:** нет, важно время доступности; зависимость и leakage не тождественны.
8. Что оценивает nested CV? **Ответ:** процедуру выбора и обучения, выполняемую внутри каждого outer train.
9. Независим ли OOF score после выбора threshold? **Ответ:** не полностью, эти predictions уже участвовали в выборе; нужен отдельный evaluation уровень.

<!-- /unit:9609e406476e81e57b -->

<!-- unit:557d8677ebed1e4e56 -->
<a id="u-557d8677ebed1e4e56"></a>

**Практика:** на бумаге разделите 12 строк от трёх пользователей за четыре даты для трёх разных целей: новые пользователи, будущие даты известных пользователей, новые пользователи в будущем. Объясните различия. Затем нарисуйте полный pipeline с местами fit и transform; отдельно укажите, где выбирается threshold и где лежит закрытый test.

<!-- /unit:557d8677ebed1e4e56 -->

## Результат главы и переход дальше

Ответьте на вопрос: **Как разделить данные так, чтобы оценить именно будущий сценарий применения?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

Следующие темы: [Качество данных, пропуски, выбросы и разметка](<df-quality.md>), [Дисбаланс, метрики, порог и калибровка](<df-metrics.md>), [Дубликаты, аугментации и синтетика](<df-augmentation.md>), [Dataset, batching и воспроизводимость](<df-loading.md>), [Drift и надёжность оценки](<df-drift.md>), [Типы признаков и контракты представлений](<../02-features/feat-types.md>), [Числовые преобразования и масштабирование](<../02-features/feat-numeric.md>), [Кардинальность, частотные кодировки и hashing](<../02-features/feat-hashing.md>), [Target statistics, smoothing, cross-fitting и WoE](<../02-features/feat-target.md>), [Обучаемые embeddings и перенос представлений](<../02-features/feat-embeddings.md>), [Временные признаки и доступность метки](<../02-features/feat-time.md>), [Географические признаки и пространственная валидация](<../02-features/feat-geography.md>), [Отбор признаков и устойчивость выбора](<../02-features/feat-selection.md>), [Сокращение размерности: PCA и SVD](<../02-features/feat-reduction.md>), [Единый pipeline и применение к новым объектам](<../02-features/feat-pipeline.md>), [Изображения, аннотации и синхронные преобразования](<../03-modalities/df-images.md>), [Текст, токенизация и разбиение документов](<../03-modalities/df-text.md>), [Обобщение, диагностика и границы теории](<../../02-regularization/reg-diagnosis.md>), [Похожесть, расстояния и постановка кластеризации](<../../03-clustering/cl-geometry.md>), [Encoder, tokenizer, attention и pooling](<../../05-language-models/02-bert-encoders/bert-interface.md>), [Задача адаптации, baseline и выбор вмешательства](<../../05-language-models/03-adaptation/adapt-choice.md>), [RAG: BM25, dense, hybrid и multi-vector retrieval](<../../05-language-models/05-applications/app-rag.md>).

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
