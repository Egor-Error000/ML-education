# Кластеризация: подробные учебные материалы

[Версия для чтения с набранными формулами](<../../Читать конспекты.html#course-4-0>) · работает без интернета.

[Маршрут и сценарии модуля](<../../План — кластеризация.md>) · [Исходный план](<../План — кластеризация.md>) · [Общее оглавление](<../../Учебные конспекты.md>)

Девять глав соответствуют девяти пунктам. Рекомендуемый порядок сохраняет маршрут плана: **А → З → Б → Г → В → Д → нужная доменная ветка → И**. Полного отдельного чата по clustering нет: курс является дополнением, с явно указанными связями с embeddings/CV/retrieval из S2/S6/S7.

1. [А — Похожесть и geometry](<А — Похожесть, расстояния и постановка кластеризации.md>). Пункт 1: задача, class/cluster, Euclidean/Manhattan/cosine/Mahalanobis, scale и unit-vector identity.
2. [Б — Основные algorithms](<Б — K-Means, иерархии, плотность и GMM.md>). Пункт 2: два шага K-Means/++, linkage/Ward, DBSCAN/HDBSCAN, полный EM GMM и covariance/AIC/BIC.
3. [В — Выбор K](<В — Выбор K, gap statistic и устойчивость.md>). Пункт 3: inertia/elbow, silhouette, gap/reference, DP mixtures и stability.
4. [Г — Качество partitions](<Г — Purity, ARI, NMI и внутренние метрики.md>). Пункт 4: permutation, fragmentation, external/internal measures, noise policy и coverage.
5. [Д — Deep clustering](<Д — Frozen embeddings, DEC и IDEC.md>). Пункт 5: frozen pipeline, Student-t/target/KL, refinement, reconstruction и независимая оценка.
6. [Е — CV](<Е — Image clustering, superpixels и поток новых объектов.md>). Пункт 6: backbones/SSL, pixel features, SLIC, unsupervised regions и dynamic/streaming groups.
7. [Ж — NLP/topics](<Ж — Текстовые representations, LDA, NMF и Top2Vec.md>). Пункт 7: lexical/semantic alternatives, topic mixtures, NMF, Top2Vec и short-text hybrids.
8. [З — Dimension reduction](<З — PCA, t-SNE, UMAP и геометрия проекции.md>). Пункт 8: PCA/SVD/whitening, visualization distortion, preprocessing и проверка исходных neighbours.
9. [И — Компонент системы](<И — Аномалии, псевдометки, IVF и итоговый pipeline.md>). Пункт 9: anomalies/regimes, pseudo-labels, ANN routing, out-of-sample и итоговый report.

## Проверка глубины и практика плана

Семь практических веток покрыты: geometry — А/З; ручной K-Means — Б; 2D GMM/EM — Б; core/border/noise и density sweep — Б; purity/ARI/NMI/DB/CH — Г; seeds/subsampling — В; fixed encoder против fixed algorithm — Д. Числовые примеры оформлены LaTeX, каждая глава содержит ошибки, контрольные вопросы с ответами и practice.

Итог — один reproducible experiment с [паспортом/split](../../01-data/Учебные%20конспекты/README.md), representation config, метриками/coverage, stability, runtime/memory и типичными/пограничными cases. Все выводы привязаны к задаче и geometry; красивое 2D-разделение само по себе не объявляется доказательством natural taxonomy.

Связанные подробные курсы: [CV](../../05-computer-vision/Учебные%20конспекты/README.md), [RAG и retrieval](../../06-llm/04-applications/Учебные%20конспекты/README.md). Упоминания проектов в историческом плане не засчитываются как выполненные эксперименты; новое изучение/практика подтверждаются отдельно.
