# Optical flow и RAFT: плотное движение

**Уровень:** углубление.

**Предпосылки:** [Изображение, карты признаков и блоки модели](<cv-input.md>), [Классификация, CNN и residual learning](<cv-classification.md>)

**Используют:** Самостоятельное применение выбранной ветки.

**Учебный вопрос:** Как RAFT находит перемещение каждого пикселя между двумя кадрами?

## Схема вычислений

![Optical flow и RAFT: плотное движение: вычислительные этапы](<../оформление/course/expanded-cv-flow.svg>)

RAFT features: Два кадра I1, I2 → Feature encoders → F1,F2 → All-pairs correlation + pyramid → Context encoder → hidden state. Iterative update: Flow estimate f → Lookup around correspondences → Recurrent block → Δf → f + Δf → learned upsampling.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Correspondence как поле смещений

Optical flow $f(x,y)=(u,v)$ указывает, куда видимый пиксель первого кадра переместился во втором: $(x+u,y+v)$. Единицы — pixels за интервал между кадрами. Это не 3D velocity: проекция камеры смешивает глубину, движение объекта и камеры. Tracking связывает object IDs, flow оценивает плотные correspondences и не создаёт identity сам по себе.

Brightness constancy мотивирует классические методы, но нарушается бликами, тенями и occlusion. Обучаемые features позволяют сопоставлять более сложные наблюдения; неоднозначность однотонной области при этом не исчезает.

## RAFT: correlation и recurrent refinement

[RAFT](https://arxiv.org/abs/2003.12039) вычисляет features обоих изображений на уменьшенной сетке, затем all-pairs correlation. Для $F_1,F_2\in\mathbb R^{H_f\times W_f\times D}$ значение $C(i,j,k,l)$ — dot product features позиций $(i,j)$ и $(k,l)$. Correlation pyramid агрегирует последние пространственные оси. Context encoder первого кадра подготавливает hidden state и context для recurrent update.

Текущее flow задаёт ожидаемую позицию correspondence; lookup читает локальные окрестности correlation на нескольких уровнях. GRU-подобный update block предсказывает поправку $\Delta f$, после чего $f\leftarrow f+\Delta f$. Features и correlation не пересчитываются с нуля на каждой итерации. Learned upsampling возвращает full-resolution field.

## Loss и допустимые pixels

Supervised обучение сравнивает последовательность flow estimates с GT, например взвешенной суммой L1 errors, исключая invalid targets. Для unsupervised pipeline нужны отдельные photometric/reprojection assumptions, occlusion masks и regularization; это не автоматически исходный supervised RAFT. На inference GT отсутствует, число iterations влияет на стоимость и иногда качество.

## Разобранный пример: endpoint error и память

Истинное flow в pixel $(3,4)$, prediction $(1,5)$. Endpoint error, EPE, равна $\sqrt{(1-3)^2+(5-4)^2}=\sqrt5\approx2.23607$ pixels. Для трёх valid pixels с EPE 1,2,3 mean EPE равна 2. Если четвёртый pixel invalid, он не должен добавлять нулевую ошибку и уменьшать среднее до 1.5.

На входе $512\times512$ с feature stride 8 получаем $64\times64=4096$ positions. All-pairs volume содержит $4096^2=16777216$ scalars, 64 MiB float32 до pyramid и gradient buffers. При удвоении обеих сторон входа positions вырастут вчетверо, correlation — в 16 раз. Это важный предел даже при малом recurrent block.

Для prediction low-resolution flow $(2,-1)$ в units feature cells stride 8 full-image displacement соответствует $(16,-8)$ pixels. Resize grid без корректного масштабирования vectors даёт геометрически неверный ответ.

## Диагностика

Оценку разбивают по occluded/non-occluded, motion magnitude и boundaries. Forward-backward consistency обнаруживает часть проблем, но согласованные ошибки могут пройти проверку. Flow не следует использовать как буквальное измерение физической скорости без калибровки и времени между кадрами.

## Самопроверка с ответами

**Практика:** увеличьте интервал между кадрами вдвое для равномерного движения. Pixel displacement примерно удвоится, но соответствия могут стать труднее из-за окклюзии и выхода из кадра.

**Почему flow не задан надёжно для исчезнувшего pixel?** Во втором кадре может не быть видимого соответствия.

**Равен ли correlation volume вероятности?** Нет. Это scores; их интерпретация определяется последующей сетью.

## Иллюстрации из первоисточников

![raft: Figure 1 из авторской статьи](<../оформление/course/papers/raft.png>)

**Источник:** [raft, Figure 1](https://arxiv.org/abs/2003.12039), PDF, с. 2. Проследите вход, преобразования и выход по стрелкам. Сопоставьте обозначения статьи с формами и targets, определёнными в тексте главы.

## Источники и продолжение

- [RAFT — Teed, Deng, ECCV 2020](https://arxiv.org/abs/2003.12039)

[Выбрать следующую ветку](<README.md>) · [Сценарии подраздела](<05-video/План.md>)
