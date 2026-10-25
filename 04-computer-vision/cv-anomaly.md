# Anomaly detection: reconstruction, PaDiM и PatchCore

**Уровень:** основа ветки.

**Предпосылки:** [Обучение и перенос: frozen features, fine-tuning и аугментации](<cv-transfer.md>)

**Используют:** [Сквозные разборы: дефекты, видеопоток и документы](<cv-cases.md>)

**Учебный вопрос:** Как искать дефекты, если при обучении есть преимущественно нормальные изделия?

## Схема вычислений

![Anomaly detection: reconstruction, PaDiM и PatchCore: вычислительные этапы](<../оформление/course/expanded-cv-anomaly.svg>)

PaDiM: Normal images → frozen features → По позиции: mean / covariance → Test features → Mahalanobis → Anomaly map → image decision. PatchCore: Normal patches → memory bank → Representative coreset → Test patch → nearest neighbours → Distances → map / image score.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Норма и anomaly score

One-class anomaly detection строит модель нормальных observations; отклонение получает большой score. Это не автоматически classifier всех видов брака. Новая нормальная текстура может быть anomaly для обученной модели, а визуально похожий дефект — остаться незамеченным. Image-level задача отвечает, есть ли подозрение на кадре; pixel-level localization показывает область.

## Reconstruction и PaDiM

Autoencoder обучается восстанавливать normal images. Reconstruction error используется как anomaly map, но expressive decoder может хорошо восстановить и дефект; noise или registration error могут дать большой residual без брака. Требуется validation обоих случаев.

[PaDiM](https://arxiv.org/abs/2011.08785) извлекает pretrained patch features нескольких уровней. Для каждой spatial position по normal train images оцениваются mean $\mu$ и covariance $\Sigma$. Score — Mahalanobis distance $s=\sqrt{(z-\mu)^T\Sigma^{-1}(z-\mu)}$, обычно с regularization covariance. Позиционная модель чувствительна к выравниванию изделий; covariance плохо оценивается при слишком большом feature dimension и малом числе изображений.

## PatchCore

[PatchCore](https://arxiv.org/abs/2106.08265) сохраняет memory bank normal patch features, сокращая его representative coreset. Test patch сравнивается с ближайшими normal vectors; большие distances указывают на необычные features. Coreset уменьшает search cost и memory, но может удалить редкую нормальную структуру. Backbone не обязательно дообучается: основной fit состоит в построении normal representation и memory bank.

Image score агрегирует patch scores, например максимум или специальную reweighting policy. Максимум способен расти с числом patches даже при прежнем уровне noise, поэтому изменение resolution требует повторной проверки thresholds. Anomaly map upsampling не создаёт точные границы дефекта из грубых features.

## Разобранный пример: covariance и порог

Mean normal feature $(1,2)$, diagonal covariance $\operatorname{diag}(4,1)$, test feature $(3,3)$. Difference $(2,1)$, squared Mahalanobis distance $2^2/4+1^2/1=2$, score $\sqrt2\approx1.41421$. Euclidean distance $\sqrt5$ не учитывает, что первая координата в норме более изменчива.

Normal memory vectors $(0,0),(1,0),(0,1)$, test patch $(2,0)$. Nearest distance равна 1 до $(1,0)$. После удаления этого vector coreset distance станет 2. Score вырос без изменения test image: compression bank является изменением модели и требует validation.

На normal validation из 1000 независимых изделий порог оставляет 10 false alarms: FPR 1%. При потоке 100000 normal изделий в день это около 1000 alarm/day при сохранении distribution. Высокий AUROC не показывает приемлемость этой нагрузки.

## Самопроверка с ответами

**Практика:** сдвиньте нормальное изделие в кадре. У PaDiM position-specific distribution может дать anomaly; проверяют alignment или представление, устойчивое к допустимому сдвигу.

**Можно ли выбрать порог на test defects?** Нет. Порог относится к настройке, test нужен для независимой оценки.

**Почему pixel AUROC может выглядеть высоким?** Большой background способен скрыть пропуск маленьких defect regions; нужны defect-level и region metrics.

## Иллюстрации из первоисточников

![padim: Figure 2 из авторской статьи](<../оформление/course/papers/padim.png>)

**Источник:** [padim, Figure 2](https://arxiv.org/abs/2011.08785), PDF, с. 2. Проследите вход, преобразования и выход по стрелкам. Сопоставьте обозначения статьи с формами и targets, определёнными в тексте главы.

![patchcore: Figure 2 из авторской статьи](<../оформление/course/papers/patchcore.png>)

**Источник:** [patchcore, Figure 2](https://arxiv.org/abs/2106.08265), PDF, с. 4. Проследите вход, преобразования и выход по стрелкам. Сопоставьте обозначения статьи с формами и targets, определёнными в тексте главы.

## Источники и продолжение

- [PaDiM — Defard et al., ICPR workshop 2020](https://arxiv.org/abs/2011.08785)
- [PatchCore — Roth et al., CVPR 2022](https://arxiv.org/abs/2106.08265)

[Следующие углубления: cv-cases](<README.md>) · [Сценарии подраздела](<07-specialized/План.md>)
