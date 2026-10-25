# CRNN, CTC и TrOCR: распознавание текста

**Уровень:** основа ветки.

**Предпосылки:** [Классификация, CNN и residual learning](<cv-classification.md>), [Vision Transformer: patches, attention и обучение](<cv-vit.md>)

**Используют:** [Document vision: layout, reading order и таблицы](<cv-documents.md>)

**Учебный вопрос:** Как читать строку без отдельной рамки каждого символа?

## Схема вычислений

![CRNN, CTC и TrOCR: распознавание текста: вычислительные этапы](<../оформление/course/expanded-cv-text-recognition.svg>)

CRNN + CTC: Rectified text crop → CNN → width sequence → BiRNN → T×(alphabet + blank) → CTC alignments → string. TrOCR: Crop → image Transformer → Image states → cross-attention → Causal decoder: BOS → tokens → EOS → decoded text.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Последовательность из изображения

Text recognizer получает rectified crop строки и возвращает последовательность symbols или tokens. Alphabet, пробелы, punctuation и Unicode normalization входят в target contract. Width сохраняет информацию о длине строки; одинаковое растяжение всех crops к квадрату может существенно изменить буквы. Recognition на GT crops и полный OCR с detector — разные протоколы.

## CRNN и CTC

[CRNN](https://arxiv.org/abs/1507.05717) использует CNN для features, переводит spatial width в последовательность и обрабатывает её bidirectional recurrent layers. Каждый timestep выдаёт probabilities symbols и специального blank. CTC, connectionist temporal classification, суммирует probabilities всех alignments, которые после collapse дают target string. Сначала сливают подряд повторённые symbols, затем удаляют blank. Blank между двумя одинаковыми буквами позволяет сохранить обе.

CTC loss $-\log\sum_{\pi:B(\pi)=y}\prod_tp_t(\pi_t)$ вычисляют dynamic programming, а не перебором всех paths. Предполагается framewise factorization probabilities при заданном encoder output. Greedy decoding выбирает максимум каждого шага, но самый вероятный path может не дать самую вероятную итоговую string. Beam search учитывает несколько paths и при необходимости lexical constraints.

## TrOCR: image encoder и autoregressive decoder

[TrOCR](https://arxiv.org/abs/2109.10282) кодирует crop image Transformer и генерирует text tokens decoder с causal self-attention и cross-attention к image features. Для training используют предыдущие истинные tokens и CE следующего token; на inference — уже предсказанные. BOS/EOS задают границы, padding исключается из loss. Decoder может выдавать правдоподобное слово даже при слабом visual evidence, поэтому language fluency не равна точности OCR.

## Разобранный пример: repeated letters и CER

CTC path `l l blank l blank` после слияния соседних повторов даёт `l blank l blank`, после удаления blank — `ll`. Path `l l l blank` даёт только `l`. Для target `ll` нужно минимум три timesteps: две буквы и разделяющий blank. В общем минимальная длина $T\ge U+r$, где $U$ — длина target, $r$ — число соседних одинаковых пар.

GT `кот`, prediction `кит`: одна substitution, CER $1/3\approx0.33333$. Для GT `кот`, prediction `котт`: одна insertion, тоже $1/3$. CER $(S+D+I)/N$ может превышать 1 при большом числе insertions; это не обычная accuracy. Case folding или удаление punctuation меняют задачу и должны быть одинаковы для всех моделей.

## Ошибки и стоимость

CTC ограничен доступными timesteps после CNN downsampling. Длинная строка может оказаться непредставимой. Autoregressive latency растёт с generated length, beam width и decoder cost. Mixed scripts, похожие символы «О/0» и потеря диакритики требуют специальных срезов. Confidence калибруют на уровне нужной единицы: symbol, word или поле документа.

## Самопроверка с ответами

**Практика:** для target `book` вычислите минимальное CTC $T$. Длина 4 плюс повтор `oo`, итого 5.

**Является ли blank пробелом?** Нет: blank — служебное состояние alignment; пробел при необходимости отдельный symbol.

**Нужен ли language-model модуль курса для чтения этой главы?** Нет. Encoder, causal decoding и targets определены локально; более общие углубления можно читать отдельно.

## Иллюстрации из первоисточников

![crnn: Figure 1 из авторской статьи](<../оформление/course/papers/crnn.png>)

**Источник:** [crnn, Figure 1](https://arxiv.org/abs/1507.05717), PDF, с. 2. Проследите вход, преобразования и выход по стрелкам. Сопоставьте обозначения статьи с формами и targets, определёнными в тексте главы.

![trocr: Figure 1 из авторской статьи](<../оформление/course/papers/trocr.png>)

**Источник:** [trocr, Figure 1](https://arxiv.org/abs/2109.10282), PDF, с. 2. Проследите вход, преобразования и выход по стрелкам. Сопоставьте обозначения статьи с формами и targets, определёнными в тексте главы.

## Источники и продолжение

- [CRNN — Shi et al., TPAMI 2017](https://arxiv.org/abs/1507.05717)
- [TrOCR — Li et al., AAAI 2023](https://arxiv.org/abs/2109.10282)

[Следующие углубления: cv-documents](<README.md>) · [Сценарии подраздела](<07-specialized/План.md>)
