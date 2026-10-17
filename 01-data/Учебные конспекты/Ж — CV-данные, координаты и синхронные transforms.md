# Ж. Почему правильная картинка может иметь неправильную аннотацию

[Оглавление](README.md). Пункт 7 плана. **Основа:** S7:1618–3712 — detection/segmentation, S7:3713–5276 — pose/OBB, S7:5277–6186 — tracking. Здесь собрана data-сторона этих задач; дополнены конвертации и streaming. [CV-конспекты](../../05-computer-vision/Учебные%20конспекты/README.md) раскрывают модели и метрики.

## 1. Формат координат — часть смысла

Bbox может задаваться углами $(x_1,y_1,x_2,y_2)$ либо origin/size $(x,y,w,h)$ либо centre/size $(c_x,c_y,w,h)$. Числа могут быть pixels или normalized fractions. Название «xywh» без origin convention недостаточно. COCO обычно хранит top-left и размеры, YOLO labels используют normalized centre/size; конкретный exporter и version всё равно проверяются на sample.

Для image шириной $W$ и высотой $H$:

$$
c_x=\frac{x_1+x_2}{2W},\quad
c_y=\frac{y_1+y_2}{2H},\quad
w=\frac{x_2-x_1}{W},\quad
h=\frac{y_2-y_1}{H}.
$$

При $W=200,H=100$ и box $(20,10,80,50)$ получаем $(0.25,0.30,0.30,0.40)$. Обратное преобразование должно вернуть исходную box с допустимой rounding error. В formulas предполагается continuous/pixel-edge convention; inclusive integer endpoints с добавлением единицы — другой контракт, который нельзя смешивать.

Class IDs тоже требуют mapping: contiguous model IDs могут отличаться от исходных dataset IDs; фон может быть отдельным классом или неявным. `ignore_index` исключает label из выбранного loss/metric, а не обозначает фон. Ошибка на единицу способна сделать весь training бессмысленным при внешне корректных масках.

## 2. Геометрическое преобразование всех labels

Resize с коэффициентами $s_x,s_y$ умножает x/y coordinates на соответствующий scale. Crop с левым верхним углом $(u,v)$ вычитает offsets и clips координаты по новой области. Если объект исчез, его label удаляется или отмечается по явному правилу visibility. Нельзя сохранять box с отрицательной площадью.

При horizontal flip continuous box преобразуется как:

$$
(x_1,x_2)\mapsto(W-x_2,W-x_1).
$$

Для discrete pixel-centre keypoints часто используется $x\mapsto W-1-x$; это другая convention. Left/right keypoint IDs меняются местами. Visibility flags обновляются при crop, а не остаются механически прежними.

Дискретную semantic mask масштабируют nearest-neighbor, чтобы не создавать дробные class IDs. Image допускает bilinear interpolation, но mask с classes 0 и 2 не должна получить новый class 1 из усреднения. Instance IDs также не смешиваются интерполяцией. Для depth map, continuous heatmap и probability map правила иные: тип target определяет подходящий transform.

## 3. Pose, OBB и неполнота annotations

OBB хранит rotation: radians/degrees, диапазон угла, направление осей и порядок width/height должны совпадать с model/loss. Rectangle с углом $\theta+\pi$ геометрически эквивалентен исходному; swap width/height может соответствовать повороту на $\pi/2$. Round-trip test лучше проверяет corners/IoU, чем raw angle equality.

В pose visibility «размечен, но закрыт» отличается от «не размечен»; evaluator может учитывать их по-разному. В detection пустое изображение без объектов — настоящий negative, а изображение с неразмеченными объектами — incomplete annotation. Если неразмеченные реальные objects считаются background, модель получает противоречивый сигнал.

Crowd, ignore regions, маленькие объекты и occlusion требуют явной политики. Редкий small object легко исчезает после resize; поэтому аудит должен включать size distribution до/после transforms. Полнота annotations важна не меньше точности boxes.

## 4. Видео и шардирование

Соседние кадры очень похожи. Random frame split часто оценивает запоминание сцены; для новых видео делить нужно sequences, а при необходимости камеры/локации/время. Tracking IDs уникальны в пределах объявленной sequence, и их нельзя смешивать между видео без namespace.

Шарды и WebDataset-подобные форматы полезны для больших потоков. Но shuffle ограниченного buffer не равен полному случайному перемешиванию. Group/source split делают до упаковки либо сохраняют надежные metadata. При restart проверяют replay/skip, counts и распределение classes по shards. Формат хранения не решает проблему leakage автоматически.

## 5. Контроль и практика

1. Почему nearest нужен для class masks? **Ответ:** он сохраняет дискретные labels, тогда как усреднение может создать несуществующий класс.
2. Можно ли random split кадров назвать проверкой на новых видео? **Ответ:** нет; связанные кадры одной sequence должны оставаться в одной части.
3. Почему одинаковая OBB может иметь разные углы? **Ответ:** существуют эквивалентные parametrizations с периодичностью и swap сторон.

**Практика:** возьмите пять изображений с empty case, small object, crowd, keypoints и rotated box. Выполните round-trip координат и transforms, визуально наложите labels. Проверьте positive areas, ID mapping, masks, visibility и неизменность классов; отдельно протестируйте границы и исчезнувший после crop объект.
