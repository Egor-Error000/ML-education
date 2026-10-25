"""First-use explanations from an explicitly authored glossary, outside code/math."""
from pathlib import Path
import json,re,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools'))
from edit_chapter import load,save,transform
TERMS={
'hidden state':'скрытое состояние позиции','token embedding':'входное векторное представление токена',
'embedding':'векторное представление','embeddings':'векторные представления',
'encoder':'кодировщик входа','decoder':'декодер','backbone':'основная сеть извлечения признаков',
'head':'выходная голова задачи','pooling':'агрегирование состояний','mean pooling':'усреднение выбранных состояний',
'logits':'логиты, оценки альтернатив до нормировки','scores':'оценки','score':'оценка',
'baseline':'базовый вариант сравнения','checkpoint':'сохранённое состояние модели',
'objective':'оптимизируемая целевая функция','loss':'функция потерь','loss mask':'маска позиций, включённых в ошибку',
'targets':'целевые значения','target':'целевая величина','labels':'метки','label':'метка',
'features':'признаки','feature':'признак','weights':'веса модели','parameters':'параметры',
'batch':'пакет примеров','batch size':'число примеров в пакете','padding':'дополнение до общей длины',
'tokenizer':'токенизатор','token':'токен','tokens':'токены','attention':'механизм внимания',
'self-attention':'самовнимание','cross-attention':'перекрёстное внимание',
'residual':'остаточная связь или её путь','gradient':'градиент','forward':'прямой проход вычисления',
'backward':'обратный проход вычисления производных','optimizer':'оптимизатор обновления параметров',
'learning rate':'шаг обучения','dropout':'случайное выключение выбранных активаций при обучении',
'weight decay':'затухание весов при обновлении','weight tying':'совместное использование одного массива параметров',
'validation':'проверочная часть для выбора решения','holdout':'отложенная часть данных','train':'обучающая часть данных',
'test':'тестовая часть независимой оценки','split':'разбиение данных','leakage':'утечка недоступной информации',
'inference':'применение обученной модели','fine-tuning':'дообучение модели','pretraining':'предобучение',
'retrieval':'поиск кандидатов','reranking':'повторное ранжирование кандидатов','retriever':'компонент поиска',
'generator':'генератор ответа','query':'запрос','document':'документ','context':'доступный модели контекст',
'prompt':'входной запрос или инструкция','template':'шаблон сериализации входа','prefix':'уже доступный префикс',
'prefill':'обработка начального контекста','decode':'вычисление очередного шага продолжения',
'serving':'обслуживание запросов к модели','latency':'задержка отдельного запроса',
'throughput':'число обработанных единиц за время','runtime':'среда исполнения модели',
'cache':'кеш повторно используемых вычислений','sampling':'случайный выбор по распределению',
'greedy':'выбор максимальной оценки на каждом шаге','temperature':'температура распределения выбора',
'out-of-sample':'применение к объектам вне обучающей выборки','drift':'изменение распределения или зависимости со временем',
'confidence':'уверенность по принятой оценке','calibration':'согласование вероятностных оценок с частотами',
'seed':'начальное состояние генератора случайных чисел','ablation':'сравнение с удалённой или изменённой частью метода',
'teacher':'модель-учитель','student':'модель-ученик','soft targets':'мягкие целевые распределения',
'reward':'награда за действие или результат','policy':'политика выбора действий','trajectory':'траектория действий и наблюдений',
'observation':'доступное наблюдение среды','environment':'среда взаимодействия','state':'состояние системы',
'orchestration':'управление порядком действий и состоянием','tool calling':'предложение вызова инструмента',
'tool':'инструмент приложения','schema':'схема допустимых данных','structured output':'ответ заданной структуры',
'distillation':'дистилляция, обучение по сигналу модели-учителя','quantization':'квантование числового представления',
'pruning':'удаление параметров или структур модели','sparsity':'разреженность, доля нулевых или исключённых элементов',
'normalization':'нормировка по выбранному правилу','augmentation':'преобразование обучающего примера с согласованной целью',
'bounding box':'ограничивающая рамка объекта','feature map':'пространственная карта признаков','patch':'фрагмент изображения',
'receptive field':'область входа, влияющая на элемент карты','stride':'шаг пространственного перемещения операции',
'upsampling':'повышение пространственного разрешения','downsampling':'уменьшение пространственного разрешения',
'ground truth':'эталонная разметка','false positive':'ложное срабатывание','false negative':'пропущенный положительный случай',
}
for ambiguous in ['head','decode','target','residual','policy','train','test','generator','weights','calibration']:TERMS.pop(ambiguous,None)
PROTECTED=r'(```[\s\S]*?```|\$\$[\s\S]*?\$\$|\$[^$\n]+\$|`[^`\n]+`|!?\[[^\]]*\]\((?:<[^>]+>|[^\n)]+)\)|<!--.*?-->)'
PATTERN=re.compile(r'(?<![\w-])('+'|'.join(re.escape(k) for k in sorted(TERMS,key=len,reverse=True))+r')(?![\w-])',re.I)
def explain(text):
    boundary=text.find('\n## ')
    if boundary<0:return text
    prefix=text[:boundary];parts=re.split(PROTECTED,text[boundary:],flags=re.S);seen=set()
    for index in range(0,len(parts),2):
        original=parts[index]
        def term(m):
            key=m[0].lower()
            if key in seen:return m[0]
            seen.add(key)
            local=original[max(0,m.start()-110):m.end()+110].lower()
            # Existing Russian definitions take precedence over annotation.
            stem=TERMS[key].split(',')[0].split()[0][:7]
            if stem in local or original[m.end():].lstrip().startswith('('):return m[0]
            return m[0]+' ('+TERMS[key]+')'
        parts[index]=PATTERN.sub(term,original)
    return prefix+''.join(parts)
def main():
    cov=load();course=json.loads((ROOT/'course.json').read_text(encoding='utf8'))
    # Fully rewritten chapters already introduce their vocabulary in Russian.
    completed={'lm-attention','lm-tokenizer','lm-states','lm-output','lm-bert','bert-interface'}
    for c in course['chapters']:
        if c['module']=='cluster' or c['id'] in completed:continue
        text=(ROOT/c['path']).read_text(encoding='utf8')
        if explain(text)==text:continue
        transform(c['path'],explain,'Добавлено русское пояснение первого употребления технического термина; формулы, код и ссылки не менялись. Требуется проверка естественности формулировок.',cov);save(cov)
    (ROOT/'editorial/terminology.json').write_text(json.dumps(TERMS,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
if __name__=='__main__':main()
