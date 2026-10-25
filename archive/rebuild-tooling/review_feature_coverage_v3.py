"""Explicit review of all eleven original numeric/selection source units."""
import sys,json,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from edit_chapter import ROOT,load,save,sections,digest,transform
cov=load()
rel='05-language-models/01-foundations/lm-transformer.md'
needle='Только последний вектор имеет ширину 2 и совместим с остаточной суммой.'
def insert(s):
    return s.replace(needle,'![Две ветви управляющей MLP с числовым примером](<../../оформление/course/lm-gated-mlp.svg>)\n\n'+needle)
if 'lm-gated-mlp.svg' not in (ROOT/rel).read_text(encoding='utf8'):
    transform(rel,insert,'Добавлена схема ветвей up/gate, произведения и возврата к D с промежуточными числами.',cov);save(cov)
destinations={
'bba7f85413a23ece40':['Единицы измерения задают геометрию'],
'd97f5177fc85149485':['Обучаемые преобразования и их состояние','Схема применения и проверка'],
'a0ba40e71b198eca23':['Новый объект не участвует в обучении масштаба'],
'fc4dd48f324471f27e':['Единицы измерения задают геометрию'],
'68261fded6b9cbabc4':['Нелинейные преобразования, интервалы и пропуски','Единицы измерения задают геометрию','Схема применения и проверка'],
'328ded2644135d9f1c':['Что выбирает процедура отбора'],
'482d8fd5d744b1acff':['Что выбирает процедура отбора','Взаимозаменяемость и взаимодействия'],
'3a11efce44bf36bcbb':['Оценки полей входят в обученное состояние','Что выбирает процедура отбора','Как оценить выбор числа признаков'],
'dfee709de9545f0823':['Бюджет поиска и генерация кандидатов'],
'39b3c45faf47307fdb':['Взаимозаменяемость и взаимодействия','Бюджет поиска и генерация кандидатов'],
'180d1aebca1b3aeb4a':['Схема и предметная проверка']}
for u in cov['units']:
    if u['id'] not in destinations:continue
    path=u['target'];parts=sections((ROOT/path).read_text(encoding='utf8'));targets=[]
    for i,title in enumerate(destinations[u['id']]):
        key=u['anchor'] if i==0 else f"t-{u['chapter']}-{u['id']}-{i}"
        a,b,body=parts[title]
        cov['spans'][key]={'path':path,'start_line':a,'end_line':b,'sha256':digest(body),'destination_section':title,
            'status':'revised','review_status':'verified',
            'rationale':'Исходная единица прочитана и сопоставлена с объяснением; повтор объединён, разные случаи и ограничения сохранены. Для IQR исправлено ошибочное обобщение нулевого знаменателя на постоянное поле.',
            'evidence':'editorial/review-evidence.md#numeric-selection'}
        targets.append(key)
    u['targets']=targets;u['status']='semantic-reviewed';u['action']='объединить с сохранением нюансов'
save(cov)
# Registry contains actual local SVGs; PNG licence entries remain in illustrations.json.
p=ROOT/'course.json';course=json.loads(p.read_text(encoding='utf8'))
for ch in course['chapters']:
    text=(ROOT/ch['path']).read_text(encoding='utf8')
    refs=re.findall(r'!\[[^\]]*\]\(<([^>]+\.svg)>\)',text)
    ch['figures']=list(dict.fromkeys((Path(ch['path']).parent/ref).resolve().relative_to(ROOT).as_posix() for ref in refs))
p.write_text(json.dumps(course,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
print('Eleven source units reviewed; explicit many-to-many destinations retained.')
