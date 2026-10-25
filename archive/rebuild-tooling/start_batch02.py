"""Record explicit user acceptance of CV and start the requested 20-card batch."""
import json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads((ROOT/p).read_text(encoding='utf8'))
def write(p,v):(ROOT/p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
cards=read('editorial/chapter-reviews.json');cv=[(k,c) for k,c in cards['chapters'].items() if c['path'].startswith('04-computer-vision/')]
assert len(cv)==60
for ident,card in cv:
    card.update(content_acceptance='closed',acceptance_basis='explicit-user-acceptance',accepted_on='2026-10-25',remaining=[])
    card['evidence']=list(dict.fromkeys(card.get('evidence',[])+['editorial/cv-user-acceptance.md']))
cards['checked_on']='2026-10-25';write('editorial/chapter-reviews.json',cards)
course=read('course.json')
ids=[c['id'] for c in course['chapters'] if c['path'].startswith(('05-language-models/03-adaptation/','05-language-models/04-inference/')) and c['id']!='infer-monitoring']
assert len(ids)==20
folder=ROOT/'editorial/batches';folder.mkdir(exist_ok=True)
previous=read('editorial/current-batch.json')
if previous['id']=='lm-batch-01':write('editorial/batches/lm-batch-01.json',previous)
else:assert previous['id']=='lm-batch-02'
write('editorial/current-batch.json',dict(id='lm-batch-02',status='in-progress',started_on='2026-10-25',batch_size=20,chapters=ids,
    acceptance='Содержательная редактура, исходные случаи, предметные схемы, арифметика и общая техническая приёмка.',
    remaining=['Редактура и приёмка 11 глав адаптации и девяти глав инференса.']))
p=ROOT/'editorial/acceptance.md';text=p.read_text(encoding='utf8')
heading='## Решение пользователя по CV — 25 октября 2026'
if heading not in text:
    at=text.index('## Пакет 01')
    text=text[:at]+heading+'\n\nВсе 60 карточек CV закрыты [по прямому указанию пользователя](<cv-user-acceptance.md>). После этого решения закрыты **92 из 168 карточек**, открыты **76**. Последующие пакеты содержат по 20 карточек. Сведения о фактически выполненных проверках и происхождении исходников сохранены.\n\n'+text[at:]
    p.write_text(text,encoding='utf8',newline='\n')
print('CV accepted by user; 92/168 closed; started',ids)
