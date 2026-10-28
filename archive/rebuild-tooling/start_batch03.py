"""Register an in-progress batch without changing completed review claims."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads((ROOT/p).read_text(encoding='utf8'))
def write(p,v):(ROOT/p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
old=read('editorial/current-batch.json');assert old['id']=='lm-batch-02' and old['status']=='complete'
cards=read('editorial/chapter-reviews.json');course=read('course.json');by={c['id']:c for c in course['chapters']}
# A separately authored chapter was already present in the working tree. Give
# it an open review card without altering its text or asserting acceptance.
for ident,c in by.items():
 if ident not in cards['chapters']:
  text=(ROOT/c['path']).read_text(encoding='utf8')
  cards['chapters'][ident]={'path':c['path'],'text_sha256':hashlib.sha256(text.encode('utf8')).hexdigest(),
   'scenario':'authored' if ident in read('editorial/chapter-scenarios.json')['chapters'] else 'missing',
   'prose':'existing-unreviewed','source_units':0,'source_units_pending_semantic_review':0,
   'content_acceptance':'open','evidence':[],'remaining':['Прочитать главу и принять её содержание отдельно.']}
write('editorial/chapter-reviews.json',cards)
ids=['df-task','df-split','df-quality','df-metrics','df-augmentation','df-loading','df-drift',
 'feat-types','feat-missing','feat-categorical','feat-hashing','feat-target','feat-embeddings',
 'feat-native','feat-interactions','feat-sets','feat-time','feat-geography','feat-reduction','feat-autofe']
assert all(x in by and cards['chapters'][x]['content_acceptance']=='open' for x in ids)
write('editorial/current-batch.json',{'id':'data-batch-03','status':'in-progress','started_on':'2026-10-28',
 'batch_size':len(ids),'chapters':ids,'plan':'editorial/batch03-plan.md',
 'remaining':['Содержательная редактура, сверка исходных единиц, схемы и приёмка партии.']})
print('20 planned; current course',len(by),'chapters;',len(cards['chapters']),'review cards.')
