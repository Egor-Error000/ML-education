"""Record the explicit authored review; build/tests never invoke this script."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools'))
from edit_chapter import load,save,digest
from editorial_io import verify_span
def read(p):return json.loads((ROOT/p).read_text(encoding='utf8'))
def write(p,o):(ROOT/p).write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
expected={'adapt-choice':52,'adapt-sft':58,'adapt-regimes':0,'adapt-lora':80,'adapt-variants':100,
 'adapt-alternatives':54,'adapt-preference':69,'adapt-merging':54,'adapt-distillation':59,
 'adapt-continual':46,'adapt-evaluation':49,'infer-generation':3,'infer-kv':63,'infer-context':73,
 'infer-decoding':79,'infer-serving':88,'infer-speculative':89,'infer-quantization':119,'infer-pruning':63,'infer-deployment':114}
batch=read('editorial/current-batch.json');assert batch['id']=='lm-batch-02' and set(batch['chapters'])==set(expected)
review=(ROOT/'editorial/batch02-review.md').read_text(encoding='utf8');cov=load();cards=read('editorial/chapter-reviews.json')
for ident,count in expected.items():
 assert f'id="{ident}"' in review
 card=cards['chapters'][ident];text=(ROOT/card['path']).read_text(encoding='utf8')
 units=[u for u in cov['units'] if u.get('chapter')==ident and not u.get('provenance_only')]
 assert len(units)==count
 evidence='editorial/batch02-review.md#'+ident
 for u in units:
  assert u['targets']
  for key in u['targets']:
   span=cov['spans'][key];assert span['path']==card['path'] and verify_span(text,span),(ident,key)
   span.update(review_status='verified',evidence=evidence,reviewed_on='2026-10-25',
    rationale='Исходный случай и назначенный раздел прочитаны; предпосылки, выводы, расчёты и исправления указаны в предметном свидетельстве пакета 02.')
  u.update(status='semantic-reviewed',review_evidence=evidence)
 card.update(text_sha256=digest(text),prose='rewritten',source_units=count,source_units_pending_semantic_review=0,
  content_acceptance='closed',acceptance_basis='authored-content-review',evidence=[evidence],remaining=[],reviewed_on='2026-10-25')
save(cov);cards['checked_on']='2026-10-25';write('editorial/chapter-reviews.json',cards)
closed=sum(c['content_acceptance']=='closed' for c in cards['chapters'].values());assert closed==112
batch.update(status='content-reviewed-validation-pending',reviewed_on='2026-10-25',source_units=sum(expected.values()),
 review='editorial/batch02-review.md',remaining=['Итоговые отчёты текущего HTML: аудит, браузер и чистый LF/CRLF-клон Windows.'])
write('editorial/current-batch.json',batch)
print('20 cards reviewed; 1312 source units; 112/168 closed, 56 open; final validation pending')
