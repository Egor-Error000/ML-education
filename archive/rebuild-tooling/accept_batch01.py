"""Explicitly record the completed manual review of the ten-card batch.

Not called by the build. Does not infer acceptance from hashes or tests.
Only the stated cards and their referenced spans are accepted here.
"""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from edit_chapter import load,save,digest
from editorial_io import verify_span
def read(rel):return json.loads((ROOT/rel).read_text(encoding='utf8'))
def write(rel,value):(ROOT/rel).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
expected={'lm-tokenizer':7,'lm-states':19,'lm-position':47,'bert-adaptation':76,
          'bert-tasks':96,'bert-embeddings':83,'bert-retrieval':89,'bert-domains':69,
          'bert-evaluation':80,'bert-operations':79}
batch=read('editorial/current-batch.json')
assert batch['id']=='lm-batch-01' and set(batch['chapters'])==set(expected)
review=(ROOT/'editorial/batch01-review.md').read_text(encoding='utf8')
c=load();cards=read('editorial/chapter-reviews.json')
for ident,count in expected.items():
    assert f'id="{ident}"' in review
    card=cards['chapters'][ident];text=(ROOT/card['path']).read_text(encoding='utf8')
    units=[u for u in c['units'] if u.get('chapter')==ident and not u.get('provenance_only')]
    assert len(units)==count,(ident,len(units))
    evidence='editorial/batch01-review.md#'+ident
    for u in units:
        assert u['targets']
        for key in u['targets']:
            span=c['spans'][key]
            assert span['path']==card['path'] and verify_span(text,span),(ident,key)
            span.update(review_status='verified',evidence=evidence,reviewed_on='2026-10-25',
                rationale='Исходный случай и назначенный раздел прочитаны; отличающиеся предпосылки, примеры и оговорки сохранены либо явно исправлены. Конкретные случаи и ограничения перечислены в свидетельстве пакета 01.')
        u.update(status='semantic-reviewed',review_evidence=evidence)
    card.update(text_sha256=digest(text),prose='rewritten',source_units=count,
        source_units_pending_semantic_review=0,content_acceptance='closed',
        evidence=[evidence],remaining=[],reviewed_on='2026-10-25')
save(c);cards['checked_on']='2026-10-25';write('editorial/chapter-reviews.json',cards)
batch.update(status='content-reviewed-validation-pending',reviewed_on='2026-10-25',
    source_units=sum(expected.values()),review='editorial/batch01-review.md',
    remaining=['Итоговые аудит, браузер и LF/CRLF-клон на текущем расширенном курсе; публикация отдельных отчётов.'])
write('editorial/current-batch.json',batch)
print('Accepted ten manually reviewed cards;',sum(expected.values()),'source units;',
      sum(v['content_acceptance']=='closed' for v in cards['chapters'].values()),'/',len(cards['chapters']),'cards closed.')
