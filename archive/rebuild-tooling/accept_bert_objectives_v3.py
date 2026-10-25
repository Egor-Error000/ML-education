"""Manual source review after reading, arithmetic and visual comparison."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from edit_chapter import load,save,ROOT
from editorial_io import verify_span
c=load();rel='05-language-models/02-bert-encoders/bert-objectives.md'
text=(ROOT/rel).read_text(encoding='utf8')
units=[u for u in c['units'] if u.get('chapter')=='bert-objectives' and not u.get('provenance_only')]
assert len(units)==64
for u in units:
    for key in u['targets']:
        span=c['spans'][key]
        assert span['path']==rel and verify_span(text,span)
        span.update(review_status='verified',evidence='editorial/review-evidence.md#bert-objectives',rationale='Исходная единица и итоговый целевой раздел прочитаны; объединение сохраняет случаи и ограничения. Арифметика проверена независимо, схемы просмотрены и исправлены. Подробное свидетельство перечисляет отдельные проверки.')
    u['status']='semantic-reviewed'
save(c)
