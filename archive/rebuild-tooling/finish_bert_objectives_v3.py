"""Explicit correction of a figure reference after visual comparison."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from edit_chapter import load,save,transform
c=load()
def op(text):
    old='../../оформление/transformer/bert-mlm-targets.svg'
    assert old in text
    return text.replace(old,'../../оформление/course/bert-mlm-targets.svg')
transform('05-language-models/02-bert-encoders/bert-objectives.md',op,'После визуальной проверки старый рисунок-памятка заменён авторской схемой с явным разветвлением оригинала на вход и цели.',c)
save(c)
