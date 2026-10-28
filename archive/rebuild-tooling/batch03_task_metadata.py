"""Make the independent first chapter's prerequisite field explicit."""
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'tools'))
from edit_chapter import load,save,transform
c=load()
def op(t):
 old='**Предпосылки:** .'
 assert old in t
 return t.replace(old,'**Предпосылки:** не требуются.',1)
transform('01-data-features/01-reliability/df-task.md',op,
          'Пакет 03: явно указано начало самостоятельного маршрута.',c)
save(c)
