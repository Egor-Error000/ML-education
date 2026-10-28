"""Place the primary FM source beside the supported mechanism."""
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'tools'))
from edit_chapter import load,save,transform
def op(t):
 old='Это ограничивает множество возможных матриц взаимодействий и не гарантирует правильности для новых значений.'
 new=old+' Исходная формулировка метода — [Factorization Machines](https://doi.org/10.1109/ICDM.2010.127).'
 assert old in t
 return t.replace(old,new,1)
c=load();transform('01-data-features/02-features/feat-interactions.md',op,
 'Пакет 03: первичный источник FM добавлен рядом с описанием механизма.',c);save(c)
