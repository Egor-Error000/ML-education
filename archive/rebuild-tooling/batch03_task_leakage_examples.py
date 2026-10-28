"""Restore distinct future-information examples found during source review."""
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'tools'))
from edit_chapter import load,save,transform
def op(t):
 old='Его использование для прогноза при оформлении — **утечка** информации, которой не будет при реальном применении.'
 new=old+' Менее очевидны число обращений после заказа, итоговая длительность процесса, агрегат по всей будущей истории пользователя или документ, изменённый после решения. Нейтральное название поля не подтверждает его доступность в момент прогноза.'
 assert old in t
 return t.replace(old,new,1)
c=load();transform('01-data-features/01-reliability/df-task.md',op,
 'Пакет 03: при сверке источника восстановлены различные примеры информации после момента прогноза.',c);save(c)
