"""Final explicit narration changes after reading the relevant source units."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from edit_chapter import ROOT,load,save,transform
c=load();rel='03-clustering/cl-deep.md'
pairs=[
('а декодер $g_\\phi$ восстанавливает вход из этого вектора.', 'а декодер $g_\\phi$ восстанавливает вход из этого вектора. При d<D скрытое представление имеет меньше координат, чем вход; одной этой узости недостаточно для полезной семантики.'),
('Обратная связь от собственных назначений требует', 'Верхний контур схемы обучает восстановление и получает начальные центры по готовым z. Нижний контур использует только кодировщик и центры: из текущего q периодически строится фиксированная цель p, затем ошибка KL обновляет параметры. Возврат вектора в исходное пространство в этой фазе отсутствует.\n\nОбратная связь от собственных назначений требует'),
('Спроектируйте сравнение frozen/DEC/IDEC при одном K и бюджете;', 'Сначала раздельно меняйте представление и алгоритм двухэтапного базового решения. Затем спроектируйте сравнение замороженной основы, DEC и IDEC при одном K и бюджете;')]
def op(s):
    for before,after in pairs:
        if before not in s:raise ValueError(before)
        s=s.replace(before,after)
    return s
transform(rel,op,'После сверки исходных единиц конкретизированы две фазы схемы DEC, ширина автокодировщика и раздельные сравнения представления и алгоритма.',c);save(c)
# The changed sections were read together with their sources; diagrams get a
# separate visual acceptance before chapter cards close.
for span in c['spans'].values():
    if span['path']==rel or (span['path']=='03-clustering/cl-stability.md' and span.get('destination_section')=='Устойчивость и практика'):
        if span.get('review_status')=='pending':
            span.update(review_status='verified',evidence='editorial/review-evidence.md#deep-clustering' if span['path']==rel else 'editorial/review-evidence.md#cluster-stability')
save(c)
