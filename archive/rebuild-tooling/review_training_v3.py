"""Explicit destinations for all 53 original training source units."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from edit_chapter import ROOT,load,save,sections,digest,transform
c=load();rel='05-language-models/01-foundations/lm-training.md'
pairs=[
('Случайное выключение активаций (dropout) работает при обучении и выключается при обычной оценке.',
 'Случайное выключение активаций (dropout) работает при обучении и выключается при обычной оценке. В распространённом варианте с вероятностью удаления p сохранённые элементы делят на $1-p$, чтобы сохранить математическое ожидание; конкретное соглашение нужно проверять в реализации.'),
('Ветвь ширины 10 нельзя сложить с потоком ширины 4 без обратной проекции.',
 'Ветвь ширины 10 нельзя сложить с потоком ширины 4 без обратной проекции. Выходная проекция внимания смешивает объединённые головы и возвращает совместимую ширину D.'),
('Сам RoPE не гарантирует больший вес ближайшей позиции: содержание и периодичность сохраняют значение.',
 'Сам RoPE не гарантирует больший вес ближайшей позиции: содержание и периодичность сохраняют значение. Архитектурная основа — [Transformer](https://arxiv.org/abs/1706.03762); варианты нормализации и управляющих ветвей сверяют с конфигурацией выбранных весов, а не считают одинаковыми для всех моделей.')]
def op(s):
    for before,after in pairs:
        if after in s:continue
        if before not in s:raise ValueError(before)
        s=s.replace(before,after)
    return s
transform(rel,op,'При чтении исходников восстановлены масштабирование dropout, назначение o_proj и оговорка о конфигурации архитектуры.',c);save(c)
groups={
'Как текст превращается в обучающие цели':'''b96fce00d3c467c937 41f749f59cd2886c28 0f9695ee01f24657a0 9c8e2135b6add2641f 4e4621bbf907d73e52 f8b76b2190aab1f532 0fd75d37a23e0ec5a4 5f8f6b2f249304ad0b 586f7b65f2fb2d3b40 cfcdc16fa7f88309bc 6072e389f5020dc2a4 476e3f5549c4c15114''',
'От вероятности правильного токена к ошибке':'''444d6c252a7be1333a 2cc99fe592a26df0ff dfd7ac56935fb4478b''',
'Маска ошибки и правило усреднения':'''0bbf93da1348bcbae8 70cf188276ce275ffc 80b5ea7b0f06efca0b a8781e05d7d1b15f27 b1c7bf7271b6138cdf 67f0ba1b7a754b1d35 00a577194763a49580 cb85498b8757a81800 76785df35514c4a5c3 4141fdcdc5633f3a35 e609ec65d59d591366 505cb2dd169dc3c001''',
'Как ошибка достигает параметров':'''e120f8421d309be285 d6663a715cccc20e70 da8095a1509ea8c2dd 80036b1db7dc2ce684 a58977e0e5e02c7770 293330fd8f27b3517e 76fe0cb69ca4ffe76f cfdae0c6c972524278''',
'Один шаг обновления и режим оценки':'''d1cbe452382e87081f e80a01f20946f774d7 d3f2c6bdcf81f3c19b 90b62e6e4b4d92aba8 8c44f66b80c3af3767 8e1ba021c259b4ef61 b7bc4291dcff17b099 eb696e8286a847d0f3''',
'Перплексия и проверка понимания':'''b0decdf9625599a734 211c610be787bf08ff 60068697a339c5d79b 3cda3c9e0f7c6e8e2f 8c3b47f97fdb621fc7 d4ccd08fba1bc73437 4496abd3a1c904bb34 2620107914c63f040a 46df627eb00dd7f0f4 519240ecd259f8a96b'''}
dest={ident:[title] for title,ids in groups.items() for ident in ids.split()}
extras={
'e609ec65d59d591366':['От вероятности правильного токена к ошибке'],
'd6663a715cccc20e70':['Маска ошибки и правило усреднения'],
'b0decdf9625599a734':['Маска ошибки и правило усреднения'],
'eb696e8286a847d0f3':['Перплексия и проверка понимания'],
'2620107914c63f040a':['Как текст превращается в обучающие цели','Как ошибка достигает параметров']}
parts=sections((ROOT/rel).read_text(encoding='utf8'))
units=[u for u in c['units'] if u.get('chapter')=='lm-training' and not u.get('provenance_only')]
assert set(dest)=={u['id'] for u in units}
for u in units:
    targets=[]
    for i,title in enumerate(dest[u['id']]+extras.get(u['id'],[])):
        a,b,body=parts[title];key=u['anchor'] if i==0 else f"t-lm-training-{u['id']}-{i}"
        c['spans'][key]={'path':rel,'start_line':a,'end_line':b,'sha256':digest(body),'destination_section':title,'status':'revised','review_status':'verified',
          'rationale':'Прочитана исходная единица и проверен итоговый разбор: сдвиг, маски, нормировки, числа, градиент, связывание весов и режимы сохранены; повторение LoRA/RoPE вынесено в явно необязательное задание.',
          'evidence':'editorial/review-evidence.md#training'}
        targets.append(key)
    u.update(targets=targets,status='semantic-reviewed',action='объединить с сохранением нюансов')
save(c)
