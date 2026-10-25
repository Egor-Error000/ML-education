"""Review all 44 source units; reference one shared attention derivation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from edit_chapter import ROOT,load,save,sections,digest
c=load();rel='05-language-models/01-foundations/lm-transformer.md'
groups={
'Вход и выход повторяемого блока':'''b46fbad22299dcff41 dac9e7fbcc7c300644 68f023c163a6091f7e 136f53b8a6f42238e8 8e5eabc60f181a1879 8ce648b17dc6cd007d''',
'Две ветви и положение нормализации':'''51a2a554b6702dbf1a 34a3389ef21cf3fbca c00e218b4d38b2b325 00d59062da61225861''',
'Нормализация одного состояния':'''efd97cee3e63cac7ed 71e922801e5bce87c6 a6ce930930f89d1d95 e20647fdc2f46ddf58 8477ddab421112cb12''',
'Обмен между позициями и возврат к ширине D':'''13453093867f0dcfc4 55a93517196f04cc2c d345732b26c41ba1b8 82d196709f9fbbe0cb 429453687a0c81dcb3 410cd4293896ebbe99''',
'MLP: расширение и сжатие каждой позиции':'''72a0e0205c04fef836 50c1a5fd8020efc8e2 234d77e674e99b9637 43789bdbb53e7500c2 592ea2376318cfac1a a7a5cb3ca4080f063b 263b67915c36d9ff06 44ebeec84c69419c44 b2f03827c013ffd5d5 819603035be1815a83 f8336a97c44bbd7286''',
'Остаточная сумма и параметры блока':'''13234436a9f462872e 140eba200cd63cd156 43c303113c473db762 af70990d327e0025bb fea68e09e3d7c08d8c ace8bf41b238827eed''',
'Блок в полной модели и контроль понимания':'''bebbb99db8ac3a9118 339e1a8c1bdfe95347 573af5f88133b61dd4 744a7512c4c21a67ad e42a60a1ff1bab3bf4 d61cbd215f485b1eb9'''}
dest={ident:title for title,ids in groups.items() for ident in ids.split()}
units=[u for u in c['units'] if u.get('chapter')=='lm-transformer' and not u.get('provenance_only')]
assert set(dest)=={u['id'] for u in units}
parts=sections((ROOT/rel).read_text(encoding='utf8'))
for u in units:
    title=dest[u['id']];a,b,body=parts[title]
    c['spans'][u['anchor']].update(path=rel,start_line=a,end_line=b,sha256=digest(body),destination_section=title,status='revised',review_status='verified',rationale='Прочитан исходный блок и итоговый разбор: формы, отличия операций, числа и ограничения сохранены; повтор вывода attention заменён ссылкой на единое объяснение.',evidence='editorial/review-evidence.md#transformer')
    u.update(targets=[u['anchor']],status='semantic-reviewed',action='объединить с сохранением нюансов')
# The old bookmark stays in the block chapter; the actual derivation has its own target.
u=next(u for u in units if u['id']=='429453687a0c81dcb3')
target='05-language-models/01-foundations/lm-attention.md';title='Оценки, масштаб, маска и нормировка'
a,b,body=sections((ROOT/target).read_text(encoding='utf8'))[title];key='t-transformer-shared-attention-scale'
c['spans'][key]={'path':target,'start_line':a,'end_line':b,'sha256':digest(body),'destination_section':title,'status':'revised','review_status':'verified','rationale':'Единственный полный вывод масштаба по предпосылкам дисперсии; глава блока явно ссылается сюда вместо повторного вывода.','evidence':'editorial/review-evidence.md#transformer'}
u['targets'].append(key)
save(c)
