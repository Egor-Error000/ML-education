"""Hand-place individual source anchors in the matching task-chapter section.

The IDs were checked against their archived text. This narrows destinations;
it does not certify equivalence or close the content review.
"""
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'tools'))
from edit_chapter import load,save
c=load()
first={
 'e0606c34ae41d51035','8e659939763ae55748','af7e2de9c7d73f4e3c',
 '22e12616d9edfa14eb','773b72a2dee62fc778','10aece3b2906ec9040',
 '07431bde53e00dc93b',
 'c9baa8958993ab6fe3','03179024572476e691','2abcb3ebca8defd550',
 '768fb0c58c1ee09a61','839885f1ec2f94d5ce','74976b0fd4f32f5a26',
 '6fc23296a75232eebb','aa57e9a5ea9c276b92'}
second={
 'b5dcb0180e38d3b130','79fc3237d2944238d0','0faf17eee5080035d6',
 'e249f2f2d33326f54d','a1c5c2517d0d38eaee','2b87826ce6a58f2fe0',
 '630c7187f542dccb29',
 '45fca1b2f4561237cf','6dc66e2c32b20cfcd6','46cf4af57bdcf330ee',
 '77b14ea39940335e89','281aec0b8e2be0964b','50f5f8fd5c1beae439',
 'eb7a5093f35a5340ce','ac12db04f28013fb83','80d6982a6c19d2262a',
 '5e21d368b3254184eb','b92b9113d594f5993f','24a77858d9aca5fc94',
 '45d1e5b273e91d3cdc','84566b2a003c4e43e5','302d743f5bed8fdffc'}
assert not first&second
units={u['id']:u for u in c['units']}
for ident in sorted(first|second):
 u=units[ident];assert u['chapter']=='df-task' and len(u['targets'])==2
 primary,extra=u['targets'];assert primary=='u-'+ident and extra.startswith('t-batch02-')
 options=[c['spans'][primary],c['spans'][extra]]
 group={x['destination_section'] for x in options}
 expected=({'Решение, объект и момент прогноза','Событие и доступность поля'} if ident in {
  'e0606c34ae41d51035','8e659939763ae55748','af7e2de9c7d73f4e3c','22e12616d9edfa14eb',
  '773b72a2dee62fc778','10aece3b2906ec9040','07431bde53e00dc93b','b5dcb0180e38d3b130',
  '79fc3237d2944238d0','0faf17eee5080035d6','e249f2f2d33326f54d','a1c5c2517d0d38eaee',
  '2b87826ce6a58f2fe0','630c7187f542dccb29'} else
  {'Цель, горизонт и наблюдаемость метки','Как задача определяет результат и ошибку'})
 assert group==expected,(ident,group)
 chosen=options[0 if ident in first else 1].copy()
 chosen['rationale']='Пакет 03: исходный фрагмент вручную направлен к соответствующему подразделу; содержательная эквивалентность ожидает приёмки.'
 chosen['review_status']='pending';c['spans'][primary]=chosen;u['targets']=[primary]
 assert all(extra not in other['targets'] for other in c['units'])
 del c['spans'][extra]
save(c)
print('Narrowed',len(first|second),'individually inspected task anchors; semantic review remains pending.')
