"""Independent arithmetic for the first seven feature lessons in batch 03."""
import math,unittest

class Batch03FeatureExamples(unittest.TestCase):
 def test_mature_labels_and_baselines(self):
  confirmed,unknown,total=105,195,1200
  self.assertAlmostEqual(confirmed/total,.0875)
  self.assertAlmostEqual((confirmed+unknown)/total,.25)
  y=[0,0,0,8]
  self.assertAlmostEqual(sum(abs(x) for x in y)/4,2)
  self.assertAlmostEqual(math.sqrt(sum(x*x for x in y)/4),4)
  self.assertAlmostEqual(sum(abs(x-2) for x in y)/4,3)
  self.assertAlmostEqual(math.sqrt(sum((x-2)**2 for x in y)/4),math.sqrt(12))
  self.assertAlmostEqual(.9*.95+.1*.6,.915);self.assertAlmostEqual(.2*.95+.8*.6,.67)
 def test_split_and_delayed_outcomes(self):
  rows=[(u,t) for u in 'ABC' for t in range(1,5)]
  self.assertEqual((sum(u in 'AB' for u,t in rows),sum(u=='C' for u,t in rows)),(8,4))
  self.assertEqual((sum(t<=2 for u,t in rows),sum(t>=3 for u,t in rows)),(6,6))
  self.assertEqual((sum(u in 'AB' and t<=2 for u,t in rows),sum(u=='C' and t>=3 for u,t in rows)),(4,2))
  self.assertEqual([t for t in [10,11,12] if t+7+2<20],[10])
  self.assertEqual(3*2*4+3,27);self.assertEqual(27+2*4+1,36)
 def test_quality_missing_and_agreement(self):
  obs=(100*10+50*30)/150
  self.assertAlmostEqual(obs,50/3);self.assertAlmostEqual((100*10+100*30)/200,20)
  self.assertAlmostEqual((150*obs+50*obs)/200,obs)
  self.assertEqual((2-1.5*(4-2),4+1.5*(4-2)),(-1,7))
  pe=.05**2+.95**2;self.assertAlmostEqual((.9-pe)/(1-pe),-.05263157895)
  self.assertAlmostEqual((20+9)/1000,.029)
 def test_imbalance_and_calibration(self):
  tp,fp,fn,tn=80,495,20,9405
  self.assertAlmostEqual((tp+tn)/10000,.9485)
  self.assertAlmostEqual(tp/(tp+fp),80/575)
  self.assertAlmostEqual(2*tp/(2*tp+fp+fn),160/675)
  self.assertAlmostEqual(.5+.5*(2/3),5/6)
  self.assertAlmostEqual(.5+.5*(.5+2/3)/2,19/24)
  self.assertAlmostEqual(sum((p-y)**2 for p,y in zip([.9,.7,.2,.1],[1,0,1,0]))/4,.2875)
  self.assertAlmostEqual((1-.2),.8);self.assertAlmostEqual(9*.2,1.8)
 def test_augmentation_and_duplicate_candidates(self):
  x1,y1,x2,y2=10,20,40,60;w=100
  self.assertEqual((w-x2,y1,w-x1,y2),(60,20,90,60))
  self.assertAlmostEqual((20*40)/(30*40),2/3)
  a=set('abcd');b=set('abce');self.assertAlmostEqual(len(a&b)/len(a|b),.6)
  self.assertAlmostEqual(10000/(1+99*.5),198.01980198)
  self.assertAlmostEqual(1-(1-.8**5)**20,.999643267,places=5)
 def test_batches_masks_and_resources(self):
  lengths=[5,6,20,21,40,41]
  first=[(5,41),(6,40),(20,21)];second=[(5,6),(20,21),(40,41)]
  self.assertEqual(sum(2*max(p) for p in first),204)
  self.assertEqual(sum(2*max(p) for p in second),136)
  self.assertEqual((2+0+0+0)/4,.5);self.assertEqual((2+0)/2,1)
  self.assertEqual([[j for j in range(12) if j%4==c] for c in range(4)],[[0,4,8],[1,5,9],[2,6,10],[3,7,11]])
  self.assertEqual(64*3*224*224*4/2**20,36.75)
  self.assertEqual(sum([12,18,4,30]),64)
 def test_shift_and_group_uncertainty(self):
  self.assertAlmostEqual(.8*.1+.2*.4,.16)
  self.assertAlmostEqual(.2*.1+.8*.4,.34)
  self.assertAlmostEqual(.8*.25*.1+.2*4*.4,.34)
  self.assertAlmostEqual(10000/(80*.25**2+20*4**2),10000/325)
  self.assertAlmostEqual(math.sqrt(.8*.2/100),.04)
  self.assertAlmostEqual(math.sqrt(.8*.2/5),.1788854382)
 def test_nominal_codes_are_not_order_invariant(self):
  scores={'A':1.,'B':2.,'C':3.}
  recoded={'A':2.,'B':3.,'C':1.}
  self.assertEqual(scores['B']-scores['A'],scores['C']-scores['B'])
  self.assertNotEqual(recoded['B']-recoded['A'],recoded['C']-recoded['B'])
 def test_one_hot_baselines_and_penalties(self):
  y=[10,13,8];codes=[1,2,3];b=37/3;w=-1
  self.assertAlmostEqual(sum((b+w*x-t)**2 for x,t in zip(codes,y))/3,32/9)
  self.assertEqual(3**2+(-2)**2,13);self.assertEqual(2**2+5**2,29)
  self.assertAlmostEqual(sum(v*v for v in [-1/3,8/3,-7/3]),38/3)
 def test_missing_mode_and_indicator(self):
  observed=['A','A','B'];filled=observed+['A','A']
  self.assertAlmostEqual(observed.count('A')/len(observed),2/3)
  self.assertAlmostEqual(filled.count('A')/len(filled),4/5)
  b,w,g=.1,.4,.2
  self.assertAlmostEqual((b+w+g)-(b+w),g)
 def test_set_pooling(self):
  es={'A':(1,0),'B':(0,2)};items=['A','B','A']
  summed=[sum(es[i][j] for i in items) for j in range(2)]
  self.assertEqual(summed,[2,2]);self.assertEqual([v/len(items) for v in summed],[2/3,2/3])
  self.assertEqual([max(es[i][j] for i in items) for j in range(2)],[1,2])
  self.assertEqual([sum(es[i][j] for i in ['A','B']) for j in range(2)],[1,2])
 def test_delayed_labels(self):
  for values,expected in [([],0.5),([1],2/3),([1,0],.5)]:
   self.assertAlmostEqual((sum(values)+2*.5)/(len(values)+2),expected)
 def test_geography_antimeridian_and_latitude(self):
  r=6371;phi=0;dl=math.radians(-179-179)
  a=math.cos(phi)**2*math.sin(dl/2)**2
  self.assertAlmostEqual(2*r*math.asin(math.sqrt(a)),222.389853289,places=6)
  self.assertAlmostEqual(r*math.pi/180,111.194926645,places=6)
  self.assertAlmostEqual(r*math.pi/180*math.cos(math.radians(60)),55.597463322,places=6)
  self.assertAlmostEqual(math.sin(math.radians(179)),.0174524064)
 def test_pca_projection_reconstruction(self):
  mu=[2,2];v=[1/math.sqrt(2)]*2;x=[5,1]
  centered=[a-b for a,b in zip(x,mu)];z=sum(a*b for a,b in zip(centered,v))
  rebuilt=[m+z*vi for m,vi in zip(mu,v)]
  for a,b in zip(rebuilt,[3,3]):self.assertAlmostEqual(a,b)
  self.assertAlmostEqual(sum((a-b)**2 for a,b in zip(x,rebuilt)),8)
 def test_as_of_window_and_search_budget(self):
  events=[(1,1),(3,3),(9,11)]
  historical=[e for e,a in events if 3<=e<10 and a<10]
  self.assertEqual(historical,[3]);self.assertEqual(20*3*5,300)
 def test_hash_collisions_by_enumeration(self):
  from itertools import product
  configurations=list(product(range(2),repeat=3))
  pair_counts=[sum(a==b for i,a in enumerate(c) for b in c[i+1:]) for c in configurations]
  occupied=[len(set(c)) for c in configurations]
  self.assertAlmostEqual(sum(pair_counts)/len(configurations),1.5)
  self.assertAlmostEqual(sum(occupied)/len(configurations),1.75)
  self.assertAlmostEqual(sum(3-n for n in occupied)/len(configurations),1.25)
  no_collision=math.prod(1-j/1000 for j in range(100))
  self.assertAlmostEqual(1-no_collision,.994,places=3)
  self.assertEqual(4*100000+4*100000+4*(100000+1),1200004)
if __name__=='__main__':unittest.main()
