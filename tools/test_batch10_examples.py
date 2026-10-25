"""Independent arithmetic and structural checks for encoder batch 01."""
import itertools
import math
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
def softmax(xs):
    ex=[math.exp(x-max(xs)) for x in xs]
    return [x/sum(ex) for x in ex]
def dot(a,b):return sum(x*y for x,y in zip(a,b))

class EncoderBatchExamples(unittest.TestCase):
    def test_lookup_and_output_axes(self):
        e=[(0,0,1),(1,0,0),(0,1,0),(1,1,0),(-1,0,1)]
        h=(.2,.9,.1);scores=[dot(h,row) for row in e]
        self.assertEqual(scores,[.1,.2,.9,1.1,-.1])
        self.assertEqual(max(range(5),key=scores.__getitem__),3)
        self.assertEqual(max(range(3),key=h.__getitem__),1)
        gradients={i:[0,0] for i in [1,2]}
        for ident,g in zip([1,2,1],[(1,2),(3,4),(5,6)]):
            gradients[ident]=[a+b for a,b in zip(gradients[ident],g)]
        self.assertEqual(gradients,{1:[6,8],2:[3,4]})

    def test_masks_and_rotations(self):
        mask=[[j<=i for j in range(3)] for i in range(3)]
        self.assertEqual([sum(r) for r in mask],[1,2,3])
        def rot(v,t):return (v[0]*math.cos(t)-v[1]*math.sin(t),v[0]*math.sin(t)+v[1]*math.cos(t))
        q,k=(2,3),(4,-1)
        self.assertAlmostEqual(dot(rot(q,.3),rot(k,.8)),dot(q,rot(k,.5)))
        self.assertAlmostEqual(math.sin(1),.8414709848)

    def test_losses_gradients_and_huber(self):
        p=softmax([0,0,0]);self.assertAlmostEqual(-math.log(p[0]),math.log(3))
        self.assertEqual([round(v-(i==0),6) for i,v in enumerate(p)],[-.666667,.333333,.333333])
        y=[1,0,1];self.assertEqual([(.5-t)/3 for t in y],[-1/6,1/6,-1/6])
        self.assertAlmostEqual(math.log1p(math.exp(0)),math.log(2))
        self.assertEqual(1*(5-.5),4.5);self.assertEqual(.5*5**2,12.5)

    def test_parameters_and_accumulation(self):
        self.assertEqual(768*3+3,2307)
        self.assertEqual(12*2*8*(768+768),294912)
        self.assertEqual(2307+294912,297219)
        self.assertEqual(100*768,76800)
        self.assertEqual(math.ceil(1000/(2*8)),63)
        self.assertEqual(math.ceil(63/4),16)
        self.assertEqual([16/40,16/40,8/40],[.4,.4,.2])
        self.assertAlmostEqual(sum(n*x for n,x in zip([16,16,8],[1,2,3]))/40,1.8)

    def test_classification_and_entity_metrics(self):
        p=softmax([2,1,0]);self.assertAlmostEqual(-math.log(p[0]),.40760596444438046)
        self.assertAlmostEqual(-math.log(p[2]),2.40760596444438)
        self.assertEqual(2*2/(2*2+0+1),.8)
        self.assertAlmostEqual((1+1+0)/3,2/3)
        self.assertEqual(3/4,.75)
        self.assertEqual(2*(95/195)/2,95/195)
        self.assertAlmostEqual(95/195,.48717948717948717)

    def test_crf_paths_and_normalizer(self):
        emissions=[{'O':2,'B':1},{'O':0,'B':3}]
        scores={a+b:emissions[0][a]+emissions[1][b]+(-2 if (a,b)==('O','B') else 0)
                for a,b in itertools.product('OB',repeat=2)}
        self.assertEqual(scores,{'OO':2,'OB':3,'BO':1,'BB':4})
        self.assertEqual(max(scores,key=scores.__getitem__),'BB')
        z=sum(math.exp(s) for s in scores.values())
        # Explicit forward recurrence, distinct from enumeration.
        alphas=[math.log(sum(math.exp(emissions[0][a]+(-2 if (a,b)==('O','B') else 0)) for a in 'OB'))+emissions[1][b] for b in 'OB']
        self.assertAlmostEqual(math.log(sum(math.exp(a) for a in alphas)),math.log(z))
        self.assertAlmostEqual(math.log(z)-3,1.4401896985611953)

    def test_qa_spans_and_offsets(self):
        start=[0,5,2];end=[4,0,2]
        scores={(i+1,j+1):start[i]+end[j] for i in range(3) for j in range(i,min(3,i+2))}
        self.assertEqual(scores,{(1,1):4,(1,2):0,(2,2):5,(2,3):7,(3,3):4})
        self.assertEqual(max(scores,key=scores.__getitem__),(2,3))
        self.assertEqual('Анна в Москве'[7:13],'Москве')
        self.assertEqual(sum(min(2,5-i) for i in range(5)),9)
        self.assertEqual(5*6//2,15);self.assertEqual(16*15//2,120)

    def test_contrastive_objectives(self):
        self.assertAlmostEqual(-math.log(.5+.3),.2231435513142097)
        self.assertAlmostEqual(-(math.log(.5)+math.log(.3))/2,.9485599924429406)
        p=softmax([1,0]);self.assertAlmostEqual(-math.log(p[0]),.3132616875182228)
        self.assertAlmostEqual(p[0]-1,-.2689414213699951)
        self.assertAlmostEqual(p[1],.2689414213699951)
        self.assertAlmostEqual(max(0,.3-.4+.2),.1)
        self.assertEqual([2-1,4-1],[1,3])

    def test_rotation_cosine_and_duplication(self):
        self.assertEqual(dot((0,1),(1,0)),0)
        self.assertEqual(dot((0,1),(0,1)),1)
        self.assertAlmostEqual(math.cos(math.radians(40)),.766044443118978)
        self.assertAlmostEqual(math.cos(math.radians(80)),.17364817766693041)
        self.assertGreater(math.cos(math.radians(40)),.7)
        self.assertLess(math.cos(math.radians(80)),.7)
        self.assertEqual(1_000_000*768*4,3_072_000_000)
        self.assertEqual(1_000_000*256*4,1_024_000_000)

    def test_bm25_full_precision(self):
        n=1000;k=1.2;b=.75;ratio=300/200
        score=sum(math.log(1+(n-df+.5)/(df+.5))*f*(k+1)/(f+k*(1-b+b*ratio))
                  for f,df in [(4,30),(2,5)])
        self.assertAlmostEqual(score,11.710677,places=6)
        text=(ROOT/'05-language-models/02-bert-encoders/bert-retrieval.md').read_text(encoding='utf8')
        self.assertIn('11.710677',text)

    def test_sparse_aggregation_and_rrf(self):
        self.assertAlmostEqual(max(math.log1p(1),math.log1p(2)),math.log(3))
        self.assertAlmostEqual(math.log1p(1)+math.log1p(2),math.log(6))
        # Batch means for two different term-use patterns; same active count.
        self.assertEqual(sum(x*x for x in [1,0]),1)
        self.assertEqual(sum(x*x for x in [.5,.5]),.5)
        self.assertAlmostEqual(1/61+1/160,.02264344262295082)
        self.assertAlmostEqual(2/70,.02857142857142857)

    def test_colbert_same_mean_different_maxsim(self):
        q=[(1,0),(0,1)];a=[(1,0),(0,1),(-1,0)];b=[(0,1),(0,1),(0,-1)]
        mean=lambda rows:tuple(sum(r[i] for r in rows)/len(rows) for i in range(2))
        score=lambda doc:sum(max(dot(row,t) for t in doc) for row in q)
        self.assertEqual(mean(a),mean(b));self.assertEqual((score(a),score(b)),(2,1))
        self.assertEqual(sum(max(dot(row,t) for t in [(1,0)]) for row in [(1,0),(1,0)]),2)
        self.assertEqual(100000*128*2,25_600_000)
        self.assertEqual(100000*128*128*2,3_276_800_000)
        self.assertEqual(10000*384*4*4,61_440_000)

    def test_overlap_guarantee_exhaustively(self):
        for c in range(2,10):
            for a in range(1,c+1):
                overlap=a-1;step=c-overlap
                for start in range(40):
                    self.assertTrue(any(w<=start and start+a<=w+c for w in range(0,50,step)))
        self.assertFalse(any(w<=5 and 9<=w+8 for w in range(0,40,6)))
        self.assertEqual(1+2*2*3,13)
        self.assertEqual(1000*33+4*1000,37000)

    def test_domain_weights_and_boxes(self):
        self.assertAlmostEqual((900*.95+100*.6)/1000,.915)
        self.assertAlmostEqual((.95+.6)/2,.775)
        self.assertEqual([100/1000*1000,400/2000*1000,500/1000*1000,600/2000*1000],[100,200,500,300])
        self.assertAlmostEqual(1-(1-.1)**2,.19)

    def test_ranking_metrics_and_calibration(self):
        self.assertEqual((1+1/2+0)/3,.5)
        self.assertAlmostEqual(3.5/(3+1/math.log2(3)),.9639404333166532)
        self.assertAlmostEqual(-(math.log(.9)+math.log(.1))/2,1.203972804325936)
        self.assertAlmostEqual(((.9-1)**2+.9**2)/2,.41)
        self.assertAlmostEqual(((.6-1)**2+.6**2)/2,.26)
        a,b=[4,0,0],[3,2.9,-10]
        self.assertGreater(max(softmax(a)),max(softmax(b)))
        self.assertLess(max(softmax([v/10 for v in a])),max(softmax([v/10 for v in b])))
        self.assertEqual(3/60,.05);self.assertEqual(60/100,.6)

    def test_integrated_gradients(self):
        # Exact integrals of 3*alpha and 2*alpha along (2alpha,3alpha).
        ig=[2*3/2,3*2/2];self.assertEqual(ig,[3,3])
        self.assertEqual(sum(ig),2*3);self.assertEqual(sum([2*3,3*2]),12)

    def test_quantization_and_resource_payloads(self):
        q=round(.26/.1);self.assertEqual(q,3);self.assertAlmostEqual(q*.1-.26,.04)
        self.assertLess((.9998-1)**2/2,2.000001e-8)
        self.assertLess(.9998,.9999)
        self.assertEqual(100_000_000*4,400_000_000)
        self.assertEqual(100_000_000*16,1_600_000_000)
        self.assertEqual(100_000_000/128*4,3_125_000)

    def test_bidirectional_cache_and_queue(self):
        self.assertEqual(dot(softmax([0]),[2]),2)
        self.assertEqual(dot(softmax([0,0]),[2,6]),4)
        completion=[(i+1)*.2 for i in range(10)]
        self.assertAlmostEqual(sum(completion)/10,1.1)
        self.assertEqual(100/completion[-1],50)
        self.assertEqual((1024/32)**2,1024)

if __name__=='__main__':unittest.main()
