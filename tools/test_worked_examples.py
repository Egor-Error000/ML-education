"""Independent calculations for the revision's worked examples, with lesson assertions."""
from pathlib import Path
import math,unittest
ROOT=Path(__file__).resolve().parents[1]
def squared(x,y):return sum((a-b)**2 for a,b in zip(x,y))
def step(points,centres):
    labels=[min(range(len(centres)),key=lambda k:squared(x,centres[k])) for x in points]
    groups=[[x for x,k in zip(points,labels) if k==j] for j in range(len(centres))]
    if any(not g for g in groups):raise ValueError('empty cluster')
    updated=[tuple(sum(x[d] for x in g)/len(g) for d in range(len(points[0]))) for g in groups]
    return labels,updated,sum(squared(x,updated[k]) for x,k in zip(points,labels))
class WorkedExamples(unittest.TestCase):
    def test_kmeans_2d(self):
        x=[(0,0),(0,2),(4,0),(4,2)]
        z,c,j=step(x,[x[0],x[1]])
        self.assertEqual((z,c,j),([0,1,0,1],[(2,0),(2,2)],16))
        self.assertEqual(sum(squared(p,x[k]) for p,k in zip(x,z)),32)
        self.assertEqual(step(x,[x[0],x[2]])[2],4)
        scaled=[(a,b*10) for a,b in x]
        self.assertEqual(step(scaled,[scaled[0],scaled[2]])[2],400)
        self.assertIn('$J=16$',(ROOT/'03-clustering/cl-kmeans.md').read_text(encoding='utf8'))
    def test_kmeans_1d(self):
        points=[(0,),(1,),(4,),(5,)]
        z,c,j=step(points,[(0,),(1,)])
        self.assertAlmostEqual(j,26/3)
        self.assertEqual(step(points,c)[2],1)
    def test_linkage_and_ward(self):
        x=[(0,0),(0,1),(3,0),(3,2)]
        distances=[math.sqrt(squared(a,b)) for a in x[:2] for b in x[2:]]
        self.assertEqual(min(distances),3)
        self.assertAlmostEqual(max(distances),math.sqrt(13))
        self.assertAlmostEqual(sum(distances)/4,3.232526648950187)
        self.assertEqual(squared((0,.5),(3,1)),9.25)
    def test_density_reachability(self):
        x=[(0,0),(0,1),(1,0),(1,1),(2,1),(5,5)]
        neighbours=[{j for j,q in enumerate(x) if squared(p,q)<=1.1**2} for p in x]
        self.assertEqual([len(n) for n in neighbours],[3,3,3,4,2,1])
        core={i for i,n in enumerate(neighbours) if len(n)>=4}
        self.assertEqual(core,{3})
        reached=set().union(*(neighbours[i] for i in core))
        self.assertEqual(reached,{1,2,3,4});self.assertNotIn(0,reached)
        original=[0,.1,.2,.4,2]
        counts=[sum(abs(a-b)<=.21 for b in original) for a in original]
        self.assertEqual(counts,[3,3,4,2,1])
    def test_gmm_em(self):
        a=1/(1+math.exp(-2));b=1-a
        self.assertAlmostEqual(2*b,.238405844044235,places=12)
        self.assertAlmostEqual(2*a,1.76159415595576,places=12)
        variance=a*(2*b)**2+b*(2-2*b)**2
        self.assertAlmostEqual(variance,.419974341614026,places=12)
        self.assertGreater(variance+.001,0)
        self.assertIn('0.420974',(ROOT/'03-clustering/cl-gmm.md').read_text(encoding='utf8'))
    def test_attention_two_heads(self):
        weights=[(.75,.25),(.25,.75)];values=[(2,6),(10,-2)]
        heads=[sum(a*b for a,b in zip(w,v)) for w,v in zip(weights,values)]
        self.assertEqual(heads,[3,1])
        self.assertEqual([heads[0]-heads[1],2*heads[0]+heads[1]],[2,7])
    def test_cross_attention_rectangular_and_pad(self):
        values=[(1,0),(0,2),(3,1)]
        def mix(w):return tuple(sum(a*v[j] for a,v in zip(w,values)) for j in range(2))
        for actual,expected in zip(mix([4/7,2/7,1/7]),(1,5/7)):self.assertAlmostEqual(actual,expected)
        for actual,expected in zip(mix([1/3]*3),(4/3,1)):self.assertAlmostEqual(actual,expected)
        self.assertEqual(mix([2/3,1/3,0]),(2/3,2/3))
        zero_pad=[(1,0),(0,2),(0,0)]
        wrong=tuple(sum(a*v[j] for a,v in zip([4/7,2/7,1/7],zero_pad)) for j in range(2))
        self.assertEqual(wrong,(4/7,4/7));self.assertNotEqual(wrong,mix([2/3,1/3,0]))
    def test_bert_counts_pooling(self):
        d=12;ff=48
        self.assertEqual(4*(d*d+d)+d*ff+ff+ff*d+d+4*d,1884)
        rows=[(1,0,0),(0,2,0),(0,0,3)]
        self.assertEqual(tuple(sum(x[j] for x in rows)/3 for j in range(3)),(1/3,2/3,1))
        self.assertEqual(4*10**2+4*100**2,40400)
    def test_numeric_geometry_and_transforms(self):
        self.assertAlmostEqual(math.sqrt(squared((0,0),(1,100))),100.00499987500625)
        self.assertLess(squared((0,0),(1,1)),squared((0,0),(10,0)))
        mean=(1+3)/2;scale=math.sqrt(((1-mean)**2+(3-mean)**2)/2)
        self.assertEqual((100-mean)/scale,98)
        self.assertEqual((100-1)/(3-1),49.5)
        self.assertEqual((100-3)/(4-2),48.5)
        self.assertAlmostEqual(math.expm1((math.log1p(0)+math.log1p(3))/2),1)
        # A zero IQR is possible for a nonconstant feature.
        values=[0,0,0,0,1]
        self.assertEqual(values[3]-values[1],0);self.assertGreater(max(values)-min(values),0)
    def test_feature_selection_budget_and_means(self):
        self.assertEqual(20*3*5,300)
        self.assertAlmostEqual((.20+.24)/2,.22)
        self.assertAlmostEqual((.18+.30)/2,.24)
        self.assertLess((.20+.24)/2,(.18+.30)/2)
    def test_frozen_vector_normalization(self):
        first=(1,0);second=(.9,.1)
        self.assertAlmostEqual(squared(first,second),.02)
        norm=math.sqrt(sum(v*v for v in second))
        normalized=tuple(v/norm for v in second)
        self.assertAlmostEqual(squared(first,normalized),.0122325306527622,places=12)
    def test_cosine_distance_triangle_counterexample(self):
        u=(1,0);v=(1/math.sqrt(2),1/math.sqrt(2));w=(0,1)
        def cosine_distance(a,b):return 1-sum(x*y for x,y in zip(a,b))
        self.assertEqual(cosine_distance(u,w),1)
        via=cosine_distance(u,v)+cosine_distance(v,w)
        self.assertAlmostEqual(via,2-math.sqrt(2))
        self.assertGreater(cosine_distance(u,w),via)
        self.assertAlmostEqual(squared(u,v),2*cosine_distance(u,v))
    def test_cluster_metrics(self):
        def ari(table):
            n=sum(map(sum,table));rows=[sum(row) for row in table]
            columns=[sum(row[j] for row in table) for j in range(len(table[0]))]
            pairs=sum(math.comb(x,2) for row in table for x in row)
            a=sum(math.comb(x,2) for x in rows);b=sum(math.comb(x,2) for x in columns)
            expected=a*b/math.comb(n,2)
            return (pairs-expected)/((a+b)/2-expected)
        self.assertAlmostEqual(ari([[1,1],[1,1]]),-.5)
        self.assertEqual(ari([[1,0],[1,0],[0,1],[0,1]]),0)
        self.assertEqual(ari([[2,0],[0,2]]),1)
        self.assertAlmostEqual(2*math.log(2)/(math.log(4)+math.log(2)),2/3)
        self.assertEqual((.5+.5)/4,.25)
        within=4*.5**2;between=2*(.5-2.5)**2+2*(4.5-2.5)**2
        self.assertEqual((between/(2-1))/(within/(4-2)),32)
        self.assertIn('ARI=−0.5',(ROOT/'03-clustering/cl-metrics.md').read_text(encoding='utf8'))
    def test_cluster_granularity_and_common_objects(self):
        points=[0,1,4,5]
        self.assertEqual(sum((x-2.5)**2 for x in points),17)
        self.assertEqual(sum(min((x-.5)**2,(x-4.5)**2) for x in points),1)
        self.assertAlmostEqual((7/9+5/7)/2,.746031746031746)
        self.assertGreaterEqual(.8,.85-.1)
        self.assertGreaterEqual(.79,.8-.1) # K=1 may qualify before K=2.
        run1=dict(zip('ABCDE',[0,0,1,1,1]));run2=dict(zip('ABCDF',[7,7,3,3,3]))
        common=sorted(run1.keys() & run2.keys())
        self.assertEqual(common,list('ABCD'))
        for i,a in enumerate(common):
            for b in common[i+1:]:self.assertEqual(run1[a]==run1[b],run2[a]==run2[b])
    def test_dec_targets_and_unequal_frequencies(self):
        def target(q,f):
            weights=[a*a/b for a,b in zip(q,f)]
            return [w/sum(weights) for w in weights]
        self.assertAlmostEqual(target([.6,.4],[1,1])[0],.6923076923076923)
        self.assertAlmostEqual(target([.8,.2],[1,1])[0],16/17)
        self.assertAlmostEqual(target([.6,.4],[9,1])[0],.2)
        for actual,expected in zip(target([.6,.4],[.6,.4]),[.6,.4]):self.assertAlmostEqual(actual,expected)
        kernel=[1,1/5]
        self.assertAlmostEqual(kernel[0]/sum(kernel),5/6)
    def test_mlm_targets_denominators_and_gradients(self):
        p=[[.1,.6,.2,.1],[.5,.2,.2,.1]];labels=[1,0]
        loss=-sum(math.log(row[y]) for row,y in zip(p,labels))/2
        self.assertAlmostEqual(loss,.6019864021629681)
        gradients=[[(value-(j==y))/2 for j,value in enumerate(row)] for row,y in zip(p,labels)]
        for actual,expected in zip(gradients,[[.05,-.2,.1,.05],[-.25,.1,.1,.05]]):
            for a,e in zip(actual,expected):self.assertAlmostEqual(a,e)
            self.assertAlmostEqual(sum(actual),0)
        token_mean=-(math.log(.6)+math.log(.5)+math.log(.9))/3
        sequence_mean=(loss-math.log(.9))/2
        self.assertAlmostEqual(token_mean,.43644443999458743)
        self.assertAlmostEqual(sequence_mean,.3536734582679772)
        original=['CLS','кот','спит','на','мягком','диване','SEP']
        changed=original.copy();changed[2]='MASK';changed[4]='старом'
        targets={i:original[i] for i in [2,4,5]}
        self.assertEqual(targets,{2:'спит',4:'мягком',5:'диване'})
        self.assertEqual([int(a!=b) for a,b in zip(original,changed)],[0,0,1,0,1,0,0])
    def test_mlm_repeated_masking_and_factorization(self):
        self.assertAlmostEqual(1-.85**10,.8031255956592774)
        self.assertEqual(10000*.15,1500)
        self.assertEqual(1500*.8,1200)
        self.assertEqual(1000*20+20*100,22000)
        self.assertEqual(1000*100,100000)
        text=(ROOT/'05-language-models/02-bert-encoders/bert-objectives.md').read_text(encoding='utf8')
        for number in ['0.436444','0.353673','0.803126','22000']:self.assertIn(number,text)
    def test_image_pooling_and_slic(self):
        cells=[(1,0),(3,2),(5,4),(7,2)]
        mean=[sum(v[j] for v in cells)/4 for j in range(2)]
        self.assertEqual(mean,[4,2]);self.assertEqual([max(v[j] for v in cells) for j in range(2)],[7,4])
        self.assertAlmostEqual(mean[0]/math.sqrt(20),.8944271909999159)
        self.assertAlmostEqual(math.sqrt(10**2+(20/10)**2*5**2),14.142135623730951)
        self.assertAlmostEqual(math.sqrt(10**2+(5/10)**2*5**2),10.307764064044152)
    def test_text_topics_and_search(self):
        self.assertAlmostEqual(2*(1+math.log((3+1)/(1+1))),3.386294361119891)
        self.assertAlmostEqual(.7*.1+.3*.01,.073)
        self.assertEqual([2*1+1*0,2*0+1*3],[2,3])
        self.assertEqual([4*.5+1*0,4*0+1*3],[2,3])
        self.assertEqual(len({'модель','обучение'}|{'модель','данные'})/4,.75)
        self.assertEqual(1000000*10/1000,10000)
        self.assertEqual(1000000*20/1000,20000)
        self.assertEqual(len({'a','b','c'} & {'a','c','d'})/3,2/3)
    def test_transformer_normalization_and_gated_mlp(self):
        h=[1,2,3,4];mean=sum(h)/len(h)
        variance=sum((x-mean)**2 for x in h)/len(h)
        self.assertEqual((mean,variance),(2.5,1.25))
        ln=[(x-mean)/math.sqrt(variance) for x in h]
        rms=[x/math.sqrt(sum(y*y for y in h)/len(h)) for x in h]
        self.assertAlmostEqual(sum(ln),0);self.assertGreater(sum(rms),0)
        up=[1,2,3];gate=[x/(1+math.exp(-x)) for x in [0,1,2]]
        product=[a*b for a,b in zip(up,gate)]
        out=[product[0]+product[2],product[1]-product[2]]
        self.assertAlmostEqual(out[0],5.284782467867294,places=12)
        self.assertAlmostEqual(out[1],-3.822665310607284,places=12)
        self.assertGreater(gate[2],1)
    def test_training_mask_perplexity_and_tied_weights(self):
        probabilities=[.5,.25,.8]
        losses=[-math.log(p) for p in probabilities]
        self.assertAlmostEqual(math.prod(probabilities),.1)
        self.assertAlmostEqual(sum(losses)/3,.7675283643313485)
        self.assertAlmostEqual(math.exp(sum(losses)/3),2.154434690031884)
        self.assertAlmostEqual((losses[0]+losses[2])/2,.4581453659370775)
        self.assertAlmostEqual((losses[0]+losses[1])/2,1.0397207708399179)
        grad=[.1,.7-1,.2]
        for actual,expected in zip(grad,[.1,-.3,.2]):self.assertAlmostEqual(actual,expected)
        self.assertEqual(50000*768,38400000)
        self.assertEqual(2*50000*768,76800000)
from test_batch10_examples import EncoderBatchExamples
from test_batch02_examples import AdaptationInferenceExamples
from test_batch03_features import Batch03FeatureExamples
if __name__=='__main__':unittest.main()
