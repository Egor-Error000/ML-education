"""Independently recompute the adaptation/inference examples, not model benchmarks."""
import json,math,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sm(z):
 e=[math.exp(x-max(z)) for x in z];return [x/sum(e) for x in e]
def dot(a,b):return sum(x*y for x,y in zip(a,b))
class AdaptationInferenceExamples(unittest.TestCase):
 def close(self,a,b,places=7):
  for x,y in zip(a,b):self.assertAlmostEqual(x,y,places=places)
 def test_budget_and_effective_batch(self):
  self.assertEqual(7e9*(2+4+8),98e9);self.assertEqual(7e9*(2+2+8+4),112e9)
  self.assertEqual(4194304*(2+2+8)/2**20,48)
  self.assertEqual(4096*4096*2/2**20,32);self.assertEqual(32*4096**2*2/2**30,1)
  self.assertEqual(2*8*2,32)
 def test_sft_shift_and_denominator(self):
  ids=['BOS','USER','question','ASSIST','answer','EOS'];mask=[0,0,0,0,1,1]
  self.assertEqual([(i,ids[i+1]) for i in range(5) if mask[i+1]],[(3,'answer'),(4,'EOS')])
  self.assertAlmostEqual((-math.log(.5)-math.log(.25))/2,1.03972077084)
  self.assertAlmostEqual((2+9*.2)/10,.38);self.assertAlmostEqual((2+.2)/2,1.1)
 def test_trainable_dimensions(self):
  self.assertEqual(4*3,12);self.assertEqual(1*(4+3),7)
  self.assertGreater(2*(5+3),5*3)
 def test_lora_two_sgd_steps(self):
  x=[1,2];a=[.1,.2];b=[0.,0.];base=[5,11];target=[6,10];losses=[]
  for _ in range(2):
   h=dot(a,x);y=[v+bi*h for v,bi in zip(base,b)];g=[v-t for v,t in zip(y,target)]
   losses.append(sum(v*v/2 for v in g));ga=[dot(b,g)*xi for xi in x];gb=[gi*h for gi in g]
   a=[v-.1*gi for v,gi in zip(a,ga)];b=[v-.1*gi for v,gi in zip(b,gb)]
  self.close(a,[.10975,.2195]);self.close(b,[.09875,-.09875])
  y=[v+bi*dot(a,x) for v,bi in zip(base,b)];self.close(y,[5.0541890625,10.9458109375])
  self.close(losses,[1,.950625]);self.assertAlmostEqual(sum((v-t)**2/2 for v,t in zip(y,target)),.8945583311,places=8)
 def test_lora_targets_and_merge(self):
  square=8*(4096+4096);mlp=8*(4096+11008)
  self.assertEqual(2*square*32,4194304);self.assertEqual((4*square+3*mlp)*32,19988480)
  self.assertEqual(19988480*2/2**20,38.125)
  w=[[1,2],[3,4]];a=[.1,.2];b=[.05,-.05]
  merged=[[v+bi*aj for v,aj in zip(row,a)] for row,bi in zip(w,b)]
  self.close([dot(r,[1,2]) for r in merged],[5.025,10.975])
 def test_dora_coordinate_norms(self):
  u=[[.61,.78],[.02,.96]];m=[5,2];x=[1,2]
  self.close([dot(r,x)*mi/math.sqrt(dot(r,r)) for r,mi in zip(u,m)],[10.957360,4.040790],places=5)
  self.assertAlmostEqual(5*8/math.sqrt(32),7.0710678119)
  self.assertEqual(8*8192+4096,69632)
 def test_adalora_budget_and_metadata(self):
  groups=[[.9,.8,.1,.05],[.7,.6,.5,.4],[.2,.15,.05,.01],[.95,.02,.01,.01]]
  selected=sorted([(v,i) for i,g in enumerate(groups) for v in g],reverse=True)[:8]
  self.assertEqual([sum(i==g for _,i in selected) for g in range(4)],[2,4,1,1])
  self.assertEqual(2*8+2*14,44);self.assertEqual(3*8+1*14,38)
  self.assertEqual(7e9*(32/64-8/64)/8,328125000)
 def test_adapter_and_prefix(self):
  h=[3,1];down=dot([1,-1],h);self.close([h[0]+.5*max(0,down),h[1]+max(0,down)],[4,3])
  self.assertEqual(2*768*64+64+768,99136)
  self.assertEqual([sum(j<2 or j-2<=i for j in range(5)) for i in range(3)],[3,4,5])
  a=sm([1,0])[0];out=a*4+(1-a)*2
  self.assertAlmostEqual(out,3.4621171573);self.assertAlmostEqual(a*(1-a)*2,.3932238665)
  self.assertEqual(2*12*10*768,184320)
 def test_reward_ppo_kl(self):
  self.assertAlmostEqual(math.log1p(math.exp(-1.2)),.2632824673)
  self.assertAlmostEqual(min(1.4*2,1.2*2),2.4)
  self.assertAlmostEqual(min(.6*-2,.8*-2),-1.6)
  self.assertAlmostEqual(.8*math.log(.8/.5)+.2*math.log(.2/.5),.192744757)
 def test_dpo_ipo_kto_orpo(self):
  self.assertAlmostEqual(math.log1p(math.exp(-.1)),.6443966601)
  self.assertAlmostEqual(-.1/(1+math.exp(.1)),-.0475020813)
  self.assertEqual([(h-5)**2 for h in [1,5,7]],[16,0,4])
  self.assertEqual([2*(h-5) for h in [1,5,7]],[-8,0,4])
  self.assertAlmostEqual(math.log1p(1/16),.0606246218)
 def test_factor_average_and_dare(self):
  self.assertEqual((1*1+(-1)*(-1))/2,1);self.assertEqual(((1-1)/2)**2,0)
  self.assertAlmostEqual(.1+.1+.1-1,-.7)
  d=.4;p=.25;out=d/p
  self.assertAlmostEqual(p*out,d);self.assertAlmostEqual(p*(out-d)**2+(1-p)*d*d,.48)
 def test_distillation_kl_temperature(self):
  teacher=[.8,.2];student=[.6,.4]
  kl=sum(p*math.log(p/q) for p,q in zip(teacher,student))
  self.assertAlmostEqual(kl,.09151622185)
  self.close(sm([2,0]),[.88079707798,.11920292202]);self.close(sm([1,0]),[.73105857863,.26894142137])
  self.close([.8*p+.2*y for p,y in zip([.1,.9],[1,0])],[.28,.72])
  self.assertEqual(100000*50000*2,10e9)
 def test_continual_constraints_ewc(self):
  self.assertGreater(.9-.2*.3,.8-.2*.05);self.assertLess(.9-.3,.8-.05)
  self.assertAlmostEqual(.5*(100*.1**2+.1**2),.505)
  theta=1/101;self.assertAlmostEqual(theta,.009900990099)
  self.assertAlmostEqual(.5*(theta-1)**2,.4901480247)
 def test_evaluation_paired_uncertainty(self):
  ds=[0]*70+[-1]*10+[1]*20;mean=sum(ds)/100
  se=math.sqrt((sum(d*d for d in ds)/100-mean**2)/100)
  self.assertAlmostEqual(mean,.1);self.assertAlmostEqual(se,.0538516481)
  self.assertAlmostEqual((45+.5*20)/100,.55);self.assertAlmostEqual(45/80,.5625)
  self.assertAlmostEqual(math.exp(.38),1.4622845894)
 def test_generation_no_cache_positions(self):
  self.assertEqual([4+i for i in range(2)],[4,5]);self.close(sm([2,1,0]),[.6652409558,.2447284711,.0900305732])
 def test_kv_incremental_attention_and_error(self):
  self.assertAlmostEqual(dot(sm([2,4]),[2,4]),3.76159415596)
  self.assertAlmostEqual(dot(sm([3,6,9]),[2,4,6]),5.8959491568)
  self.assertAlmostEqual(dot(sm([0,.1]),[0,10]),5.2497918748)
  self.assertAlmostEqual(dot(sm([0,0]),[.1,10.1]),5.1)
 def test_kv_head_and_batch_memory(self):
  per=2*32*8*128*2;self.assertEqual(per,131072)
  self.assertEqual(per*4096/2**20,512);self.assertEqual(per*(512+1024+4096)/2**20,704)
  self.assertEqual(16*2**30//(per*4096),32)
  self.assertEqual(2*80*8*128*2*8192/2**30,2.5)
 def test_context_positions_and_prefix_cost(self):
  self.assertEqual(32000/4,8000);self.assertEqual(1+3*(4-1),10)
  self.assertEqual(23000*500+500*501//2,11625250)
  threshold=(1.25-1)/(1.25-.1);self.assertAlmostEqual(threshold,.2173913043)
  self.assertAlmostEqual(1000*(1-.9)*23000,2300000)
 def test_decoding_truncation_order(self):
  self.close(sm([1,.5,0]),[.5064803911,.3071958857,.1863237232])
  self.close([.5/.8,.3/.8],[.625,.375])
  squared=[v*v for v in [.6,.25,.15]];p=[v/sum(squared) for v in squared]
  self.assertAlmostEqual(p[0],.8089887640)
  p=[.4,.3,.2,.1];first=[v/sum(p[:2]) for v in p[:2]]
  self.assertGreater(first[0],.55);self.assertLess(p[0],.55)
 def test_grammar_local_vs_global(self):
  self.close([.02/.03,.01/.03],[2/3,1/3])
  masses=[.9*.01,.1*1];self.close([v/sum(masses) for v in masses],[.08256880734,.91743119266])
  self.assertNotAlmostEqual(masses[0]/sum(masses),.9)
  self.assertAlmostEqual(.1**3,.001);self.assertAlmostEqual(1+.1+.01,1.11)
 def test_serving_slots_and_page_waste(self):
  self.assertEqual(sum([2,5,1,4]),12);self.assertEqual(sum([4,4,4,4,2]),18)
  sizes=[1,16,17,31];slots=sum(math.ceil(n/16)*16 for n in sizes)
  self.assertEqual(slots,96);self.assertEqual(sum(sizes),65)
  table=['B1','B3','B7'];self.assertEqual((table[34//16],34%16),('B7',2))
 def test_paged_softmax_and_shared_blocks(self):
  self.assertAlmostEqual(dot(sm([0,math.log(3)]),[0,4]),3)
  self.assertEqual(4*3,12);self.assertEqual(2+4,6)
  self.assertAlmostEqual((1-.2)/(5-1),.2);self.assertAlmostEqual((1-.6)/(5-1),.1)
  self.assertEqual(4*2,8)
 def test_speculative_mass_exactness(self):
  p=[.5,.3,.2];q=[.6,.1,.3];accepted=[min(x,y) for x,y in zip(p,q)];reject=1-sum(accepted)
  residual=[max(0,x-y)/reject for x,y in zip(p,q)]
  self.close(residual,[0,1,0]);self.close([a+reject*r for a,r in zip(accepted,residual)],p)
  self.close([a+reject*v for a,v in zip(accepted,p)],[.6,.16,.24])
  self.assertAlmostEqual(1-sum(abs(x-y) for x,y in zip(p,q))/2,.8)
 def test_speculative_progress_and_scheduler(self):
  progress=sum(.8**i for i in range(5));self.assertAlmostEqual(progress,3.3616)
  self.assertAlmostEqual(progress*10/16,2.101)
  self.assertAlmostEqual(sum(.9**i for i in range(5))*10/24,1.7062916667)
  depths=[1.9,2.35,2.44]
  self.assertEqual(max(range(3),key=lambda i:depths[i]/[10,12,18][i]),1)
  self.assertEqual(max(range(3),key=lambda i:depths[i]/[10,16,18][i]),0)
 def test_speculative_beam_counterexample(self):
  scores={'AX':.55*.49,'AY':.55*.51,'BZ':.45*.6,'BW':.45*.4}
  self.assertEqual(sorted(scores,key=scores.get,reverse=True)[:2],['AY','BZ'])
  self.assertNotIn('AY',['AX','BZ']);self.assertAlmostEqual(1-1/1.6,.375)
  self.assertAlmostEqual(100e6/86400,1157.4074074074)
 def test_affine_rounding_and_group_metadata(self):
  xs=[-1.4,-.3,.2,1.3]
  def away(v):return math.copysign(math.floor(abs(v)+.5+1e-10),v)
  rec=[min(7,max(-7,away(x/.2)))*.2 for x in xs]
  self.close(rec,[-1.4,-.4,.2,1.4]);self.assertAlmostEqual(sum((a-b)**2 for a,b in zip(xs,rec))/4,.005)
  self.assertEqual(4+16/128,4.125);self.assertEqual((128+12+4)*8/256,4.5)
  self.assertAlmostEqual(2/15,.1333333333333)
 def test_outlier_and_smoothquant(self):
  self.assertEqual([round(x/(100/127)) for x in [.1,.3]],[0,0])
  self.assertEqual(dot([100,1],[.01,1]),dot([10,1],[.1,1]))
  self.assertAlmostEqual(1.4+2.2,3.6);self.assertLess(abs(4-3.6),abs(3-3.6))
 def test_gptq_and_awq(self):
  def cost(d):return .5*(2*d[0]**2+2*d[0]*d[1]+2*d[1]**2)
  self.assertAlmostEqual(cost([-.4,0]),.16);self.assertAlmostEqual(cost([-.4,.2]),.12)
  self.assertAlmostEqual((-.4)**2/(2*(2/3)),.12)
  self.assertAlmostEqual(dot([.13,.77],[10,1]),2.07)
  self.assertAlmostEqual(dot([.55,.77],[2.5,1]),2.145)
  self.assertEqual(2*4-3*3,-1)
 def test_quantization_format_budget_and_amdahl(self):
  self.assertEqual(7e9*4/8,3.5e9);self.assertAlmostEqual(3.5e9/2**30,3.259629011)
  self.assertAlmostEqual(24/13.5,1.7777777778)
  self.assertAlmostEqual(1+2**-10,1.0009765625)
 def test_pruning_coupled_shapes_and_csr(self):
  w=[[.1,.5,0,.3],[0,.2,.8,0],[.4,0,.1,.6],[0,.7,0,.2]]
  pruned=[[0 if abs(v)==.1 else v for v in row] for row in w]
  self.assertEqual(sum(v==0 for row in pruned for v in row),8)
  self.close([dot(row,[1,2,3,4]) for row in pruned],[2.2,2.8,2.8,2.2])
  self.assertEqual(8*(2+4)+5*4,68);self.assertEqual(3*4*10,120);self.assertEqual(3*4*6,72)
 def test_pruning_iterations_and_speed_limit(self):
  self.assertAlmostEqual(.9**7,.4782969);self.assertAlmostEqual(.7*4/16,.175)
  self.assertAlmostEqual(1/(.6+.4/2),1.25)
 def test_deployment_lifecycle_and_memory(self):
  self.assertEqual(sum([4,12,18,10,6,8]),58)
  tokens=math.ceil((1000+512)/16)*16;self.assertEqual(tokens,1520)
  self.assertEqual(tokens*128/1024,190);self.assertEqual(8*1024//190,43)
  self.assertEqual(10-4-1,5)
 def test_queue_scaling_latency_and_quantiles(self):
  self.assertEqual((3-2)*60,60);self.assertEqual(math.ceil(30/(10*.75)),4)
  self.assertEqual(math.ceil(2*12/5),5)
  self.assertAlmostEqual(sum(.2*i for i in range(1,11))/10,1.1)
  samples=[100]*4+[5000];self.assertEqual(sum(samples)/5,1080)
  self.assertEqual(sorted(samples)[math.ceil(.95*5)-1],5000)
  combined=[100]*100+[5000];self.assertEqual(combined[math.ceil(.95*101)-1],100)
 def test_authored_scenarios_figures_and_clean_sources(self):
  ids=['adapt-choice','adapt-sft','adapt-regimes','adapt-lora','adapt-variants','adapt-alternatives',
   'adapt-preference','adapt-merging','adapt-distillation','adapt-continual','adapt-evaluation',
   'infer-generation','infer-kv','infer-context','infer-decoding','infer-serving','infer-speculative',
   'infer-quantization','infer-pruning','infer-deployment']
  scenarios=json.loads((ROOT/'editorial/chapter-scenarios.json').read_text(encoding='utf8'))['chapters']
  course=json.loads((ROOT/'course.json').read_text(encoding='utf8'));by={c['id']:c for c in course['chapters']}
  for ident in ids:
   t=(ROOT/by[ident]['path']).read_text(encoding='utf8');s=scenarios[ident]
   self.assertRegex(t.lower(),r'## .*практик');self.assertNotIn('<!-- unit',t);self.assertNotIn('archive/',t)
   self.assertGreater(len(s['definitions_and_formulas']),100)
   self.assertTrue(all((ROOT/p).is_file() for p in by[ident]['figures']))
if __name__=='__main__':unittest.main()
