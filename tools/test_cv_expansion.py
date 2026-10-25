"""Independent numerical and contract checks for the 60-chapter CV expansion.

These checks evaluate mathematical examples and source-figure integrity. They
do not assert model benchmark results or replace a subject-matter review.
"""
from pathlib import Path
import hashlib,json,math,re,unittest

ROOT=Path(__file__).resolve().parents[1]
COURSE=json.loads((ROOT/'course.json').read_text(encoding='utf8'))
MANIFEST=json.loads((ROOT/'editorial/cv-expansion.json').read_text(encoding='utf8'))
BY={c['id']:c for c in COURSE['chapters']}

class CVExpansion(unittest.TestCase):
    def text(self,id):return (ROOT/BY[id]['path']).read_text(encoding='utf8')
    def shown(self,id,value,printed,tolerance=0.000006):
        self.assertIn(printed,self.text(id),id+' must contain its checked example')
        self.assertAlmostEqual(value,float(printed),delta=tolerance,msg=id+': '+printed)

    def test_inventory_routes_and_existing_ids(self):
        cv=[c for c in COURSE['chapters'] if c['module']=='cv']
        self.assertEqual(len(cv),60)
        self.assertEqual(len({c['section'] for c in cv}),9)
        self.assertEqual(len(MANIFEST['original_ids']),13)
        self.assertEqual(sum(c['kind']=='new-authored' for c in MANIFEST['chapters']),47)
        for id in MANIFEST['original_ids']:
            self.assertEqual(BY[id]['path'],'04-computer-vision/'+id+'.md')
        for c in cv:
            self.assertIn('## Самопроверка с ответами',self.text(c['id']))
            self.assertIn('## Разобранный пример',self.text(c['id']))
            self.assertIn('**Практика:**',self.text(c['id']))
        for route in COURSE['routes']:
            if route['id'].startswith('vision'):
                self.assertFalse(any(id.startswith(('lm-','bert-','adapt-')) for id in route['chapters']))
                seen=set()
                for id in route['chapters']:
                    self.assertTrue(set(BY[id]['prerequisites'])<=seen,(route['id'],id))
                    seen.add(id)

    def test_convolution_and_backbones(self):
        self.shown('cv-classical',7*15*4*9,'3780',0)
        self.shown('cv-resnet',2*9*64**2,'73728',0)
        self.shown('cv-resnet',64*128,'8192',0)
        self.shown('cv-efficient',28**2*9*32*64,'14450688',0)
        self.shown('cv-efficient',28**2*(9*32+32*64),'1831424',0)
        self.shown('cv-convnext',49*96+2*96*384,'78432',0)
        self.shown('cv-input',7+2*2*4,'23',0)
        self.shown('cv-classification',512*10+10,'5130',0)
        self.shown('cv-hierarchy',64*49**2,'153664',0)

    def test_transfer_and_probability_losses(self):
        self.shown('cv-transfer',10000*768*4,'30720000',0)
        smoothed=-(.9+.1/3)*math.log(.8)-2*(.1/3)*math.log(.1)
        self.shown('cv-losses',smoothed,'0.36177')
        self.shown('cv-losses',-(1-.2)**2*math.log(.2),'1.03004')
        self.shown('cv-dino',-.8*math.log(.6)-.2*math.log(.4),'0.59192')
        self.shown('cv-dino',16*16*768,'196608',0)
        self.shown('cv-pretraining',147*768,'112896',0)

    def test_contrastive_denominator(self):
        p=math.exp(4)/(math.exp(4)+math.exp(1)+1)
        self.shown('cv-contrastive',-math.log(p),'0.06588')
        hard=math.exp(4)/(math.exp(4)+math.exp(3.5)+1)
        self.shown('cv-contrastive',hard,'0.61544')
        self.shown('cv-contrastive',-math.log(hard),'0.48541')
        self.shown('cv-retrieval',(1+2/3+3/5)/3,'0.75556')

    def test_detection_geometry(self):
        anchor=(50,40,20,10);gt=(54,38,30,10)
        encoded=((gt[0]-anchor[0])/anchor[2],(gt[1]-anchor[1])/anchor[3],math.log(gt[2]/anchor[2]),math.log(gt[3]/anchor[3]))
        decoded=(anchor[0]+encoded[0]*anchor[2],anchor[1]+encoded[1]*anchor[3],anchor[2]*math.exp(encoded[2]),anchor[3]*math.exp(encoded[3]))
        for a,b in zip(gt,decoded):self.assertAlmostEqual(a,b)
        self.shown('cv-faster-rcnn',encoded[2],'0.40547')
        self.shown('cv-dense-detectors',math.sqrt(10/30),'0.57735')
        self.shown('cv-yolo',8400*3*(80+5),'2142000',0)
        self.shown('cv-yolo',8400*(80+4*16),'1209600',0)
        self.shown('cv-pyramids',80**2+40**2+20**2,'8400',0)
        self.shown('cv-detr',min(1+100,2+1.1),'3.1')
        self.shown('cv-deformable',.375*0+.125*2+.375*4+.125*6,'2.5')

    def test_masks_and_panoptic(self):
        self.shown('cv-unet',2*1.7/(2.1+2),'0.82927')
        self.shown('cv-segformers',4*256*128**2,'16777216',0)
        self.shown('cv-prototypes',1/(1+math.exp(1)),'0.26894')
        self.shown('cv-prototypes',138**2*32*100,'60940800',0)
        self.shown('cv-mask2former',(0.8+0.6)/(2+.5+.5),'0.46667')
        self.shown('cv-instance',100*80*28**2,'6272000',0)

    def test_motion_and_video(self):
        self.shown('cv-pose-models',17*48*64,'52224',0)
        self.shown('cv-pose-models',17*(384+512),'15232',0)
        self.shown('cv-tracking-models',380/420,'0.90476')
        self.shown('cv-flow',math.sqrt(5),'2.23607')
        self.shown('cv-video',(8*196)**2,'2458624',0)
        self.shown('cv-video',196*8**2+8*196**2,'319872',0)
        self.shown('cv-tracking',11+.8*(12-11),'11.8')

    def test_original_anchor_chapter_examples(self):
        self.shown('cv-vit',(3/4)*2+(1/4)*0,'1.5')
        self.assertIn('TP=1,FP=1,FN=1,TN=1',self.text('cv-semantic'))
        truth={0,1};predicted={0,2}
        self.assertAlmostEqual(len(truth&predicted)/len(truth|predicted),1/3)
        self.assertAlmostEqual(2*len(truth&predicted)/(len(truth)+len(predicted)),.5)
        self.shown('cv-pose',100+40/2,'120',0)
        angle_error=abs((math.radians(89-(-89))+math.pi/2)%math.pi-math.pi/2)
        self.assertAlmostEqual(math.degrees(angle_error),2)
        self.assertIn('минимальная physical difference равна 2°',self.text('cv-obb'))
        self.shown('cv-depth',sum(abs(p-y)/y for p,y in zip([1,2,4],[2,4,8]))/3,'0.5')
        self.shown('cv-open-vocabulary',400*1.28,'512',0)
        self.assertIn('минимальная длина',self.text('cv-text-recognition'))

    def test_camera_and_stereo(self):
        self.shown('cv-camera',math.sqrt(17.25),'4.15331')
        self.shown('cv-camera',800*1/4+320,'520',0)
        self.shown('cv-stereo',96/3.1,'30.96774')
        self.shown('cv-stereo',.2*48+.5*32+.3*24,'32.8')
        self.shown('cv-sfm',(-2)**2+3**2,'13',0)
        self.shown('cv-bev',352*400*64*4/2**20,'34.375')
        self.shown('cv-pointnet',1024**2,'1048576',0)

    def test_rendering(self):
        alphas=[.5,.25];colours=[.8,.2];trans=1.;output=0.
        for a,c in zip(alphas,colours):
            output+=trans*a*c;trans*=1-a
        self.shown('cv-nerf',output,'0.425')
        self.shown('cv-nerf',-math.log(.5)/.1,'6.93147')
        self.shown('cv-gaussians',.6*1+(1-.6)*.5*.2,'0.64')
        self.shown('cv-gaussians',59*4*10**6/2**20,'225.07',.005)

    def test_ocr_and_reconstruction(self):
        def collapse(path):
            merged=[c for i,c in enumerate(path) if i==0 or path[i-1]!=c]
            return ''.join(c for c in merged if c!='_')
        self.assertEqual(collapse('ll_l_'),'ll')
        self.assertEqual(collapse('lll_'),'l')
        self.shown('cv-text-detection',1/(1+math.exp(-1)),'0.73106')
        self.shown('cv-documents',72/100,'0.72')
        self.shown('cv-anomaly',math.sqrt(4/4+1),'1.41421')
        self.shown('cv-restoration',10*math.log10(1/.0025),'26.02060')
        self.shown('cv-domains',(math.ceil((2048-512)/384)+1)**2,'25',0)

    def test_generative_models(self):
        self.shown('cv-generative',.5*(1+.25-math.log(.25)-1),'0.81815')
        noisy=math.sqrt(.64)*.8+math.sqrt(.36)*(-.5)
        self.shown('cv-ddpm',noisy,'0.34')
        self.shown('cv-ddpm',(noisy-math.sqrt(.36)*(-.4))/math.sqrt(.64),'0.725')
        self.shown('cv-sampling',.9*.8+math.sqrt(.19)*(-.5),'0.50206')
        self.shown('cv-latent-diffusion',1024*77,'78848',0)
        self.shown('cv-dit-flow',1024**2,'1048576',0)
        self.shown('cv-control',2*1024*8,'16384',0)
        self.shown('cv-generation-eval',1+1+4-2*math.sqrt(4),'2',0)

    def test_operational_cost(self):
        self.shown('cv-diagnosis',80/90,'0.88889')
        self.shown('cv-robustness',40/800,'0.05')
        self.shown('cv-deployment',1000/26,'38.46',.005)
        self.shown('cv-cases',90/(90+198),'0.3125')
        self.shown('cv-cases',10*100+198,'1198',0)
        self.shown('cv-cases',5*100+396,'896',0)

    def test_manifest_and_external_figure_integrity(self):
        for c in MANIFEST['chapters']:
            normalized=self.text(c['id']).replace('\r\n','\n')
            self.assertEqual(hashlib.sha256(normalized.encode('utf8')).hexdigest(),c['text_sha256'],c['id'])
        figures=MANIFEST['external_figures']
        self.assertGreaterEqual(len(figures),55)
        for f in figures:
            raw=(ROOT/f['path']).read_bytes()
            self.assertTrue(raw.startswith((b'\x89PNG\r\n\x1a\n',b'\xff\xd8\xff')),f['path'])
            self.assertEqual(hashlib.sha256(raw).hexdigest(),f['sha256'])
            self.assertIn('../'+f['path'],self.text(f['chapter']))
            self.assertIn(f['source'],self.text(f['chapter']))
            self.assertIn(f['visual_review'],['manually-corrected','contact-sheet-inspected'])

if __name__=='__main__':unittest.main()
