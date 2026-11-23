from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
import numpy as np
OUT=Path(__file__).parent
colors={'input':'#93c5fd','conv':'#60a5fa','residual':'#a78bfa','pool':'#fb923c','head':'#34d399','output':'#fbbf24'}
def diagram(filename,title,subtitle,stages,footnote):
 fig,ax=plt.subplots(figsize=(19,7.5)); fig.patch.set_facecolor('#fafbfe');ax.set_facecolor('#fafbfe')
 count=len(stages); spacing=2.1
 for i,(label,shape,kind,height,detail,skip) in enumerate(stages):
  x=i*spacing; y=2.4; w=.43; d=.38;h=height
  # Perspective feature volume, diagrammatic rather than to scale.
  ax.add_patch(Polygon([(x,y-h/2),(x+w,y-h/2),(x+w,y+h/2),(x,y+h/2)],facecolor=colors[kind],edgecolor='#334155',lw=1.1))
  ax.add_patch(Polygon([(x+w,y-h/2),(x+w+d,y-h/2+.3),(x+w+d,y+h/2+.3),(x+w,y+h/2)],facecolor=colors[kind],alpha=.55,edgecolor='#334155'))
  ax.add_patch(Polygon([(x,y+h/2),(x+d,y+h/2+.3),(x+w+d,y+h/2+.3),(x+w,y+h/2)],facecolor=colors[kind],alpha=.75,edgecolor='#334155'))
  ax.text(x+.35, y+h/2+.55,shape,ha='center',fontsize=10,weight='bold',color='#0f172a')
  ax.text(x+.35,.35,label,ha='center',va='top',fontsize=10,weight='bold',color='#0f172a')
  ax.text(x+.35,-.18,detail,ha='center',va='top',fontsize=8.5,color='#475569',linespacing=1.5)
  if i<count-1:
   ax.annotate('',xy=((i+1)*spacing-.12,y),xytext=(x+w+d+.1,y),arrowprops=dict(arrowstyle='->',color='#64748b',lw=1.4))
  if skip:
   ax.annotate('',xy=(x+.9,y+.65),xytext=(x-.22,y+.65),arrowprops=dict(arrowstyle='->',connectionstyle='bar,fraction=.65',color='#7c3aed',lw=1.5))
 ax.set_xlim(-.6,(count-1)*spacing+1.3);ax.set_ylim(-1.7,5.7);ax.axis('off')
 fig.text(.04,.94,title,fontsize=23,weight='bold',color='#0f172a')
 fig.text(.04,.89,subtitle,fontsize=12,color='#475569')
 fig.text(.04,.055,footnote,fontsize=10,color='#475569')
 fig.legend(handles=[Patch(facecolor=colors[k],label=v) for k,v in [('conv','Convolution + BN + ReLU'),('residual','Residual blocks'),('pool','Pooling'),('head','Classifier'),('output','Class logits')]],loc='lower center',bbox_to_anchor=(.5,.095),ncol=5,frameon=False)
 for ext in ['png','svg','pdf']:fig.savefig(OUT/f'{filename}.{ext}',dpi=200,facecolor=fig.get_facecolor())
 plt.close(fig)
S=lambda label,shape,kind,h,detail='',skip=False:(label,shape,kind,h,detail,skip)
diagram('scratch_cnn','Scratch CNN','108,842 parameters • Random initialization • Shapes: height × width × channels',[
 S('RGB input','192 × 256 × 3','input',3.2),
 S('Conv block 1','96 × 128 × 32','conv',2.8,'3×3 Conv → BN → ReLU\n2×2 max pool'),
 S('Conv block 2','48 × 64 × 64','conv',2.3,'3×3 Conv → BN → ReLU\n2×2 max pool'),
 S('Conv block 3','24 × 32 × 128','conv',1.8,'3×3 Conv → BN → ReLU\n2×2 max pool'),
 S('Adaptive max pool','3 × 4 × 128','pool',1.1),
 S('Classifier','1,536 → 10','head',.7,'Flatten → Dropout(0.3)\nLinear'),
 S('Prediction','10','output',.4,'10 unnormalized logits')], 'Feature volumes are schematic. Cross-entropy consumes logits directly; softmax is only needed for probabilities.')
diagram('custom_residual_cnn','Custom residual CNN','692,138 parameters • Random initialization • Five residual blocks • Shapes: H × W × C',[
 S('RGB input','192 × 256 × 3','input',3.2),
 S('Stem','96 × 128 × 32','conv',2.8,'3×3 Conv, stride 2\nBN → ReLU'),
 S('Residual block 1','96 × 128 × 32','residual',2.8,'32 → 32; stride 1\nIdentity shortcut',True),
 S('Residual blocks 2–3','48 × 64 × 64','residual',2.2,'32 → 64, stride 2; then 64 → 64\nProjection; then identity',True),
 S('Residual blocks 4–5','24 × 32 × 128','residual',1.7,'64 → 128, stride 2; then 128 → 128\nProjection; then identity',True),
 S('Adaptive max pool','3 × 4 × 128','pool',1),
 S('Classifier','1,536 → 10','head',.65,'Flatten → Dropout(0.3)\nLinear'),
 S('Prediction','10','output',.4,'10 logits')], 'Each residual block: 3×3 Conv → BN → ReLU → 3×3 Conv → BN → add shortcut → ReLU. Projection: 1×1 Conv + BN.')
diagram('pretrained_resnet18','Pretrained ResNet18','ImageNet-1K pretrained backbone • New 512 → 10 classifier • Shapes: H × W × C',[
 S('RGB input','224 × 224 × 3','input',3.2,'Resize 256 → center crop 224\nImageNet normalization'),
 S('Stem','56 × 56 × 64','conv',2.7,'7×7 Conv, stride 2 → BN → ReLU\n3×3 max pool, stride 2'),
 S('layer1','56 × 56 × 64','residual',2.7,'2 residual blocks\nIdentity shortcuts',True),
 S('layer2','28 × 28 × 128','residual',2.1,'2 residual blocks\nFirst downsamples (stride 2)',True),
 S('layer3','14 × 14 × 256','residual',1.6,'2 residual blocks\nFirst downsamples (stride 2)',True),
 S('layer4','7 × 7 × 512','residual',1.2,'2 residual blocks\nFirst downsamples (stride 2)',True),
 S('Global average pool','1 × 1 × 512','pool',.8,'Flatten → 512 features'),
 S('New classifier','512 → 10','head',.5,'Linear → 10 logits')], 'Epochs 1–3: train classifier only; backbone and BN statistics frozen. Then fine-tune all layers (backbone LR 1e−5; classifier LR 1e−4).')
print('Saved three diagrams as PNG, SVG and PDF.')
