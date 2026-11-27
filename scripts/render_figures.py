"""Re-export the fixed evidence at final paper size; never modify sealed inputs."""
from pathlib import Path
import sys, json, csv, hashlib, re, fitz
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[1]
import argparse
parser=argparse.ArgumentParser(description="Regenerate the nine paper figures from bundled evidence.")
parser.add_argument("--output",type=Path,default=ROOT/"regenerated-figures")
args=parser.parse_args()
if args.output.exists(): parser.error("Choose a new output folder.")
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
RUN=ROOT/'experiments/offline'
F=args.output; (F/'sources').mkdir(parents=True,exist_ok=True)
(F/'previews').mkdir(exist_ok=True)
S=json.loads((RUN/'results/summary.json').read_text())
rows=list(csv.DictReader((RUN/'results/tables/full_geometry.csv').open()))
plt.rcParams.update({'font.size':10,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'svg.fonttype':'path','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'lines.linewidth':1.3})
green='#276657'; orange='#a65c28'; gray='#616a71'
# Text accents on white exceed the author kit's 4.5:1 contrast requirement.
# Orange: 5.0204:1; green: 6.7212:1; gray: 5.5142:1.
manifest=[]
def save(fig,name,scope):
    fig.savefig(F/'sources'/f'{name}.svg')
    fig.savefig(F/'previews'/f'{name}.png',dpi=160)
    drawing=svg2rlg(str(F/'sources'/f'{name}.svg'))
    renderPDF.drawToFile(drawing,str(F/f'{name}.pdf'))
    # SVG labels are outlined. Remove only empty ReportLab text-state blocks
    # and their unused standard-font resource, keeping every drawn vector.
    doc=fitz.open(F/f'{name}.pdf'); page=doc[0]
    before=page.get_pixmap().samples
    stream=page.read_contents()
    for block in re.findall(rb'BT\b.*?\bET',stream,re.S):
        assert not re.search(rb'\bTj\b|\bTJ\b',block), 'Unexpected live text'
    stream=re.sub(rb'BT\b.*?\bET',b'',stream,flags=re.S)
    page.clean_contents()
    doc.update_stream(page.get_contents()[0],stream)
    kind,res=doc.xref_get_key(page.xref,'Resources')
    if kind=='xref':
        doc.xref_set_key(int(res.split()[0]),'Font','<<>>')
    else:
        assert kind=='dict'
        doc.xref_set_key(page.xref,'Resources/Font','<<>>')
    assert page.get_pixmap().samples==before, 'Font cleanup changed rendering'
    data=doc.tobytes(garbage=4,deflate=True);doc.close()
    (F/f'{name}.pdf').write_bytes(data)
    manifest.append({'file':f'{name}.pdf','scope':scope,'width_inches':fig.get_figwidth(),'height_inches':fig.get_figheight(),'minimum_label_pt':9,'sha256':hashlib.sha256((F/f'{name}.pdf').read_bytes()).hexdigest(),'text':'Outlined vector glyphs; editable Matplotlib source retained.'})
    plt.close(fig)
def txt(ax,x,y,s,**kw): ax.text(x,y,s,ha='center',va='center',**kw)
fig,ax=plt.subplots(figsize=(3.3,2.25)); ax.set_axis_off(); ax.set_xlim(0,1); ax.set_ylim(0,1)
txt(ax,.5,.94,r'Same outcomes: $R=[1,1,0,0]$',fontsize=10)
for x,title,reward,adv,note in [(.26,'Equal penalties',r'$[1,1,-2,-2]$',r'$[1,1,-1,-1]$','Same credit'),(.76,'Unequal penalties',r'$[1,1,-1,-3]$',r'$\frac{[3,3,-1,-5]}{\sqrt{11}}$','Different credit')]:
    txt(ax,x,.76,title,fontsize=9,fontweight='bold')
    txt(ax,x,.61,reward,fontsize=10)
    ax.annotate('',xy=(x,.37),xytext=(x,.51),arrowprops={'arrowstyle':'->','color':gray})
    txt(ax,x,.29,adv,fontsize=11)
    txt(ax,x,.08,note,fontsize=9,color=green if x<.5 else orange)
fig.subplots_adjust(left=0,right=1,bottom=0,top=1); save(fig,'normalization_example','Synthetic, population SD; epsilon=0.')
fig,axes=plt.subplots(1,2,figsize=(6.875,2.25),sharey=True)
for ax,n,title in zip(axes,['std','mean'],['SD + stabilizer','Mean only']):
    rr=[next(r for r in rows if r['rule']==rule and r['normalization']==n) for rule in ['learned','constant','length_global']]
    w=np.array([float(r['within_energy'])*100 for r in rr]); t=np.array([float(r['weight_energy'])*100 for r in rr])
    ax.bar(range(3),w,color=green,label='Within-group',width=.6)
    ax.bar(range(3),t,bottom=w,color='#e3bfa0',edgecolor=orange,label='Prompt weight',width=.6)
    # Explicit line segments survive SVG-to-PDF conversion, unlike SVG patterns.
    for i,(bottom,height) in enumerate(zip(w,t)):
        for y in np.arange(bottom-1.2,bottom+height,.85):
            lo=max(0.,(bottom-y)/2); hi=min(.6,(bottom+height-y)/2)
            if hi>lo:ax.plot([i-.3+lo,i-.3+hi],[y+2*lo,y+2*hi],color=orange,lw=.7)
    for i,total in enumerate(w+t): ax.text(i,total+.35,'~0' if total<1e-5 else f'{total:.2f}',ha='center',fontsize=9)
    ax.set_xticks(range(3),['Learned','Constant','Length']); ax.set_title(title,fontsize=10)
    ax.set_ylim(0,15.5); ax.set_yticks([0,5,10,15])
axes[0].set_ylabel('Orthogonal energy (%)'); axes[1].legend(loc='upper left',frameon=False)
fig.subplots_adjust(left=.08,right=.99,bottom=.19,top=.86,wspace=.18); save(fig,'energy_accounting','500 prompts, 185 mixed; original full_geometry.csv.')
fig,axes=plt.subplots(1,3,figsize=(6.875,2.7))
ax=axes[0]; ax.set_xlim(-.35,3.7); ax.set_ylim(-.4,2.1); ax.set_aspect('equal'); ax.set_axis_off()
for end,start,col in [((3,0),(0,0),gray),((3,2**.5),(3,0),orange),((3,2**.5),(0,0),green)]:
    ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','color':col,'lw':1.4})
ax.text(1.4,-.32,r'$s\|x\|=3$',ha='center',fontsize=9)
ax.text(3.06,.65,r'$\sqrt{2}$',fontsize=9)
ax.text(1.2,1.65,r'$c=3/\sqrt{11}$',ha='center',fontsize=10)
ax.set_title('(a) Components',fontsize=10)
t=np.linspace(0,5,201)
for ax,y,title,xlab in [(axes[1],np.sqrt(.5/(.5+(t/(1+2*t))**2)),r'(b) Scale fixed distances',r'Coefficient $\beta$'),(axes[2],np.sqrt(.5/(.5+(1/(3+t))**2)),r'(c) Shift at fixed spread',r'Added penalty $a$')]:
    ax.plot(t,y,color=green); ax.set_xlim(0,5); ax.set_ylim(.8,1.01); ax.set_yticks([.8,.9,1]); ax.set_xticks([0,1,3,5]); ax.set_title(title,fontsize=9); ax.set_xlabel(xlab)
axes[1].axhline(np.sqrt(2/3),color=orange,ls='--'); axes[1].plot(1,3/np.sqrt(11),'o',color=orange)
axes[1].set_ylabel('Per-group cosine',labelpad=1)
axes[2].plot(0,3/np.sqrt(11),'o',color=orange)
fig.subplots_adjust(left=.01,right=.98,bottom=.23,top=.84,wspace=.56)
save(fig,'components_and_interventions','Synthetic: R=[1,1,0,0], distances [1,3]; not a fitted empirical sweep.')
EXT=ROOT/'experiments/assignment/results'
pair_rows=[r for r in csv.DictReader((EXT/'per_group.csv').open()) if r['normalization']=='std']
assert len(pair_rows)==185
opportunity=[r for r in pair_rows if r['opportunity']=='True']
assert len(opportunity)==132
fig,axes=plt.subplots(1,2,figsize=(6.875,2.55),sharey=True)
for ax,rr,title in zip(axes,[pair_rows,opportunity],['All mixed groups (185)','At least two incorrect (132)']):
    for key,label,col,style in [('learned_energy','Learned distance',green,'-'),('constant_energy','Constant',gray,':'),('length_energy','Token length',orange,'--')]:
        values=np.sort([float(r[key])*100 for r in rr])
        ax.step(np.r_[values[0],values,51],np.r_[0,np.arange(1,len(rr)+1)/len(rr),1],where='post',label=label,color=col,ls=style)
    ax.set_xlim(-1,51); ax.set_ylim(0,1.03); ax.set_title(title,fontsize=10)
    ax.set_xlabel('Orthogonal energy (%)')
axes[0].set_ylabel('Fraction of groups'); axes[1].legend(frameon=False,loc='lower right')
fig.subplots_adjust(left=.08,right=.99,bottom=.23,top=.87,wspace=.15)
save(fig,'group_novelty_ecdf','185 mixed and 132 opportunity prompts; recorded assignment extension, constant-score groups retained.')
fig,axes=plt.subplots(1,2,figsize=(6.875,2.55))
x=np.array([float(r['learned_energy'])*100 for r in opportunity])
y=np.array([float(r['length_energy'])*100 for r in opportunity])
axes[0].scatter(x,y,s=10,alpha=.55,color=green,edgecolor='none')
axes[0].plot([0,51],[0,51],ls='--',color=gray,lw=.8)
axes[0].set(xlim=(-.05,51),ylim=(-.05,51),xlabel='Learned energy (%)',ylabel='Length energy (%)')
axes[0].set_xscale('symlog',linthresh=1,linscale=.65)
axes[0].set_yscale('symlog',linthresh=1,linscale=.65)
axes[0].set_xticks([0,1,5,20,50],['0','1','5','20','50'])
axes[0].set_yticks([0,1,5,20,50],['0','1','5','20','50'])
axes[0].set_title('Opportunity groups (132)',fontsize=10)
values=np.sort([float(r['residual_cosine']) for r in pair_rows if r['residual_cosine']!=''])
assert len(values)==131
axes[1].step(np.r_[values[0],values],np.r_[0,np.arange(1,132)/131],where='post',color=green)
axes[1].axvline(0,ls=':',color=gray,lw=.8)
axes[1].set(xlim=(-1.03,1.03),ylim=(0,1.03),xlabel='Residual cosine',ylabel='Fraction of defined groups')
axes[1].set_title('Defined residual pairs (131)',fontsize=10)
fig.subplots_adjust(left=.08,right=.99,bottom=.23,top=.87,wspace=.40)
save(fig,'direct_residual_comparison','Fixed 132 opportunity groups and 131 defined residual cosines; no population inference.')

def box(ax,x,y,w,h,label,col='#f4f5f5'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.006',fc=col,ec=gray,lw=.7))
    txt(ax,x+w/2,y+h/2,label,fontsize=9)
def arrow(ax,start,end,dashed=False):
    ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','color':gray,'lw':1,'linestyle':'--' if dashed else '-'})
fig,ax=plt.subplots(figsize=(6.875,2.9));ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
box(ax,.01,.68,.18,.23,'One prompt\n8 completions')
box(ax,.26,.68,.22,.23,'Verifier labels\nCorrect references\nIncorrect queries')
box(ax,.55,.68,.20,.23,'Frozen VAE\nLearned projection\nNonnegative metric')
box(ax,.82,.68,.17,.23,'Incorrect penalty\nthen group\nnormalization')
for a,b in [((.19,.8),(.26,.8)),((.48,.8),(.55,.8)),((.75,.8),(.82,.8))]:arrow(ax,a,b)
for x,title,count,note in [(.01,'All wrong','47 / 500','No correct reference'),(.35,'Mixed','185 / 500','650 incorrect queries'),(.69,'All correct','268 / 500','No incorrect query')]:
    box(ax,x,.16,.30,.33,f'{title}: {count}\n{note}\n'+('Shaping active' if title=='Mixed' else 'Shaping inactive'))
txt(ax,.5,.04,'Counts: post-hoc audit, k = 8; not historical training coverage.',fontsize=9)
fig.subplots_adjust(0,0,1,1); save(fig,'signal_route','Conceptual path with measured 500-prompt audit counts.')
fig,ax=plt.subplots(figsize=(6.875,3.05));ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
box(ax,.01,.7,.18,.22,'Frozen latents\n(detached)')
box(ax,.27,.7,.20,.22,'Learned head\nprojection + metric')
box(ax,.56,.7,.17,.22,'Shaped rewards')
box(ax,.81,.7,.18,.22,'Detached\nadvantages')
for a,b in [((.19,.81),(.27,.81)),((.47,.81),(.56,.81)),((.73,.81),(.81,.81))]:arrow(ax,a,b)
box(ax,.27,.37,.20,.20,'Auxiliary loss')
box(ax,.66,.37,.33,.20,'Policy loss + log probabilities')
arrow(ax,(.37,.70),(.37,.57));arrow(ax,(.41,.57),(.41,.70),True)
arrow(ax,(.9,.70),(.9,.57));txt(ax,.67,.62,'Stop-gradient boundary',fontsize=9,color=orange)
ax.plot([.885,.915],[.64,.64],color=orange,lw=2)
txt(ax,.37,.29,'Live head-training route',fontsize=9,color=green)
txt(ax,.82,.29,'Gradients reach policy',fontsize=9,color=green)
box(ax,.03,.04,.94,.16,'Clear gradients  →  backward  →  clip  →  step\nHistorical head omitted from normal clearing and clipping')
fig.subplots_adjust(0,0,1,1);save(fig,'gradient_lifecycle','Source-audit schematic; solid forward, dashed auxiliary backward.')
REPLAY=ROOT/'experiments/replay'
result_path=REPLAY/'run/result.json'
replay=json.loads(result_path.read_text())
summary=json.loads((REPLAY/'results/summary.json').read_text())
assert replay['status']=='COMPLETE' and replay['groups_complete']==185
fig,axes=plt.subplots(1,2,figsize=(6.875,3.0))
draws=[replay['metrics'][f'residual_shuffle_{i:02d}'] for i in range(32)]
xs=[r['vs_original_residual']['cosine'] for r in draws]
ys=[100*r['vs_binary']['norm_ratio'] for r in draws]
ax=axes[0]
ax.scatter(xs,ys,s=18,color=gray,alpha=.8,zorder=3)
learned=summary['exact_reference']['learned'];length=summary['exact_reference']['length']
binary_norm=summary['primary']['learned']['reference_norm']
ax.scatter([1],[100*learned['residual_norm_over_binary_norm']],s=48,color=green,marker='D',zorder=4,label='Learned')
ax.scatter([summary['residual_cross_rule']['cosine']],[100*length['residual_norm_over_binary_norm']],s=55,color=orange,marker='^',zorder=4,label='Length')
ax.axhline(100*learned['root_expected_squared_norm']/binary_norm,ls='--',color=green,lw=.9)
ax.axvline(0,ls=':',color='#bbbbbb',lw=.6)
ax.set(xlim=(-.24,1.1),ylim=(14.5,24.5),xticks=[0,.5,1],yticks=[15,18,21,24],
       xlabel='Cosine with learned\nresidual gradient',ylabel='Residual norm / binary norm (%)',title='(a) Scale and assignment')
ax.legend(frameon=False,loc='upper right',fontsize=9)
ax=axes[1]
ratios=np.array([r['vs_binary']['norm_b']**2 for r in draws])/learned['exact_expected_squared_norm']
ax.scatter(ratios,1+.035*(np.arange(32)%7-3),s=15,color=gray,alpha=.8)
ax.scatter([np.mean(ratios)],[1],s=43,marker='D',facecolor='white',edgecolor='black',zorder=5)
ax.scatter([learned['observed_over_expected_squared_norm']],[2],s=48,marker='D',color=green)
ax.scatter([length['observed_over_expected_squared_norm']],[0],s=55,marker='^',color=orange)
ax.axvline(1,ls='--',color=gray,lw=.9)
ax.set(xlim=(.8,2.5),ylim=(-.45,2.5),xticks=[1,1.5,2,2.5],yticks=[0,1,2],yticklabels=['Length','Shuffled','Learned'],
       xlabel='Squared residual norm /\nexact expectation for that rule',title='(b) Exact reference')
fig.subplots_adjust(left=.085,right=.972,bottom=.27,top=.87,wspace=.48)
save(fig,'assignment_gradient_replay','All 185 mixed groups, 1480 completions, 32 fixed learned-penalty permutations; exact conditional second-moment reference for each rule. Different reference denominators in panel (b); no learning-quality estimate.')
from render_assignment_example import main as render_assignment_example
render_assignment_example(F)
manifest.append({'file':'assignment_example.pdf','scope':'Exact four-completion illustration; fixed score gradients and within-incorrect penalty swap. No empirical inputs.','width_inches':6.875,'height_inches':2.35,'minimum_label_pt':9,'sha256':hashlib.sha256((F/'assignment_example.pdf').read_bytes()).hexdigest(),'text':'Outlined vector glyphs; generator in scripts/render_assignment_example.py.'})
(F/'manifest.json').write_text(json.dumps({'figures':manifest,'input_hashes':{str(p.relative_to(RUN)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [RUN/'results/summary.json',RUN/'results/group_metrics.csv',RUN/'results/tables/full_geometry.csv']},'assignment_input_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [EXT/'summary.json',EXT/'per_group.csv']},'replay_result_sha256':hashlib.sha256(result_path.read_bytes()).hexdigest(),'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
print(f'Exported {len(manifest)} vector figures.')
