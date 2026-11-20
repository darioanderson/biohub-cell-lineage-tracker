"""Create scientific report figures from the preserved real Sample18 predictions."""
from pathlib import Path
import json, hashlib, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.reference_review import match_centres
DATA=ROOT/'results/real_review_20261118'
OUT=DATA/'report_figures';OUT.mkdir(exist_ok=True)
font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
plt.rcParams.update({'font.family':'Times New Roman','font.size':11,'axes.titlesize':13,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':120})
nodes=pd.read_csv(DATA/'predicted_cells.csv')
edges=pd.read_csv(DATA/'predicted_links.csv')
reference=pd.read_csv(ROOT/'data/cshaper_demo/reference_centres.csv')
summary=pd.read_csv(DATA/'reference_agreement.csv')
volume=np.load(ROOT/'data/cshaper_demo/Sample18_t080_082.npy',allow_pickle=False)
colors=['#285E88','#D38926','#288B75']
manifest=[]
def save(fig,name,description):
    path=OUT/name;fig.savefig(path,dpi=190,bbox_inches='tight');plt.close(fig)
    manifest.append(dict(file=name,description=description,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),synthetic=False))

# All three frames; the markers come only from actual predicted/reference coordinates.
fig,axes=plt.subplots(1,3,figsize=(12,6),constrained_layout=True)
match_rows=[]
for t,ax in enumerate(axes):
    image=volume[t].max(axis=0);lo,hi=np.percentile(image,(1,99.8))
    p=nodes[nodes.t.eq(t)];r=reference[reference.t.eq(t)]
    pairs,errors=match_centres(r[['z','y','x']],p[['z','y','x']])
    good=set(pairs[:,1]);seen=set(pairs[:,0])
    ax.imshow(image,cmap='gray',vmin=lo,vmax=hi)
    for inds,c,marker,label,df in [(sorted(good),'#24DAC4','o','Matched prediction',p),
        ([i for i in range(len(p)) if i not in good],'#ED69CD','o','Unmatched prediction',p),
        ([i for i in range(len(r)) if i not in seen],'#FFAE48','x','Unmatched reference',r)]:
        q=df.iloc[inds]
        if len(q):
            if marker=='x':ax.scatter(q.x,q.y,s=30,c=c,marker=marker,label=label)
            else:ax.scatter(q.x,q.y,s=28,facecolors='none',edgecolors=c,linewidths=.9,label=label)
    ax.set_title(f'Frame {80+t}: {len(pairs)} matched\n{len(p)} predicted / {len(r)} reference')
    ax.set_xlabel('X (pixels)');ax.set_ylabel('Y (pixels)')
    for (ri,pi),error in zip(pairs,errors):match_rows.append(dict(frame=80+t,reference_id=int(r.iloc[ri].reference_id),predicted_node_id=int(p.iloc[pi].node_id),distance_voxels=float(error)))
axes[-1].legend(loc='lower right',fontsize=8)
fig.suptitle('Real temporal output comparison: generated reference, fixed 3-voxel 3D gate')
save(fig,'temporal_output_comparison.png','All three real test frames with actual matched and unmatched centres; 2D maximum projections of 3D matching.')
matches=pd.DataFrame(match_rows);matches.to_csv(OUT/'matched_localization_distances.csv',index=False)

fig,ax=plt.subplots(figsize=(8.4,4.4),constrained_layout=True)
x=np.arange(3)
for j,(field,label,c) in enumerate(zip(['reference_centres','predicted_centres','matched'],['Generated reference','Predicted','Matched'],colors)):
    bars=ax.bar(x+(j-1)*.24,summary[field],.24,label=label,color=c)
    ax.bar_label(bars,padding=3,fontsize=10)
ax.set_xticks(x,['80','81','82']);ax.set_xlabel('Original Sample18 frame');ax.set_ylabel('Nucleus centres');ax.set_ylim(0,122)
ax.set_title('Predictions versus generated-reference counts');ax.legend(ncols=3,loc='upper center',frameon=False)
save(fig,'reference_count_comparison.png','Grouped actual counts: 97/98/99 references, 100/102/99 predictions, 94 matches per frame.')

fig,ax=plt.subplots(figsize=(8.4,4.4),constrained_layout=True)
for j,(field,label,c) in enumerate(zip(['precision','recall','f1'],['Precision','Recall','F1'],colors)):
    bars=ax.bar(x+(j-1)*.24,100*summary[field],.24,label=label,color=c)
    ax.bar_label(bars,labels=[f'{v:.1f}' for v in 100*summary[field]],padding=3,fontsize=9)
ax.set_xticks(x,['80','81','82']);ax.set_xlabel('Original Sample18 frame');ax.set_ylabel('Agreement (%)');ax.set_ylim(0,119)
ax.set_title('Per-frame detection agreement (not tracking accuracy)');ax.legend(ncols=3,loc='upper center',frameon=False)
save(fig,'per_frame_metrics.png','Per-frame precision, recall and F1 against generated labels, using the same fixed match gate.')

fig,ax=plt.subplots(figsize=(8.4,4.4),constrained_layout=True)
for (frame,group),c in zip(matches.groupby('frame'),colors):
    ax.hist(group.distance_voxels,bins=np.linspace(0,3,16),histtype='step',linewidth=2,color=c,label=f'Frame {frame} (n={len(group)})')
ax.set(xlim=(0,3),xlabel='Accepted 3D match distance (voxels)',ylabel='Matched pairs',title='Localization-error distribution for accepted matches')
ax.legend(frameon=False)
save(fig,'localization_error_distribution.png','Measured 3D distances for the 282 accepted matches. Unmatched centres are not included.')

indexed=nodes.set_index('node_id');a=indexed.loc[edges.source_id];b=indexed.loc[edges.target_id]
motion=pd.DataFrame({'source_frame':a.t.to_numpy()+80,'source_id':edges.source_id.to_numpy(),'target_id':edges.target_id.to_numpy(),
    'distance_per_frame':np.linalg.norm(b[['z','y','x']].to_numpy()-a[['z','y','x']].to_numpy(),axis=1)/(b.t.to_numpy()-a.t.to_numpy())})
motion.to_csv(OUT/'link_displacements.csv',index=False)
fig,ax=plt.subplots(figsize=(8.4,4.4),constrained_layout=True)
for (frame,g),c in zip(motion.groupby('source_frame'),colors):
    ax.hist(g.distance_per_frame,bins=np.linspace(0,max(motion.distance_per_frame.max(),1),19),histtype='step',linewidth=2,color=c,label=f'{frame} to {frame+1} (n={len(g)})')
ax.set(xlabel='Voxel distance per frame (uncalibrated)',ylabel='Predicted links',title='Movement along the actual predicted links');ax.legend(frameon=False)
save(fig,'link_movement_distribution.png','All 198 predicted links split by transition; describes model/tracker output without certifying edge correctness.')

parents=edges.groupby('source_id').size();parent=int(parents[parents.eq(2)].index.min())
levels={parent:0};frontier=[parent]
for depth in (1,2):
    following=[]
    for node in frontier:
        children=edges.loc[edges.source_id.eq(node),'target_id'].astype(int).tolist()
        for child in children:levels[child]=depth;following.append(child)
    frontier=following
positions={}
for depth in range(3):
    ids=[n for n,d in levels.items() if d==depth]
    for n,xpos in zip(ids,np.linspace(.2,.8,len(ids)) if len(ids)>1 else [.5]):positions[n]=(xpos,1-depth*.4)
fig,ax=plt.subplots(figsize=(8.4,4.6),constrained_layout=True)
for row in edges.itertuples():
    if row.source_id in positions and row.target_id in positions:
        ax.annotate('',xy=positions[row.target_id],xytext=positions[row.source_id],arrowprops=dict(arrowstyle='->',color='#687B8B',lw=1.6,shrinkA=18,shrinkB=18))
for node,(xx,yy) in positions.items():
    rec=indexed.loc[node];label=f'Node {node}\nFrame {int(rec.t)+80}\n(z,y,x)=({rec.z:.0f},{rec.y:.0f},{rec.x:.0f})'
    ax.text(xx,yy,label,ha='center',va='center',fontsize=10,bbox=dict(boxstyle='round,pad=.55',fc='#EAF2F8' if node!=parent else '#FFF0D6',ec='#435F78'))
ax.set(xlim=(0,1),ylim=(-.05,1.18),title=f'Actual predicted family rooted at node {parent}');ax.axis('off')
save(fig,'predicted_family_graph.png',f'Actual parent/daughter and subsequent links rooted at node {parent}; unverified division hypothesis, not named biological identity.')
(OUT/'figure_provenance.json').write_text(json.dumps(dict(source='Saved real Sample18 pipeline outputs; no new training or fabricated control/variant data',input_sha256=hashlib.sha256((ROOT/'data/cshaper_demo/Sample18_t080_082.npy').read_bytes()).hexdigest(),figures=manifest),indent=2))
print(json.dumps({'figures':len(manifest),'matches':len(matches),'predicted_links':len(motion),'division_parent':parent},indent=2))
