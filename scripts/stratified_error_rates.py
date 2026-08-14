from __future__ import annotations
import argparse, csv, hashlib, json, subprocess, time
from pathlib import Path
import h5py, numpy as np
from scipy.stats import fisher_exact
from maskedbike_ml.gjs import distance_spectrum

parser=argparse.ArgumentParser(description='Stratify frozen held-out classifier errors by BIKE distance-spectrum overlap.')
parser.add_argument('--dataset',required=True,type=Path)
parser.add_argument('--run-dir',required=True,type=Path)
parser.add_argument('--helper',required=True,type=Path)
parser.add_argument('--key-seed',required=True)
parser.add_argument('--output',required=True,type=Path)
parser.add_argument('--block-length',type=int,default=12323)
parser.add_argument('--bootstrap',type=int,default=5000)
args=parser.parse_args()
ROOT=args.dataset.resolve(); RUN=args.run_dir.resolve(); OUT=args.output.resolve(); HELPER=args.helper.resolve(); KEY=args.key_seed
R=args.block_length
if len(bytes.fromhex(KEY)) != 32: raise ValueError('key seed must be 32 bytes')
THRESH=json.load(open(RUN/'final-results.json'))['result']['threshold']
OUT.mkdir(parents=True,exist_ok=True)

def positions(s): return np.fromstring(s,sep=',',dtype=np.int32) if s else np.empty(0,np.int32)
def pair_mult(pos):
    p=np.unique(pos); a,b=np.triu_indices(len(p),1); d=np.abs(p[a]-p[b]); d=np.minimum(d,R-d)
    return np.bincount(d,minlength=R//2+1)
def ci_boot(y,p,w=None,B=5000,seed=52):
    y=np.asarray(y); p=np.asarray(p); w=np.ones(len(y)) if w is None else np.asarray(w,float)
    rng=np.random.default_rng(seed); out={}
    for val,name,correct in [(1,'tpr',1),(0,'tnr',0)]:
      ix=np.flatnonzero(y==val)
      if len(ix)==0: out[name]=[None,None,None]; continue
      point=float(np.average(p[ix]==correct,weights=w[ix])); vals=np.empty(B)
      for b in range(B):
        j=rng.choice(ix,len(ix),replace=True); vals[b]=np.average(p[j]==correct,weights=w[j])
      out[name]=[point,float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]
    return out

keyline=subprocess.check_output([str(HELPER),KEY,'--key-only'],text=True).strip().split('\t')
h0,h1=map(positions,keyline)
km=pair_mult(h0)+pair_mult(h1); km[0]=0
key_present=km>0
key_counts={str(g):int(np.sum(km==g)) for g in range(1,int(km.max())+1)}

snap=json.load(open(RUN/'final-dataset-snapshot.json'))
z=np.load(RUN/'final-80000-semisupervised-predictions.npz')
prob=z['r2_probability'].astype(float); truth=z['r2_truth'].astype(np.uint8); pred=(prob>=THRESH).astype(np.uint8)
rows=[]; index=0
proc=subprocess.Popen([str(HELPER),KEY],text=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
t0=time.time()
for bundle in snap['r2_heldout']['bundle_rows']:
    path=ROOT/bundle['h5_path']
    with h5py.File(path,'r') as h:
      seeds=np.asarray(h['case_seeds'],np.uint8); cts=np.asarray(h['ciphertexts'],np.uint8)
      hws=np.asarray(h['hamming_weights']); tids=np.asarray(h['trace_ids']).astype('U')
    for local,(seed,ct,hw,tid) in enumerate(zip(seeds,cts,hws,tids)):
      sh=seed.tobytes().hex(); proc.stdin.write(sh+'\n'); proc.stdin.flush(); line=proc.stdout.readline()
      rs,canonical,e0s,e1s=line.rstrip('\n').split('\t')
      if rs!=sh: raise RuntimeError('seed mismatch')
      # Verify canonical ciphertext against transport format.
      raw=bytes.fromhex(canonical); transport=bytearray(1576); transport[:1541]=raw[:1541]; transport[1544:]=raw[1541:]
      if bytes(transport)!=ct.tobytes(): raise RuntimeError(f'ciphertext mismatch {path}:{local}')
      e0,e1=positions(e0s),positions(e1s)
      present=np.zeros(len(km),bool); present[distance_spectrum(e0,R)]=1; present[distance_spectrum(e1,R)]=1
      ov_unique=int(np.sum(present & key_present)); ov_mult=int(np.sum(km[present]))
      by_mult={g:int(np.sum(present & (km==g))) for g in range(0,int(km.max())+1)}
      rows.append({'global_index':index,'h5_path':bundle['h5_path'],'h5_row':local,'trace_id':tid,
        'case_seed':sh,'ciphertext_sha256':hashlib.sha256(ct.tobytes()).hexdigest(),'delta':int(hw==0),
        'delta_hat':int(pred[index]),'probability_hw_zero':float(prob[index]),'hamming_weight':int(hw),
        'e0_weight':len(e0),'e1_weight':len(e1),'overlap_unique':ov_unique,'overlap_multiplicity_sum':ov_mult,
        **{f'present_key_mult_{g}':v for g,v in by_mult.items()}})
      index+=1
      if index%1000==0: print('processed',index,'seconds',round(time.time()-t0,1),flush=True)
proc.stdin.close(); rc=proc.wait()
if rc or index!=len(truth): raise RuntimeError((rc,index,len(truth)))
if not np.array_equal(np.array([r['delta'] for r in rows]),truth): raise RuntimeError('truth order mismatch')

# Exact requested trace-level groups.
ov=np.array([r['overlap_unique'] for r in rows]); y=truth; ph=pred
cats=np.where(ov==0,'0',np.where(ov==1,'1',np.where(ov==2,'2','>=3')))
def table(groups,names):
 out=[]
 for name in names:
  ix=np.flatnonzero(groups==name); yy=y[ix]; pp=ph[ix]
  ci=ci_boot(yy,pp,B=args.bootstrap,seed=52+len(out)) if len(ix) else {'tpr':[None]*3,'tnr':[None]*3}
  out.append({'group':name,'n':len(ix),'delta_1':int(np.sum(yy==1)),'delta_0':int(np.sum(yy==0)),
    'tp':int(np.sum((yy==1)&(pp==1))),'fn':int(np.sum((yy==1)&(pp==0))),
    'tn':int(np.sum((yy==0)&(pp==0))),'fp':int(np.sum((yy==0)&(pp==1))),
    'tpr':ci['tpr'][0],'tpr_ci95':[ci['tpr'][1],ci['tpr'][2]],'tnr':ci['tnr'][0],'tnr_ci95':[ci['tnr'][1],ci['tnr'][2]]})
 return out
literal=table(cats,['0','1','2','>=3'])

# Exploratory equal-frequency overlap strata, useful because literal groups may be degenerate.
edges=np.quantile(ov,[0,.25,.5,.75,1]); q=np.digitize(ov,edges[1:-1],right=True); qnames=np.array(['Q1','Q2','Q3','Q4'])[q]
quant=table(qnames,['Q1','Q2','Q3','Q4'])
# Fisher each quantile vs Q1, separately on class-conditional correct/error counts.
base={r['group']:r for r in quant}
for row in quant:
 if row['group']=='Q1': row['fisher_tpr_vs_q1_p']=1.0; row['fisher_tnr_vs_q1_p']=1.0
 else:
  b=base['Q1']; row['fisher_tpr_vs_q1_p']=float(fisher_exact([[row['tp'],row['fn']],[b['tp'],b['fn']]]).pvalue)
  row['fisher_tnr_vs_q1_p']=float(fisher_exact([[row['tn'],row['fp']],[b['tn'],b['fp']]]).pvalue)

fields=list(rows[0])
with open(OUT/'heldout-trace-overlap.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
report={'schema':'maskedbike-stratified-error-rate.v1','dataset':'D1 first-order / two shares','heldout_traces':len(rows),
 'positive_event':'delta=1 iff hamming_weight==0 (decryptable)','threshold':THRESH,
 'prediction_source':str((RUN/'final-80000-semisupervised-predictions.npz').resolve()),
 'snapshot_source':str((RUN/'final-dataset-snapshot.json').resolve()),
 'distance_definition':'union of unique nonzero minimal cyclic pair distances in reconstructed e0 and e1',
 'key_spectrum_definition':'pair-distance multiplicities summed across h0 and h1','key_support_weights':[len(h0),len(h1)],
 'key_distance_multiplicity_distribution':key_counts,'overlap_unique_range':[int(ov.min()),int(ov.max())],
 'requested_groups':literal,'exploratory_overlap_quantile_edges':edges.tolist(),'exploratory_groups':quant,
 'interpretation':('requested 0/1/2/>=3 trace-level grouping is degenerate' if len(set(cats))==1 else 'requested grouping is estimable')}
(OUT/'stratified-error-rates.json').write_text(json.dumps(report,indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(7.2,4.4))
x=np.arange(4); tpr=np.array([v['tpr'] for v in quant]); tnr=np.array([v['tnr'] for v in quant])
tprerr=np.array([[tpr[i]-quant[i]['tpr_ci95'][0],quant[i]['tpr_ci95'][1]-tpr[i]] for i in range(4)]).T
tnrerr=np.array([[tnr[i]-quant[i]['tnr_ci95'][0],quant[i]['tnr_ci95'][1]-tnr[i]] for i in range(4)]).T
ax.errorbar(x-.07,tpr,yerr=tprerr,fmt='o-',capsize=4,label='TPR (delta=1)')
ax.errorbar(x+.07,tnr,yerr=tnrerr,fmt='s-',capsize=4,label='TNR (delta=0)')
ax.set_xticks(x,[f"Q{i+1}\n(n={quant[i]['n']})" for i in range(4)]); ax.set_ylim(.84,.97)
ax.set_ylabel('Class-conditional correct rate'); ax.set_xlabel('Ciphertext/key overlap-count quartile')
ax.grid(axis='y',alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(OUT/'overlap-quartile-tpr-tnr.png',dpi=180); plt.close(fig)
# compact markdown
with open(OUT/'REPORT.md','w') as f:
 f.write('# D1 held-out stratified error-rate analysis\n\n')
 f.write(f'- Frozen R2 held-out traces: {len(rows)}\n- Positive event: `delta=1 iff HW=0`\n- Frozen threshold: `{THRESH:.10f}`\n')
 f.write(f'- Literal overlap range: {ov.min()}–{ov.max()} unique key-spectrum distances per ciphertext.\n\n')
 f.write('## Requested groups\n\n|group|n|TPR (95% bootstrap CI)|TNR (95% bootstrap CI)|\n|---:|---:|---:|---:|\n')
 for x in literal:
  fmt=lambda v: 'NA' if v is None else f'{v:.4f}'
  f.write(f"|{x['group']}|{x['n']}|{fmt(x['tpr'])} [{fmt(x['tpr_ci95'][0])}, {fmt(x['tpr_ci95'][1])}]|{fmt(x['tnr'])} [{fmt(x['tnr_ci95'][0])}, {fmt(x['tnr_ci95'][1])}]|\n")
 f.write('\n## Exploratory overlap-count quartiles\n\n|group|n|TPR|TNR|Fisher TPR vs Q1|Fisher TNR vs Q1|\n|---|---:|---:|---:|---:|---:|\n')
 for x in quant: f.write(f"|{x['group']}|{x['n']}|{x['tpr']:.4f}|{x['tnr']:.4f}|{x['fisher_tpr_vs_q1_p']:.3g}|{x['fisher_tnr_vs_q1_p']:.3g}|\n")
 f.write('\nThe requested categorical test is not identifiable on this valid-ciphertext held-out set because every ciphertext falls in `>=3`. Quartile results are exploratory and do not replace a GJS chosen-ciphertext stratification.\n')
print(json.dumps({'out':str(OUT.resolve()),'literal':literal,'quantiles':quant,'range':[int(ov.min()),int(ov.max())]},indent=2))