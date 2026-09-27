from __future__ import annotations
import json, os, time
from pathlib import Path
import numpy as np
from train_suneung_math import task_sets
from train_suneung_math_v2 import load_graph, discover_nodes, encode, standardize, apply_standardize, ridge_multi, predict_multi
from run_fly_v783_fullbrain import ActiveEngine


def exponents(dim,degree):
    out=[]
    def rec(prefix,left,pos):
        if pos==dim-1: out.append(tuple(prefix+[left])); return
        for k in range(left+1): rec(prefix+[k],left-k,pos+1)
    for total in range(degree+1): rec([],total,0)
    return out

def poly(X,degree):
    X=np.asarray(X,float); E=exponents(X.shape[1],degree); P=np.ones((len(X),len(E)),float)
    for j,e in enumerate(E):
        z=np.ones(len(X),float)
        for k,p in enumerate(e):
            if p: z*=X[:,k]**p
        P[:,j]=z
    return P

def fit(P,y):
    s=P.std(0); s[s<1e-10]=1.0
    Q=P/s
    c=np.linalg.lstsq(Q,y.astype(float),rcond=1e-12)[0]
    return c,s

def pred(P,c,s): return (P/s)@c

def score(y,p):
    return float(np.mean(np.rint(p)==y)), float(np.mean(np.abs(p-y)))

def metrics(y,p):
    return {'exact_rounded_accuracy':float(np.mean(np.rint(p)==y)),'mae':float(np.mean(np.abs(p-y))),'rmse':float(np.sqrt(np.mean((p-y)**2))),'max_abs_error':float(np.max(np.abs(p-y)))}

def task(name,e,stim,nodes):
    tr,te=task_sets(name)
    def enc(rows):
        X=[];R=[];Y=[]
        for row in rows:
            X.append(encode(e,stim,row[:4],nodes)); R.append(row[:4]); Y.append(row[4])
        return np.stack(X),np.asarray(R,float),np.asarray(Y,float)
    Xtr,Rtr,ytr=enc(tr); Xte,Rte,yte=enc(te)
    Ztr,mu,sd,keep=standardize(Xtr); Zte=apply_standardize(Xte,mu,sd,keep)
    dec=ridge_multi(Ztr,Rtr,0.3); Dtr=np.rint(predict_multi(Ztr,dec)); Dte=np.rint(predict_multi(Zte,dec))
    decode=float(np.mean(np.all(Dte==Rte,axis=1)))
    cut=120; best=None
    for degree in (1,2,3,4):
        c,s=fit(poly(Dtr[:cut],degree),ytr[:cut]); pv=pred(poly(Dtr[cut:],degree),c,s); acc,mae=score(ytr[cut:],pv)
        key=(-acc,mae,degree)
        if best is None or key<best[0]: best=(key,degree)
    degree=best[1]; c,s=fit(poly(Dtr,degree),ytr); p=pred(poly(Dte,degree),c,s)
    out={'task':name,'selected_degree':degree,'decoded_full_problem_accuracy':decode,'test':metrics(yte,p)}
    print('V3_TASK='+json.dumps(out,sort_keys=True),flush=True); return out

def main():
    t=time.perf_counter(); W,rows,n=load_graph(); stim=np.flatnonzero(np.diff(W.indptr)>0)[:4].astype(np.int64); e=ActiveEngine(W,2027); nodes=discover_nodes(e,stim)
    res=[task(x,e,stim,nodes) for x in ('sequence','derivative','integral')]
    summary={'source_commit':os.environ.get('UPSTREAM_COMMIT'),'graph_neurons':n,'unique_edges':int(W.nnz),'stimulated_real_neurons':stim.tolist(),'feature_nodes':int(len(nodes)),'tasks':res,'mean_exact_rounded_accuracy':float(np.mean([r['test']['exact_rounded_accuracy'] for r in res])),'total_wall_s':time.perf_counter()-t,'scope_note':'Trained external decoder/readout over the real v783 fly-brain reservoir; not a claim that the biological connectome itself acquired natural-language mathematical understanding.'}
    Path('suneung_math_training_v3_result.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8'); print('V3_FINAL='+json.dumps(summary,ensure_ascii=False,sort_keys=True),flush=True)
if __name__=='__main__': main()
