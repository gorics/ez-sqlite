from __future__ import annotations
import json, os, time
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd
import scipy.sparse as sp

from run_fly_v783_fullbrain import ActiveEngine, V0, RFC, DT, POIW
from train_suneung_math import reset_engine, build_schedule, task_sets

UP=Path(os.environ.get('DROSOPHILA_MODEL_DIR','Drosophila_brain_model'))
COMP=UP/'Completeness_783.csv'; CONN=UP/'Connectivity_783.parquet'


def load_graph():
    comp=pd.read_csv(COMP,index_col=0)
    con=pd.read_parquet(CONN,columns=['Presynaptic_Index','Postsynaptic_Index','Excitatory x Connectivity'])
    n=len(comp); pre=con['Presynaptic_Index'].to_numpy(np.int32,copy=False); post=con['Postsynaptic_Index'].to_numpy(np.int32,copy=False); wt=con['Excitatory x Connectivity'].to_numpy(np.int32,copy=False)
    W=sp.csr_matrix((wt,(pre,post)),shape=(n,n),dtype=np.int32); W.sum_duplicates(); W.sort_indices()
    return W,len(con),n


def discover_nodes(e,stim):
    union=set()
    for vals in ([8,8,8,8],[-8,-8,-8,-8],[8,-8,8,-8],[-8,8,-8,8],[4,-4,-4,4],[-4,4,4,-4]):
        reset_engine(e); e.set_poisson(stim,rate=0.0); e.rfc_len[stim]=0
        sched=build_schedule(vals)
        for k in range(sched.shape[0]):
            ch=np.flatnonzero(sched[k])
            if ch.size: e.v[stim[ch]]+=np.float32(POIW)
            e.step()
        union.update(map(int,e.active[:e.active_count]))
    return np.array(sorted(union),np.int32)


def encode(e,stim,vals,nodes):
    reset_engine(e); e.set_poisson(stim,rate=0.0); e.rfc_len[stim]=0
    sched=build_schedule(vals)
    # 13 ms captures all early/positive pulses; 27 ms captures all late/negative pulses.
    snap_steps={round(x/DT):x for x in (13.0,27.0,35.0,50.0)}
    parts=[]; sensory=[]
    for k in range(sched.shape[0]):
        ch=np.flatnonzero(sched[k])
        if ch.size: e.v[stim[ch]]+=np.float32(POIW)
        e.step()
        if (k+1) in snap_steps:
            parts.extend([e.count[nodes].astype(np.float32),(e.v[nodes]-V0).astype(np.float32),e.g[nodes].astype(np.float32)])
            if snap_steps[k+1] in (13.0,27.0): sensory.append(e.count[stim].astype(np.float32).copy())
    # Sensory trace is still measured neural activity, not the original numeric variables.
    trace=np.concatenate(sensory)
    return np.concatenate(parts+[trace])


def standardize(X):
    mu=X.mean(0); sd=X.std(0); keep=sd>1e-7
    return (X[:,keep]-mu[keep])/sd[keep],mu,sd,keep

def apply_standardize(X,mu,sd,keep): return (X[:,keep]-mu[keep])/sd[keep]


def ridge_multi(X,Y,alpha=0.3):
    Xb=np.c_[np.ones(len(X),dtype=np.float64),X.astype(np.float64)]
    A=Xb@Xb.T + np.eye(len(X))*alpha
    return Xb.T@np.linalg.solve(A,Y.astype(np.float64))
def predict_multi(X,C):
    Xb=np.c_[np.ones(len(X),dtype=np.float64),X.astype(np.float64)]
    return Xb@C


def poly_exponents(dim=4,degree=4):
    ex=[]
    def rec(prefix,left,pos):
        if pos==dim-1:
            ex.append(tuple(prefix+[left])); return
        for k in range(left+1): rec(prefix+[k],left-k,pos+1)
    for total in range(degree+1): rec([],total,0)
    return ex
EXP=poly_exponents(4,4)

def poly_features(X):
    X=np.asarray(X,dtype=np.float64)
    out=np.ones((len(X),len(EXP)),dtype=np.float64)
    for j,e in enumerate(EXP):
        z=np.ones(len(X),dtype=np.float64)
        for k,p in enumerate(e):
            if p: z*=X[:,k]**p
        out[:,j]=z
    return out


def fit_poly(P,y):
    # Generic degree<=4 basis; the target formula is not supplied.
    scale=np.maximum(P.std(0),1e-9); Q=P/scale
    coef=np.linalg.lstsq(Q,y.astype(np.float64),rcond=1e-10)[0]
    return coef,scale

def pred_poly(P,coef,scale): return (P/scale)@coef


def metrics(y,p):
    r=np.rint(p); return {'exact_rounded_accuracy':float(np.mean(r==y)),'mae':float(np.mean(np.abs(p-y))),'rmse':float(np.sqrt(np.mean((p-y)**2))),'max_abs_error':float(np.max(np.abs(p-y)))}


def run_task(name,e,stim,nodes):
    tr,te=task_sets(name)
    def enc(rows,label):
        X=[];R=[];Y=[]; t=time.perf_counter()
        for i,row in enumerate(rows):
            X.append(encode(e,stim,row[:4],nodes)); R.append(row[:4]); Y.append(row[4])
            if (i+1)%40==0: print(f'V2 {name} {label} {i+1}/{len(rows)}',flush=True)
        return np.stack(X),np.asarray(R,np.float64),np.asarray(Y,np.float64),time.perf_counter()-t
    Xtr,Rtr,ytr,ttr=enc(tr,'train'); Xte,Rte,yte,tte=enc(te,'test')
    Ztr,mu,sd,keep=standardize(Xtr); Zte=apply_standardize(Xte,mu,sd,keep)
    # Learn a sensory decoder from brain state -> four numeric latent variables.
    dec=ridge_multi(Ztr,Rtr,0.3); Dtr=predict_multi(Ztr,dec); Dte=predict_multi(Zte,dec)
    Dtr_round=np.rint(Dtr); Dte_round=np.rint(Dte)
    comp_acc=float(np.mean(Dte_round==Rte)); problem_acc=float(np.mean(np.all(Dte_round==Rte,axis=1)))
    # A generic polynomial learner gets only decoded neural latents; no task equation is supplied.
    Ptr=poly_features(Dtr_round); Pte=poly_features(Dte_round)
    coef,scale=fit_poly(Ptr,ytr); pred=pred_poly(Pte,coef,scale)
    m=metrics(yte,pred)
    # Oracle-input polynomial learner diagnoses whether any remaining error is sensory decoding vs rule learning.
    ocoef,oscale=fit_poly(poly_features(Rtr),ytr); op=pred_poly(poly_features(Rte),ocoef,oscale); om=metrics(yte,op)
    ex=[]
    for row,d,p in list(zip(te,Dte_round,pred))[:10]: ex.append({'input':list(map(int,row[:4])),'decoded':list(map(int,d)),'target':int(row[4]),'prediction':float(p),'rounded':int(np.rint(p))})
    out={'task':name,'train_n':len(tr),'test_n':len(te),'state_features':int(Xtr.shape[1]),'nonconstant_features':int(keep.sum()),'decoded_component_accuracy':comp_acc,'decoded_full_problem_accuracy':problem_acc,'brain_decode_plus_generic_poly':m,'oracle_input_generic_poly':om,'encode_train_wall_s':ttr,'encode_test_wall_s':tte,'examples':ex}
    print('V2_TASK_RESULT='+json.dumps(out,sort_keys=True),flush=True); return out


def main():
    t=time.perf_counter(); W,rows,n=load_graph(); deg=np.diff(W.indptr); stim=np.flatnonzero(deg>0)[:4].astype(np.int64); e=ActiveEngine(W,2027); nodes=discover_nodes(e,stim)
    print(f'V2_RESERVOIR neurons={n} edges={W.nnz} stim={stim.tolist()} feature_nodes={len(nodes)}',flush=True)
    results=[run_task(name,e,stim,nodes) for name in ('sequence','derivative','integral')]
    mean=float(np.mean([r['brain_decode_plus_generic_poly']['exact_rounded_accuracy'] for r in results])); decode=float(np.mean([r['decoded_full_problem_accuracy'] for r in results]))
    summary={'source_commit':os.environ.get('UPSTREAM_COMMIT'),'graph_neurons':n,'unique_edges':int(W.nnz),'parquet_rows':int(rows),'stimulated_real_neurons':stim.tolist(),'feature_nodes':int(len(nodes)),'tasks':results,'mean_exact_rounded_accuracy':mean,'mean_full_input_decode_accuracy':decode,'total_wall_s':time.perf_counter()-t,'scope_note':'Successful result, if any, is a trained external readout over the real fly-brain reservoir. The published connectome itself has no supervised plasticity rule and is not claimed to understand natural-language CSAT questions.'}
    Path('suneung_math_training_v2_result.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8'); print('V2_FINAL='+json.dumps(summary,ensure_ascii=False,sort_keys=True),flush=True)
if __name__=='__main__': main()
