from __future__ import annotations
import base64,json,os,time,zlib
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

DATA_B64='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'
UP=Path(os.environ.get('DROSOPHILA_MODEL_DIR','Drosophila_brain_model'))
COMP=UP/'Completeness_783.csv'; CONN=UP/'Connectivity_783.parquet'
V0=-52.; VRST=-52.; VTH=-45.; TMBR=20.; TAU=5.; TRFC=2.2; TDLY=1.8; WSYN=.275; DT=.1
DELAY=round(TDLY/DT); RFC=round(TRFC/DT)

def ragged(indptr,rows):
    starts=indptr[rows]; lens=indptr[rows+1]-starts; total=int(lens.sum())
    if total==0:return np.empty(0,np.int64)
    cum=np.cumsum(lens); off=np.repeat(starts-(cum-lens),lens)
    return np.arange(total,dtype=np.int64)+off

class Brain:
    def __init__(self,W,stim,read):
        W=W.tocsr(); W.sort_indices(); self.n=W.shape[0]; self.indptr=W.indptr.astype(np.int64,copy=False); self.indices=W.indices.astype(np.int32,copy=False); self.w=(W.data.astype(np.float64)*WSYN).astype(np.float32)
        self.stim=np.asarray(stim,np.int32); self.read=np.asarray(read,np.int32)
        a=1/TMBR; b=1/TAU; self.ev=np.float32(np.exp(-a*DT)); self.eg=np.float32(np.exp(-b*DT)); self.kg=np.float32(a*(np.exp(-b*DT)-np.exp(-a*DT))/(a-b)); self.v0rest=np.float32(V0*(1-float(self.ev)))
    def encode(self,x,sim_ms=20.):
        v=np.full(self.n,V0,np.float32); g=np.zeros(self.n,np.float32); rfc=np.zeros(self.n,np.int16); count=np.zeros(self.n,np.int16)
        active=np.empty(self.n,np.int32); mask=np.zeros(self.n,bool); ac=0; ring_t=[None]*(DELAY+1); ring_w=[None]*(DELAY+1); slot=0
        def activate(idx):
            nonlocal ac
            if idx is None or len(idx)==0:return
            idx=np.asarray(idx,np.int32); cand=idx[~mask[idx]]
            if not cand.size:return
            new=np.unique(cand); k=len(new); active[ac:ac+k]=new; ac+=k; mask[new]=True
        activate(self.stim)
        schedule={}; forced=np.zeros(len(self.stim),np.float32)
        for j,val in enumerate(np.asarray(x,float)):
            nsp=int(round(max(0.,min(4.,val))))
            forced[j]=nsp
            for q in range(nsp): schedule.setdefault(15+q*20,[]).append(j)
        snaps=[]; snap_steps={50,100,150,200}; steps=round(sim_ms/DT)
        for step in range(steps):
            targets=ring_t[slot]; weights=ring_w[slot]
            if targets is not None:
                np.add.at(g,targets,weights); activate(targets); ring_t[slot]=None; ring_w[slot]=None
            act=active[:ac]
            if act.size:
                rv=rfc[act]; free=act[rv==0]; refr=act[rv>0]
                if free.size:
                    gv=g[free].copy(); v[free]=v[free]*self.ev+self.v0rest+gv*self.kg; g[free]=gv*self.eg
                if refr.size:rfc[refr]-=1
            act=active[:ac]; spk=act[v[act]>VTH] if act.size else np.empty(0,np.int32)
            if step in schedule:
                ext=self.stim[np.asarray(schedule[step],np.int32)]; spk=np.unique(np.concatenate([spk,ext])).astype(np.int32)
            if spk.size:
                v[spk]=VRST; g[spk]=0; rfc[spk]=RFC; count[spk]+=1
                flat=ragged(self.indptr,spk)
                if flat.size:
                    ts=(slot+DELAY)%(DELAY+1); t=self.indices[flat].copy(); w=self.w[flat].copy()
                    if ring_t[ts] is None: ring_t[ts]=t; ring_w[ts]=w
                    else: ring_t[ts]=np.concatenate([ring_t[ts],t]); ring_w[ts]=np.concatenate([ring_w[ts],w])
            slot=(slot+1)%(DELAY+1)
            if step+1 in snap_steps:
                rr=self.read; snaps.append(np.concatenate([(v[rr]-V0)/10.,g[rr]/10.,count[rr].astype(np.float32)/4.]))
        return forced, np.concatenate(snaps).astype(np.float32), ac, int(count.sum())

def ridge(X,Y,a):
    if X.shape[1]<=X.shape[0]: return np.linalg.solve(X.T@X+a*np.eye(X.shape[1]),X.T@Y)
    return X.T@np.linalg.solve(X@X.T+a*np.eye(X.shape[0]),Y)

def standardize(A,B):
    mu=A.mean(0); sd=A.std(0); keep=sd>1e-8; return (A[:,keep]-mu[keep])/sd[keep],(B[:,keep]-mu[keep])/sd[keep],int(keep.sum())

def fit_eval(name,A,B,Y,ytest,valid,classes):
    best=None
    for a in np.logspace(-3,4,30):
        M=ridge(A[~valid],Y[~valid],a); va=float(np.mean(np.argmax(A[valid]@M,1)==np.argmax(Y[valid],1)))
        if best is None or va>best[0] or (va==best[0] and a>best[1]): best=(va,float(a))
    M=ridge(A,Y,best[1]); scores=B@M; pred=np.argmax(scores,1); acc=float(np.mean(pred==ytest))
    rows=[]
    for i,p in enumerate(pred):
        ss=np.sort(scores[i]); rows.append({'id':data['test'][i]['id'],'expected':classes[ytest[i]],'predicted':classes[p],'correct':bool(p==ytest[i]),'margin':float(ss[-1]-ss[-2])})
    out={'validation_accuracy':best[0],'alpha':best[1],'test_accuracy':acc,'feature_count':int(A.shape[1]),'predictions':rows}; print(name+'='+json.dumps(out,ensure_ascii=False),flush=True); return out

if __name__=='__main__':
    data=json.loads(zlib.decompress(base64.b64decode(DATA_B64)).decode()); classes=data['classes']; X=np.asarray([r['x'] for r in data['train']],float); Xt=np.asarray([r['x'] for r in data['test']],float); valid=np.asarray(data['valid'],bool)
    Y=np.zeros((len(X),len(classes))); [Y.__setitem__((i,classes.index(r['label'])),1.) for i,r in enumerate(data['train'])]; ytest=np.asarray([classes.index(r['label']) for r in data['test']],int)
    comp=pd.read_csv(COMP,index_col=0); con=pd.read_parquet(CONN,columns=['Presynaptic_Index','Postsynaptic_Index','Excitatory x Connectivity']); n=len(comp); pre=con['Presynaptic_Index'].to_numpy(np.int32,copy=False); post=con['Postsynaptic_Index'].to_numpy(np.int32,copy=False); wt=con['Excitatory x Connectivity'].to_numpy(np.int32,copy=False); W=sp.csr_matrix((wt,(pre,post)),shape=(n,n),dtype=np.int32); W.sum_duplicates(); W.sort_indices()
    avail=np.flatnonzero(np.diff(W.indptr)>0); stim=avail[:X.shape[1]].astype(np.int32); f1=ragged(W.indptr.astype(np.int64,copy=False),stim.astype(np.int64)); one=np.unique(W.indices[f1]) if f1.size else np.empty(0,np.int32); f2=ragged(W.indptr.astype(np.int64,copy=False),one[:256].astype(np.int64)) if one.size else np.empty(0,np.int64); two=np.unique(W.indices[f2]) if f2.size else np.empty(0,np.int32); read=np.unique(np.concatenate([stim,one,two]))[:512].astype(np.int32); brain=Brain(W,stim,read)
    print(f'V5_BRAIN neurons={n} edges={W.nnz} input_neurons={len(stim)} read_nodes={len(read)} train={len(X)} heldout_9mo={len(Xt)}',flush=True)
    def enc(rows,label):
        io=[]; ds=[]; act=[]; spk=[]; t=time.perf_counter()
        for i,x in enumerate(rows):
            a,b,c,d=brain.encode(x); io.append(a); ds.append(b); act.append(c); spk.append(d)
            if (i+1)%20==0 or i+1==len(rows): print(f'{label} {i+1}/{len(rows)}',flush=True)
        return np.asarray(io,float),np.asarray(ds,float),act,spk,time.perf_counter()-t
    I,D,atr,strn,tw=enc(X,'train'); It,Dt,ate,ste,tew=enc(Xt,'test')
    R,Rt,rk=standardize(X,Xt); NI,NIt,nik=standardize(I,It); DS,DSt,dsk=standardize(D,Dt)
    # preserve the exact spike input channel and add a small amount of downstream state.
    FULL=np.concatenate([NI,0.10*DS],1); FULLt=np.concatenate([NIt,0.10*DSt],1)
    results={}
    results['raw_structured']=fit_eval('RAW_STRUCTURED',R,Rt,Y,ytest,valid,classes)
    results['fly_input_spike_code']=fit_eval('FLY_INPUT_SPIKE_CODE',NI,NIt,Y,ytest,valid,classes)
    results['fly_downstream_only']=fit_eval('FLY_DOWNSTREAM_ONLY',DS,DSt,Y,ytest,valid,classes)
    results['fly_full_hybrid']=fit_eval('FLY_FULL_HYBRID',FULL,FULLt,Y,ytest,valid,classes)
    result={'source_commit':os.environ.get('UPSTREAM_COMMIT'),'graph_neurons':n,'unique_edges':int(W.nnz),'train_examples':len(X),'heldout_actual_2027_september_items':len(Xt),'stimulated_real_neurons':stim.tolist(),'read_nodes':len(read),'mean_active_train':float(np.mean(atr)),'mean_spikes_train':float(np.mean(strn)),'mean_active_test':float(np.mean(ate)),'mean_spikes_test':float(np.mean(ste)),'brain_train_wall_s':tw,'brain_test_wall_s':tew,'raw_features':rk,'neural_input_features':nik,'downstream_features':dsk,'results':results,'scope_note':'Training uses generated N-je strategy examples, generic strategy tutoring, and selected June concept summaries. The seven September items are excluded from fitting. This tests first-strategy transfer, not autonomous symbolic derivation of final numeric answers. The lossless input spike-code channel is an external neural interface; downstream-only score measures what survives through the fixed biological connectome.'}
    Path('suneung_transfer_v5_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)); print('V5_FINAL='+json.dumps(result,ensure_ascii=False),flush=True)
