from __future__ import annotations
import json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

UP=Path(os.environ.get('DROSOPHILA_MODEL_DIR','Drosophila_brain_model'))
DATA_B64='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'
COMP=UP/'Completeness_783.csv'; CONN=UP/'Connectivity_783.parquet'
V0=-52.0; VRST=-52.0; VTH=-45.0; TMBR=20.0; TAU=5.0; TRFC=2.2; TDLY=1.8; WSYN=0.275; DT=0.1
DELAY=round(TDLY/DT); RFC=round(TRFC/DT)

def ragged(indptr,rows):
    starts=indptr[rows]; lens=indptr[rows+1]-starts; total=int(lens.sum())
    if total==0:return np.empty(0,np.int64)
    cum=np.cumsum(lens); offsets=np.repeat(starts-(cum-lens),lens)
    return np.arange(total,dtype=np.int64)+offsets

class Brain:
    def __init__(self,W,stim,read_nodes):
        W=W.tocsr(); W.sort_indices(); self.n=W.shape[0]
        self.indptr=W.indptr.astype(np.int64,copy=False); self.indices=W.indices.astype(np.int32,copy=False)
        self.w=(W.data.astype(np.float64)*WSYN).astype(np.float32)
        self.stim=np.asarray(stim,np.int32); self.read=np.asarray(read_nodes,np.int32)
        a=1/TMBR; b=1/TAU
        self.ev=np.float32(np.exp(-a*DT)); self.eg=np.float32(np.exp(-b*DT)); self.kg=np.float32(a*(np.exp(-b*DT)-np.exp(-a*DT))/(a-b)); self.v0rest=np.float32(V0*(1-float(self.ev)))
    def encode(self,x,sim_ms=32.0):
        v=np.full(self.n,V0,np.float32); g=np.zeros(self.n,np.float32); rfc=np.zeros(self.n,np.int16); count=np.zeros(self.n,np.int16)
        active=np.empty(self.n,np.int32); mask=np.zeros(self.n,bool); ac=0
        ring_t=[None]*(DELAY+1); ring_w=[None]*(DELAY+1); slot=0
        def activate(idx):
            nonlocal ac
            if idx is None or len(idx)==0:return
            idx=np.asarray(idx,np.int32); cand=idx[~mask[idx]]
            if cand.size==0:return
            new=np.unique(cand); k=len(new); active[ac:ac+k]=new; ac+=k; mask[new]=True
        activate(self.stim)
        x=np.asarray(x,float); mx=float(x.max(initial=0.0)); schedule={}
        if mx>0:
            rel=x/mx
            for j,r in enumerate(rel):
                if r<=0: continue
                nsp=max(1,min(4,int(np.ceil(r*4)))); base=max(1,int(round((1.0-r)*40)))
                for q in range(nsp): schedule.setdefault(base+q*25,[]).append(j)
        snaps=[]; snap_steps={80,160,240,320}; steps=round(sim_ms/DT)
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
                forced=self.stim[np.asarray(schedule[step],np.int32)]; spk=np.unique(np.concatenate([spk,forced])).astype(np.int32)
            if spk.size:
                v[spk]=VRST; g[spk]=0; rfc[spk]=RFC; count[spk]+=1
                flat=ragged(self.indptr,spk)
                if flat.size:
                    ts=(slot+DELAY)%(DELAY+1); t=self.indices[flat].copy(); w=self.w[flat].copy()
                    if ring_t[ts] is None: ring_t[ts]=t; ring_w[ts]=w
                    else: ring_t[ts]=np.concatenate([ring_t[ts],t]); ring_w[ts]=np.concatenate([ring_w[ts],w])
            slot=(slot+1)%(DELAY+1)
            if step+1 in snap_steps:
                rr=self.read; snaps.append(np.concatenate([(v[rr]-V0)/10.0,g[rr]/10.0,count[rr].astype(np.float32)/4.0]))
        if not snaps:
            rr=self.read; snaps=[np.concatenate([(v[rr]-V0)/10.0,g[rr]/10.0,count[rr].astype(np.float32)/4.0])]
        return np.concatenate(snaps).astype(np.float32), ac, int(count.sum())

def ridge_fit(X,Y,alpha): return np.linalg.solve(X.T@X+alpha*np.eye(X.shape[1]),X.T@Y)

def choose_alpha(X,Y,labels,valid_mask,alphas):
    tr=~valid_mask; va=valid_mask; best=None
    for a in alphas:
        W=ridge_fit(X[tr],Y[tr],a); pred=np.argmax(X[va]@W,axis=1); true=np.argmax(Y[va],axis=1); acc=float(np.mean(pred==true))
        if best is None or acc>best[0] or (acc==best[0] and a>best[1]): best=(acc,float(a))
    return best

def main():
    import zlib,base64
    d=json.loads(zlib.decompress(base64.b64decode(DATA_B64)).decode()); classes=d['classes']; train=d['train']; test=d['test']
    Xraw=np.asarray([r['x'] for r in train],np.float64); Xtestraw=np.asarray([r['x'] for r in test],np.float64)
    Y=np.zeros((len(train),len(classes)),np.float64)
    for i,r in enumerate(train):Y[i,classes.index(r['label'])]=1.0
    ytest=np.asarray([classes.index(r['label']) for r in test],int)
    comp=pd.read_csv(COMP,index_col=0); con=pd.read_parquet(CONN,columns=['Presynaptic_Index','Postsynaptic_Index','Excitatory x Connectivity'])
    n=len(comp); pre=con['Presynaptic_Index'].to_numpy(np.int32,copy=False); post=con['Postsynaptic_Index'].to_numpy(np.int32,copy=False); wt=con['Excitatory x Connectivity'].to_numpy(np.int32,copy=False)
    W=sp.csr_matrix((wt,(pre,post)),shape=(n,n),dtype=np.int32); W.sum_duplicates(); W.sort_indices()
    avail=np.flatnonzero(np.diff(W.indptr)>0); stim=avail[:64].astype(np.int32)
    flat1=ragged(W.indptr.astype(np.int64,copy=False),stim.astype(np.int64)); one=np.unique(W.indices[flat1]) if flat1.size else np.empty(0,np.int32)
    seed2=one[:256].astype(np.int64); flat2=ragged(W.indptr.astype(np.int64,copy=False),seed2) if seed2.size else np.empty(0,np.int64); two=np.unique(W.indices[flat2]) if flat2.size else np.empty(0,np.int32)
    read=np.unique(np.concatenate([stim,one,two]))[:512].astype(np.int32); brain=Brain(W,stim,read)
    print(f'STRATEGY_V4_BRAIN neurons={n} edges={W.nnz} stim={len(stim)} read={len(read)} train={len(train)} test={len(test)}',flush=True)
    def enc(rows,name):
        out=[]; active=[]; spikes=[]; t=time.perf_counter()
        for i,r in enumerate(rows):
            z,a,s=brain.encode(r['x']);out.append(z);active.append(a);spikes.append(s)
            if (i+1)%20==0 or i+1==len(rows):print(f'{name} encoded {i+1}/{len(rows)}',flush=True)
        return np.asarray(out,np.float64),active,spikes,time.perf_counter()-t
    Xb,atr,sps,trwall=enc(train,'train'); Xtb,ate,spte,tewall=enc(test,'test')
    def std_pair(A,B):
        mu=A.mean(0); sd=A.std(0); keep=sd>1e-7; return (A[:,keep]-mu[keep])/sd[keep],(B[:,keep]-mu[keep])/sd[keep],int(keep.sum())
    Br,Bt,bk=std_pair(Xb,Xtb); Rr,Rt,rk=std_pair(Xraw,Xtestraw); Hr=np.concatenate([Br,Rr],1); Ht=np.concatenate([Bt,Rt],1)
    ids=np.asarray([int(r['id']) if str(r['id']).isdigit() else 10000+i for i,r in enumerate(train)]); valid=(ids<=100)&(ids%4==0); alphas=np.logspace(-3,4,24); results={}
    for name,A,B in [('raw_structured',Rr,Rt),('fly_reservoir',Br,Bt),('fly_hybrid',Hr,Ht)]:
        vacc,alpha=choose_alpha(A,Y,classes,valid,alphas); M=ridge_fit(A,Y,alpha); scores=B@M; pred=np.argmax(scores,1); acc=float(np.mean(pred==ytest)); rows=[]
        for i,r in enumerate(test):rows.append({'id':r['id'],'expected':classes[ytest[i]],'predicted':classes[pred[i]],'correct':bool(pred[i]==ytest[i]),'margin':float(np.sort(scores[i])[-1]-np.sort(scores[i])[-2])})
        results[name]={'validation_accuracy':vacc,'alpha':alpha,'test_accuracy':acc,'predictions':rows,'feature_count':int(A.shape[1])}; print(name.upper()+'='+json.dumps(results[name],ensure_ascii=False),flush=True)
    result={'source_commit':os.environ.get('UPSTREAM_COMMIT'),'graph_neurons':n,'unique_edges':int(W.nnz),'stimulated_real_neurons':stim.tolist(),'read_nodes':int(len(read)),'train_examples':len(train),'heldout_actual_kice_items':len(test),'brain_train_wall_s':trwall,'brain_test_wall_s':tewall,'mean_active_train':float(np.mean(atr)),'mean_spikes_train':float(np.mean(sps)),'mean_active_test':float(np.mean(ate)),'mean_spikes_test':float(np.mean(spte)),'brain_nonconstant_features':bk,'raw_nonconstant_features':rk,'results':results,'scope_note':'Held-out targets are the seven 2027 June KICE high-difficulty items 14,15,21,22,28,29,30. This benchmark tests first-strategy selection from structured problem features, not autonomous full symbolic solving.'}
    Path('suneung_strategy_v4_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print('STRATEGY_V4_FINAL='+json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
