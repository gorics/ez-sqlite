from __future__ import annotations
import json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

UP = Path(os.environ.get('DROSOPHILA_MODEL_DIR', 'Drosophila_brain_model'))
COMP = UP / 'Completeness_783.csv'
CONN = UP / 'Connectivity_783.parquet'
V0=-52.0; VRST=-52.0; VTH=-45.0; TMBR=20.0; TAU=5.0; TRFC=2.2; TDLY=1.8; WSYN=0.275; FPOI=250.0; DT=0.1
DELAY=round(TDLY/DT); RFC=round(TRFC/DT); POIW=WSYN*FPOI

def ragged(indptr, rows):
    starts=indptr[rows]; lens=indptr[rows+1]-starts; total=int(lens.sum())
    if total==0: return np.empty(0,dtype=np.int64)
    cum=np.cumsum(lens); offsets=np.repeat(starts-(cum-lens), lens)
    return np.arange(total,dtype=np.int64)+offsets

class DenseEngine:
    def __init__(self,W,seed=27):
        W=W.tocsr(); W.sort_indices(); self.n=W.shape[0]
        self.indptr=W.indptr.astype(np.int64,copy=False); self.indices=W.indices.astype(np.int32,copy=False)
        self.wdata=(W.data.astype(np.float64)*WSYN).astype(np.float32); self.rng=np.random.default_rng(seed)
        a=1/TMBR; b=1/TAU
        self.ev=np.float32(np.exp(-a*DT)); self.eg=np.float32(np.exp(-b*DT)); self.kg=np.float32(a*(np.exp(-b*DT)-np.exp(-a*DT))/(a-b)); self.v0rest=np.float32(V0*(1-float(self.ev)))
        self.v=np.full(self.n,V0,np.float32); self.g=np.zeros(self.n,np.float32); self.tmp=np.zeros(self.n,np.float32)
        self.rfc=np.zeros(self.n,np.int16); self.rfc_len=np.full(self.n,RFC,np.int16); self.ring=np.zeros((DELAY+1,self.n),np.float32); self.slot=0
        self.count=np.zeros(self.n,np.int32); self.poi=np.empty(0,np.int64); self.pp=np.empty(0,np.float64)
    def set_poisson(self,idx,rate=150.0):
        self.poi=np.asarray(idx,np.int64); self.pp=np.full(len(self.poi),rate*DT*1e-3); self.rfc_len[self.poi]=0
    def step(self):
        p=self.ring[self.slot]; self.g+=p; p.fill(0)
        ridx=np.flatnonzero(self.rfc)
        if ridx.size: vh=self.v[ridx].copy(); gh=self.g[ridx].copy()
        np.multiply(self.g,self.kg,out=self.tmp); self.v*=self.ev; self.v+=self.v0rest; self.v+=self.tmp; self.g*=self.eg
        if ridx.size: self.v[ridx]=vh; self.g[ridx]=gh; self.rfc[ridx]-=1
        fired=self.rng.random(self.poi.size)<self.pp
        if fired.any(): self.v[self.poi[fired]]+=np.float32(POIW)
        spk=np.flatnonzero(self.v>VTH)
        if spk.size:
            self.v[spk]=VRST; self.g[spk]=0; self.rfc[spk]=self.rfc_len[spk]; self.count[spk]+=1
            flat=ragged(self.indptr,spk)
            if flat.size: np.add.at(self.ring[(self.slot+DELAY)%(DELAY+1)],self.indices[flat],self.wdata[flat])
        self.slot=(self.slot+1)%(DELAY+1)

class ActiveEngine:
    def __init__(self,W,seed=27):
        W=W.tocsr(); W.sort_indices(); self.n=W.shape[0]
        self.indptr=W.indptr.astype(np.int64,copy=False); self.indices=W.indices.astype(np.int32,copy=False)
        self.wdata=(W.data.astype(np.float64)*WSYN).astype(np.float32); self.rng=np.random.default_rng(seed)
        a=1/TMBR; b=1/TAU
        self.ev=np.float32(np.exp(-a*DT)); self.eg=np.float32(np.exp(-b*DT)); self.kg=np.float32(a*(np.exp(-b*DT)-np.exp(-a*DT))/(a-b)); self.v0rest=np.float32(V0*(1-float(self.ev)))
        self.v=np.full(self.n,V0,np.float32); self.g=np.zeros(self.n,np.float32)
        self.rfc=np.zeros(self.n,np.int16); self.rfc_len=np.full(self.n,RFC,np.int16); self.count=np.zeros(self.n,np.int32)
        self.poi=np.empty(0,np.int64); self.pp=np.empty(0,np.float64)
        self.active=np.empty(self.n,np.int32); self.active_mask=np.zeros(self.n,bool); self.active_count=0
        self.ev_targets=[None]*(DELAY+1); self.ev_weights=[None]*(DELAY+1); self.slot=0
    def _activate(self,idx):
        if idx.size==0: return
        cand=idx[~self.active_mask[idx]]
        if cand.size==0: return
        new=np.unique(cand).astype(np.int32,copy=False); k=len(new)
        self.active[self.active_count:self.active_count+k]=new; self.active_count+=k; self.active_mask[new]=True
    def set_poisson(self,idx,rate=150.0):
        self.poi=np.asarray(idx,np.int64); self.pp=np.full(len(self.poi),rate*DT*1e-3); self.rfc_len[self.poi]=0; self._activate(self.poi.astype(np.int32))
    def step(self):
        targets=self.ev_targets[self.slot]; weights=self.ev_weights[self.slot]
        if targets is not None:
            np.add.at(self.g,targets,weights); self._activate(targets); self.ev_targets[self.slot]=None; self.ev_weights[self.slot]=None
        act=self.active[:self.active_count]
        if act.size:
            rv=self.rfc[act]; free=act[rv==0]; refr=act[rv>0]
            if free.size:
                gv=self.g[free].copy(); self.v[free]=self.v[free]*self.ev+self.v0rest+gv*self.kg; self.g[free]=gv*self.eg
            if refr.size: self.rfc[refr]-=1
        fired=self.rng.random(self.poi.size)<self.pp
        if fired.any(): self.v[self.poi[fired]]+=np.float32(POIW)
        act=self.active[:self.active_count]; spk=act[self.v[act]>VTH]
        if spk.size:
            self.v[spk]=VRST; self.g[spk]=0; self.rfc[spk]=self.rfc_len[spk]; self.count[spk]+=1
            flat=ragged(self.indptr,spk)
            if flat.size:
                ts=(self.slot+DELAY)%(DELAY+1); self.ev_targets[ts]=self.indices[flat].copy(); self.ev_weights[ts]=self.wdata[flat].copy()
        self.slot=(self.slot+1)%(DELAY+1)

def run_steps(engine,ms):
    for _ in range(round(ms/DT)): engine.step()

def main():
    print('SOURCE_FILES',COMP,COMP.stat().st_size,CONN,CONN.stat().st_size,flush=True)
    t=time.perf_counter(); comp=pd.read_csv(COMP,index_col=0); con=pd.read_parquet(CONN,columns=['Presynaptic_Index','Postsynaptic_Index','Excitatory x Connectivity'])
    n=len(comp); pre=con['Presynaptic_Index'].to_numpy(np.int32,copy=False); post=con['Postsynaptic_Index'].to_numpy(np.int32,copy=False); wt=con['Excitatory x Connectivity'].to_numpy(np.int32,copy=False)
    max_index=int(max(pre.max(initial=0),post.max(initial=0))); W=sp.csr_matrix((wt,(pre,post)),shape=(n,n),dtype=np.int32); W.sum_duplicates(); W.sort_indices(); load=time.perf_counter()-t
    if max_index>=n or n<130000: raise RuntimeError(f'bad v783 dimensions n={n} max_index={max_index}')
    avail=np.flatnonzero(np.diff(W.indptr)>0); check_stim=avail[:64]; stim=avail[:16]
    print(f'REAL_V783_LOADED neurons={n} parquet_rows={len(con)} unique_edges={W.nnz} max_index={max_index} abs_weight_sum={int(np.abs(W.data.astype(np.int64)).sum())} load_s={load:.3f}',flush=True)

    # exactness check against the dense implementation on the same full graph
    d=DenseEngine(W,27); a=ActiveEngine(W,27); d.set_poisson(check_stim); a.set_poisson(check_stim)
    run_steps(d,50.0); run_steps(a,50.0)
    exact=bool(np.array_equal(d.count,a.count))
    print(f'ACTIVE_EXACTNESS_50MS={exact} dense_spikes={int(d.count.sum())} active_spikes={int(a.count.sum())} active_neurons={a.active_count}',flush=True)
    if not exact: raise RuntimeError('active event engine diverged from dense reference')

    e=ActiveEngine(W,27); e.set_poisson(stim)
    bench_ms=500.0; t=time.perf_counter(); run_steps(e,bench_ms); bw=time.perf_counter()-t; bx=(bench_ms/1000)/bw
    print(f'ACTIVE_RAW_BENCH stim={len(stim)} sim_s={bench_ms/1000:.3f} wall_s={bw:.6f} x_realtime={bx:.6f} spikes={int(e.count.sum())} active_neurons={e.active_count}',flush=True)

    e=ActiveEngine(W,27); e.set_poisson(stim)
    target_s=2.0; steps=round(target_s*1000/DT); start=time.perf_counter(); maxlag=0.0
    for k in range(steps):
        e.step(); target=(k+1)*DT/1000; elapsed=time.perf_counter()-start
        if elapsed<target: time.sleep(target-elapsed); elapsed=time.perf_counter()-start
        maxlag=max(maxlag,elapsed-target)
        if (k+1)%1000==0: print(f'LIVE sim={target:.3f}s wall={elapsed:.3f}s lag_ms={(elapsed-target)*1000:.3f} spikes={int(e.count.sum())} active_neurons={e.active_count}',flush=True)
    wall=time.perf_counter()-start; stimsp=int(e.count[stim].sum()); total=int(e.count.sum())
    result={'source_commit':os.environ.get('UPSTREAM_COMMIT'),'source_model_neurons':n,'parquet_rows':int(len(con)),'unique_edges':int(W.nnz),'max_connectivity_index':max_index,'abs_signed_synapse_count_sum':int(np.abs(W.data.astype(np.int64)).sum()),'connectivity_file_bytes':CONN.stat().st_size,'completeness_file_bytes':COMP.stat().st_size,'dt_ms':DT,'engine':'exact active-event LIF over full CSR graph','dense_exactness_50ms':exact,'stimulated_real_neurons':int(len(stim)),'benchmark_sim_s':bench_ms/1000,'benchmark_wall_s':bw,'raw_x_realtime':bx,'paced_sim_s':target_s,'paced_wall_s':wall,'paced_x_realtime':target_s/wall,'max_lag_ms':maxlag*1000,'all_spikes':total,'stimulated_spikes':stimsp,'downstream_spikes':total-stimsp,'active_neurons_touched':int(e.active_count),'load_s':load}
    Path('fly_v783_result.json').write_text(json.dumps(result,indent=2)); print('FULL_BRAIN_RESULT='+json.dumps(result,sort_keys=True),flush=True)
if __name__=='__main__': main()
