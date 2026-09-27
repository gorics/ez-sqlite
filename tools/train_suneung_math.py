from __future__ import annotations
import json, os, time
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

from run_fly_v783_fullbrain import ActiveEngine, V0, RFC, DT, POIW

UP = Path(os.environ.get('DROSOPHILA_MODEL_DIR', 'Drosophila_brain_model'))
COMP = UP / 'Completeness_783.csv'
CONN = UP / 'Connectivity_783.parquet'
RNG = np.random.default_rng(20270927)


def load_graph():
    comp = pd.read_csv(COMP, index_col=0)
    con = pd.read_parquet(CONN, columns=['Presynaptic_Index','Postsynaptic_Index','Excitatory x Connectivity'])
    n = len(comp)
    pre = con['Presynaptic_Index'].to_numpy(np.int32, copy=False)
    post = con['Postsynaptic_Index'].to_numpy(np.int32, copy=False)
    wt = con['Excitatory x Connectivity'].to_numpy(np.int32, copy=False)
    W = sp.csr_matrix((wt, (pre, post)), shape=(n, n), dtype=np.int32)
    W.sum_duplicates(); W.sort_indices()
    return W, len(con), n


def reset_engine(e: ActiveEngine):
    if e.active_count:
        old = e.active[:e.active_count].copy()
        e.v[old] = V0
        e.g[old] = 0
        e.rfc[old] = 0
        e.count[old] = 0
        e.rfc_len[old] = RFC
        e.active_mask[old] = False
    e.active_count = 0
    e.ev_targets = [None] * len(e.ev_targets)
    e.ev_weights = [None] * len(e.ev_weights)
    e.slot = 0
    e.poi = np.empty(0, np.int64)
    e.pp = np.empty(0, np.float64)


def build_schedule(vals, total_ms=50.0):
    steps = round(total_ms / DT)
    sched = np.zeros((steps, 4), dtype=np.bool_)
    for ch, raw in enumerate(vals[:4]):
        val = int(raw)
        mag = min(abs(val), 8)
        if mag == 0:
            continue
        # Sign is temporal: positive = early phase, negative = late phase.
        start = 15 if val > 0 else 145
        spacing = 15  # 1.5 ms; input neurons have refractory disabled.
        for j in range(mag):
            s = start + j * spacing
            if s < steps:
                sched[s, ch] = True
    return sched


def run_encoded(e: ActiveEngine, stim, vals, feature_nodes, total_ms=50.0):
    reset_engine(e)
    e.set_poisson(stim, rate=0.0)
    e.rfc_len[stim] = 0
    sched = build_schedule(vals, total_ms)
    snap_steps = {round(30.0/DT), round(40.0/DT), round(50.0/DT)}
    parts = []
    for k in range(sched.shape[0]):
        chans = np.flatnonzero(sched[k])
        if chans.size:
            e.v[stim[chans]] += np.float32(POIW)
        e.step()
        if (k + 1) in snap_steps:
            nodes = feature_nodes
            parts.extend([
                e.count[nodes].astype(np.float32),
                (e.v[nodes] - V0).astype(np.float32),
                e.g[nodes].astype(np.float32),
            ])
    return np.concatenate(parts)


def discover_feature_nodes(e, stim):
    patterns = [
        [8,8,8,8], [-8,-8,-8,-8], [8,-8,8,-8], [-8,8,-8,8],
        [8,0,-8,0], [0,8,0,-8], [4,-4,-4,4], [-4,4,4,-4],
    ]
    union = set()
    for vals in patterns:
        reset_engine(e); e.set_poisson(stim, rate=0.0); e.rfc_len[stim] = 0
        sched = build_schedule(vals)
        for k in range(sched.shape[0]):
            chans = np.flatnonzero(sched[k])
            if chans.size: e.v[stim[chans]] += np.float32(POIW)
            e.step()
        union.update(map(int, e.active[:e.active_count]))
    nodes = np.array(sorted(union), dtype=np.int32)
    if len(nodes) < 32:
        raise RuntimeError(f'reservoir too small: {len(nodes)} active nodes')
    return nodes


def task_sets(name):
    if name == 'sequence':
        rows = [(a1,d,n,0, a1+(n-1)*d) for a1,d,n in product(range(-5,6), range(-3,4), range(1,8))]
    elif name == 'derivative':
        rows = [(a,b,c,x, 2*a*x+b) for a,b,c,x in product(range(-3,4), range(-4,5), range(-2,3), range(-3,4))]
    elif name == 'integral':
        # f(t)=3*a*t^2 + 2*b*t + c, answer = integral_0^x f(t)dt
        rows = [(a,b,c,x, a*x**3+b*x**2+c*x) for a,b,c,x in product(range(-2,3), range(-2,3), range(-2,3), range(-3,4))]
    else:
        raise ValueError(name)
    idx = RNG.permutation(len(rows))
    rows = [rows[i] for i in idx]
    ntrain, ntest = 160, 80
    return rows[:ntrain], rows[ntrain:ntrain+ntest]


def standardize_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    keep = sd > 1e-6
    return mu, sd, keep


def standardize_apply(X, mu, sd, keep):
    return (X[:,keep] - mu[keep]) / sd[keep]


def ridge_fit_dual(X, y, alpha):
    Xb = np.concatenate([np.ones((len(X),1), np.float32), X.astype(np.float32)], axis=1)
    K = Xb @ Xb.T
    A = K + np.eye(len(X), dtype=np.float32) * np.float32(alpha)
    coef = Xb.T @ np.linalg.solve(A, y.astype(np.float32))
    return coef


def ridge_predict(X, coef):
    Xb = np.concatenate([np.ones((len(X),1), np.float32), X.astype(np.float32)], axis=1)
    return Xb @ coef


def fit_select_alpha(X, y):
    n = len(X)
    cut = int(n * 0.8)
    tr = np.arange(cut); va = np.arange(cut,n)
    best = None
    for alpha in (0.03,0.1,0.3,1.0,3.0,10.0,30.0,100.0):
        c = ridge_fit_dual(X[tr], y[tr], alpha)
        p = ridge_predict(X[va], c)
        mae = float(np.mean(np.abs(p-y[va])))
        if best is None or mae < best[0]: best = (mae, alpha)
    return float(best[1])


def metrics(y, p):
    rounded = np.rint(p)
    exact = float(np.mean(rounded == y))
    mae = float(np.mean(np.abs(p-y)))
    rmse = float(np.sqrt(np.mean((p-y)**2)))
    denom = float(np.sum((y-y.mean())**2))
    r2 = 1.0 - float(np.sum((p-y)**2))/denom if denom else 1.0
    return {'exact_rounded_accuracy':exact,'mae':mae,'rmse':rmse,'r2':r2}


def train_task(name, e, stim, nodes):
    train, test = task_sets(name)
    def featurize(rows):
        X=[]; Y=[]; raw=[]
        t0=time.perf_counter()
        for i,row in enumerate(rows):
            vals=row[:4]; y=row[4]
            X.append(run_encoded(e, stim, vals, nodes))
            raw.append(np.array(vals,dtype=np.float32))
            Y.append(float(y))
            if (i+1)%40==0: print(f'{name} encoded {i+1}/{len(rows)}', flush=True)
        return np.stack(X), np.asarray(Y,np.float32), np.stack(raw), time.perf_counter()-t0
    Xtr,ytr,Rtr,ttr = featurize(train)
    Xte,yte,Rte,tte = featurize(test)

    mu,sd,keep = standardize_fit(Xtr)
    Ztr=standardize_apply(Xtr,mu,sd,keep); Zte=standardize_apply(Xte,mu,sd,keep)
    ymu=float(ytr.mean()); ysd=float(ytr.std() or 1.0); yn=(ytr-ymu)/ysd
    alpha=fit_select_alpha(Ztr,yn)
    coef=ridge_fit_dual(Ztr,yn,alpha)
    pred=(ridge_predict(Zte,coef)*ysd+ymu).astype(np.float32)

    # Raw-input linear baseline: same train/test split and ridge selection.
    rmu,rsd,rkeep=standardize_fit(Rtr)
    Brtr=standardize_apply(Rtr,rmu,rsd,rkeep); Brte=standardize_apply(Rte,rmu,rsd,rkeep)
    balpha=fit_select_alpha(Brtr,yn)
    bcoef=ridge_fit_dual(Brtr,yn,balpha)
    bpred=(ridge_predict(Brte,bcoef)*ysd+ymu).astype(np.float32)

    examples=[]
    for row,pp in list(zip(test,pred))[:12]:
        examples.append({'input':list(map(int,row[:4])),'target':int(row[4]),'prediction':float(pp),'rounded':int(np.rint(pp))})
    out={
        'task':name,'train_n':len(train),'test_n':len(test),'reservoir_features_total':int(Xtr.shape[1]),
        'reservoir_features_nonconstant':int(keep.sum()),'alpha':alpha,'encode_train_wall_s':ttr,'encode_test_wall_s':tte,
        'reservoir':metrics(yte,pred),'raw_linear_baseline':metrics(yte,bpred),'examples':examples,
    }
    print('TASK_RESULT='+json.dumps(out,sort_keys=True),flush=True)
    return out


def main():
    t0=time.perf_counter(); W, parquet_rows, n = load_graph(); load_s=time.perf_counter()-t0
    deg=np.diff(W.indptr); avail=np.flatnonzero(deg>0)
    # Use the same real four-neuron input budget that previously sustained real-time execution.
    stim=avail[:4].astype(np.int64)
    e=ActiveEngine(W,seed=2027)
    nodes=discover_feature_nodes(e,stim)
    print(f'SUNEUNG_RESERVOIR graph_neurons={n} edges={W.nnz} parquet_rows={parquet_rows} stim={stim.tolist()} feature_nodes={len(nodes)} load_s={load_s:.3f}',flush=True)
    results=[]
    for name in ('sequence','derivative','integral'):
        results.append(train_task(name,e,stim,nodes))
    mean_acc=float(np.mean([r['reservoir']['exact_rounded_accuracy'] for r in results]))
    mean_base=float(np.mean([r['raw_linear_baseline']['exact_rounded_accuracy'] for r in results]))
    summary={
        'source_commit':os.environ.get('UPSTREAM_COMMIT'),'graph_neurons':n,'unique_edges':int(W.nnz),'parquet_rows':int(parquet_rows),
        'stimulated_real_neurons':stim.tolist(),'feature_nodes':int(len(nodes)),'tasks':results,
        'mean_exact_rounded_accuracy':mean_acc,'mean_raw_linear_baseline_accuracy':mean_base,
        'learned_above_raw_linear_baseline':bool(mean_acc>mean_base),'total_wall_s':time.perf_counter()-t0,
        'scope_note':'Numeric CSAT-style generated tasks only; this does not establish natural-language understanding or general CSAT reasoning.'
    }
    Path('suneung_math_training_result.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('SUNEUNG_TRAINING_RESULT='+json.dumps(summary,ensure_ascii=False,sort_keys=True),flush=True)

if __name__=='__main__': main()
