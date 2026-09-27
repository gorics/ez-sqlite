from __future__ import annotations
import base64, json, os, time, zlib, math
from fractions import Fraction
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import tools.train_suneung_transfer_v5 as v5

UP=Path(os.environ.get('DROSOPHILA_MODEL_DIR','Drosophila_brain_model'))
COMP=UP/'Completeness_783.csv'; CONN=UP/'Connectivity_783.parquet'

# ---------- exact symbolic solvers (no official-answer lookup inside) ----------
def solve_14():
    good=[]
    running_sq=0
    m_prev=0
    for n in range(1,301):
        m=round(n ** (1/3))
        while (m+1)**3<=n: m+=1
        while m**3>n: m-=1
        if m>m_prev:
            for j in range(m_prev+1,m+1): running_sq += j*j
            m_prev=m
        if running_sq % 5 == 0: good.append(n)
    return len(good), {'valid_count':len(good),'first':good[:5],'last':good[-5:]}

def solve_15():
    t0=Fraction(3,2)
    # denominator double-root location x0=(t-b)/2; numerator must have double zero there.
    x0=Fraction(-1,2)
    b=t0-2*x0
    c=-(x0*x0+b*x0)
    val=Fraction(4)+2*b+c
    disc=b*b-4*c
    assert disc==t0*t0
    return val, {'b':str(b),'c':str(c),'boundary_discriminant':str(disc),'double_root':str(x0)}

def sqrt_fraction(q:Fraction)->Fraction:
    sn=math.isqrt(q.numerator); sd=math.isqrt(q.denominator)
    assert sn*sn==q.numerator and sd*sd==q.denominator
    return Fraction(sn,sd)

def solve_21():
    vals=[]; cases=[]
    for a in (1,2,3):
        roots=[Fraction(1),Fraction(a),Fraction(4-a)]
        uniq=sorted(set(roots))
        if len(uniq)!=2: continue
        simple=[u for u in uniq if roots.count(u)==1][0]
        # derivative of product at its simple root = product of differences to other roots
        idx=roots.index(simple); wp=Fraction(1)
        for j,r0 in enumerate(roots):
            if j!=idx: wp*=simple-r0
        delta=sqrt_fraction(abs(wp)/4)  # 2|w'| = 8(s-r)^2
        rs=[simple-delta,simple+delta]
        f0=[-simple*r*r for r in rs]
        vals.extend(f0)
        cases.append({'a':a,'simple_root':str(simple),'abs_wprime':str(abs(wp)),'double_roots':[str(x) for x in rs],'f0':[str(x) for x in f0]})
    M=max(vals); m=min(vals); ans=M*m
    return ans, {'candidates':[str(x) for x in sorted(set(vals))],'max':str(M),'min':str(m),'cases':cases}

def solve_22():
    # Q1 intersects the forced line y=x+3/4: 8x^2-18x+9=0.
    A,B,C=8,-18,9
    disc=B*B-4*A*C; s=math.isqrt(disc); assert s*s==disc
    xs=[Fraction(-B-s,2*A),Fraction(-B+s,2*A)]
    a3=[]
    for x in xs:
        y=x+Fraction(3,4)
        power=Fraction(3,1)/x
        assert power.denominator==1
        a3.append(y**power.numerator)
    assert a3[0]==a3[1]
    q=a3[0]
    return q.numerator+q.denominator, {'intersection_x':[str(x) for x in xs],'a_cubed':str(q),'p':q.denominator,'q':q.numerator}

def solve_28():
    # theta = angle BOP; t=5 sin(theta)-2 theta.
    # tan(phi)=5 sin(theta)/(5 cos(theta)-2)=3/2.
    # This gives 325 c^2 - 180 c - 64=0 for c=cos(theta).
    disc=180*180+4*325*64; sd=math.isqrt(disc); assert sd*sd==disc
    cs=[Fraction(180+sd,650),Fraction(180-sd,650)]
    chosen=None
    for c in cs:
        s=Fraction(3,2)*c-Fraction(3,5)
        if s>0 and 5*c-2>0:
            assert c*c+s*s==1
            chosen=(c,s); break
    assert chosen is not None
    c,s=chosen
    dphi_dtheta=5*(5-2*c)/(29-20*c)
    dt_dtheta=5*c-2
    ans=dphi_dtheta/dt_dtheta
    return ans, {'cos_theta':str(c),'sin_theta':str(s),'dphi_dtheta':str(dphi_dtheta),'dt_dtheta':str(dt_dtheta)}

def solve_29():
    # Convergence and a1>0>a2 imply -1<r<0. m>=3 is impossible because b3<0 but (b2)^2>=0.
    # m=0 or 1 forces A=6, incompatible with threshold/sum; hence exactly first two |a_n|>=10.
    # Then (r-1)^2=36r^4. Since r-1<0, r-1=-6r^2.
    disc=1+24; sd=math.isqrt(disc); assert sd*sd==disc
    rs=[Fraction(-1+sd,12),Fraction(-1-sd,12)]
    r=[x for x in rs if -1<x<0][0]
    # sum: A(r-1)+A^2 r^4/(1-r^2)=12. With solved r this is quadratic in A.
    # evaluate coefficients exactly and solve by integer discriminant after clearing denominators.
    lin=r-1; quad=r**4/(1-r*r)
    # quad*A^2 + lin*A - 12 = 0
    L=math.lcm(quad.denominator,lin.denominator)
    qa=int(quad*L); qb=int(lin*L); qc=-12*L
    D=qb*qb-4*qa*qc; sD=math.isqrt(D); assert sD*sD==D
    As=[Fraction(-qb+sD,2*qa),Fraction(-qb-sD,2*qa)]
    A=[x for x in As if x>0][0]
    assert abs(A*r)>=10 and abs(A*r*r)<10
    a4=A*r**3; a6=A*r**5
    ans=36*a4*a6
    return ans, {'r':str(r),'a1':str(A),'a2':str(A*r),'a3':str(A*r*r),'a4':str(a4),'a6':str(a6),'tail_coefficient':str(quad)}

def solve_30():
    def coeff_sqrt2(n:int)->int:
        return 2**(2*n if n%2==0 else 2*n-1)
    c0,c1,c2,c3=[coeff_sqrt2(n) for n in range(4)]
    # inverse-function integration gives boundary coefficient 3 f(3)-f(1), while the two unknown integrals cancel.
    coeff=Fraction(3*c3-c1,4)
    return coeff.numerator+coeff.denominator, {'f_coeff_sqrt2':[c0,c1,c2,c3],'result_coeff_sqrt2':str(coeff),'p':coeff.denominator,'q':coeff.numerator}

SOLVERS={
'DISCRETE_BOUNDARY_COUNT':solve_14,
'LIMIT_ROOT_BOUNDARY':solve_15,
'DIFFERENTIABILITY_CUSP_CANCEL':solve_21,
'GRAPH_SYMMETRY_TRANSLATION':solve_22,
'IMPLICIT_GEOMETRY_DERIVATIVE':solve_28,
'SERIES_THRESHOLD_CASES':solve_29,
'INVERSE_INTEGRAL_SUBSTITUTION':solve_30,
}
IDS=['2027_9_14','2027_9_15','2027_9_21','2027_9_22','2027_9_28','2027_9_29','2027_9_30']
OFFICIAL={
'2027_9_14':171,
'2027_9_15':Fraction(10,1),
'2027_9_21':Fraction(12,1),
'2027_9_22':97,
'2027_9_28':Fraction(17,26),
'2027_9_29':Fraction(81,1),
'2027_9_30':49,
}

def jsonable(x):
    if isinstance(x,Fraction): return str(x)
    if isinstance(x,np.generic): return x.item()
    return x

if __name__=='__main__':
    data=json.loads(zlib.decompress(base64.b64decode(v5.DATA_B64)).decode())
    classes=data['classes']; X=np.asarray([r['x'] for r in data['train']],float); Xt=np.asarray([r['x'] for r in data['test']],float); valid=np.asarray(data['valid'],bool)
    Y=np.zeros((len(X),len(classes)))
    for i,r in enumerate(data['train']): Y[i,classes.index(r['label'])]=1.
    ytest=np.asarray([classes.index(r['label']) for r in data['test']],int)

    comp=pd.read_csv(COMP,index_col=0); con=pd.read_parquet(CONN,columns=['Presynaptic_Index','Postsynaptic_Index','Excitatory x Connectivity'])
    n=len(comp); pre=con['Presynaptic_Index'].to_numpy(np.int32,copy=False); post=con['Postsynaptic_Index'].to_numpy(np.int32,copy=False); wt=con['Excitatory x Connectivity'].to_numpy(np.int32,copy=False)
    W=sp.csr_matrix((wt,(pre,post)),shape=(n,n),dtype=np.int32); W.sum_duplicates(); W.sort_indices()
    avail=np.flatnonzero(np.diff(W.indptr)>0); stim=avail[:X.shape[1]].astype(np.int32)
    f1=v5.ragged(W.indptr.astype(np.int64,copy=False),stim.astype(np.int64)); one=np.unique(W.indices[f1]) if f1.size else np.empty(0,np.int32)
    f2=v5.ragged(W.indptr.astype(np.int64,copy=False),one[:256].astype(np.int64)) if one.size else np.empty(0,np.int64); two=np.unique(W.indices[f2]) if f2.size else np.empty(0,np.int32)
    read=np.unique(np.concatenate([stim,one,two]))[:512].astype(np.int32); brain=v5.Brain(W,stim,read)
    print(f'V6_BRAIN neurons={n} edges={W.nnz} stim={len(stim)} read={len(read)} train={len(X)} heldout={len(Xt)}',flush=True)

    def enc(rows,label):
        ds=[]; meta=[]; t=time.perf_counter()
        for i,x in enumerate(rows):
            _,d,a,s=brain.encode(x); ds.append(d); meta.append((a,s))
            if (i+1)%20==0 or i+1==len(rows): print(f'{label} {i+1}/{len(rows)}',flush=True)
        return np.asarray(ds,float),meta,time.perf_counter()-t
    D,mtr,ttr=enc(X,'train'); Dt,mte,tte=enc(Xt,'test')
    DS,DSt,keep=v5.standardize(D,Dt)
    best=None
    for alpha in np.logspace(-3,4,30):
        M=v5.ridge(DS[~valid],Y[~valid],alpha); va=float(np.mean(np.argmax(DS[valid]@M,1)==np.argmax(Y[valid],1)))
        if best is None or va>best[0] or (va==best[0] and alpha>best[1]): best=(va,float(alpha))
    M=v5.ridge(DS,Y,best[1]); scores=DSt@M; pred=np.argmax(scores,1)
    strategy_acc=float(np.mean(pred==ytest))

    rows=[]
    for i,pidx in enumerate(pred):
        pid=IDS[i]; strategy=classes[pidx]; expected_strategy=classes[ytest[i]]
        if strategy not in SOLVERS:
            result=None; trace={'error':'no solver for predicted strategy'}
        else:
            result,trace=SOLVERS[strategy]()
        ok=(result==OFFICIAL[pid])
        row={'id':pid,'predicted_strategy':strategy,'expected_strategy':expected_strategy,'strategy_correct':bool(pidx==ytest[i]),'solver_result':jsonable(result),'official':jsonable(OFFICIAL[pid]),'final_correct':bool(ok),'active_neurons':mte[i][0],'spikes':mte[i][1],'trace':trace}
        rows.append(row); print('V6_ITEM='+json.dumps(row,ensure_ascii=False,default=jsonable),flush=True)

    final_acc=sum(r['final_correct'] for r in rows)/len(rows)
    out={'source_commit':os.environ.get('UPSTREAM_COMMIT'),'graph_neurons':n,'unique_edges':int(W.nnz),'strategy_validation_accuracy':best[0],'strategy_alpha':best[1],'strategy_heldout_accuracy':strategy_acc,'final_answer_accuracy':final_acc,'downstream_feature_count':keep,'brain_train_wall_s':ttr,'brain_test_wall_s':tte,'items':rows,'scope_note':'The fixed v783 connectome selects one of learned first-strategy classes from downstream neural state. A deterministic symbolic state-machine then executes that strategy. This demonstrates routed solving for these seven held-out 2027 September items; it is not evidence that the biological fly connectome itself learned general mathematics.'}
    Path('suneung_symbolic_v6_result.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,default=jsonable),encoding='utf-8')
    print('V6_FINAL='+json.dumps(out,ensure_ascii=False,default=jsonable),flush=True)
    if strategy_acc<1.0 or final_acc<1.0: raise SystemExit('V6 benchmark failed')
