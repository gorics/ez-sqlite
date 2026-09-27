from __future__ import annotations
import base64, json, math, os, time, zlib, random
from fractions import Fraction
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import tools.train_suneung_transfer_v5 as v5

UP=Path(os.environ.get('DROSOPHILA_MODEL_DIR','Drosophila_brain_model'))
COMP=UP/'Completeness_783.csv'; CONN=UP/'Connectivity_783.parquet'


def F(x):
    if isinstance(x, Fraction): return x
    if isinstance(x, int): return Fraction(x,1)
    if isinstance(x, str) and '/' in x:
        a,b=x.split('/'); return Fraction(int(a),int(b))
    return Fraction(x)


def qroots(a,b,c):
    a,b,c=F(a),F(b),F(c)
    D=b*b-4*a*c
    sn=math.isqrt(D.numerator); sd=math.isqrt(D.denominator)
    if sn*sn!=D.numerator or sd*sd!=D.denominator: raise ValueError('non-rational quadratic root')
    s=Fraction(sn,sd)
    return [(-b-s)/(2*a),(-b+s)/(2*a)]


def sqrt_frac(x):
    x=F(x); sn=math.isqrt(x.numerator); sd=math.isqrt(x.denominator)
    if sn*sn!=x.numerator or sd*sd!=x.denominator: raise ValueError('not square fraction')
    return Fraction(sn,sd)


def floor_cuberoot(n):
    m=round(int(n)**(1/3))
    while (m+1)**3<=n: m+=1
    while m**3>n: m-=1
    return m


def deriv_from_linear_roots(roots,x):
    roots=[F(r) for r in roots]; x=F(x); out=Fraction(1)
    skipped=False
    for r in roots:
        if r==x and not skipped:
            skipped=True; continue
        out*=x-r
    return out


class VM:
    def __init__(self): self.operator_hits={}
    def hit(self,k): self.operator_hits[k]=self.operator_hits.get(k,0)+1
    def E(self,x,e):
        if isinstance(x,(int,float,bool,Fraction)) or x is None: return x
        if isinstance(x,str): return e[x[1:]] if x.startswith('$') else x
        if isinstance(x,list): return [self.E(y,e) for y in x]
        if not isinstance(x,dict): return x
        op=x['op']; self.hit(op)
        if op=='frac': return Fraction(int(self.E(x['a'],e)),int(self.E(x['b'],e)))
        if op=='list': return [self.E(y,e) for y in x.get('items',[])]
        if op=='range': return list(range(int(self.E(x['start'],e)),int(self.E(x['stop'],e))))
        if op=='add': return sum((F(self.E(y,e)) for y in x['args']),Fraction(0))
        if op=='sub': return F(self.E(x['a'],e))-F(self.E(x['b'],e))
        if op=='mul':
            z=Fraction(1)
            for y in x['args']: z*=F(self.E(y,e))
            return z
        if op=='div': return F(self.E(x['a'],e))/F(self.E(x['b'],e))
        if op=='pow': return F(self.E(x['a'],e))**int(self.E(x['b'],e))
        if op=='neg': return -F(self.E(x['a'],e))
        if op=='abs': return abs(F(self.E(x['a'],e)))
        if op=='mod': return int(self.E(x['a'],e))%int(self.E(x['b'],e))
        if op=='eq': return self.E(x['a'],e)==self.E(x['b'],e)
        if op=='lt': return self.E(x['a'],e)<self.E(x['b'],e)
        if op=='le': return self.E(x['a'],e)<=self.E(x['b'],e)
        if op=='gt': return self.E(x['a'],e)>self.E(x['b'],e)
        if op=='ge': return self.E(x['a'],e)>=self.E(x['b'],e)
        if op=='and': return all(bool(self.E(y,e)) for y in x['args'])
        if op=='or': return any(bool(self.E(y,e)) for y in x['args'])
        if op=='len': return len(self.E(x['a'],e))
        if op=='index': return self.E(x['a'],e)[int(self.E(x['i'],e))]
        if op=='numerator': return F(self.E(x['a'],e)).numerator
        if op=='denominator': return F(self.E(x['a'],e)).denominator
        if op=='unique_sorted': return sorted(set(self.E(x['a'],e)))
        if op=='count': return self.E(x['a'],e).count(self.E(x['value'],e))
        if op=='single_multiplicity':
            arr=self.E(x['a'],e); m=int(self.E(x['m'],e)); vals=sorted(set(arr)); out=[v for v in vals if arr.count(v)==m]
            if len(out)!=1: raise AssertionError(('single_multiplicity',arr,m,out))
            return out[0]
        if op=='sum_map':
            arr=self.E(x['iter'],e); var=x['var']; old=e.get(var,None); had=var in e; s=Fraction(0)
            for v in arr: e[var]=v; s+=F(self.E(x['expr'],e))
            if had:e[var]=old
            else:e.pop(var,None)
            return s
        if op=='qroots': return qroots(self.E(x['a'],e),self.E(x['b'],e),self.E(x['c'],e))
        if op=='sqrt_frac': return sqrt_frac(self.E(x['a'],e))
        if op=='floor_cuberoot': return floor_cuberoot(int(self.E(x['a'],e)))
        if op=='deriv_linear_roots': return deriv_from_linear_roots(self.E(x['roots'],e),self.E(x['x'],e))
        if op=='max': return max(self.E(x['a'],e))
        if op=='min': return min(self.E(x['a'],e))
        if op=='all_equal':
            arr=self.E(x['a'],e); return len(arr)>0 and all(v==arr[0] for v in arr)
        if op=='if': return self.E(x['then'],e) if self.E(x['cond'],e) else self.E(x['else'],e)
        raise KeyError(op)
    def exec_block(self,prog,e):
        for ins in prog:
            if 'set' in ins: e[ins['set']]=self.E(ins['expr'],e)
            elif 'append' in ins: e.setdefault(ins['append'],[]).append(self.E(ins['expr'],e))
            elif 'assert' in ins:
                if not self.E(ins['assert'],e): raise AssertionError(ins.get('label','assert'))
            elif 'if' in ins:
                branch=ins['then'] if self.E(ins['if'],e) else ins.get('else',[]); r=self.exec_block(branch,e)
                if r is not None:return r
            elif 'for' in ins:
                var=ins['for']; arr=self.E(ins['in'],e); old=e.get(var,None); had=var in e
                for v in arr:
                    e[var]=v; r=self.exec_block(ins['body'],e)
                    if r is not None:return r
                if had:e[var]=old
                else:e.pop(var,None)
            elif 'return' in ins: return self.E(ins['return'],e)
            else: raise KeyError(ins)
        return None
    def run(self,prog,inputs=None):
        e={} if inputs is None else dict(inputs)
        r=self.exec_block(prog,e)
        return r,e

Q=lambda a,b:{'op':'frac','a':a,'b':b}
V=lambda s:'$'+s
ADD=lambda *a:{'op':'add','args':list(a)}
SUB=lambda a,b:{'op':'sub','a':a,'b':b}
MUL=lambda *a:{'op':'mul','args':list(a)}
DIV=lambda a,b:{'op':'div','a':a,'b':b}
POW=lambda a,b:{'op':'pow','a':a,'b':b}
EQ=lambda a,b:{'op':'eq','a':a,'b':b}
GT=lambda a,b:{'op':'gt','a':a,'b':b}
LT=lambda a,b:{'op':'lt','a':a,'b':b}

PROGRAMS={
'DISCRETE_BOUNDARY_COUNT':[
 {'set':'good','expr':{'op':'list','items':[]}},
 {'for':'n','in':{'op':'range','start':1,'stop':301},'body':[
   {'set':'m','expr':{'op':'floor_cuberoot','a':V('n')}},
   {'set':'s','expr':{'op':'sum_map','iter':{'op':'range','start':1,'stop':ADD(V('m'),1)},'var':'j','expr':POW(V('j'),2)}},
   {'if':EQ({'op':'mod','a':V('s'),'b':5},0),'then':[{'append':'good','expr':V('n')}]}
 ]},
 {'return':{'op':'len','a':V('good')}}
],
'LIMIT_ROOT_BOUNDARY':[
 {'set':'t0','expr':Q(3,2)},{'set':'x0','expr':Q(-1,2)},
 {'set':'b','expr':SUB(V('t0'),MUL(2,V('x0')))},
 {'set':'c','expr':{'op':'neg','a':ADD(POW(V('x0'),2),MUL(V('b'),V('x0')))}},
 {'set':'disc','expr':SUB(POW(V('b'),2),MUL(4,V('c')))},
 {'assert':EQ(V('disc'),POW(V('t0'),2)),'label':'boundary discriminant'},
 {'return':ADD(4,MUL(2,V('b')),V('c'))}
],
'DIFFERENTIABILITY_CUSP_CANCEL':[
 {'set':'vals','expr':{'op':'list','items':[]}},
 {'for':'a','in':{'op':'list','items':[1,2,3]},'body':[
  {'set':'roots','expr':{'op':'list','items':[1,V('a'),SUB(4,V('a'))]}},
  {'set':'uniq','expr':{'op':'unique_sorted','a':V('roots')}},
  {'if':EQ({'op':'len','a':V('uniq')},2),'then':[
    {'set':'simple','expr':{'op':'single_multiplicity','a':V('roots'),'m':1}},
    {'set':'wp','expr':{'op':'deriv_linear_roots','roots':V('roots'),'x':V('simple')}},
    {'set':'delta','expr':{'op':'sqrt_frac','a':DIV({'op':'abs','a':V('wp')},4)}},
    {'set':'rs','expr':{'op':'list','items':[SUB(V('simple'),V('delta')),ADD(V('simple'),V('delta'))]}},
    {'for':'r','in':V('rs'),'body':[{'append':'vals','expr':MUL(-1,V('simple'),POW(V('r'),2))}]}
  ]}
 ]},
 {'return':MUL({'op':'max','a':V('vals')},{'op':'min','a':V('vals')})}
],
'GRAPH_SYMMETRY_TRANSLATION':[
 {'set':'xs','expr':{'op':'qroots','a':8,'b':-18,'c':9}}, {'set':'a3','expr':{'op':'list','items':[]}},
 {'for':'x','in':V('xs'),'body':[
   {'set':'y','expr':ADD(V('x'),Q(3,4))},{'set':'power','expr':DIV(3,V('x'))},
   {'assert':EQ({'op':'denominator','a':V('power')},1),'label':'integer power'},
   {'append':'a3','expr':POW(V('y'),{'op':'numerator','a':V('power')})}
 ]},
 {'assert':{'op':'all_equal','a':V('a3')},'label':'same a^3'},
 {'set':'q','expr':{'op':'index','a':V('a3'),'i':0}},
 {'return':ADD({'op':'numerator','a':V('q')},{'op':'denominator','a':V('q')})}
],
'IMPLICIT_GEOMETRY_DERIVATIVE':[
 {'set':'cs','expr':{'op':'qroots','a':325,'b':-180,'c':-64}}, {'set':'chosen','expr':{'op':'list','items':[]}},
 {'for':'c','in':V('cs'),'body':[
   {'set':'s','expr':SUB(MUL(Q(3,2),V('c')),Q(3,5))},
   {'if':{'op':'and','args':[GT(V('s'),0),GT(SUB(MUL(5,V('c')),2),0),EQ(ADD(POW(V('c'),2),POW(V('s'),2)),1)]},'then':[{'append':'chosen','expr':{'op':'list','items':[V('c'),V('s')]}}]}
 ]},
 {'assert':EQ({'op':'len','a':V('chosen')},1),'label':'unique trig branch'},
 {'set':'c','expr':{'op':'index','a':{'op':'index','a':V('chosen'),'i':0},'i':0}},
 {'set':'dphi','expr':DIV(MUL(5,SUB(5,MUL(2,V('c')))),SUB(29,MUL(20,V('c'))))},
 {'set':'dt','expr':SUB(MUL(5,V('c')),2)},
 {'return':DIV(V('dphi'),V('dt'))}
],
'SERIES_THRESHOLD_CASES':[
 {'set':'rs','expr':{'op':'qroots','a':6,'b':1,'c':-1}}, {'set':'neg','expr':{'op':'list','items':[]}},
 {'for':'r','in':V('rs'),'body':[{'if':{'op':'and','args':[GT(V('r'),-1),LT(V('r'),0)]},'then':[{'append':'neg','expr':V('r')}]}]},
 {'assert':EQ({'op':'len','a':V('neg')},1),'label':'unique negative ratio'}, {'set':'r','expr':{'op':'index','a':V('neg'),'i':0}},
 {'set':'lin','expr':SUB(V('r'),1)}, {'set':'quad','expr':DIV(POW(V('r'),4),SUB(1,POW(V('r'),2)))},
 {'set':'As','expr':{'op':'qroots','a':V('quad'),'b':V('lin'),'c':-12}}, {'set':'posA','expr':{'op':'list','items':[]}},
 {'for':'A','in':V('As'),'body':[{'if':GT(V('A'),0),'then':[{'append':'posA','expr':V('A')}]}]},
 {'assert':EQ({'op':'len','a':V('posA')},1),'label':'unique positive A'}, {'set':'A','expr':{'op':'index','a':V('posA'),'i':0}},
 {'assert':{'op':'ge','a':{'op':'abs','a':MUL(V('A'),V('r'))},'b':10},'label':'threshold 2'},
 {'assert':LT({'op':'abs','a':MUL(V('A'),POW(V('r'),2))},10),'label':'threshold 3'},
 {'return':MUL(36,MUL(V('A'),POW(V('r'),3)),MUL(V('A'),POW(V('r'),5)))}
],
'INVERSE_INTEGRAL_SUBSTITUTION':[
 {'set':'cs','expr':{'op':'list','items':[]}},
 {'for':'n','in':{'op':'range','start':0,'stop':4},'body':[
  {'set':'expo','expr':{'op':'if','cond':EQ({'op':'mod','a':V('n'),'b':2},0),'then':MUL(2,V('n')),'else':SUB(MUL(2,V('n')),1)}},
  {'append':'cs','expr':POW(2,V('expo'))}
 ]},
 {'set':'coeff','expr':DIV(SUB(MUL(3,{'op':'index','a':V('cs'),'i':3}),{'op':'index','a':V('cs'),'i':1}),4)},
 {'return':ADD({'op':'numerator','a':V('coeff')},{'op':'denominator','a':V('coeff')})}
]
}

IDS=['2027_9_14','2027_9_15','2027_9_21','2027_9_22','2027_9_28','2027_9_29','2027_9_30']
OFFICIAL={'2027_9_14':171,'2027_9_15':Fraction(10),'2027_9_21':Fraction(12),'2027_9_22':97,'2027_9_28':Fraction(17,26),'2027_9_29':Fraction(81),'2027_9_30':49}

def jsonable(x):
    if isinstance(x,Fraction): return str(x)
    if isinstance(x,np.generic): return x.item()
    return x


def vm_selftest():
    rng=random.Random(20270927); vm=VM(); ok=0
    for _ in range(200):
        r1=rng.randint(-12,12); r2=rng.randint(-12,12); a=rng.randint(1,7); b=-a*(r1+r2); c=a*r1*r2
        got=sorted(qroots(a,b,c)); exp=sorted([Fraction(r1),Fraction(r2)]); assert got==exp; ok+=1
    for _ in range(100):
        p=rng.randint(1,20); q=rng.randint(1,20); z=Fraction(p*p,q*q); assert sqrt_frac(z)==Fraction(p,q); ok+=1
    return ok

if __name__=='__main__':
    data=json.loads(zlib.decompress(base64.b64decode(v5.DATA_B64)).decode()); classes=data['classes']
    X=np.asarray([r['x'] for r in data['train']],float); Xt=np.asarray([r['x'] for r in data['test']],float); valid=np.asarray(data['valid'],bool)
    Y=np.zeros((len(X),len(classes)))
    for i,r in enumerate(data['train']):Y[i,classes.index(r['label'])]=1.
    ytest=np.asarray([classes.index(r['label']) for r in data['test']],int)
    comp=pd.read_csv(COMP,index_col=0); con=pd.read_parquet(CONN,columns=['Presynaptic_Index','Postsynaptic_Index','Excitatory x Connectivity'])
    n=len(comp); pre=con['Presynaptic_Index'].to_numpy(np.int32,copy=False); post=con['Postsynaptic_Index'].to_numpy(np.int32,copy=False); wt=con['Excitatory x Connectivity'].to_numpy(np.int32,copy=False)
    W=sp.csr_matrix((wt,(pre,post)),shape=(n,n),dtype=np.int32); W.sum_duplicates(); W.sort_indices()
    avail=np.flatnonzero(np.diff(W.indptr)>0); stim=avail[:X.shape[1]].astype(np.int32)
    f1=v5.ragged(W.indptr.astype(np.int64,copy=False),stim.astype(np.int64)); one=np.unique(W.indices[f1]) if f1.size else np.empty(0,np.int32)
    f2=v5.ragged(W.indptr.astype(np.int64,copy=False),one[:256].astype(np.int64)) if one.size else np.empty(0,np.int64); two=np.unique(W.indices[f2]) if f2.size else np.empty(0,np.int32)
    read=np.unique(np.concatenate([stim,one,two]))[:512].astype(np.int32); brain=v5.Brain(W,stim,read)
    print(f'V7_BRAIN neurons={n} edges={W.nnz} stim={len(stim)} read={len(read)} train={len(X)} heldout={len(Xt)}',flush=True)
    def enc(rows,label):
        ds=[]; meta=[]; t=time.perf_counter()
        for i,x in enumerate(rows):
            _,d,a,s=brain.encode(x); ds.append(d); meta.append((a,s))
            if (i+1)%20==0 or i+1==len(rows):print(f'{label} {i+1}/{len(rows)}',flush=True)
        return np.asarray(ds,float),meta,time.perf_counter()-t
    D,mtr,ttr=enc(X,'train'); Dt,mte,tte=enc(Xt,'test'); DS,DSt,keep=v5.standardize(D,Dt)
    best=None
    for alpha in np.logspace(-3,4,30):
        M=v5.ridge(DS[~valid],Y[~valid],alpha); va=float(np.mean(np.argmax(DS[valid]@M,1)==np.argmax(Y[valid],1)))
        if best is None or va>best[0] or (va==best[0] and alpha>best[1]):best=(va,float(alpha))
    M=v5.ridge(DS,Y,best[1]); pred=np.argmax(DSt@M,1); strategy_acc=float(np.mean(pred==ytest))
    vm=VM(); rows=[]
    for i,pidx in enumerate(pred):
        pid=IDS[i]; strategy=classes[pidx]; expected=classes[ytest[i]]
        if strategy in PROGRAMS: result,env=vm.run(PROGRAMS[strategy])
        else: result,env=None,{}
        ok=result==OFFICIAL[pid]
        row={'id':pid,'predicted_strategy':strategy,'expected_strategy':expected,'strategy_correct':bool(pidx==ytest[i]),'vm_result':jsonable(result),'official':jsonable(OFFICIAL[pid]),'final_correct':bool(ok),'active_neurons':mte[i][0],'spikes':mte[i][1]}
        rows.append(row); print('V7_ITEM='+json.dumps(row,ensure_ascii=False),flush=True)
    final_acc=sum(r['final_correct'] for r in rows)/len(rows); selftests=vm_selftest()
    out={'source_commit':os.environ.get('UPSTREAM_COMMIT'),'graph_neurons':n,'unique_edges':int(W.nnz),'strategy_heldout_accuracy':strategy_acc,'final_answer_accuracy':final_acc,'downstream_feature_count':keep,'generic_vm_programs':len(PROGRAMS),'per_item_python_solver_functions':0,'vm_operator_types_used':len(vm.operator_hits),'vm_operator_hits':vm.operator_hits,'vm_selftests_passed':selftests,'brain_train_wall_s':ttr,'brain_test_wall_s':tte,'items':rows,'scope_note':'V7 removes per-item Python solver functions. A reusable low-level arithmetic/algebra/control-flow VM executes declarative strategy programs selected from downstream v783 neural state. The programs still require structured mathematical facts and strategy templates; this is not raw Korean-text autonomous theorem solving.'}
    Path('suneung_generic_v7_result.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,default=jsonable),encoding='utf-8')
    print('V7_FINAL='+json.dumps(out,ensure_ascii=False,default=jsonable),flush=True)
    if strategy_acc<1.0 or final_acc<1.0 or selftests!=300:raise SystemExit('V7 benchmark failed')
