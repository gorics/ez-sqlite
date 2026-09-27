from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag

import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp
from bs4 import BeautifulSoup

from run_fly_v783_fullbrain import ActiveEngine, V0, DT

UP = Path(os.environ.get("DROSOPHILA_MODEL_DIR", "Drosophila_brain_model"))
COMP = UP / "Completeness_783.csv"
CONN = UP / "Connectivity_783.parquet"
ROOT = Path("agent_workspace").resolve()
ROOT.mkdir(exist_ok=True)

ACTIONS = ["LIST_FILES", "READ_FILE", "WRITE_FILE", "WEB_OPEN", "WEB_FOLLOW", "WEB_BACK", "RUN_PROGRAM"]
TRAIN_PHRASES = {
    "LIST_FILES": ["파일 목록 보여줘", "현재 폴더 확인", "디렉터리 내용 확인", "파일들을 살펴봐"],
    "READ_FILE": ["파일 읽어", "노트 내용을 확인해", "텍스트 파일 열어봐", "저장한 문서를 읽어"],
    "WRITE_FILE": ["파일에 써", "노트를 저장해", "문서로 기록해", "결과를 파일에 남겨"],
    "WEB_OPEN": ["웹사이트 열어", "인터넷 페이지 접속", "파이썬 홈페이지 열어", "웹 페이지로 이동해"],
    "WEB_FOLLOW": ["첫 링크 열어", "링크 따라가", "페이지 링크를 눌러", "다음 링크로 이동해"],
    "WEB_BACK": ["뒤로 가", "이전 페이지로", "웹에서 뒤로 이동", "직전 페이지로 돌아가"],
    "RUN_PROGRAM": ["프로그램 실행", "파이썬 계산 돌려", "컴퓨터에서 코드 실행해", "테스트 프로그램을 실행해"],
}
EVAL_PHRASES = {
    "LIST_FILES": "현재 폴더 파일을 살펴봐",
    "READ_FILE": "저장한 파일 내용을 읽어봐",
    "WRITE_FILE": "작업 결과를 문서로 기록해",
    "WEB_OPEN": "인터넷 사이트에 접속해",
    "WEB_FOLLOW": "페이지의 첫 링크로 이동해",
    "WEB_BACK": "이전 웹 페이지로 돌아가",
    "RUN_PROGRAM": "컴퓨터에서 테스트 프로그램을 실행해",
}
TOKENS = ["파일","목록","폴더","디렉터리","내용","읽","열어","문서","노트","쓰","저장","기록","남겨","웹","사이트","인터넷","페이지","접속","링크","따라","눌러","이동","뒤로","이전","직전","돌아","프로그램","컴퓨터","코드","실행","파이썬","계산","테스트","결과","현재","확인"]

@dataclass
class Page:
    url: str
    status: int
    title: str
    text: str
    links: list[str]

class PublicBrowser:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": "FruitFly-v783-autonomous-lab/1.0"})
        self.current = None
        self.history = []
    @staticmethod
    def validate(url):
        url, _ = urldefrag(url)
        p = urlparse(url)
        if p.scheme not in {"http","https"} or not p.hostname:
            raise ValueError("public http(s) only")
        for info in socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == "https" else 80), type=socket.SOCK_STREAM):
            if not ipaddress.ip_address(info[4][0]).is_global:
                raise ValueError("non-public address blocked")
        return url
    def fetch(self, url, push=True):
        url = self.validate(url)
        if push and self.current is not None:
            self.history.append(self.current)
        r = self.s.get(url, timeout=12, allow_redirects=True)
        r.raise_for_status()
        final = self.validate(r.url)
        html = r.content[:1_000_000].decode(r.encoding or "utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        text = " ".join(soup.stripped_strings)[:8000]
        links, seen = [], set()
        for a in soup.find_all("a", href=True):
            try:
                href = self.validate(urljoin(final, a.get("href")))
            except Exception:
                continue
            if href not in seen:
                seen.add(href); links.append(href)
            if len(links) >= 30:
                break
        self.current = Page(final, int(r.status_code), title, text, links)
        return self.current
    def follow(self):
        if self.current is None or not self.current.links:
            raise RuntimeError("no link")
        for link in self.current.links:
            if link != self.current.url:
                return self.fetch(link)
        return self.fetch(self.current.links[0])
    def back(self):
        if not self.history:
            return self.current
        p = self.history.pop()
        return self.fetch(p.url, push=False)

class IsolatedComputer:
    def __init__(self):
        self.root = ROOT
        self.browser = PublicBrowser()
    def _path(self, rel):
        p = (self.root / rel).resolve()
        if p != self.root and self.root not in p.parents:
            raise ValueError("path escapes workspace")
        return p
    def execute(self, action, payload):
        if action == "LIST_FILES":
            return {"cwd": str(self.root), "entries": [p.name for p in sorted(self.root.iterdir())]}
        if action == "READ_FILE":
            p = self._path(payload.get("path", "brain_note.txt"))
            return {"path": p.name, "text": p.read_text(errors="replace")[:5000]}
        if action == "WRITE_FILE":
            p = self._path(payload.get("path", "brain_note.txt"))
            p.write_text(payload.get("text", "fruit fly brain was here\n"))
            return {"path": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        if action == "WEB_OPEN":
            p = self.browser.fetch(payload.get("url", "https://www.python.org/"))
            return {"url": p.url, "status": p.status, "title": p.title[:120], "links": len(p.links)}
        if action == "WEB_FOLLOW":
            p = self.browser.follow()
            return {"url": p.url, "status": p.status, "title": p.title[:120], "links": len(p.links)}
        if action == "WEB_BACK":
            p = self.browser.back()
            return {"url": p.url, "status": p.status, "title": p.title[:120], "links": len(p.links)}
        if action == "RUN_PROGRAM":
            code = "print('fly-program-ok', sum(i*i for i in range(20)))"
            cp = subprocess.run(["python", "-c", code], cwd=self.root, capture_output=True, text=True, timeout=10)
            return {"returncode": cp.returncode, "stdout": cp.stdout.strip(), "stderr": cp.stderr.strip()}
        raise ValueError(action)

def load_connectome():
    t = time.perf_counter()
    comp = pd.read_csv(COMP, index_col=0)
    con = pd.read_parquet(CONN, columns=["Presynaptic_Index","Postsynaptic_Index","Excitatory x Connectivity"])
    n = len(comp)
    pre = con["Presynaptic_Index"].to_numpy(np.int32, copy=False)
    post = con["Postsynaptic_Index"].to_numpy(np.int32, copy=False)
    wt = con["Excitatory x Connectivity"].to_numpy(np.int32, copy=False)
    W = sp.csr_matrix((wt,(pre,post)), shape=(n,n), dtype=np.int32)
    W.sum_duplicates(); W.sort_indices()
    if n < 130000 or max(int(pre.max()), int(post.max())) >= n:
        raise RuntimeError("v783 integrity failure")
    return W, len(con), time.perf_counter()-t

def reset_engine(e, seed):
    e.rng = np.random.default_rng(seed)
    e.v.fill(V0); e.g.fill(0); e.rfc.fill(0); e.count.fill(0)
    e.rfc_len.fill(round(2.2/DT)); e.poi=np.empty(0,np.int64); e.pp=np.empty(0,np.float64)
    e.active_mask.fill(False); e.active_count=0; e.ev_targets=[None]*len(e.ev_targets); e.ev_weights=[None]*len(e.ev_weights); e.slot=0

def choose_neurons(W):
    ranked = np.argsort(np.diff(W.indptr))[::-1].astype(np.int64)
    sensory = ranked[:len(TOKENS)]
    ss = set(map(int,sensory)); score={}
    for src in sensory:
        a,b=W.indptr[src],W.indptr[src+1]
        for dst,w in zip(W.indices[a:b],W.data[a:b]):
            d=int(dst)
            if d not in ss: score[d]=score.get(d,0)+abs(int(w))
    probes=np.array([k for k,_ in sorted(score.items(), key=lambda kv:kv[1], reverse=True)[:192]], dtype=np.int64)
    return sensory, probes

def token_indices(text):
    idx=[i for i,t in enumerate(TOKENS) if t in text]
    if idx: return idx
    h=hashlib.sha256(text.encode()).digest()
    return sorted({b%len(TOKENS) for b in h[:4]})

def encode(e,text,sensory,probes,seed):
    reset_engine(e,seed)
    ti=token_indices(text); stim=sensory[np.asarray(ti,dtype=np.int64)]
    e.set_poisson(stim, rate=250.0)
    for _ in range(round(55.0/DT)): e.step()
    fi=np.concatenate([sensory,probes])
    x=np.log1p(e.count[fi].astype(np.float64))
    x=np.r_[x,np.log1p(float(e.count.sum())),np.log1p(float(e.active_count))]
    n=np.linalg.norm(x)
    if n: x/=n
    return x,{"tokens":[TOKENS[i] for i in ti],"spikes":int(e.count.sum()),"active_neurons":int(e.active_count)}

def train(X,y,nc):
    Xb=np.c_[X,np.ones(len(X))]; R=np.zeros((nc,Xb.shape[1])); rng=np.random.default_rng(20260927); hist=[]
    for _ in range(120):
        m=0
        for i in rng.permutation(len(Xb)):
            p=int(np.argmax(R@Xb[i])); t=int(y[i])
            if p!=t: R[t]+=Xb[i]; R[p]-=Xb[i]; m+=1
        hist.append(m)
        if m==0: break
    return R,hist

def pred(R,x): return int(np.argmax(R@np.r_[x,1.0]))

def next_goal(state):
    if not state.get("listed"): return EVAL_PHRASES["LIST_FILES"],"LIST_FILES",{}
    if not state.get("wrote"): return EVAL_PHRASES["WRITE_FILE"],"WRITE_FILE",{"path":"brain_note.txt","text":"real FlyWire v783 autonomous computer session\n"}
    if not state.get("read"): return EVAL_PHRASES["READ_FILE"],"READ_FILE",{"path":"brain_note.txt"}
    if not state.get("web_opened"): return EVAL_PHRASES["WEB_OPEN"],"WEB_OPEN",{"url":"https://www.python.org/"}
    if not state.get("web_followed"): return EVAL_PHRASES["WEB_FOLLOW"],"WEB_FOLLOW",{}
    if not state.get("web_back"): return EVAL_PHRASES["WEB_BACK"],"WEB_BACK",{}
    if not state.get("program"): return EVAL_PHRASES["RUN_PROGRAM"],"RUN_PROGRAM",{}
    return None,None,None

def update_state(state,expected,action,result):
    if action!=expected: return
    key={"LIST_FILES":"listed","WRITE_FILE":"wrote","READ_FILE":"read","WEB_OPEN":"web_opened","WEB_FOLLOW":"web_followed","WEB_BACK":"web_back","RUN_PROGRAM":"program"}[expected]
    state[key]=True

def main():
    W,rows,load_s=load_connectome(); sensory,probes=choose_neurons(W); e=ActiveEngine(W,seed=1)
    print(f"REAL_V783 neurons={W.shape[0]} edges={W.nnz} rows={rows} load_s={load_s:.3f}",flush=True)
    X=[]; y=[]; seed=1000
    for ai,a in enumerate(ACTIONS):
        for phrase in TRAIN_PHRASES[a]:
            for _ in range(2):
                x,_m=encode(e,phrase,sensory,probes,seed); X.append(x); y.append(ai); seed+=1
    X=np.asarray(X); y=np.asarray(y); R,h=train(X,y,len(ACTIONS))
    train_acc=float(np.mean([pred(R,x)==int(t) for x,t in zip(X,y)]))
    print(f"TRAIN samples={len(X)} accuracy={train_acc:.6f} mistakes={h}",flush=True)
    ev=[]
    for ai,a in enumerate(ACTIONS):
        x,m=encode(e,EVAL_PHRASES[a],sensory,probes,9000+ai); pa=ACTIONS[pred(R,x)]
        row={"expected":a,"predicted":pa,"correct":pa==a,"phrase":EVAL_PHRASES[a],**m}; ev.append(row); print("EVAL "+json.dumps(row,ensure_ascii=False),flush=True)
    c=IsolatedComputer(); state={}; trace=[]
    for step in range(12):
        intent,expected,payload=next_goal(state)
        if intent is None: break
        x,m=encode(e,intent,sensory,probes,12000+step); action=ACTIONS[pred(R,x)]
        try: result=c.execute(action,payload); ok=True; err=None
        except Exception as ex: result={}; ok=False; err=f"{type(ex).__name__}: {ex}"
        if ok: update_state(state,expected,action,result)
        row={"step":step+1,"intent":intent,"expected":expected,"brain_action":action,"classification_correct":action==expected,"executed_ok":ok,"error":err,"brain":m,"result":result,"state":dict(state)}
        trace.append(row); print("AUTONOMOUS_STEP "+json.dumps(row,ensure_ascii=False),flush=True)
    complete=all(state.get(k) for k in ["listed","wrote","read","web_opened","web_followed","web_back","program"])
    out={"source_commit":os.environ.get("UPSTREAM_COMMIT"),"v783_neurons":int(W.shape[0]),"v783_edges":int(W.nnz),"training_samples":len(X),"train_accuracy":train_acc,"heldout_accuracy":float(np.mean([r["correct"] for r in ev])),"eval":ev,"autonomous_trace":trace,"autonomous_goal_complete":bool(complete),"permissions":{"filesystem":"read/write inside disposable agent_workspace","program_execution":"predefined local Python process","internet":"public HTTP/HTTPS GET navigation","private_network":"blocked","github_credentials":"not exposed","user_device_access":False},"load_s":load_s}
    Path("fly_autonomous_result.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)); np.save("fly_autonomous_readout.npy",R)
    print("AUTONOMOUS_RESULT="+json.dumps(out,ensure_ascii=False,sort_keys=True),flush=True)
    if not complete: raise RuntimeError("autonomous goal incomplete")
if __name__=="__main__": main()
