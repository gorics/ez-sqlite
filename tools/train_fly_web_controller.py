from __future__ import annotations

import gc
import hashlib
import ipaddress
import json
import os
import socket
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

ACTIONS = ["OPEN_HOME", "FOLLOW_1", "FOLLOW_2", "BACK", "REFRESH"]
COMMANDS = {
    "OPEN_HOME": "파이썬 홈페이지 열어",
    "FOLLOW_1": "첫 번째 링크 열어",
    "FOLLOW_2": "두 번째 링크 열어",
    "BACK": "뒤로 가",
    "REFRESH": "새로고침",
}
HOME = "https://www.python.org/"
MAX_BYTES = 1_000_000
TIMEOUT = 12


@dataclass
class Page:
    url: str
    status: int
    title: str
    text: str
    links: list[str]


class ReadOnlyWeb:
    """GET-only browser. No forms, POST, cookies, credentials, or private-network URLs."""

    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": "FruitFly-v783-readonly-research/1.0"})
        self.history: list[Page] = []
        self.current: Page | None = None

    @staticmethod
    def _public_http_url(url: str) -> str:
        url, _ = urldefrag(url)
        p = urlparse(url)
        if p.scheme not in {"http", "https"} or not p.hostname:
            raise ValueError("only public http(s) URLs are allowed")
        host = p.hostname.lower()
        if host in {"localhost", "localhost.localdomain"}:
            raise ValueError("local hosts are blocked")
        infos = socket.getaddrinfo(host, p.port or (443 if p.scheme == "https" else 80), type=socket.SOCK_STREAM)
        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if not ip.is_global:
                raise ValueError(f"non-public address blocked: {ip}")
        return url

    def fetch(self, url: str, push_history: bool = True) -> Page:
        url = self._public_http_url(url)
        if push_history and self.current is not None:
            self.history.append(self.current)
        with self.s.get(url, timeout=TIMEOUT, allow_redirects=True, stream=True) as r:
            r.raise_for_status()
            final = self._public_http_url(r.url)
            ctype = (r.headers.get("content-type") or "").lower()
            if "text/html" not in ctype and "text/plain" not in ctype:
                raise ValueError(f"unsupported content type: {ctype}")
            chunks = []
            total = 0
            for chunk in r.iter_content(65536):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_BYTES:
                    raise ValueError("page exceeds read-only 1 MB limit")
                chunks.append(chunk)
            raw = b"".join(chunks)
            enc = r.encoding or "utf-8"
            html = raw.decode(enc, errors="replace")
            soup = BeautifulSoup(html, "html.parser")
            title = soup.title.get_text(" ", strip=True) if soup.title else ""
            text = " ".join(soup.stripped_strings)[:12000]
            links: list[str] = []
            seen = set()
            for a in soup.find_all("a", href=True):
                href = urljoin(final, a.get("href"))
                href, _ = urldefrag(href)
                try:
                    href = self._public_http_url(href)
                except Exception:
                    continue
                if href not in seen:
                    seen.add(href)
                    links.append(href)
                if len(links) >= 30:
                    break
            page = Page(final, int(r.status_code), title, text, links)
            self.current = page
            return page

    def execute(self, action: str) -> Page:
        if action == "OPEN_HOME":
            return self.fetch(HOME)
        if self.current is None:
            raise RuntimeError("no current page")
        if action == "FOLLOW_1":
            if len(self.current.links) < 1:
                raise RuntimeError("no first link")
            return self.fetch(self.current.links[0])
        if action == "FOLLOW_2":
            if len(self.current.links) < 2:
                raise RuntimeError("no second link")
            return self.fetch(self.current.links[1])
        if action == "BACK":
            if not self.history:
                return self.current
            page = self.history.pop()
            return self.fetch(page.url, push_history=False)
        if action == "REFRESH":
            return self.fetch(self.current.url, push_history=False)
        raise ValueError(action)


def load_connectome():
    t = time.perf_counter()
    comp = pd.read_csv(COMP, index_col=0)
    con = pd.read_parquet(CONN, columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
    n = len(comp)
    pre = con["Presynaptic_Index"].to_numpy(np.int32, copy=False)
    post = con["Postsynaptic_Index"].to_numpy(np.int32, copy=False)
    wt = con["Excitatory x Connectivity"].to_numpy(np.int32, copy=False)
    W = sp.csr_matrix((wt, (pre, post)), shape=(n, n), dtype=np.int32)
    W.sum_duplicates(); W.sort_indices()
    if n < 130000 or int(max(pre.max(), post.max())) >= n:
        raise RuntimeError("v783 connectome integrity check failed")
    return W, len(con), time.perf_counter() - t


def page_bits(page: Page, nbits: int = 4) -> np.ndarray:
    payload = (page.title + "\n" + page.text[:2000] + "\n" + "\n".join(page.links[:8])).encode("utf-8", "replace")
    digest = hashlib.sha256(payload).digest()
    return np.array([(digest[0] >> i) & 1 for i in range(nbits)], dtype=np.int8)


def reset_engine(e: ActiveEngine, seed: int):
    e.rng = np.random.default_rng(seed)
    e.v.fill(V0); e.g.fill(0); e.rfc.fill(0); e.count.fill(0)
    e.rfc_len.fill(round(2.2 / DT))
    e.poi = np.empty(0, np.int64); e.pp = np.empty(0, np.float64)
    e.active_mask.fill(False); e.active_count = 0
    e.ev_targets = [None] * len(e.ev_targets); e.ev_weights = [None] * len(e.ev_weights); e.slot = 0


def choose_sensory_and_probes(W: sp.csr_matrix):
    outdeg = np.diff(W.indptr)
    sensory = np.argsort(outdeg)[-14:].astype(np.int64)
    cmd = sensory[:10].reshape(5, 2)
    context = sensory[10:14]
    score: dict[int, int] = {}
    sensory_set = set(map(int, sensory))
    for src in sensory:
        a, b = W.indptr[src], W.indptr[src + 1]
        for dst, w in zip(W.indices[a:b], W.data[a:b]):
            d = int(dst)
            if d in sensory_set:
                continue
            score[d] = score.get(d, 0) + abs(int(w))
    probes = np.array([k for k, _ in sorted(score.items(), key=lambda kv: kv[1], reverse=True)[:128]], dtype=np.int64)
    if len(probes) < 32:
        raise RuntimeError("not enough downstream probe neurons")
    return cmd, context, probes


def encode_and_run(e: ActiveEngine, action_idx: int, page: Page, cmd_neurons, context_neurons, probes, seed: int):
    reset_engine(e, seed)
    bits = page_bits(page, len(context_neurons))
    stim = list(map(int, cmd_neurons[action_idx]))
    stim += [int(n) for n, bit in zip(context_neurons, bits) if bit]
    e.set_poisson(np.array(stim, np.int64), rate=250.0)
    for _ in range(round(55.0 / DT)):
        e.step()
    feature_idx = np.concatenate([cmd_neurons.reshape(-1), context_neurons, probes])
    x = np.log1p(e.count[feature_idx].astype(np.float64))
    x = np.concatenate([x, [np.log1p(float(e.count.sum())), np.log1p(float(e.active_count))]])
    norm = np.linalg.norm(x)
    if norm > 0:
        x /= norm
    return x, int(e.count.sum()), int(e.active_count), stim


def train_perceptron(X: np.ndarray, y: np.ndarray, classes: int, epochs: int = 80):
    Xb = np.c_[X, np.ones(len(X))]
    W = np.zeros((classes, Xb.shape[1]), dtype=np.float64)
    rng = np.random.default_rng(20260927)
    mistakes = []
    for epoch in range(epochs):
        m = 0
        for i in rng.permutation(len(Xb)):
            pred = int(np.argmax(W @ Xb[i]))
            target = int(y[i])
            if pred != target:
                W[target] += Xb[i]
                W[pred] -= Xb[i]
                m += 1
        mistakes.append(m)
        if m == 0:
            break
    return W, mistakes


def predict(Wr: np.ndarray, x: np.ndarray) -> int:
    return int(np.argmax(Wr @ np.r_[x, 1.0]))


def main():
    browser = ReadOnlyWeb()
    training_urls = [
        "https://example.com/",
        "https://www.python.org/",
        "https://www.iana.org/help/example-domains",
    ]
    pages = []
    web_checks = []
    for url in training_urls:
        try:
            p = browser.fetch(url, push_history=False)
            pages.append(p)
            web_checks.append({"url": p.url, "status": p.status, "title": p.title[:120], "links": len(p.links)})
            print("WEB_OK " + json.dumps(web_checks[-1], ensure_ascii=False), flush=True)
        except Exception as ex:
            web_checks.append({"url": url, "error": f"{type(ex).__name__}: {ex}"})
            print("WEB_FAIL " + json.dumps(web_checks[-1], ensure_ascii=False), flush=True)
    if len(pages) < 2:
        raise RuntimeError("fewer than two live public pages were reachable")

    W, parquet_rows, load_s = load_connectome()
    print(f"REAL_V783 neurons={W.shape[0]} edges={W.nnz} parquet_rows={parquet_rows} load_s={load_s:.3f}", flush=True)
    cmd_neurons, context_neurons, probes = choose_sensory_and_probes(W)
    e = ActiveEngine(W, seed=1)

    X = []
    y = []
    sample_meta = []
    seed = 1000
    for p in pages:
        for ai, action in enumerate(ACTIONS):
            for rep in range(3):
                x, spikes, active, stim = encode_and_run(e, ai, p, cmd_neurons, context_neurons, probes, seed)
                X.append(x); y.append(ai)
                sample_meta.append({"action": action, "url": p.url, "seed": seed, "spikes": spikes, "active": active, "stim": stim})
                seed += 1
    X = np.asarray(X); y = np.asarray(y)
    readout, mistakes = train_perceptron(X, y, len(ACTIONS))
    train_pred = np.array([predict(readout, x) for x in X])
    train_acc = float(np.mean(train_pred == y))
    print(f"LEARN train_samples={len(X)} train_accuracy={train_acc:.6f} epochs={len(mistakes)} final_mistakes={mistakes[-1]}", flush=True)

    # Generalisation: unseen Poisson seeds on live pages.
    eval_rows = []
    correct = 0
    total = 0
    for pi, p in enumerate(pages):
        for ai, action in enumerate(ACTIONS):
            x, spikes, active, stim = encode_and_run(e, ai, p, cmd_neurons, context_neurons, probes, 9000 + pi * 100 + ai)
            pred = predict(readout, x)
            ok = pred == ai
            correct += int(ok); total += 1
            row = {"command": COMMANDS[action], "expected": action, "predicted": ACTIONS[pred], "correct": ok, "url": p.url, "spikes": spikes, "active": active}
            eval_rows.append(row)
            print("EVAL " + json.dumps(row, ensure_ascii=False), flush=True)
    eval_acc = correct / total

    # Actual live read-only browser manipulation. All actions below are selected by the learned readout.
    browser = ReadOnlyWeb()
    browser.fetch("https://example.com/", push_history=False)
    live_sequence = ["OPEN_HOME", "FOLLOW_1", "BACK", "FOLLOW_2", "REFRESH"]
    executed = []
    for step, target_action in enumerate(live_sequence):
        before = browser.current
        ai = ACTIONS.index(target_action)
        x, spikes, active, stim = encode_and_run(e, ai, before, cmd_neurons, context_neurons, probes, 12000 + step)
        pred_action = ACTIONS[predict(readout, x)]
        try:
            after = browser.execute(pred_action)
            status = "ok"
            err = None
        except Exception as ex:
            after = browser.current
            status = "error"
            err = f"{type(ex).__name__}: {ex}"
        row = {
            "step": step + 1,
            "command": COMMANDS[target_action],
            "target_action": target_action,
            "brain_action": pred_action,
            "correct_action": pred_action == target_action,
            "before_url": before.url if before else None,
            "after_url": after.url if after else None,
            "http_status": after.status if after else None,
            "result": status,
            "error": err,
            "spikes": spikes,
            "active_neurons": active,
        }
        executed.append(row)
        print("LIVE_ACTION " + json.dumps(row, ensure_ascii=False), flush=True)

    result = {
        "mode": "read-only public internet GET navigation",
        "no_post_no_forms_no_login": True,
        "source_commit": os.environ.get("UPSTREAM_COMMIT"),
        "v783_neurons": int(W.shape[0]),
        "v783_edges": int(W.nnz),
        "parquet_rows": int(parquet_rows),
        "web_checks": web_checks,
        "actions": ACTIONS,
        "commands_ko": COMMANDS,
        "learning_layer": "online multiclass perceptron readout over real v783 LIF spike features",
        "fixed_connectome_note": "upstream v783 LIF synapses are fixed; learning occurs in the attached plastic action readout",
        "training_samples": int(len(X)),
        "train_accuracy": train_acc,
        "perceptron_epoch_mistakes": mistakes,
        "unseen_seed_eval_accuracy": eval_acc,
        "eval": eval_rows,
        "live_actions": executed,
        "all_live_actions_correct": bool(all(r["correct_action"] and r["result"] == "ok" for r in executed)),
        "sensory_command_neurons": cmd_neurons.tolist(),
        "sensory_context_neurons": context_neurons.tolist(),
        "probe_neurons": probes.tolist(),
        "load_s": load_s,
    }
    Path("fly_web_learning_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    Path("fly_web_readout.npy").write_bytes(readout.astype(np.float64).tobytes())
    print("WEB_LEARNING_RESULT=" + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    gc.collect()


if __name__ == "__main__":
    main()
