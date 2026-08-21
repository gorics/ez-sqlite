from __future__ import annotations

import hashlib, json, re, sys, time
from pathlib import Path
from urllib.parse import urljoin

import fitz
import requests
from bs4 import BeautifulSoup

ROOT = Path('tmp_soc_bundle')
SRC = ROOT / 'sources'
OUT = ROOT / 'output'
SRC.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36'
S = requests.Session()
S.headers.update({'User-Agent': UA, 'Accept': '*/*'})

SDF = {y: f'https://sdfallyy.tistory.com/{n}' for y, n in {
    2014:386, 2015:389, 2016:391, 2017:393, 2018:395, 2019:397,
    2020:399, 2021:401, 2022:403, 2023:405, 2024:407,
}.items()}

HORAENG = {
    (2023,'6월'):'https://horaeng.com/223', (2023,'9월'):'https://horaeng.com/252',
    (2024,'6월'):'https://horaeng.com/263', (2024,'9월'):'https://horaeng.com/267', (2024,'수능'):'https://horaeng.com/350',
    (2025,'6월'):'https://horaeng.com/279', (2025,'9월'):'https://horaeng.com/353', (2025,'수능'):'https://horaeng.com/420',
    (2026,'6월'):'https://horaeng.com/438', (2026,'9월'):'https://horaeng.com/442', (2026,'수능'):'https://horaeng.com/446',
    (2027,'6월'):'https://horaeng.com/462',
}

LEGEND = {
    (2014,'6월'):'https://legendstudy.com/280',
    (2014,'9월'):'https://legendstudy.com/281',
    (2014,'수능'):'https://legendstudy.com/268',
}

SESSIONS = [(y,s) for y in range(2014,2027) for s in ('6월','9월','수능')] + [(2027,'6월')]
assert len(SESSIONS)==40


def clean(s: str) -> str:
    return re.sub(r'[^0-9a-zA-Z가-힣]+','', (s or '')).lower()


def session_token(s: str) -> str:
    return s


def fetch(url: str, referer: str | None = None, binary: bool = False, tries: int = 5):
    err = None
    for i in range(tries):
        try:
            h = {'Referer': referer} if referer else {}
            r = S.get(url, headers=h, timeout=45, allow_redirects=True)
            if r.status_code == 200:
                if binary:
                    b = r.content
                    if b[:4] == b'%PDF':
                        return r
                    # Some hosts prepend whitespace/BOM. Find a near-leading PDF header.
                    j = b.find(b'%PDF',0,1024)
                    if j >= 0:
                        r._content = b[j:]
                        return r
                else:
                    return r
            err = f'HTTP {r.status_code} final={r.url} ct={r.headers.get("content-type")} len={len(r.content)}'
        except Exception as e:
            err = repr(e)
        time.sleep(1.2*(i+1))
    raise RuntimeError(f'fetch failed {url}: {err}')


def source_pages(year: int, sess: str):
    pages=[]
    if (year,sess) in HORAENG: pages.append(HORAENG[(year,sess)])
    if (year,sess) in LEGEND: pages.append(LEGEND[(year,sess)])
    if year in SDF: pages.append(SDF[year])
    # de-dupe
    return list(dict.fromkeys(pages))


def relevant_anchor(label: str, year: int, sess: str, kind: str):
    n = clean(label)
    if not ('사회문화' in n or '사문' in n):
        return False
    if str(year) not in n:
        # Horaeng session-specific pages sometimes use shortened labels; permit there at scoring stage.
        pass
    if sess == '6월' and '6월' not in n: return False
    if sess == '9월' and '9월' not in n: return False
    if sess == '수능' and '수능' not in n: return False
    if kind == 'problem':
        return ('문제' in n) and not any(x in n for x in ('정답','해설'))
    # answer: any explicitly answer-ish file, with score preferring 정답표.
    return any(x in n for x in ('정답표','정답','해설')) and '문제' not in n


def extract_candidates(page_url: str, year: int, sess: str, kind: str):
    r=fetch(page_url)
    soup=BeautifulSoup(r.text,'html.parser')
    out=[]
    for a in soup.find_all('a',href=True):
        label=' '.join(a.stripped_strings).strip()
        href=urljoin(r.url,a.get('href'))
        if not href.startswith('http'): continue
        if not relevant_anchor(label,year,sess,kind): continue
        n=clean(label)
        score=0
        if str(year) in n: score+=20
        if '사회문화' in n: score+=20
        if sess in n: score+=15
        if kind=='problem':
            if '문제지' in n: score+=10
            elif '문제' in n: score+=8
        else:
            if '정답표' in n: score+=30
            elif '정답' in n and '해설' not in n: score+=20
            elif '정답' in n: score+=12
            elif '해설' in n: score+=5
        if any(host in href for host in ('daumcdn.net','kakaocdn.net','horaeng.com','ebsi.co.kr','kice.re.kr')): score+=5
        out.append((score,label,href,page_url))
    out.sort(key=lambda x:x[0], reverse=True)
    return out


def validate_problem(path: Path, year: int, sess: str, label: str):
    d=fitz.open(path)
    pages=len(d)
    txt='\n'.join(d[i].get_text('text') for i in range(min(2,pages)))
    nt=clean(txt)
    labeln=clean(label)
    subject_text=('사회문화' in nt) or ('사회' in nt and '문화' in nt)
    title_text=(str(year) in nt and (sess in nt or (sess=='수능' and '대학수학능력시험' in nt)))
    anchor_strong=('사회문화' in labeln and str(year) in labeln and sess in labeln)
    ok=(pages==4 and (subject_text or anchor_strong))
    return ok, {'pages':pages,'subject_text':subject_text,'title_text':title_text,'anchor_strong':anchor_strong,'text_head':txt[:500]}


def validate_answer(path: Path):
    d=fitz.open(path)
    return len(d)>=1, {'pages':len(d),'text_head':d[0].get_text('text')[:500] if len(d) else ''}


def resolve(year: int, sess: str, kind: str):
    logs=[]; cands=[]
    for page in source_pages(year,sess):
        try:
            found=extract_candidates(page,year,sess,kind)
            logs.append({'source_page':page,'candidate_count':len(found),'labels':[x[1] for x in found[:8]]})
            cands.extend(found)
        except Exception as e:
            logs.append({'source_page':page,'error':repr(e)})
    # De-dupe URLs while preserving score order globally.
    cands.sort(key=lambda x:x[0], reverse=True)
    seen=set(); uniq=[]
    for c in cands:
        if c[2] in seen: continue
        seen.add(c[2]); uniq.append(c)
    for idx,(score,label,url,referer) in enumerate(uniq):
        dest=SRC / f'{year}_{sess}_{kind}_{idx}.pdf'
        try:
            rr=fetch(url,referer=referer,binary=True)
            dest.write_bytes(rr.content)
            if kind=='problem': ok,info=validate_problem(dest,year,sess,label)
            else: ok,info=validate_answer(dest)
            logs.append({'candidate':url,'label':label,'final_url':rr.url,'bytes':len(rr.content),'validation':info,'ok':ok})
            if ok:
                return dest, rr.url, label, logs
            dest.unlink(missing_ok=True)
        except Exception as e:
            logs.append({'candidate':url,'label':label,'error':repr(e)})
            dest.unlink(missing_ok=True)
    return None,None,None,logs


def make_answer_appendix(answer_items, dest: Path):
    out=fitz.open()
    a4=fitz.paper_rect('a4')
    margin=24
    slots=2
    slot_h=(a4.height-2*margin)/slots
    for i,item in enumerate(answer_items):
        if i%slots==0:
            page=out.new_page(width=a4.width,height=a4.height)
        slot=i%slots
        top=margin + slot*slot_h
        year,sess,path=item
        page.insert_text((margin,top+13),f'{year}학년도 {sess}  빠른정답',fontsize=10,fontname='helv')
        src=fitz.open(path)
        # Prefer the page that visibly contains '정답'; otherwise use first page.
        pi=0
        for j in range(min(3,len(src))):
            t=clean(src[j].get_text('text'))
            if '정답' in t:
                pi=j; break
        sp=src[pi]
        # Separate answer-table PDFs are generally one page. If source is an explanation PDF,
        # crop the upper region where the official answer table appears, excluding explanation body.
        full_text=clean(sp.get_text('text'))
        is_answer_table=(len(src)==1 and ('정답' in full_text or len(full_text)<3000))
        frac=0.96 if is_answer_table else 0.34
        clip=fitz.Rect(sp.rect.x0,sp.rect.y0,sp.rect.x1,sp.rect.y0+sp.rect.height*frac)
        target=fitz.Rect(margin,top+22,a4.width-margin,top+slot_h-10)
        page.show_pdf_page(target,src,pi,clip=clip,keep_proportion=True)
    out.save(dest,garbage=4,deflate=True)


def merge(problem_items, appendix: Path, dest: Path):
    out=fitz.open()
    for year,sess,path in problem_items:
        src=fitz.open(path)
        out.insert_pdf(src)
    app=fitz.open(appendix); out.insert_pdf(app)
    out.set_metadata({'title':'KICE 사회문화 2014학년도-2027학년도 6월 원문 시험지 + 빠른정답'})
    out.save(dest,garbage=4,deflate=True)


def main():
    manifest=[]; problems=[]; answers=[]
    for i,(year,sess) in enumerate(SESSIONS,1):
        print(f'[{i:02d}/40] {year}학년도 {sess}',flush=True)
        pp,purl,plabel,plogs=resolve(year,sess,'problem')
        ap,aurl,alabel,alogs=resolve(year,sess,'answer')
        row={'order':i,'year':year,'session':sess,'problem_url':purl,'problem_label':plabel,'answer_url':aurl,'answer_label':alabel,'problem_logs':plogs,'answer_logs':alogs}
        if pp and ap:
            row['status']='OK'; problems.append((year,sess,pp)); answers.append((year,sess,ap))
        else:
            row['status']='FAIL'
        manifest.append(row)
        (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    failed=[x for x in manifest if x['status']!='OK']
    if failed:
        print('FAILED SESSIONS:',[(x['year'],x['session'],bool(x['problem_url']),bool(x['answer_url'])) for x in failed])
        raise SystemExit(f'{len(failed)} sessions failed; refusing false complete file')
    if len(problems)!=40 or len(answers)!=40:
        raise SystemExit('40/40 gate failed')
    problem_pages=sum(len(fitz.open(p)) for _,_,p in problems)
    if problem_pages!=160:
        raise SystemExit(f'problem page gate failed: {problem_pages}')
    appendix=OUT/'빠른정답_부록.pdf'
    make_answer_appendix(answers,appendix)
    final=OUT/'KICE_사회문화_2014학년도-2027학년도6월_원문시험지_빠른정답_완성본.pdf'
    merge(problems,appendix,final)
    final_doc=fitz.open(final)
    report={
        'complete':True,'sessions':40,'problem_pages':160,
        'appendix_pages':len(fitz.open(appendix)),'final_pages':len(final_doc),
        'bytes':final.stat().st_size,'sha256':hashlib.sha256(final.read_bytes()).hexdigest(),
    }
    if report['final_pages'] != 160 + report['appendix_pages']:
        raise SystemExit('final page count mismatch')
    (OUT/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    main()
