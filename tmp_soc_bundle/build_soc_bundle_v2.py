from pathlib import Path
import fitz
import build_soc_bundle as base

DIRECT = {
    (2019,'9월','problem'):[
        ('https://t1.daumcdn.net/cfile/tistory/99DF5C3F5BB335EF09','https://legendstudy.com/1353','2019학년도 9월 모의평가 사회탐구 - 사회문화 문제.pdf'),
        ('https://t1.daumcdn.net/cfile/tistory/994B7E385B91B1630E','https://sdfallyy.tistory.com/397','2019학년도 9월 모의평가 사회탐구 - 사회문화 문제.pdf'),
    ],
    (2025,'수능','problem'):[('https://horaeng.com/wp-content/uploads/2025%ED%95%99%EB%85%84%EB%8F%84-%EB%8C%80%ED%95%99%EC%88%98%ED%95%99%EB%8A%A5%EB%A0%A5%EC%8B%9C%ED%97%98-%EC%82%AC%ED%9A%8C%EB%AC%B8%ED%99%94-%EB%AC%B8%EC%A0%9C.pdf','https://horaeng.com/420','2025학년도 대학수학능력시험 사회문화 문제')],
    (2025,'수능','answer'):[('https://horaeng.com/wp-content/uploads/2025%ED%95%99%EB%85%84%EB%8F%84-%EB%8C%80%ED%95%99%EC%88%98%ED%95%99%EB%8A%A5%EB%A0%A5%EC%8B%9C%ED%97%98-%EC%82%AC%ED%9A%8C%EB%AC%B8%ED%99%94-%EC%A0%95%EB%8B%B5.pdf','https://horaeng.com/420','2025학년도 대학수학능력시험 사회문화 정답')],
    (2026,'6월','problem'):[('https://horaeng.com/wp-content/uploads/2026%ED%95%99%EB%85%84%EB%8F%84-6%EC%9B%94-%EB%AA%A8%EC%9D%98%ED%8F%89%EA%B0%80-%EC%82%AC%ED%9A%8C%EB%AC%B8%ED%99%94-%EB%AC%B8%EC%A0%9C.pdf','https://horaeng.com/438','2026학년도 6월 모의평가 사회문화 문제')],
    (2026,'6월','answer'):[('https://horaeng.com/wp-content/uploads/2026%ED%95%99%EB%85%84%EB%8F%84-6%EC%9B%94-%EB%AA%A8%EC%9D%98%ED%8F%89%EA%B0%80-%EC%82%AC%ED%9A%8C%EB%AC%B8%ED%99%94-%EC%A0%95%EB%8B%B5.pdf','https://horaeng.com/438','2026학년도 6월 모의평가 사회문화 정답')],
    (2026,'수능','problem'):[('https://horaeng.com/wp-content/uploads/2026%ED%95%99%EB%85%84%EB%8F%84-%EB%8C%80%ED%95%99%EC%88%98%ED%95%99%EB%8A%A5%EB%A0%A5%EC%8B%9C%ED%97%98-%EC%82%AC%ED%9A%8C%EB%AC%B8%ED%99%94-%EB%AC%B8%EC%A0%9C.pdf','https://horaeng.com/446','2026학년도 대학수학능력시험 사회문화 문제')],
    (2026,'수능','answer'):[('https://horaeng.com/wp-content/uploads/2026%ED%95%99%EB%85%84%EB%8F%84-%EB%8C%80%ED%95%99%EC%88%98%ED%95%99%EB%8A%A5%EB%A0%A5%EC%8B%9C%ED%97%98-%EC%82%AC%ED%9A%8C%EB%AC%B8%ED%99%94-%EC%A0%95%EB%8B%B5.pdf','https://horaeng.com/446','2026학년도 대학수학능력시험 사회문화 정답')],
}

orig_resolve = base.resolve

def _normalize_combined_problem(raw: Path, dest: Path, year: int, sess: str, label: str):
    d=fitz.open(raw)
    if len(d) != 5:
        return None, {'reason':f'not_5_pages:{len(d)}'}
    last=base.clean(d[4].get_text('text'))
    # A known KICE/EBS mirror format packages the four original problem pages + one answer page.
    # Only accept it when the fifth page is answer-like; never silently trim an arbitrary 5-page file.
    if not any(tok in last for tok in ('정답','답','문항')):
        return None, {'reason':'fifth_page_not_answer_like','fifth_head':d[4].get_text('text')[:400]}
    out=fitz.open(); out.insert_pdf(d,from_page=0,to_page=3); out.save(dest,garbage=4,deflate=True)
    ok,info=base.validate_problem(dest,year,sess,label)
    if not ok:
        dest.unlink(missing_ok=True)
        return None, {'reason':'trimmed_validation_failed','validation':info}
    return dest, {'reason':'accepted_four_original_pages_from_4plus1_package','validation':info,'fifth_head':d[4].get_text('text')[:400]}

def patched_resolve(year:int,sess:str,kind:str):
    logs=[]
    for idx,(url,referer,label) in enumerate(DIRECT.get((year,sess,kind),[])):
        raw=base.SRC/f'direct_{year}_{sess}_{kind}_{idx}_raw.pdf'
        try:
            rr=base.fetch(url,referer=referer,binary=True)
            raw.write_bytes(rr.content)
            if kind=='answer':
                ok,info=base.validate_answer(raw)
                logs.append({'direct':url,'final_url':rr.url,'label':label,'validation':info,'ok':ok})
                if ok: return raw,rr.url,label,logs
            else:
                ok,info=base.validate_problem(raw,year,sess,label)
                logs.append({'direct':url,'final_url':rr.url,'label':label,'validation':info,'ok':ok})
                if ok: return raw,rr.url,label,logs
                # Special verified 2019.9 mirror package: 4 original exam pages + answer page.
                if year==2019 and sess=='9월' and info.get('pages')==5:
                    norm=base.SRC/f'direct_{year}_{sess}_{kind}_{idx}_normalized4.pdf'
                    got,ninfo=_normalize_combined_problem(raw,norm,year,sess,label)
                    logs.append({'normalization':ninfo})
                    if got: return got,rr.url+'#pages=1-4',label,logs
            raw.unlink(missing_ok=True)
        except Exception as e:
            logs.append({'direct':url,'label':label,'error':repr(e)})
            raw.unlink(missing_ok=True)
    p,u,l,more=orig_resolve(year,sess,kind)
    return p,u,l,logs+more

base.resolve=patched_resolve
base.main()
