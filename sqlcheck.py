def sqlcheck(sql_query, *parms):
    # 불필요한 공백 제거
    sql_query = sql_query.strip()

    # 쿼리문에 semicolon이 있는 경우(쿼리문이 여러개)
    if ";" in sql_query:
        return False
    
    # 쿼리문에 단어 검색
    # sql injection 관련 단어 목록
    sql_injection_keywords = ['DROP', 'TRUNCATE', 'DELETE', 'ALTER', 'UPDATE', 'UNION']
    for keyword in sql_injection_keywords:
        if keyword in sql_query.upper():
            return False

    # 파라미터에 단어 검색
    for parm in parms:
        for keyword in sql_injection_keywords:
            if keyword in str(parm).upper():
                return False
    return True
