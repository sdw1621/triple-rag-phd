"""
university_data.py
확장된 대학 행정 마스터 데이터
논문 데이터셋 규모 대폭 확장:
  교수 30명 / 학과 8개 / 과목 40개 / 프로젝트 15개
  문서 200건 / 그래프 노드 150+ / 골드QA 1000개
"""

# ── 학과 (8개) ─────────────────────────────────────────────
DEPARTMENTS = [
    {"id":"d1",  "name":"컴퓨터공학과",       "code":"CS",   "college":"공과대학"},
    {"id":"d2",  "name":"인공지능학과",        "code":"AI",   "college":"공과대학"},
    {"id":"d3",  "name":"소프트웨어학과",      "code":"SW",   "college":"공과대학"},
    {"id":"d4",  "name":"데이터사이언스학과",   "code":"DS",   "college":"이과대학"},
    {"id":"d5",  "name":"정보보안학과",        "code":"SEC",  "college":"공과대학"},
    {"id":"d6",  "name":"전자공학과",          "code":"EE",   "college":"공과대학"},
    {"id":"d7",  "name":"수학과",             "code":"MATH", "college":"이과대학"},
    {"id":"d8",  "name":"통계학과",            "code":"STAT", "college":"이과대학"},
]

# ── 교수 (30명) ────────────────────────────────────────────
PROFESSORS = [
    # 컴퓨터공학과 (6명)
    {"id":"p01","name":"김철수","dept":"컴퓨터공학과","age":45,"rank":"정교수",
     "research":["머신러닝","자연어처리"],"email":"kimcs@univ.ac.kr"},
    {"id":"p02","name":"박민수","dept":"컴퓨터공학과","age":52,"rank":"정교수",
     "research":["자연어처리","정보검색"],"email":"parkms@univ.ac.kr"},
    {"id":"p03","name":"최준혁","dept":"컴퓨터공학과","age":41,"rank":"부교수",
     "research":["알고리즘","그래프이론"],"email":"choijh@univ.ac.kr"},
    {"id":"p04","name":"한상우","dept":"컴퓨터공학과","age":38,"rank":"조교수",
     "research":["분산시스템","클라우드컴퓨팅"],"email":"hansw@univ.ac.kr"},
    {"id":"p05","name":"윤지현","dept":"컴퓨터공학과","age":35,"rank":"조교수",
     "research":["임베디드시스템","IoT"],"email":"yoonjh@univ.ac.kr"},
    {"id":"p06","name":"임태양","dept":"컴퓨터공학과","age":48,"rank":"정교수",
     "research":["데이터베이스","빅데이터"],"email":"limty@univ.ac.kr"},

    # 인공지능학과 (6명)
    {"id":"p07","name":"이영희","dept":"인공지능학과","age":38,"rank":"부교수",
     "research":["딥러닝","컴퓨터비전"],"email":"leeyh@univ.ac.kr"},
    {"id":"p08","name":"정수진","dept":"인공지능학과","age":36,"rank":"조교수",
     "research":["딥러닝","컴퓨터비전"],"email":"jungsj@univ.ac.kr"},
    {"id":"p09","name":"강현도","dept":"인공지능학과","age":44,"rank":"정교수",
     "research":["강화학습","로보틱스"],"email":"kanghd@univ.ac.kr"},
    {"id":"p10","name":"서민준","dept":"인공지능학과","age":40,"rank":"부교수",
     "research":["생성모델","GAN"],"email":"seomj@univ.ac.kr"},
    {"id":"p11","name":"오지은","dept":"인공지능학과","age":33,"rank":"조교수",
     "research":["설명가능AI","XAI"],"email":"ohje@univ.ac.kr"},
    {"id":"p12","name":"남기훈","dept":"인공지능학과","age":50,"rank":"정교수",
     "research":["지식그래프","온톨로지"],"email":"namkh@univ.ac.kr"},

    # 소프트웨어학과 (4명)
    {"id":"p13","name":"장미래","dept":"소프트웨어학과","age":39,"rank":"부교수",
     "research":["소프트웨어공학","애자일"],"email":"jangmr@univ.ac.kr"},
    {"id":"p14","name":"권오준","dept":"소프트웨어학과","age":46,"rank":"정교수",
     "research":["프로그래밍언어","컴파일러"],"email":"kwonOJ@univ.ac.kr"},
    {"id":"p15","name":"백승현","dept":"소프트웨어학과","age":37,"rank":"조교수",
     "research":["모바일컴퓨팅","UX"],"email":"baeksh@univ.ac.kr"},
    {"id":"p16","name":"류성민","dept":"소프트웨어학과","age":43,"rank":"부교수",
     "research":["클라우드네이티브","DevOps"],"email":"ryusm@univ.ac.kr"},

    # 데이터사이언스학과 (4명)
    {"id":"p17","name":"전혜린","dept":"데이터사이언스학과","age":41,"rank":"부교수",
     "research":["데이터마이닝","패턴인식"],"email":"jeonhr@univ.ac.kr"},
    {"id":"p18","name":"신동현","dept":"데이터사이언스학과","age":35,"rank":"조교수",
     "research":["시계열분석","예측모델"],"email":"shindh@univ.ac.kr"},
    {"id":"p19","name":"홍석진","dept":"데이터사이언스학과","age":49,"rank":"정교수",
     "research":["빅데이터처리","하둡"],"email":"hongsj@univ.ac.kr"},
    {"id":"p20","name":"문지혜","dept":"데이터사이언스학과","age":34,"rank":"조교수",
     "research":["시각화","인포그래픽"],"email":"moonjh@univ.ac.kr"},

    # 정보보안학과 (3명)
    {"id":"p21","name":"조성환","dept":"정보보안학과","age":47,"rank":"정교수",
     "research":["암호학","네트워크보안"],"email":"joshh@univ.ac.kr"},
    {"id":"p22","name":"김보람","dept":"정보보안학과","age":36,"rank":"조교수",
     "research":["사이버포렌식","취약점분석"],"email":"kimbr@univ.ac.kr"},
    {"id":"p23","name":"이재원","dept":"정보보안학과","age":42,"rank":"부교수",
     "research":["블록체인","분산보안"],"email":"leejw@univ.ac.kr"},

    # 전자공학과 (3명)
    {"id":"p24","name":"박태준","dept":"전자공학과","age":53,"rank":"정교수",
     "research":["반도체","VLSI"],"email":"parktj@univ.ac.kr"},
    {"id":"p25","name":"손민아","dept":"전자공학과","age":40,"rank":"부교수",
     "research":["신호처리","통신시스템"],"email":"sonma@univ.ac.kr"},
    {"id":"p26","name":"안준호","dept":"전자공학과","age":37,"rank":"조교수",
     "research":["임베디드","FPGA"],"email":"anjh@univ.ac.kr"},

    # 수학과 (2명)
    {"id":"p27","name":"정기범","dept":"수학과","age":51,"rank":"정교수",
     "research":["최적화이론","볼록최적화"],"email":"jungkb@univ.ac.kr"},
    {"id":"p28","name":"노현아","dept":"수학과","age":44,"rank":"부교수",
     "research":["확률론","통계적학습이론"],"email":"noha@univ.ac.kr"},

    # 통계학과 (2명)
    {"id":"p29","name":"하윤석","dept":"통계학과","age":48,"rank":"정교수",
     "research":["베이지안통계","머신러닝"],"email":"hays@univ.ac.kr"},
    {"id":"p30","name":"송나연","dept":"통계학과","age":39,"rank":"부교수",
     "research":["생존분석","임상통계"],"email":"songny@univ.ac.kr"},
]

# ── 과목 (40개) ────────────────────────────────────────────
COURSES = [
    # CS
    {"id":"c01","name":"인공지능개론",       "dept":"컴퓨터공학과","credits":3,"profs":["p01"]},
    {"id":"c02","name":"딥러닝",             "dept":"컴퓨터공학과","credits":3,"profs":["p01","p07"]},
    {"id":"c03","name":"자연어처리",         "dept":"컴퓨터공학과","credits":3,"profs":["p02"]},
    {"id":"c04","name":"알고리즘",           "dept":"컴퓨터공학과","credits":3,"profs":["p03"]},
    {"id":"c05","name":"강화학습",           "dept":"컴퓨터공학과","credits":3,"profs":["p01","p09"]},
    {"id":"c06","name":"분산시스템",         "dept":"컴퓨터공학과","credits":3,"profs":["p04"]},
    {"id":"c07","name":"클라우드컴퓨팅",     "dept":"컴퓨터공학과","credits":3,"profs":["p04","p16"]},
    {"id":"c08","name":"데이터베이스",       "dept":"컴퓨터공학과","credits":3,"profs":["p06"]},
    {"id":"c09","name":"운영체제",           "dept":"컴퓨터공학과","credits":3,"profs":["p03"]},
    {"id":"c10","name":"컴퓨터네트워크",     "dept":"컴퓨터공학과","credits":3,"profs":["p04"]},
    # AI
    {"id":"c11","name":"컴퓨터비전",         "dept":"인공지능학과","credits":3,"profs":["p07","p08"]},
    {"id":"c12","name":"생성모델",           "dept":"인공지능학과","credits":3,"profs":["p10"]},
    {"id":"c13","name":"설명가능AI",         "dept":"인공지능학과","credits":3,"profs":["p11"]},
    {"id":"c14","name":"지식그래프",         "dept":"인공지능학과","credits":3,"profs":["p12"]},
    {"id":"c15","name":"로봇공학",           "dept":"인공지능학과","credits":3,"profs":["p09"]},
    {"id":"c16","name":"자연어이해",         "dept":"인공지능학과","credits":3,"profs":["p01","p02"]},
    {"id":"c17","name":"멀티모달AI",         "dept":"인공지능학과","credits":3,"profs":["p10","p11"]},
    {"id":"c18","name":"AI윤리",             "dept":"인공지능학과","credits":2,"profs":["p11","p12"]},
    # SW
    {"id":"c19","name":"소프트웨어공학",     "dept":"소프트웨어학과","credits":3,"profs":["p13"]},
    {"id":"c20","name":"프로그래밍언어론",   "dept":"소프트웨어학과","credits":3,"profs":["p14"]},
    {"id":"c21","name":"모바일프로그래밍",   "dept":"소프트웨어학과","credits":3,"profs":["p15"]},
    {"id":"c22","name":"DevOps실습",         "dept":"소프트웨어학과","credits":3,"profs":["p16"]},
    # DS
    {"id":"c23","name":"데이터마이닝",       "dept":"데이터사이언스학과","credits":3,"profs":["p17"]},
    {"id":"c24","name":"시계열분석",         "dept":"데이터사이언스학과","credits":3,"profs":["p18"]},
    {"id":"c25","name":"빅데이터처리",       "dept":"데이터사이언스학과","credits":3,"profs":["p19"]},
    {"id":"c26","name":"데이터시각화",       "dept":"데이터사이언스학과","credits":3,"profs":["p20"]},
    {"id":"c27","name":"통계적학습",         "dept":"데이터사이언스학과","credits":3,"profs":["p17","p29"]},
    # SEC
    {"id":"c28","name":"암호학",             "dept":"정보보안학과","credits":3,"profs":["p21"]},
    {"id":"c29","name":"네트워크보안",       "dept":"정보보안학과","credits":3,"profs":["p21","p22"]},
    {"id":"c30","name":"블록체인",           "dept":"정보보안학과","credits":3,"profs":["p23"]},
    # EE
    {"id":"c31","name":"신호처리",           "dept":"전자공학과","credits":3,"profs":["p25"]},
    {"id":"c32","name":"VLSI설계",           "dept":"전자공학과","credits":3,"profs":["p24"]},
    {"id":"c33","name":"임베디드시스템",     "dept":"전자공학과","credits":3,"profs":["p05","p26"]},
    # MATH
    {"id":"c34","name":"최적화이론",         "dept":"수학과","credits":3,"profs":["p27"]},
    {"id":"c35","name":"확률론",             "dept":"수학과","credits":3,"profs":["p28"]},
    # STAT
    {"id":"c36","name":"베이지안통계",       "dept":"통계학과","credits":3,"profs":["p29"]},
    {"id":"c37","name":"생존분석",           "dept":"통계학과","credits":3,"profs":["p30"]},
    # 공통
    {"id":"c38","name":"머신러닝기초",       "dept":"데이터사이언스학과","credits":3,"profs":["p01","p17","p29"]},
    {"id":"c39","name":"수치해석",           "dept":"수학과","credits":3,"profs":["p27","p28"]},
    {"id":"c40","name":"선형대수학",         "dept":"수학과","credits":3,"profs":["p27"]},
]

# ── 프로젝트 (15개) ────────────────────────────────────────
PROJECTS = [
    {"id":"pr01","name":"AI융합연구소",      "profs":["p01","p07","p09"],"field":"AI"},
    {"id":"pr02","name":"NLP연구센터",       "profs":["p02","p01","p12"],"field":"NLP"},
    {"id":"pr03","name":"자율주행AI프로젝트","profs":["p07","p08","p09"],"field":"Vision/RL"},
    {"id":"pr04","name":"스마트팩토리연구",  "profs":["p04","p26","p25"],"field":"IoT"},
    {"id":"pr05","name":"의료AI연구단",      "profs":["p10","p11","p30"],"field":"Healthcare"},
    {"id":"pr06","name":"블록체인보안연구",  "profs":["p21","p23","p22"],"field":"Security"},
    {"id":"pr07","name":"빅데이터플랫폼",   "profs":["p06","p19","p17"],"field":"BigData"},
    {"id":"pr08","name":"지능형로봇연구",   "profs":["p09","p26","p25"],"field":"Robotics"},
    {"id":"pr09","name":"메타버스연구소",   "profs":["p15","p10","p20"],"field":"Metaverse"},
    {"id":"pr10","name":"양자컴퓨팅연구",  "profs":["p27","p24","p21"],"field":"Quantum"},
    {"id":"pr11","name":"클라우드인프라",   "profs":["p04","p16","p06"],"field":"Cloud"},
    {"id":"pr12","name":"데이터거버넌스",   "profs":["p17","p29","p30"],"field":"Governance"},
    {"id":"pr13","name":"사이버보안센터",   "profs":["p21","p22","p23"],"field":"Cybersecurity"},
    {"id":"pr14","name":"지식그래프플랫폼","profs":["p12","p01","p17"],"field":"KG"},
    {"id":"pr15","name":"그린AI연구소",     "profs":["p11","p28","p20"],"field":"GreenAI"},
]

# ── 교수 간 협력 관계 ─────────────────────────────────────
COLLABORATIONS = [
    ("p01","p02"),("p01","p07"),("p01","p09"),("p01","p12"),
    ("p02","p03"),("p07","p08"),("p07","p10"),("p09","p08"),
    ("p04","p16"),("p06","p19"),("p10","p11"),("p11","p12"),
    ("p17","p29"),("p19","p17"),("p21","p22"),("p21","p23"),
    ("p24","p25"),("p25","p26"),("p27","p28"),("p29","p30"),
    ("p03","p14"),("p13","p16"),("p15","p10"),("p18","p20"),
]

def get_prof_by_id(pid):
    return next((p for p in PROFESSORS if p["id"]==pid), None)

def get_dept_profs(dept_name):
    return [p for p in PROFESSORS if p["dept"]==dept_name]

def get_course_profs(course_id):
    c = next((c for c in COURSES if c["id"]==course_id), None)
    if not c: return []
    return [get_prof_by_id(pid) for pid in c["profs"] if get_prof_by_id(pid)]

def get_prof_courses(prof_id):
    return [c for c in COURSES if prof_id in c["profs"]]

def get_proj_profs(proj_id):
    pr = next((p for p in PROJECTS if p["id"]==proj_id), None)
    if not pr: return []
    return [get_prof_by_id(pid) for pid in pr["profs"] if get_prof_by_id(pid)]
