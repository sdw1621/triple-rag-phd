"""
Gold QA Dataset Generator — 확장판
5,000개 질문-정답-참조근거 쌍 생성
60개 학과 / ~600명 교수 / ~1,500개 과목 / 400개 프로젝트
Simple 40% / Multi-hop 35% / Conditional 25%
"""
import json
import os
import random
from typing import Dict, List

# ── 학과 목록 (60개) ──────────────────────────────────────────────────
DEPARTMENTS = [
    # 공과대학 (15개)
    "컴퓨터공학과", "소프트웨어공학과", "전기공학과", "전자공학과", "기계공학과",
    "화학공학과", "토목공학과", "산업공학과", "건축공학과", "재료공학과",
    "항공우주공학과", "에너지공학과", "환경공학과", "생체공학과", "로봇공학과",
    # AI/IT (5개)
    "인공지능학과", "데이터사이언스학과", "정보보안학과", "사이버보안학과", "정보통신학과",
    # 자연과학 (7개)
    "수학과", "물리학과", "화학과", "생명과학과", "지구환경과학과", "통계학과", "천문학과",
    # 경영 (7개)
    "경영학과", "회계학과", "금융학과", "마케팅학과", "경제학과", "무역학과", "경영정보학과",
    # 인문 (7개)
    "국어국문학과", "영어영문학과", "사학과", "철학과", "심리학과", "중어중문학과", "일어일문학과",
    # 사회과학 (5개)
    "사회학과", "행정학과", "정치외교학과", "언론정보학과", "사회복지학과",
    # 의약 (5개)
    "의학과", "간호학과", "약학과", "보건학과", "바이오의공학과",
    # 예체능 (4개)
    "음악학과", "미술학과", "체육학과", "디자인학과",
    # 융합/특수 (5개)
    "법학과", "융합기술학과", "AI융합학과", "스마트시스템학과", "글로벌학과",
]

DEPT_CATEGORY = {
    "컴퓨터공학과": "공학", "소프트웨어공학과": "공학", "전기공학과": "공학",
    "전자공학과": "공학", "기계공학과": "공학", "화학공학과": "공학",
    "토목공학과": "공학", "산업공학과": "공학", "건축공학과": "공학",
    "재료공학과": "공학", "항공우주공학과": "공학", "에너지공학과": "공학",
    "환경공학과": "공학", "생체공학과": "공학", "로봇공학과": "공학",
    "인공지능학과": "AI", "데이터사이언스학과": "AI", "정보보안학과": "AI",
    "사이버보안학과": "AI", "정보통신학과": "AI",
    "수학과": "자연과학", "물리학과": "자연과학", "화학과": "자연과학",
    "생명과학과": "자연과학", "지구환경과학과": "자연과학", "통계학과": "자연과학", "천문학과": "자연과학",
    "경영학과": "경영", "회계학과": "경영", "금융학과": "경영",
    "마케팅학과": "경영", "경제학과": "경영", "무역학과": "경영", "경영정보학과": "경영",
    "국어국문학과": "인문사회", "영어영문학과": "인문사회", "사학과": "인문사회",
    "철학과": "인문사회", "심리학과": "인문사회", "중어중문학과": "인문사회",
    "일어일문학과": "인문사회", "사회학과": "인문사회", "행정학과": "인문사회",
    "정치외교학과": "인문사회", "언론정보학과": "인문사회", "사회복지학과": "인문사회",
    "법학과": "인문사회",
    "의학과": "의약", "간호학과": "의약", "약학과": "의약",
    "보건학과": "의약", "바이오의공학과": "의약",
    "음악학과": "예체능", "미술학과": "예체능", "체육학과": "예체능", "디자인학과": "예체능",
    "융합기술학과": "공학", "AI융합학과": "AI", "스마트시스템학과": "공학", "글로벌학과": "인문사회",
}

COURSE_TOPICS = {
    "공학": [
        "자료구조론", "알고리즘분석", "운영체제론", "컴퓨터네트워크", "데이터베이스시스템",
        "소프트웨어공학", "컴파일러이론", "임베디드시스템", "분산시스템", "클라우드컴퓨팅",
        "시스템프로그래밍", "컴퓨터구조론", "병렬처리", "네트워크보안", "웹프로그래밍",
        "모바일프로그래밍", "객체지향프로그래밍", "IoT시스템", "엣지컴퓨팅", "정보보안개론",
        "신호처리", "제어공학", "전력공학", "열역학응용", "유체역학응용",
        "재료역학", "구조해석", "CADCAM", "생산자동화", "품질공학",
    ],
    "AI": [
        "인공지능개론", "머신러닝", "딥러닝", "컴퓨터비전", "자연어처리",
        "강화학습", "생성AI", "설명가능AI", "페더레이티드러닝", "AutoML",
        "그래프신경망", "트랜스포머모델", "멀티모달AI", "AI윤리", "AI보안론",
        "지식그래프", "추론시스템", "지능형에이전트", "AI최적화", "데이터마이닝",
        "빅데이터처리", "통계적학습이론", "패턴인식", "음성인식", "추천시스템",
        "시계열분석", "이상감지", "자동화시스템", "AI플랫폼", "클라우드AI",
    ],
    "자연과학": [
        "미적분학1", "미적분학2", "선형대수학", "확률및통계", "현대물리학",
        "양자역학", "열역학", "전자기학", "광학", "유체역학",
        "유기화학", "무기화학", "물리화학", "분석화학", "생화학",
        "분자생물학", "유전학", "생태학", "지질학", "천문학개론",
        "수치해석", "위상수학", "대수학", "복소해석학", "편미분방정식",
        "응용통계학", "베이즈통계학", "실험설계론", "비모수통계", "수리통계학",
    ],
    "경영": [
        "경영학원론", "회계원리", "재무관리", "마케팅원론", "경제학원론",
        "조직행동론", "국제경영", "전략경영", "기업윤리", "창업론",
        "인사관리", "생산운영관리", "공급망관리", "빅데이터경영분석", "경영정보시스템",
        "금융공학", "계량경제학", "소비자행동론", "광고론", "무역실무",
        "재무회계", "관리회계", "세무회계", "재무제표분석", "글로벌마케팅",
        "디지털마케팅", "ESG경영", "플랫폼비즈니스", "스타트업경영", "MA전략",
    ],
    "인문사회": [
        "한국어문법론", "영어작문", "서양사개론", "동양사개론", "논리학",
        "인식론", "형이상학", "심리학개론", "사회학이론", "정치학원론",
        "언론이론", "미디어리터러시", "사회조사방법론", "복지정책론", "법학개론",
        "국제관계론", "행정학원론", "사회복지실천론", "범죄학", "도시사회학",
        "언어학개론", "문학비평론", "한국현대사", "비교정치론", "여론조사론",
        "문화인류학", "젠더연구", "지역사회복지", "공공정책론", "헌법학",
    ],
    "의약": [
        "해부학", "생리학", "약리학", "병리학", "임상실습개론",
        "간호학개론", "약제학", "보건통계학", "역학", "공중보건학",
        "의료윤리학", "임상약학", "의료기기공학", "재활의학개론", "의료AI활용",
        "정신건강론", "응급의학개론", "예방의학", "의료정보학", "디지털헬스케어",
        "기초의학", "임상의학", "약학개론", "보건행정학", "의약품개발론",
        "임상시험방법론", "건강증진론", "의학통계", "원격의료", "의료데이터분석",
    ],
    "예체능": [
        "음악이론", "화성학", "연주실기", "미술사", "회화기법",
        "조소기법", "스포츠과학", "체육교육론", "시각디자인", "UXUI디자인",
        "영상디자인", "제품디자인", "타이포그래피", "브랜드디자인", "인터랙션디자인",
        "운동생리학", "스포츠심리학", "체육측정평가", "디지털아트", "미디어아트",
        "작곡법", "지휘법", "국악개론", "현대미술론", "스포츠마케팅",
        "아트디렉션", "전시기획론", "스포츠경영", "애니메이션", "게임디자인",
    ],
}

RESEARCH_AREAS = [
    "머신러닝", "딥러닝", "자연어처리", "컴퓨터비전", "강화학습",
    "데이터마이닝", "정보검색", "소프트웨어공학", "분산시스템", "클라우드컴퓨팅",
    "사이버보안", "암호학", "네트워크", "임베디드시스템", "로봇공학",
    "생체인식", "의료AI", "금융공학", "회계정보", "경영과학",
    "통계학", "확률론", "최적화이론", "시뮬레이션", "환경공학",
    "에너지공학", "재료공학", "나노기술", "바이오공학", "인지과학",
    "언어학", "디지털미디어", "교육공학", "사회데이터분석", "양자컴퓨팅",
]

# 20 성 × 30 이름 = 600 고유 조합
LAST_NAMES  = ["김", "이", "박", "정", "최", "조", "한", "강", "윤", "장",
               "임", "오", "신", "홍", "권", "황", "안", "서", "유", "전"]
FIRST_NAMES = ["민준", "서준", "도윤", "예준", "시우", "주원", "하준", "지호", "준서", "준우",
               "서연", "서윤", "지우", "서현", "하은", "하윤", "민서", "지유", "윤서", "채원",
               "성민", "현우", "태양", "정훈", "재원", "승현", "민혁", "동현", "지훈", "성준"]

PROJECT_PREFIXES = [
    "AI", "스마트", "지능형", "차세대", "첨단", "융합", "디지털", "미래",
    "혁신", "빅데이터", "클라우드", "IoT", "블록체인", "메타버스", "양자",
    "바이오", "그린", "글로벌", "에듀", "시큐리티",
]
PROJECT_SUFFIXES = [
    "기술연구", "시스템개발", "플랫폼구축", "모델링연구", "최적화연구",
    "데이터분석", "솔루션개발", "알고리즘연구", "서비스혁신", "인프라구축",
    "응용연구", "기초연구", "융합연구", "실용화연구", "표준화연구",
    "성능평가", "보안강화", "자동화시스템", "품질향상", "지능화연구",
]


def generate_university_data(seed: int = 42) -> Dict:
    """
    대학 전체 엔티티 생성 (결정론적, seed 고정)
    Returns:
        depts        : 학과명 리스트 (60개)
        professors   : 교수 dict 리스트 (~600명)
        courses      : 과목 dict 리스트 (~1,500개)
        projects     : {프로젝트명: [참여교수명, ...]} (400개)
        dept_profs   : {학과: [교수명, ...]}
        dept_courses : {학과: [과목명, ...]}
        course_profs : {과목명: [교수명, ...]}
    """
    rng = random.Random(seed)

    # ── 교수 생성 (학과당 8~12명, 총 ~600명) ────────────────────────
    professors: List[Dict] = []
    prof_idx = 0
    for dept in DEPARTMENTS:
        count = rng.randint(8, 12)
        for _ in range(count):
            ln = LAST_NAMES[prof_idx % len(LAST_NAMES)]
            fn = FIRST_NAMES[prof_idx // len(LAST_NAMES)]
            cat = DEPT_CATEGORY.get(dept, "공학")
            professors.append({
                "id":       f"p{prof_idx}",
                "name":     ln + fn,
                "dept":     dept,
                "age":      rng.randint(34, 65),
                "research": ", ".join(rng.sample(RESEARCH_AREAS, 2)),
                "category": cat,
                "courses":  [],
                "collab":   [],
            })
            prof_idx += 1

    # ── 과목 생성 (학과당 20~30개, 총 ~1,500개) ──────────────────────
    courses: List[Dict] = []
    dept_courses: Dict[str, List[str]] = {}
    for dept in DEPARTMENTS:
        cat        = DEPT_CATEGORY.get(dept, "공학")
        topic_pool = COURSE_TOPICS.get(cat, COURSE_TOPICS["공학"])
        dept_short = dept.replace("학과", "").replace("학부", "")
        count      = rng.randint(20, 30)
        selected   = rng.sample(topic_pool, min(count, len(topic_pool)))
        dept_courses[dept] = []
        for topic in selected:
            cname = f"{dept_short} {topic}"
            courses.append({"name": cname, "dept": dept})
            dept_courses[dept].append(cname)

    # ── 교수-과목 배정 (교수당 3~6개) ────────────────────────────────
    for prof in professors:
        available    = dept_courses.get(prof["dept"], [])
        n            = min(rng.randint(3, 6), len(available))
        prof["courses"] = rng.sample(available, n) if available else []

    # course → profs 역방향 인덱스
    course_profs: Dict[str, List[str]] = {}
    for prof in professors:
        for c in prof["courses"]:
            course_profs.setdefault(c, []).append(prof["name"])

    # ── 협력 관계 (같은 학과 내 1~3명) ──────────────────────────────
    dept_prof_list: Dict[str, List[Dict]] = {}
    for prof in professors:
        dept_prof_list.setdefault(prof["dept"], []).append(prof)

    for prof in professors:
        mates = [p for p in dept_prof_list[prof["dept"]] if p["name"] != prof["name"]]
        if mates:
            n = min(rng.randint(1, 3), len(mates))
            prof["collab"] = [p["name"] for p in rng.sample(mates, n)]

    # ── 프로젝트 생성 (400개) ─────────────────────────────────────────
    projects: Dict[str, List[str]] = {}
    used_pnames = set()
    for i in range(400):
        pname = rng.choice(PROJECT_PREFIXES) + rng.choice(PROJECT_SUFFIXES)
        if pname in used_pnames:
            pname = f"{pname}{i}"
        used_pnames.add(pname)
        participants = rng.sample(professors, rng.randint(2, 4))
        projects[pname] = [p["name"] for p in participants]

    # dept → profs 인덱스
    dept_profs: Dict[str, List[str]] = {}
    for prof in professors:
        dept_profs.setdefault(prof["dept"], []).append(prof["name"])

    return {
        "depts":        DEPARTMENTS,
        "professors":   professors,
        "courses":      courses,
        "projects":     projects,
        "dept_profs":   dept_profs,
        "dept_courses": dept_courses,
        "course_profs": course_profs,
    }


# ── 사전 계산 인덱스 헬퍼 ──────────────────────────────────────────────

def _prof_by_name(professors: List[Dict]) -> Dict[str, Dict]:
    return {p["name"]: p for p in professors}


def _same_dept_pairs_with_shared_courses(professors, dept_prof_list):
    """같은 학과이면서 공동 담당 과목이 있는 교수 쌍 목록"""
    pairs = []
    for dept, profs in dept_prof_list.items():
        for i in range(len(profs)):
            for j in range(i + 1, len(profs)):
                p1, p2 = profs[i], profs[j]
                shared = list(set(p1["courses"]) & set(p2["courses"]))
                if shared:
                    pairs.append((p1, p2, shared))
    return pairs


# ══════════════════════════════════════════════════════════════════════
# QA 생성
# ══════════════════════════════════════════════════════════════════════

def build_gold_dataset(seed: int = 42, total: int = 5000) -> List[Dict]:
    """
    5,000개 Gold QA 쌍 생성
    Simple 40% / Multi-hop 35% / Conditional 25%
    """
    data   = generate_university_data(seed)
    rng    = random.Random(seed + 1)   # QA 생성용 별도 시드

    professors   = data["professors"]
    depts        = data["depts"]
    projects     = data["projects"]
    dept_profs   = data["dept_profs"]
    dept_courses = data["dept_courses"]
    course_profs = data["course_profs"]

    pbn = _prof_by_name(professors)  # name → prof dict

    dept_prof_list: Dict[str, List[Dict]] = {}
    for p in professors:
        dept_prof_list.setdefault(p["dept"], []).append(p)

    shared_pairs = _same_dept_pairs_with_shared_courses(professors, dept_prof_list)

    n_simple      = int(total * 0.40)  # 2,000
    n_multihop    = int(total * 0.35)  # 1,750
    n_conditional = total - n_simple - n_multihop  # 1,250

    dataset: List[Dict] = []
    idx = 1

    # ── Simple ──────────────────────────────────────────────────────
    simple_templates = [
        lambda p, c, d, pr: (
            f"{p['name']} 교수의 소속 학과는?",
            p["dept"],
            f"{p['name']} 교수 소속 정보",
        ),
        lambda p, c, d, pr: (
            f"{p['name']} 교수의 나이는?",
            f"{p['age']}세",
            f"{p['name']} 교수 기본 정보",
        ),
        lambda p, c, d, pr: (
            f"{p['name']} 교수가 담당하는 과목은?",
            ", ".join(p["courses"]) or "없음",
            f"{p['name']} 교수 담당 과목 정보",
        ),
        lambda p, c, d, pr: (
            f"{p['name']} 교수의 연구 분야는?",
            p["research"],
            f"{p['name']} 교수 연구 정보",
        ),
        lambda p, c, d, pr: (
            f"{d}에 소속된 교수는 몇 명인가?",
            f"{len(dept_profs.get(d, []))}명",
            f"{d} 학과 교수 현황",
        ),
        lambda p, c, d, pr: (
            f"{d} 소속 교수 목록은?",
            ", ".join(dept_profs.get(d, [])) or "없음",
            f"{d} 학과 교수 목록",
        ),
        lambda p, c, d, pr: (
            f"{c} 과목을 담당하는 교수는?",
            ", ".join(course_profs.get(c, [])) or "없음",
            f"{c} 과목 담당 교수 정보",
        ),
        lambda p, c, d, pr: (
            f"{pr} 프로젝트 참여 교수는?",
            ", ".join(projects.get(pr, [])) or "없음",
            f"{pr} 프로젝트 참여 현황",
        ),
    ]

    proj_list = list(projects.keys())

    while len([x for x in dataset if x["type"] == "simple"]) < n_simple:
        prof  = rng.choice(professors)
        dept  = rng.choice(depts)
        c_list = dept_courses.get(prof["dept"], [])
        course = rng.choice(c_list) if c_list else (dept_courses[depts[0]][0])
        proj  = rng.choice(proj_list)
        fn    = rng.choice(simple_templates)
        q, a, ref = fn(prof, course, dept, proj)
        dataset.append({"id": idx, "query": q, "answer": a,
                        "reference": ref, "type": "simple"})
        idx += 1

    # ── Multi-hop ────────────────────────────────────────────────────
    while len([x for x in dataset if x["type"] == "multi_hop"]) < n_multihop:
        choice = rng.randint(0, 3)

        if choice == 0 and shared_pairs:
            # 같은 학과 두 교수가 공동 담당하는 과목
            p1, p2, shared = rng.choice(shared_pairs)
            q   = f"{p1['name']} 교수와 {p2['name']} 교수가 공동 담당하는 과목은?"
            a   = ", ".join(shared)
            ref = f"{p1['name']}, {p2['name']} 담당 과목 교차 검색"

        elif choice == 1:
            # 교수 → 소속 학과 → 동일 학과 다른 교수
            prof  = rng.choice(professors)
            mates = [n for n in dept_profs.get(prof["dept"], []) if n != prof["name"]]
            q   = f"{prof['name']} 교수가 소속된 학과의 다른 교수는?"
            a   = ", ".join(mates) or "없음"
            ref = f"{prof['name']} 소속 학과 → 동일 학과 교수 목록"

        elif choice == 2:
            # 교수 → 협력 교수 → 협력 교수 담당 과목
            prof = rng.choice([p for p in professors if p["collab"]])
            collab_names  = prof["collab"]
            collab_courses = []
            for cn in collab_names:
                cp = pbn.get(cn)
                if cp:
                    collab_courses.extend(cp["courses"])
            q   = f"{prof['name']} 교수와 협력하는 교수의 담당 과목은?"
            a   = ", ".join(set(collab_courses)) or "없음"
            ref = f"{prof['name']} 협력 교수 → 해당 교수 담당 과목"

        else:
            # 프로젝트 → 참여 교수 → 소속 학과
            proj  = rng.choice(proj_list)
            profs = projects[proj]
            depts_set = list({pbn[n]["dept"] for n in profs if n in pbn})
            q   = f"{proj} 프로젝트 참여 교수들의 소속 학과는?"
            a   = ", ".join(depts_set) or "없음"
            ref = f"{proj} 프로젝트 참여 교수 → 소속 학과 추적"

        dataset.append({"id": idx, "query": q, "answer": a,
                        "reference": ref, "type": "multi_hop"})
        idx += 1

    # ── Conditional ──────────────────────────────────────────────────
    age_thresholds = [40, 45, 50, 55]

    while len([x for x in dataset if x["type"] == "conditional"]) < n_conditional:
        dept  = rng.choice(depts)
        age   = rng.choice(age_thresholds)
        choice = rng.randint(0, 3)

        if choice == 0:
            # 특정 학과 X세 이하 교수
            result = [p["name"] for p in dept_prof_list.get(dept, []) if p["age"] <= age]
            q   = f"{dept} 소속 {age}세 이하 교수는?"
            a   = ", ".join(result) or "없음"
            ref = f"{dept} 나이 조건 필터 (≤{age})"

        elif choice == 1:
            # 전체 X세 이상 교수 담당 과목
            result = list({c for p in professors if p["age"] >= age for c in p["courses"]})
            q   = f"{age}세 이상 교수가 담당하는 과목 목록은?"
            a   = ", ".join(result[:10]) + ("..." if len(result) > 10 else "") or "없음"
            ref = f"전체 교수 나이 조건 필터 (≥{age}) → 담당 과목"

        elif choice == 2:
            # X세 미만 전임 교수
            result = [p["name"] for p in professors if p["age"] < age]
            q   = f"전임 교수 중 {age}세 미만인 사람은?"
            a   = f"총 {len(result)}명 (" + ", ".join(result[:5]) + ("..." if len(result) > 5 else "") + ")"
            ref = f"전체 교수 나이 조건 필터 (<{age})"

        else:
            # 특정 학과 X세 이상 교수 연구 분야
            result = list({p["research"] for p in dept_prof_list.get(dept, []) if p["age"] >= age})
            q   = f"{dept} 소속 {age}세 이상 교수의 연구 분야는?"
            a   = ", ".join(result) or "없음"
            ref = f"{dept} 나이·연구 복합 조건 필터 (≥{age})"

        dataset.append({"id": idx, "query": q, "answer": a,
                        "reference": ref, "type": "conditional"})
        idx += 1

    rng.shuffle(dataset)
    for i, d in enumerate(dataset):
        d["id"] = i + 1
    return dataset


def save_dataset(path: str = "data/gold_qa_5000.json", total: int = 5000):
    """데이터셋 JSON 저장"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds     = build_gold_dataset(total=total)
    counts = {t: sum(1 for d in ds if d["type"] == t)
              for t in ["simple", "multi_hop", "conditional"]}

    with open(path, "w", encoding="utf-8") as f:
        json.dump(ds, f, ensure_ascii=False, indent=2)

    print(f"Gold QA 데이터셋 저장: {path}")
    print(f"   총 {len(ds)}개 | {counts}")
    return ds


if __name__ == "__main__":
    # 규모 확인
    data = generate_university_data()
    print(f"학과:   {len(data['depts'])}개")
    print(f"교수:   {len(data['professors'])}명")
    print(f"과목:   {len(data['courses'])}개")
    print(f"프로젝트: {len(data['projects'])}개")
    save_dataset()
