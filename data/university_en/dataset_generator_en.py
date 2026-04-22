"""
English synthetic university dataset — minimal viable dataset for L-DWA
cross-lingual transfer experiments (Option B of the 2026-04-22 cross-domain
analysis, Ch.6 §7).

Scale: 50 departments, 100 professors, 200 courses, 100 projects.
Output: documents + knowledge graph + gold_qa_500.json with the same
schema as data/university/gold_qa_5000.json but English content.

Run (from repo root, inside docker container):
    python data/university_en/dataset_generator_en.py

Produces in this folder:
    university_data_en.py    (department / professor / course / project tables)
    gold_qa_500_en.json      (500 QA pairs, simple 40% / multi 35% / cond 25%)
    corpus_en.json           (document corpus for VectorStore)
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

random.seed(42)  # reproducibility

ROOT = Path(__file__).resolve().parent

# ---- Universe ----
DEPARTMENTS = [
    # Engineering (15)
    "Computer Engineering", "Software Engineering", "Electrical Engineering",
    "Mechanical Engineering", "Chemical Engineering", "Civil Engineering",
    "Industrial Engineering", "Materials Engineering", "Aerospace Engineering",
    "Energy Engineering", "Environmental Engineering", "Biomedical Engineering",
    "Robotics Engineering", "Naval Engineering", "Urban Engineering",
    # AI / IT (6)
    "Artificial Intelligence", "Data Science", "Information Security",
    "Cybersecurity", "Information Communication", "Machine Learning",
    # Natural Science (6)
    "Mathematics", "Physics", "Chemistry", "Life Science", "Statistics", "Astronomy",
    # Business (6)
    "Business Administration", "Accounting", "Finance", "Marketing", "Economics", "Trade",
    # Humanities (6)
    "Korean Literature", "English Literature", "History", "Philosophy",
    "Psychology", "Sociology",
    # Medical (5)
    "Medicine", "Nursing", "Pharmacy", "Public Health", "Dentistry",
    # Arts (3)
    "Music", "Fine Arts", "Design",
    # Law & Policy (3)
    "Law", "Public Administration", "International Relations",
]
assert len(DEPARTMENTS) == 50, f"expected 50, got {len(DEPARTMENTS)}"

DEPT_CATEGORY = {d: ("Engineering" if any(k in d for k in ("Engineering", "Computer", "Cyber"))
                     else "AI" if any(k in d for k in ("Artificial", "Data", "Information"))
                     else "Science" if d in ("Mathematics", "Physics", "Chemistry", "Life Science", "Statistics", "Astronomy")
                     else "Business" if d in ("Business Administration", "Accounting", "Finance", "Marketing", "Economics", "Trade")
                     else "Humanities" if d in ("Korean Literature", "English Literature", "History", "Philosophy", "Psychology", "Sociology")
                     else "Medical" if d in ("Medicine", "Nursing", "Pharmacy", "Public Health")
                     else "Arts" if d in ("Music", "Fine Arts", "Design")
                     else "Law")
                  for d in DEPARTMENTS}

FIRST_NAMES = [
    "John", "Jane", "Michael", "Sarah", "David", "Emily", "Robert", "Jessica",
    "James", "Linda", "William", "Susan", "Richard", "Karen", "Thomas", "Nancy",
    "Charles", "Mary", "Daniel", "Patricia", "Matthew", "Ashley", "Anthony",
    "Maria", "Mark", "Angela", "Steven", "Helen", "Paul", "Rachel", "Andrew",
    "Donna", "Joshua", "Lisa", "Kenneth", "Sandra", "Kevin", "Ruth",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Taylor", "Thomas", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Clark", "Lewis", "Walker",
    "Young", "Allen", "King", "Wright", "Scott", "Torres", "Hill", "Green",
]

RESEARCH_FIELDS = {
    "Engineering": ["Robotics", "Control Systems", "Mechatronics", "Manufacturing"],
    "AI": ["Machine Learning", "NLP", "Computer Vision", "Reinforcement Learning",
           "Knowledge Graphs", "Speech Processing", "Information Retrieval"],
    "Science": ["Quantum Physics", "Thermodynamics", "Organic Chemistry",
                "Genetics", "Bayesian Inference", "Astronomy"],
    "Business": ["Corporate Finance", "Consumer Behavior", "Supply Chain",
                 "Game Theory", "Econometrics"],
    "Humanities": ["Linguistics", "Medieval History", "Ethics", "Cognitive Psychology",
                   "Social Theory"],
    "Medical": ["Oncology", "Cardiology", "Neuroscience", "Epidemiology",
                "Pharmacokinetics"],
    "Arts": ["Contemporary Art", "Musicology", "Visual Design", "Digital Media"],
    "Law": ["Constitutional Law", "International Law", "Criminal Law",
            "Administrative Policy"],
}


def gen_professor(idx: int, dept: str) -> dict:
    category = DEPT_CATEGORY[dept]
    return {
        "id": f"prof_{idx:04d}",
        "name": f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
        "dept": dept,
        "age": random.randint(32, 68),
        "title": random.choice(["Professor", "Associate Professor", "Assistant Professor"]),
        "research": random.choice(RESEARCH_FIELDS[category]),
    }


def gen_course(idx: int, dept: str, profs_in_dept: list[dict]) -> dict:
    category = DEPT_CATEGORY[dept]
    topic = random.choice(RESEARCH_FIELDS[category])
    teacher_ids = random.sample([p["id"] for p in profs_in_dept],
                                min(random.randint(1, 3), len(profs_in_dept)))
    return {
        "id": f"course_{idx:04d}",
        "title": f"{topic} {random.choice(['I', 'II', 'III', 'Introduction', 'Advanced', 'Seminar'])}",
        "dept": dept,
        "teachers": teacher_ids,
        "level": random.choice(["Undergraduate", "Graduate"]),
    }


def gen_project(idx: int, depts: list[str], all_profs: list[dict]) -> dict:
    lead_dept = random.choice(depts)
    collab_depts = random.sample([d for d in depts if d != lead_dept],
                                  random.randint(0, 2))
    profs_in_lead = [p["id"] for p in all_profs if p["dept"] == lead_dept]
    lead = random.choice(profs_in_lead) if profs_in_lead else all_profs[0]["id"]
    return {
        "id": f"project_{idx:04d}",
        "title": f"{random.choice(['AI', 'Green', 'Smart', 'Next-Gen', 'Quantum', 'Bio'])} "
                 f"{random.choice(['Energy', 'Health', 'Learning', 'Security', 'Mobility', 'Research'])} Project",
        "lead_dept": lead_dept,
        "collab_depts": collab_depts,
        "lead_prof": lead,
    }


def generate_corpus() -> dict:
    professors: list[dict] = []
    # Distribute ~100 professors across 50 departments (2 each + bonus)
    prof_idx = 0
    for dept in DEPARTMENTS:
        n = 2 + random.randint(0, 1)
        for _ in range(n):
            professors.append(gen_professor(prof_idx, dept))
            prof_idx += 1
    # Trim to 100
    professors = professors[:100]

    courses: list[dict] = []
    course_idx = 0
    for dept in DEPARTMENTS:
        profs_dept = [p for p in professors if p["dept"] == dept]
        if not profs_dept:
            continue
        n = 3 + random.randint(0, 2)
        for _ in range(n):
            courses.append(gen_course(course_idx, dept, profs_dept))
            course_idx += 1
    courses = courses[:200]

    projects = [gen_project(i, DEPARTMENTS, professors) for i in range(100)]

    return {
        "departments": DEPARTMENTS,
        "professors": professors,
        "courses": courses,
        "projects": projects,
    }


def prof_ids_to_names(ids: list[str], profs: list[dict]) -> list[str]:
    m = {p["id"]: p["name"] for p in profs}
    return [m[i] for i in ids if i in m]


def gen_simple_qa(corpus: dict) -> list[dict]:
    profs = corpus["professors"]
    courses = corpus["courses"]
    qas: list[dict] = []

    # Type A: "Which department does Prof X belong to?"
    for p in random.sample(profs, min(60, len(profs))):
        qas.append({
            "id": len(qas) + 1,
            "query": f"Which department does Professor {p['name']} belong to?",
            "answer": p["dept"],
            "reference": f"Profile of Professor {p['name']}",
            "type": "simple",
        })

    # Type B: "Who teaches course X?"
    for c in random.sample(courses, min(60, len(courses))):
        names = prof_ids_to_names(c["teachers"], profs)
        if not names:
            continue
        qas.append({
            "id": len(qas) + 1,
            "query": f"Who teaches {c['title']}?",
            "answer": ", ".join(names),
            "reference": f"Course record for {c['title']}",
            "type": "simple",
        })

    # Type C: "How old is Prof X?"
    for p in random.sample(profs, min(40, len(profs))):
        qas.append({
            "id": len(qas) + 1,
            "query": f"How old is Professor {p['name']}?",
            "answer": f"{p['age']} years old",
            "reference": f"Profile of Professor {p['name']}",
            "type": "simple",
        })

    # Type D: "What is Prof X's research area?"
    for p in random.sample(profs, min(40, len(profs))):
        qas.append({
            "id": len(qas) + 1,
            "query": f"What is Professor {p['name']}'s research area?",
            "answer": p["research"],
            "reference": f"Profile of Professor {p['name']}",
            "type": "simple",
        })

    return qas[:200]  # 40%


def gen_multi_hop_qa(corpus: dict) -> list[dict]:
    profs = corpus["professors"]
    courses = corpus["courses"]
    projects = corpus["projects"]
    qas: list[dict] = []

    # Type: "Which departments participate in project X?"
    for pr in random.sample(projects, min(60, len(projects))):
        all_depts = [pr["lead_dept"]] + pr["collab_depts"]
        if len(all_depts) < 2:
            continue
        qas.append({
            "id": 1000 + len(qas),
            "query": f"Which departments participate in the {pr['title']}?",
            "answer": ", ".join(all_depts),
            "reference": f"Project record: {pr['title']}",
            "type": "multi_hop",
        })

    # Type: "What courses does department X offer?"
    for d in random.sample(DEPARTMENTS, min(40, len(DEPARTMENTS))):
        dept_courses = [c["title"] for c in courses if c["dept"] == d]
        if len(dept_courses) < 2:
            continue
        qas.append({
            "id": 1000 + len(qas),
            "query": f"What courses does the {d} Department offer?",
            "answer": ", ".join(dept_courses[:5]),
            "reference": f"{d} Department catalog",
            "type": "multi_hop",
        })

    # Type: "Which professors teach course X together with Prof Y?"
    for c in random.sample(courses, min(50, len(courses))):
        names = prof_ids_to_names(c["teachers"], profs)
        if len(names) < 2:
            continue
        anchor = random.choice(names)
        co = [n for n in names if n != anchor]
        qas.append({
            "id": 1000 + len(qas),
            "query": f"Who co-teaches {c['title']} with Professor {anchor}?",
            "answer": ", ".join(co),
            "reference": f"Course record: {c['title']}",
            "type": "multi_hop",
        })

    return qas[:175]  # 35%


def gen_conditional_qa(corpus: dict) -> list[dict]:
    profs = corpus["professors"]
    qas: list[dict] = []

    # Type: "List professors in department X younger than Y"
    for d in random.sample(DEPARTMENTS, min(50, len(DEPARTMENTS))):
        threshold = random.choice([45, 50, 55])
        matches = [p for p in profs if p["dept"] == d and p["age"] < threshold]
        if not matches:
            continue
        qas.append({
            "id": 5000 + len(qas),
            "query": f"List professors in the {d} Department younger than {threshold}.",
            "answer": ", ".join(p["name"] for p in matches),
            "reference": f"{d} Department faculty filter (age < {threshold})",
            "type": "conditional",
        })

    # Type: "Which professors in department X have research area in Y?"
    for d in random.sample(DEPARTMENTS, min(50, len(DEPARTMENTS))):
        dept_profs = [p for p in profs if p["dept"] == d]
        if len(dept_profs) < 2:
            continue
        target_field = dept_profs[0]["research"]
        matches = [p for p in dept_profs if p["research"] == target_field]
        if len(matches) < 2:
            continue
        qas.append({
            "id": 5000 + len(qas),
            "query": f"Which professors in the {d} Department work on {target_field}?",
            "answer": ", ".join(p["name"] for p in matches),
            "reference": f"{d} Department research filter",
            "type": "conditional",
        })

    # Type: "Professors older than X with title Y"
    for _ in range(40):
        threshold = random.choice([50, 55, 60])
        title = random.choice(["Professor", "Associate Professor", "Assistant Professor"])
        matches = [p for p in profs if p["age"] > threshold and p["title"] == title]
        if not matches:
            continue
        qas.append({
            "id": 5000 + len(qas),
            "query": f"Which {title}s are older than {threshold}?",
            "answer": ", ".join(p["name"] for p in matches[:10]),
            "reference": f"Faculty filter: age > {threshold}, title = {title}",
            "type": "conditional",
        })

    return qas[:125]  # 25%


def build_documents(corpus: dict) -> list[str]:
    """Document corpus for VectorStore — descriptive English paragraphs."""
    docs: list[str] = []
    for p in corpus["professors"]:
        docs.append(
            f"Professor {p['name']} is affiliated with the {p['dept']} Department. "
            f"They are {p['age']} years old and hold the title of {p['title']}. "
            f"Their research area is {p['research']}."
        )
    for c in corpus["courses"]:
        teacher_names = prof_ids_to_names(c["teachers"], corpus["professors"])
        docs.append(
            f"The {c['title']} course is offered by the {c['dept']} Department as a "
            f"{c['level']} course. It is taught by {', '.join(teacher_names)}."
        )
    for pr in corpus["projects"]:
        all_depts = [pr["lead_dept"]] + pr["collab_depts"]
        docs.append(
            f"The {pr['title']} is led by the {pr['lead_dept']} Department, with "
            f"collaboration from {', '.join(pr['collab_depts']) or 'no additional departments'}. "
            f"The lead professor is {pr['lead_prof']}."
        )
    for d in DEPARTMENTS:
        n_profs = sum(1 for p in corpus["professors"] if p["dept"] == d)
        docs.append(
            f"The {d} Department is part of the {DEPT_CATEGORY[d]} division. "
            f"It has {n_profs} professors affiliated."
        )
    return docs


def main() -> None:
    corpus = generate_corpus()
    docs = build_documents(corpus)

    # Shuffle gold QA to avoid type-grouping in saved file
    qas = gen_simple_qa(corpus) + gen_multi_hop_qa(corpus) + gen_conditional_qa(corpus)
    random.shuffle(qas)
    for i, q in enumerate(qas, 1):
        q["id"] = i

    (ROOT / "corpus_en.json").write_text(
        json.dumps({"documents": docs, **corpus}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    (ROOT / "gold_qa_500_en.json").write_text(
        json.dumps(qas, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(docs)} documents → corpus_en.json")
    print(f"Wrote {len(qas)} gold QA ({sum(1 for q in qas if q['type']=='simple')} simple, "
          f"{sum(1 for q in qas if q['type']=='multi_hop')} multi-hop, "
          f"{sum(1 for q in qas if q['type']=='conditional')} conditional) → gold_qa_500_en.json")


if __name__ == "__main__":
    main()
