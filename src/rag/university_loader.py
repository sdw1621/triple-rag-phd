"""
Synthetic university corpus loader (V/G/O builder for the JKSCI 2025 dataset).

Part of PhD Thesis: Triple-Hybrid RAG with PPO-based L-DWA
Author: Shin Dong-wook <sdw@hoseo.ac.kr>
Thesis Reference: Chapter 6 Section 1.1 (synthetic university benchmark)

Wraps ``data/university/dataset_generator.generate_university_data(seed=42)``,
which deterministically builds the **extended** corpus that gold_qa_5000.json
was generated from:

    60 departments / 577 professors / 1,505 courses / 400 projects.

The smaller ``university_data.py`` (8/30/40/15) is a separate baseline kept
for unit testing the loader contract — not used in production builds.

Usage:
    docs = build_documents()
    graph = build_graph()
    ontology = build_ontology()
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from src.rag.graph_store import GraphStore
from src.rag.ontology_store import OntologyStore, PersonInstance

logger = logging.getLogger(__name__)

DATASET_GENERATOR_PATH = Path("/workspace/data/university/dataset_generator.py")
DEFAULT_SEED: int = 42


def _load_generator(path: Path = DATASET_GENERATOR_PATH) -> ModuleType:
    """Import ``dataset_generator.py`` from a non-package path."""
    spec = importlib.util.spec_from_file_location("university_dataset_generator", str(path))
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"dataset_generator.py not found: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("university_dataset_generator", module)
    spec.loader.exec_module(module)
    return module


_DATA_CACHE: dict[int, dict[str, Any]] = {}


def load_university_data(seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Generate (and cache) the extended university corpus dict.

    Returns:
        Dict with keys ``depts``, ``professors``, ``courses``, ``projects``,
        ``dept_profs``, ``dept_courses``, ``course_profs``.
    """
    if seed not in _DATA_CACHE:
        gen = _load_generator()
        _DATA_CACHE[seed] = gen.generate_university_data(seed=seed)
    return _DATA_CACHE[seed]


# ---------- documents ----------

def build_documents(data: dict[str, Any] | None = None) -> list[str]:
    """Generate the document corpus for VectorStore.

    Per-source counts (with seed=42):
        - 577 professor profile docs
        - 60 department blurbs
        - 1,505 course descriptions
        - 400 project descriptions
        ≈ 2,542 documents total.
    """
    d = data or load_university_data()

    docs: list[str] = []
    for prof in d["professors"]:
        docs.append(
            f"{prof['name']} 교수는 {prof['age']}세이며 {prof['dept']} 소속이다. "
            f"연구 분야는 {prof['research']}이며 담당 과목은 {', '.join(prof['courses'])}이다."
        )
    for dept_name in d["depts"]:
        prof_count = len(d["dept_profs"].get(dept_name, []))
        docs.append(f"{dept_name}은(는) 본 대학 소속 학과로 {prof_count}명의 교수가 재직 중이다.")
    for course in d["courses"]:
        prof_names = d["course_profs"].get(course["name"], [])
        if prof_names:
            docs.append(
                f"{course['name']} 과목은 {course['dept']} 개설 과목이며 "
                f"담당 교수는 {', '.join(prof_names)}이다."
            )
        else:
            docs.append(f"{course['name']} 과목은 {course['dept']} 개설 과목이다.")
    for project_name, participants in d["projects"].items():
        docs.append(
            f"{project_name} 프로젝트에는 {', '.join(participants)} 등이 참여하고 있다."
        )
    return docs


# ---------- graph ----------

def build_graph(data: dict[str, Any] | None = None, max_depth: int = 3) -> GraphStore:
    """Construct the knowledge graph from the extended corpus.

    Node types: Department, Professor, Course, Project.
    Edge types: 소속 (prof→dept), 담당 (prof→course), 협력 (prof↔prof),
                참여 (prof→project), 개설 (course→dept).
    """
    d = data or load_university_data()

    g = GraphStore(max_depth=max_depth)

    dept_id_by_name: dict[str, str] = {}
    for i, dept_name in enumerate(d["depts"]):
        node_id = f"d{i}"
        dept_id_by_name[dept_name] = node_id
        g.add_node(node_id, dept_name, "Department")

    prof_id_by_name: dict[str, str] = {}
    for prof in d["professors"]:
        prof_id_by_name[prof["name"]] = prof["id"]
        g.add_node(
            prof["id"], prof["name"], "Professor",
            age=prof["age"], dept=prof["dept"], research=prof["research"],
        )
        if prof["dept"] in dept_id_by_name:
            g.add_edge(prof["id"], "소속", dept_id_by_name[prof["dept"]])

    course_id_by_name: dict[str, str] = {}
    for i, course in enumerate(d["courses"]):
        node_id = f"c{i}"
        course_id_by_name[course["name"]] = node_id
        g.add_node(node_id, course["name"], "Course", dept=course["dept"])
        if course["dept"] in dept_id_by_name:
            g.add_edge(node_id, "개설", dept_id_by_name[course["dept"]])

    # Prof → Course (담당)
    for prof in d["professors"]:
        for course_name in prof["courses"]:
            cid = course_id_by_name.get(course_name)
            if cid:
                g.add_edge(prof["id"], "담당", cid)

    # Prof ↔ Prof (협력) — undirected, store one direction (graph adds inverse).
    added_collab: set[tuple[str, str]] = set()
    for prof in d["professors"]:
        for collab_name in prof.get("collab", []):
            collab_id = prof_id_by_name.get(collab_name)
            if not collab_id:
                continue
            pair = tuple(sorted([prof["id"], collab_id]))
            if pair in added_collab:
                continue
            added_collab.add(pair)
            g.add_edge(prof["id"], "협력", collab_id)

    # Prof → Project (참여)
    for i, (project_name, participants) in enumerate(d["projects"].items()):
        proj_id = f"pr{i}"
        g.add_node(proj_id, project_name, "Project")
        for participant_name in participants:
            pid = prof_id_by_name.get(participant_name)
            if pid:
                g.add_edge(pid, "참여", proj_id)

    return g


# ---------- ontology ----------

def build_ontology(
    data: dict[str, Any] | None = None,
    max_instances: int | None = None,
) -> OntologyStore:
    """Construct the ontology with one PersonInstance per professor.

    Args:
        data: Optional pre-loaded corpus dict.
        max_instances: Cap the instance count (debugging / smoke tests).
            Default None = use all 577.
    """
    d = data or load_university_data()
    profs = d["professors"]
    if max_instances:
        profs = profs[:max_instances]

    instances = tuple(
        PersonInstance(
            name=prof["name"],
            person_type="FullProfessor",  # generator does not assign rank
            age=int(prof["age"]),
            dept=prof["dept"],
            courses=list(prof["courses"]),
        )
        for prof in profs
    )
    return OntologyStore(instances=instances, try_owlready=False)


# ---------- introspection ----------

def stats(data: dict[str, Any] | None = None) -> dict[str, int]:
    """Return corpus counts."""
    d = data or load_university_data()
    n_collab = sum(len(p.get("collab", [])) for p in d["professors"])
    return {
        "n_departments": len(d["depts"]),
        "n_professors": len(d["professors"]),
        "n_courses": len(d["courses"]),
        "n_projects": len(d["projects"]),
        "n_collaborations": n_collab,
    }
