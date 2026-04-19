# 🔧 Code Specifications — Module-by-Module Implementation Guide

> This is the implementation bible. Each module below must match these specs.

---

## 📐 Common Conventions

### File Header Template
```python
"""
<Module short description>

Part of PhD Thesis: Triple-Hybrid RAG with PPO-based L-DWA
Author: Shin Dong-wook <sdw@hoseo.ac.kr>
Thesis Reference: Chapter X, Section Y
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional  # or other types

import numpy as np  # if needed
import torch  # if needed

logger = logging.getLogger(__name__)
```

### Class Structure Template
```python
from dataclasses import dataclass

@dataclass
class ModuleOutput:
    """Return value of module main method."""
    result: Any
    metadata: dict


class ModuleName:
    """Short description.
    
    Args:
        param1: Description
    
    Example:
        >>> m = ModuleName(param1="val")
        >>> result = m.method()
    """
    
    def __init__(self, param1: str) -> None:
        self.param1 = param1
        logger.info(f"Initialized {self.__class__.__name__}")
    
    def main_method(self, input: str) -> ModuleOutput:
        """Do the main thing."""
        # Implementation
        return ModuleOutput(result=..., metadata=...)
```

---

## 📁 src/rag/ — Triple-Hybrid Retrieval

### 1. `src/rag/vector_store.py`

**Port from**: `hybrid-rag-comparsion/src/vector_store.py`

**Purpose**: FAISS-based vector retrieval over unstructured documents.

**Required Class/Functions**:

```python
class VectorStore:
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        index_type: str = "IndexFlatIP",
        api_key: Optional[str] = None,
    ):
        """Initialize FAISS vector store.
        
        Args:
            embedding_model: OpenAI embedding model name.
            index_type: FAISS index type (IndexFlatIP for cosine similarity).
            api_key: OpenAI API key (falls back to OPENAI_API_KEY env var).
        """
    
    def add_documents(self, documents: list[str]) -> None:
        """Embed and add documents to index.
        
        Args:
            documents: List of text chunks.
        """
    
    def search(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        """Retrieve top-k documents.
        
        Returns:
            List of (document, similarity_score) tuples.
        """
    
    def save(self, path: Path) -> None:
        """Save index + documents to disk."""
    
    def load(self, path: Path) -> None:
        """Load index + documents from disk."""
```

**Thesis Reference**: Ch.3 Sec 2.1, Ch.6 Sec 1.2

**Test Cases** (`tests/test_vector_store.py`):
- Empty store search returns empty
- Add 10 docs, search returns 3 with scores
- Save/load roundtrip
- Batch embedding (100 docs)

---

### 2. `src/rag/graph_store.py`

**Port from**: `hybrid-rag-comparsion/src/knowledge_graph.py`

**Purpose**: NetworkX-based graph retrieval via BFS.

**Required Class/Functions**:

```python
class GraphStore:
    def __init__(self, max_depth: int = 3):
        """Initialize empty directed graph."""
    
    def build_from_data(self, data: dict) -> None:
        """Build graph from structured data.
        
        Args:
            data: Must have 'nodes' and 'edges' keys.
                  nodes: list[dict with 'id', 'type', 'attrs']
                  edges: list[dict with 'source', 'target', 'relation']
        """
    
    def search(
        self,
        query_entities: list[str],
        query_relations: list[str] | None = None,
        top_k: int = 3,
    ) -> list[tuple[dict, float]]:
        """BFS 3-hop retrieval.
        
        Returns:
            List of (subgraph_dict, relevance_score) tuples.
        """
    
    def get_stats(self) -> dict:
        """Return {n_nodes, n_edges, avg_degree}."""
```

**Thesis Reference**: Ch.3 Sec 2.2, expected 2,542 nodes / 6,889 edges

---

### 3. `src/rag/ontology_store.py`

**Port from**: `hybrid-rag-comparsion/src/ontology_engine.py`

**Purpose**: Owlready2 + HermiT ontology reasoning.

**Required Class/Functions**:

```python
class OntologyStore:
    def __init__(self, ontology_path: Path):
        """Load OWL ontology.
        
        Args:
            ontology_path: Path to .owl file.
        """
    
    def reason(self) -> None:
        """Run HermiT reasoner (may take several seconds)."""
    
    def search(
        self,
        query_constraints: list[str],
        top_k: int = 3,
    ) -> list[tuple[dict, float]]:
        """Retrieve entities matching constraints.
        
        Returns:
            List of (entity_info, match_score) tuples.
        """
    
    def validate_constraints(self, entity: str, constraints: dict) -> bool:
        """Check if entity satisfies all constraints."""
```

**Thesis Reference**: Ch.3 Sec 2.3

---

### 4. `src/rag/triple_hybrid_rag.py`

**Port + Adapt from**: `hybrid-rag-comparsion/src/triple_hybrid_rag.py`

**Critical Adaptation**: Must support pluggable DWA (R-DWA or L-DWA).

```python
from src.dwa.base import BaseDWA  # New abstract class

class TripleHybridRAG:
    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        ontology_store: OntologyStore,
        dwa: BaseDWA,  # R-DWA or L-DWA, same interface
        llm_model: str = "gpt-4o-mini",
        top_k: int = 3,
    ):
        """Initialize pipeline."""
    
    def query(self, text: str) -> RAGResult:
        """Execute full pipeline."""
        # 1. Query Analyzer
        intent = self.analyze(text)
        
        # 2. DWA (pluggable: R-DWA or L-DWA)
        weights = self.dwa.compute(text, intent)
        
        # 3. Three retrievals
        vec_results = self.vector_store.search(text, self.top_k)
        graph_results = self.graph_store.search(intent.entities, intent.relations)
        onto_results = self.ontology_store.search(intent.constraints)
        
        # 4. Weighted fusion
        fused = self.fuse(vec_results, graph_results, onto_results, weights)
        
        # 5. LLM generation
        answer = self.generate(text, fused)
        
        return RAGResult(
            answer=answer,
            weights=weights,
            sources_used=...,
            latency=...,
        )
```

---

## 🎯 src/intent/ — Query Intent Classification

### 5. `src/intent/rule_based.py`

**Port from**: `hybrid-rag-comparsion/src/query_analyzer.py`

**Purpose**: Rule-based (baseline) intent classification.

```python
@dataclass
class QueryIntent:
    entities: list[str]
    relations: list[str]
    constraints: list[str]
    density: tuple[float, float, float]  # (s_e, s_r, s_c)
    query_type: Literal["simple", "multi_hop", "conditional"]


class RuleBasedIntent:
    def analyze(self, query: str) -> QueryIntent:
        """Extract entities, relations, constraints via regex/rules."""
```

---

### 6. `src/intent/bert_classifier.py` ⭐ NEW

**Thesis Reference**: Ch.3 Sec 3, Ch.5

**Purpose**: BERT multi-label classification for boundary queries.

```python
class BertIntentClassifier:
    def __init__(
        self,
        model_name: str = "klue/bert-base",  # or PubMedBERT for medical
        device: str = "cuda",
    ):
        """Load BERT model for multi-label classification.
        
        Output classes: [simple, multi_hop, conditional]
        (Multi-label: a query can be both multi_hop AND conditional)
        """
    
    def predict(self, query: str) -> dict[str, float]:
        """Return probabilities for each class.
        
        Returns:
            {"simple": 0.1, "multi_hop": 0.7, "conditional": 0.85}
        """
    
    def train(
        self,
        train_data: list[tuple[str, list[str]]],
        epochs: int = 3,
        lr: float = 2e-5,
    ) -> None:
        """Fine-tune on labeled queries."""
    
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...
```

**Key innovation**: Multi-label output → captures boundary queries
**Example boundary**: "35세 이하 AND 컴공과 소속 교수 중 누가 ICML 논문 썼나?"
- multi_hop: 0.9 (교수 → 논문)
- conditional: 0.95 (35세 이하, 컴공과)

---

## ⚖️ src/dwa/ — Dynamic Weighting

### 7. `src/dwa/base.py` ⭐ NEW

**Purpose**: Abstract interface for R-DWA and L-DWA.

```python
from abc import ABC, abstractmethod

class BaseDWA(ABC):
    """Abstract base class for Dynamic Weighting Algorithms."""
    
    @abstractmethod
    def compute(self, query: str, intent: QueryIntent) -> DWAWeights:
        """Compute (α, β, γ) weights on simplex."""


@dataclass
class DWAWeights:
    alpha: float  # Vector weight
    beta: float   # Graph weight  
    gamma: float  # Ontology weight
    
    def __post_init__(self):
        total = self.alpha + self.beta + self.gamma
        assert abs(total - 1.0) < 1e-5, f"Weights don't sum to 1: {total}"
```

---

### 8. `src/dwa/rdwa.py`

**Port from**: `hybrid-rag-comparsion/src/dwa.py`

**Thesis Reference**: Ch.4

```python
class RuleBasedDWA(BaseDWA):
    def __init__(self, lambda_: float = 0.3):
        """Initialize R-DWA.
        
        Args:
            lambda_: Adjustment strength (thesis Eq. 4-2 to 4-5).
        """
    
    def compute(self, query: str, intent: QueryIntent) -> DWAWeights:
        # Stage 1: Query type → base weights
        base = self._type_to_base(intent.query_type)
        
        # Stage 2: Density-based adjustment
        adjusted = self._adjust(base, intent.density)
        
        # Stage 3: Normalization
        return self._normalize(adjusted)
    
    def _type_to_base(self, qtype: str) -> tuple[float, float, float]:
        """Thesis Table 4-1."""
        return {
            "simple":      (0.6, 0.2, 0.2),
            "multi_hop":   (0.2, 0.6, 0.2),
            "conditional": (0.2, 0.2, 0.6),
        }[qtype]
```

---

### 9. `src/dwa/ldwa.py` ⭐ CORE

**Thesis Reference**: Ch.5

**Purpose**: PPO-based Learned DWA.

```python
class LearnedDWA(BaseDWA):
    def __init__(
        self,
        policy_checkpoint: Path,
        intent_classifier: BertIntentClassifier,
        device: str = "cuda",
    ):
        """Initialize with trained PPO policy."""
        self.actor_critic = torch.load(policy_checkpoint)
        self.actor_critic.eval()
        self.intent = intent_classifier
    
    def compute(self, query: str, intent: QueryIntent) -> DWAWeights:
        # 1. State extraction
        state = self._extract_state(query, intent)  # 18-dim
        
        # 2. Policy forward
        with torch.no_grad():
            dirichlet_params = self.actor_critic.actor(state)
            dist = torch.distributions.Dirichlet(dirichlet_params)
            weights = dist.mean  # Use mean for inference
        
        alpha, beta, gamma = weights.tolist()
        return DWAWeights(alpha, beta, gamma)
    
    def _extract_state(self, query: str, intent: QueryIntent) -> torch.Tensor:
        """Thesis Eq. 5-1 to 5-3 (18-dim state)."""
```

---

## 🧠 src/ppo/ — PPO Infrastructure ⭐ ALL NEW

### 10. `src/ppo/mdp.py`

**Thesis Reference**: Ch.5 Sec 1

```python
@dataclass
class State:
    """18-dim state vector."""
    density: tuple[float, float, float]       # s_e, s_r, s_c
    intent_logits: tuple[float, float, float]  # BERT outputs
    source_stats: list[float]                  # 9-dim (3 sources × 3 stats)
    query_meta: tuple[float, float, float]     # length, entities, negation
    
    def to_tensor(self) -> torch.Tensor:
        """Return 18-dim tensor."""


@dataclass  
class Action:
    """Simplex-constrained action."""
    alpha: float
    beta: float
    gamma: float


def compute_reward(
    f1: float,
    em: float,
    faithfulness: float,
    latency: float,
) -> float:
    """Thesis Eq. 5-7."""
    return 0.5*f1 + 0.3*em + 0.2*faithfulness - 0.1*max(0, latency - 5.0)
```

---

### 11. `src/ppo/actor_critic.py`

**Thesis Reference**: Ch.5 Sec 2 (Figure 5-3)

```python
class ActorCritic(nn.Module):
    """Policy network: ~6,081 parameters total."""
    
    def __init__(self, state_dim: int = 18):
        super().__init__()
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        # Actor head (outputs Dirichlet concentrations α1, α2, α3)
        self.actor_head = nn.Sequential(
            nn.Linear(64, 3),
            nn.Softplus(),  # Ensure positive
        )
        # Critic head
        self.critic_head = nn.Linear(64, 1)
        
        # Orthogonal init (key for stable training!)
        self._init_weights()
    
    def _init_weights(self):
        """Orthogonal init → initial policy near uniform (1/3, 1/3, 1/3)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor) -> tuple[Tensor, Tensor]:
        """Returns (dirichlet_params, value)."""
        features = self.backbone(state)
        return self.actor_head(features) + 1e-6, self.critic_head(features)
```

---

### 12. `src/ppo/trainer.py`

**Thesis Reference**: Ch.5 Sec 3, Table 5-4

```python
class PPOTrainer:
    def __init__(
        self,
        actor_critic: ActorCritic,
        env: TripleHybridEnv,  # Wraps TripleHybridRAG with cached rewards
        cache: OfflineCache,
        config: PPOConfig,
    ):
        self.ac = actor_critic
        self.env = env
        self.cache = cache
        self.optimizer = torch.optim.Adam(
            self.ac.parameters(), 
            lr=config.learning_rate,
        )
        self.config = config
    
    def train(self, total_episodes: int = 10000):
        for episode in range(total_episodes):
            # 1. Rollout
            rollout = self._collect_rollout(batch_size=32)
            
            # 2. Compute advantages (GAE)
            advantages, returns = self._compute_gae(
                rollout.rewards, rollout.values,
                gamma=0.99, lambda_=0.95,
            )
            
            # 3. PPO update (4 epochs, minibatch 8)
            self._ppo_update(rollout, advantages, returns)
            
            # 4. Logging
            if episode % 100 == 0:
                self._log_metrics(episode, rollout)
                self._save_checkpoint(episode)
```

---

## 💾 src/utils/offline_cache.py ⭐ CRITICAL

**Thesis Reference**: Ch.5 Sec 4, Ch.6 Sec 10

**Purpose**: Pre-compute rewards for all (query, weight) combinations to eliminate runtime API calls during PPO training.

```python
class OfflineCache:
    """SQLite-based reward cache.
    
    Storage:
      (query_id, alpha_int, beta_int, gamma_int) → RewardComponents
      
    Discretization: alpha/beta/gamma to 0.1 grid
    Total entries: 5000 queries × 66 weight combos = 330,000
    """
    
    def __init__(self, db_path: Path):
        """Initialize SQLite DB."""
    
    def build(
        self,
        queries: list[dict],
        rag_pipeline: TripleHybridRAG,
        step: float = 0.1,
        n_workers: int = 10,
    ) -> None:
        """Build cache for all queries × all discretized weights.
        
        With 10 parallel workers, estimated 10-15 hours.
        """
    
    def lookup(
        self,
        query_id: str,
        weights: DWAWeights,
    ) -> Optional[RewardComponents]:
        """Look up reward for (query, discretized weights)."""
    
    def stats(self) -> dict:
        """Return {total_entries, total_queries, avg_f1, cache_hits, ...}"""


@dataclass
class RewardComponents:
    f1: float
    em: float
    faithfulness: float
    latency: float
    
    def total_reward(self) -> float:
        """Apply thesis Eq. 5-7."""
        return 0.5*self.f1 + 0.3*self.em + 0.2*self.faithfulness \
               - 0.1*max(0, self.latency - 5.0)
```

**Discretization logic**:
```python
def discretize(w: float, step: float = 0.1) -> int:
    return round(w / step)

# Valid combos (alpha_int, beta_int, gamma_int) where sum == 10:
# (10,0,0), (9,1,0), (9,0,1), (8,2,0), (8,1,1), (8,0,2), ..., (0,0,10)
# Total = C(12, 2) = 66 combinations
```

---

## 📊 src/eval/ — Evaluation

### 13. `src/eval/metrics.py`

**Port from**: `hybrid-rag-comparsion/src/evaluator.py`

```python
def f1_score(pred: str, gold: str) -> float:
    """Token-level F1 with Korean-aware tokenization."""

def exact_match(pred: str, gold: str) -> bool:
    """Normalized exact match."""

def recall_at_k(retrieved: list[str], relevant: list[str], k: int = 3) -> float:
    """Recall@k for retrieval evaluation."""

def faithfulness(answer: str, contexts: list[str], llm_judge: str = "gpt-4o-mini") -> float:
    """RAGAS-style faithfulness (0-1)."""
```

---

### 14. `src/eval/benchmark.py` ⭐ NEW

**Thesis Reference**: Ch.6

```python
class BenchmarkRunner:
    def __init__(self, rag: TripleHybridRAG):
        self.rag = rag
    
    def run_university(self, qa_file: Path) -> BenchmarkResult: ...
    def run_hotpotqa(self, qa_file: Path) -> BenchmarkResult: ...
    def run_musique(self, qa_file: Path) -> BenchmarkResult: ...
    def run_pubmedqa(self, qa_file: Path) -> BenchmarkResult: ...
    
    def run_all(self, with_seeds: list[int] = [42, 123, 999]) -> dict:
        """Run all 4 benchmarks × 3 seeds."""
```

---

## 🛠️ src/utils/seed.py

```python
def set_seed(seed: int = 42) -> None:
    """Reproducibility setup.
    
    Sets:
    - random.seed
    - numpy.random.seed
    - torch.manual_seed
    - torch.cuda.manual_seed_all
    - torch.backends.cudnn.deterministic = True
    - torch.backends.cudnn.benchmark = False
    - os.environ['PYTHONHASHSEED']
    """
```

---

## 📝 Scripts (CLI runners)

### `scripts/build_cache.py`
```bash
python scripts/build_cache.py \
    --qa data/university/gold_qa_5000.json \
    --output cache/reward/university.sqlite \
    --step 0.1 \
    --workers 10
```

### `scripts/train_ppo.py`
```bash
python scripts/train_ppo.py \
    --cache cache/reward/university.sqlite \
    --config configs/ppo_default.yaml \
    --seed 42 \
    --output cache/ppo_checkpoints/seed_42/
```

### `scripts/run_benchmarks.py`
```bash
python scripts/run_benchmarks.py \
    --policy cache/ppo_checkpoints/seed_42/final.pt \
    --benchmarks university,hotpotqa,musique,pubmedqa \
    --output results/table_6_2.json
```

---

## 🧪 Testing Requirements

Every module MUST have:
- `tests/test_<module>.py` with at least 3 test cases
- Smoke test: minimal valid input produces expected output type
- Edge case: empty/invalid input handled gracefully
- Integration: module works with actual dependencies (when possible)

Run all tests:
```bash
docker-compose exec triple_rag pytest tests/ -v
```

---

## 📏 Code Quality Requirements

1. **All public functions must have**:
   - Type hints on args and return
   - Google-style docstring
   - Example usage in docstring (if non-trivial)

2. **No magic numbers**: Use named constants at module top.

3. **Logging over print**: Use `logger.info/warning/error`.

4. **Path handling**: Always `pathlib.Path`, never string concatenation.

5. **Configuration**: YAML for all hyperparameters, loaded via pydantic.
