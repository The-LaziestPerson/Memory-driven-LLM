#!/usr/bin/env python3
"""
graph_rag_infer_with_gate.py

RAG-style inference over ClaimGraph JSON produced by graph_make.py with a
relevance gate:

 - If retrieved top hit is below an absolute similarity threshold -> REJECT retrieval
 - Optionally require a minimum gap between top-1 and top-2 -> REJECT if ambiguous
 - Short queries (e.g. "Hi") are forced to pure QA
 - Backends supported: sentence-transformers (SBERT), OpenAI embeddings, TF-IDF fallback
 - Embedding cache supported (numpy .npz) for SBERT/OpenAI
 - Uses LLMInterface from graph_make.py if available, otherwise a minimal fallback

Usage examples:
  python graph_rag_infer_with_gate.py --graph integrated_graph_revision_fixed.json --backend sbert --top-k 4 --verbose
  python graph_rag_infer_with_gate.py --backend openai --min-sim 0.22

Notes:
 - If you choose --backend auto the script will pick SBERT if installed,
   otherwise OpenAI if OPENAI_API_KEY is set, otherwise TF-IDF.
 - If --min-sim is not provided, the script picks a conservative default
   depending on backend.
"""
import argparse
import json
import os
# HF cache defaults (adjust if you want)
os.environ.setdefault('HF_HOME', './hf_cache')
os.environ.setdefault("TRANSFORMERS_CACHE", "./hf_cache/transformers")

import sys
import time
from typing import List, Dict, Tuple, Optional
import numpy as np

# Optional libs
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import openai
except Exception:
    openai = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
except Exception:
    TfidfVectorizer = None
    sk_cosine_similarity = None

# Try import graph_make's LLMInterface
try:
    from graph_make import LLMInterface
except Exception:
    class LLMInterface:
        def __init__(self, model_name="dummy", **kwargs):
            self.model_name = model_name
        def load(self, adapter_dir=None):
            return
        def generate(self, prompt, max_new_tokens=256, temperature=0.7):
            return "LLMInterface not available. Install or provide OpenAI key and sentence-transformers."

# -----------------------
# Helpers: load graph & maps
# -----------------------
def load_graph(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "graph" in data:
        return data["graph"]
    return data

def build_node_maps(graph: Dict) -> Tuple[Dict[int,str], Dict[str, Dict[int, List[int]]]]:
    nodes = {int(n["id"]): n.get("text", "") for n in graph.get("nodes", [])}
    edges = graph.get("edges", [])
    children = {}
    parents = {}
    for e in edges:
        typ = e.get("type")
        s = e.get("source")
        t = e.get("target")
        try:
            s = int(s); t = int(t)
        except Exception:
            pass
        if typ == "derived":
            children.setdefault(s, []).append(t)
            parents.setdefault(t, []).append(s)
    return nodes, {"children": children, "parents": parents}

# -----------------------
# Embeddings backends
# -----------------------
def pick_backend(cli_backend: Optional[str]) -> str:
    if cli_backend and cli_backend != "auto":
        return cli_backend
    if SentenceTransformer is not None:
        return "sbert"
    if openai is not None and os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "tfidf"

def compute_embeddings_sbert(texts: List[str], model_name="all-MiniLM-L6-v2"):
    m = SentenceTransformer(model_name)
    return m.encode(texts, show_progress_bar=True, convert_to_numpy=True)

def compute_embeddings_openai(texts: List[str], engine="text-embedding-3-small", batch=16):
    if openai is None:
        raise RuntimeError("openai package missing")
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    all_emb = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        resp = openai.Embeddings.create(model=engine, input=chunk)
        all_emb.extend([np.array(x["embedding"], dtype=float) for x in resp["data"]])
    return np.vstack(all_emb)

def compute_embeddings_tfidf(texts: List[str]):
    if TfidfVectorizer is None:
        raise RuntimeError("sklearn not available")
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=32768)
    X = vec.fit_transform(texts)
    return X, vec

# -----------------------
# Retrieval returns (hits, sims)
# hits = list of (node_id, score) sorted desc
# sims = full score vector aligned to 'ids' list
# -----------------------
def retrieve_nodes(query: str,
                   ids: List[int],
                   docs: List[str],
                   node_embs,
                   backend: str,
                   tfidf_vec=None,
                   top_k: int = 5) -> Tuple[List[Tuple[int,float]], np.ndarray]:
    if backend == "sbert":
        model = SentenceTransformer("all-MiniLM-L6-v2")
        qemb = model.encode([query], convert_to_numpy=True)[0]
        # node_embs: numpy array shape (N, d)
        norms_nodes = np.linalg.norm(node_embs, axis=1) + 1e-12
        qnorm = np.linalg.norm(qemb) + 1e-12
        sims = (node_embs @ qemb) / (norms_nodes * qnorm)
        order = np.argsort(-sims)[:top_k]
        hits = [(ids[i], float(sims[i])) for i in order]
        return hits, sims

    if backend == "openai":
        if openai is None:
            raise RuntimeError("openai missing")
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        resp = openai.Embeddings.create(model="text-embedding-3-small", input=[query])
        qemb = np.array(resp["data"][0]["embedding"], dtype=float)
        norms_nodes = np.linalg.norm(node_embs, axis=1) + 1e-12
        qnorm = np.linalg.norm(qemb) + 1e-12
        sims = (node_embs @ qemb) / (norms_nodes * qnorm)
        order = np.argsort(-sims)[:top_k]
        hits = [(ids[i], float(sims[i])) for i in order]
        return hits, sims

    # tfidf fallback: node_embs is sparse matrix, tfidf_vec provided
    if backend == "tfidf":
        if tfidf_vec is None:
            raise RuntimeError("tfidf vectorizer missing")
        qv = tfidf_vec.transform([query])
        sims = sk_cosine_similarity(qv, node_embs)[0]  # (N,)
        order = np.argsort(-sims)[:top_k]
        hits = [(ids[i], float(sims[i])) for i in order]
        return hits, sims

    raise RuntimeError(f"unknown backend {backend}")

# -----------------------
# Expansion: ancestors (chain) + direct children
# -----------------------
def expand_with_ancestors_and_children(hit_ids: List[int],
                                       node_texts: Dict[int,str],
                                       adjacency: Dict[str, Dict[int, List[int]]],
                                       max_ancestor_depth: int = 4) -> List[Tuple[int,List[str]]]:
    parents = adjacency.get("parents", {})
    children = adjacency.get("children", {})
    expanded = []
    for nid in hit_ids:
        # build ancestor chain (root-first)
        chain = []
        cur = nid
        depth = 0
        seen = set([cur])
        while depth < max_ancestor_depth:
            ps = parents.get(cur, [])
            if not ps:
                break
            # take first parent heuristically
            cur = ps[0]
            if cur in seen:
                break
            seen.add(cur)
            chain.append(node_texts.get(cur, ""))
            depth += 1
        chain.reverse()
        child_texts = [node_texts.get(c, "") for c in children.get(nid, [])[:4]]
        block = chain + [node_texts.get(nid, "")] + child_texts
        expanded.append((nid, block))
    return expanded

# -----------------------
# Prompt builders
# -----------------------
def build_rag_prompt(query: str, retrieved_blocks: List[Tuple[int,List[str]]], max_chars: int = 4000) -> str:
    parts = ["You are answering question. Use english only, no need to translate anything. Answer fully as much as you can. You may use mathematical expressions to explain some equation and formulas. Feel free to use emojis and express your emotion!"]
    parts.append("Below is your thoughts and beliefs. (Use this to help answering).")
    for i, (nid, lines) in enumerate(retrieved_blocks):
        joined = " | ".join([l.strip() for l in lines if l and l.strip()])
        if len(joined) > 900:
            joined = joined[:890] + "…"
        parts.append(f" {i} : {joined}")
    parts.append(f"\nQuestion: {query.strip()}")
    parts.append("Answer:")
    prompt = "\n\n".join(parts)
    if len(prompt) > max_chars:
        prompt = prompt[:max_chars-1] + "\n…"
    return prompt

def build_pure_qa_prompt(query: str) -> str:
    return (
        "YOUR MEMORY ABOUT YOURSELF:\My name is Gooner. My creator named me. He is a high-school student that is very smart. He told me to study and help others, so I like helping others and answering question expertly!\n"
        "ROLE:\nYou are an expert assistant.\n"
        "INFO:\nYou can use emojis to help bring expressions.\n You can use equations/math expressions to answer the question more specificly.\nAnswer as much as you can. (You may answer once)\n\n"
        f"Question: {query}\n\nFull Answer:"
    )

# -----------------------
# Relevance gate
# -----------------------
def retrieval_is_valid(hits: List[Tuple[int,float]],
                       sims: np.ndarray,
                       min_sim: float,
                       min_gap: float,
                       require_gap: bool) -> bool:
    if not hits or len(hits) == 0:
        return False
    top_sim = hits[0][1]
    if top_sim < min_sim:
        return False
    if require_gap and len(hits) > 1:
        second_sim = hits[1][1]
        if (top_sim - second_sim) < min_gap:
            return False
    return True

# -----------------------
# Main CLI
# -----------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graph", type=str, default="integrated_graph_revision_fixed.json", help="Graph JSON")
    p.add_argument("--emb-cache", type=str, default="graph_emb_cache.npz", help="Embedding cache")
    p.add_argument("--backend", type=str, default="auto", choices=["auto","sbert","openai","tfidf"])
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--max-anc-depth", type=int, default=3)
    p.add_argument("--min-sim", type=float, default=None, help="Absolute similarity gate (auto-defaults per backend)")
    p.add_argument("--min-gap", type=float, default=0.05, help="Minimum gap between top1 and top2 (if --require-gap)")
    p.add_argument("--require-gap", action="store_true", help="Require min-gap check")
    p.add_argument("--short-query-cutoff", type=int, default=3, help="Force pure QA when query length <= this")
    p.add_argument("--model-name", type=str, default="LiquidAI/LFM2-1.2B", help="LLMInterface model")
    p.add_argument("--adapter-dir", type=str, default=None, help="Adapter dir for LLMInterface")
    p.add_argument("--verbose", action="store_true", default=False)
    args = p.parse_args()

    # load graph
    graph = load_graph(args.graph)
    node_texts, adjacency = build_node_maps(graph)
    ids = sorted(list(node_texts.keys()))
    docs = [node_texts[i] for i in ids]

    backend = pick_backend(args.backend)
    if args.verbose:
        print(f"[INFO] chosen backend: {backend}")

    # pick sensible default min_sim if not provided
    if args.min_sim is None:
        if backend == "sbert":
            min_sim = 0.30
        elif backend == "openai":
            min_sim = 0.22
        else:  # tfidf
            min_sim = 0.08
        if args.verbose:
            print(f"[INFO] auto min_sim set to {min_sim} for backend {backend}")
    else:
        min_sim = args.min_sim

    # load or compute embeddings (SBERT/OpenAI) or tfidf vectorizer
    node_embs = None
    tfidf_vec = None
    # try load cache
    if backend in ("sbert","openai") and os.path.exists(args.emb_cache):
        try:
            d = np.load(args.emb_cache, allow_pickle=True)
            saved_backend = d.get("backend", None).item() if "backend" in d else None
            if saved_backend == backend:
                node_embs = d["embs"]
                saved_ids = [int(x) for x in d["ids"].tolist()]
                if saved_ids != ids:
                    # reorder according to ids
                    id_to_idx = {saved_ids[i]: i for i in range(len(saved_ids))}
                    node_embs = np.array([node_embs[id_to_idx[i]] if i in id_to_idx else np.zeros(node_embs.shape[1]) for i in ids])
                if args.verbose:
                    print(f"[INFO] loaded embedding cache {args.emb_cache}")
            else:
                if args.verbose:
                    print("[INFO] emb cache backend mismatch or missing; recomputing")
        except Exception as e:
            if args.verbose:
                print("[WARN] couldn't load emb cache:", e)

    if backend == "sbert" and node_embs is None:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed; please pip install sentence-transformers or choose another backend")
        if args.verbose:
            print("[INFO] computing SBERT embeddings...")
        node_embs = compute_embeddings_sbert(docs, model_name="all-MiniLM-L6-v2")
        np.savez_compressed(args.emb_cache, embs=node_embs, ids=np.array(ids), backend=np.array([backend]))

    if backend == "openai" and node_embs is None:
        if openai is None or not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("openai not configured; set OPENAI_API_KEY or use another backend")
        if args.verbose:
            print("[INFO] computing OpenAI embeddings (API calls)...")
        node_embs = compute_embeddings_openai(docs)
        np.savez_compressed(args.emb_cache, embs=node_embs, ids=np.array(ids), backend=np.array([backend]))

    if backend == "tfidf":
        if TfidfVectorizer is None:
            raise RuntimeError("sklearn not available; install scikit-learn or choose another backend")
        if args.verbose:
            print("[INFO] computing TF-IDF matrix...")
        X, vec = compute_embeddings_tfidf(docs)
        node_embs = X
        tfidf_vec = vec

    # prepare LLMInterface
    llm = LLMInterface(model_name=args.model_name)
    try:
        llm.load(adapter_dir=args.adapter_dir)
    except Exception:
        if args.verbose:
            print("[WARN] LLMInterface.load failed or absent; continuing with fallback generate()")

    print("Ready. Type your query and press Enter (Ctrl+C to quit).")
    try:
        while True:
            query = input("\n> ").strip()
            if not query:
                continue

            # short query guard
            if len(query.strip()) <= args.short_query_cutoff:
                if args.verbose:
                    print("[INFO] Short query -> forcing pure QA (no retrieval)")
                prompt = build_pure_qa_prompt(query)
                out = llm.generate(prompt, max_new_tokens=256, temperature=0.2)
                print("\n---\nAnswer:\n", out, "\n---")
                continue

            # retrieve
            hits, sims = retrieve_nodes(query, ids, docs, node_embs, backend, tfidf_vec=tfidf_vec, top_k=args.top_k)
            if args.verbose:
                print(f"[DEBUG] hits: {hits}")
                top_sim = hits[0][1] if hits else None
                print(f"[DEBUG] top_sim={top_sim:.4f}  min_sim={min_sim:.4f}")

            # gate decision
            valid = retrieval_is_valid(hits, sims, min_sim=min_sim, min_gap=args.min_gap, require_gap=args.require_gap)
            if not valid:
                if args.verbose:
                    print("[INFO] Retrieval rejected by gate — falling back to pure QA")
                prompt = build_pure_qa_prompt(query)
                out = llm.generate(prompt, max_new_tokens=256, temperature=0.2)
                print("\n---\nAnswer:\n", out, "\n---")
                continue

            # expand contexts
            hit_ids = [h[0] for h in hits]
            expanded = expand_with_ancestors_and_children(hit_ids, node_texts, adjacency, max_ancestor_depth=args.max_anc_depth)
            prompt = build_rag_prompt(query, expanded)
            if args.verbose:
                print("[PROMPT]\n", prompt[:1600], "\n---")

            # generate
            out = llm.generate(prompt, max_new_tokens=256, temperature=0.15)
            print("\n---\nAnswer:\n", out, "\n---")
            print("Used contexts:", [f"node_{nid}" for nid in hit_ids])

    except KeyboardInterrupt:
        print("\nbye")
        sys.exit(0)

if __name__ == "__main__":
    main()
