import streamlit as st
import json
import os
import numpy as np
# HF cache defaults (adjust if you want)
os.environ.setdefault('HF_HOME', './hf_cache')
os.environ.setdefault("TRANSFORMERS_CACHE", "./hf_cache/transformers")
# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Gooner bot Chat",
    layout="wide"
)
# =========================
# Lazy imports (important for Streamlit)
# =========================
@st.cache_resource(show_spinner=True)
def load_backend_and_graph():
    from inference_final import (
        load_graph,
        build_node_maps,
        pick_backend,
        retrieve_nodes,
        retrieval_is_valid,
        expand_with_ancestors_and_children,
        build_rag_prompt,
        build_pure_qa_prompt,
        LLMInterface
    )

    GRAPH_PATH = "integrated_graph_revision_with_memory.json"

    graph = load_graph(GRAPH_PATH)
    node_texts, adjacency = build_node_maps(graph)
    ids = sorted(node_texts.keys())
    docs = [node_texts[i] for i in ids]

    backend = pick_backend("auto")

    # embeddings
    if backend == "sbert":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        node_embs = model.encode(docs, convert_to_numpy=True)
        tfidf_vec = None
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=32768)
        node_embs = vec.fit_transform(docs)
        tfidf_vec = vec

    llm = LLMInterface(model_name="LiquidAI/LFM2-1.2B")
    try:
        llm.load()
    except Exception:
        pass

    return {
        "node_texts": node_texts,
        "adjacency": adjacency,
        "ids": ids,
        "docs": docs,
        "backend": backend,
        "node_embs": node_embs,
        "tfidf_vec": tfidf_vec,
        "llm": llm,
        "retrieve_nodes": retrieve_nodes,
        "retrieval_is_valid": retrieval_is_valid,
        "expand": expand_with_ancestors_and_children,
        "rag_prompt": build_rag_prompt,
        "qa_prompt": build_pure_qa_prompt,
    }

# =========================
# Load everything ONCE
# =========================
with st.spinner("Loading graph and model..."):
    ctx = load_backend_and_graph()

# =========================
# Session state
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# Header
# =========================
st.title("Graph-RAG Assistant")
st.caption("ClaimGraph retrieval with gated evidence")

# =========================
# Display chat history
# =========================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================
# Chat input (THIS IS THE LOOP)
# =========================
query = st.chat_input("Ask a question...")



if query:
    # show user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })
    with st.chat_message("user"):
        st.markdown(query)
    raw_answer = ""
    # assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # short query gate
            prefix_prompt = ""
            prompt = ""
            if len(st.session_state.messages) > 0:
                history_block = ""
                for i in range(0,len(st.session_state.messages)-1):
                    if st.session_state.messages[i]["role"] == "user":
                        history_block = history_block + "User: " +  st.session_state.messages[i]["content"] + "\n"
                    if st.session_state.messages[i]["role"] == "assistant":
                        history_block = history_block + "You: " +  st.session_state.messages[i]["content"] + "\n"


                prefix_prompt = f""" {history_block}
\nAbove is your previous chat with this person.
                """
            print(prefix_prompt)
            if len(query.strip()) <= 2:
                prompt = prefix_prompt + ctx["qa_prompt"](query)
                answer = ctx["llm"].generate(prompt, max_new_tokens=1024, temperature=0.2)
            
            else:
                hits, sims = ctx["retrieve_nodes"](
                    query,
                    ctx["ids"],
                    ctx["docs"],
                    ctx["node_embs"],
                    ctx["backend"],
                    tfidf_vec=ctx["tfidf_vec"],
                    top_k=4
                )

                valid = ctx["retrieval_is_valid"](
                    hits,
                    sims,
                    min_sim=0.30 if ctx["backend"] == "sbert" else 0.08,
                    min_gap=0.05,
                    require_gap=False
                )
                answer = "Im not sure."
                if not valid:
                    prompt = prefix_prompt + ctx["qa_prompt"](query)
                    answer = ctx["llm"].generate(prompt, max_new_tokens=1024, temperature=0.2)
                else:
                    expanded = ctx["expand"](
                        [h[0] for h in hits],
                        ctx["node_texts"],
                        ctx["adjacency"],
                        max_ancestor_depth=3
                    )
                    prompt = prefix_prompt + ctx["rag_prompt"](query, expanded)
                    answer = ctx["llm"].generate(prompt, max_new_tokens=1024, temperature=0.15)
                raw_answer = answer
                valid_block = f"\n\n USED CONTEXT {hits}"
                answer = answer + valid_block

        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": raw_answer
    })
