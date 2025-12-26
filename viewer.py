import json
import streamlit as st
import networkx as nx
from pyvis.network import Network
import tempfile
import os
from collections import defaultdict
from typing import List

# -----------------------------
# Load JSON
# -----------------------------
@st.cache_data
def load_graph(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# -----------------------------
# Build derived graph (topics + claim nodes + claim->claim edges)
# -----------------------------
def build_topic_graph(data):
    # Use directed graph so we can keep edge direction / types
    G = nx.DiGraph()

    nodes = data["graph"]["nodes"]
    edges = data["graph"].get("edges", [])

    topic_to_nodes = defaultdict(list)

    # Register claim nodes
    for node in nodes:
        # Node IDs from JSON are integers; keep them as ints here
        nid = node["id"]
        text = node["text"]
        prov = node.get("provenance", [])

        # topics listed in provenance (topic_idx)
        topics = sorted({p["topic_idx"] for p in prov}) if prov else []
        dominant_topic = topics[0] if topics else -1

        G.add_node(
            nid,
            label=(text[:80] + ("..." if len(text) > 80 else "")),
            full_text=text,
            topics=topics,
            dominant_topic=dominant_topic,
            provenance=prov,
            is_topic_root=False,
        )

        for t in topics:
            topic_to_nodes[t].append(nid)

    # Create virtual topic roots (string IDs to avoid collision with numeric claim IDs)
    for topic_idx, nids in topic_to_nodes.items():
        root_id = f"topic_{topic_idx}"
        G.add_node(
            root_id,
            label=f"TOPIC {topic_idx}",
            full_text=f"Topic {topic_idx}",
            topics=[topic_idx],
            dominant_topic=topic_idx,
            provenance=[],
            is_topic_root=True,
        )
        # connect root -> claims (undirected feel; use directed edges for clarity)
        for nid in nids:
            # attach topic root as parent-like index
            G.add_edge(root_id, nid, edge_type="topic_index", score=1.0)

    # Add claim->claim edges from JSON (if any); keep their types and scores
    for e in edges:
        # JSON edges might use keys named differently; assume source,target,type,score
        try:
            src = e.get("source", e.get("src", e.get("s")))
            dst = e.get("target", e.get("dst", e.get("t")))
            typ = e.get("type", e.get("edge_type", "related"))
            score = float(e.get("score", 1.0))
        except Exception:
            continue

        # Only add the edge if both endpoints already exist in G (skip otherwise)
        if src in G.nodes and dst in G.nodes:
            G.add_edge(src, dst, edge_type=typ, score=score)
        else:
            # If endpoint is missing, still add nodes minimally and attach the edge
            if src not in G.nodes:
                G.add_node(src, label=f"[{src}]", full_text=str(src), topics=[], dominant_topic=-1, provenance=[], is_topic_root=False)
            if dst not in G.nodes:
                G.add_node(dst, label=f"[{dst}]", full_text=str(dst), topics=[], dominant_topic=-1, provenance=[], is_topic_root=False)
            G.add_edge(src, dst, edge_type=typ, score=score)

    return G, sorted(topic_to_nodes.keys())

# -----------------------------
# Coloring helpers
# -----------------------------
def topic_color(topic):
    palette = [
        "#ff5555", "#50fa7b", "#8be9fd",
        "#bd93f9", "#f1fa8c", "#ff79c6",
        "#ffaa00", "#8affc1", "#c7a1ff", "#ffd480"
    ]
    return palette[int(topic) % len(palette)]

def edge_style_by_type(edge_type):
    # returns dict of kwargs for pyvis add_edge
    if edge_type == "contradicts":
        return {"color": "#ff5555", "dashes": True, "arrows": "to"}
    if edge_type == "derived":
        return {"color": "#50fa7b", "dashes": False, "arrows": "to"}
    if edge_type == "related":
        return {"color": "#8be9fd", "dashes": True, "arrows": "to"}
    if edge_type == "supports":
        return {"color": "#8affc1", "dashes": False, "arrows": "to"}
    if edge_type == "topic_index":
        return {"color": "#44475a", "dashes": False, "arrows": None}
    # default
    return {"color": "#cccccc", "dashes": False, "arrows": "to"}

# -----------------------------
# Render PyVis
# -----------------------------
def render_graph(G: nx.DiGraph, show_topics: List[int], show_claim_edges: bool):
    net = Network(
        height="750px",
        width="100%",
        bgcolor="#0e1117",
        font_color="white",
        directed=True
    )

    added_node_ids = set()

    # Add nodes: topic roots first (so they appear as boxes)
    for node, data in G.nodes(data=True):
        if not data.get("is_topic_root", False):
            continue
        # Only include topic roots that are selected
        if data["dominant_topic"] not in show_topics:
            continue
        net.add_node(
            node,
            label=data["label"],
            shape="box",
            color="#44475a",
            font={"size": 18},
            title=data.get("full_text", ""),
            group=f"topic_{data['dominant_topic']}"
        )
        added_node_ids.add(node)

    # Add claim nodes (only claims that intersect show_topics)
    for node, data in G.nodes(data=True):
        if data.get("is_topic_root", False):
            continue
        # filter claims that have any topic in show_topics
        node_topics = set(data.get("topics", []))
        if show_topics and not (node_topics & set(show_topics)):
            continue

        tooltip = data.get("full_text", "") + "\n\nTopics: " + ", ".join(map(str, data.get("topics", [])))
        dom = data.get("dominant_topic", -1)
        color = topic_color(dom) if dom >= 0 else "#888888"
        net.add_node(
            node,
            label=f"[{node}] {data.get('label', '')}",
            title=tooltip,
            color=color,
            shape="dot",
            size=10
        )
        added_node_ids.add(node)

    # Add edges:
    # - topic_index edges are added between topic roots and claims already because topic roots exist
    # - claim->claim edges are only added if show_claim_edges=True
    for u, v, d in G.edges(data=True):
        # Skip edges that connect to nodes not in this visualization subset
        if u not in added_node_ids or v not in added_node_ids:
            continue

        etype = d.get("edge_type", "related")
        style = edge_style_by_type(etype)
        # Optionally hide non-derived/contradicts edges if show_claim_edges is False
        if not show_claim_edges and etype not in ("topic_index",):
            continue

        # pyvis add_edge accepts IDs which can be strings or ints
        net.add_edge(u, v, **style)

    # Force physics options for nicer layout
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -25000,
          "centralGravity": 0.3,
          "springLength": 150,
          "springConstant": 0.02,
          "damping": 0.4
        },
        "minVelocity": 0.75
      }
    }
    """)

    # attach the set of added node ids for caller inspection
    net._added_node_ids = added_node_ids
    return net

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(layout="wide")
st.title("Semantic Claim Graph Explorer")

path = st.sidebar.text_input("Path to JSON", value="integrated_graph_revision.json")

if not os.path.exists(path):
    st.error("File not found.")
    st.stop()

data = load_graph(path)
G, topics = build_topic_graph(data)

st.sidebar.markdown("### Topic Filter")
selected_topics = st.sidebar.multiselect(
    "Show topics",
    topics,
    default=topics
)

if not selected_topics:
    st.warning("No topics selected — select at least one to visualize.")
    st.stop()

# Edge toggle
show_claim_edges = st.sidebar.checkbox("Show claim→claim edges (derived/related/contradicts)", value=True)

# Node inspection
st.sidebar.markdown("### Inspect Node")
# build list of numeric claim node ids only
claim_node_ids = [n for n, d in G.nodes(data=True) if not d.get("is_topic_root", False)]
claim_node_ids_sorted = sorted(claim_node_ids, key=lambda x: int(x) if isinstance(x, (int, str)) and str(x).isdigit() else str(x))
selected_node = st.sidebar.selectbox("Node ID", claim_node_ids_sorted)

if selected_node is not None:
    node = G.nodes[selected_node]
    st.sidebar.write("**Text**")
    st.sidebar.info(node.get("full_text", ""))
    st.sidebar.write("**Topics:**", node.get("topics", []))
    st.sidebar.write("**Provenance:**")
    st.sidebar.json(node.get("provenance", []))

# Legend
st.sidebar.markdown("### Legend")
st.sidebar.markdown(
    """
- Topic roots are **boxes** (dark gray).
- Claims are **dots**, colored by dominant topic.
- Edge color & style indicates relation:
  - **green solid** = derived (child / derived)
  - **red dashed** = contradicts
  - **cyan dashed** = related
  - **gray** = other / default
"""
)

# Render graph
net = render_graph(G, selected_topics, show_claim_edges)

# Save pyvis HTML into a temp file, display
with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
    tmp_path = tmp.name
    net.save_graph(tmp_path)

with open(tmp_path, "r", encoding="utf-8") as f:
    html = f.read()

st.components.v1.html(html, height=820, scrolling=True)

# cleanup
try:
    os.remove(tmp_path)
except Exception:
    pass

