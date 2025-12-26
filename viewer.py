# viewer.py
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
def load_graph(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# -----------------------------
# Build graph (defensive)
# -----------------------------
def build_topic_graph(data):
    G = nx.DiGraph()

    # Defensive access into structure
    graph_obj = data.get("graph", {})
    nodes = graph_obj.get("nodes", [])
    edges = graph_obj.get("edges", [])

    topic_to_nodes = defaultdict(list)

    for node in nodes:
        # Extract id (keep original type)
        nid = node.get("id")
        text = node.get("text", "")
        prov = node.get("provenance") or []

        # safe topic extraction: provenance entries might be heterogeneous
        topic_set = set()
        for p in prov:
            if isinstance(p, dict) and "topic_idx" in p:
                try:
                    topic_set.add(int(p["topic_idx"]))
                except Exception:
                    # non-int topic indices are ignored
                    pass
        topics = sorted(list(topic_set))
        dominant_topic = topics[0] if topics else -1

        # normalize status and revised_by
        raw_status = node.get("status", "ACTIVE")
        status = str(raw_status).lower() if raw_status is not None else "active"
        revised_by = node.get("revised_by")
        if revised_by is None:
            revised_by = []
        elif isinstance(revised_by, (int, str)):
            revised_by = [revised_by]
        elif isinstance(revised_by, list):
            # ensure elements are strings/ints
            revised_by = [r for r in revised_by if r is not None]
        else:
            revised_by = []

        # numeric-ish fields
        confidence = float(node.get("confidence") or 0.0)
        support = float(node.get("support_weight") or node.get("support", 0.0))
        attack = float(node.get("attack_weight") or node.get("attack", 0.0))

        G.add_node(
            nid,
            label=(text[:80] + ("..." if len(text) > 80 else "")),
            full_text=text,
            topics=topics,
            dominant_topic=dominant_topic,
            provenance=prov,
            status=status,
            confidence=confidence,
            support_weight=support,
            attack_weight=attack,
            revised_by=revised_by,
            is_topic_root=False,
        )

        for t in topics:
            topic_to_nodes[t].append(nid)

    # create topic root virtual nodes (string IDs to avoid collision with numeric claim IDs)
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
        for nid in nids:
            G.add_edge(root_id, nid, edge_type="topic_index", score=1.0)

    # Add edges defensively
    for e in edges:
        # support several key name variants
        src = e.get("source", e.get("src", e.get("s")))
        dst = e.get("target", e.get("dst", e.get("t")))
        typ = e.get("type", e.get("edge_type", "related"))
        score = e.get("score", 1.0)
        try:
            score = float(score)
        except Exception:
            score = 1.0

        # If nodes missing, add minimal node placeholders
        if src not in G.nodes:
            G.add_node(src, label=str(src), full_text=str(src), topics=[], dominant_topic=-1, provenance=[], status="active", is_topic_root=False)
        if dst not in G.nodes:
            G.add_node(dst, label=str(dst), full_text=str(dst), topics=[], dominant_topic=-1, provenance=[], status="active", is_topic_root=False)

        # normalize edge type names (the trainer used 'revises' for supersession)
        if typ == "revises":
            # make it explicit for the viewer
            typ = "supersedes"
        G.add_edge(src, dst, edge_type=typ, score=score)

    return G, sorted(topic_to_nodes.keys())

# -----------------------------
# Styling helpers
# -----------------------------
def topic_color(topic: int):
    palette = [
        "#ff5555", "#50fa7b", "#8be9fd",
        "#bd93f9", "#f1fa8c", "#ff79c6",
        "#ffaa00", "#8affc1", "#c7a1ff", "#ffd480"
    ]
    try:
        return palette[int(topic) % len(palette)]
    except Exception:
        return "#888888"

def edge_style_by_type(edge_type: str):
    if edge_type == "contradicts":
        return {"color": "#ff5555", "dashes": True, "arrows": "to"}
    if edge_type == "derived":
        return {"color": "#50fa7b", "dashes": False, "arrows": "to"}
    if edge_type == "supersedes":
        return {"color": "#ffaa00", "dashes": False, "arrows": "to", "width": 3}
    if edge_type == "related":
        return {"color": "#8be9fd", "dashes": True, "arrows": "to"}
    if edge_type == "supports":
        return {"color": "#8affc1", "dashes": False, "arrows": "to"}
    if edge_type == "topic_index":
        return {"color": "#44475a", "dashes": False, "arrows": None}
    return {"color": "#cccccc", "dashes": False, "arrows": "to"}

# -----------------------------
# Render PyVis graph
# -----------------------------
def render_graph(G: nx.DiGraph, show_topics: List[int], show_claim_edges: bool, hide_superseded: bool):
    net = Network(
        height="750px",
        width="100%",
        bgcolor="#0e1117",
        font_color="white",
        directed=True
    )

    added_node_ids = set()

    # add topic roots first
    for node, data in G.nodes(data=True):
        if not data.get("is_topic_root", False):
            continue
        if data["dominant_topic"] not in show_topics:
            continue
        net.add_node(
            node,
            label=data["label"],
            shape="box",
            color="#44475a",
            font={"size": 18},
            title=data.get("full_text", "")
        )
        added_node_ids.add(node)

    # add claim nodes
    for node, data in G.nodes(data=True):
        if data.get("is_topic_root", False):
            continue
        # topic filter
        node_topics = set(data.get("topics", []))
        if show_topics and not (node_topics & set(show_topics)):
            continue

        status = data.get("status", "active")
        is_superseded = (status == "superseded")
        if hide_superseded and is_superseded:
            # skip adding visually
            continue

        dom = data.get("dominant_topic", -1)
        color = topic_color(dom) if dom >= 0 else "#888888"
        # fade color if superseded
        if is_superseded:
            color = "#555555"

        tooltip = (
            f"<b>Text:</b> {data.get('full_text','')}<br><br>"
            f"<b>Status:</b> {status}<br>"
            f"<b>Confidence:</b> {data.get('confidence', 0):.3f}<br>"
            f"<b>Support:</b> {data.get('support_weight', 0):.3f}<br>"
            f"<b>Attack:</b> {data.get('attack_weight', 0):.3f}<br>"
            f"<b>Revised by:</b> {', '.join(map(str, data.get('revised_by', [])))}<br>"
        )

        net.add_node(
            node,
            label=f"[{node}] {data.get('label','')}",
            title=tooltip,
            color=color,
            shape="dot",
            size=8 if is_superseded else 12,
        )
        added_node_ids.add(node)

    # add edges
    for u, v, d in G.edges(data=True):
        if u not in added_node_ids or v not in added_node_ids:
            continue
        etype = d.get("edge_type", "related")
        if not show_claim_edges and etype not in ("topic_index",):
            continue
        style = edge_style_by_type(etype)
        # net.add_edge accepts IDs of mixed types
        net.add_edge(u, v, **style)

    # nicer physics
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
        "minVelocity": 0.5
      }
    }
    """)
    # attach helper
    net._added_node_ids = added_node_ids
    return net

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(layout="wide")
st.title("Claim Graph (Revision-aware) Explorer")

path = st.sidebar.text_input("Path to JSON", value="integrated_graph_revision.json")

if not os.path.exists(path):
    st.error("File not found: " + path)
    st.stop()

data = load_graph(path)
G, topics = build_topic_graph(data)

st.sidebar.markdown("### Topic Filter")
selected_topics = st.sidebar.multiselect("Show topics", topics, default=topics)

if not selected_topics:
    st.warning("No topics selected — select at least one to visualize.")
    st.stop()

show_claim_edges = st.sidebar.checkbox("Show claim→claim edges (derived/related/contradicts/supersedes)", value=True)
hide_superseded = st.sidebar.checkbox("Hide superseded nodes", value=False)

# Build mapping for inspector: display string -> node id
claim_nodes = [(n, d) for n, d in G.nodes(data=True) if not d.get("is_topic_root", False)]
# only nodes that match topic filter
filter_nodes = []
for n, d in claim_nodes:
    if set(d.get("topics", [])) & set(selected_topics):
        filter_nodes.append((n, d))

# create display strings sorted
display_map = {}
display_list = []
for n, d in sorted(filter_nodes, key=lambda kv: (str(kv[0]))):
    short = d.get("label", "")[:80]
    disp = f"[{n}] {short}"
    display_map[disp] = n
    display_list.append(disp)

st.sidebar.markdown("### Inspect Node")
selected_display = st.sidebar.selectbox("Node", ["(none)"] + display_list)

if selected_display and selected_display != "(none)":
    selected_node_id = display_map[selected_display]
    nd = G.nodes[selected_node_id]
    st.sidebar.markdown("**Full text**")
    st.sidebar.info(nd.get("full_text", ""))
    st.sidebar.markdown("**Metadata**")
    st.sidebar.write(f"Status: {nd.get('status')}")
    st.sidebar.write(f"Confidence: {nd.get('confidence'):.4f}")
    st.sidebar.write(f"Support weight: {nd.get('support_weight'):.4f}")
    st.sidebar.write(f"Attack weight: {nd.get('attack_weight'):.4f}")
    st.sidebar.write("Revised by: " + ", ".join(map(str, nd.get("revised_by", []))))
    st.sidebar.markdown("**Provenance (first 10)**")
    prov = nd.get("provenance", []) or []
    st.sidebar.json(prov[:10])

# Render and show
net = render_graph(G, selected_topics, show_claim_edges, hide_superseded)

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
