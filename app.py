# app/app.py
import re
import io
import csv
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

import chromadb
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter


# ----------------------------
# Paths / Constants
# ----------------------------
APP_ROOT = Path(__file__).resolve().parent.parent  # project root
DATA_DIR = APP_ROOT / "data"
KB_DIR = APP_ROOT / "kb"
PERSIST_DIR = str(APP_ROOT / "chroma_db")

KB_PATH = KB_DIR / "materials_kb.csv"
MENTIONS_PATH = KB_DIR / "materials_mentions.csv"

DATA_DIR.mkdir(exist_ok=True)
KB_DIR.mkdir(exist_ok=True)


# ----------------------------
# KB helpers (silent backend)
# ----------------------------
def join_unique(series, sep="; "):
    vals = []
    for x in series.dropna().astype(str):
        parts = re.split(r"[;,]", x)
        for p in parts:
            p = p.strip()
            if p and p not in vals:
                vals.append(p)
    return sep.join(vals)


def top_snips(series, k=3):
    snips = []
    for s in series.dropna().astype(str):
        s = s.strip()
        if s and s not in snips:
            snips.append(s)
        if len(snips) >= k:
            break
    return " || ".join(snips)


def extract_material_kb_rows(nodes):
    """
    Returns:
      raw_df: one row per mention
      agg_df: one row per material (consolidated)
    """
    known_materials = [
        "PEG-DA", "alginate", "GelMA", "PCL", "PLA", "PLGA", "PVA",
        "PU", "PEU", "PCU", "PTFE", "silicone", "TPU", "PDMS",
        "hydrogel", "resin", "PEEK", "HA", "β-TCP", "bioink"
    ]

    canonical_map = {
        "polyurethane": "PU",
        "pu": "PU",
        "polyether urethane": "PEU",
        "peu": "PEU",
        "polycarbonate-urethane": "PCU",
        "pcu": "PCU",
        "polytetrafluoroethylene": "PTFE",
        "ptfe": "PTFE",
        "polydimethylsiloxane": "PDMS",
        "pdms": "PDMS",
        "gelma": "GelMA",
        "peg-da": "PEG-DA",
        "pegda": "PEG-DA",
        "hydroxyapatite": "HA",
        "ha": "HA",
        "beta-tcp": "β-TCP",
        "β-tcp": "β-TCP",
    }

    generic_materials = {"hydrogel", "resin", "bioink"}

    rows = []
    for n in nodes:
        text = n.get_text() or ""
        meta = n.metadata or {}
        file_name = meta.get("file_name") or meta.get("filename") or meta.get("source") or "unknown"
        flat_text = " ".join(text.split())

        # direct matches
        for mat in known_materials:
            if mat.lower() in flat_text.lower():
                canonical = canonical_map.get(mat.lower(), mat)
                rows.append({
                    "material": canonical,
                    "file": file_name,
                    "printers_tags": str(meta.get("printers", "")),
                    "settings_tags": str(meta.get("settings", "")),
                    "materials_tags": str(meta.get("materials", "")),
                    "context_snippet": flat_text[:700],
                })

        # synonym matches
        for syn, canon in canonical_map.items():
            if syn in flat_text.lower():
                rows.append({
                    "material": canon,
                    "file": file_name,
                    "printers_tags": str(meta.get("printers", "")),
                    "settings_tags": str(meta.get("settings", "")),
                    "materials_tags": str(meta.get("materials", "")),
                    "context_snippet": flat_text[:700],
                })

    raw_df = pd.DataFrame(rows, columns=[
        "material", "file", "printers_tags", "settings_tags", "materials_tags", "context_snippet"
    ])

    if raw_df.empty:
        agg_df = pd.DataFrame(columns=[
            "material", "mentions", "files", "printers_tags", "settings_tags", "example_snippets"
        ])
        return raw_df, agg_df

    raw_for_agg = raw_df[~raw_df["material"].str.lower().isin(generic_materials)].copy()
    if raw_for_agg.empty:
        agg_df = pd.DataFrame(columns=[
            "material", "mentions", "files", "printers_tags", "settings_tags", "example_snippets"
        ])
        return raw_df, agg_df

    agg = raw_for_agg.groupby("material", as_index=False).agg(
        mentions=("material", "count"),
        files=("file", lambda s: "; ".join(sorted(set(s.astype(str))))),
        printers_tags=("printers_tags", join_unique),
        settings_tags=("settings_tags", join_unique),
        example_snippets=("context_snippet", top_snips),
    )
    agg_df = agg.sort_values("mentions", ascending=False).reset_index(drop=True)
    return raw_df, agg_df


def kb_rank(materials_kb_df: pd.DataFrame, printer: str, feel: str, priority):
    """
    Simple scoring over consolidated KB.
    (Use-case is still used in the LLM prompt; KB rank is just a hint.)
    """
    p = printer.lower()
    priorities = [x.lower() for x in priority]

    def score_row(r):
        s = 0.0
        printers = str(r.get("printers_tags", "")).lower()
        settings = str(r.get("settings_tags", "")).lower()
        mat = str(r.get("material", "")).lower()

        # Printer compatibility boost
        if "sla" in p or "dlp" in p:
            if any(k in printers for k in ["sla", "dlp", "photocrosslink", "uv"]):
                s += 3.0
        elif "fdm" in p:
            if "fdm" in printers or "extrusion" in printers:
                s += 3.0
        elif "hydrogel" in p or "bioprint" in p:
            if any(k in printers for k in ["bioprint", "extrusion", "photocrosslink"]):
                s += 3.0

        # settings signal
        if settings.strip():
            s += 0.5

        # feel heuristic
        if feel.lower().startswith("rigid"):
            if mat in ["peek", "pla", "pcl", "ha", "β-tcp", "beta-tcp"]:
                s += 1.5
        else:
            if mat in ["tpu", "pu", "silicone", "pdms", "peg-da", "gelma", "alginate", "hydrogel"]:
                s += 1.5

        # priorities heuristic
        if "anatomical fidelity" in priorities and any(k in printers for k in ["sla", "dlp", "photocrosslink"]):
            s += 1.0
        if "compliance" in priorities and any(k in settings for k in ["viscosity", "temp", "temperature", "modulus", "stiffness"]):
            s += 0.8
        if "suturability/tear resistance" in priorities and any(k in settings for k in ["tough", "tear", "elongation"]):
            s += 0.6

        # mentions as weak signal
        try:
            s += min(float(r.get("mentions", 0)) / 100.0, 1.0)
        except Exception:
            pass

        return s

    df = materials_kb_df.copy()
    df["kb_score"] = df.apply(score_row, axis=1)
    return df.sort_values("kb_score", ascending=False)


# ----------------------------
# Index build (cached)
# ----------------------------
@st.cache_resource
def build_index(_ollama_model: str):
    """
    Build persistent Chroma index from PDFs in ./data.
    Also generates kb/ CSVs, but DOES NOT display them.
    """
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection("heart_pdfs")

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    docs = SimpleDirectoryReader(str(DATA_DIR)).load_data()

    splitter = SentenceSplitter(chunk_size=900, chunk_overlap=150)
    nodes = splitter.get_nodes_from_documents(docs)

    material_re = re.compile(
        r"\b(PEG-DA|PEGDA|alginate|GelMA|PU|PEU|PCU|PTFE|silicone|PVA|TPU|PDMS|hydrogel|resin|PEEK|HA|hydroxyapatite|β-TCP|beta-?TCP|bioink|photocrosslink\w*)\b",
        re.I,
    )
    printer_re = re.compile(
        r"\b(SLA|DLP|FDM|PolyJet|bioprint\w*|extrusion|3D print\w*|photocrosslink\w*)\b",
        re.I,
    )
    setting_re = re.compile(
        r"\b(layer|layers|nozzle|temp(?:erature)?|speed|infill|cure|exposure|UV|wavelength|viscosity|photoinitiator|LAP|I2959)\b",
        re.I,
    )

    for n in nodes:
        t = n.get_text() or ""
        materials = sorted(set(m.group(0) for m in material_re.finditer(t)))[:30]
        printers = sorted(set(p.group(0) for p in printer_re.finditer(t)))[:30]
        settings = sorted(set(s.group(0) for s in setting_re.finditer(t)))[:30]

        # Chroma requires flat metadata values (str/int/float/None)
        n.metadata = n.metadata or {}
        n.metadata["materials"] = ", ".join(materials)
        n.metadata["printers"] = ", ".join(printers)
        n.metadata["settings"] = ", ".join(settings)

    # Build/update KB CSVs (silent)
    raw_df, agg_df = extract_material_kb_rows(nodes)
    raw_df.to_csv(MENTIONS_PATH, index=False)
    agg_df.to_csv(KB_PATH, index=False)

    return VectorStoreIndex(nodes, storage_context=storage_context)


# ----------------------------
# App UI (chat-only front end)
# ----------------------------
st.set_page_config(page_title="Heart Materials AI (Local)", layout="wide")
st.title("🫀 Heart Materials AI (Local)")

# Sidebar (keep controls here)
st.sidebar.header("Model (Ollama)")
ollama_model = st.sidebar.text_input("Ollama model name", value="llama3.1:8b")
st.sidebar.caption("Run `ollama list` to see installed models.")

llm = Ollama(model=ollama_model, request_timeout=360.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.llm = llm

st.sidebar.header("Answer style")
mode = st.sidebar.radio(
    "Mode",
    ["Recommend (strong outputs)", "Extract (settings/steps)"],
    index=0,
)

st.sidebar.header("Context (used to steer answers)")
use_case = st.sidebar.selectbox(
    "Use-case",
    ["Surgical simulation", "Surgical planning (rigid)", "Education model", "Flow loop testing", "Bioprinting / tissue engineering"],
)
printer = st.sidebar.selectbox(
    "Printer type",
    ["SLA/DLP (resin)", "FDM (filament)", "Hydrogel extrusion / bioprinting", "Other"],
)
feel = st.sidebar.selectbox(
    "Target feel",
    ["Rigid", "Flexible", "Very soft/compliant"],
)
priority = st.sidebar.multiselect(
    "Top priorities",
    ["Anatomical fidelity", "Compliance", "Suturability/tear resistance", "Transparency", "Low cost", "Fast print time"],
    default=["Compliance", "Anatomical fidelity"],
)

st.sidebar.header("PDFs")
uploads = st.sidebar.file_uploader(
    "Upload one or more PDF protocols/papers",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploads:
    for f in uploads:
        (DATA_DIR / f.name).write_bytes(f.getbuffer())
    st.sidebar.success(f"Saved {len(uploads)} file(s) to ./data")

pdf_files = sorted(DATA_DIR.glob("*.pdf"))
st.sidebar.caption(f"Detected {len(pdf_files)} PDF(s) in ./data")

st.sidebar.header("Index")
if st.sidebar.button("Build / Rebuild Index"):
    build_index.clear()
    st.sidebar.info("Rebuilding index…")
    _ = build_index(ollama_model)
    st.sidebar.success("Index ready!")

index = build_index(ollama_model) if len(pdf_files) > 0 else None

# Chat session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_msg = st.chat_input("Ask about materials, print settings, or what to avoid…")

def make_prompt(query: str) -> str:
    # Pull KB hints silently (do NOT show them)
    kb_top = ""
    if KB_PATH.exists():
        try:
            kb_df = pd.read_csv(KB_PATH)
            ranked = kb_rank(kb_df, printer, feel, priority)
            kb_top = ranked.head(8)[["material", "kb_score", "printers_tags", "settings_tags", "files", "mentions"]].to_csv(index=False)
        except Exception:
            kb_top = ""

    if mode.startswith("Extract"):
        return f"""
You are extracting fabrication / printing parameters from retrieved documents.

Top candidate materials from local KB (use as hints; only cite if supported by retrieved text):
{kb_top}

User context (steer extraction):
- Use-case: {use_case}
- Printer type: {printer}
- Target feel: {feel}
- Priorities: {", ".join(priority)}

User question:
{query}

Rules:
1) ONLY report a parameter if it is explicitly stated in retrieved text.
2) If not stated, write "not specified".
3) Focus on: resin/bioink composition, photoinitiator, wavelength/UV, exposure, layer height, nozzle size, print speed, temperature, viscosity/rheology, crosslinking, post-cure, washing, sterilization.
4) Output format EXACTLY:
- Parameter | Value | Material/Context | Source #
"""
    else:
        return f"""
You are a biomaterials recommendation assistant.

Top candidate materials from local KB (use as hints; only cite if supported by retrieved text):
{kb_top}

User constraints:
- Use-case: {use_case}
- Printer type: {printer}
- Target feel: {feel}
- Priorities: {", ".join(priority)}

User question:
{query}

Output requirements:
1) Give a strong, specific answer. Use bullets and short sections.
2) Ground claims in retrieved text. Cite sources as [S1], [S2], etc.
3) Separate into:
   A) Best options (ranked, with rationale + tradeoffs)
   B) What to avoid / failure modes (if evidence exists)
   C) Concrete next steps (what to test next, what knobs to tune)
4) If something isn’t in the PDFs, say so plainly.
"""

def collect_sources(response):
    sources = []
    if hasattr(response, "source_nodes") and response.source_nodes:
        for i, sn in enumerate(response.source_nodes, start=1):
            meta = getattr(sn.node, "metadata", {}) or {}
            filename = meta.get("file_name") or meta.get("filename") or meta.get("source") or "unknown"
            snippet = (sn.node.get_text() or "").replace("\n", " ")[:700]
            sources.append({"i": i, "file": filename, "snippet": snippet})
    return sources

if user_msg:
    # show user message
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    if index is None:
        assistant_text = "No PDFs found yet. Upload PDFs in the sidebar, then click **Build / Rebuild Index**."
        st.session_state.messages.append({"role": "assistant", "content": assistant_text})
        with st.chat_message("assistant"):
            st.markdown(assistant_text)
        st.stop()

    # query engine
    qe = index.as_query_engine(llm=llm, similarity_top_k=8)

    with st.chat_message("assistant"):
        with st.spinner("Searching your PDFs…"):
            prompt = make_prompt(user_msg)
            response = qe.query(prompt)

            assistant_text = str(response)
            st.markdown(assistant_text)

            sources = collect_sources(response)
            st.session_state.last_sources = sources

            if sources:
                with st.expander("Sources used (snippets)"):
                    for s in sources:
                        st.markdown(f"**[S{s['i']}] {s['file']}**")
                        st.write(s["snippet"] + "…")

            # download result
            timestamp = datetime.now().isoformat(timespec="seconds")
            export = {
                "timestamp": timestamp,
                "mode": mode,
                "use_case": use_case,
                "printer_type": printer,
                "target_feel": feel,
                "priorities": "; ".join(priority),
                "question": user_msg,
                "answer": assistant_text,
                "sources": " || ".join([f"S{s['i']}|{s['file']}|{s['snippet']}" for s in sources]) if sources else "none",
            }
            csv_buf = io.StringIO()
            writer = csv.DictWriter(csv_buf, fieldnames=list(export.keys()))
            writer.writeheader()
            writer.writerow(export)

            st.download_button(
                "⬇️ Download this answer (CSV)",
                data=csv_buf.getvalue().encode("utf-8"),
                file_name=f"heart_materials_ai_{timestamp.replace(':','-')}.csv",
                mime="text/csv",
            )

    # save assistant message
    st.session_state.messages.append({"role": "assistant", "content": assistant_text})