# app/app.py
import re
import io
import csv
import json
import hashlib
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
PERSIST_PATH = APP_ROOT / "chroma_db"
PERSIST_DIR = str(PERSIST_PATH)
COLLECTION_NAME = "heart_pdfs"
INDEX_STATE_PATH = PERSIST_PATH / "index_state.json"

KB_PATH = KB_DIR / "materials_kb.csv"
MENTIONS_PATH = KB_DIR / "materials_mentions.csv"
DISCOVERED_MATERIALS_PATH = KB_DIR / "discovered_materials.csv"
CUSTOM_CONDITIONS_PATH = KB_DIR / "custom_conditions.json"

DATA_DIR.mkdir(exist_ok=True)
KB_DIR.mkdir(exist_ok=True)
PERSIST_PATH.mkdir(exist_ok=True)


def list_pdf_files():
    return sorted(DATA_DIR.glob("*.pdf"))


def get_pdf_inventory(pdf_files):
    inventory = []
    for path in pdf_files:
        stat = path.stat()
        inventory.append(
            {
                "name": path.name,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            }
        )
    return inventory


def compute_corpus_fingerprint(pdf_files):
    digest = hashlib.sha256()
    for path in pdf_files:
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8"))
    return digest.hexdigest()


def load_index_state():
    if not INDEX_STATE_PATH.exists():
        return {}
    try:
        return json.loads(INDEX_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_index_state(state):
    INDEX_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def load_custom_conditions():
    default_state = {
        "use_cases": [],
        "printer_types": [],
        "target_feels": [],
        "priorities": [],
    }
    if not CUSTOM_CONDITIONS_PATH.exists():
        return default_state
    try:
        loaded = json.loads(CUSTOM_CONDITIONS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return default_state

    if not isinstance(loaded, dict):
        return default_state

    merged = default_state.copy()
    for key in merged:
        value = loaded.get(key, [])
        if isinstance(value, list):
            merged[key] = [str(item).strip() for item in value if str(item).strip()]
    return merged


def save_custom_conditions(state):
    CUSTOM_CONDITIONS_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def load_ranked_kb(printer: str, feel: str, priority):
    if not KB_PATH.exists():
        return pd.DataFrame()
    try:
        kb_df = pd.read_csv(KB_PATH)
    except Exception:
        return pd.DataFrame()
    if kb_df.empty:
        return kb_df
    return kb_rank(kb_df, printer, feel, priority)


def parse_custom_values(raw_value: str):
    return [item.strip() for item in str(raw_value or "").split(",") if item.strip()]


def merge_unique_values(*groups):
    merged = []
    for group in groups:
        for item in group:
            if item not in merged:
                merged.append(item)
    return merged


def curated_material_aliases():
    return {
        "polyurethane": "PU",
        "pu": "PU",
        "polyether urethane": "PEU",
        "peu": "PEU",
        "polycarbonate-urethane": "PCU",
        "polycarbonate urethane": "PCU",
        "pcu": "PCU",
        "polytetrafluoroethylene": "PTFE",
        "ptfe": "PTFE",
        "polydimethylsiloxane": "PDMS",
        "pdms": "PDMS",
        "gelma": "GelMA",
        "gelatin methacryloyl": "GelMA",
        "peg-da": "PEG-DA",
        "pegda": "PEG-DA",
        "polyethylene glycol diacrylate": "PEG-DA",
        "hydroxyapatite": "HA",
        "ha": "HA",
        "beta-tcp": "β-TCP",
        "β-tcp": "β-TCP",
        "tricalcium phosphate": "β-TCP",
        "polycaprolactone": "PCL",
        "pcl": "PCL",
        "polylactic acid": "PLA",
        "pla": "PLA",
        "poly(lactic-co-glycolic acid)": "PLGA",
        "plga": "PLGA",
        "polyvinyl alcohol": "PVA",
        "pva": "PVA",
        "thermoplastic polyurethane": "TPU",
        "tpu": "TPU",
        "polyether ether ketone": "PEEK",
        "peek": "PEEK",
    }


def normalize_material_name(name: str) -> str:
    token = re.sub(r"\s+", " ", str(name or "").strip(" ,;:.()[]{}"))
    if not token:
        return ""

    alias = curated_material_aliases().get(token.lower())
    if alias:
        return alias

    if re.fullmatch(r"[A-Z0-9]+(?:[-/][A-Z0-9]+)*", token):
        return token

    if re.fullmatch(r"[a-z0-9]+(?:[-/][a-z0-9]+)+", token.lower()):
        return token.upper()

    if token.lower().startswith("poly"):
        return token

    return token


def split_material_tags(value: str):
    if not value:
        return []
    return [part.strip() for part in re.split(r"[;,]", str(value)) if part.strip()]


def discover_material_candidates(text: str, metadata_materials: str = ""):
    flat_text = " ".join((text or "").split())
    lower_text = flat_text.lower()
    candidates = set()

    for token in split_material_tags(metadata_materials):
        normalized = normalize_material_name(token)
        if normalized:
            candidates.add(normalized)

    alias_map = curated_material_aliases()
    for alias, canonical in alias_map.items():
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", lower_text):
            candidates.add(canonical)

    polymer_pattern = re.compile(
        r"\b("
        r"[A-Z]{2,6}(?:[-/][A-Z0-9]{1,6})*"
        r"|[A-Z][a-z]+(?:[- ][A-Z][a-z]+){0,3}"
        r"|poly[a-zA-Z0-9()/-]{3,}"
        r"|[A-Za-z]+(?:[- ][A-Za-z]+){0,2}\s(?:hydrogel|resin|elastomer|copolymer|composite|acrylate|methacrylate)"
        r")\b"
    )
    stopwords = {
        "DNA", "RNA", "UV", "SLA", "DLP", "FDM", "LAP", "I2959", "TABLE", "FIG",
        "REVIEW", "INTRODUCTION", "MATERIALS", "METHODS", "RESULTS", "DISCUSSION",
        "PolyJet", "Extrusion", "Bioprinting", "Printer", "Temperature", "Exposure",
    }
    generic_only = {"hydrogel", "resin", "bioink", "polymer", "copolymer", "composite", "elastomer"}

    for match in polymer_pattern.finditer(flat_text):
        token = normalize_material_name(match.group(1))
        token_lower = token.lower()
        if not token or token in stopwords:
            continue
        if token_lower in generic_only:
            continue
        if len(token) < 3:
            continue
        candidates.add(token)

    return sorted(candidates)


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
    generic_materials = {"hydrogel", "resin", "bioink", "polymer", "copolymer", "composite", "elastomer"}

    rows = []
    for n in nodes:
        text = n.get_text() or ""
        meta = n.metadata or {}
        file_name = meta.get("file_name") or meta.get("filename") or meta.get("source") or "unknown"
        flat_text = " ".join(text.split())
        discovered = discover_material_candidates(text, str(meta.get("materials", "")))
        for material in discovered:
            rows.append({
                "material": material,
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
        materials_tags=("materials_tags", join_unique),
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
def build_index(_ollama_model: str, corpus_fingerprint: str, rebuild_version: int):
    """
    Build persistent Chroma index from PDFs in ./data.
    Also generates kb/ CSVs, but DOES NOT display them.
    """
    pdf_files = list_pdf_files()
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    state = load_index_state()
    force_rebuild = rebuild_version > 0
    should_rebuild = (
        force_rebuild
        or state.get("corpus_fingerprint") != corpus_fingerprint
        or state.get("ollama_model") != _ollama_model
    )

    if should_rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    existing_count = 0
    try:
        existing_count = collection.count()
    except Exception:
        existing_count = 0

    if should_rebuild or existing_count == 0:
        docs = SimpleDirectoryReader(str(DATA_DIR)).load_data()

        splitter = SentenceSplitter(chunk_size=900, chunk_overlap=150)
        nodes = splitter.get_nodes_from_documents(docs)

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
            materials = discover_material_candidates(t)
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
        agg_df[["material", "mentions", "files", "materials_tags"]].to_csv(DISCOVERED_MATERIALS_PATH, index=False)

        index = VectorStoreIndex(nodes, storage_context=storage_context)
        save_index_state(
            {
                "collection_name": COLLECTION_NAME,
                "corpus_fingerprint": corpus_fingerprint,
                "indexed_at": datetime.now().isoformat(timespec="seconds"),
                "ollama_model": _ollama_model,
                "pdf_count": len(pdf_files),
                "chunk_count": len(nodes),
                "kb_material_count": int(len(agg_df)),
                "discovered_material_count": int(agg_df["material"].nunique()) if not agg_df.empty else 0,
                "mention_count": int(len(raw_df)),
                "rebuild_version": rebuild_version,
            }
        )
        return index

    return VectorStoreIndex.from_vector_store(vector_store=vector_store)


# ----------------------------
# App UI (chat-only front end)
# ----------------------------
st.set_page_config(page_title="Heart Materials AI (Local)", layout="wide")
st.title("🫀 Heart Materials AI (Local)")
st.caption(
    "Local evidence workspace for heart-model biomaterials recommendations. "
    "The app indexes uploaded PDFs, extracts a lightweight material knowledge base, "
    "and uses Ollama for grounded answers."
)

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
saved_custom_conditions = load_custom_conditions()

default_use_cases = [
    "Surgical simulation",
    "Surgical planning (rigid)",
    "Education model",
    "Flow loop testing",
    "Bioprinting / tissue engineering",
]
custom_use_cases = parse_custom_values(
    st.sidebar.text_input(
        "Add custom use-cases",
        value=", ".join(saved_custom_conditions["use_cases"]),
        placeholder="Patient-specific rehearsal, device testing",
    )
)
use_case_options = merge_unique_values(default_use_cases, custom_use_cases)
use_case = st.sidebar.selectbox("Use-case", use_case_options)

default_printers = [
    "SLA/DLP (resin)",
    "FDM (filament)",
    "Hydrogel extrusion / bioprinting",
    "Other",
]
custom_printers = parse_custom_values(
    st.sidebar.text_input(
        "Add custom printer types",
        value=", ".join(saved_custom_conditions["printer_types"]),
        placeholder="Material jetting, SLS, DIW",
    )
)
printer_options = merge_unique_values(default_printers, custom_printers)
printer = st.sidebar.selectbox("Printer type", printer_options)

default_feels = ["Rigid", "Flexible", "Very soft/compliant"]
custom_feels = parse_custom_values(
    st.sidebar.text_input(
        "Add custom target feels",
        value=", ".join(saved_custom_conditions["target_feels"]),
        placeholder="Elastic, tacky, anisotropic",
    )
)
feel_options = merge_unique_values(default_feels, custom_feels)
feel = st.sidebar.selectbox("Target feel", feel_options)

default_priorities = [
    "Anatomical fidelity",
    "Compliance",
    "Suturability/tear resistance",
    "Transparency",
    "Low cost",
    "Fast print time",
]
custom_priorities = parse_custom_values(
    st.sidebar.text_input(
        "Add custom priorities",
        value=", ".join(saved_custom_conditions["priorities"]),
        placeholder="Hemodynamic realism, sterilizability",
    )
)
priority_options = merge_unique_values(default_priorities, custom_priorities)
default_priority_selection = [p for p in ["Compliance", "Anatomical fidelity"] if p in priority_options]
priority = st.sidebar.multiselect(
    "Top priorities",
    priority_options,
    default=default_priority_selection,
)

updated_custom_conditions = {
    "use_cases": custom_use_cases,
    "printer_types": custom_printers,
    "target_feels": custom_feels,
    "priorities": custom_priorities,
}
if updated_custom_conditions != saved_custom_conditions:
    save_custom_conditions(updated_custom_conditions)

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

pdf_files = list_pdf_files()
pdf_inventory = get_pdf_inventory(pdf_files)
corpus_fingerprint = compute_corpus_fingerprint(pdf_files)
index_state = load_index_state()
corpus_synced = bool(pdf_files) and index_state.get("corpus_fingerprint") == corpus_fingerprint
st.sidebar.caption(f"Detected {len(pdf_files)} PDF(s) in ./data")

st.sidebar.header("Index")
if st.sidebar.button("Build / Rebuild Index"):
    st.session_state["rebuild_version"] = st.session_state.get("rebuild_version", 0) + 1
    build_index.clear()
    st.sidebar.info("Rebuilding index…")

rebuild_version = st.session_state.get("rebuild_version", 0)
index = build_index(ollama_model, corpus_fingerprint, rebuild_version) if pdf_files else None
index_state = load_index_state()
corpus_synced = bool(pdf_files) and index_state.get("corpus_fingerprint") == corpus_fingerprint

if not pdf_files:
    st.sidebar.info("Upload PDFs to create a searchable corpus.")
elif corpus_synced:
    st.sidebar.success(
        f"Index synced: {index_state.get('chunk_count', 0)} chunks from {index_state.get('pdf_count', len(pdf_files))} PDFs."
    )
else:
    st.sidebar.warning("The current PDFs do not match the saved index state yet.")

ranked_kb = load_ranked_kb(printer, feel, priority)

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("PDFs", len(pdf_files))
metric_col2.metric("Indexed chunks", index_state.get("chunk_count", 0) if corpus_synced else 0)
metric_col3.metric("Ranked materials", len(ranked_kb))
metric_col4.metric("KB mentions", index_state.get("mention_count", 0) if corpus_synced else 0)

if pdf_files:
    st.caption(
        f"Last indexed: {index_state.get('indexed_at', 'not yet indexed')} | "
        f"Model: {index_state.get('ollama_model', ollama_model)}"
    )

# Chat session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

chat_tab, corpus_tab = st.tabs(["Chat", "Corpus & KB"])

def make_prompt(query: str) -> str:
    # Pull KB hints silently (do NOT show them)
    kb_top = ""
    if not ranked_kb.empty:
        kb_top = ranked_kb.head(8)[["material", "kb_score", "printers_tags", "settings_tags", "files", "mentions"]].to_csv(index=False)

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

with chat_tab:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_msg = st.chat_input("Ask about materials, print settings, or what to avoid…")

    if user_msg:
        st.session_state.messages.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        if index is None:
            assistant_text = "No PDFs found yet. Upload PDFs in the sidebar, then click **Build / Rebuild Index**."
            st.session_state.messages.append({"role": "assistant", "content": assistant_text})
            with st.chat_message("assistant"):
                st.markdown(assistant_text)
            st.stop()

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

        st.session_state.messages.append({"role": "assistant", "content": assistant_text})

with corpus_tab:
    st.subheader("Corpus inventory")
    if pdf_inventory:
        st.dataframe(pd.DataFrame(pdf_inventory), use_container_width=True, hide_index=True)
    else:
        st.info("No PDFs uploaded yet.")

    st.subheader("Ranked material candidates")
    if ranked_kb.empty:
        st.info("Build the index to generate the material knowledge base.")
    else:
        st.dataframe(
            ranked_kb[
                ["material", "kb_score", "mentions", "files", "printers_tags", "settings_tags"]
            ].head(15),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Latest retrieved evidence")
    if st.session_state.last_sources:
        for s in st.session_state.last_sources:
            st.markdown(f"**[S{s['i']}] {s['file']}**")
            st.write(s["snippet"] + "…")
    else:
        st.caption("Ask a question in the chat tab to populate the latest retrieval snippets.")
