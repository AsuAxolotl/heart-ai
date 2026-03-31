# app/app.py
import re
import io
import csv
import json
import hashlib
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import httpx
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
DEFAULT_LLM_TIMEOUT_SECONDS = 900.0
DEFAULT_SIMILARITY_TOP_K = 5
NON_MATERIAL_TERMS = {
    "the", "this", "that", "these", "those", "for", "from", "into", "with", "without",
    "author", "authors", "author manuscript", "manuscript", "page", "han page", "pmc",
    "figure", "fig", "table", "review", "abstract", "introduction", "discussion",
    "results", "methods", "materials", "conclusion", "supplementary", "copyright",
    "october", "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "november", "december", "compr physiol", "physiol",
    "received", "accepted", "published", "preprint", "journal", "article",
}
MATERIAL_KEYWORDS = {
    "poly", "gel", "hydrogel", "resin", "elastomer", "copolymer", "composite",
    "acrylate", "methacrylate", "urethane", "siloxane", "silicone", "alginate",
    "gelatin", "gelma", "collagen", "elastin", "fibrin", "chitosan", "cellulose",
    "albumin", "bioink", "tcp", "hydroxyapatite", "ceramic",
}
PRINTER_TAG_ALIASES = {
    "fdm/fff": "FDM",
    "fff": "FDM",
    "fdm": "FDM",
    "sla/dlp": "Vat photopolymerization (SLA/DLP)",
    "dlp/sla": "Vat photopolymerization (SLA/DLP)",
    "sla": "SLA",
    "dlp": "DLP",
    "polyjet": "PolyJet",
    "extrusion": "Extrusion / bioprinting",
    "bioprinting": "Extrusion / bioprinting",
    "3d printer": "General 3D printing",
    "3d printers": "General 3D printing",
    "3d printing": "General 3D printing",
    "3d printing systems": "General 3D printing",
    "photocrosslinking": "Photocrosslinking",
    "photocrosslinkable": "Photocrosslinking",
    "uv": "Photocrosslinking",
}
PRINTER_DESCRIPTIONS = {
    "FDM": "Filament extrusion process commonly used for thermoplastics such as PLA, PCL, and PEEK.",
    "SLA": "Laser-based vat photopolymerization process that cures liquid resin layer by layer with high detail.",
    "DLP": "Projector-based vat photopolymerization process that cures entire resin layers at once.",
    "Vat photopolymerization (SLA/DLP)": "Resin-based light-curing family that is useful for high-detail anatomical models and photocurable materials.",
    "PolyJet": "Material-jetting process that deposits and UV-cures droplets of photopolymer for multi-material, high-detail prints.",
    "Extrusion / bioprinting": "Nozzle-based deposition process used for hydrogels, bioinks, pastes, and other soft materials.",
    "General 3D printing": "General corpus tag indicating 3D-printing knowledge without a single clearly specified printing modality.",
    "Photocrosslinking": "Light-triggered curing workflow often paired with resin or hydrogel systems rather than a standalone printer family.",
}

KB_PATH = KB_DIR / "materials_kb.csv"
MENTIONS_PATH = KB_DIR / "materials_mentions.csv"
DISCOVERED_MATERIALS_PATH = KB_DIR / "discovered_materials.csv"
CUSTOM_CONDITIONS_PATH = KB_DIR / "custom_conditions.json"
FEEDBACK_DIR = APP_ROOT / "feedback"
FEEDBACK_RAW_DIR = FEEDBACK_DIR / "raw"
FEEDBACK_RULES_DIR = FEEDBACK_DIR / "rules"

DATA_DIR.mkdir(exist_ok=True)
KB_DIR.mkdir(exist_ok=True)
PERSIST_PATH.mkdir(exist_ok=True)
FEEDBACK_RAW_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_RULES_DIR.mkdir(parents=True, exist_ok=True)


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


def fetch_url_text(url: str, timeout: int = 60):
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "heart-materials-ai/1.0 (local research tool)",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="ignore")


def fetch_url_bytes(url: str, timeout: int = 120):
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "heart-materials-ai/1.0 (local research tool)",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def search_pmc_articles(query: str, retmax: int = 5):
    encoded_query = urllib.parse.quote_plus(query)
    search_url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        f"?db=pmc&term={encoded_query}&retmax={retmax}&retmode=json"
    )
    payload = json.loads(fetch_url_text(search_url))
    id_list = payload.get("esearchresult", {}).get("idlist", [])
    if not id_list:
        return []

    summary_url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        f"?db=pmc&id={','.join(id_list)}&retmode=json"
    )
    summary_payload = json.loads(fetch_url_text(summary_url))
    results = []
    for pmc_numeric_id in id_list:
        summary = summary_payload.get("result", {}).get(str(pmc_numeric_id), {})
        if not summary:
            continue
        article_ids = summary.get("articleids", [])
        pmcid = next(
            (item.get("value") for item in article_ids if item.get("idtype") == "pmc"),
            f"PMC{pmc_numeric_id}",
        )
        results.append(
            {
                "pmcid": pmcid if str(pmcid).startswith("PMC") else f"PMC{pmcid}",
                "title": summary.get("title", ""),
            }
        )
    return results


def find_pmc_pdf_url(pmcid: str):
    article_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
    html = fetch_url_text(article_url)

    meta_match = re.search(r'citation_pdf_url"\s+content="([^"]+)"', html, re.I)
    if meta_match:
        return html_unescape_url(meta_match.group(1))

    href_match = re.search(r'href="([^"]+/pdf/[^"]+\.pdf[^"]*)"', html, re.I)
    if href_match:
        href = html_unescape_url(href_match.group(1))
        if href.startswith("http"):
            return href
        return urllib.parse.urljoin(article_url, href)

    fallback_match = re.search(r'href="([^"]+/pdf/[^"]*)"', html, re.I)
    if fallback_match:
        href = html_unescape_url(fallback_match.group(1))
        if href.startswith("http"):
            return href
        return urllib.parse.urljoin(article_url, href)

    raise ValueError(f"Could not find a downloadable PDF link for {pmcid}.")


def html_unescape_url(url: str):
    return url.replace("&amp;", "&").strip()


def download_pmc_pdf(pmcid: str, title: str = ""):
    pdf_url = find_pmc_pdf_url(pmcid)
    pdf_bytes = fetch_url_bytes(pdf_url)
    safe_title = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("_")
    filename = f"{pmcid}_{safe_title[:80]}.pdf" if safe_title else f"{pmcid}.pdf"
    destination = DATA_DIR / filename
    destination.write_bytes(pdf_bytes)
    return destination


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


def split_feedback_segments(comment: str):
    segments = []
    for part in re.split(r"[.\n;]+", str(comment or "")):
        cleaned = part.strip()
        if cleaned:
            segments.append(cleaned)
    return segments[:5]


def infer_feedback_themes(comment: str, query_route: str, answer_mode: str, score: int):
    text = str(comment or "").lower()
    themes = []
    theme_keywords = {
        "conciseness": ["too long", "verbose", "shorter", "concise", "brief"],
        "evidence": ["source", "citation", "evidence", "support", "grounded"],
        "clarity": ["clear", "confusing", "clarity", "understand", "organized", "structure"],
        "accuracy": ["wrong", "incorrect", "accurate", "accuracy", "hallucinated"],
        "speed": ["slow", "faster", "quick", "timeout", "lag"],
        "inventory": ["inventory", "list", "catalog", "too many materials"],
        "recommendation": ["recommend", "best", "alternative", "tradeoff"],
        "avoidance": ["avoid", "risk", "warning", "unsafe"],
        "extraction": ["parameter", "extract", "table", "not specified"],
    }
    for theme, keywords in theme_keywords.items():
        if any(keyword in text for keyword in keywords):
            themes.append(theme)

    if not themes:
        if answer_mode.startswith("Extract"):
            themes.append("extraction")
        elif query_route == "avoid":
            themes.append("avoidance")
        elif query_route == "recommend":
            themes.append("recommendation")
        else:
            themes.append("general")

    if score <= 2 and "accuracy" not in themes:
        themes.append("accuracy")
    return list(dict.fromkeys(themes))


def build_feedback_rule_summary(score: int, themes, query_route: str, answer_mode: str, comment_segments):
    route_label = "extract" if answer_mode.startswith("Extract") else query_route
    if score >= 4:
        if "evidence" in themes:
            return f"Preserve the current {route_label} pattern of grounding answers in explicit evidence."
        if "clarity" in themes:
            return f"Preserve the current {route_label} structure because the answer format is working well."
        return f"Keep the current {route_label} response style as a positive baseline."

    rule_parts = []
    if "conciseness" in themes:
        rule_parts.append("Prioritize the top few findings before longer detail.")
    if "evidence" in themes or "accuracy" in themes:
        rule_parts.append("Make evidence support explicit before making claims.")
    if "clarity" in themes:
        rule_parts.append("Use clearer sectioning and simpler phrasing.")
    if "inventory" in themes:
        rule_parts.append("Cap inventory-style outputs and clean noisy entries before display.")
    if "recommendation" in themes:
        rule_parts.append("Surface tradeoffs and alternatives alongside the primary recommendation.")
    if "avoidance" in themes:
        rule_parts.append("State concrete risks and safer alternatives when advising against an option.")
    if "extraction" in themes:
        rule_parts.append("Prefer compact parameter tables and say not specified only when necessary.")
    if not rule_parts and comment_segments:
        rule_parts.append(comment_segments[0])
    if not rule_parts:
        rule_parts.append("Improve answer quality without changing the app's built-in rules.")
    return " ".join(dict.fromkeys(rule_parts))


def persist_feedback_artifacts(response_record: dict, score: int, comment: str):
    timestamp = datetime.now().isoformat(timespec="seconds")
    fingerprint_source = f"{timestamp}|{response_record.get('question', '')}|{score}|{comment}"
    feedback_id = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()[:12]
    comment_segments = split_feedback_segments(comment)
    themes = infer_feedback_themes(
        comment,
        response_record.get("query_route", "default"),
        response_record.get("mode", ""),
        score,
    )
    rule_summary = build_feedback_rule_summary(
        score,
        themes,
        response_record.get("query_route", "default"),
        response_record.get("mode", ""),
        comment_segments,
    )

    raw_feedback = {
        "feedback_id": feedback_id,
        "created_at": timestamp,
        "score": score,
        "comment": comment.strip(),
        "comment_segments": comment_segments,
        "response_record": response_record,
    }
    concise_rule = {
        "rule_id": feedback_id,
        "created_at": timestamp,
        "advisory_only": True,
        "source_feedback_id": feedback_id,
        "query_route": response_record.get("query_route", "default"),
        "mode": response_record.get("mode", ""),
        "score": score,
        "themes": themes,
        "rule_summary": rule_summary,
        "question_digest": str(response_record.get("question", ""))[:240],
        "comment_digest": comment_segments[:3],
    }

    raw_path = FEEDBACK_RAW_DIR / f"{timestamp.replace(':', '-')}_{feedback_id}.json"
    rule_path = FEEDBACK_RULES_DIR / f"{timestamp.replace(':', '-')}_{feedback_id}.json"
    raw_path.write_text(json.dumps(raw_feedback, indent=2), encoding="utf-8")
    rule_path.write_text(json.dumps(concise_rule, indent=2), encoding="utf-8")
    return feedback_id, concise_rule


def load_ranked_kb(printer: str, feel: str, priority):
    if not KB_PATH.exists():
        return pd.DataFrame()
    try:
        kb_df = pd.read_csv(KB_PATH)
    except Exception:
        return pd.DataFrame()
    if kb_df.empty:
        return kb_df
    if "material_groups" not in kb_df.columns:
        kb_df = add_material_groups(kb_df)
    return kb_rank(kb_df, printer, feel, priority)


def load_material_inventory():
    if not KB_PATH.exists():
        return pd.DataFrame()
    try:
        kb_df = pd.read_csv(KB_PATH)
    except Exception:
        return pd.DataFrame()
    if kb_df.empty:
        return kb_df
    return kb_df.sort_values(["mentions", "material"], ascending=[False, True]).reset_index(drop=True)


def load_printer_inventory():
    if not KB_PATH.exists():
        return pd.DataFrame(columns=["printer_system", "mentions", "materials", "files"])
    try:
        kb_df = pd.read_csv(KB_PATH)
    except Exception:
        return pd.DataFrame(columns=["printer_system", "mentions", "materials", "files"])
    if kb_df.empty or "printers_tags" not in kb_df.columns:
        return pd.DataFrame(columns=["printer_system", "mentions", "materials", "files"])

    rows = []
    for row in kb_df.itertuples():
        printer_tags = split_material_tags(getattr(row, "printers_tags", ""))
        files = str(getattr(row, "files", ""))
        material = str(getattr(row, "material", ""))
        mentions = int(getattr(row, "mentions", 0) or 0)
        for printer_tag in printer_tags:
            normalized_printer = normalize_printer_tag(printer_tag)
            if not is_plausible_printer_tag(normalized_printer):
                continue
            rows.append(
                {
                    "printer_system": normalized_printer,
                    "mentions": mentions,
                    "material": material,
                    "files": files,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["printer_system", "mentions", "materials", "files"])

    printer_df = pd.DataFrame(rows)
    aggregated = (
        printer_df.groupby("printer_system", as_index=False)
        .agg(
            mentions=("mentions", "sum"),
            materials=("material", join_unique),
            files=("files", join_unique),
        )
        .sort_values(["mentions", "printer_system"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return aggregated


def parse_custom_values(raw_value: str):
    return [item.strip() for item in str(raw_value or "").split(",") if item.strip()]


def merge_unique_values(*groups):
    merged = []
    for group in groups:
        for item in group:
            if item not in merged:
                merged.append(item)
    return merged


def looks_like_recommendation_question(query: str) -> bool:
    q = str(query or "").strip().lower()
    if not q:
        return False

    recommendation_terms = [
        "best", "better", "recommend", "recommended", "recommendation", "suitable",
        "ideal", "good for", "use for", "should i use", "what should", "tradeoff",
        "compare", "comparison", "pros and cons", "for an", "for a", "for the",
    ]
    return any(term in q for term in recommendation_terms)


def looks_like_avoidance_question(query: str) -> bool:
    q = str(query or "").strip().lower()
    if not q:
        return False

    avoidance_terms = [
        "avoid", "shouldn't use", "should not use", "not recommend", "would not recommend",
        "unsafe", "bad for", "poor choice", "worst", "failure mode", "failure modes",
        "what not to use", "what should i avoid", "risks", "downsides",
    ]
    return any(term in q for term in avoidance_terms)


def classify_query_route(query: str) -> str:
    if is_material_inventory_question(query):
        return "material_inventory"
    if is_printer_inventory_question(query):
        return "printer_inventory"
    if looks_like_avoidance_question(query):
        return "avoid"
    if looks_like_recommendation_question(query):
        return "recommend"
    return "default"


def is_material_inventory_question(query: str) -> bool:
    q = str(query or "").strip().lower()
    if not q:
        return False
    if looks_like_recommendation_question(q):
        return False

    material_terms = ["material", "materials"]
    inventory_terms = ["inventory", "corpus", "knowledge", "know about", "available", "list", "show"]
    inventory_phrases = [
        "what materials",
        "which materials",
        "list materials",
        "show materials",
        "materials do you have",
        "materials are in",
        "what do you know about",
        "what is in your corpus",
        "what's in your corpus",
        "inventory of materials",
    ]
    return any(phrase in q for phrase in inventory_phrases) or (
        any(term in q for term in material_terms) and any(term in q for term in inventory_terms)
    )


def is_printer_inventory_question(query: str) -> bool:
    q = str(query or "").strip().lower()
    if not q:
        return False
    if looks_like_recommendation_question(q):
        return False

    printer_terms = [
        "printer",
        "printers",
        "printing system",
        "printing systems",
        "3d printing system",
        "3d printing systems",
        "printer types",
        "print technologies",
        "fabrication systems",
        "machines",
    ]
    inventory_phrases = [
        "what printers",
        "which printers",
        "what printing systems",
        "which printing systems",
        "what 3d printing systems",
        "which 3d printing systems",
        "what printer types",
        "what print technologies",
        "do you know about printers",
        "knowledge of 3d printing systems",
    ]
    inventory_terms = ["inventory", "corpus", "knowledge", "know about", "available", "list", "show"]
    return any(phrase in q for phrase in inventory_phrases) or (
        any(term in q for term in printer_terms) and any(term in q for term in inventory_terms)
    )


def build_material_inventory_answer(material_inventory_df: pd.DataFrame, index_state, pdf_files):
    if material_inventory_df.empty:
        return (
            "I do not have a built material inventory yet. Upload PDFs and rebuild the index first, "
            "then I can answer inventory questions directly from the corpus."
        )

    material_names = material_inventory_df["material"].astype(str).tolist()
    top_materials = material_inventory_df.head(12)
    top_lines = [
        f"- **{row.material}**: {int(row.mentions)} mentions across {len(str(row.files).split('; '))} source file(s)"
        for row in top_materials.itertuples()
    ]

    remaining_count = max(len(material_names) - len(top_materials), 0)
    remainder_line = (
        f"\nThere are {remaining_count} additional materials in the indexed inventory beyond the top list above."
        if remaining_count
        else ""
    )

    return (
        f"I currently have indexed knowledge of **{len(material_names)} materials** across **{len(pdf_files)} PDFs** "
        f"and **{index_state.get('chunk_count', 0)} chunks**.\n\n"
        f"Top materials by mentions:\n" + "\n".join(top_lines) +
        f"\n\nFull inventory:\n{', '.join(material_names)}"
        f"{remainder_line}"
    )


def build_printer_inventory_answer(printer_inventory_df: pd.DataFrame, index_state, pdf_files):
    if printer_inventory_df.empty:
        return (
            "I do not have a built 3D-printing-system inventory yet. Upload PDFs and rebuild the index first, "
            "then I can answer printer inventory questions directly from the corpus."
        )

    printer_names = printer_inventory_df["printer_system"].astype(str).tolist()
    top_printers = printer_inventory_df.head(12)
    top_lines = [
        f"- **{row.printer_system}**: {describe_printer_system(row.printer_system)} "
        f"Linked mentions: {int(row.mentions)} | example materials: {summarize_examples(row.materials)}"
        for row in top_printers.itertuples()
    ]

    remaining_count = max(len(printer_names) - len(top_printers), 0)
    remainder_line = (
        f"\nThere are {remaining_count} additional printing systems in the indexed inventory beyond the top list above."
        if remaining_count
        else ""
    )

    return (
        f"I currently have indexed knowledge of **{len(printer_names)} printing systems / printer tags** across "
        f"**{len(pdf_files)} PDFs** and **{index_state.get('chunk_count', 0)} chunks**.\n\n"
        f"Top printing systems by linked mentions:\n" + "\n".join(top_lines) +
        f"\n\nFull printer inventory:\n{', '.join(printer_names)}"
        f"{remainder_line}"
    )


def build_recommendation_prompt(query: str, kb_top: str) -> str:
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


def build_avoidance_prompt(query: str, kb_top: str) -> str:
    return f"""
You are a biomaterials risk-screening assistant.

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
1) Focus first on what should be avoided or treated cautiously.
2) Ground claims in retrieved text. Cite sources as [S1], [S2], etc.
3) Separate into:
   A) Materials/processes to avoid or treat cautiously
   B) Why they are risky or mismatched
   C) Safer alternatives or mitigation strategies supported by the PDFs
   D) Evidence gaps or uncertainty
4) If the PDFs do not support a strong warning, say that plainly.
"""


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


def normalize_printer_tag(tag: str) -> str:
    token = re.sub(r"\s+", " ", str(tag or "").strip(" ,;:.()[]{}"))
    if not token:
        return ""

    normalized = PRINTER_TAG_ALIASES.get(token.lower())
    if normalized:
        return normalized

    if token.lower().startswith("3d print"):
        return "General 3D printing"

    return token


def describe_printer_system(printer_system: str) -> str:
    description = PRINTER_DESCRIPTIONS.get(str(printer_system or "").strip())
    if description:
        return description
    return "Printing-related tag found in the indexed corpus."


def is_plausible_printer_tag(tag: str) -> bool:
    token = str(tag or "").strip()
    if not token:
        return False

    token_lower = token.lower()
    if token_lower in {"nan", "none", "null", "unknown"}:
        return False

    allowed_keywords = {
        "fdm", "sla", "dlp", "polyjet", "extrusion", "bioprint", "photocrosslink",
        "printing", "printer", "vat",
    }
    return any(keyword in token_lower for keyword in allowed_keywords)


def summarize_examples(values: str, limit: int = 5) -> str:
    examples = []
    for value in split_material_tags(values):
        normalized = normalize_material_name(value)
        if not is_plausible_material_candidate(normalized):
            continue
        if normalized not in examples:
            examples.append(normalized)
        if len(examples) >= limit:
            break
    if not examples:
        return "no clean material examples yet"
    return ", ".join(examples)


def is_plausible_material_candidate(token: str) -> bool:
    if not token:
        return False

    token = str(token).strip()
    token_lower = token.lower()

    if token_lower in NON_MATERIAL_TERMS:
        return False
    if any(part in NON_MATERIAL_TERMS for part in token_lower.split()):
        return False
    if len(token) < 3:
        return False
    if len(token.split()) > 4:
        return False

    alias_values = {value.lower() for value in curated_material_aliases().values()}
    alias_keys = set(curated_material_aliases().keys())
    if token_lower in alias_keys or token_lower in alias_values:
        return True

    if token.isupper():
        return bool(re.search(r"[-/\d]", token)) or token_lower in alias_values

    if token_lower.startswith("poly"):
        return True

    if any(keyword in token_lower for keyword in MATERIAL_KEYWORDS):
        return True

    if re.search(r"[-/\d]", token) and any(ch.isalpha() for ch in token):
        return True

    return False


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
        if is_plausible_material_candidate(normalized):
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
    generic_only = {"hydrogel", "resin", "bioink", "polymer", "copolymer", "composite", "elastomer"}

    for match in polymer_pattern.finditer(flat_text):
        token = normalize_material_name(match.group(1))
        token_lower = token.lower()
        if not is_plausible_material_candidate(token):
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


def infer_material_groups(material: str, printers_tags: str, settings_tags: str, materials_tags: str, example_snippets: str):
    text = " ".join(
        [
            str(material or ""),
            str(printers_tags or ""),
            str(settings_tags or ""),
            str(materials_tags or ""),
            str(example_snippets or ""),
        ]
    ).lower()
    material_lower = str(material or "").lower()
    groups = []

    flexible_markers = {
        "pu", "peu", "pcu", "tpu", "silicone", "pdms", "elastomer", "flexible", "viscoelastic",
        "compliant", "soft", "elastic",
    }
    rigid_markers = {
        "pla", "peek", "ha", "β-tcp", "beta-tcp", "ceramic", "rigid", "stiff", "hard",
        "polypropylene", "ptfe",
    }
    soft_hydrogel_markers = {
        "gelma", "alginate", "collagen", "fibrin", "hydrogel", "bioink", "peg-da", "ha-ma",
        "gelatin", "very soft",
    }
    photocurable_markers = {
        "sla", "dlp", "photocrosslink", "uv", "resin", "methacrylate", "acrylate", "photoinitiator",
        "peg-da", "gelma",
    }
    extrusion_markers = {
        "extrusion", "bioprint", "nozzle", "viscosity", "bioink", "hydrogel",
    }
    fdm_markers = {
        "fdm", "filament", "fff", "pla", "pcl", "peek", "plga", "pva",
    }
    composite_markers = {
        "composite", "ha/", "/ha", "/tcp", "tcp", "fiber", "cf-", "ceramic",
    }

    if any(marker in text or marker == material_lower for marker in flexible_markers):
        groups.append("flexible")
    if any(marker in text or marker == material_lower for marker in rigid_markers):
        groups.append("rigid")
    if any(marker in text or marker == material_lower for marker in soft_hydrogel_markers):
        groups.append("soft_compliant")
    if any(marker in text for marker in photocurable_markers):
        groups.append("photocurable")
    if any(marker in text for marker in extrusion_markers):
        groups.append("extrusion_compatible")
    if any(marker in text for marker in fdm_markers):
        groups.append("fdm_compatible")
    if any(marker in text for marker in composite_markers):
        groups.append("composite")

    if not groups:
        groups.append("general")
    return "; ".join(dict.fromkeys(groups))


def add_material_groups(agg_df: pd.DataFrame):
    agg_columns = [
        "material", "mentions", "files", "printers_tags", "settings_tags", "materials_tags",
        "example_snippets", "material_groups"
    ]
    if agg_df.empty:
        return pd.DataFrame(columns=agg_columns)

    grouped = agg_df.copy()
    grouped["material_groups"] = grouped.apply(
        lambda row: infer_material_groups(
            row.get("material", ""),
            row.get("printers_tags", ""),
            row.get("settings_tags", ""),
            row.get("materials_tags", ""),
            row.get("example_snippets", ""),
        ),
        axis=1,
    )
    return grouped[agg_columns]


def select_grouped_candidates(materials_kb_df: pd.DataFrame, printer: str, feel: str):
    if materials_kb_df.empty or "material_groups" not in materials_kb_df.columns:
        return materials_kb_df

    desired_groups = []
    feel_lower = str(feel or "").lower()
    printer_lower = str(printer or "").lower()

    if "rigid" in feel_lower:
        desired_groups.append("rigid")
    elif "soft" in feel_lower or "compliant" in feel_lower:
        desired_groups.extend(["soft_compliant", "flexible"])
    elif "flexible" in feel_lower:
        desired_groups.append("flexible")

    if "sla" in printer_lower or "dlp" in printer_lower or "resin" in printer_lower:
        desired_groups.append("photocurable")
    elif "fdm" in printer_lower or "filament" in printer_lower:
        desired_groups.append("fdm_compatible")
    elif "hydrogel" in printer_lower or "bioprint" in printer_lower or "extrusion" in printer_lower:
        desired_groups.append("extrusion_compatible")

    desired_groups = list(dict.fromkeys(desired_groups))
    if not desired_groups:
        return materials_kb_df

    filtered = materials_kb_df[
        materials_kb_df["material_groups"].fillna("").apply(
            lambda value: any(group in split_material_tags(str(value).replace("; ", ";")) for group in desired_groups)
        )
    ].copy()
    return filtered if not filtered.empty else materials_kb_df


def clean_tag_value(value: str, validator=None):
    cleaned = []
    for tag in split_material_tags(value):
        normalized = re.sub(r"\s+", " ", str(tag).strip())
        if not normalized:
            continue
        if validator and not validator(normalized):
            continue
        if normalized not in cleaned:
            cleaned.append(normalized)
    return "; ".join(cleaned)


def clean_material_kb(raw_df: pd.DataFrame, agg_df: pd.DataFrame):
    raw_columns = [
        "material", "file", "printers_tags", "settings_tags", "materials_tags", "context_snippet"
    ]
    agg_columns = [
        "material", "mentions", "files", "printers_tags", "settings_tags", "materials_tags",
        "example_snippets", "material_groups"
    ]

    if raw_df.empty:
        return pd.DataFrame(columns=raw_columns), pd.DataFrame(columns=agg_columns)

    cleaned = raw_df.copy()
    cleaned["material"] = cleaned["material"].apply(normalize_material_name)
    cleaned = cleaned[cleaned["material"].apply(is_plausible_material_candidate)].copy()

    generic_materials = {"hydrogel", "resin", "bioink", "polymer", "copolymer", "composite", "elastomer"}
    cleaned = cleaned[~cleaned["material"].str.lower().isin(generic_materials)].copy()

    cleaned["file"] = cleaned["file"].astype(str).str.strip()
    cleaned["printers_tags"] = cleaned["printers_tags"].apply(clean_tag_value)
    cleaned["settings_tags"] = cleaned["settings_tags"].apply(clean_tag_value)
    cleaned["materials_tags"] = cleaned["materials_tags"].apply(
        lambda value: clean_tag_value(
            value,
            validator=lambda token: is_plausible_material_candidate(normalize_material_name(token)),
        )
    )
    cleaned["context_snippet"] = cleaned["context_snippet"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    cleaned = cleaned.drop_duplicates(
        subset=["material", "file", "context_snippet"]
    ).reset_index(drop=True)

    if cleaned.empty:
        return pd.DataFrame(columns=raw_columns), pd.DataFrame(columns=agg_columns)

    rebuilt_agg = cleaned.groupby("material", as_index=False).agg(
        mentions=("material", "count"),
        files=("file", lambda s: "; ".join(sorted(set(s.astype(str))))),
        printers_tags=("printers_tags", join_unique),
        settings_tags=("settings_tags", join_unique),
        materials_tags=("materials_tags", join_unique),
        example_snippets=("context_snippet", top_snips),
    )
    rebuilt_agg = rebuilt_agg.sort_values(["mentions", "material"], ascending=[False, True]).reset_index(drop=True)
    rebuilt_agg = add_material_groups(rebuilt_agg)
    return cleaned[raw_columns], rebuilt_agg[agg_columns]


def extract_material_kb_rows(nodes):
    """
    Returns:
      raw_df: one row per mention
      agg_df: one row per material (consolidated)
    """
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
    return clean_material_kb(raw_df, pd.DataFrame())


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

    df = select_grouped_candidates(materials_kb_df.copy(), printer, feel).copy()
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
        raw_df, agg_df = clean_material_kb(raw_df, agg_df)
        raw_df.to_csv(MENTIONS_PATH, index=False)
        agg_df.to_csv(KB_PATH, index=False)
        agg_df[["material", "mentions", "files", "materials_tags", "material_groups"]].to_csv(
            DISCOVERED_MATERIALS_PATH, index=False
        )

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
llm_timeout_seconds = st.sidebar.number_input(
    "LLM timeout (seconds)",
    min_value=60,
    max_value=3600,
    value=int(DEFAULT_LLM_TIMEOUT_SECONDS),
    step=30,
)
retrieval_top_k = st.sidebar.slider(
    "Retrieved chunks per answer",
    min_value=2,
    max_value=8,
    value=DEFAULT_SIMILARITY_TOP_K,
)

llm = Ollama(model=ollama_model, request_timeout=float(llm_timeout_seconds))
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

st.sidebar.header("PubMed Central")
pmc_query = st.sidebar.text_input(
    "Search PMC for open-access papers",
    placeholder="heart valve biomaterials 3D printing",
)
pmc_retmax = st.sidebar.slider("PMC papers to download", min_value=1, max_value=10, value=3)
if st.sidebar.button("Search and download PMC PDFs"):
    if not pmc_query.strip():
        st.sidebar.warning("Enter a PubMed Central search query first.")
    else:
        try:
            pmc_results = search_pmc_articles(pmc_query.strip(), retmax=pmc_retmax)
            if not pmc_results:
                st.sidebar.warning("No PubMed Central papers matched that query.")
            else:
                downloaded_files = []
                failed_downloads = []
                for result in pmc_results:
                    try:
                        downloaded_path = download_pmc_pdf(result["pmcid"], result.get("title", ""))
                        downloaded_files.append(downloaded_path.name)
                    except Exception as exc:
                        failed_downloads.append(f"{result['pmcid']}: {exc}")

                if downloaded_files:
                    st.sidebar.success(f"Downloaded {len(downloaded_files)} PMC PDF(s) to ./data.")
                    build_index.clear()
                    with st.sidebar.expander("Downloaded PMC files"):
                        for filename in downloaded_files:
                            st.write(filename)
                if failed_downloads:
                    with st.sidebar.expander("PMC download issues"):
                        for failure in failed_downloads:
                            st.write(failure)
        except Exception as exc:
            st.sidebar.error(f"PMC download failed: {exc}")

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
material_inventory_df = load_material_inventory()
printer_inventory_df = load_printer_inventory()

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
if "latest_response_record" not in st.session_state:
    st.session_state.latest_response_record = None
if "feedback_status" not in st.session_state:
    st.session_state.feedback_status = ""

chat_tab, corpus_tab = st.tabs(["Chat", "Corpus & KB"])

def make_prompt(query: str, query_route: str) -> str:
    # Pull KB hints silently (do NOT show them)
    kb_top = ""
    if not ranked_kb.empty:
        prompt_columns = ["material", "kb_score", "material_groups", "printers_tags", "settings_tags", "files", "mentions"]
        kb_top = ranked_kb.head(8)[prompt_columns].to_csv(index=False)

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

    if query_route == "avoid":
        return build_avoidance_prompt(query, kb_top)

    return build_recommendation_prompt(query, kb_top)

def collect_sources(response):
    sources = []
    if hasattr(response, "source_nodes") and response.source_nodes:
        for i, sn in enumerate(response.source_nodes, start=1):
            meta = getattr(sn.node, "metadata", {}) or {}
            filename = meta.get("file_name") or meta.get("filename") or meta.get("source") or "unknown"
            snippet = (sn.node.get_text() or "").replace("\n", " ")[:700]
            sources.append({"i": i, "file": filename, "snippet": snippet})
    return sources


def build_terminal_failure_message(reason: str, query_route: str, llm_timeout_seconds: int, answer_mode: str) -> str:
    if reason == "timeout":
        return (
            f"I couldn't complete this answer because the model timed out after {llm_timeout_seconds} seconds. "
            "This is a terminal failure for the current run, so I am not returning a low-confidence guess. "
            "Try a smaller model, fewer retrieved chunks, or a narrower question."
        )

    if reason == "insufficient_knowledge":
        if answer_mode.startswith("Extract"):
            route_hint = (
                "The indexed PDFs do not appear to contain enough explicit parameter-level evidence to support a trustworthy extraction."
            )
        elif query_route in {"recommend", "avoid", "default"}:
            route_hint = (
                "The indexed PDFs do not appear to contain enough direct evidence to support a trustworthy recommendation."
            )
        else:
            route_hint = "The indexed corpus does not contain enough direct evidence for a trustworthy answer."
        return (
            f"{route_hint} This is a terminal failure for the current query, so I am stopping instead of guessing. "
            "Try uploading more relevant PDFs or asking a narrower evidence-backed question."
        )

    return "I couldn't produce a trustworthy answer for this query, so I am stopping instead of guessing."


def response_has_sufficient_evidence(answer_text: str, sources, query_route: str, answer_mode: str) -> bool:
    text = str(answer_text or "").strip().lower()
    if not text:
        return False

    if query_route in {"material_inventory", "printer_inventory"}:
        return True

    if len(sources or []) == 0:
        return False

    general_insufficiency_markers = [
        "i don't know",
        "i do not know",
        "not enough information",
        "insufficient information",
        "insufficient evidence",
        "evidence is weak",
        "cannot determine",
        "can't determine",
        "unable to determine",
        "not in the pdfs",
        "not in the retrieved text",
        "no direct evidence",
    ]
    general_marker_hits = sum(marker in text for marker in general_insufficiency_markers)

    if answer_mode.startswith("Extract"):
        extraction_failure_markers = [
            "no parameters found",
            "no explicit parameters found",
            "no extraction possible",
        ]
        if any(marker in text for marker in extraction_failure_markers):
            return False
        if general_marker_hits >= 2:
            return False
        if "|" not in answer_text and len(text) < 60:
            return False
        return True

    if query_route == "avoid":
        avoidance_markers = [
            "no strong warning",
            "no direct warning",
            "no clear reason to avoid",
        ]
        if general_marker_hits >= 2 or all(marker in text for marker in avoidance_markers[:2]):
            return False
        return len(text) >= 100

    if query_route in {"recommend", "default"}:
        recommendation_markers = [
            "not specified",
            "unclear from the pdfs",
            "weak evidence",
        ]
        marker_hits = general_marker_hits + sum(marker in text for marker in recommendation_markers)
        if marker_hits >= 2:
            return False
        return len(text) >= 100

    return len(text) >= 80


def run_query_with_fallback(index, llm, prompt: str, retrieval_top_k: int):
    primary_qe = index.as_query_engine(
        llm=llm,
        similarity_top_k=retrieval_top_k,
        response_mode="compact",
    )
    try:
        return primary_qe.query(prompt), False
    except httpx.ReadTimeout:
        fallback_qe = index.as_query_engine(
            llm=llm,
            similarity_top_k=max(2, min(3, retrieval_top_k)),
            response_mode="simple_summarize",
        )
        return fallback_qe.query(prompt), True

with chat_tab:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_msg = st.chat_input("Ask about materials, print settings, or what to avoid…")

    if user_msg:
        st.session_state.feedback_status = ""
        st.session_state.messages.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)
        query_route = classify_query_route(user_msg)

        if index is None:
            assistant_text = "No PDFs found yet. Upload PDFs in the sidebar, then click **Build / Rebuild Index**."
            st.session_state.messages.append({"role": "assistant", "content": assistant_text})
            with st.chat_message("assistant"):
                st.markdown(assistant_text)
            st.stop()

        with st.chat_message("assistant"):
            if query_route == "material_inventory":
                assistant_text = build_material_inventory_answer(material_inventory_df, index_state, pdf_files)
                st.markdown(assistant_text)
                inventory_sources = []
                for i, row in enumerate(material_inventory_df.head(8).itertuples(), start=1):
                    inventory_sources.append(
                        {
                            "i": i,
                            "file": "materials_kb.csv",
                            "snippet": f"{row.material}: {int(row.mentions)} mentions | files: {row.files}",
                        }
                    )
                st.session_state.last_sources = inventory_sources
            elif query_route == "printer_inventory":
                assistant_text = build_printer_inventory_answer(printer_inventory_df, index_state, pdf_files)
                st.markdown(assistant_text)
                printer_sources = []
                for i, row in enumerate(printer_inventory_df.head(8).itertuples(), start=1):
                    printer_sources.append(
                        {
                            "i": i,
                            "file": "materials_kb.csv",
                            "snippet": (
                                f"{row.printer_system}: {int(row.mentions)} linked mentions | "
                                f"example materials: {summarize_examples(row.materials)}"
                            ),
                        }
                    )
                st.session_state.last_sources = printer_sources
            else:
                with st.spinner("Searching your PDFs…"):
                    prompt = make_prompt(user_msg, query_route)
                    try:
                        response, used_timeout_fallback = run_query_with_fallback(index, llm, prompt, retrieval_top_k)
                    except httpx.ReadTimeout:
                        assistant_text = build_terminal_failure_message("timeout", query_route, llm_timeout_seconds, mode)
                        st.error(assistant_text)
                        st.session_state.last_sources = []
                        st.session_state.messages.append({"role": "assistant", "content": assistant_text})
                        st.stop()

                    assistant_text = str(response)
                    if used_timeout_fallback:
                        st.caption("Used timeout fallback mode with fewer chunks to complete this answer.")
                    sources = collect_sources(response)
                    if not response_has_sufficient_evidence(assistant_text, sources, query_route, mode):
                        assistant_text = build_terminal_failure_message(
                            "insufficient_knowledge", query_route, llm_timeout_seconds, mode
                        )
                        st.warning(assistant_text)
                        st.session_state.last_sources = sources
                        st.session_state.messages.append({"role": "assistant", "content": assistant_text})
                        st.stop()

                    st.markdown(assistant_text)
                    st.session_state.last_sources = sources

                    if sources:
                        with st.expander("Sources used (snippets)"):
                            for s in sources:
                                st.markdown(f"**[S{s['i']}] {s['file']}**")
                                st.write(s["snippet"] + "…")

            timestamp = datetime.now().isoformat(timespec="seconds")
            sources = st.session_state.last_sources
            csv_buf = io.StringIO()
            export = {
                "timestamp": timestamp,
                "mode": mode,
                "query_route": query_route,
                "use_case": use_case,
                "printer_type": printer,
                "target_feel": feel,
                "priorities": "; ".join(priority),
                "question": user_msg,
                "answer": assistant_text,
                "sources": " || ".join([f"S{s['i']}|{s['file']}|{s['snippet']}" for s in sources]) if sources else "none",
            }
            st.session_state.latest_response_record = export
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

    latest_response_record = st.session_state.get("latest_response_record")
    if latest_response_record:
        st.divider()
        st.subheader("Response Feedback")
        if st.session_state.feedback_status:
            st.caption(st.session_state.feedback_status)

        with st.form("response_feedback_form", clear_on_submit=False):
            score = st.slider("Score this response", min_value=1, max_value=5, value=3)
            comment = st.text_area(
                "Comment on what should improve or stay the same",
                placeholder="Example: Keep the evidence structure, but make the answer shorter and clearer.",
            )
            submitted = st.form_submit_button("Save feedback")

        if submitted:
            feedback_id, concise_rule = persist_feedback_artifacts(latest_response_record, score, comment)
            st.session_state.feedback_status = (
                f"Saved feedback {feedback_id}. Advisory rule stored separately with themes: "
                f"{', '.join(concise_rule['themes'])}."
            )
            st.success(st.session_state.feedback_status)

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
                ["material", "kb_score", "material_groups", "mentions", "files", "printers_tags", "settings_tags"]
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

