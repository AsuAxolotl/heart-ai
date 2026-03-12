# Heart Materials AI

Heart Materials AI is a local AI research assistant designed to help biomedical engineers and biomaterials researchers analyze scientific literature and identify candidate materials for **3D-printed heart models and cardiovascular biomaterials applications**.

The system allows users to upload research papers and protocols (PDFs) and query them using a **retrieval-augmented large language model (RAG)** that returns answers grounded in the uploaded documents.

Unlike generic chatbots, this tool supports **constraint-aware material recommendations**, enabling researchers to explore candidate materials based on:

- Use case
- Printer type
- Target mechanical behavior
- Research priorities

All inference runs **locally**, enabling secure analysis of unpublished protocols and proprietary research data.

---

## Overview

Modern biomaterials research requires synthesizing large volumes of literature across materials science, biomedical engineering, and additive manufacturing.

Heart Materials AI addresses this challenge by combining:

- Local large language models
- Vector search across research papers
- Constraint-aware reasoning

The system retrieves relevant sections from uploaded papers and uses them to generate answers supported by citations.

This enables rapid exploration of questions such as:

- What materials have been used for 3D printed heart valve scaffolds?
- Which hydrogels are suitable for compliant cardiovascular models?
- What polymers are commonly used in prosthetic heart valves?
- What materials are compatible with SLA or hydrogel printing?

---

## Key Features

### Literature-Grounded Answers
Responses are generated only from retrieved document content and include supporting source snippets.

### PDF Knowledge Base
Researchers can upload scientific papers, protocols, and reviews to build a searchable knowledge base.

### Constraint-Aware Recommendations
Users can specify design constraints such as use case, printer type, target mechanical properties, and design priorities. The system then recommends candidate materials supported by literature.

### Local AI Inference
All language model inference runs locally via **Ollama**, enabling secure analysis of sensitive research data.

### Source Transparency
Every response includes supporting text extracted from the source documents.

---

## Example Use Cases

| Domain | Description |
|---|---|
| Biomaterials Research | Identify candidate materials for cardiovascular scaffolds |
| Surgical Simulation | Explore materials for compliant anatomical heart models |
| Tissue Engineering | Investigate hydrogels and polymer scaffolds for heart valve engineering |
| Literature Exploration | Rapidly summarize materials and fabrication approaches across papers |

---

## Example Queries

```
What materials are used in 3D printed heart valve scaffolds?
```
```
Recommend materials for a compliant heart model printed using SLA.
```
```
Which hydrogels support cardiovascular scaffold fabrication?
```
```
What biomaterials are used for surface modification of polyurethane valves?
```

---

## System Architecture

Heart Materials AI uses a **Retrieval-Augmented Generation (RAG)** architecture.

| Component | Technology |
|---|---|
| Large Language Model | Ollama (e.g. Llama 3) |
| Embeddings | Sentence Transformers — `all-MiniLM-L6-v2` |
| Vector Database | ChromaDB |
| RAG Framework | LlamaIndex |
| User Interface | Streamlit |

---

## Project Structure

```
heart-materials-ai
│
├── app
│   └── app.py
│
├── data
│   └── uploaded research papers
│
├── chroma_db
│   └── vector index (created automatically)
│
├── scripts
│   └── run_mac.sh
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Requirements

- macOS (Apple Silicon recommended)
- Python 3.12
- [Ollama](https://ollama.com)

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/heart-materials-ai.git
cd heart-materials-ai

# Run the application
./scripts/run_mac.sh
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Using the Application

**1. Upload Research Papers**
Upload PDFs in the sidebar — biomaterials papers, additive manufacturing protocols, cardiovascular scaffold research, or surgical simulation studies.

**2. Build the Index**
Click **Build / Rebuild Index** to create the document vector database.

**3. Ask Questions**
Enter research questions in the query box. The system retrieves relevant passages and generates literature-grounded answers.

**4. Explore Recommendations**
Specify constraints (use case, printer type, target properties) and the model will suggest candidate materials supported by your uploaded literature.

---

## Data Policy

Uploaded PDFs remain on the local machine. The system does **not transmit or redistribute documents**, making it suitable for unpublished research, lab protocols, and proprietary datasets.

> Users should ensure they have appropriate rights to analyze any uploaded documents.

---

## Roadmap

- [ ] Material comparison tables
- [ ] Exportable recommendation reports (CSV)
- [ ] Structured protocol extraction
- [ ] Automated printer parameter extraction
- [ ] Multi-paper material ranking
- [ ] Expanded biomaterials knowledge base
- [ ] Docker deployment
- [ ] Web-hosted demonstration version

---

## Contributing

Contributions are welcome. Areas where help would be valuable:

- Additional biomaterials datasets
- Improved recommendation ranking
- Extraction of material properties from literature
- UI improvements
- Deployment workflows

Pull requests and issues are encouraged.

---

## License

[MIT License](LICENSE)

---

## Citation

If you use this software in research, please cite the repository:

```
Heart Materials AI — Local AI assistant for biomaterials literature analysis
GitHub: https://github.com/YOUR_USERNAME/heart-materials-ai
```

---

## Acknowledgements

This project builds upon several open-source tools:

- [Ollama](https://ollama.com)
- [LlamaIndex](https://www.llamaindex.ai)
- [ChromaDB](https://www.trychroma.com)
- [Sentence Transformers](https://www.sbert.net)
- [Streamlit](https://streamlit.io)
