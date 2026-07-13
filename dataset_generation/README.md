# Dataset Generation

This folder contains scripts for constructing the CTD-Align dataset and graph resources.

## Contents

- `scripts/download_ctd_data.py`: download CTD source resources.
- `scripts/extract_ctd_entities.py`: extract CTD entity inventories.
- `scripts/extract_1m_triples.py`: extract graph triples.
- `scripts/pubtator_api_queries.py`: query PubTator for evidence.
- `scripts/verify_initial_graph.py`: validate graph construction.

## Overview

This pipeline extracts **1 million RDF triples** from the CTD (Comparative Toxicogenomics Database) and finds corresponding text evidence from **PubTator 3.0** to create a knowledge graph aligned with biomedical literature.

Phase 1: CTD Knowledge Graph Extraction
  └─> Extract 1M triples from CTD Bio2RDF files
Phase 2: Entity Inventory 
  └─> Extract all entities from KG
Phase 3: PubTator API Queries
  └─> Query PubTator for each entity
Phase 4: Document Filtering
  └─> Minimize document set while keeping coverage
Phase 5: Text Collection
  └─> Fetch full PubTator annotations
 
```bash
# Phase 1: Download CTD data (~5-10 minutes, 416 MB) and extract 1M triples (~49 seconds)
python3 scripts/download_ctd_data.py
python3 scripts/extract_1m_triples.py

# Phase 2: Extract entities (~10 seconds)
python3 scripts/extract_ctd_entities.py

# Phase 3: Query PubTator (~5.5 hours)
python3 scripts/pubtator_api_queries.py
```


## Expected Outputs

The downstream alignment code expects:

```text
kg_data/
├── evidence_aligned_ided_medcpt_dedup.tsv
├── kg_full.tsv
└── metadata_full.tsv
```


## External Resources

### Databases
- **CTD Database**: https://ctdbase.org/
- **CTD 2021 Update Paper**: https://pubmed.ncbi.nlm.nih.gov/33068428/
- **Bio2RDF**: http://bio2rdf.org/
- **Bio2RDF CTD Stats**: https://download.bio2rdf.org/files/release/3/ctd/ctd.html
- **PubTator 3.0**: https://www.ncbi.nlm.nih.gov/research/pubtator3/

### APIs & Documentation
- **PubTator API**: https://www.ncbi.nlm.nih.gov/research/pubtator3/api
- **PubTator FTP**: https://ftp.ncbi.nlm.nih.gov/pub/lu/PubTator3/
- **Bio2RDF Paper**: https://pmc.ncbi.nlm.nih.gov/articles/PMC3632999/

### Standards & Tools
- **RDF Specification**: https://www.w3.org/RDF/
- **SPARQL 1.1**: https://www.w3.org/TR/sparql11-query/
- **RDFLib Documentation**: https://rdflib.readthedocs.io/
- **N-Quads Format**: https://www.w3.org/TR/n-quads/
