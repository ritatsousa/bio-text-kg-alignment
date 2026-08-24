# Dataset Files

This folder contains the data files required by the alignment experiments. The
dataset was generated from CTD knowledge graph resources and PubTator evidence
text.

## Folder Structure

The code assumes the paths below unless the paths are changed in the configuration files.

```text
Data/
|-- CTD_to_PubTator/
|   `-- kg_data/
|       |-- evidence_aligned_ided_medcpt_dedup.tsv
|       |-- kg_full.tsv
|       `-- metadata_full.tsv
|-- text_embeddings/
|   |-- embeddings_biobert_mcpt.npy
|   |-- embeddings_pubmedbert_mcpt.npy
|   `-- index.tsv
|-- kg_embeddings/
|   `-- outputs/
|       `-- selected/
|           |-- rdf2vec/
|           |-- rotate/
|           |-- tucker/
|           `-- manifest.json
`-- structure.txt
```

## Evidence-Aligned Dataset

`CTD_to_PubTator/kg_data/evidence_aligned_ided_medcpt_dedup.tsv` is the main
alignment dataset. It contains one row per evidence/triple pair.

Columns:

- `pair_id`: row-level identifier for the evidence/triple pair.
- `triple_id`: identifier of the KG triple.
- `subject_uri`: CTD URI of the triple subject.
- `predicate_uri`: relation URI.
- `object_uri`: CTD URI of the triple object.
- `pmid`: PubMed identifier of the evidence document.
- `text_id`: identifier of the evidence sentence/text span.
- `text`: evidence text associated with the triple.
- `subject_name`: readable subject label.
- `object_name`: readable object label.

The `text_id` column maps to `text_embeddings/index.tsv`. The `subject_uri` and
`object_uri` columns map to `entity_embeddings.csv` files. The `predicate_uri`
column maps to `relation_embeddings.csv` files.


## Knowledge Graph Files

`CTD_to_PubTator/kg_data/kg_full.tsv` contains the extended KG triples. It contains 1,048,611 triples and 14 predicate types.

Format:

```text
subject_uri<TAB>predicate_uri<TAB>object_uri
```

`CTD_to_PubTator/kg_data/metadata_full.tsv` contains metadata triples. It contains labels and type assertions, where `rdf:type` rows provide node types and `rdfs:label` rows provide readable entity labels.

Format:

```text
entity_uri<TAB>metadata_predicate<TAB>metadata_value
```


## Text Embeddings

The `text_embeddings/` folder should contain sentence-level biomedical text
embeddings aligned to the evidence table.

Expected files:

- `embeddings_biobert_mcpt.npy`: BioBERT/MedCPT text embedding matrix.
- `embeddings_pubmedbert_mcpt.npy`: PubMedBERT/MedCPT text embedding matrix.
- `index.tsv`: mapping between embedding rows and `text_id`.

The embedding files may contain duplicate text rows. The alignment code uses
`index.tsv` and `text_id` to align these embeddings to the deduplicated evidence
dataset.



## KG Embeddings

The `kg_embeddings/outputs/selected/` folder should contain selected KG embedding
configurations used by the alignment experiments.

Families:

- `rdf2vec/`: RDF2Vec embeddings trained from random walks and Skip-gram Word2Vec.
- `rotate/`: RotatE embeddings trained with PyKEEN.
- `tucker/`: TuckER embeddings trained with PyKEEN.

Each selected configuration is expected to contain:

- `entity_embeddings.csv`: entity vectors keyed by `entity_id`.
- `relation_embeddings.csv`: relation vectors keyed by `relation_id`.
- `training_summary.json`: metadata about the KG embedding run.


## Notes 

Text embedding and KG embedding files are large generated artifacts so they are intentionally not included in this GitHub repo.  
