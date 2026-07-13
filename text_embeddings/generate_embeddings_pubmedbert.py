"""
Generate PubMedBERT embeddings for evidence texts.
Usage: python generate_embeddings_pubmedbert.py [--resume] [--batch-size 32]
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "pubmedbert"
HF_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
EMBEDDING_DIM = 768
BATCH_SIZE = 32
CHECKPOINT_INTERVAL = 1000


def mean_pooling(model_output, attention_mask):
    """Mean pooling over token embeddings."""
    token_embeddings = model_output.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (16 for 4GB RAM, 64 for 16GB)")
    args = parser.parse_args()

    batch_size = args.batch_size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Paths (relative to script location, works from any working directory)
    script_dir = Path(__file__).resolve().parent
    evidence_path = script_dir / ".." / "CTD_to_PubTator" / "kg_data" / "evidence_aligned_ided.tsv"
    output_dir = script_dir
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    # Load data
    print("Loading data...")
    df = pd.read_csv(evidence_path, sep="\t")
    texts = df["text"].tolist()
    print(f"Loaded {len(texts)} texts")

    # Save index (text_id mapping)
    index_df = pd.DataFrame({"idx": range(len(df)), "text_id": df["text_id"]})
    index_df.to_csv(output_dir / "index.tsv", sep="\t", index=False)

    # Load model
    print(f"Loading {HF_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    model = AutoModel.from_pretrained(HF_MODEL)
    model.to(device)
    model.eval()
    if device == "cuda":
        model.half()

    # Check for checkpoint
    checkpoint_file = output_dir / "checkpoints" / f"{MODEL_NAME}_checkpoint.json"
    start_idx = 0
    embeddings = []

    if args.resume and checkpoint_file.exists():
        with open(checkpoint_file) as f:
            ckpt = json.load(f)
        start_idx = ckpt["last_index"]
        partial = np.load(output_dir / f"embeddings_{MODEL_NAME}_partial.npy")
        embeddings = list(partial)
        print(f"Resuming from index {start_idx}")

    # Generate embeddings
    print(f"Generating embeddings (batch_size={batch_size})...")
    with torch.no_grad():
        for i in tqdm(range(start_idx, len(texts), batch_size)):
            batch = texts[i:i + batch_size]

            encoded = tokenizer(batch, padding=True, truncation=True,
                              max_length=512, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}

            output = model(**encoded)
            batch_emb = mean_pooling(output, encoded["attention_mask"])
            embeddings.extend(batch_emb.cpu().numpy().astype(np.float16))

            # Checkpoint
            if (i + batch_size) % CHECKPOINT_INTERVAL == 0:
                np.save(output_dir / f"embeddings_{MODEL_NAME}_partial.npy", np.array(embeddings))
                with open(checkpoint_file, "w") as f:
                    json.dump({"last_index": i + batch_size}, f)

    # Save final
    embeddings = np.array(embeddings)
    np.save(output_dir / f"embeddings_{MODEL_NAME}.npy", embeddings)
    print(f"Saved embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")

    # Cleanup checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    partial_file = output_dir / f"embeddings_{MODEL_NAME}_partial.npy"
    if partial_file.exists():
        partial_file.unlink()

    # Update metadata
    meta_file = output_dir / "metadata.json"
    metadata = json.load(open(meta_file)) if meta_file.exists() else {"models": {}}
    metadata["num_texts"] = len(texts)
    metadata["source_file"] = str(evidence_path)
    metadata["models"][MODEL_NAME] = {
        "hf_model": HF_MODEL,
        "dim": EMBEDDING_DIM,
        "precision": "float16",
        "file": f"embeddings_{MODEL_NAME}.npy",
        "created_at": datetime.now().isoformat()
    }
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
