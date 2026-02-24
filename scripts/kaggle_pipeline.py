# ruff: noqa: E402
# %% [markdown]
# # Sage: Kaggle GPU Pipeline
#
# Runs the full data pipeline on Kaggle with 1M reviews using GPU acceleration.
# Uploads embeddings to Qdrant Cloud.
#
# **Setup:**
# 1. Enable GPU (Settings -> Accelerator -> GPU T4 x2)
# 2. Add secrets: `QDRANT_URL`, `QDRANT_API_KEY`
# 3. Run all cells

# %% [markdown]
# ## Environment Setup

# %%
import os
import sys
import time
from pathlib import Path

IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

if IS_KAGGLE:
    # Add sage package from Kaggle dataset
    sys.path.insert(0, "/kaggle/input/sage-package")

    # Override data directory (Kaggle input is read-only)
    os.environ["SAGE_DATA_DIR"] = "/kaggle/working/data"

    import subprocess

    packages = ["qdrant-client>=1.7.0", "sentence-transformers>=2.2.0"]
    for pkg in packages:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            stdout=subprocess.DEVNULL,
        )
    print("Packages installed")

    from kaggle_secrets import UserSecretsClient

    secrets = UserSecretsClient()
    os.environ["QDRANT_URL"] = secrets.get_secret("QDRANT_URL")
    os.environ["QDRANT_API_KEY"] = secrets.get_secret("QDRANT_API_KEY")
    print("Secrets loaded")
else:
    from dotenv import load_dotenv

    load_dotenv()
    print("Using local .env")

print(f"QDRANT_URL: {os.environ.get('QDRANT_URL', 'NOT SET')[:40]}...")

# %% [markdown]
# ## Check GPU

# %%
import torch

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
else:
    print("WARNING: No GPU detected, embeddings will be slow")

# %% [markdown]
# ## Load and Filter Data

# %%
from sage.data import prepare_data, get_review_stats

SUBSET_SIZE = 1_000_000 if IS_KAGGLE else 100_000

print(f"Loading {SUBSET_SIZE:,} reviews...")
start = time.time()
df = prepare_data(subset_size=SUBSET_SIZE, force=True)
print(f"Prepared {len(df):,} reviews in {time.time() - start:.1f}s")

stats = get_review_stats(df)
print(f"  Users: {stats['unique_users']:,}")
print(f"  Items: {stats['unique_items']:,}")
print(f"  Sparsity: {stats['sparsity']:.4f}")

# %% [markdown]
# ## Chunk Reviews

# %%
from sage.adapters.embeddings import get_embedder
from sage.core.chunking import chunk_reviews_batch

# Prepare reviews for chunking
reviews = df.to_dict("records")
for i, review in enumerate(reviews):
    review["review_id"] = f"review_{i}"
    review["product_id"] = review.get("parent_asin", review.get("asin", ""))

print("Loading E5-small embedding model...")
embedder = get_embedder()

print(f"Chunking {len(reviews):,} reviews...")
start = time.time()
chunks = chunk_reviews_batch(reviews, embedder=embedder)
print(f"Created {len(chunks):,} chunks in {time.time() - start:.1f}s")
print(f"Expansion ratio: {len(chunks) / len(reviews):.2f}x")

# %% [markdown]
# ## Generate Embeddings (GPU)

# %%
import numpy as np

chunk_texts = [c.text for c in chunks]

cache_dir = Path("/kaggle/working") if IS_KAGGLE else Path("data")
cache_dir.mkdir(exist_ok=True)
cache_path = cache_dir / f"embeddings_{len(chunks)}.npy"

print(f"Embedding {len(chunks):,} chunks...")
start = time.time()
embeddings = embedder.embed_passages(
    chunk_texts,
    cache_path=cache_path,
    force=True,
    batch_size=64,
)
embed_time = time.time() - start

print(f"Embeddings: {embeddings.shape} in {embed_time:.1f}s")
print(f"Throughput: {len(chunks) / embed_time:.0f} chunks/sec")

# Validate
assert embeddings.shape[1] == 384, f"Wrong dims: {embeddings.shape[1]}"
assert np.isnan(embeddings).sum() == 0, "NaN values"
norms = np.linalg.norm(embeddings, axis=1)
assert np.allclose(norms, 1.0, atol=0.01), "Not normalized"
print("Validation: PASSED")

# %% [markdown]
# ## Upload to Qdrant Cloud

# %%
from sage.adapters.vector_store import (
    get_client,
    create_collection,
    upload_chunks,
    get_collection_info,
    create_payload_indexes,
)

qdrant_url = os.environ.get("QDRANT_URL")
print(f"Uploading to: {qdrant_url[:40]}...")

client = get_client()
create_collection(client)
create_payload_indexes(client)

start = time.time()
upload_chunks(client, chunks, embeddings)
print(f"Upload complete in {time.time() - start:.1f}s")

info = get_collection_info(client)
print("\nCollection info:")
for key, value in info.items():
    print(f"  {key}: {value}")

# %% [markdown]
# ## Test Search

# %%
from sage.adapters.vector_store import search

query = "wireless headphones with noise cancellation"
query_emb = embedder.embed_single_query(query)
results = search(client, query_emb.tolist(), limit=5)

print(f"Query: '{query}'\n")
for i, r in enumerate(results):
    print(f"{i + 1}. [{r['rating']:.0f}*] {r['text'][:70]}...")

# %%
client.close()
print(
    f"\nDone! {info.get('points_count', len(chunks)):,} chunks indexed to Qdrant Cloud"
)
