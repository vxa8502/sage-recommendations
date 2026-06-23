# ruff: noqa: E402
# %% [markdown]
# # Sage: Kaggle GPU Pipeline
#
# Runs the full data pipeline on Kaggle with 1M reviews using GPU acceleration.
# Uploads embeddings to Qdrant Cloud when Qdrant credentials are available.
#
# **Setup:**
# 1. Enable GPU (Settings -> Accelerator -> GPU T4 x2)
# 2. Attach dataset: `stardewcvalley/sage-package`
# 3. Optional for upload: in the Kaggle editor open `Add-ons -> Secrets`,
#    enable `QDRANT_URL` and `QDRANT_API_KEY`, then save the notebook version
# 4. Run all cells
#
# Note: a CLI-pushed kernel can start running before those editor-side secrets
# are attached. Saving from the editor may restart the run, and that is normal.

# %% [markdown]
# ## Environment Setup

# %%
import json
import os
import subprocess
import sys
import time
from importlib import metadata as importlib_metadata
from pathlib import Path

IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
KAGGLE_PACKAGE_DATASET = os.getenv(
    "SAGE_KAGGLE_PACKAGE_DATASET", "stardewcvalley/sage-package"
)
KAGGLE_DATA_DIR = Path("/kaggle/working/data")
KAGGLE_ARTIFACT_DIR = Path("/kaggle/working")
LOCAL_ARTIFACT_DIR = Path("data")
KAGGLE_SUBSET_SIZE = 1_000_000
LOCAL_SUBSET_SIZE = 1_000_000
DEFAULT_KERNEL_CONFIG_NAME = "sage_kernel_config.json"
EMBED_BATCH_SIZE = 64
QDRANT_TIMEOUT = 30
QDRANT_TEST_QUERY = "example product query"
TRUTHY_ENV_TOKENS = {"1", "true", "yes", "on"}
KAGGLE_PACKAGE_SPECS = [
    "python-dotenv==1.0.1",
    "protobuf>=5.26.1,<6.0.0",
    "qdrant-client==1.12.1",
    "sentence-transformers==3.3.1",
    "transformers==4.47.1",
    "safetensors==0.4.5",
]
PACKAGE_VERSION_NAMES = (
    "protobuf",
    "qdrant-client",
    "sentence-transformers",
    "transformers",
    "huggingface-hub",
)


def _configure_optional_backends() -> None:
    """Prefer the PyTorch path and reduce optional TensorFlow/JAX noise."""
    os.environ.setdefault("USE_TORCH", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def _require_package_mount(path: Path) -> None:
    """Ensure the Kaggle dataset mount exists before importing from it."""
    if not path.exists():
        raise FileNotFoundError(
            "Expected Sage package dataset at "
            f"{path}. Reattach the dataset if the mount changed."
        )

    sys.path.insert(0, str(path))
    print(f"Using package path: {path}")


def _load_kernel_config() -> dict[str, object]:
    """Load optional ingestion CLI config bundled with the pushed kernel."""
    candidate_paths: list[Path] = []
    if "__file__" in globals():
        candidate_paths.append(Path(__file__).with_name(DEFAULT_KERNEL_CONFIG_NAME))
    candidate_paths.append(Path(DEFAULT_KERNEL_CONFIG_NAME))

    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            with path.open(encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Invalid ingestion kernel config at {path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Invalid ingestion kernel config at {path}: expected a JSON object."
            )
        print(f"Loaded ingestion kernel config from {path}")
        return payload
    return {}


def _resolve_subset_size(*, is_kaggle: bool, config: dict[str, object]) -> int:
    """Resolve the configured subset size for this ingestion run."""
    default = KAGGLE_SUBSET_SIZE if is_kaggle else LOCAL_SUBSET_SIZE
    configured = config.get("subset_size")
    if configured is None:
        return default
    if isinstance(configured, bool) or not isinstance(configured, int) or configured < 1:
        raise RuntimeError(
            "Ingestion kernel config field 'subset_size' must be a positive integer."
        )
    return configured


def _resolve_kaggle_package_path() -> Path:
    """Find the mounted Sage package dataset inside the Kaggle runtime."""
    dataset_slug = KAGGLE_PACKAGE_DATASET.rsplit("/", 1)[-1]
    candidate_paths: list[Path] = []

    explicit_path = os.getenv("SAGE_KAGGLE_PACKAGE_PATH", "").strip()
    if explicit_path:
        candidate_paths.append(Path(explicit_path))

    candidate_paths.extend(
        [
            Path("/kaggle/input") / dataset_slug,
            Path("/kaggle/input/datasets") / KAGGLE_PACKAGE_DATASET,
            Path("/kaggle/input") / KAGGLE_PACKAGE_DATASET,
        ]
    )

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate

    kaggle_input_root = Path("/kaggle/input")
    if kaggle_input_root.exists():
        for candidate in kaggle_input_root.glob("**/*"):
            if not candidate.is_dir():
                continue
            if candidate.name != dataset_slug:
                continue
            if (candidate / "pyproject.toml").exists() or (candidate / "sage").exists():
                return candidate

    raise FileNotFoundError(
        "Could not locate the mounted Sage package dataset inside /kaggle/input. "
        f"Expected a mount for {KAGGLE_PACKAGE_DATASET}."
    )


def _install_packages(package_specs: list[str]) -> dict[str, str]:
    """Install all Kaggle runtime dependencies in a single resolver transaction."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--disable-pip-version-check",
            *package_specs,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.stderr.strip():
            print(result.stderr.strip())
        raise subprocess.CalledProcessError(
            result.returncode,
            result.args,
            output=result.stdout,
            stderr=result.stderr,
        )

    installed = {
        name: importlib_metadata.version(name) for name in PACKAGE_VERSION_NAMES
    }
    protobuf_major = int(installed["protobuf"].split(".", 1)[0])
    if protobuf_major >= 6:
        raise RuntimeError(
            "Kaggle resolved protobuf>=6, which is incompatible with the "
            f"TensorFlow 2.19 runtime in this notebook: {installed['protobuf']}"
        )

    return installed


def _print_installed_packages(installed: dict[str, str]) -> None:
    """Print the exact package versions the Kaggle kernel resolved."""
    print("Packages installed")
    for name, version in installed.items():
        print(f"  {name}: {version}")


def _env_flag(name: str) -> bool:
    """Interpret common truthy environment-variable values."""
    return os.getenv(name, "").strip().lower() in TRUTHY_ENV_TOKENS


def _qdrant_upload_required() -> bool:
    """Allow Kaggle runs to require Qdrant upload when explicitly requested."""
    return _env_flag("SAGE_KAGGLE_REQUIRE_QDRANT_UPLOAD")


def _load_kaggle_secret_if_available(secrets, label: str) -> str | None:
    """Read one Kaggle secret with a warning instead of a fatal traceback."""
    try:
        value = secrets.get_secret(label)
    except Exception as exc:
        print(f"WARNING: Could not load Kaggle secret '{label}'.")
        print(f"  Details: {exc.__class__.__name__}: {exc}")
        return None

    if value is None:
        print(f"WARNING: Kaggle secret '{label}' is not configured.")
        return None

    stripped = value.strip()
    if not stripped:
        print(f"WARNING: Kaggle secret '{label}' is empty.")
        return None
    return stripped


def _print_saved_artifacts(
    chunk_manifest_path: Path,
    indexed_product_ids_path: Path,
    cache_path: Path | None,
) -> None:
    """Print the ingestion artifacts that should be downloaded from Kaggle."""
    print("\nSaved artifacts:")
    print(f"  Chunk manifest: {chunk_manifest_path}")
    print(f"  Indexed product IDs: {indexed_product_ids_path}")
    if cache_path is not None:
        print(f"  Embeddings: {cache_path}")


def _load_runtime_environment() -> None:
    """Configure Kaggle or local environment variables and credentials."""
    if IS_KAGGLE:
        _require_package_mount(_resolve_kaggle_package_path())
        os.environ["SAGE_DATA_DIR"] = str(KAGGLE_DATA_DIR)
        _print_installed_packages(_install_packages(KAGGLE_PACKAGE_SPECS))

        if os.environ.get("QDRANT_URL", "").strip():
            print("Using preconfigured Qdrant environment variables")
            return

        from kaggle_secrets import UserSecretsClient

        secrets = UserSecretsClient()
        qdrant_url = _load_kaggle_secret_if_available(secrets, "QDRANT_URL")
        qdrant_api_key = _load_kaggle_secret_if_available(secrets, "QDRANT_API_KEY")

        if qdrant_url:
            os.environ["QDRANT_URL"] = qdrant_url
        if qdrant_api_key:
            os.environ["QDRANT_API_KEY"] = qdrant_api_key

        if qdrant_url:
            print("Qdrant settings loaded from Kaggle secrets")
        else:
            print("WARNING: Qdrant is not configured for this Kaggle run.")
            print(
                "  Ingestion artifacts will still be written, but Qdrant upload "
                "will be skipped."
            )
        return

    from dotenv import load_dotenv

    load_dotenv()
    print("Using local .env")


def _preflight_qdrant() -> bool:
    """Check whether the configured Qdrant cluster is reachable."""
    qdrant_url = os.environ.get("QDRANT_URL")
    if not qdrant_url:
        print(
            "WARNING: QDRANT_URL is not configured; upload will be skipped/fail later."
        )
        return False

    from qdrant_client import QdrantClient

    probe_client = None
    try:
        probe_client = QdrantClient(
            url=qdrant_url,
            api_key=os.environ.get("QDRANT_API_KEY"),
            timeout=QDRANT_TIMEOUT,
        )
        collections = probe_client.get_collections().collections
        print(f"Qdrant reachable ({len(collections)} collections visible)")
        return True
    except Exception as e:
        print("WARNING: Qdrant preflight failed.")
        print(f"  Error: {e}")
        print(
            "  The cluster may be deleted, inactive, or the Kaggle secrets may be stale."
        )
        print(
            "  You can still finish chunking/embedding to preserve local artifacts, "
            "but upload will fail until QDRANT_URL/QDRANT_API_KEY point to a live cluster."
        )
        return False
    finally:
        if probe_client is not None:
            probe_client.close()


def _print_gpu_status() -> None:
    """Report whether PyTorch can see a usable GPU."""
    import torch

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        return

    print("WARNING: No GPU detected, embeddings will be slow")


def _configure_embedding_device() -> None:
    """Force CPU embeddings when the detected GPU is unsupported."""
    if os.getenv("SAGE_EMBEDDING_DEVICE", "").strip():
        print(
            "Embedding device override:"
            f" {os.environ['SAGE_EMBEDDING_DEVICE'].strip()}"
        )
        return

    try:
        import torch
    except Exception as exc:
        print(
            "WARNING: Could not import torch while selecting embedding device."
        )
        print(f"  Details: {exc.__class__.__name__}: {exc}")
        return

    if not torch.cuda.is_available():
        return

    capability = torch.cuda.get_device_capability(0)
    arch = f"sm_{capability[0]}{capability[1]}"
    supported_arches = set(getattr(torch.cuda, "get_arch_list", lambda: [])())
    if not supported_arches or arch in supported_arches:
        return

    gpu_name = torch.cuda.get_device_name(0)
    os.environ["SAGE_EMBEDDING_DEVICE"] = "cpu"
    print("WARNING: Current PyTorch build does not support the detected GPU.")
    print(f"  GPU: {gpu_name} ({arch})")
    print(
        "  Supported CUDA arches in this runtime: "
        + ", ".join(sorted(supported_arches))
    )
    print("  Forcing SAGE_EMBEDDING_DEVICE=cpu for embeddings.")


def _get_artifact_dir() -> Path:
    """Return the run artifact directory and ensure it exists."""
    artifact_dir = KAGGLE_ARTIFACT_DIR if IS_KAGGLE else LOCAL_ARTIFACT_DIR
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def _prepare_reviews_for_chunking(df) -> list[dict]:
    """Convert the prepared DataFrame into chunk-ready review records."""
    reviews = df.to_dict("records")
    for i, review in enumerate(reviews):
        review["review_id"] = f"review_{i}"
        review["product_id"] = review.get("parent_asin") or review.get("asin", "")
    return reviews


def _chunk_to_manifest_row(chunk) -> dict[str, object]:
    """Serialize a chunk into one JSONL manifest row."""
    return {
        "text": chunk.text,
        "chunk_index": chunk.chunk_index,
        "total_chunks": chunk.total_chunks,
        "review_id": chunk.review_id,
        "product_id": chunk.product_id,
        "rating": chunk.rating,
        "timestamp": chunk.timestamp,
        "verified_purchase": chunk.verified_purchase,
    }


def _save_chunk_manifest(chunks: list, artifact_dir: Path) -> Path:
    """Persist chunk metadata so failed uploads still leave a usable artifact."""
    chunk_manifest_path = artifact_dir / f"chunks_{len(chunks)}.jsonl"
    print(f"Saving chunk manifest to {chunk_manifest_path}...")
    with chunk_manifest_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(
                json.dumps(_chunk_to_manifest_row(chunk), ensure_ascii=False) + "\n"
            )
    print("Chunk manifest saved")
    return chunk_manifest_path


def _save_indexed_product_ids(
    chunks: list,
    artifact_dir: Path,
    *,
    subset_size: int,
    review_count: int,
    dataset_category: str,
    chunk_manifest_path: Path,
) -> Path:
    """Persist product IDs for the actual indexed corpus."""
    from sage.data.corpus_anchor import build_corpus_anchor

    indexed_product_ids_path = artifact_dir / "indexed_product_ids.json"
    product_ids = sorted(
        {
            chunk.product_id
            for chunk in chunks
            if isinstance(chunk.product_id, str) and chunk.product_id.strip()
        }
    )
    payload = build_corpus_anchor(
        product_ids=product_ids,
        dataset_category=dataset_category,
        subset_size=subset_size,
        review_count=review_count,
        chunk_count=len(chunks),
        source_kind="kaggle_chunk_index",
        source_ref=chunk_manifest_path.name,
    )

    print(f"Saving indexed product IDs to {indexed_product_ids_path}...")
    with indexed_product_ids_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print("Indexed product IDs saved")
    return indexed_product_ids_path


def _validate_embeddings(embeddings, expected_dim: int) -> None:
    """Fail early if embeddings are malformed or unexpectedly shaped."""
    import numpy as np

    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got shape {embeddings.shape}")
    if embeddings.shape[1] != expected_dim:
        raise ValueError(
            f"Wrong embedding dimensions: {embeddings.shape[1]}, expected {expected_dim}"
        )
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        raise ValueError("Embeddings contain NaN or Inf values")

    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=0.01):
        raise ValueError("Embeddings are not normalized")


_configure_optional_backends()
_load_runtime_environment()
print(f"QDRANT_URL: {'configured' if os.environ.get('QDRANT_URL') else 'NOT SET'}")

# %% [markdown]
# ## Check Qdrant Connectivity

# %%
QDRANT_OK = _preflight_qdrant()

# %% [markdown]
# ## Check GPU

# %%
_configure_embedding_device()
_print_gpu_status()

# %% [markdown]
# ## Load and Filter Data

# %%
from sage.data.loader import get_review_stats, prepare_data

SUBSET_SIZE = KAGGLE_SUBSET_SIZE if IS_KAGGLE else LOCAL_SUBSET_SIZE
KERNEL_CONFIG = _load_kernel_config()
SUBSET_SIZE = _resolve_subset_size(
    is_kaggle=IS_KAGGLE,
    config=KERNEL_CONFIG,
)
ARTIFACT_DIR = _get_artifact_dir()

print(f"Loading {SUBSET_SIZE:,} reviews...")
start = time.time()
# Kaggle kernels are ephemeral - no persistent cache between runs, always regenerate
df = prepare_data(subset_size=SUBSET_SIZE, force=True)
print(f"Prepared {len(df):,} reviews in {time.time() - start:.1f}s")

stats = get_review_stats(df)
print(f"  Users: {stats['unique_users']:,}")
print(f"  Items: {stats['unique_items']:,}")
print(f"  Sparsity: {stats['sparsity']:.4f}")

# %% [markdown]
# ## Chunk Reviews

# %%
from sage.core.chunking import chunk_reviews_batch

# Prepare reviews for chunking
reviews = _prepare_reviews_for_chunking(df)

embedder = None
if QDRANT_OK:
    from sage.adapters.embeddings import get_embedder

    print("Loading E5-small embedding model...")
    embedder = get_embedder()
else:
    print(
        "Artifact-only ingestion run: skipping embedder load and using "
        "non-semantic chunking."
    )

print(f"Chunking {len(reviews):,} reviews...")
start = time.time()
chunks = chunk_reviews_batch(reviews, embedder=embedder)
print(f"Created {len(chunks):,} chunks in {time.time() - start:.1f}s")
print(f"Expansion ratio: {len(chunks) / len(reviews):.2f}x")

chunk_manifest_path = _save_chunk_manifest(chunks, ARTIFACT_DIR)

# %% [markdown]
# ## Generate Embeddings (GPU)

# %%
from sage.config import DATASET_CATEGORY, EMBEDDING_DIM

indexed_product_ids_path = _save_indexed_product_ids(
    chunks,
    ARTIFACT_DIR,
    subset_size=SUBSET_SIZE,
    review_count=len(reviews),
    dataset_category=DATASET_CATEGORY,
    chunk_manifest_path=chunk_manifest_path,
)

if not QDRANT_OK:
    message = (
        "WARNING: Skipping embedding generation and Qdrant upload because the "
        "cluster is unreachable or Qdrant credentials are not configured."
    )
    if _qdrant_upload_required():
        _print_saved_artifacts(chunk_manifest_path, indexed_product_ids_path, None)
        raise RuntimeError(
            message
            + " Set SAGE_KAGGLE_REQUIRE_QDRANT_UPLOAD=0 to allow artifact-only runs."
        )
    print(message)
    print("  Ingestion only requires the saved artifacts below.")
    _print_saved_artifacts(chunk_manifest_path, indexed_product_ids_path, None)
    raise SystemExit(0)

chunk_texts = [c.text for c in chunks]

cache_path = ARTIFACT_DIR / f"embeddings_{len(chunks)}.npy"

print(f"Embedding {len(chunks):,} chunks...")
start = time.time()
embeddings = embedder.embed_passages(
    chunk_texts,
    cache_path=cache_path,
    force=True,
    batch_size=EMBED_BATCH_SIZE,
)
embed_time = time.time() - start

print(f"Embeddings: {embeddings.shape} in {embed_time:.1f}s")
print(f"Throughput: {len(chunks) / embed_time:.0f} chunks/sec")

# Validate embeddings (explicit checks instead of assert - survives python -O)
_validate_embeddings(embeddings, EMBEDDING_DIM)
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
    search,
)
from sage.services.corpus_alignment import stamp_corpus_anchor

client = None
try:
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

    stamp_result = stamp_corpus_anchor(
        anchor_path=indexed_product_ids_path,
        client=client,
    )
    print("\nCorpus anchor stamp:")
    for key, value in stamp_result.items():
        print(f"  {key}: {value}")

    # Test search
    query = QDRANT_TEST_QUERY
    query_emb = embedder.embed_single_query(query)
    results = search(client, query_emb.tolist(), limit=5)

    print(f"\nQuery: '{query}'\n")
    for i, r in enumerate(results):
        print(f"{i + 1}. [{r['rating']:.0f}*] {r['text'][:70]}...")

    print(
        f"\nDone! {info.get('points_count', len(chunks)):,} chunks indexed to Qdrant Cloud"
    )
except Exception as e:
    print("\nQdrant upload failed.")
    print(f"  Error: {e}")
    print(
        "  Create a new Qdrant cluster, update the Kaggle secrets, and rerun "
        "the upload cell or the full notebook."
    )
    if _qdrant_upload_required():
        raise
    print("  Continuing because ingestion only requires the saved artifacts.")
finally:
    if client is not None:
        client.close()

_print_saved_artifacts(chunk_manifest_path, indexed_product_ids_path, cache_path)
