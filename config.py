# config.py
import os
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    BinaryQuantization,
    BinaryQuantizationConfig,
    SparseVectorParams,
)

# Qdrant Configuration - Point to the port-forwarded address
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")  # Should resolve to 127.0.0.1
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))  # Kubernetes port-forwarded
QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
TIMEOUT = 500
# If using API keys set in Helm values:
# QDRANT_API_KEY = "qdrant-api-key" # Ensure this matches the Helm value if set

# Collection Configuration - These settings now apply to the K8s cluster
COLLECTION_NAME = "beir_nq"

SHARD_NUMBER = 3  # Will be distributed across the 2 nodes
REPLICATION_FACTOR = 2  # Each shard replica will live on a different node

# --- Vector Config, Models, Data Config, Indexing Config, Search Config ---

# Vector Configuration
DENSE_MODEL_NAME = "BAAI/bge-large-en-v1.5"
DENSE_VECTOR_SIZE = 1024  # 384 for bge-small-en-v1.5 and 1024 for bge-large-en-v1.5
DENSE_VECTOR_NAME = "dense_vector"
# I know this is not a recommended model to apply binary quantization to, but at least it has more than 1000 dimensions, and this is just for testing

SPARSE_MODEL_NAME = "Qdrant/bm42-all-minilm-l6-v2-attentions"
SPARSE_VECTOR_NAME = "sparse_vector"

LATE_INTERACTION_MODEL_NAME = "colbert-ir/colbertv2.0"
LATE_INTERACTION_VECTOR_SIZE = (
    128  # 128 for colbertv2.0 and 96 for answerai-colbert-small-v1
)
LATE_INTERACTION_VECTOR_NAME = "late_interaction_vector"


# Binary Quantization for Dense Vectors
QUANTIZATION_CONFIG = BinaryQuantization(
    binary=BinaryQuantizationConfig(always_ram=True)
)

# Define Vector Parameters
VECTOR_PARAMS = {
    DENSE_VECTOR_NAME: VectorParams(
        size=DENSE_VECTOR_SIZE,
        distance=Distance.COSINE,
        quantization_config=QUANTIZATION_CONFIG,
        on_disk=True,
    ),
    SPARSE_VECTOR_NAME: SparseVectorParams(),
}

# Payload Indexing
PAYLOAD_INDEX_FIELD = "user_id"

# Data Configuration
DATASET_NAME = "BeIR/nq"
DATASET_CONFIG = "corpus"
DATASET_TEXT_FIELD = "text"
MAX_DOCUMENTS = (
    1_000_000  # TODO: Change to higher value if you want to index the entire dataset
)
NUM_USERS = 10

# Indexing Configuration
BATCH_SIZE = 5
UPSERT_WAIT = False

# Search Configuration
PREFETCH_LIMIT = 50
RERANK_LIMIT = 10
