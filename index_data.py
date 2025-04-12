from fastembed import TextEmbedding
from fastembed import SparseTextEmbedding
from fastembed import LateInteractionTextEmbedding
from qdrant_client import QdrantClient, models
import tqdm
from datasets import load_dataset
from loguru import logger
import config

logger.info(f"Loading dataset: {config.DATASET_NAME}...")
dataset = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG, split="corpus")
logger.info(f"Dataset loaded: {config.DATASET_NAME}")

# Dense Embeddings
logger.info(f"Initializing dense embedding model: {config.DENSE_MODEL_NAME}...")
dense_embedding_model = TextEmbedding(
    model_name=config.DENSE_MODEL_NAME,
    model_type="text",
)
logger.info(f"Dense embedding model initialized: {config.DENSE_MODEL_NAME}")

logger.info("Embedding dataset...")
dense_embeddings = list(dense_embedding_model.passage_embed(dataset["text"][0:1]))
logger.info(f"Dense embeddings embedded: {len(dense_embeddings)}")
logger.info(f"Embedding size of first document: {len(dense_embeddings[0])}")

# Sparse Embeddings
logger.info(f"Initializing sparse embedding model: {config.SPARSE_MODEL_NAME}...")
sparse_embedding_model = SparseTextEmbedding(
    model_name=config.SPARSE_MODEL_NAME,
    model_type="text",
)
logger.info(f"Sparse embedding model initialized: {config.SPARSE_MODEL_NAME}")

logger.info("Embedding dataset...")
sparse_embeddings = list(sparse_embedding_model.passage_embed(dataset["text"][0:1]))
logger.info(f"Sparse embeddings embedded: {len(sparse_embeddings)}")
logger.info(f"Embedding size of first document: {len(sparse_embeddings[0])}")

# Late Interaction Embeddings
logger.info(
    f"Initializing late interaction embedding model: {config.LATE_INTERACTION_MODEL_NAME}..."
)
late_interaction_embedding_model = LateInteractionTextEmbedding(
    model_name=config.LATE_INTERACTION_MODEL_NAME,
    model_type="text",
)
logger.info(
    f"Late interaction embedding model initialized: {config.LATE_INTERACTION_MODEL_NAME}"
)

logger.info("Embedding dataset...")
late_interaction_embeddings = list(
    late_interaction_embedding_model.passage_embed(dataset["text"][0:1])
)
logger.info(f"Late interaction embeddings embedded: {len(late_interaction_embeddings)}")
logger.info(
    f"Embedding size of first document: {len(late_interaction_embeddings[0][0])}"
)

# Create Qdrant client
logger.info("Creating Qdrant client...")
qdrant_client = QdrantClient(url=config.QDRANT_URL, timeout=config.TIMEOUT)
logger.info(f"Qdrant client created: {config.QDRANT_URL}")

# Create collection
logger.info("Creating collection...")
qdrant_client.create_collection(
    collection_name=config.COLLECTION_NAME,
    vectors_config={
        config.DENSE_VECTOR_NAME: models.VectorParams(
            size=config.DENSE_VECTOR_SIZE,
            distance=models.Distance.COSINE,
        ),
        config.LATE_INTERACTION_VECTOR_NAME: models.VectorParams(
            size=len(
                late_interaction_embeddings[0][0]
            ),  # TODO: Move the value to config and use it here
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM,
            ),
        ),
        config.SPARSE_VECTOR_NAME: models.SparseVectorParams(
            modifier=models.Modifier.IDF,
        ),
    },
)
logger.info(f"Collection created: {config.COLLECTION_NAME}")
