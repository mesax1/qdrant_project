import config
from qdrant_client import QdrantClient, models
from fastembed import (
    TextEmbedding,
    SparseTextEmbedding,
    LateInteractionTextEmbedding,
)
from loguru import logger

# Initialize Client
logger.info("Creating Qdrant client...")

qdrant_client = QdrantClient(url=config.QDRANT_URL, timeout=config.TIMEOUT)
logger.info(f"Qdrant client connected: {config.QDRANT_URL}")

# Check if collection exists
if not qdrant_client.collection_exists(collection_name=config.COLLECTION_NAME):
    logger.error(
        f"Collection '{config.COLLECTION_NAME}' does not exist. Please run index_data.py first."
    )
    exit()
else:
    logger.info(f"Using collection: '{config.COLLECTION_NAME}'")


# Initialize Models
logger.info(f"Initializing dense embedding model: {config.DENSE_MODEL_NAME}...")
dense_embedding_model = TextEmbedding(model_name=config.DENSE_MODEL_NAME)
logger.info(f"Dense embedding model initialized: {config.DENSE_MODEL_NAME}")

logger.info(f"Initializing sparse embedding model: {config.SPARSE_MODEL_NAME}...")
sparse_embedding_model = SparseTextEmbedding(model_name=config.SPARSE_MODEL_NAME)
logger.info(f"Sparse embedding model initialized: {config.SPARSE_MODEL_NAME}")


logger.info(
    f"Initializing late interaction embedding model: {config.LATE_INTERACTION_MODEL_NAME}..."
)
late_interaction_embedding_model = LateInteractionTextEmbedding(
    model_name=config.LATE_INTERACTION_MODEL_NAME
)
logger.info(
    f"Late interaction embedding model initialized: {config.LATE_INTERACTION_MODEL_NAME}"
)


# Search Parameters
query_text = "impact of renewable energy on climate change"
target_user_id = "user_5"  # Example user ID to filter for
rerank_limit = config.RERANK_LIMIT  # Final number of results to retrieve
prefetch_limit = (
    config.PREFETCH_LIMIT  # Number of candidates to fetch per vector type before fusion/reranking
)


# Embed Query
logger.info(f"Embedding query: '{query_text}'")
dense_query_vector = next(dense_embedding_model.query_embed(query_text))
sparse_query_vector = next(sparse_embedding_model.query_embed(query_text))
late_query_vector = next(late_interaction_embedding_model.query_embed(query_text))


# Filter ensures the final results belong to the target user_id.
query_filter = models.Filter(
    must=[
        models.FieldCondition(
            key=config.PAYLOAD_INDEX_FIELD,  # Field indexed in index_data.py
            match=models.MatchValue(value=target_user_id),
        )
    ]
)

# Build Prefetch Queries for initial candidate retrieval
# These retrieve top candidates based on dense and sparse similarity independently.
prefetch_queries = [
    models.Prefetch(
        query=dense_query_vector.tolist(),
        using=config.DENSE_VECTOR_NAME,
        limit=prefetch_limit,
        filter=query_filter,
    ),
    models.Prefetch(
        query=models.SparseVector(**sparse_query_vector.as_object()),
        using=config.SPARSE_VECTOR_NAME,
        limit=prefetch_limit,
        filter=query_filter,
    ),
]

# Reranking with Late Interaction
logger.info(
    f"Performing hybrid search WITH LATE INTERACTION RERANKING for user '{target_user_id}'..."
)
search_result_reranked = qdrant_client.query_points(
    collection_name=config.COLLECTION_NAME,
    prefetch=prefetch_queries,
    query=late_query_vector,  # Use the late interaction vector for reranking
    using=config.LATE_INTERACTION_VECTOR_NAME,  # Specify the late interaction vector name
    limit=rerank_limit,
    with_payload=["title", "user_id"],
    with_vectors=False,
)
logger.info(f"Search results with reranking (Top {rerank_limit}):")
if search_result_reranked.points:
    for i, point in enumerate(search_result_reranked.points):
        logger.info(f"{i + 1}. ID: {point.id}, Score: {point.score:.4f}")
        logger.info(f"   Title: {point.payload.get('title', 'N/A')}")
        logger.info(f"   User ID: {point.payload.get('user_id', 'N/A')}")
else:
    logger.info("No results found matching the criteria.")
