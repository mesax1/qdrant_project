import config
from qdrant_client import QdrantClient, models
from fastembed import (
    TextEmbedding,
    SparseTextEmbedding,
    LateInteractionTextEmbedding,
)
from loguru import logger


class HybridSearcher:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.rerank_limit = config.RERANK_LIMIT
        self.prefetch_limit = config.PREFETCH_LIMIT

        # Initialize Client
        logger.info("Creating Qdrant client...")
        self.qdrant_client = QdrantClient(url=config.QDRANT_URL, timeout=config.TIMEOUT)
        logger.info(f"Qdrant client connected: {config.QDRANT_URL}")

        # Check if collection exists
        if not self.qdrant_client.collection_exists(
            collection_name=self.collection_name
        ):
            logger.error(
                f"Collection '{self.collection_name}' does not exist. Please run index_data.py first."
            )
        else:
            logger.info(f"Using collection: '{self.collection_name}'")

        # Initialize Models
        logger.info(f"Initializing dense embedding model: {config.DENSE_MODEL_NAME}...")
        self.dense_embedding_model = TextEmbedding(model_name=config.DENSE_MODEL_NAME)
        logger.info(f"Dense embedding model initialized: {config.DENSE_MODEL_NAME}")

        logger.info(
            f"Initializing sparse embedding model: {config.SPARSE_MODEL_NAME}..."
        )
        self.sparse_embedding_model = SparseTextEmbedding(
            model_name=config.SPARSE_MODEL_NAME
        )
        logger.info(f"Sparse embedding model initialized: {config.SPARSE_MODEL_NAME}")

        logger.info(
            f"Initializing late interaction embedding model: {config.LATE_INTERACTION_MODEL_NAME}..."
        )
        self.late_interaction_embedding_model = LateInteractionTextEmbedding(
            model_name=config.LATE_INTERACTION_MODEL_NAME
        )
        logger.info(
            f"Late interaction embedding model initialized: {config.LATE_INTERACTION_MODEL_NAME}"
        )

    def search(self, query_text: str, target_user_id: str | None = None):
        """
        Performs a hybrid search with dense, sparse, and late interaction reranking.

        Args:
            query_text: The text to search for.
            target_user_id: Optional user ID to filter results for.

        Returns:
            A list of search results (points).
        """
        logger.info(f"Embedding query: '{query_text}'")
        dense_query_vector = next(self.dense_embedding_model.query_embed(query_text))
        sparse_query_vector = next(self.sparse_embedding_model.query_embed(query_text))
        late_query_vector = next(
            self.late_interaction_embedding_model.query_embed(query_text)
        )

        # Build filter if target_user_id is provided
        query_filter = None
        if target_user_id:
            logger.info(f"Applying filter for user_id: {target_user_id}")
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key=config.PAYLOAD_INDEX_FIELD,
                        match=models.MatchValue(value=target_user_id),
                    )
                ]
            )

        # Build Prefetch Queries
        prefetch_queries = [
            models.Prefetch(
                query=dense_query_vector.tolist(),
                using=config.DENSE_VECTOR_NAME,
                limit=self.prefetch_limit,
                filter=query_filter,
            ),
            models.Prefetch(
                query=models.SparseVector(**sparse_query_vector.as_object()),
                using=config.SPARSE_VECTOR_NAME,
                limit=self.prefetch_limit,
                filter=query_filter,
            ),
        ]

        # Reranking with Late Interaction
        log_message = f"Performing hybrid search WITH LATE INTERACTION RERANKING"
        if target_user_id:
            log_message += f" for user '{target_user_id}'"
        log_message += "..."
        logger.info(log_message)

        search_result_reranked = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch_queries,
            query=late_query_vector,
            using=config.LATE_INTERACTION_VECTOR_NAME,
            limit=self.rerank_limit,
            with_payload=[
                "title",
                "user_id",
                "text",
            ],  # Adjust payload fields as needed
            with_vectors=False,
        )

        logger.info(f"Search returned {len(search_result_reranked.points)} results.")
        return search_result_reranked.points


# Example usage (optional, for testing)
if __name__ == "__main__":
    searcher = HybridSearcher(collection_name=config.COLLECTION_NAME)
    query = "What do you know about the lord of the rings?"
    user = "user_5"
    results = searcher.search(query_text=query, target_user_id=user)

    if results:
        logger.info(f"Search results for query '{query}' and user '{user}':")
        for i, point in enumerate(results):
            logger.info(f"{i + 1}. ID: {point.id}, Score: {point.score:.4f}")
            logger.info(f"Title: {point.payload.get('title', 'N/A')}")
            logger.info(f"User ID: {point.payload.get('user_id', 'N/A')}")
            logger.info(f"Text: {point.payload.get('text', 'N/A')}")
    else:
        logger.info(f"No results found for query '{query}' and user '{user}'.")

    # Example without user filter
    results_all = searcher.search(query_text=query)
    if results_all:
        logger.info(f"\nSearch results for query '{query}' (no user filter):")
        for i, point in enumerate(results_all):
            logger.info(f"{i + 1}. ID: {point.id}, Score: {point.score:.4f}")
            logger.info(f"Title: {point.payload.get('title', 'N/A')}")
            logger.info(f"User ID: {point.payload.get('user_id', 'N/A')}")
            logger.info(f"Text: {point.payload.get('text', 'N/A')}")
    else:
        logger.info(f"No results found for query '{query}' (no user filter).")
