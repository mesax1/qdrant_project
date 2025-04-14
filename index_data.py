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

# Limit dataset size based on config
if config.MAX_DOCUMENTS is not None and config.MAX_DOCUMENTS < len(dataset):
    logger.warning(f"Limiting dataset size to {config.MAX_DOCUMENTS} documents.")
    dataset = dataset.select(range(config.MAX_DOCUMENTS))
else:
    logger.info(f"Using full dataset size: {len(dataset)} documents.")

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
logger.info("Sparse embeddings embedded")


# Late Interaction Embeddings
# logger.info(
#     f"Initializing late interaction embedding model: {config.LATE_INTERACTION_MODEL_NAME}..."
# )
# late_interaction_embedding_model = LateInteractionTextEmbedding(
#     model_name=config.LATE_INTERACTION_MODEL_NAME,
#     model_type="text",
# )
# logger.info(
#     f"Late interaction embedding model initialized: {config.LATE_INTERACTION_MODEL_NAME}"
# )

# logger.info("Embedding dataset...")
# late_interaction_embeddings = list(
#     late_interaction_embedding_model.passage_embed(dataset["text"][0:1])
# )
# logger.info(f"Late interaction embeddings embedded: {len(late_interaction_embeddings)}")
# logger.info(
#     f"Embedding size of first document: {len(late_interaction_embeddings[0][0])}"
# )

# Create Qdrant client
logger.info("Creating Qdrant client...")
qdrant_client = QdrantClient(url=config.QDRANT_URL, timeout=config.TIMEOUT)
logger.info(f"Qdrant client created: {config.QDRANT_URL}")

# Check if collection exists, create if not
collection_name = config.COLLECTION_NAME
if qdrant_client.collection_exists(collection_name=collection_name):
    logger.info(f"Collection '{collection_name}' already exists. Skipping creation.")
else:
    logger.info(f"Creating collection: {collection_name}...")
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config={
            config.DENSE_VECTOR_NAME: models.VectorParams(
                size=config.DENSE_VECTOR_SIZE,
                distance=models.Distance.COSINE,
                quantization_config=models.BinaryQuantization(
                    binary=models.BinaryQuantizationConfig(always_ram=True),
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=0,  # Defer HNSW construction, to store vectors and index them later after upload of all points
                ),
            ),
            config.LATE_INTERACTION_VECTOR_NAME: models.VectorParams(
                size=config.LATE_INTERACTION_VECTOR_SIZE,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=0,  # Disable HNSW graph creation given that late interaction will only be used for reranking purposes
                ),
            ),
        },
        sparse_vectors_config={
            config.SPARSE_VECTOR_NAME: models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            ),
        },
        shard_number=config.SHARD_NUMBER,
        replication_factor=config.REPLICATION_FACTOR,
    )
    logger.info(f"Collection created: {collection_name}")

# Upload data to Qdrant by batches
logger.info(f"Uploading data to Qdrant by batches of {config.BATCH_SIZE}...")

# We are simulating 10 users, so we need to assign user_id to each document based on its index
# We split the dataset into 10 parts and assign user_id to each part
docs_per_user = len(dataset) // config.NUM_USERS
processed_docs = 0

for batch in tqdm.tqdm(
    dataset.iter(batch_size=config.BATCH_SIZE),
    total=len(dataset) // config.BATCH_SIZE,
):
    dense_embeddings = list(dense_embedding_model.passage_embed(batch["text"]))
    sparse_embeddings = list(sparse_embedding_model.passage_embed(batch["text"]))

    points_to_upload = []
    for i, _ in enumerate(batch["_id"]):
        global_index = processed_docs + i
        # Assign user_id from 0 to 9 based on the global index
        # Use min to handle the last few documents if len(dataset) is not perfectly divisible by 10
        user_id_num = min(global_index // docs_per_user, config.NUM_USERS - 1)
        user_id_str = f"user_{user_id_num}"

        points_to_upload.append(
            models.PointStruct(
                id=int(
                    batch["_id"][i].replace("doc", "")
                ),  # Assuming _id is like 'doc0', 'doc1', etc. Adjust if needed.
                vector={
                    config.DENSE_VECTOR_NAME: dense_embeddings[i].tolist(),
                    config.SPARSE_VECTOR_NAME: sparse_embeddings[i].as_object(),
                },
                payload={
                    "_id": batch["_id"][i],
                    "user_id": user_id_str,  # Assign the calculated user_id
                    "title": batch["title"][i],
                    "text": batch["text"][i],
                },
            )
        )

    qdrant_client.upload_points(
        config.COLLECTION_NAME,  # Use config variable for collection name
        points=points_to_upload,
        batch_size=config.BATCH_SIZE,  # This batch_size is for the upload_points call itself
    )

    processed_docs += len(batch["_id"])  # Update the count of processed documents

# Add payload index field
logger.info("Adding payload index field...")
qdrant_client.create_payload_index(
    collection_name=config.COLLECTION_NAME,
    field_name=config.PAYLOAD_INDEX_FIELD,
    field_schema="keyword",  # We could use uuid instead of keyword if user_id uses UUID values, as its better optimized for that case
)
logger.info("Payload index field added...")

# Re-enable HNSW construction after upload of all points
logger.info("Re-enabling HNSW construction...")
qdrant_client.update_collection(
    collection_name=config.COLLECTION_NAME,
    vectors_config={
        config.DENSE_VECTOR_NAME: models.VectorParamsDiff(
            hnsw_config=models.HnswConfigDiff(
                m=16,  # Re-enable HNSW construction after upload of all points
            ),
        ),
    },
)
logger.info("HNSW construction re-enabled...")
