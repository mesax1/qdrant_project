from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel  # For potential request/response models
from typing import List, Optional, Dict, Any
import uvicorn
import config
from hybrid_searcher import HybridSearcher  # Import the class
from loguru import logger

# Initialize FastAPI app
app = FastAPI(
    title="Hybrid Search API",
    description="API for performing hybrid search using dense and sparse vectors, with late-interaction reranking.",
    version="1.0.0",
)


class SearchResult(BaseModel):
    id: str | int
    score: float
    payload: Dict[str, Any]


class SearchResponse(BaseModel):
    results: List[SearchResult]


# Initialize the Hybrid Searcher
# This happens once when the service starts
try:
    searcher = HybridSearcher(collection_name=config.COLLECTION_NAME)
    logger.info("HybridSearcher initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize HybridSearcher: {e}")
    searcher = None  # Set to None to indicate failure


@app.get(
    "/api/search",
    response_model=SearchResponse,
    summary="Perform Hybrid Search",
    description="Searches the collection using a combination of dense, sparse, and late-interaction reranking. Optionally filters by user ID. The user_id created by index_data.py is a string of the form 'user_0', 'user_1', etc. from user_0 to user_9.",
)
def search_documents(
    query: str = Query(
        ...,  # Ellipsis indicates it's required
        title="Search Query",
        description="The text query to search for.",
    ),
    user_id: Optional[str] = Query(
        None,  # Default value is None
        title="User ID Filter",
        description="Optional user ID to filter results for a specific user. The user_id created by index_data.py is a string of the form 'user_0', 'user_1', etc. from user_0 to user_9.",
    ),
):
    """
    Search endpoint.

    - **query**: The search query string.
    - **user_id**: (Optional) The user ID to filter results for.
    """
    if searcher is None:
        logger.error("Search endpoint called but HybridSearcher is not initialized.")
        raise HTTPException(
            status_code=503,
            detail="Search service is not available due to initialization error.",
        )

    if not query:
        raise HTTPException(
            status_code=400, detail="Query parameter 'query' cannot be empty."
        )

    try:
        logger.info(f"Received search request: query='{query}', user_id='{user_id}'")
        results = searcher.search(query_text=query, target_user_id=user_id)
        # Format results to match the response model
        formatted_results = [
            SearchResult(id=point.id, score=point.score, payload=point.payload)
            for point in results
        ]
        return SearchResponse(results=formatted_results)
    except Exception as e:
        logger.error(f"Error during search for query '{query}': {e}")
        raise HTTPException(
            status_code=500, detail=f"An internal error occurred during search: {e}"
        )


# Health check endpoint (optional but good practice)
@app.get(
    "/health",
    summary="Health Check",
    description="Returns 'OK' if the service is running.",
)
def health_check():
    # Could add more checks here, e.g., Qdrant client connectivity
    if searcher and searcher.qdrant_client:
        # Basic check: try getting cluster info
        try:
            searcher.qdrant_client.get_collections()  # Simple operation to check connection
            return {"status": "OK"}
        except Exception as e:
            logger.error(f"Health check failed: Qdrant connection error: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Service Unavailable: Qdrant connection error: {e}",
            )
    else:
        logger.warning("Health check called but HybridSearcher is not initialized.")
        raise HTTPException(
            status_code=503, detail="Service Unavailable: Searcher not initialized"
        )


# Run the service with Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
