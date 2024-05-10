import os  # Importing the os module for operating system related functionalities
from abc import ABC  # Importing the ABC class from the abc module for creating abstract base classes
from dataclasses import dataclass  # Importing the dataclass decorator for creating data classes
from typing import (  # Importing various types from the typing module for type hints
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    List,
    Optional,
    TypedDict,
    Union,
    cast,
)
from urllib.parse import urljoin  # Importing the urljoin function from the urllib.parse module for URL manipulation

import aiohttp  # Importing the aiohttp library for making asynchronous HTTP requests
from azure.search.documents.aio import SearchClient  # Importing the SearchClient class from the azure.search.documents.aio module for interacting with Azure Search
from azure.search.documents.models import (  # Importing various models from the azure.search.documents.models module for Azure Search
    QueryCaptionResult,
    QueryType,
    VectorizedQuery,
    VectorQuery,
)
from openai import AsyncOpenAI  # Importing the AsyncOpenAI class from the openai module for interacting with OpenAI

from core.authentication import AuthenticationHelper  # Importing the AuthenticationHelper class from the core.authentication module for authentication related functionalities
from text import nonewlines  # Importing the nonewlines function from the text module for removing newlines from a string

@dataclass  # Decorator for creating a data class
class Document:
    id: Optional[str]  # Optional string field for document ID
    content: Optional[str]  # Optional string field for document content
    embedding: Optional[List[float]]  # Optional list of floats field for document embedding
    image_embedding: Optional[List[float]]  # Optional list of floats field for document image embedding
    category: Optional[str]  # Optional string field for document category
    sourcepage: Optional[str]  # Optional string field for document source page
    sourcefile: Optional[str]  # Optional string field for document source file
    oids: Optional[List[str]]  # Optional list of strings field for document oids
    groups: Optional[List[str]]  # Optional list of strings field for document groups
    captions: List[QueryCaptionResult]  # List of QueryCaptionResult objects for document captions
    score: Optional[float] = None  # Optional float field for document score
    reranker_score: Optional[float] = None  # Optional float field for document reranker score

    def serialize_for_results(self) -> dict[str, Any]:
        """Serializes the Document object into a dictionary for results."""
        return {
            "id": self.id,
            "content": self.content,
            "embedding": Document.trim_embedding(self.embedding),
            "imageEmbedding": Document.trim_embedding(self.image_embedding),
            "category": self.category,
            "sourcepage": self.sourcepage,
            "sourcefile": self.sourcefile,
            "oids": self.oids,
            "groups": self.groups,
            "captions": (
                [
                    {
                        "additional_properties": caption.additional_properties,
                        "text": caption.text,
                        "highlights": caption.highlights,
                    }
                    for caption in self.captions
                ]
                if self.captions
                else []
            ),
            "score": self.score,
            "reranker_score": self.reranker_score,
        }

    @classmethod
    def trim_embedding(cls, embedding: Optional[List[float]]) -> Optional[str]:
        """Returns a trimmed list of floats from the vector embedding."""
        if embedding:
            if len(embedding) > 2:
                # Format the embedding list to show the first 2 items followed by the count of the remaining items.
                return f"[{embedding[0]}, {embedding[1]} ...+{len(embedding) - 2} more]"
            else:
                return str(embedding)

        return None


@dataclass  # Decorator for creating a data class
class ThoughtStep:
    title: str  # String field for thought step title
    description: Optional[Any]  # Optional field for thought step description
    props: Optional[dict[str, Any]] = None  # Optional dictionary field for thought step properties


class Approach(ABC):  # Abstract base class for different approaches
    def __init__(
        self,
        search_client: SearchClient,
        openai_client: AsyncOpenAI,
        auth_helper: AuthenticationHelper,
        query_language: Optional[str],
        query_speller: Optional[str],
        embedding_deployment: Optional[str],  # Not needed for non-Azure OpenAI or for retrieval_mode="text"
        embedding_model: str,
        embedding_dimensions: int,
        openai_host: str,
        vision_endpoint: str,
        vision_token_provider: Callable[[], Awaitable[str]],
    ):
        self.search_client = search_client  # Initializing the search client
        self.openai_client = openai_client  # Initializing the OpenAI client
        self.auth_helper = auth_helper  # Initializing the authentication helper
        self.query_language = query_language  # Setting the query language
        self.query_speller = query_speller  # Setting the query speller
        self.embedding_deployment = embedding_deployment  # Setting the embedding deployment
        self.embedding_model = embedding_model  # Setting the embedding model
        self.embedding_dimensions = embedding_dimensions  # Setting the embedding dimensions
        self.openai_host = openai_host  # Setting the OpenAI host
        self.vision_endpoint = vision_endpoint  # Setting the vision endpoint
        self.vision_token_provider = vision_token_provider  # Setting the vision token provider

    def build_filter(self, overrides: dict[str, Any], auth_claims: dict[str, Any]) -> Optional[str]:
        """Builds the filter based on overrides and authentication claims."""
        exclude_category = overrides.get("exclude_category")
        security_filter = self.auth_helper.build_security_filters(overrides, auth_claims)
        filters = []
        if exclude_category:
            filters.append("category ne '{}'".format(exclude_category.replace("'", "''")))
        if security_filter:
            filters.append(security_filter)
        return None if len(filters) == 0 else " and ".join(filters)

    async def search(
        self,
        top: int,
        query_text: Optional[str],
        filter: Optional[str],
        vectors: List[VectorQuery],
        use_semantic_ranker: bool,
        use_semantic_captions: bool,
        minimum_search_score: Optional[float],
        minimum_reranker_score: Optional[float],
    ) -> List[Document]:
        """Performs a search and returns a list of qualified documents."""
        if use_semantic_ranker and query_text:
            # Use semantic ranker if requested and if retrieval mode is text or hybrid (vectors + text)
            results = await self.search_client.search(
                search_text=query_text,
                filter=filter,
                query_type=QueryType.SEMANTIC,
                query_language=self.query_language,
                query_speller=self.query_speller,
                semantic_configuration_name="default",
                top=top,
                query_caption="extractive|highlight-false" if use_semantic_captions else None,
                vector_queries=vectors,
            )
        else:
            results = await self.search_client.search(
                search_text=query_text or "", filter=filter, top=top, vector_queries=vectors
            )

        documents = []
        async for page in results.by_page():
            async for document in page:
                documents.append(
                    Document(
                        id=document.get("id"),
                        content=document.get("content"),
                        embedding=document.get("embedding"),
                        image_embedding=document.get("imageEmbedding"),
                        category=document.get("category"),
                        sourcepage=document.get("sourcepage"),
                        sourcefile=document.get("sourcefile"),
                        oids=document.get("oids"),
                        groups=document.get("groups"),
                        captions=cast(List[QueryCaptionResult], document.get("@search.captions")),
                        score=document.get("@search.score"),
                        reranker_score=document.get("@search.reranker_score"),
                    )
                )

            qualified_documents = [
                doc
                for doc in documents
                if (
                    (doc.score or 0) >= (minimum_search_score or 0)
                    and (doc.reranker_score or 0) >= (minimum_reranker_score or 0)
                )
            ]

        return qualified_documents

    def get_sources_content(
        self, results: List[Document], use_semantic_captions: bool, use_image_citation: bool
    ) -> list[str]:
        """Returns the content of the sources based on the search results."""
        if use_semantic_captions:
            return [
                (self.get_citation((doc.sourcepage or ""), use_image_citation))
                + ": "
                + nonewlines(" . ".join([cast(str, c.text) for c in (doc.captions or [])]))
                for doc in results
            ]
        else:
            return [
                (self.get_citation((doc.sourcepage or ""), use_image_citation)) + ": " + nonewlines(doc.content or "")
                for doc in results
            ]

    def get_citation(self, sourcepage: str, use_image_citation: bool) -> str:
        """Returns the citation for a source page."""
        if use_image_citation:
            return sourcepage
        else:
            path, ext = os.path.splitext(sourcepage)
            if ext.lower() == ".png":
                page_idx = path.rfind("-")
                page_number = int(path[page_idx + 1 :])
                return f"{path[:page_idx]}.pdf#page={page_number}"

            return sourcepage

    async def compute_text_embedding(self, q: str):
        """Computes the text embedding using OpenAI."""
        SUPPORTED_DIMENSIONS_MODEL = {
            "text-embedding-ada-002": False,
            "text-embedding-3-small": True,
            "text-embedding-3-large": True,
        }

        class ExtraArgs(TypedDict, total=False):
            dimensions: int

        dimensions_args: ExtraArgs = (
            {"dimensions": self.embedding_dimensions} if SUPPORTED_DIMENSIONS_MODEL[self.embedding_model] else {}
        )
        embedding = await self.openai_client.embeddings.create(
            model=self.embedding_deployment if self.embedding_deployment else self.embedding_model,
            input=q,
            **dimensions_args,
        )
        query_vector = embedding.data[0].embedding
        return VectorizedQuery(vector=query_vector, k_nearest_neighbors=50, fields="embedding")

    async def compute_image_embedding(self, q: str):
        """Computes the image embedding using Azure Computer Vision."""
        endpoint = urljoin(self.vision_endpoint, "computervision/retrieval:vectorizeText")
        headers = {"Content-Type": "application/json"}
        params = {"api-version": "2023-02-01-preview", "modelVersion": "latest"}
        data = {"text": q}

        headers["Authorization"] = "Bearer " + await self.vision_token_provider()

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=endpoint, params=params, headers=headers, json=data, raise_for_status=True
            ) as response:
                json = await response.json()
                image_query_vector = json["vector"]
        return VectorizedQuery(vector=image_query_vector, k_nearest_neighbors=50, fields="imageEmbedding")

    async def run(
        self, messages: list[dict], stream: bool = False, session_state: Any = None, context: dict[str, Any] = {}
    ) -> Union[dict[str, Any], AsyncGenerator[dict[str, Any], None]]:
        """Runs the approach."""
        raise NotImplementedError
