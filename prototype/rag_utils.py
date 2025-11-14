"""
Utility module for RAG embeddings.
Extracted from wiki_rag.rag for standalone use.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings


class PromptedBGE(HuggingFaceEmbeddings):
    """
    BGE embeddings with prompt prefixes for better retrieval.
    
    Uses BAAI/bge-base-en model with query/document prompts.
    """
    
    def embed_documents(self, texts):
        """Embed documents with document prompt."""
        return super().embed_documents(
            [f"Represent this document for retrieval: {t}" for t in texts]
        )
    
    def embed_query(self, text):
        """Embed query with query prompt."""
        return super().embed_query(
            f"Represent this query for retrieval: {text}"
        )

