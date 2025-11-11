from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional
import uvicorn
import base64
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os
from pathlib import Path
#import faiss
import sys
import uvicorn

# 🔐 Symmetric encryption key (must be securely shared after attestation)
AES_KEY = os.environ.get("RAG_AES_KEY")  # 256-bit key as base64

app = FastAPI()

# HACK: Disable encryption for now
do_encryption = False

# 🧠 Load tokenizer + model
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 🔒 FHE KNN support
USE_FHE_KNN = os.environ.get("USE_FHE_KNN", "false").lower() == "true"
fhe_rag = None  # Will be initialized if USE_FHE_KNN is True


class PromptedBGE(HuggingFaceEmbeddings):

    def embed_documents(self, texts):
        return super().embed_documents(
            [f"Represent this document for retrieval: {t}" for t in texts])

    def embed_query(self, text):
        return super().embed_query(
            f"Represent this query for retrieval: {text}")


# BAAI_embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en")

BAAI_embedding = PromptedBGE(model_name="BAAI/bge-base-en")  # or bge-large-en

# 📚 Load FAISS index (in-memory)
# You'd load your document embeddings here

### LOAD RAG
FAISS_PATH = Path("/Users/roy/data/wikipedia/hugging_face/")
FAISS_PATH = FAISS_PATH / "faiss_index__top_100000__2025-04-11"
FAISS_PATH = os.environ.get("FAISS_PATH", FAISS_PATH)
print(f"FAISS_PATH {FAISS_PATH}")


# 📚 Load FAISS index (path optionally provided via CLI)
def load_vectorstore(faiss_path: Optional[str] = FAISS_PATH):
    """ adjusts global vectorstore variable """
    global vectorstore
    if faiss_path is None:
        default_FAISS_PATH = Path("/Users/roy/data/wikipedia/hugging_face")
        default_FAISS_PATH = default_FAISS_PATH / "wiki_index__top_100000__2025-04-11"
        faiss_path = default_FAISS_PATH

    else:
        faiss_path = Path(faiss_path)
    print(f"loaded vector store")
    vectorstore = FAISS.load_local(faiss_path,
                                   BAAI_embedding,
                                   allow_dangerous_deserialization=True)
    return vectorstore


# Will be set in main()
vectorstore = load_vectorstore()

# Initialize FHE KNN if enabled
if USE_FHE_KNN:
    print("Initializing FHE KNN wrapper...")
    try:
        from wiki_rag.fhe_knn_rag import FHEKNNRAG
        fhe_rag = FHEKNNRAG(
            vectorstore=vectorstore,
            embeddings=BAAI_embedding,
            n_neighbors=5,
            n_bits=3,
            pca_components=50,
            compile_fhe=True,
            cache_dir=Path("fhe_knn_cache")
        )
        print("FHE KNN wrapper initialized successfully")
    except Exception as e:
        print(f"Warning: Could not initialize FHE KNN: {e}")
        print("Falling back to regular FAISS search")
        USE_FHE_KNN = False


# 🔒 Decrypt helper
def decrypt_message(enc_b64: str, key: bytes) -> str:
    data = base64.b64decode(enc_b64)
    nonce, ciphertext = data[:12], data[12:]
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, None).decode()


# 🔒 Encrypt helper
def encrypt_message(message: str, key: bytes) -> str:
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ciphertext = aesgcm.encrypt(nonce, message.encode(), None)
    return base64.b64encode(nonce + ciphertext).decode()


# 🧾 Request schema
class Query(BaseModel):
    encrypted_query: str
    fhe_mode: Optional[str] = "simulate"  # "simulate", "execute", or "clear"
    k: Optional[int] = 1  # Number of results


@app.post("/rag")
async def rag_endpoint(query: Query):
    global vectorstore, fhe_rag, USE_FHE_KNN
    
    # Get query mode from query or use default
    fhe_mode = getattr(query, 'fhe_mode', 'simulate') if hasattr(query, 'fhe_mode') else 'simulate'
    k = getattr(query, 'k', 1) if hasattr(query, 'k') else 1

    if do_encryption:
        key = base64.b64decode(AES_KEY)
        user_query = decrypt_message(query.encrypted_query, key)
    else:
        user_query = query.encrypted_query  # use plaintext

    # Use FHE KNN if enabled, otherwise use regular FAISS
    if USE_FHE_KNN and fhe_rag is not None:
        print(f"Using FHE KNN with mode: {fhe_mode}")
        responses = fhe_rag.similarity_search(user_query, k=k, fhe_mode=fhe_mode)
    else:
        print("Using regular FAISS search")
        responses = vectorstore.similarity_search(user_query, k=k)
    
    # Serialize responses
    serialized_responses = []
    for doc in responses:
        if isinstance(doc, Document):
            serialized_responses.append({
                "title": doc.metadata.get("title", "Unknown"),
                "content": doc.page_content[:500],  # First 500 chars
                "metadata": doc.metadata
            })
        else:
            serialized_responses.append(str(doc))
    
    # Return first result (or all results if k > 1)
    if do_encryption:
        key = base64.b64decode(AES_KEY)
        encrypted_responses = [encrypt_message(str(r), key) for r in serialized_responses]
        return {"results": encrypted_responses}
    else:
        return {"results": serialized_responses}


@app.post("/fhe_rag")
async def fhe_rag_endpoint(query: Query):
    """Dedicated endpoint for FHE KNN queries."""
    global fhe_rag, USE_FHE_KNN
    
    if not USE_FHE_KNN or fhe_rag is None:
        return {"error": "FHE KNN is not enabled. Set USE_FHE_KNN=true environment variable."}
    
    if do_encryption:
        key = base64.b64decode(AES_KEY)
        user_query = decrypt_message(query.encrypted_query, key)
    else:
        user_query = query.encrypted_query
    
    fhe_mode = query.fhe_mode if hasattr(query, 'fhe_mode') else "simulate"
    k = query.k if hasattr(query, 'k') else 5
    
    print(f"FHE KNN query: '{user_query}', mode: {fhe_mode}, k: {k}")
    
    responses = fhe_rag.similarity_search(user_query, k=k, fhe_mode=fhe_mode)
    
    # Serialize responses
    serialized_responses = []
    for doc in responses:
        if isinstance(doc, Document):
            serialized_responses.append({
                "title": doc.metadata.get("title", "Unknown"),
                "content": doc.page_content[:500],
                "metadata": doc.metadata
            })
        else:
            serialized_responses.append(str(doc))
    
    return {
        "results": serialized_responses,
        "fhe_mode": fhe_mode,
        "k": k
    }


@app.post("/test")
async def test_endpoint(query: Query):
    global vectorstore

    print(f"query {query}")
    return {"results": [f"Vectorstore has {vectorstore.index.ntotal} documents"]}


@app.post("/provision")
async def provision_key(payload: dict):
    raw = payload["aes_key"]
    os.environ["RAG_AES_KEY"] = raw
    AES_KEY = raw
    return {"status": "ok"}


def main():
    global vectorstore
    # Optional CLI arg: FAISS path
    faiss_arg = sys.argv[1] if len(sys.argv) > 1 else None
    vectorstore = load_vectorstore(faiss_arg)
    print(f"vectorstore {vectorstore}")
    uvicorn.run("rag_server:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    print("hello!")
    main()
