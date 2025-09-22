from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import shutil
import os
from pathlib import Path
from typing import List, Optional
import uvicorn 
import requests
from pydantic import BaseModel

# NEW: Redis
#import redis

# ----------------- Processing helpers -----------------
# Atomic upsert helpers and the Chroma DB live here (no circular import)
from upsert_helpers import (
    add_image_document,
    add_video_document,
    add_text_document,
    vector_db,
)

# High-level pipeline functions (RAG + loaders)
from pipeline_multimodal import (
    multimodal_rag_answer,
    load_images_from_folder,   # optional, if you want CLI/bulk loader access
    load_pdfs_from_folder,     # optional
)

# PDF extraction helper (calls add_text_document internally)
from pdf_utils import add_pdf_document


app = FastAPI(title="Multimodal RAG API", version="1.0")

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# # ---------- REDIS CHAT HISTORY ----------
# # NEW: connect to Redis (adjust host/port if in Docker)
# redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

# # NEW: helper to namespace keys per project + college + user
# def make_chat_key(college_id: str, user_id: str) -> str:
#     return f"rag:college:{college_id}:user:{user_id}:history"


# NEW: request body schema
class RAGRequest(BaseModel):
    question: str
    top_k: int = 3
    college_id: str = "default_college"
    user_id: str = "default_user"


# ---------- INSERT DATA ----------
@app.post("/insert/image")
async def insert_image(file: UploadFile, title: Optional[str] = None):
    filepath = UPLOAD_DIR / file.filename
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    metadata = {"id": filepath.stem, "title": title or file.filename}
    add_image_document(str(filepath), metadata, caption="Uploaded image")

    return {"status": "success", "file": str(filepath)}


# @app.post("/insert/text")
# async def insert_text(content: str = Form(...), title: Optional[str] = None):
#     metadata = {"id": str(hash(content)), "title": title or "text-snippet"}
#     add_text_document(content, metadata)
#     return {"status": "success", "content": content[:50]}


@app.post("/insert/video")
async def insert_video(file: UploadFile, title: Optional[str] = None, fps: int = 1):
    filepath = UPLOAD_DIR / file.filename
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    metadata = {"id": filepath.stem, "title": title or file.filename}
    add_video_document(str(filepath), metadata, fps=fps)

    return {"status": "success", "file": str(filepath)}


@app.post("/insert/pdf")
async def insert_pdf(file: UploadFile, title: Optional[str] = None):
    filepath = UPLOAD_DIR / file.filename
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    metadata = {"id": filepath.stem, "title": title or file.filename}
    add_pdf_document(str(filepath), metadata, add_text_document)

    return {"status": "success", "file": str(filepath)}


#----Document ingestion through Supabase---------------(not used in local for now!)

@app.post("/supabase/ingest")
async def ingest_from_supabase(file_url: str, file_type: str, title: Optional[str] = None):
    filepath = UPLOAD_DIR / Path(file_url).name
    r = requests.get(file_url)
    with open(filepath, "wb") as f:
        f.write(r.content)

    metadata = {"id": filepath.stem, "title": title or filepath.name}

    if file_type == "pdf":
        add_pdf_document(str(filepath), metadata, add_text_document)
    elif file_type == "image":
        add_image_document(str(filepath), metadata, caption="From Supabase")
    elif file_type == "video":
        add_video_document(str(filepath), metadata, fps=1)
    else:
        return {"status": "error", "message": "Unsupported file type"}

    return {"status": "success", "file": str(filepath)}



# ---------- VIEW CONTENT ----------
@app.get("/chroma/list")
async def list_chroma(limit: int = 10):
    results = vector_db._collection.get(limit=100)
    return {
        "ids": results["ids"],
        "metadatas": results["metadatas"],
        "documents": results["documents"],
    }


# ---------- DELETE CONTENT ----------
@app.delete("/chroma/delete")
async def delete_chroma(ids: Optional[List[str]] = None):
    if ids:
        vector_db._collection.delete(ids=ids)
        return {"status": "deleted", "ids": ids}
    else:
        # Fetch all existing IDs
        all_data = vector_db._collection.get(include=[])
        all_ids = all_data.get("ids", [])
        if all_ids:
            vector_db._collection.delete(ids=all_ids)
        return {"status": "cleared all"}




# ---------- 1:- RAG ANSWERING ----------
# @app.post("/rag/answer")
# async def rag_answer(question: str, top_k: int = 3):
#     result = multimodal_rag_answer(question, top_k=top_k)
#     return JSONResponse(result)


# ---------- 2:- RAG ANSWERING WITH CHAT HISTORY ----------
# @app.post("/rag/answer")
# async def rag_answer(
#     question: str,
#     top_k: int = 3,
#     college_id: str = "default_college",  # NEW
#     user_id: str = "default_user"         # NEW
# ):
#     # NEW: fetch chat history
#     chat_key = make_chat_key(college_id, user_id)
#     history = redis_client.lrange(chat_key, 0, -1) or []

#     # NEW: include history context if available
#     context = "\n".join(history[-5:])  # last 5 turns for context

#     # Existing retrieval logic
#     result = multimodal_rag_answer(question, top_k=top_k)

#     # NEW: store question+answer in history
#     redis_client.rpush(chat_key, f"User: {question}")
#     redis_client.rpush(chat_key, f"Assistant: {result['answer']}")

#     # Return with history
#     return JSONResponse({
#         **result,
#         "chat_history": history + [f"User: {question}", f"Assistant: {result['answer']}"]
#     })


# ---------- 3:- RAG ANSWERING WITH CHAT HISTORY ----------
@app.post("/rag/answer")
async def rag_answer(req: RAGRequest):  # 🔹 CHANGED: accept the request model
    # 🔹 CHANGED: build session_id from college_id + user_id
    session_id = f"{req.college_id}:{req.user_id}"

    # 🔹 CHANGED: call pipeline with session_id (pipeline expects session_id)
    result = multimodal_rag_answer(req.question, session_id=session_id, top_k=req.top_k)

    # The pipeline now handles reading/saving history in Redis and returns full result
    return JSONResponse(result)


# Entry point for running with: python api_multimodal.py
# --------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "api_multimodal:app",  # filename:variable_name
        host="127.0.0.1",
        port=8000,
        reload=True
    )