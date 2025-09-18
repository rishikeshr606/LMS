
import os
import base64
import shutil
import time
import gc
import hashlib   # 🔹 CHANGED
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
from langchain_chroma import Chroma
from langchain.docstore.document import Document

from model_manager import ModelManager
from frame_utils import extract_frames   

# ------------- CONFIG -------------
CHROMA_DB_PATH = "./chroma_db_multimodal"
TOP_K_DEFAULT = 3
MULTIMODAL_MODEL = os.getenv("MULTIMODAL_MODEL", "gpt-4o-mini")

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------- INIT MODEL & DB -------------
model = ModelManager()

# if os.path.exists(CHROMA_DB_PATH):
#     shutil.rmtree(CHROMA_DB_PATH)

# def safe_rmtree(path, retries=5, delay=1):
#     for i in range(retries):
#         try:
#             if os.path.exists(path):
#                 shutil.rmtree(path)
#             break
#         except PermissionError:
#             time.sleep(delay)

vectordb = None

gc.collect()

# REMOVE THIS: it was deleting DB on every run
#safe_rmtree(CHROMA_DB_PATH)


vector_db = Chroma(
    collection_name="coca_multimodal",
    persist_directory=CHROMA_DB_PATH,
    embedding_function=None,  # manual embedding
)


# 🔹 CHANGED: Utility to compute hash from text or file content
def compute_content_hash(content: str = None, file_path: str = None) -> str:
    hasher = hashlib.sha256()
    if content is not None:
        hasher.update(content.encode("utf-8"))
    elif file_path is not None and os.path.exists(file_path):
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
    return hasher.hexdigest()


# ------------- UPSERT HELPERS -------------
def add_text_document(text: str, metadata: Dict[str, Any]):
    #doc_id = metadata.get("id") or metadata.get("title") or str(hash(text))
    doc_id = compute_content_hash(content=text)   # 🔹 CHANGED
    emb = model.get_text_embedding(text)
    vector_db._collection.upsert(
        ids=[doc_id], embeddings=[emb], metadatas=[metadata], documents=[text]
    )
    print(f"Inserted TEXT doc '{doc_id}'")

def add_image_document(image_path: str, metadata: Dict[str, Any], caption: str = ""):
    #doc_id = metadata.get("id") or Path(image_path).stem
    doc_id = compute_content_hash(file_path=image_path)   # 🔹 CHANGED
    meta = dict(metadata)
    meta["file_path"] = str(Path(image_path).resolve())
    meta["image_url"] = f"/static/{Path(image_path).name}"

    emb = model.get_image_embedding(image_path)
    vector_db._collection.upsert(
        ids=[doc_id], embeddings=[emb], metadatas=[meta], documents=[caption]
    )
    print(f"Inserted IMAGE doc '{doc_id}'")

def add_multimodal_document(text: str, image_path: str, metadata: Dict[str, Any]):
    #doc_id = metadata.get("id") or str(hash(text + image_path))

    # 🔹 CHANGED: combined hash of text + image content
    text_hash = compute_content_hash(content=text)
    image_hash = compute_content_hash(file_path=image_path)
    doc_id = compute_content_hash(content=text_hash + image_hash)

    meta = dict(metadata)
    meta["file_path"] = str(Path(image_path).resolve())
    meta["image_url"] = f"/static/{Path(image_path).name}"

    emb = model.get_text_image_embedding(text, image_path)
    vector_db._collection.upsert(
        ids=[doc_id], embeddings=[emb], metadatas=[meta], documents=[text]
    )
    print(f"Inserted MULTIMODAL doc '{doc_id}'")

def add_video_document(video_path: str, metadata: Dict[str, Any], fps: int = 1):
    """Extract frames from video, embed each, and upsert."""
    frames_dir = Path(video_path).with_suffix("").as_posix() + "_frames"
    os.makedirs(frames_dir, exist_ok=True)

    extract_frames(video_path, frames_dir, fps)

    for frame_file in Path(frames_dir).glob("*.jpg"):
        frame_metadata = {**metadata, "frame": frame_file.name, "video": video_path}
        add_image_document(str(frame_file), frame_metadata, caption="Video frame")
