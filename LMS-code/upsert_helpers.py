
import os
import base64
import shutil
import time
import gc
import hashlib   
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

# Initialize vector DB
vector_db = Chroma(
    collection_name="coca_multimodal",
    persist_directory=CHROMA_DB_PATH,
    embedding_function=None  # manual embedding
)

# CHANGED: check metric and default to 'cosine' if not set
# try:
#     current_metric = vector_db._collection.get()['metric']
#     if not current_metric:
#         # update to cosine if metric is None
#         vector_db._collection.set_metric("cosine")
#         print("Metric was None. Set to 'cosine'.")
#     else:
#         print(f"Vector DB metric: {current_metric}")
# except Exception as e:
#     print(f"Warning: could not check metric, defaulting to cosine. Error: {e}")
#     vector_db._collection.set_metric("cosine")


# CHANGED: Utility to compute hash from text or file content
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
    #vector_db.persist()  # persist immediately
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
    #vector_db.persist()  # persist immediately
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
    #vector_db.persist()  # persist immediately
    print(f"Inserted MULTIMODAL doc '{doc_id}'")


#-----2 ----------------------

def add_video_document(video_path: str, metadata: Dict[str, Any], fps: int = 1):
    """
    Extract frames + audio text from video, embed each, and upsert.
    🔹 CHANGED: now includes speech-to-text per video
    """
    frames_dir = Path(video_path).with_suffix("").as_posix() + "_frames"
    os.makedirs(frames_dir, exist_ok=True)

    # --- Extract frames
    extract_frames(video_path, frames_dir, fps)

    # --- NEW: extract audio transcript for video
    try:
        audio_text = model.transcribe_video(video_path)  # 🔹 requires ModelManager.transcribe_video
    except Exception as e:
        print(f"[WARN] Could not extract audio text: {e}")
        audio_text = ""

    # --- NEW: upsert audio transcript as separate text doc
    if audio_text.strip():
        print(f"[DEBUG] Transcript extracted:\n{audio_text}\n")  # 🔹 ADD THIS PRINT
        transcript_metadata = {**metadata, "video": video_path, "type": "transcript"}  # 🔹 mark as transcript
        add_text_document(audio_text, transcript_metadata)  # 🔹 store transcript separately
        print(f"[DEBUG] Inserted transcript for video {video_path}")  # 🔹

    # --- Upsert frames as image docs with captions
    for frame_file in Path(frames_dir).glob("*.jpg"):
        # 🔹 compute hash for frame
        frame_id = compute_content_hash(file_path=str(frame_file))

        # 🔹 skip if already exists
        existing = vector_db._collection.get(ids=[frame_id])
        if existing and existing["ids"]:
            print(f"Skipping FRAME (already exists): {frame_file.name}")
            continue

        # 🔹 include audio_text in caption for multimodal guardrail
        frame_metadata = {
            **metadata,
            "frame": frame_file.name,
            "video": video_path,
            "id": frame_id
        }
        # 🔹 optionally include a snippet if transcript is too long
        snippet = audio_text[:200] + "..." if audio_text and len(audio_text) > 200 else audio_text
        caption = f"Video frame. Transcript snippet: {snippet}" if snippet else "Video frame"
        add_image_document(str(frame_file), frame_metadata, caption=caption)

#-----1 -------------------

# def add_video_document(video_path: str, metadata: Dict[str, Any], fps: int = 1):
#     """Extract frames from video, embed each, and upsert."""
#     frames_dir = Path(video_path).with_suffix("").as_posix() + "_frames"
#     os.makedirs(frames_dir, exist_ok=True)

#     extract_frames(video_path, frames_dir, fps)

#     for frame_file in Path(frames_dir).glob("*.jpg"):

#         # CHANGED: compute hash for each frame
#         frame_id = compute_content_hash(file_path=str(frame_file))

#         # CHANGED: check if already in DB
#         existing = vector_db._collection.get(ids=[frame_id])
#         if existing and existing["ids"]:
#             print(f"Skipping FRAME (already exists): {frame_file.name}")
#             continue

#         frame_metadata = {**metadata, "frame": frame_file.name, "video": video_path, "id": frame_id}
#         add_image_document(str(frame_file), frame_metadata, caption="Video frame")
