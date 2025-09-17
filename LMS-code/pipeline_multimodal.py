import os
import base64
from pathlib import Path
from typing import List, Dict, Any

from openai import OpenAI
from dotenv import load_dotenv
from langchain.docstore.document import Document

# Import helpers
from upsert_helpers import (
    add_text_document,
    add_image_document,
    add_multimodal_document,
    add_video_document,
    vector_db,
    model,
)
from pdf_utils import add_pdf_document  # NEW: for PDF support

# ------------- CONFIG -------------
TOP_K_DEFAULT = 3
MULTIMODAL_MODEL = os.getenv("MULTIMODAL_MODEL", "gpt-4o-mini")

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------- BULK LOADERS -------------
def load_images_from_folder(folder_path: str):
    folder = Path(folder_path)
    for img_file in folder.glob("*.*"):
        if img_file.suffix.lower() in [
            ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"
        ]:
            metadata = {"id": img_file.stem, "title": img_file.name}
            add_image_document(str(img_file), metadata, caption="Dataset image")

def load_texts_from_folder(folder_path: str):
    folder = Path(folder_path)
    for txt_file in folder.glob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            text_content = f.read().strip()
        metadata = {"id": txt_file.stem, "title": txt_file.name}
        add_text_document(text_content, metadata)

def load_pdfs_from_folder(folder_path: str):
    folder = Path(folder_path)
    for pdf_file in folder.glob("*.pdf"):
        metadata = {"id": pdf_file.stem, "title": pdf_file.name}
        add_pdf_document(str(pdf_file), metadata)

# ------------- LLM HELPERS -------------
def encode_image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def gpt4o_answer_over_images(question: str, image_paths: List[str]) -> str:
    contents = [{"type": "text", "text": question}]
    for path in image_paths:
        base64_img = encode_image_to_base64(path)
        contents.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
        })
    resp = client.chat.completions.create(
        model=MULTIMODAL_MODEL, messages=[{"role": "user", "content": contents}]
    )
    return resp.choices[0].message.content

def llm_fallback(question: str, context: str = "") -> str:
    prompt = f"Question: {question}\n\nContext:\n{context}"
    resp = client.chat.completions.create(
        model=MULTIMODAL_MODEL, messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip()

# ------------- QUERY PIPELINE -------------
def retrieve_top_k_by_text(query: str, k: int = TOP_K_DEFAULT):
    query_embedding = model.get_text_embedding(query).tolist()
    results = vector_db._collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    docs, scores = [], []
    for i, doc_text in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i] if results["metadatas"] else {}
        docs.append(Document(page_content=doc_text, metadata=meta))
        scores.append(results["distances"][0][i])
    return docs, scores

def multimodal_rag_answer(question: str, top_k: int = TOP_K_DEFAULT) -> Dict[str, Any]:
    docs, scores = retrieve_top_k_by_text(question, k=top_k)

    image_paths: List[str] = []
    for d in docs:
        fp = d.metadata.get("file_path") if d.metadata else None
        if fp and Path(fp).exists() and fp.lower().endswith((".jpg", ".png", ".jpeg")):
            image_paths.append(fp)

    if image_paths:
        answer = gpt4o_answer_over_images(question, image_paths)
    else:
        context_snippets = "\n\n".join([d.page_content or "" for d in docs])
        answer = llm_fallback(question, context_snippets)

    image_urls = [d.metadata.get("image_url") for d in docs if d.metadata]

    return {
        "answer": answer,
        "images": image_urls,
        "local_paths": image_paths,
        "scores": scores,
        "docs": [d.page_content for d in docs],
    }





# # ------------- BULK LOADERS -------------
# def load_images_from_folder(folder_path: str):
#     folder = Path(folder_path)
#     for img_file in folder.glob("*.*"):
#         if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"]:
#             metadata = {"id": img_file.stem, "title": img_file.name}
#             add_image_document(str(img_file), metadata, caption="Dataset image")

# def load_texts_from_folder(folder_path: str):
#     folder = Path(folder_path)
#     for txt_file in folder.glob("*.txt"):
#         with open(txt_file, "r", encoding="utf-8") as f:
#             text_content = f.read().strip()
#         metadata = {"id": txt_file.stem, "title": txt_file.name}
#         add_text_document(text_content, metadata)

# # ------------- LLM HELPERS -------------
# def encode_image_to_base64(path: str) -> str:
#     with open(path, "rb") as f:
#         return base64.b64encode(f.read()).decode("utf-8")

# def gpt4o_answer_over_images(question: str, image_paths: List[str]) -> str:
#     contents = [{"type": "text", "text": question}]
#     for path in image_paths:
#         base64_img = encode_image_to_base64(path)
#         contents.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}})
#     resp = client.chat.completions.create(
#         model=MULTIMODAL_MODEL, messages=[{"role": "user", "content": contents}]
#     )
#     return resp.choices[0].message.content

# def llm_fallback(question: str, context: str = "") -> str:
#     prompt = f"Question: {question}\n\nContext:\n{context}"
#     resp = client.chat.completions.create(
#         model=MULTIMODAL_MODEL, messages=[{"role": "user", "content": prompt}]
#     )
#     return resp.choices[0].message.content.strip()

# # ------------- QUERY PIPELINE -------------
# def retrieve_top_k_by_text(query: str, k: int = TOP_K_DEFAULT):
#     query_embedding = model.get_text_embedding(query).tolist()
#     results = vector_db._collection.query(
#         query_embeddings=[query_embedding],
#         n_results=k,
#         include=["documents", "metadatas", "distances"],
#     )

#     docs, scores = [], []
#     for i, doc_text in enumerate(results["documents"][0]):
#         meta = results["metadatas"][0][i] if results["metadatas"] else {}
#         docs.append(Document(page_content=doc_text, metadata=meta))
#         scores.append(results["distances"][0][i])
#     return docs, scores

# def multimodal_rag_answer(question: str, top_k: int = TOP_K_DEFAULT) -> Dict[str, Any]:
#     docs, scores = retrieve_top_k_by_text(question, k=top_k)

#     image_paths: List[str] = []
#     for d in docs:
#         fp = d.metadata.get("file_path") if d.metadata else None
#         if fp and Path(fp).exists():
#             image_paths.append(fp)

#     if image_paths:
#         answer = gpt4o_answer_over_images(question, image_paths)
#     else:
#         context_snippets = "\n\n".join([d.page_content or "" for d in docs])
#         answer = llm_fallback(question, context_snippets)

#     image_urls = [d.metadata.get("image_url") for d in docs if d.metadata]

#     return {"answer": answer, "images": image_urls, "local_paths": image_paths, "scores": scores, "docs": [d.page_content for d in docs]}

# ------------- MAIN -------------
# if __name__ == "__main__":
#     from upsert_helpers import load_images_from_folder, load_texts_from_folder, add_video_document
#     from rag_query import multimodal_rag_answer   # assuming you split query into its own file

#     image_folder = r"./video_image_frames"
#     text_folder = r"./audio_text"
#     video_file = r"./sample_video.mp4"

#     # Ingest
#     load_images_from_folder(image_folder)
#     load_texts_from_folder(text_folder)
#     add_video_document(video_file, {"id": "vid1", "title": "Sample Video"}, fps=1)

#     print("\n Ready for Q/A. Type 'exit' to quit.\n")
#     while True:
#         query = input("Enter your query: ").strip()
#         if query.lower() == "exit":
#             break
#         result = multimodal_rag_answer(query)
#         print("\nAnswer:", result["answer"])
#         print("Images:", result["images"])
#         print("Local paths:", result["local_paths"])
#         print("Scores:", result["scores"], "\n")
