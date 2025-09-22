import os
import base64
from pathlib import Path
from typing import List, Dict, Any

from openai import OpenAI
from dotenv import load_dotenv
from langchain.docstore.document import Document
import redis  ### NEW

# Import helpers
from upsert_helpers import (
    add_text_document,
    add_image_document,
    add_multimodal_document,
    add_video_document,
    vector_db,
    model,
    compute_content_hash,   # NEW: import content hashing
)
from pdf_utils import add_pdf_document  # for PDF support

# ------------- CONFIG -------------
TOP_K_DEFAULT = 3
MULTIMODAL_MODEL = os.getenv("MULTIMODAL_MODEL", "gpt-4o-mini")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")  ### NEW
MAX_HISTORY_TURNS = 5  ### NEW

### CHANGE: centralized relevance threshold
#RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", 0.75))

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)  ### NEW

# ------------- BULK LOADERS -------------
def load_images_from_folder(folder_path: str):
    folder = Path(folder_path)
    for img_file in folder.glob("*.*"):
        if img_file.suffix.lower() in [
            ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"
        ]:

            # NEW: compute hash-based ID
            doc_id = compute_content_hash(str(img_file))
            # NEW: check if already in DB
            existing = vector_db._collection.get(ids=[doc_id])
            if existing and existing["ids"]:
                print(f"Skipping IMAGE (already exists): {img_file.name}")
                continue

            metadata = {"id": img_file.stem, "title": img_file.name}
            add_image_document(str(img_file), metadata, caption="Dataset image")

def load_texts_from_folder(folder_path: str):
    folder = Path(folder_path)
    for txt_file in folder.glob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            text_content = f.read().strip()

        # NEW: compute hash
        doc_id = compute_content_hash(text_content)
        # NEW: check if already in DB
        existing = vector_db._collection.get(ids=[doc_id])
        if existing and existing["ids"]:
            print(f"Skipping TEXT (already exists): {txt_file.name}")
            continue

        metadata = {"id": txt_file.stem, "title": txt_file.name}
        add_text_document(text_content, metadata)

def load_pdfs_from_folder(folder_path: str):
    folder = Path(folder_path)
    for pdf_file in folder.glob("*.pdf"):

        # NEW: hash on file content
        doc_id = compute_content_hash(str(pdf_file))
        existing = vector_db._collection.get(ids=[doc_id])
        if existing and existing["ids"]:
            print(f"Skipping PDF (already exists): {pdf_file.name}")
            continue

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


# ------------- CHAT HISTORY HELPERS -------------  ### NEW
def save_turn(session_id: str, role: str, content: str):
    """Append a chat turn to Redis list"""
    redis_client.rpush(f"chat:{session_id}", f"{role}:{content}")

def get_history(session_id: str, max_turns: int = MAX_HISTORY_TURNS):
    """Fetch last N turns"""
    key = f"chat:{session_id}"
    turns = redis_client.lrange(key, -max_turns * 2, -1)  # each Q/A is 2 turns
    history = []
    for t in turns:
        if t.startswith("user:"):
            history.append({"role": "user", "content": t[5:]})
        elif t.startswith("assistant:"):
            history.append({"role": "assistant", "content": t[10:]})
    return history


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


# ------4 ----------

def multimodal_rag_answer(
    question: str, session_id: str, top_k: int = TOP_K_DEFAULT
) -> Dict[str, Any]:
    # --- retrieve
    docs, scores = retrieve_top_k_by_text(question, k=top_k)

    # --- CHANGE 1: check Chroma metric to handle distances vs similarities
    try:
        chroma_metric = vector_db._collection.get().get("metric", "cosine")
        print(f"[DEBUG] Chroma metric: {chroma_metric}")
    except Exception as e:
        print(f"[DEBUG] Could not read Chroma metric, defaulting to cosine. Error: {e}")
        chroma_metric = "cosine"

    normalized_scores = []
    # --- CHANGE 2: normalize scores based on metric
    for s in scores:
        if chroma_metric in ["cosine", "dot"]:
            # larger = better
            sim = max(0.0, min(1.0, s))  # assume already in [0,1]
        else:
            # distance metrics: smaller = better
            sim = max(0.0, min(1.0, 1.0 - s))  # normalize to [0,1]
        normalized_scores.append(sim)

    # --- CHANGE 3: filter relevant docs using centralized threshold
    RELEVANCE_THRESHOLD = 0.75  ### CHANGED: define threshold here
    relevant_docs = [d for d, s in zip(docs, normalized_scores) if s >= RELEVANCE_THRESHOLD]

    # --- CHANGE 4: early return if no relevant docs
    if not relevant_docs:
        answer = "I don’t know based on the provided documents."
        save_turn(session_id, "user", question)
        save_turn(session_id, "assistant", answer)
        return {
            "answer": answer,
            "images": [],
            "local_paths": [],
            "scores": normalized_scores,
            "docs": [],
            "history": get_history(session_id) + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
        }

    # --- history
    history = get_history(session_id)

    # --- context
    image_paths: List[str] = []
    for d in relevant_docs:
        fp = d.metadata.get("file_path") if d.metadata else None
        if fp and Path(fp).exists() and fp.lower().endswith((".jpg", ".png", ".jpeg")):
            image_paths.append(fp)

    # --- multimodal input with strict prompt
    if image_paths:
        contents = [{"type": "text", "text": f"""
You are a helpful assistant. Answer the question using ONLY the provided documents and images.
- Quote or summarize from text.
- For images, mention relevant info if possible.
- Do NOT make up information.
- If nothing in the documents/images answers the question, reply exactly:
"I don’t know based on the provided documents."

Question: {question}
"""}]
        for path in image_paths:
            base64_img = encode_image_to_base64(path)
            contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
            })
        messages = history + [{"role": "user", "content": contents}]
        resp = client.chat.completions.create(model=MULTIMODAL_MODEL, messages=messages)
        answer = resp.choices[0].message.content.strip()
    else:
        # --- STRICT PROMPT for text-only docs
        context_snippets = "\n\n".join([d.page_content or "" for d in relevant_docs])
        prompt = f"""
You are a helpful assistant. Answer the user’s question using ONLY the following documents.
- Quote or summarize from the documents wherever possible.
- Do NOT make up information.
- If the answer cannot be found in the documents, reply exactly:
"I don’t know based on the provided documents."

Documents:
{context_snippets}

Question: {question}
Answer:
"""
        messages = history + [{"role": "user", "content": prompt}]
        resp = client.chat.completions.create(model=MULTIMODAL_MODEL, messages=messages)
        answer = resp.choices[0].message.content.strip()

    ### 🔹 CHANGED 5: improved guardrail for videos + images
    has_text_context = any(
        d.page_content and d.page_content.strip() != "" and "video frame" not in d.page_content.lower()
        for d in relevant_docs
    )
    has_image_context = any(d.metadata.get("image_url") for d in relevant_docs)

    if not has_text_context and not has_image_context:
        print("[DEBUG] Guardrail triggered: no text or image context found.")
        answer = "I don’t know based on the provided documents."

    # --- save turn
    save_turn(session_id, "user", question)
    save_turn(session_id, "assistant", answer)

    # --- only include image URLs from relevant docs
    image_urls = [d.metadata.get("image_url") for d in relevant_docs if d.metadata]

    return {
        "answer": answer,
        "images": image_urls,
        "local_paths": image_paths,
        "scores": normalized_scores,
        "docs": [d.page_content for d in relevant_docs],
        "history": history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
    }





#-----3 --------------

# def multimodal_rag_answer(
#     question: str, session_id: str, top_k: int = TOP_K_DEFAULT
# ) -> Dict[str, Any]:
#     # --- retrieve
#     docs, scores = retrieve_top_k_by_text(question, k=top_k)

#     # --- CHANGE 1: check Chroma metric to handle distances vs similarities
#     try:
#         chroma_metric = vector_db._collection.get().get("metric", "cosine")
#         print(f"[DEBUG] Chroma metric: {chroma_metric}")
#     except Exception as e:
#         print(f"[DEBUG] Could not read Chroma metric, defaulting to cosine. Error: {e}")
#         chroma_metric = "cosine"

#     normalized_scores = []
#     # --- CHANGE 2: normalize scores based on metric
#     for s in scores:
#         if chroma_metric in ["cosine", "dot"]:
#             # larger = better
#             sim = max(0.0, min(1.0, s))  # assume already in [0,1]
#         else:
#             # distance metrics: smaller = better
#             sim = max(0.0, min(1.0, 1.0 - s))  # normalize to [0,1]
#         normalized_scores.append(sim)

#     # --- CHANGE 3: filter relevant docs using centralized threshold
#     RELEVANCE_THRESHOLD = 0.75  ### CHANGED: define threshold here
#     relevant_docs = [d for d, s in zip(docs, normalized_scores) if s >= RELEVANCE_THRESHOLD]

#     # --- CHANGE 4: early return if no relevant docs
#     if not relevant_docs:
#         answer = "I don’t know based on the provided documents."
#         save_turn(session_id, "user", question)
#         save_turn(session_id, "assistant", answer)
#         return {
#             "answer": answer,
#             "images": [],
#             "local_paths": [],
#             "scores": normalized_scores,
#             "docs": [],
#             "history": get_history(session_id) + [
#                 {"role": "user", "content": question},
#                 {"role": "assistant", "content": answer},
#             ],
#         }

#     # --- history
#     history = get_history(session_id)

#     # --- context
#     image_paths: List[str] = []
#     for d in relevant_docs:
#         fp = d.metadata.get("file_path") if d.metadata else None
#         if fp and Path(fp).exists() and fp.lower().endswith((".jpg", ".png", ".jpeg")):
#             image_paths.append(fp)

#     # --- multimodal input with strict prompt
#     if image_paths:
#         contents = [{"type": "text", "text": f"""
# You are a helpful assistant. Answer the question using ONLY the provided documents and images.
# - Quote or summarize from text.
# - For images, mention relevant info if possible.
# - Do NOT make up information.
# - If nothing in the documents/images answers the question, reply exactly:
# "I don’t know based on the provided documents."

# Question: {question}
# """}]
#         for path in image_paths:
#             base64_img = encode_image_to_base64(path)
#             contents.append({
#                 "type": "image_url",
#                 "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
#             })
#         messages = history + [{"role": "user", "content": contents}]
#         resp = client.chat.completions.create(model=MULTIMODAL_MODEL, messages=messages)
#         answer = resp.choices[0].message.content.strip()
#     else:
#         # --- STRICT PROMPT for text-only docs
#         context_snippets = "\n\n".join([d.page_content or "" for d in relevant_docs])
#         prompt = f"""
# You are a helpful assistant. Answer the user’s question using ONLY the following documents.
# - Quote or summarize from the documents wherever possible.
# - Do NOT make up information.
# - If the answer cannot be found in the documents, reply exactly:
# "I don’t know based on the provided documents."

# Documents:
# {context_snippets}

# Question: {question}
# Answer:
# """
#         messages = history + [{"role": "user", "content": prompt}]
#         resp = client.chat.completions.create(model=MULTIMODAL_MODEL, messages=messages)
#         answer = resp.choices[0].message.content.strip()

#     ### CHANGED 5: improved guardrail - soft semantic check instead of substring
#     combined_text = " ".join([d.page_content for d in relevant_docs]).lower()
#     question_words = [w.lower() for w in question.split() if len(w) > 3]
#     if question_words and not any(word in combined_text for word in question_words):
#         print("[DEBUG] Guardrail triggered: question words not found in retrieved docs.")
#         answer = "I don’t know based on the provided documents."

#     # --- save turn
#     save_turn(session_id, "user", question)
#     save_turn(session_id, "assistant", answer)

#     # --- only include image URLs from relevant docs
#     image_urls = [d.metadata.get("image_url") for d in relevant_docs if d.metadata]

#     return {
#         "answer": answer,
#         "images": image_urls,
#         "local_paths": image_paths,
#         "scores": normalized_scores,
#         "docs": [d.page_content for d in relevant_docs],
#         "history": history + [
#             {"role": "user", "content": question},
#             {"role": "assistant", "content": answer},
#         ],
#     }



#-----2 ------------

# def multimodal_rag_answer(
#     question: str, session_id: str, top_k: int = TOP_K_DEFAULT
# ) -> Dict[str, Any]:  ### CHANGED (added session_id)
#     # --- retrieve
#     docs, scores = retrieve_top_k_by_text(question, k=top_k)

#     # --- history
#     history = get_history(session_id)  ### NEW

#     # --- context
#     image_paths: List[str] = []
#     for d in docs:
#         fp = d.metadata.get("file_path") if d.metadata else None
#         if fp and Path(fp).exists() and fp.lower().endswith((".jpg", ".png", ".jpeg")):
#             image_paths.append(fp)

#     if image_paths:
#         # multimodal input
#         contents = [{"type": "text", "text": question}]
#         for path in image_paths:
#             base64_img = encode_image_to_base64(path)
#             contents.append({
#                 "type": "image_url",
#                 "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
#             })
#         messages = history + [{"role": "user", "content": contents}]  ### NEW
#         resp = client.chat.completions.create(model=MULTIMODAL_MODEL, messages=messages)
#         answer = resp.choices[0].message.content
#     else:
#         context_snippets = "\n\n".join([d.page_content or "" for d in docs])
#         prompt = f"Question: {question}\n\nContext:\n{context_snippets}"
#         messages = history + [{"role": "user", "content": prompt}]  ### NEW
#         resp = client.chat.completions.create(model=MULTIMODAL_MODEL, messages=messages)
#         answer = resp.choices[0].message.content.strip()

#     # --- save turn
#     save_turn(session_id, "user", question)     ### NEW
#     save_turn(session_id, "assistant", answer)  ### NEW

#     image_urls = [d.metadata.get("image_url") for d in docs if d.metadata]

#     return {
#         "answer": answer,
#         "images": image_urls,
#         "local_paths": image_paths,
#         "scores": scores,
#         "docs": [d.page_content for d in docs],
#         "history": history + [
#             {"role": "user", "content": question},
#             {"role": "assistant", "content": answer},
#         ],  ### NEW
#     }




#---------1 :-------------

# def multimodal_rag_answer(question: str, top_k: int = TOP_K_DEFAULT) -> Dict[str, Any]:
#     docs, scores = retrieve_top_k_by_text(question, k=top_k)

#     image_paths: List[str] = []
#     for d in docs:
#         fp = d.metadata.get("file_path") if d.metadata else None
#         if fp and Path(fp).exists() and fp.lower().endswith((".jpg", ".png", ".jpeg")):
#             image_paths.append(fp)

#     if image_paths:
#         answer = gpt4o_answer_over_images(question, image_paths)
#     else:
#         context_snippets = "\n\n".join([d.page_content or "" for d in docs])
#         answer = llm_fallback(question, context_snippets)

#     image_urls = [d.metadata.get("image_url") for d in docs if d.metadata]

#     return {
#         "answer": answer,
#         "images": image_urls,
#         "local_paths": image_paths,
#         "scores": scores,
#         "docs": [d.page_content for d in docs],
#     }





