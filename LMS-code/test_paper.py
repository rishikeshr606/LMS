# app_simple.py
import os
import io
import sqlite3
import pdfplumber
from flask import Flask, request, jsonify, Response
import json
import pandas as pd

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

import openai
from openai import OpenAI

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
PREV_FOLDER = "Previous_Year_Question_Papers"
BOOKS_FOLDER = "Books"

app = Flask(__name__)

### --- SQLite Setup ---
def init_db():
    conn = sqlite3.connect("cache.db")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS file_cache (
            file_id TEXT PRIMARY KEY,
            text_content TEXT
        )
    """)
    # --- NEW TABLE FOR QA STORAGE ---
    cur.execute("DROP TABLE IF EXISTS question_papers")  # remove old version comment later!!!!!!!
    cur.execute("""
        CREATE TABLE IF NOT EXISTS question_papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT,
            chapter TEXT,
            difficulty TEXT,       
            questions TEXT,
            answers TEXT,
            user_answers TEXT,
            evaluation TEXT
        )
    """)  # MARKED CHANGE
    conn.commit()
    conn.close()

def get_cached_text(file_id):
    conn = sqlite3.connect("cache.db")
    cur = conn.cursor()
    cur.execute("SELECT text_content FROM file_cache WHERE file_id = ?", (file_id,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None

def set_cached_text(file_id, text):
    conn = sqlite3.connect("cache.db")
    cur = conn.cursor()
    cur.execute("REPLACE INTO file_cache (file_id, text_content) VALUES (?, ?)", (file_id, text))
    conn.commit()
    conn.close()

# --- NEW HELPER FUNCTION TO SAVE Q&A ---
def save_question_paper(subject, chapter, questions, answers):  # MARKED CHANGE
    conn = sqlite3.connect("cache.db")
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO question_papers (subject, chapter, questions, answers)
        VALUES (?, ?, ?, ?)
    """, (subject, chapter, questions, answers))
    conn.commit()
    conn.close()

    #print(f" Saved: {subject} | {chapter} | {len(questions)} chars of questions | {len(answers)} chars of answers")
# ----------------------------------------

def get_drive_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as f:
            f.write(creds.to_json())
    return build('drive', 'v3', credentials=creds)

def find_folder_id(service, name):
    q = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    res = service.files().list(q=q, fields="files(id)").execute()
    files = res.get("files", [])
    return files[0]['id'] if files else None

# --- NEW FUNCTION ---
def find_subfolder_id(service, parent_id, name):
    """
    Find a subfolder by name inside a parent folder.
    Returns the folder ID or None if not found.
    """
    q = f"'{parent_id}' in parents and name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    res = service.files().list(q=q, fields="files(id)").execute()
    files = res.get("files", [])
    return files[0]['id'] if files else None
# -------------------

def list_files(service, folder_id):
    q = f"'{folder_id}' in parents and trashed=false and mimeType='application/pdf'"
    res = service.files().list(q=q, fields="files(id,name)").execute()
    return res.get("files", [])

def download_text(service, file_id, refresh=False):
    # Check cache unless refresh requested
    if not refresh:
        cached = get_cached_text(file_id)
        if cached:
            return cached

    # Download from Drive
    request = service.files().get_media(fileId=file_id)
    bio = io.BytesIO()
    downloader = MediaIoBaseDownload(bio, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    bio.seek(0)

    text = []
    with pdfplumber.open(bio) as pdf:
        #for page in pdf.pages:
        for i, page in enumerate(pdf.pages[:5]):
            text.append(page.extract_text() or "")
    final_text = "\n".join(text)

    # Save to cache
    set_cached_text(file_id, final_text)
    return final_text

# 4 ---------------------------------------------------------

@app.route("/generate_question_paper", methods=["GET"])
def generate_question_paper():
    subject = request.args.get("subject", "pharmacology").lower()
    chapter = request.args.get("chapter", "").lower()
    difficulty = request.args.get("difficulty", "medium").lower()   
    refresh = request.args.get("refresh", "false").lower() == "true"

    # key excludes count, includes subject+chapter
    key = f"{subject}_{chapter}"

    if not refresh and key in session_cache:
        paper = session_cache[key]
        return jsonify({"subject": subject, "difficulty": difficulty, "paper": paper})

    service = get_drive_service()
    prev_id = find_folder_id(service, PREV_FOLDER)
    books_id = find_folder_id(service, BOOKS_FOLDER)

    prev_text, book_text = "", ""

    if prev_id:
        files = list_files(service, prev_id)
        subject_files = [f for f in files if subject in f['name'].lower()]
        if not subject_files:
            subject_files = files[:2]
        for f in subject_files[:2]:
            prev_text += download_text(service, f['id'], refresh=refresh) + "\n"

    if books_id:
        query = f"'{books_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        subject_folders = results.get("files", [])

        subject_folder_id = None
        for f in subject_folders:
            if f["name"].lower() == subject.lower():
                subject_folder_id = f["id"]
                break

        if subject_folder_id:
            files = list_files(service, subject_folder_id)
            subject_files = [
                f for f in files
                if chapter.replace(".pdf", "").lower() in f["name"].lower()
            ]
            if not subject_files:
                app.logger.warning(f"No files matched for chapter={chapter}, subject={subject}")
            else:
                for f in subject_files[:1]:
                    book_text += download_text(service, f["id"], refresh=refresh) + "\n"
                app.logger.info(
                    f"[DEBUG] Using file: {subject_files[0]['name']}, size={len(book_text)} chars"
                )
        else:
            app.logger.warning(f"Subject folder not found under Books for subject={subject}")

    # --- Extract keywords from the chapter ---
    keyword_prompt = f"""
    Extract a comma-separated list of the most important medical terms, 
    drug classes, and key topics from the following text.
    Only include words that are explicitly present in the text.
    Text:
    {book_text[:3000]}
    """
    client = OpenAI()
    kw_resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": keyword_prompt}],
        max_tokens=200,
        temperature=0.3
    )
    keywords = getattr(kw_resp.choices[0].message, "content", "")
    keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]

    app.logger.info(f"[DEBUG] Extracted Keywords for Chapter: {chapter}")
    app.logger.info(f"Keywords: {keywords_list}")

    # --- STEP 1: Generate Questions ---
    prompt_q = f"""
    You are an experienced medical exam setter.

    Subject: {subject}, Chapter: {chapter}.
    Difficulty Level: {difficulty}  

    --- Instructions ---
    1. Use the **book content below as the ONLY knowledge source**. 
       STRICT RULE: All questions must come **only from the given chapter**.
    2. Use the **past papers ONLY as formatting/style reference** 
       (sections, numbering, marks distribution, MCQ/short/long patterns).
    3. Limit yourself to the following extracted keywords and their context:
       {", ".join(keywords_list[:30])}
    4. Avoid repeating or rephrasing the same question.
    5. Ensure a balanced mix of essays, short notes, and MCQs.
    6. Format clearly with numbering.
    7. Generate a question paper that follows the **common structure of the provided past papers**.
       If there are differences in total question counts, choose the **most frequently occurring count**.

    --- Book Content (knowledge source) ---
    {book_text[:3000]}

    --- Past Papers (style reference only) ---
    {prev_text[:3000]}

    --- Task ---
    Generate the exam paper strictly using the chapter content 
    but mirroring the formatting and style of past papers.
    """

    resp_q = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_q}],
        max_tokens=1500,
        temperature=0.7
    )
    raw_questions = getattr(resp_q.choices[0].message, "content", "No content returned")

    # Deduplicate
    seen = set()
    final_questions = []
    for line in raw_questions.split("\n"):
        cleaned = line.strip()
        if not cleaned:
            continue
        if cleaned not in seen:
            final_questions.append(cleaned)
            seen.add(cleaned)
    questions = "\n".join(final_questions)

    # --- STEP 2: Generate Answers (debug only) ---
    prompt_a = f"""
    Provide correct, concise answers for the following exam questions.
    Only use the chapter content provided earlier (do not hallucinate):

    {questions}
    """
    resp_a = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_a}],
        max_tokens=2000,
        temperature=0.7
    )
    answers = getattr(resp_a.choices[0].message, "content", "No answers returned")

    # --- SAVE TO DB (keep answers for debugging, difficulty for progress reports) ---
    conn = sqlite3.connect("cache.db")
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO question_papers (subject, chapter, difficulty, questions, answers)
        VALUES (?, ?, ?, ?, ?)
    """, (subject, chapter, difficulty, questions, answers))
    conn.commit()
    conn.close()

    # --- RETURN ONLY QUESTIONS TO USER ---
    return Response(
        json.dumps({"subject": subject, "chapter": chapter, "difficulty": difficulty, "paper": questions}, indent=4),
        mimetype="application/json"
    )



@app.route("/submit_answers", methods=["POST"])
def submit_answers():
    data = request.json
    paper_id = data.get("paper_id")
    user_answers = data.get("answers", "")

    # Normalize list input to plain text
    if isinstance(user_answers, list):
        user_answers = "\n".join(user_answers)

    if not paper_id or not user_answers:
        return jsonify({"error": "paper_id and answers are required"}), 400

    # Fetch questions & difficulty
    conn = sqlite3.connect("cache.db")
    cur = conn.cursor()
    cur.execute("SELECT subject, chapter, difficulty, questions FROM question_papers WHERE id = ?", (paper_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return jsonify({"error": "Paper not found"}), 404

    subject, chapter, difficulty, questions = row

    # Prompt LLM to evaluate without reference
    eval_prompt = f"""
    You are an expert examiner.

    Subject: {subject}
    Chapter: {chapter}
    Difficulty Level: {difficulty}

    Questions:
    {questions}

    Student Answers:
    {user_answers}

    --- Evaluation Instructions ---
    1. Judge answers purely on accuracy, clarity, completeness, and depth of knowledge.
    2. Assign a score (0–10) per question.
    3. Provide short feedback per question.
    4. Give an overall percentage and brief remarks (strengths, weaknesses, advice).
    """

    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": eval_prompt}],
        max_tokens=3000,
        temperature=0.5
    )
    evaluation = getattr(resp.choices[0].message, "content", "Evaluation failed")

    # Save user answers + evaluation
    conn = sqlite3.connect("cache.db")
    cur = conn.cursor()
    cur.execute("""
        UPDATE question_papers
        SET user_answers = ?, evaluation = ?
        WHERE id = ?
    """, (user_answers, evaluation, paper_id))
    conn.commit()
    conn.close()

    return jsonify({
        "paper_id": paper_id,
        "difficulty": difficulty,
        "questions": questions,
        "user_answers": user_answers,
        "evaluation": evaluation
    })




@app.route("/preview", methods=["GET"])
def preview_papers():
    conn = sqlite3.connect("cache.db")
    cur = conn.cursor()
    cur.execute("""
        SELECT id, LOWER(TRIM(subject)), LOWER(TRIM(chapter)), questions, answers
        FROM question_papers
    """)
    rows = cur.fetchall()
    conn.close()

    papers = []
    for row in rows:
        papers.append({
            "id": row[0],
            "subject": row[1],
            "chapter": row[2],
            "questions": row[3],
            "answers": row[4],
        })
    return jsonify(papers)


@app.route("/delete", methods=["DELETE"])
def delete_papers():
    subject = request.args.get("subject")
    chapter = request.args.get("chapter")
    paper_id = request.args.get("id")

    conn = sqlite3.connect("cache.db")
    cur = conn.cursor()

    query = "DELETE FROM question_papers WHERE 1=1"
    params = []

    if paper_id:
        query += " AND id = ?"
        params.append(paper_id)

    if subject:
        query += " AND LOWER(TRIM(subject)) = ?"
        params.append(subject.lower().strip())

    if chapter:
        query += " AND LOWER(TRIM(chapter)) = ?"
        params.append(chapter.lower().strip())

    cur.execute(query, params)
    deleted = cur.rowcount
    conn.commit()
    conn.close()

    return jsonify({"deleted": deleted})




if __name__ == "__main__":
    init_db()
    session_cache = {}   # MARKED CHANGE (global cache dict, was missing)

    # --- MARKED CHANGE: Pandas preview of DB ---
    
    conn = sqlite3.connect("cache.db")
    try:
        df = pd.read_sql_query("SELECT * FROM question_papers LIMIT 5", conn)
        #print("\n Preview of saved question papers:")
        #print(df.to_markdown(index=False))   # clean markdown table in terminal
    except Exception as e:
        print("Could not preview DB:", e)
    finally:
        conn.close()
    # --- END CHANGE ---

    app.run(port=5000, debug=True)




#------------------------------make all the api call refresh--------------------------------------------------------------------------

# @app.route("/generate_question_paper", methods=["GET"])
# def generate_question_paper():
#     subject = request.args.get("subject", "pharmacology").lower()
#     chapter = request.args.get("chapter", "").lower()
#     difficulty = request.args.get("difficulty", "medium").lower()   # ✅ still comes from user if needed
#     refresh = True  # ✅ always regenerate, ignore cache

#     # key excludes count, includes subject+chapter
#     key = f"{subject}_{chapter}"

#     # ❌ skip cache check completely, always go fresh
#     # if not refresh and key in session_cache:
#     #     paper = session_cache[key]
#     #     return jsonify({"subject": subject, "difficulty": difficulty, "paper": paper})
