import os
import json
import time
import fitz
from dotenv import load_dotenv
from google import genai
from supabase import create_client
from groq import Groq

load_dotenv()

gemini = genai.Client(api_key=os.getenv("GEMINI_KEY"))
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_KEY"))

PDF_FOLDER = "jee_pdfs"
PROGRESS_FILE = "pdf_progress.json"

try:
    with open(PROGRESS_FILE, "r") as f:
        progress = json.load(f)
except FileNotFoundError:
    progress = {"completed_files": [], "failed_files": []}

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()

    if len(full_text.strip()) < 100:
        print(f"  🔍 Scanned PDF — running OCR...")
        try:
            from pdf2image import convert_from_path
            import pytesseract
            images = convert_from_path(pdf_path, dpi=300)
            full_text = ""
            for i, image in enumerate(images):
                print(f"    OCR page {i+1}/{len(images)}...")
                full_text += pytesseract.image_to_string(image)
        except Exception as e:
            print(f"  ❌ OCR failed: {str(e)[:80]}")
            return ""

    return full_text

def parse_questions_with_ai(text_chunk, filename):
    prompt = f"""You are parsing a JEE (IIT-JEE) exam PDF. Extract ALL questions from this text.

SOURCE FILE: {filename}

RAW TEXT FROM PDF:
{text_chunk}

Extract every question and return a JSON array. For each question use this exact format:
[
  {{
    "question": "full question text here",
    "options": ["option A text", "option B text", "option C text", "option D text"],
    "answer": "correct answer or option letter",
    "solution": "solution/explanation if present, else null",
    "subject": "physics or chemistry or math",
    "topic": "topic name if you can identify it",
    "exam": "JEE Main or JEE Advanced",
    "year": 2023,
    "difficulty": "simple or medium or hard"
  }}
]

RULES:
- If options are not present set options to []
- If year is not clear guess from filename: {filename}
- If subject is not clear guess from question content
- If solution not in text set to null
- Return ONLY the JSON array, no other text
- If no questions found return []"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        questions = json.loads(raw)
        return questions
    except Exception as e:
        print(f"    ⚠️ AI parsing failed: {str(e)[:100]}")
        return []

def embed_and_store(questions, filename):
    stored = 0
    for q in questions:
        if not q.get("question") or len(q["question"].strip()) < 10:
            continue
        try:
            result = gemini.models.embed_content(
                model="gemini-embedding-001",
                contents=q["question"]
            )
            embedding = result.embeddings[0].values

            record = {
                "question": q.get("question", ""),
                "solution": q.get("solution"),
                "answer": q.get("answer", ""),
                "subject": q.get("subject", "unknown"),
                "topic": q.get("topic", ""),
                "subtopic": None,
                "exam": q.get("exam", "JEE Main"),
                "year": q.get("year"),
                "session": None,
                "shift": None,
                "question_type": "MCQ" if q.get("options") else "numerical",
                "difficulty": q.get("difficulty"),
                "source": filename,
                "embedding": embedding
            }

            supabase.table("jee_questions").insert(record).execute()
            stored += 1
            time.sleep(0.3)

        except Exception as e:
            print(f"    ❌ Failed to store: {str(e)[:80]}")
            continue

    return stored

def chunk_text(text, chunk_size=8000):
    chunks = []
    words = text.split()
    current_chunk = []
    current_size = 0

    for word in words:
        current_chunk.append(word)
        current_size += len(word)
        if current_size >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_size = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Main processing
pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
print(f"Found {len(pdf_files)} PDF files")
print(f"Already completed: {len(progress['completed_files'])} files\n")

total_stored = 0

for pdf_file in pdf_files:
    if pdf_file in progress["completed_files"]:
        print(f"⏭️  Skipping {pdf_file} (already done)")
        continue

    pdf_path = os.path.join(PDF_FOLDER, pdf_file)
    print(f"\n📄 Processing: {pdf_file}")

    try:
        text = extract_text_from_pdf(pdf_path)
        print(f"  Extracted {len(text)} characters")

        if len(text.strip()) < 100:
            print(f"  ❌ Could not extract text. Skipping.")
            progress["failed_files"].append(pdf_file)
            with open(PROGRESS_FILE, "w") as f:
                json.dump(progress, f)
            continue

        chunks = chunk_text(text)
        print(f"  Split into {len(chunks)} chunks")

        file_total = 0
        for i, chunk in enumerate(chunks):
            print(f"  🤖 Parsing chunk {i+1}/{len(chunks)}...")
            questions = parse_questions_with_ai(chunk, pdf_file)
            print(f"     Found {len(questions)} questions")

            if questions:
                stored = embed_and_store(questions, pdf_file)
                file_total += stored
                print(f"     ✅ Stored {stored} questions")

            time.sleep(5)  # pause between chunks

        print(f"  📊 Total from {pdf_file}: {file_total} questions stored")
        total_stored += file_total

        progress["completed_files"].append(pdf_file)
        with open(PROGRESS_FILE, "w") as f:
            json.dump(progress, f)

    except Exception as e:
        print(f"  ❌ Failed: {str(e)[:100]}")
        progress["failed_files"].append(pdf_file)
        with open(PROGRESS_FILE, "w") as f:
            json.dump(progress, f)
        continue

print(f"\n{'='*50}")
print(f"✅ Done! Total questions stored: {total_stored}")
print(f"✅ Completed: {len(progress['completed_files'])} files")
print(f"❌ Failed: {len(progress['failed_files'])} files")
if progress["failed_files"]:
    print("Failed files:", progress["failed_files"])