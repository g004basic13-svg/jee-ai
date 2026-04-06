import os
import json
import time
import re
from dotenv import load_dotenv
from google import genai
from supabase import create_client

load_dotenv()

gemini = genai.Client(api_key=os.getenv("GEMINI_KEY"))
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Load data
with open("jee_mains_data.json", "r") as f:
    data = json.load(f)

print(f"Total questions to process: {len(data)}")

# Track progress
PROGRESS_FILE = "embed_progress.json"
try:
    with open(PROGRESS_FILE, "r") as f:
        progress = json.load(f)
        start_from = progress["last_index"] + 1
except FileNotFoundError:
    start_from = 0

print(f"Starting from index: {start_from}")


def extract_year(paper_id):
    match = re.search(r'(\d{4})', paper_id or "")
    return int(match.group(1)) if match else None


def extract_session(paper_id):
    if not paper_id:
        return None
    if "january" in paper_id or "february" in paper_id:
        return "Jan"
    if "march" in paper_id:
        return "Mar"
    if "april" in paper_id:
        return "Apr"
    if "june" in paper_id:
        return "Jun"
    if "july" in paper_id:
        return "Jul"
    if "august" in paper_id or "september" in paper_id:
        return "Sep"
    return None


def extract_shift(paper_id):
    if not paper_id:
        return None
    if "morning" in paper_id:
        return "Morning"
    if "evening" in paper_id:
        return "Evening"
    return None


def save_progress(index):
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"last_index": index}, f)


failed = 0
skipped = 0

# ✅ FIX: Use while loop instead of for loop
#    so that i -= 1 on rate-limit retry actually works.
i = start_from
while i < len(data):
    q = data[i]

    # Clean question text (remove HTML tags)
    question_text = re.sub('<[^<]+?>', '', q.get("question", ""))

    # ✅ FIX: Save progress even when skipping empty questions,
    #    so restarts don't re-process already-handled indices.
    if not question_text.strip():
        skipped += 1
        save_progress(i)
        i += 1
        continue

    try:
        # Embed
        result = gemini.models.embed_content(
            model="gemini-embedding-001",
            contents=question_text
        )
        embedding = result.embeddings[0].values

        # Prepare record
        record = {
            "question": q.get("question", ""),
            "solution": q.get("explanation") or q.get("solution", ""),
            "answer": q.get("correct_option") or q.get("answer", ""),
            "subject": q.get("subject", "unknown"),
            "topic": q.get("chapter", ""),
            "subtopic": q.get("topic", ""),
            "exam": "JEE Main",
            "year": extract_year(q.get("paper_id")),
            "session": extract_session(q.get("paper_id")),
            "shift": extract_shift(q.get("paper_id")),
            "question_type": q.get("question_type", "mcq"),
            "difficulty": None,
            "source": "huggingface",
            "embedding": embedding
        }

        supabase.table("jee_questions").insert(record).execute()

        # ✅ Save progress after every successful insert
        save_progress(i)

        if (i + 1) % 10 == 0:
            print(f"  ✅ {i + 1}/{len(data)} embedded  |  skipped so far: {skipped}  |  failed: {failed}")

        # ✅ Only advance i on success
        i += 1

    except Exception as e:
        error_msg = str(e)

        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            # ✅ FIX: Don't increment i — the while loop will retry the same index
            print(f"  ⏳ Rate limited at {i}. Waiting 60s then retrying...")
            save_progress(i - 1)  # last safely completed index
            time.sleep(60)
            # Do NOT do i -= 1 here. Just don't increment. Loop retries same i.
            continue

        else:
            # Non-rate-limit error: log, save, skip
            failed += 1
            print(f"  ❌ Failed at {i}: {error_msg[:120]}")
            save_progress(i)  # mark as handled so we don't retry on restart
            i += 1

    # Small delay to avoid hitting rate limits
    time.sleep(0.5)

print(f"\n{'='*50}")
print(f"Done!")
print(f"  Processed up to index : {i - 1}")
print(f"  Total in dataset      : {len(data)}")
print(f"  Skipped (empty)       : {skipped}")
print(f"  Failed (errors)       : {failed}")
print(f"{'='*50}")