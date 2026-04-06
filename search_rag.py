import os
from dotenv import load_dotenv
from google import genai
from supabase import create_client

load_dotenv()

gemini = genai.Client(api_key=os.getenv("GEMINI_KEY"))
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

def find_similar(question, top_k=3):
    result = gemini.models.embed_content(
        model="gemini-embedding-001",
        contents=question
    )
    query_embedding = result.embeddings[0].values

    response = supabase.rpc("match_questions", {
        "query_embedding": query_embedding,
        "match_count": top_k
    }).execute()

    return response.data

print("🔍 JEE RAG Search (type 'quit' to exit)\n")

while True:
    question = input("Ask a JEE question: ")
    if question.lower() == "quit":
        break

    results = find_similar(question)

    if not results:
        print("\nNo similar questions found.\n")
        continue

    print(f"\n📚 Found {len(results)} similar JEE questions:\n")
    for i, r in enumerate(results, 1):
        print(f"--- Match {i} (similarity: {r['similarity']:.3f}) ---")
        print(f"Q: {r['question']}")
        if r.get('solution'):
            print(f"A: {r['solution']}")
        if r.get('answer'):
            print(f"Answer: {r['answer']}")
        print(f"📌 {r.get('subject','?')}/{r.get('topic','?')} | {r.get('exam','?')} {r.get('year','?')} | {r.get('difficulty','?')}")
        print()

print("Done!")