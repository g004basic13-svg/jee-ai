import os
from dotenv import load_dotenv
from google import genai
from supabase import create_client
from openai import OpenAI

load_dotenv()

gemini = genai.Client(api_key=os.getenv("GEMINI_KEY"))
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
openrouter = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_KEY")
)

MODELS = [
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "google/gemma-3-4b-it:free",
    "minimax/minimax-m2.5:free",
]

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

def build_context(similar_questions):
    if not similar_questions:
        return "No similar PYQs found."
    context = ""
    for i, q in enumerate(similar_questions, 1):
        context += f"\n--- Similar JEE PYQ {i} (similarity: {q['similarity']:.2f}) ---\n"
        context += f"Exam: {q.get('exam', '?')} {q.get('year', '?')}\n"
        context += f"Topic: {q.get('subject', '?')}/{q.get('topic', '?')}\n"
        context += f"Question: {q.get('question', '')}\n"
        if q.get('solution'):
            context += f"Solution: {q['solution'][:500]}\n"
        if q.get('answer'):
            context += f"Answer: {q['answer']}\n"
    return context

def solve(question):
    print("  🔍 Searching similar PYQs...")
    similar = find_similar(question)
    context = build_context(similar)

    system_prompt = f"""You are an expert JEE tutor. You solve problems using the EXACT methodology expected by IIT JEE examiners.

SIMILAR JEE PAST YEAR QUESTIONS (use these as reference for methodology):
{context}

RULES:
1. Solve step-by-step. Number each step.
2. For EACH step, state the concept being applied in [brackets]
3. Write all math in PLAIN TEXT. Examples:
   - Write: (1/2)mv^2   NOT: \\frac{{1}}{{2}}mv^2
   - Write: sqrt(2)     NOT: \\sqrt{{2}}
   - Write: omega       NOT: \\omega
   - Write: x^2 + y^2   NOT: x^2 + y^2 in LaTeX
   - Write: integral of x dx = x^2/2   NOT: \\int x dx
   - Write: delta(KE)   NOT: \\Delta KE
4. Show all mathematical work clearly
5. Use the SAME approach as the PYQ solutions above
6. After the solution, state: Subject, Topic, Difficulty (Simple/Medium/Hard)
7. If the question is similar to a PYQ, mention it: "This is similar to JEE [year] — same concept"
8. Be concise but thorough. No fluff.
DO NOT use any LaTeX formatting. Write everything as plain readable text."""

    print("  🧠 Solving...")
    for model in MODELS:
        try:
            response = openrouter.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": system_prompt},
                    {"role": "assistant", "content": "Ready. Give me the JEE question."},
                    {"role": "user", "content": question}
                ]
            )
            reply = response.choices[0].message.content
            if reply:
                return reply, model, similar
        except Exception:
            continue
    return "All models are down. Try later.", "none", similar

print("=" * 50)
print("  🎯 JEE DOUBT SOLVER")
print("  Powered by RAG + AI")
print("=" * 50)
print("\nType a JEE question. Type 'quit' to exit.\n")

if __name__ == "__main__":
    print("=" * 50)
    print("  🎯 JEE DOUBT SOLVER")
    print("=" * 50)

    while True:
        question = input("📝 Your question: ")
        if not question.strip():
            continue
        if question.lower() == "quit":
            break
        solution, model_used, similar = solve(question)
        print(f"\n{solution}\n")

    print(f"\n{'=' * 50}")
    print(f"🤖 Solution (via {model_used}):\n")
    print(solution)
    print(f"\n{'=' * 50}")

    if similar:
        print(f"\n📚 Referenced {len(similar)} similar PYQs:")
        for s in similar:
            print(f"   • {s.get('exam', '?')} {s.get('year', '?')} — {s.get('subject', '?')}/{s.get('topic', '?')} (similarity: {s['similarity']:.2f})")
    print()

print("Done!")