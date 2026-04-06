import os
from dotenv import load_dotenv
from google import genai
from supabase import create_client
load_dotenv()
gemini = genai.Client(api_key=os.getenv("GEMINI_KEY"))
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
questions = [
    {
        "question": "A particle of mass 2 kg is moving with velocity 3 m/s. Find its kinetic energy.",
        "solution": "Step 1: Use KE = (1/2)mv². Step 2: KE = (1/2)(2)(3²) = (1/2)(2)(9) = 9 J.",
        "answer": "9 J",
        "subject": "physics",
        "topic": "mechanics",
        "subtopic": "kinetic_energy",
        "exam": "JEE Main",
        "year": 2023,
        "question_type": "numerical",
        "difficulty": "simple",
        "source": "manual"
    },
    {
        "question": "Find the derivative of sin²x with respect to x.",
        "solution": "Step 1: Apply chain rule. d/dx[sin²x] = 2sinx · cosx = sin2x.",
        "answer": "sin2x",
        "subject": "math",
        "topic": "calculus",
        "subtopic": "differentiation",
        "exam": "JEE Main",
        "year": 2022,
        "question_type": "MCQ",
        "difficulty": "simple",
        "source": "manual"
    },
    {
        "question": "Calculate the pH of a 0.01 M HCl solution.",
        "solution": "Step 1: HCl is strong acid, fully dissociates. [H+] = 0.01 M. Step 2: pH = -log(0.01) = 2.",
        "answer": "2",
        "subject": "chemistry",
        "topic": "physical_chemistry",
        "subtopic": "ionic_equilibrium",
        "exam": "JEE Main",
        "year": 2024,
        "question_type": "numerical",
        "difficulty": "simple",
        "source": "manual"
    },
    {
        "question": "A block of mass 5 kg is placed on a rough inclined plane of angle 30°. If the coefficient of friction is 0.3, find whether the block slides down.",
        "solution": "Step 1: mgsinθ = 5(10)sin30° = 25 N. Step 2: N = mgcosθ = 43.3 N. Step 3: Friction = μN = 13 N. Step 4: mgsinθ > friction, block slides.",
        "answer": "Block slides down",
        "subject": "physics",
        "topic": "mechanics",
        "subtopic": "friction",
        "exam": "JEE Main",
        "year": 2023,
        "question_type": "MCQ",
        "difficulty": "medium",
        "source": "manual"
    },
    {
        "question": "Evaluate the integral ∫(x²+1)/(x⁴+1) dx.",
        "solution": "Step 1: Divide by x². Step 2: Substitute t = x - 1/x. Step 3: ∫dt/(t²+2) = (1/√2)arctan(t/√2) + C.",
        "answer": "(1/√2)arctan((x²-1)/(x√2)) + C",
        "subject": "math",
        "topic": "calculus",
        "subtopic": "integration",
        "exam": "JEE Advanced",
        "year": 2022,
        "question_type": "MCQ",
        "difficulty": "hard",
        "source": "manual"
    }
]
for q in questions:
    print(f"Embedding: {q['question'][:50]}...")
    result = gemini.models.embed_content(
        model="gemini-embedding-001",
        contents=q["question"]
    )
    embedding = result.embeddings[0].values
    data = {**q, "embedding": embedding}
    supabase.table("jee_questions").insert(data).execute()
    print(f"  ✅ Stored! ({len(embedding)} dimensions)")
print(f"\nDone! {len(questions)} questions embedded and stored.")