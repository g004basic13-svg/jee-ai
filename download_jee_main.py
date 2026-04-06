from datasets import load_dataset
import json

print("Downloading JEE Mains dataset...")
ds = load_dataset("ruh-ai/grafite-jee-mains-qna-no-img")

questions = []
for item in ds["train"]:
    questions.append(dict(item))

with open("jee_mains_data.json", "w") as f:
    json.dump(questions, f, indent=2, default=str)

print(f"Downloaded {len(questions)} JEE Mains questions!")

print("\nDownloading JEE Advanced dataset...")
ds2 = load_dataset("daman1209arora/jeebench")

advanced = []
for item in ds2["train"]:
    advanced.append({
        "question": item["description"],
        "answer": str(item["gold"]),
        "subject": item["subject"],
        "exam": "JEE Advanced",
        "question_type": item["type"]
    })

with open("jee_advanced_data.json", "w") as f:
    json.dump(advanced, f, indent=2)

print(f"Downloaded {len(advanced)} JEE Advanced questions!")
print(f"\nTotal: {len(questions) + len(advanced)} questions ready!")

