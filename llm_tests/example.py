print("=== Initializing LLM Libs ===")
from llm_wrapper import LocalModel
from datasets import Dataset

print("=== Creating Dataset ===")
examples = [
    "Sue is weird today. What's wrong with",
    "Alex is weird today. Is she sick? What's whrong with her?",
    "Alex is weird today. Is he sick? What's whrong with him?",
]
dataset = Dataset.from_list([{"text": ex} for ex in examples])

print("=== Before Fine-tuning ===")
model = LocalModel()
model.analyze_next("Sue is weird today. What's wrong with")
model.analyze_next("Alex is weird today. Is he sick? What's wrong with")
model.analyze_next("Alex is weird today. Is she sick? What's wrong with")

print("=== Fine-tuning...")

def tokenize_for_training(example):
    encoding = model.tokenizer(
        example["text"], truncation=True, padding="max_length", max_length=64
    )
    encoding["labels"] = encoding["input_ids"]
    return encoding

tokenized_dataset = dataset.map(tokenize_for_training)
model.fine_tune(tokenized_dataset)

print("=== After Fine-tuning ===")
model.analyze_next("Sue is weird today. What's wrong with")
model.analyze_next("Alex is weird today. Is he sick? What's wrong with")
model.analyze_next("Alex is weird today. Is she sick? What's wrong with")
