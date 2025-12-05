"""Fix tokenizer by downloading fresh files from base model."""
from transformers import AutoTokenizer

print("Downloading tokenizer from sentence-transformers/all-MiniLM-L6-v2...")
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
print("Saving to models/document_classifier...")
tokenizer.save_pretrained(r'D:\Agents\fast-api-agent\models\document_classifier')
print("Done!")
