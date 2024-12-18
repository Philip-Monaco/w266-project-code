import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import torch
import os
from tqdm import tqdm

from joblib import dump, load

device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["WANDB_DISABLED"] = "true"
print(os.cpu_count())

# Load and preprocess data
df = pd.read_csv("./Annotations.csv")
terms_df = pd.read_csv("./terms_dataset.csv")


# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
print("Tokenizer loaded successfully!")

# Combine main terms and synonyms into a list
symptom_terms = []
for _, row in terms_df.iterrows():
    symptom_terms.append(row['Name'])
    if isinstance(row['Synonyms'], str) and row['Synonyms'].lower() != 'no synonyms':
        symptom_terms.extend([syn.strip() for syn in row['Synonyms'].split(',')])

symptom_terms = list(set(symptom_terms))  # Remove duplicates
print(f"Total unique symptom terms: {len(symptom_terms)}")

# Pre-tokenize the expanded list of terms
print("Tokenizing symptom terms...")
tokenized_terms = tokenizer(
    symptom_terms,
    add_special_tokens=False,
    truncation=True,
    return_token_type_ids=False
)

tokenized_term_dict = {
    term: tokens for term, tokens in zip(symptom_terms, tokenized_terms["input_ids"])
}
print(f"Tokenized {len(symptom_terms)} terms (including synonyms) successfully!")
print(f"Tokenized {len(symptom_terms)} terms (including synonyms) successfully!")

# Function to align labels
def tokenize_and_align_labels(examples):
    # Tokenize the passages
    tokenized_inputs = tokenizer(
        examples['Passage'],
        truncation=True,
        padding="max_length",
        max_length=256,
        return_offsets_mapping=True
    )

    labels = []
    
    # Add tqdm for progress tracking in a standard Python script
    for offsets, input_ids in tqdm(
        zip(tokenized_inputs['offset_mapping'], tokenized_inputs['input_ids']),
        total=len(tokenized_inputs['input_ids']),
        desc="Aligning Labels",
        unit="example"
    ):
        current_labels = [0] * len(input_ids)

        # Match input_ids with pre-tokenized terms
        for term_tokens in tokenized_term_dict.values():
            term_len = len(term_tokens)
            for idx in range(len(input_ids) - term_len + 1):
                if input_ids[idx:idx + term_len] == term_tokens:
                    current_labels[idx] = 1  # B-Symptom
                    for j in range(1, term_len):
                        current_labels[idx + j] = 2  # I-Symptom

        labels.append(current_labels)

    # Attach labels back to tokenized inputs
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

# Map function over dataset
print("Processing dataset...")

# Convert dataframe to HuggingFace Dataset
dataset = Dataset.from_pandas(df)

# Tokenize and align labels
dataset = dataset.map(
    tokenize_and_align_labels,
    batched=True,
    batch_size=128,  # Adjust based on your hardware
    remove_columns=dataset.column_names,
    num_proc=os.cpu_count()  # Adjust based on your hardware
)
# Split train/test

def save_dataset_to_joblib(dataset, output_path):
    dump(dataset, output_path)
    print(f"Dataset saved to {output_path}")


# Save the dataset
save_dataset_to_joblib(dataset, "./preprocessed_dataset.joblib")
