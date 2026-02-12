#!/usr/bin/env python3
"""
Script to split eti_documents.jsonl into train and test sets.

The test set will contain 50 randomly selected documents from eti_documents.jsonl.
The train set will contain the remaining documents.

Usage:
    python scripts/split_eti_documents.py
"""

import json
import random
from pathlib import Path


def split_by_random_sample(test_size=50):
    """
    Split eti_documents.jsonl into train and test sets by randomly selecting documents.
    
    Args:
        test_size: Number of documents to include in test set (default 50)
    """
    # Define file paths
    input_file = Path("data/raw/eti_documents.jsonl")
    test_file = Path("data/raw/eti_test.jsonl")
    train_file = Path("data/raw/eti_train.jsonl")
    
    # Validate input file exists
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"Reading documents from {input_file}...")
    
    # Read all documents
    documents = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    doc = json.loads(line)
                    documents.append(doc)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}")
    
    total_docs = len(documents)
    print(f"Total documents loaded: {total_docs}")
    
    if total_docs < test_size:
        raise ValueError(f"Not enough documents: {total_docs} (need at least {test_size})")
    
    # Randomly select documents for test set
    random.seed(42)  # For reproducibility
    test_indices = set(random.sample(range(total_docs), test_size))
    
    test_docs = []
    train_docs = []
    
    for idx, doc in enumerate(documents):
        if idx in test_indices:
            test_docs.append(doc)
        else:
            train_docs.append(doc)
    
    # Write test set
    print(f"Writing {len(test_docs)} test documents to {test_file}...")
    with open(test_file, 'w', encoding='utf-8') as f:
        for doc in test_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    # Write train set
    print(f"Writing {len(train_docs)} train documents to {train_file}...")
    with open(train_file, 'w', encoding='utf-8') as f:
        for doc in train_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    print("\nRandom split complete!")
    print(f"Test set: {test_file} ({len(test_docs)} documents)")
    print(f"Train set: {train_file} ({len(train_docs)} documents)")

def split_by_dataset_test():
    """
    Split eti_documents.jsonl based on documents referenced in dataset_test.jsonl.
    
    Documents with 'name' matching 'dokument_id' in dataset_test.jsonl will be written to eti_test.jsonl.
    Remaining documents will be written to eti_train.jsonl.
    """
    # Define file paths
    input_file = Path("data/raw/eti_documents.jsonl")
    dataset_file = Path("data/raw/dataset_test.jsonl")  # ← Endret fra processed/train_dataset.json
    test_file = Path("data/raw/eti_test.jsonl")
    train_file = Path("data/raw/eti_train.jsonl")
    
    # Validate input files exist
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
    
    # Read dataset_test.jsonl to get dokument_id values
    print(f"Reading document IDs from {dataset_file}...")
    test_doc_ids = set()
    
    with open(dataset_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    doc = json.loads(line)
                    doc_id = doc.get('dokument_id')
                    if doc_id:
                        test_doc_ids.add(doc_id)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num} in dataset: {e}")
    
    print(f"Found {len(test_doc_ids)} unique document IDs in dataset_test.jsonl")
    
    # Read all documents from eti_documents.jsonl
    print(f"Reading documents from {input_file}...")
    test_docs = []
    train_docs = []
    found_ids = set()
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    doc = json.loads(line)
                    doc_name = doc.get('name')
                    
                    if doc_name in test_doc_ids:
                        test_docs.append(doc)
                        found_ids.add(doc_name)
                    else:
                        train_docs.append(doc)
                        
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num} in eti_documents: {e}")
    
    # Calculate statistics
    total_docs = len(test_docs) + len(train_docs)
    missing_ids = test_doc_ids - found_ids
    
    print(f"\nTotal documents loaded: {total_docs}")
    print(f"Documents matched: {len(found_ids)}/{len(test_doc_ids)}")
    
    if missing_ids:
        print(f"Warning: {len(missing_ids)} document IDs from dataset not found in eti_documents.jsonl")
        if len(missing_ids) <= 10:
            print(f"Missing IDs: {sorted(list(missing_ids))}")
    
    # Write test set
    print(f"\nWriting {len(test_docs)} test documents to {test_file}...")
    with open(test_file, 'w', encoding='utf-8') as f:
        for doc in test_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    # Write train set
    print(f"Writing {len(train_docs)} train documents to {train_file}...")
    with open(train_file, 'w', encoding='utf-8') as f:
        for doc in train_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    print("\n" + "=" * 60)
    print("Dataset-based split complete!")
    print("=" * 60)
    print(f"Test set: {test_file} ({len(test_docs)} documents)")
    print(f"Train set: {train_file} ({len(train_docs)} documents)")
    print(f"Documents matched: {len(found_ids)}/{len(test_doc_ids)}")
    print(f"Match rate: {len(found_ids)/len(test_doc_ids)*100:.1f}%")


def main():
    """
    Main function to choose between split methods.
    
    To split by random sampling (50 documents):
        - Call split_by_random_sample()
    
    To split by dataset_test.json corpus documents:
        - Call split_by_dataset_test()
    """
    print("Choose split method:")
    print("1. Random sampling (50 documents)")
    print("2. Based on dataset_test.json corpus")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        split_by_random_sample()
    elif choice == "2":
        split_by_dataset_test()
    else:
        print("Invalid choice. Please enter 1 or 2.")


if __name__ == "__main__":
    main()
