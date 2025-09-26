#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT Quick Start Example - RedNote-Vibe Dataset

A simplified BERT binary classification example for quick start and testing.
"""

import json
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
from datasets import Dataset

def load_data():
    """Load data"""
    print("Loading data...")
    
    def load_jsonl(file_path):
        data = []
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data.append(json.loads(line))
                    except:
                        continue
        return data
    
    # Get data paths
    human_file = os.path.join('datasets', 'training_set_human.jsonl')
    ai_file = os.path.join('datasets', 'training_set_aigc.jsonl')
    
    human_data = load_jsonl(human_file)
    ai_data = load_jsonl(ai_file)
    
    print(f"Human data: {len(human_data)} samples")
    print(f"AI data: {len(ai_data)} samples")
    
    return human_data, ai_data

def prepare_dataset(human_data, ai_data, max_samples=1000):
    """Prepare dataset"""
    print(f"Preparing dataset (max {max_samples} samples per class)...")
    
    texts = []
    labels = []
    
    # Sample human data
    for i, item in enumerate(human_data[:max_samples]):
        content = item.get('note_content', '').strip()
        if content and len(content) > 10:
            texts.append(content[:300])  # Truncate to 300 characters
            labels.append(0)  # Human
    
    # Sample AI data
    for i, item in enumerate(ai_data[:max_samples]):
        content = item.get('note_content', '').strip()
        if content and len(content) > 10:
            texts.append(content[:300])
            labels.append(1)  # AI
    
    print(f"Total samples: {len(texts)}")
    print(f"   - Human: {labels.count(0)}")
    print(f"   - AI: {labels.count(1)}")
    
    return texts, labels

def tokenize_function(examples, tokenizer):
    """Data tokenization"""
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=256
    )

def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {'accuracy': accuracy}

def main():
    """Main function"""
    print("BERT Quick Start Example")
    print("="*40)
    
    # 1. Load data
    human_data, ai_data = load_data()
    
    if not human_data or not ai_data:
        print("Data loading failed!")
        return
    
    # 2. Prepare data
    texts, labels = prepare_dataset(human_data, ai_data, max_samples=500)  # Small sample for quick testing
    
    # 3. Split dataset
    print("\nSplitting dataset...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.4, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # 4. Initialize tokenizer and model
    print("\nInitializing BERT model...")
    model_name = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # 5. Create Dataset objects
    train_dataset = Dataset.from_dict({
        'text': X_train,
        'labels': y_train
    })
    val_dataset = Dataset.from_dict({
        'text': X_val,
        'labels': y_val
    })
    test_dataset = Dataset.from_dict({
        'text': X_test,
        'labels': y_test
    })
    
    # 6. Tokenization
    print("\nTokenizing texts...")
    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    
    # 7. Training configuration
    training_args = TrainingArguments(
        output_dir='./bert_results',
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        eval_strategy="epoch",  
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )
    
    # 8. Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # 9. Start training
    print("\nStarting training...")
    trainer.train()
    
    # 10. Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"Test accuracy: {test_results['eval_accuracy']:.4f}")
    
    # 11. Prediction examples
    print("\nPrediction examples...")
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    
    # Classification report
    print("\nDetailed classification report:")
    print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))
    
    # 12. Save model
    model_save_path = './bert_model_saved'
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"\nModel saved to: {model_save_path}")
    
    # 13. Show some prediction examples
    print("\nPrediction samples:")
    for i in range(min(5, len(X_test))):
        true_label = "Human" if y_test[i] == 0 else "AI"
        pred_label = "Human" if y_pred[i] == 0 else "AI"
        correct = "Correct" if y_test[i] == y_pred[i] else "Wrong"
        
        print(f"\nSample {i+1}: {correct}")
        print(f"Text: {X_test[i][:100]}...")
        print(f"True: {true_label}, Predicted: {pred_label}")
    
    print(f"\nTraining completed! Final accuracy: {test_results['eval_accuracy']:.4f}")

if __name__ == "__main__":
    main()
