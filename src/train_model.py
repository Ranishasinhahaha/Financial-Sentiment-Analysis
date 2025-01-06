# src/train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
from src.data_preprocessing import load_data, tokenize_data

# Load dataset and preprocess
def load_and_preprocess_data(file_path):
    df = load_data(file_path)
    tokenized_data = tokenize_data(df)
    return df, tokenized_data

# Define metrics function
def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Train the model
def train():
    df, tokenized_data = load_and_preprocess_data(r"C:\\Users\\KIIT\\Desktop\\Sentiment_Analysis\\data\\data.csv")

    # Split the data into train and test sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Load the model
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_df,
        eval_dataset=val_df,
        compute_metrics=compute_metrics
    )

    # Start training
    trainer.train()

    # Save the model
    model.save_pretrained('./results/model')
    print("Model saved successfully!")

# Evaluate the model
def evaluate():
    # Load the model
    model = RobertaForSequenceClassification.from_pretrained('./results/model')
    trainer = Trainer(model=model)

    # Evaluate on the validation dataset
    results = trainer.evaluate()
    print(f"Evaluation results: {results}")

if __name__ == "__main__":
    train()
    evaluate()
