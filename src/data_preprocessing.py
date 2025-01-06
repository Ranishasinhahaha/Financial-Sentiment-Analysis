import pandas as pd
from transformers import AutoTokenizer

# Load the dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Tokenization function
def tokenize_function(text):
    tokenizer = AutoTokenizer.from_pretrained(r"C:\\Users\\KIIT\\Desktop\\Sentiment_Analysis\\roberta-base")
    return tokenizer(text, padding="max_length", truncation=True)


# Apply tokenization to the 'Sentence' column
def tokenize_data(df):
    # Use the 'Sentence' column for tokenization
    tokenized_data = df['Sentence'].apply(tokenize_function)
    return tokenized_data

# Main script execution
if __name__ == "__main__":
    # Load data from the dataset
    df = load_data(r"C:\\Users\\KIIT\\Desktop\\Sentiment_Analysis\\data\\data.csv")

    # Debugging: Print column names and first few rows
    print("Columns in dataset:", df.columns)
    print("First few rows of the dataset:\n", df.head())

    # Tokenize the data
    print("\nStarting tokenization process...")
    tokenized_data = tokenize_data(df)

    # Debugging: Print a sample of tokenized data
    print("Tokenized data sample:\n", tokenized_data.head())

    # Optional: Save tokenized data to a file for further use
    tokenized_data.to_csv(r"C:\\Users\\KIIT\\Desktop\\Sentiment_Analysis\\data\\tokenized_data.csv", index=False)
    print("\nTokenized data saved to 'data/tokenized_data.csv'.")
