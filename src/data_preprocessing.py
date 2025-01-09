import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
import string

# Load the dataset
def load_data(file_path):
    """Load CSV data from the specified file path."""
    return pd.read_csv(file_path)

# Tokenization function
def tokenize_function(text):
    """Tokenize the given text using a pretrained tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(r"C:\\Users\\KIIT\\Desktop\\Sentiment_Analysis\\roberta-base")
    return tokenizer(text, padding="max_length", truncation=True)

# Apply tokenization to the 'Sentence' column
def tokenize_data(df):
    """Tokenize the 'Sentence' column in the DataFrame."""
    tokenized_data = df['Sentence'].apply(tokenize_function)
    return tokenized_data

# Visualize Sentiment Distribution
def plot_sentiment_distribution(df):
    """Plot a bar chart showing the distribution of sentiment labels."""
    sentiment_counts = df['Sentiment'].value_counts()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis", hue=sentiment_counts.index, dodge=False, legend=False)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()

# Visualize Word Cloud of Text Data
def plot_word_cloud(df):
    """Generate and plot a word cloud of the text data."""
    text = ' '.join(df['Sentence'].dropna())  # Combine all text into one string
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Sentences')
    plt.show()

# Encode sentiment labels
def encode_sentiment(df):
    """Encode the 'Sentiment' column as numeric labels."""
    label_encoder = LabelEncoder()
    df['Sentiment_Encoded'] = label_encoder.fit_transform(df['Sentiment'])
    return df

# Add new text-based features
def add_text_features(df):
    """Add additional text-based features to the DataFrame."""
    positive_words = ["good", "great", "excellent", "positive", "happy", "success"]
    negative_words = ["bad", "poor", "negative", "sad", "failure", "angry"]

    def count_punctuation(text):
        return sum(1 for char in text if char in string.punctuation)

    def count_words(text):
        return len(text.split())

    def avg_word_length(text):
        words = text.split()
        return np.mean([len(word) for word in words]) if words else 0

    def count_positive_words(text):
        return sum(1 for word in text.split() if word.lower() in positive_words)

    def count_negative_words(text):
        return sum(1 for word in text.split() if word.lower() in negative_words)

    # Add new features
    df['Sentence_Length'] = df['Sentence'].str.len()
    df['Word_Count'] = df['Sentence'].apply(count_words)
    df['Avg_Word_Length'] = df['Sentence'].apply(avg_word_length)
    df['Punctuation_Count'] = df['Sentence'].apply(count_punctuation)
    df['Positive_Word_Count'] = df['Sentence'].apply(count_positive_words)
    df['Negative_Word_Count'] = df['Sentence'].apply(count_negative_words)

    return df

# Plot a correlation heatmap
def plot_correlation_heatmap(df):
    """Plot a heatmap of correlations between numeric columns."""
    df = encode_sentiment(df)  # Ensure 'Sentiment' is encoded
    df = add_text_features(df)  # Add additional text-based features

    # Select numeric columns for correlation analysis
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] > 1:  # Check if there are multiple numeric columns
        correlation_matrix = numeric_df.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title('Correlation Heatmap')
        plt.show()
    else:
        print("Not enough numeric columns for a meaningful heatmap.")

# Main script execution
if __name__ == "__main__":
    # File path to the dataset
    file_path = r"C:\\Users\\KIIT\\Desktop\\Sentiment_Analysis\\data\\data.csv"

    # Load data from the dataset
    df = load_data(file_path)

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

    # Call the visualization functions
    plot_sentiment_distribution(df)
    plot_word_cloud(df)
    plot_correlation_heatmap(df)
