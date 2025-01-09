import pandas as pd
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

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

# Visualize Sentiment Distribution
def plot_sentiment_distribution(df):
    # Assuming there's a 'sentiment' column (if not, adjust as needed)
    sentiment_counts = df['sentiment'].value_counts()
    plt.figure(figsize=(8,6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis")
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()

# Visualize Word Cloud of Text Data
def plot_word_cloud(df):
    text = ' '.join(df['Sentence'].dropna())  # Combine all text into one string
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10,7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# Visualize Correlation Heatmap (for numerical features)
def plot_correlation_heatmap(df):
    correlation_matrix = df.corr()  # Only works for numeric columns
    plt.figure(figsize=(10,8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()

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

    # Call the visualization functions
    plot_sentiment_distribution(df)
    plot_word_cloud(df)
    plot_correlation_heatmap(df)
