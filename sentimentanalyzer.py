import torch
import gradio as gr
from transformers import pipeline
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# model_path = "C:\\Users\\abdul\\Documents\\genaiproj\\genai\\Models\\models--distilbert--distilbert-base-uncased-finetuned-sst-2-english"
analyzer = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")     
# analyzer = pipeline("text-classification", model=model_path)

# print(analyzer(["Nice to meet you!", "very expensive"]))

def sentiment_analysis(review):
    sentiment = analyzer(review)
    return sentiment[0]['label']

def plot_sentiment_distribution(df):
    # Check if required columns are present
    if 'Review' not in df.columns or 'Sentiment' not in df.columns:
        raise ValueError("DataFrame must contain 'Review' and 'Sentiment' columns.")
    
    # Count positive and negative sentiments
    sentiment_counts = df['Sentiment'].value_counts()

    # Create a bar chart
    fig, ax = plt.subplots()
    sentiment_counts.plot(kind='bar', ax=ax, color=['skyblue', 'salmon'])
    
    # Set chart labels and title
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    ax.set_title('Sentiment Distribution')
    
    # Return the figure object
    return fig

def analyze_reviews(file):
    if not file.name.endswith('.xlsx'):
        return "Invalid file type. Please upload an Excel file."
    
    # Read the Excel file
    df = pd.read_excel(file)
    
    if 'Review' not in df.columns:
        return "The Excel file must contain a column named 'Review'."
    
    # Apply get_sentiment function to each review and create new column
    df['Sentiment'] = df['Review'].apply(sentiment_analysis)
    chart_object = plot_sentiment_distribution(df)
    return df, chart_object

# Result = analyze_reviews("C:\\Users\\abdul\\Documents\\genaiproj\\genai\\Files\\app_reviews.xlsx")
# print(Result)

# Example usage
# file_path = 'path_to_your_excel_file.xlsx'  # Update with your actual file path
# result_df = analyze_reviews(file_path)
# print(result_df)


gr.close_all()

# demo = gr.Interface(fn=summary, inputs="text", outputs="text")

demo = gr.Interface(
    fn=analyze_reviews, 
    inputs=[gr.File(label="Input file to analyze")], 
    outputs=[gr.Dataframe(label="Sentiments"), gr.Plot(label="Sentiment Distribution")], 
    title="Sentiment Analyzer", 
    theme="soft",
    description="Analyze the sentiment of any review in seconds!")
    
demo.launch(share=True)




# Example usage
# data = {'Review': ['Great product!', 'Not good', 'Excellent service', 'Bad experience'], 
#         'Sentiment': ['Positive', 'Negative', 'Positive', 'Negative']}
# df = pd.DataFrame(data)
# fig = plot_sentiment_distribution(df)
# fig.show()
