import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

def train_from_csv():
    try:
        print("Loading dataset...")
        # Read CSV with explicit string type for sentiment column
        df = pd.read_csv('sentiment_analysis.csv', dtype={'sentiment': str})
        
        # Clean the sentiment column by stripping whitespace
        df['sentiment'] = df['sentiment'].str.strip().str.lower()
        
        # Map sentiment to our format (1: positive, 0: neutral, -1: negative)
        sentiment_map = {
            'positive': 1,
            'neutral': 0,
            'negative': -1
        }
        
        # Get text and sentiment from the dataset
        texts = df['text'].tolist()
        
        # Convert sentiments with error checking
        sentiments = []
        for sent in df['sentiment']:
            if sent not in sentiment_map:
                print(f"Warning: Unknown sentiment value found: '{sent}'")
                continue
            sentiments.append(sentiment_map[sent])
            
        print(f"Loaded {len(texts)} training examples")
        
        if len(texts) != len(sentiments):
            raise ValueError("Number of texts and sentiments don't match")
            
        if not texts or not sentiments:
            raise ValueError("No valid training data found")
            
        # Train the model
        vectorizer = CountVectorizer()
        classifier = MultinomialNB()
        
        X = vectorizer.fit_transform(texts)
        classifier.fit(X, sentiments)
        
        # Save the trained model and vectorizer
        print("Saving model and vectorizer...")
        with open('model.pkl', 'wb') as f:
            pickle.dump(classifier, f)
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
            
        print("Training complete! Model saved.")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    train_from_csv() 