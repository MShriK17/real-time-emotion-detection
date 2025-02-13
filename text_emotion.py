import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the required NLTK data
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

def detect_text_emotion(text):
    scores = sia.polarity_scores(text)
    if scores["compound"] >= 0.05:
        return "Positive"
    elif scores["compound"] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Example usage
if __name__ == "__main__":
    text = input("Enter a sentence: ")
    print(f"Detected Emotion: {detect_text_emotion(text)}")
