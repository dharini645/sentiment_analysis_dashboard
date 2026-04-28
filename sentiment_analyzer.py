#used for text cleaning, removes junk like symbols, numbers
import re
#converts words into numbers [0.2, 0.5, 0.3]
from sklearn.feature_extraction.text import TfidfVectorizer
#probability-based classifier DECIDES THINGS
from sklearn.naive_bayes import MultinomialNB
#connets things together text -> pipeline -> result
from sklearn.pipeline import Pipeline
#numeric calculations
import numpy as np


class SentimentAnalyzer:
    #using scikit-learn

    def __init__(self):

        # ─────────────────────────────────────────────────────────────────
        # TRAINING DATA
        # ─────────────────────────────────────────────────────────────────

        self.training_texts = [
            # POSITIVE EXAMPLES - happy, enthusiastic, satisfied words
            "I absolutely love this! It's fantastic!",
            "This is amazing and wonderful!",
            "Best experience ever! Highly recommend!",
            "Brilliant! Perfect quality and service!",
            "I'm so happy with this purchase!",
            "Excellent work! Very impressed!",
            "This made my day! Thank you!",
            "Wonderful and delightful!",
            "Outstanding! Exceeded expectations!",
            "Love it! Best decision ever!",
            "Simply the best! Incredibly satisfied!",
            "Perfect! Everything is great!",
            "Amazing quality! Highly satisfied!",
            "Fantastic! Will definitely buy again!",
            "Superb! You've got my business!",
            "Wonderful! I'm delighted!",
            "Excellent! Top notch!",
            "Brilliant! Absolutely perfect!",
            "Great! Very happy!",
            "Awesome! Loved every minute!",

            # NEGATIVE EXAMPLES - sad, frustrated, disappointed words
            "This is terrible and awful!",
            "I hate this! Complete waste!",
            "Worst purchase ever!",
            "Horrible quality and service!",
            "I'm very disappointed!",
            "This is garbage!",
            "Absolutely disgusting!",
            "Terrible experience! Never again!",
            "Useless and broken!",
            "Worst money I've spent!",
            "Frustrating and disappointing!",
            "Bad quality! Not worth it!",
            "Terrible! Very upset!",
            "Awful experience!",
            "Don't waste your money!",
            "Horrible! Absolutely terrible!",
            "Bad! Very dissatisfied!",
            "Disgusting! Never coming back!",
            "Worst ever! Hate it!",
            "Terrible! Complete disappointment!",

            # NEUTRAL EXAMPLES - objective, no strong emotion
            "It is a product.",
            "This is okay.",
            "It does what it should.",
            "Average quality.",
            "Neither good nor bad.",
            "It works as described.",
            "Acceptable but nothing special.",
            "Decent product.",
            "Standard quality.",
            "Fair price.",
            "It's fine.",
            "Nothing remarkable.",
            "As expected.",
            "Adequate service.",
            "Ordinary experience.",
            "Not bad, not great.",
            "Meets expectations.",
            "Regular quality.",
            "Normal price.",
            "Typical product.",
        ]

        # LABELS - what sentiment each training example represents
        # Must be in the same order as training_texts
        self.training_labels = [
            'positive', 'positive', 'positive', 'positive', 'positive',
            'positive', 'positive', 'positive', 'positive', 'positive',
            'positive', 'positive', 'positive', 'positive', 'positive',
            'positive', 'positive', 'positive', 'positive', 'positive',

            'negative', 'negative', 'negative', 'negative', 'negative',
            'negative', 'negative', 'negative', 'negative', 'negative',
            'negative', 'negative', 'negative', 'negative', 'negative',
            'negative', 'negative', 'negative', 'negative', 'negative',

            'neutral', 'neutral', 'neutral', 'neutral', 'neutral',
            'neutral', 'neutral', 'neutral', 'neutral', 'neutral',
            'neutral', 'neutral', 'neutral', 'neutral', 'neutral',
            'neutral', 'neutral', 'neutral', 'neutral', 'neutral',
        ]

        # ─────────────────────────────────────────────────────────────────
        # CREATE MACHINE LEARNING PIPELINE
        # ─────────────────────────────────────────────────────────────────
        # A pipeline combines preprocessing and model training

        self.model = Pipeline([
            # Step 1: TF-IDF Vectorizer
            # Converts text to numerical features that the model can understand
            ('tfidf', TfidfVectorizer(
                lowercase=True,  # Convert to lowercase for consistency
                stop_words='english',  # Remove common words like "the", "and"
                max_features=1000,  # Use top 1000 most important words
                ngram_range=(1, 2),  # Consider individual words and pairs
                min_df=1,  # Include words that appear at least once
                max_df=0.9  # Exclude words that appear in >90% of texts
            )),

            # Step 2: Multinomial Naive Bayes Classifier
            # Learns to classify text based on word probabilities
            ('classifier', MultinomialNB())
        ])

        # ─────────────────────────────────────────────────────────────────
        # TRAIN THE MODEL
        # ─────────────────────────────────────────────────────────────────
        # The model learns patterns from the training data

        print("🤖 Training sentiment analysis model...")
        self.model.fit(self.training_texts, self.training_labels)
        print("✅ Model training complete!")

    def preprocess_text(self, text):
        # Remove special characters and extra whitespace
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
        text = ' '.join(text.split())  # Remove extra spaces
        text = text.lower()  # Convert to lowercase
        return text

    def analyze_sentiment(self, text):
        # Clean the input text
        cleaned_text = self.preprocess_text(text)

        # Predict the sentiment class
        # predict() returns an array with one element
        prediction = self.model.predict([cleaned_text])[0]

        # Get prediction probabilities for all classes
        # This tells us the model's confidence in each possible sentiment
        probabilities = self.model.predict_proba([cleaned_text])[0]

        # Get all possible sentiment classes the model knows
        classes = self.model.classes_

        # Find the probability of the predicted sentiment
        # This is our confidence score
        predicted_index = list(classes).index(prediction)
        confidence = float(probabilities[predicted_index])

        # Return results as a dictionary
        return {
            'text': text,  # Original text
            'sentiment': prediction,  # Predicted sentiment
            'confidence': confidence  # Confidence score
        }

    def analyze_batch(self, texts):
        results = []

        # Analyze each text
        for text in texts:
            result = self.analyze_sentiment(text)
            results.append(result)

        return results

    def get_model_info(self):
        # Access the vectorizer from the pipeline
        vectorizer = self.model.named_steps['tfidf']

        return {
            'classes': list(self.model.classes_),  # Sentiments it can predict
            'n_features': vectorizer.n_features_in_,  # Number of features used
            'training_samples': len(self.training_texts),  # Samples used to train
        }


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE (uncomment to test)
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Create analyzer
    analyzer = SentimentAnalyzer()

    # Test with example texts
    test_texts = [
        "I love this product! It's amazing!",
        "This is terrible and I hate it.",
        "It's okay, nothing special.",
    ]

    # Analyze
    results = analyzer.analyze_batch(test_texts)

    # Display results
    for result in results:
        print(f"\nText: {result['text']}")
        print(f"Sentiment: {result['sentiment'].upper()}")
        print(f"Confidence: {result['confidence']:.2%}")