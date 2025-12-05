import joblib
import os

class EmailClassifier:
    def __init__(self, model_dir='.'):
        self.model_dir = model_dir
        self.clf = None
        self.vectorizer = None
        self.mlb = None
        self.load_model()

    def load_model(self):
        """Loads the model artifacts from disk."""
        try:
            self.clf = joblib.load(os.path.join(self.model_dir, 'multilabel_model.pkl'))
            self.vectorizer = joblib.load(os.path.join(self.model_dir, 'multilabel_vectorizer.pkl'))
            self.mlb = joblib.load(os.path.join(self.model_dir, 'multilabel_binarizer.pkl'))
            print("Model artifacts loaded successfully.")
        except FileNotFoundError as e:
            print(f"Error loading model: {e}")
            raise

    def predict(self, text: str) -> list:
        """
        Classifies the input text and returns a list of categories.
        
        Args:
            text (str): The email body or text to classify.
            
        Returns:
            list: A list of strings representing the categories (e.g., ['HR', 'Project Leader']).
        """
        if not text:
            return []
        
        # Vectorize input
        input_tfidf = self.vectorizer.transform([text])
        
        # Predict
        pred_matrix = self.clf.predict(input_tfidf)
        
        # Convert binary format back to labels
        pred_labels = self.mlb.inverse_transform(pred_matrix)
        
        # Return as a simple list
        return list(pred_labels[0])

# Usage example for testing
if __name__ == "__main__":
    classifier = EmailClassifier()
    result = classifier.predict("please update ravi's salary, revision and attach his medical documents")
    print(f"Predicted Categories: {result}")
