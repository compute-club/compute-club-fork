from typing import List
from profanity_check import predict_prob


class ProfanityPredictor:
    """
    Returns a profanity probability for a list of texts.
    """

    def predict(self, texts: List[str]) -> List[float]:
        """
        Returns profanity probability for each text
        """
        profanity_scores = predict_prob(texts)
        return profanity_scores
