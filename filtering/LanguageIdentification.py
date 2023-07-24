import os
import fasttext
from typing import Tuple


class LanguageIdentification:
    """
    Identifies the primary language of text using fasttext's
    pretrained model lid.176.bin
    https://fasttext.cc/docs/en/language-identification.html#content
    """

    def __init__(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(current_dir, "models/lid.176.bin")
        self.model = fasttext.load_model(path)

    def predict_lang(self, text: str) -> Tuple[str, float]:
        """
        Returns the top language
        """
        lang_tuple, prob = self.model.predict(text, k=1)
        lang = lang_tuple[0].replace('__label__', '')
        return (lang, round(prob[0], 2))
