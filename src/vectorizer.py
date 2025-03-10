from collections import Counter
import pandas as pd
from pathlib import Path
import numpy as np
import spacy
from spacy.tokens import Doc
import feature_counter_utils as fcu
from sys import stderr


class BiberGenreVectorizer:
    def __init__(self, language: str, normalize: bool = False):

        self.biber_feature_types = [
            "first_second_person_pronouns", "third_person_pronouns", "pronoun_it",
            "copula_verbs", "named_entities", "NEs_without_date", "token_VB", 
            "token_VBD", "token_VBG", "token_VBN", "token_VBP", "token_VBZ",
            "pos_nouns", "pos_verbs", "pos_proper_nouns", "pos_adjectives",
            "pos_adverbs", "token_superlatives", "pos_pronouns",
            "pos_adpositions", "token_FW", "token_EX", "pos_interjections",
            "NEs_person", "NEs_date", "NEs_location_loc", "NEs_location_gpe", "NEs_organization",
            ### NEED OWEN TO TELL US SUASIVE AND STATIVE VERBS ###
            "avg_noun_chunk_length", "avg_verb_chunk_length",
            "avg_tokens_per_sentence", "avg_chars_per_token",
            "punct_periods", "punct_questions", "punct_exclamations",
            "punct_commas"
            ]
        
        self.normalize = normalize
        
        if language == "en":
            try:
                self.nlp = spacy.load("en_core_web_lg")
            except OSError:
                print(f"Downloading spaCy language model 'en_core_web_lg'", file=stderr)
                from spacy.cli import download
                download("en_core_web_lg")
                
        elif language == "ru":
            try:
                self.nlp = spacy.load("ru_core_news_lg")
            except OSError:
                print(f"Downloading spaCy language model 'ru_core_news_lg'", file=stderr)
                from spacy.cli import download
                download("ru_core_news_lg")
    
    def init_vectorizer(self, text: str) -> pd.Series:
        feature_counter = {}
        for feature in self.biber_feature_types:
            feature_counter[feature] = 0
            
        return feature_counter
    
    '''
        returns a dataframe with the feature names as column names and the feature values as the values
    '''
    def vectorize_text(self, text: Doc) -> pd.DataFrame:
        feature_counter = self.init_vectorizer(text)

        # Get all token tags for analysis (not included in the final feature set)
        all_token_tags = fcu.count_token_by_tag(text)
        
        # Extract all features
        features = fcu.extract_features(text, self.biber_feature_types, normalize=self.normalize)
        
        # Create a DataFrame with a single row containing all feature values
        return pd.DataFrame([features], columns=self.biber_feature_types)

    def vectorize_directory(self, directory_path: Path) -> pd.DataFrame:
        pass

    def process_texts(self, documents: list[str]) -> list[Doc]:
        nlp_docs = self.nlp.pipe(documents)

        vectorized_docs = []
        for doc in nlp_docs:
            vectorized_docs.append(self.vectorize_text(doc))

        return pd.concat(vectorized_docs)
    
if __name__ == "__main__":    
    #vectorizer = BiberGenreVectorizer("en", normalize=False)
    normalized_vectorizer = BiberGenreVectorizer("en", normalize=True)
    
    text = ["Hello, world! I like to kick things and punch the wall. I like to jump on the ground.",
            "If you're visiting this page, you're likely here because you're searching for a random sentence. Sometimes a random word just isn't enough, and that is where the random sentence generator comes into play. By inputting the desired number, you can make a list of as many random sentences as you want or need. Producing random sentences can be helpful in a number of different ways.",
            "This dark chocolate is the best chocolate I have ever had!🥳",  
            "My cat is smaller than my dog...",  
            "This city is more beautiful than that city.", 
            "Maria lives in Mexico City on January 1st, 2023.",  
            "Apple Inc. announced a new product yesterday.",  
            "She is a doctor. The sky was blue. They are happy.",  
            "What is the capital of France??!"
            ]  
    

    
    normalized_features = normalized_vectorizer.process_texts(text)
    normalized_features.to_csv("vectorized_docs_normalized.csv", index=False)
    
    #doc = vectorizer.process_texts(text)
    #doc.to_csv("vectorized_docs.csv", index=False)

