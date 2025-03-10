from collections import Counter
import pandas as pd
from pathlib import Path
import numpy as np
import spacy
from spacy.tokens import Doc
import feature_counter_utils as fcu
from sys import stderr


class BiberGenreVectorizer:
    def __init__(self, language: str):

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
    
    # 0 0 0 0 0 0 0 0 0 0 
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
        
        feature_counter["avg_chars_per_token"] = fcu.avg_chars_per_token(text)
        feature_counter["avg_tokens_per_sentence"] = fcu.avg_tokens_per_sentence(text)
        feature_counter["avg_noun_chunk_length"] = fcu.avg_noun_chunk_length(text)
        feature_counter["avg_verb_chunk_length"] = fcu.avg_verb_chunk_length(text)
        feature_counter["punct_periods"] = fcu.count_punctuation(text, "periods")
        feature_counter["punct_questions"] = fcu.count_punctuation(text, "questions")
        feature_counter["punct_exclamations"] = fcu.count_punctuation(text, "exclamations")
        feature_counter["punct_commas"] = fcu.count_punctuation(text, "commas")
        feature_counter["pos_adpositions"] = fcu.count_adpositions(text)
        feature_counter["pos_interjections"] = fcu.count_interjections(text)
        feature_counter["pos_adverbs"] = fcu.count_adverbs(text)
        feature_counter["pos_adjectives"] = fcu.count_adjectives(text)
        feature_counter["pos_nouns"] = fcu.count_nouns(text)
        feature_counter["pos_verbs"] = fcu.count_verbs(text)
        feature_counter["pos_proper_nouns"] = fcu.count_proper_nouns(text)
        
        # Add token tag counting
        feature_counter["token_VB"] = fcu.count_VB(text)
        feature_counter["token_VBD"] = fcu.count_VBD(text)
        feature_counter["token_VBG"] = fcu.count_VBG(text)
        feature_counter["token_VBN"] = fcu.count_VBN(text)
        feature_counter["token_VBP"] = fcu.count_VBP(text)
        feature_counter["token_VBZ"] = fcu.count_VBZ(text)
        feature_counter["token_EX"] = fcu.count_EX(text)
        feature_counter["token_FW"] = fcu.count_FW(text)
        feature_counter["token_superlatives"] = fcu.count_superlatives(text)
        
        # Add pronoun counting
        feature_counter["first_second_person_pronouns"] = fcu.count_first_second_person_pronouns(text)
        feature_counter["third_person_pronouns"] = fcu.count_third_person_pronouns(text)
        feature_counter["pronoun_it"] = fcu.count_pronoun_it(text)
        feature_counter["pos_pronouns"] = fcu.count_pos_tags(text, "PRON")
        
        # Add named entity counting
        feature_counter["named_entities"] = fcu.count_named_entities(text)
        feature_counter["NEs_without_date"] = fcu.count_named_entities(text, "except_date")
        feature_counter["NEs_person"] = fcu.count_named_entities(text, "person")
        feature_counter["NEs_date"] = fcu.count_named_entities(text, "date")
        feature_counter["NEs_location_loc"] = fcu.count_named_entities(text, "location_loc")
        feature_counter["NEs_location_gpe"] = fcu.count_named_entities(text, "location_gpe")
        feature_counter["NEs_organization"] = fcu.count_named_entities(text, "organization")
        
        # Add copula verbs counting
        feature_counter["copula_verbs"] = fcu.count_copula_verbs(text)

        # Create a DataFrame with a single row containing all feature values
        return pd.DataFrame([feature_counter], columns=self.biber_feature_types)

    def vectorize_directory(self, directory_path: Path) -> pd.DataFrame:
        pass

    def process_texts(self, documents: list[str]) -> list[Doc]:
        nlp_docs = self.nlp.pipe(documents)

        vectorized_docs = []
        for doc in nlp_docs:
            vectorized_docs.append(self.vectorize_text(doc))

        return pd.concat(vectorized_docs)
    
if __name__ == "__main__":
    vectorizer = BiberGenreVectorizer("en")
    text = ["Hello, world! I like to kick things and punch the wall. I like to jump on the ground.",
            "If you're visiting this page, you're likely here because you're searching for a random sentence. Sometimes a random word just isn't enough, and that is where the random sentence generator comes into play. By inputting the desired number, you can make a list of as many random sentences as you want or need. Producing random sentences can be helpful in a number of different ways.",
            "This dark chocolate is the best chocolate I have ever had!🥳",
            "My cat is smaller than my dog...",
            "This city is more beautiful than that city.",
            "Maria lives in Mexico City on January 1st, 2023.",
            "Apple Inc. announced a new product yesterday.",
            "He is playing with his dog."]
    
    doc = vectorizer.process_texts(text)
    doc.to_csv("vectorized_docs.csv", index=False)