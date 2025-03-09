from collections import Counter
import pandas as pd
from pathlib import Path
import numpy as np


class BiberGenreVectorizer:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.feature_types = [
            "pos_tags", 
            "pos_verbs", # redundant
            "pos_adjectives", # redundant
            "pos_adverbs", # redundant
            "pos_proper_nouns", # redundant
            "pos_adpositions", # redundant
            "pos_interjections", # redundant
            "dep_labels", 
            "morph_tags",
            "pos_bigrams",
            "sentences", 
            "func_words", 
            "punctuation", 
            "punct_periods", # redundant
            "punct_commas", # redundant
            "punct_colons", # redundant
            "punct_semicolons", # redundant
            "punct_exclamations", # redundant
            "punct_questions", # redundant
            "letters", 
            "tokens", 
            "named_entities",
            "NEs_person", 
            "NEs_location_loc",
            "NEs_location_gpe",
            "NEs_organization",
            "NEs_date",
            "NEs_without_date",
            "token_VB", # Verb base form
            "token_VBD", # Verb past tense
            "token_VBG", # Verb gerund
            "token_VBN", # Verb past participle
            "token_VBP", # Verb present participle
            "token_VBZ", # Verb present tense
            "token_EX", # Existential there
            "token_FW", # Foreign word
            "token_PRP", # Personal pronoun
            "token_superlatives", # Superlative
            "token_comparatives", # Comparative
            "first_second_person_pronouns", # First or second person pronoun
            "third_person_pronouns", # Third person pronoun
            "pronoun_it"]
        self.vector_length = len(self.feature_types)
        
    def init_vectorizer(self, text: str) -> pd.Series:
        vector = np.zeros(self.vector_length)
        return vector

    def vectorize_file(self, file_path: Path) -> pd.Series:
        pass

    def vectorize_directory(self, directory_path: Path) -> pd.DataFrame:
        pass
