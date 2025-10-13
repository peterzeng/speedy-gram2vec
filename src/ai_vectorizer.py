import pandas as pd
from pathlib import Path
import spacy
from spacy.tokens import Doc
from collections import Counter
from typing import Dict, List, Optional
from sys import stderr
import emoji
import sys
import os

# Add the src directory to the path to allow imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from matcher import SyntaxRegexMatcher


class Feature:
    """Base class for features that can be extracted from documents."""
    
    def __init__(self, name: str, counter: Counter, vocab: Optional[List[str]] = None):
        self.name = name
        self.counter = counter
        self.vocab = vocab or []
        
    def get_vector(self) -> Dict[str, float]:
        """Convert counter to normalized vector with zero counts for missing vocab items."""
        vector = {}
        total = sum(self.counter.values()) if self.counter else 1
        
        # Add counts for all vocab items (including zeros for missing ones)
        for item in self.vocab:
            ### HANNAH ###
            if self.vocab == ["count"]:
                key = "count"
                vector[f"{self.name}:{key}"] = self.counter[key]
            ### ^^^^^^ ###
            else:
                vector[f"{self.name}:{item}"] = self.counter.get(item, 0) / total
            
        return vector


class Gram2VecVectorizer:
    """Main vectorizer class that implements gram2vec functionality."""
    
    def __init__(self, language: str = "en", normalize: bool = True):
        self.language = language
        self.normalize = normalize
        
        # Load spaCy model
        if language == "en":
            try:
                self.nlp = spacy.load("en_core_web_lg")
            except OSError:
                print(f"Downloading spaCy language model 'en_core_web_lg'", file=stderr)
                from spacy.cli import download
                download("en_core_web_lg")
                self.nlp = spacy.load("en_core_web_lg")
        
        # Initialize sentence matcher
        self.sentence_matcher = SyntaxRegexMatcher(language)
        
        # Load vocabularies
        self.vocabs = self._load_vocabs()
        
        # Register features
        self.registered_features = {}
        self._register_features()
    
    def _load_vocabs(self) -> Dict[str, List[str]]:
        """Load vocabulary files from the vocab directory."""
        vocab_dir = Path(__file__).parent.parent / "vocab"
        vocabs = {}
        
        # Load all vocabulary files
        for vocab_file in vocab_dir.glob("*.txt"):
            feature_name = vocab_file.stem
            with open(vocab_file, 'r', encoding='utf-8') as f:
                vocabs[feature_name] = [line.strip() for line in f if line.strip()]
        
        return vocabs
    
    def _register_features(self):
        """Register all feature extraction functions."""
        self.registered_features = {
            "pos_unigrams": self._extract_pos_unigrams,
            "pos_bigrams": self._extract_pos_bigrams,
            "dep_labels": self._extract_dep_labels,
            "morph_tags": self._extract_morph_tags,
            "sentence_types": self._extract_sentence_types,
            "emojis": self._extract_emojis,
            "func_words": self._extract_func_words,
            "punctuation": self._extract_punctuation,
            "letters": self._extract_letters,
            "transitions": self._extract_transition_words,
            "unique_transitions": self._extract_unique_transitions,
            "tokens": self._extract_tokens,
            "types": self._extract_types,
            "sentence_count": self._extract_sentence_count,
            "named_entities": self._extract_named_entities,
            "suasive_verbs": self._extract_suasive_verbs,
            "stative_verbs": self._extract_stative_verbs,
        }
    
    def _extract_pos_unigrams(self, doc: Doc) -> Feature:
        """Extract POS unigrams."""
        pos_counts = Counter([token.pos_ for token in doc])
        return Feature("pos_unigrams", pos_counts, self.vocabs.get("pos_unigrams", []))
    
    def _extract_pos_bigrams(self, doc: Doc) -> Feature:
        """Extract POS bigrams."""
        pos_bigrams = []
        for i in range(len(doc) - 1):
            pos_bigrams.append(f"{doc[i].pos_} {doc[i+1].pos_}")
        pos_bigram_counts = Counter(pos_bigrams)
        return Feature("pos_bigrams", pos_bigram_counts, self.vocabs.get("pos_bigrams", []))
    
    def _extract_dep_labels(self, doc: Doc) -> Feature:
        """Extract dependency labels."""
        dep_counts = Counter([token.dep_ for token in doc])
        return Feature("dep_labels", dep_counts, self.vocabs.get("dep_labels", []))
    
    def _extract_morph_tags(self, doc: Doc) -> Feature:
        """Extract morphological tags."""
        morph_tags = []
        for token in doc:
            morph_str = str(token.morph)
            if morph_str:
                morph_tags.extend(morph_str.split("|"))
        morph_counts = Counter(morph_tags)
        return Feature("morph_tags", morph_counts, self.vocabs.get("morph_tags", []))
    
    def _extract_sentence_types(self, doc: Doc) -> Feature:
        """Extract sentence types using regex patterns on dependency parse trees."""
        # Use the SyntaxRegexMatcher to find sentence constructions
        matches = self.sentence_matcher.match_document(doc)
        
        # Extract pattern names from matches
        sentence_types = [match.pattern_name for match in matches]
        
        # If no matches found, add basic sentence type detection as fallback
        if not sentence_types:
            for sent in doc.sents:
                sent_text = sent.text.strip()
                if sent_text.endswith('.'):
                    sentence_types.append("declarative")
                elif sent_text.endswith('?'):
                    sentence_types.append("interrogative")
                elif sent_text.endswith('!'):
                    sentence_types.append("exclamatory")
                else:
                    sentence_types.append("declarative")
        
        sent_counts = Counter(sentence_types)
        return Feature("sentence_types", sent_counts, self.vocabs.get("sentences", []))
    
    def _extract_emojis(self, doc: Doc) -> Feature:
        """Extract emojis from text."""
        # Extract emoji characters from the emoji list (each item is a dict with 'emoji' key)
        emoji_chars = [item['emoji'] for item in emoji.emoji_list(doc.text)]
        emoji_counts = Counter(emoji_chars)
        return Feature("emojis", emoji_counts, self.vocabs.get("emojis", []))
    
    def _extract_func_words(self, doc: Doc) -> Feature:
        """Extract function words."""
        func_word_counts = Counter([token.text.lower() for token in doc 
                                  if token.text.lower() in self.vocabs.get("func_words", [])])
        return Feature("func_words", func_word_counts, self.vocabs.get("func_words", []))
    
    def _extract_punctuation(self, doc: Doc) -> Feature:
        """Extract punctuation."""
        punct_counts = Counter([token.text for token in doc if token.is_punct])
        return Feature("punctuation", punct_counts, self.vocabs.get("punctuation", []))
    
    def _extract_letters(self, doc: Doc) -> Feature:
        """Extract individual letters."""
        letter_counts = Counter([char for token in doc for char in token.text 
                               if char.isalpha()])
        return Feature("letters", letter_counts, self.vocabs.get("letters", []))
    
    ### HANNAH ###
    def _extract_transition_words(self, doc: Doc) -> Feature:
        """Extrat sentence-initial transition words.
        Must be followed by a comma, but can be preceded by 'and' or 'but'"""
        transitions = [trans.lower() for trans in self.vocabs.get("transitions", [])]
        transition_count = 0
        for sent in doc.sents:
            sent_tokens = [token.text.lower() for token in sent]

            for transition in transitions:
                trans_tokens = transition.split()
                n = len(trans_tokens)
                # initial quotation mark is ok
                if sent_tokens[0] in {"'", '"'}:
                    n += 1

                # starts with transition word
                if len(sent_tokens) > n+1 and sent_tokens[:n] == trans_tokens and sent_tokens[n] == ",":
                    transition_count += 1
                    break
                # starts with and/but
                elif len(sent_tokens) > n+2 and sent_tokens[0] in {"and", "but"} and sent_tokens[1:n+1] == trans_tokens and sent_tokens[n+1] == ",":
                    transition_count += 1
                    break

        return Feature("transitions", Counter({"count": transition_count}), ["count"])
    
    def _extract_unique_transitions(self, doc: Doc) -> Feature:
        """Extract unique sentence-initial transition words."""
        transitions = [trans.lower() for trans in self.vocabs.get("transitions", [])]
        unique_transitions = set()
        for sent in doc.sents:
            sent_tokens = [token.text.lower() for token in sent]

            for transition in transitions:
                trans_tokens = transition.split()
                n = len(trans_tokens)
                # initial quotation mark is ok
                if sent_tokens[0] in {"'", '"'}:
                    n += 1

                # starts with transition word
                if len(sent_tokens) > n+1 and sent_tokens[:n] == trans_tokens and sent_tokens[n] == ",":
                    unique_transitions.add(transition)
                    break
                # starts with and/but
                elif len(sent_tokens) > n+2 and sent_tokens[0] in {"and", "but"} and sent_tokens[1:n+1] == trans_tokens and sent_tokens[n+1] == ",":
                    unique_transitions.add(transition)
                    break

        return Feature("unique_transitions", Counter({"count": len(unique_transitions)}), ["count"])
    ### ^^^^^^^ ###
    
    def _extract_tokens(self, doc: Doc) -> Feature:
        """Extract token count (not normalized)."""
        # For tokens, we just return the total count as a single feature
        # This will be normalized by document length if normalize=True
        token_count = len(doc)
        return Feature("tokens", Counter({"count": token_count}), ["count"]) # H: added "count" as 3rd arg

    ### HANNAH ###
    def _extract_types(self, doc: Doc) -> Feature:
        """Extract type count (not normalized)."""
        type_count = len({token.text.lower() for token in doc})
        return Feature("types", Counter({"count": type_count}), ["count"])
    
    def _extract_sentence_count(self, doc: Doc) -> Feature:
        """Extract overall sentence count (not normalized)"""
        sentence_count = len(list(doc.sents))
        return Feature("sentence_count", Counter({"count": sentence_count}), ["count"])
    ### ^^^^^^ ###
    
    def _extract_named_entities(self, doc: Doc) -> Feature:
        """Extract named entities."""
        ne_counts = Counter([ent.label_ for ent in doc.ents])
        return Feature("named_entities", ne_counts, [])
    
    def _extract_suasive_verbs(self, doc: Doc) -> Feature:
        """Extract suasive verbs."""
        suasive_verbs = self.vocabs.get("suasive_verbs", [])
        suasive_counts = Counter([token.lemma_.lower() for token in doc 
                                if token.lemma_.lower() in suasive_verbs])
        return Feature("suasive_verbs", suasive_counts, suasive_verbs)
    
    def _extract_stative_verbs(self, doc: Doc) -> Feature:
        """Extract stative verbs."""
        stative_verbs = self.vocabs.get("stative_verbs", [])
        stative_counts = Counter([token.lemma_.lower() for token in doc 
                                if token.lemma_.lower() in stative_verbs])
        return Feature("stative_verbs", stative_counts, stative_verbs)
    
    def vectorize_document(self, text: str) -> Dict[str, float]:
        """Vectorize a single document."""
        doc = self.nlp(text)
        
        # Extract all features
        feature_vectors = {}
        # for feature_name, extractor in self.registered_features.items():
        #     feature = extractor(doc)
        #     feature_vectors.update(feature.get_vector())
        ### HANNAH ###
        for feature_name in default_config.keys():
            if feature_name in self.registered_features:
                feature = self.registered_features[feature_name](doc)
                feature_vectors.update(feature.get_vector())
        ### ^^^^^^ ###
        
        return feature_vectors
    
    def vectorize_documents(self, texts: List[str]) -> pd.DataFrame:
        """Vectorize multiple documents."""
        vectors = []
        for text in texts:
            vector = self.vectorize_document(text)
            vectors.append(vector)
        
        return pd.DataFrame(vectors)
    
    def from_jsonlines(self, directory_path: str, include_content_embedding: bool = False) -> pd.DataFrame:
        """Load documents from JSONL files and vectorize them."""
        # This is a placeholder - you would implement JSONL loading here
        # For now, we'll return an empty DataFrame
        return pd.DataFrame()
    
    def from_documents(self, documents: List[str], author_ids: List[str] = None, document_ids: List[str] = None) -> pd.DataFrame:
        """Vectorize documents with optional author and document IDs."""
        df = self.vectorize_documents(documents)
        
        if author_ids:
            df['authorID'] = author_ids
        if document_ids:
            df['documentID'] = document_ids
        
        return df


# Default configuration for features
default_config = {
    "pos_unigrams": 1,
    "pos_bigrams": 1,
    "dep_labels": 1,
    "morph_tags": 1,
    "sentence_types": 1,
    "emojis": 1,
    "func_words": 1,
    "punctuation": 1,
    "letters": 1,
    "transitions": 1,
    "unique_transitions": 1,
    "tokens": 1,
    "types": 1,
    "sentence_count": 1,
    "named_entities": 1,
    "suasive_verbs": 1,
    "stative_verbs": 1,
}


if __name__ == "__main__":
    # Test the vectorizer
    vectorizer = Gram2VecVectorizer("en", normalize=True)
    
    test_texts = [
        "Hello, world! I like to kick things and punch the wall. I like to jump on the ground.",
        "If you're visiting this page, you're likely here because you're searching for a random sentence. Sometimes a random word just isn't enough, and that is where the random sentence generator comes into play. By inputting the desired number, you can make a list of as many random sentences as you want or need. Producing random sentences can be helpful in a number of different ways.",
        "This dark chocolate is the best chocolate I have ever had!🥳",
        "My cat is smaller than my dog...",
        "This city is more beautiful than that city.",
        "Maria lives in Mexico City on January 1st, 2023.",
        "She lives in Mexico City on January 1st, 2023.",
        "Apple Inc. announced a new product yesterday.",
        "She is a doctor. The sky was blue. They are happy.",
        "What is the capital of France??!",
        "I suggest you try the new restaurant. The manager insists that you order the special. We recommend the chocolate cake for dessert.",
        "I know the answer. She believes in magic. The box contains three items. This book belongs to me. The recipe involves using fresh ingredients.",
        "He weighed the bananas carefully. The bananas weigh 2 pounds.",
        "They see this <PERSON> and that <PERSON>.",
        "However, <PERSON> lives in Mexico and works at Apple Inc. Nevertheless, he persisted. However, he died later.",
    ]

    parallel_texts = []
    author_ids = []
    doc_ids = []
    
    # Vectorize documents
    #test_vectors = vectorizer.vectorize_documents(test_texts)
    vectors = vectorizer.from_documents(parallel_texts, author_ids, doc_ids)

    """
        def from_documents(self, documents: List[str], author_ids: List[str] = None, 
                      document_ids: List[str] = None) -> pd.DataFrame:
    """
    
    # Save to CSV
    #test_vectors.to_csv("vectorized_docs_normalized.csv", index=False)
    vectors.to_csv("parallel_vectorized_docs.csv", index=False)
    print(f"Vectorized {len(test_texts)} documents with {len(vectors.columns)} features")
    print(f"Features: {list(vectors.columns[:10])}...")  # Show first 10 features

