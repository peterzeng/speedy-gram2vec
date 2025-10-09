import pandas as pd
from pathlib import Path
import spacy
from spacy.tokens import Doc
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple
from sys import stderr
import emoji
import sys
import os
from tqdm import tqdm

# Add the src directory to the path to allow imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from matcher import SyntaxRegexMatcher


class Feature:
    """Base class for features that can be extracted from documents."""
    
    def __init__(self, name: str, counter: Counter, vocab: Optional[List[str]] = None, norm_total: Optional[int] = None):
        self.name = name
        self.counter = counter
        self.vocab = vocab or []
        self.norm_total = norm_total
        
    def get_vector(self) -> Dict[str, float]:
        """Convert counter to vector. Normalize by total unless feature is 'num_tokens'."""
        normalized_vector: Dict[str, float] = {}
        vector = {}
        if self.name == "num_tokens":
            # Emit raw token count without normalization
            items = self.vocab or list(self.counter.keys())
            for item in items:
                vector[f"{self.name}:{item}"] = float(self.counter.get(item, 0))
            return vector

        # Choose normalization denominator
        if self.norm_total is not None:
            total = max(1, int(self.norm_total))
        else:
            total = sum(self.counter.values()) if self.counter else 1
        # Add counts for all vocab items (including zeros for missing ones)
        for item in self.vocab:
            if self.vocab == ["count"]:
                key = "count"
                vector[f"{self.name}:{key}"] = self.counter[key]
            vector[f"{self.name}:{item}"] = self.counter.get(item, 0) / total
        return vector


class Gram2VecVectorizer:
    """Main vectorizer class that implements gram2vec functionality."""
    
    def __init__(self, language: str = "en", normalize: bool = True,
                 enabled_features: Optional[Dict[str, int]] = None,
                 spacy_model: str = "en_core_web_lg",
                 n_process: Optional[int] = 1,
                 batch_size: int = 1000):
        self.language = language
        self.normalize = normalize
        self.enabled_features = enabled_features or default_config
        self.n_process = n_process
        self.batch_size = batch_size
        
        # Load spaCy model
        if language == "en":
            exclude = []
            if not self.enabled_features.get("named_entities", 1):
                exclude.append("ner")
            try:
                self.nlp = spacy.load(spacy_model, exclude=exclude)
            except OSError:
                print(f"Downloading spaCy language model '{spacy_model}'", file=stderr)
                from spacy.cli import download
                download(spacy_model)
                self.nlp = spacy.load(spacy_model, exclude=exclude)
        
        # Initialize sentence matcher
        self.sentence_matcher = SyntaxRegexMatcher(language)
        
        # Load vocabularies
        self.vocabs = self._load_vocabs()
        # Cached set versions for fast membership
        self.vocab_sets = {
            name: set(values) for name, values in self.vocabs.items()
        }
        
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
            "sentences": self._extract_sentence_types,
            "emojis": self._extract_emojis,
            "func_words": self._extract_func_words,
            "punctuation": self._extract_punctuation,
            "letters": self._extract_letters,
            "tokens": self._extract_tokens,
            "num_tokens": self._extract_num_tokens,
            "types": self._extract_types,
            "sentence_count": self._extract_sentence_count,
            "transitions": self._extract_transition_words,
            "unique_transitions": self._extract_unique_transitions,
            "named_entities": self._extract_named_entities,
            "suasive_verbs": self._extract_suasive_verbs,
            "stative_verbs": self._extract_stative_verbs,
        }
    
    def _extract_pos_unigrams(self, doc: Doc) -> Feature:
        """Extract POS unigrams."""
        pos_counts = Counter([token.pos_ for token in doc])
        return Feature("pos_unigrams", pos_counts, self.vocabs.get("pos_unigrams", []))
    
    def _extract_pos_bigrams(self, doc: Doc) -> Feature:
        """Extract POS bigrams using the same BOS/EOS insertion logic as gram2vec."""
        # Reconstruct sentence spans as (start, end) indices
        spans: List[tuple] = [(s.start, s.end) for s in doc.sents]
        pos_tags: List[str] = [t.pos_ for t in doc]

        seq: List[str] = []
        for i, pos in enumerate(pos_tags):
            # Insert BOS or EOS based on span boundaries (mirror gram2vec's elif ordering)
            for start, end in spans:
                if i == start:
                    seq.append("BOS")
                elif i == end:
                    seq.append("EOS")
            seq.append(pos)
        # Append final EOS after all tokens
        seq.append("EOS")

        pos_bigrams: List[str] = []
        for i in range(len(seq) - 1):
            pos_bigrams.append(f"{seq[i]} {seq[i+1]}")
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
        # Normalize by number of tokens to match gram2vec behavior
        return Feature("sentences", sent_counts, self.vocabs.get("sentences", []), norm_total=len(doc))
    
    def _extract_emojis(self, doc: Doc) -> Feature:
        """Extract emojis from text."""
        # Extract emoji characters from the emoji list (each item is a dict with 'emoji' key)
        emoji_chars = [item['emoji'] for item in emoji.emoji_list(doc.text)]
        emoji_counts = Counter(emoji_chars)
        # Normalize by number of tokens to match gram2vec behavior
        return Feature("emojis", emoji_counts, self.vocabs.get("emojis", []), norm_total=len(doc))
    
    def _extract_func_words(self, doc: Doc) -> Feature:
        """Extract function words (case-sensitive to match gram2vec)."""
        vocab_list = self.vocabs.get("func_words", [])
        vocab = set(vocab_list)
        func_word_counts = Counter([token.text for token in doc if token.text in vocab])
        # Normalize by number of tokens to match gram2vec behavior
        return Feature("func_words", func_word_counts, vocab_list, norm_total=len(doc))
    
    def _extract_punctuation(self, doc: Doc) -> Feature:
        """Extract punctuation as character-level counts to match gram2vec behavior."""
        vocab = set(self.vocabs.get("punctuation", []))
        counts: Counter = Counter()
        for token in doc:
            text = token.text
            for ch in text:
                if ch in vocab:
                    counts[ch] += 1
        return Feature("punctuation", counts, list(vocab), norm_total=len(doc))
    
    def _extract_letters(self, doc: Doc) -> Feature:
        """Extract individual letters."""
        letter_counts = Counter([char for token in doc for char in token.text 
                               if char.isalpha()])
        return Feature("letters", letter_counts, self.vocabs.get("letters", []))
    
    def _extract_tokens(self, doc: Doc) -> Feature:
        """Extract token count (normalized by document length)."""
        # For tokens, we just return the total count as a single feature
        # This will be normalized by document length if normalize=True
        token_count = len(doc)
        return Feature("tokens", Counter({"count": token_count}), [])

    def _extract_num_tokens(self, doc: Doc) -> Feature:
        """Extract raw token count to match gram2vec's 'num_tokens:num_tokens' feature."""
        token_count = len(doc)
        # Provide vocab with the single expected key to force stable column emission
        return Feature("num_tokens", Counter({"num_tokens": token_count}), ["num_tokens"])
    
    ### HANNAH ###
    def _extract_types(self, doc: Doc) -> Feature:
        """Extract type count (unique words, not normalized)."""
        type_count = len({token.text.lower() for token in doc})
        return Feature("types", Counter({"count": type_count}), ["count"])
    
    def _extract_sentence_count(self, doc: Doc) -> Feature:
        """Extract overall sentence count (not normalized)."""
        sentence_count = len(list(doc.sents))
        return Feature("sentence_count", Counter({"count": sentence_count}), ["count"])
    
    def _extract_transition_words(self, doc: Doc) -> Feature:
        """Extract sentence-initial transition words.
        Must be followed by a comma, but can be preceded by 'and' or 'but'."""
        transitions = [trans.lower() for trans in self.vocabs.get("transitions", [])]
        transition_count = 0
        for sent in doc.sents:
            sent_tokens = [token.text.lower() for token in sent]
            
            for transition in transitions:
                trans_tokens = transition.split()
                n = len(trans_tokens)
                # initial quotation mark is ok
                if sent_tokens and sent_tokens[0] in {"'", '"'}:
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
                if sent_tokens and sent_tokens[0] in {"'", '"'}:
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
        for feature_name, extractor in self.registered_features.items():
            feature = extractor(doc)
            feature_vectors.update(feature.get_vector())
        
        return feature_vectors
    
    def vectorize_documents(self, texts: List[str]) -> pd.DataFrame:
        """Vectorize multiple documents using batched spaCy processing."""
        vectors: List[Dict[str, float]] = []
        # Determine processes
        n_proc = self.n_process if self.n_process is not None else os.cpu_count()
        for doc in tqdm(self.nlp.pipe(texts, n_process=n_proc, batch_size=self.batch_size), 
                        total=len(texts), desc="Vectorizing documents", unit="doc"):
            vector = self._vectorize_doc_fast(doc)
            vectors.append(vector)
        return pd.DataFrame(vectors)

    def _vectorize_doc_fast(self, doc: Doc) -> Dict[str, float]:
        """Single-pass extraction for core features, fallback to existing extractors for others."""
        token_count = len(doc)
        # Core counters
        pos_counts: Counter = Counter()
        dep_counts: Counter = Counter()
        morph_counts: Counter = Counter()
        func_counts: Counter = Counter()
        punct_counts: Counter = Counter()
        letter_counts: Counter = Counter()

        # Single pass over tokens
        func_vocab = self.vocab_sets.get("func_words", set())
        punct_vocab = self.vocab_sets.get("punctuation", set())
        letters_vocab = self.vocab_sets.get("letters", set())

        for tok in doc:
            pos_counts[tok.pos_] += 1
            dep_counts[tok.dep_] += 1
            morph_str = str(tok.morph)
            if morph_str:
                for m in morph_str.split("|"):
                    if m:
                        morph_counts[m] += 1
            text = tok.text
            if text in func_vocab:
                func_counts[text] += 1
            # char-level punctuation/letters
            for ch in text:
                if ch in punct_vocab:
                    punct_counts[ch] += 1
                if ch in letters_vocab:
                    letter_counts[ch] += 1

        # POS bigrams with BOS/EOS
        spans: List[Tuple[int, int]] = [(s.start, s.end) for s in doc.sents]
        pos_tags: List[str] = [t.pos_ for t in doc]
        seq: List[str] = []
        for i, pos in enumerate(pos_tags):
            for start, end in spans:
                if i == start:
                    seq.append("BOS")
                elif i == end:
                    seq.append("EOS")
            seq.append(pos)
        seq.append("EOS")
        bigram_counts: Counter = Counter()
        for i in range(len(seq) - 1):
            bigram_counts[f"{seq[i]} {seq[i+1]}"] += 1

        # Sentences via matcher
        matches = self.sentence_matcher.match_document(doc)
        sentence_types = [m.pattern_name for m in matches]
        sent_counts = Counter(sentence_types)

        # Emojis from full text
        emoji_chars = [item['emoji'] for item in emoji.emoji_list(doc.text)]
        emoji_counts = Counter(emoji_chars)

        # Build vectors using Feature wrappers
        vectors: Dict[str, float] = {}
        def add_feature(name: str, counter: Counter, vocab_key: Optional[str], norm_total: Optional[int] = None):
            if not self.enabled_features.get(name, 1):
                return
            vocab = self.vocabs.get(vocab_key or name, [])
            feat = Feature(name, counter, vocab, norm_total=norm_total)
            vectors.update(feat.get_vector())

        add_feature("pos_unigrams", pos_counts, "pos_unigrams")
        add_feature("pos_bigrams", bigram_counts, "pos_bigrams")
        add_feature("dep_labels", dep_counts, "dep_labels")
        add_feature("morph_tags", morph_counts, "morph_tags")
        add_feature("func_words", func_counts, "func_words", norm_total=token_count)
        add_feature("punctuation", punct_counts, "punctuation", norm_total=token_count)
        add_feature("letters", letter_counts, "letters")
        add_feature("sentences", sent_counts, "sentences", norm_total=token_count)
        add_feature("emojis", emoji_counts, "emojis", norm_total=token_count)
        # tokens / num_tokens
        if self.enabled_features.get("tokens", 1):
            vectors.update(Feature("tokens", Counter({"count": token_count}), []).get_vector())
        if self.enabled_features.get("num_tokens", 1):
            vectors.update(Feature("num_tokens", Counter({"num_tokens": token_count}), ["num_tokens"]).get_vector())

        ### HANNAH ###
        # types (unique words)
        if self.enabled_features.get("types", 1):
            type_count = len({tok.text.lower() for tok in doc})
            vectors.update(Feature("types", Counter({"count": type_count}), ["count"]).get_vector())
        
        # sentence_count
        if self.enabled_features.get("sentence_count", 1):
            sentence_count = len(list(doc.sents))
            vectors.update(Feature("sentence_count", Counter({"count": sentence_count}), ["count"]).get_vector())
        
        # Remaining features using existing methods if enabled
        if self.enabled_features.get("transitions", 1):
            vectors.update(self._extract_transition_words(doc).get_vector())
        if self.enabled_features.get("unique_transitions", 1):
            vectors.update(self._extract_unique_transitions(doc).get_vector())
        ### ^^^^^^ ###
        
        if self.enabled_features.get("named_entities", 1):
            vectors.update(self._extract_named_entities(doc).get_vector())
        if self.enabled_features.get("suasive_verbs", 1):
            vectors.update(self._extract_suasive_verbs(doc).get_vector())
        if self.enabled_features.get("stative_verbs", 1):
            vectors.update(self._extract_stative_verbs(doc).get_vector())

        return vectors
    
    def from_jsonlines(self, directory_path: str, include_content_embedding: bool = False) -> pd.DataFrame:
        """Load documents from JSONL files and vectorize them."""
        # This is a placeholder - you would implement JSONL loading here
        # For now, we'll return an empty DataFrame
        return pd.DataFrame()
    
    def from_documents(self, documents: List[str], author_ids: List[str] = None, 
                      document_ids: List[str] = None) -> pd.DataFrame:
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
    "sentences": 1,
    "emojis": 1,
    "func_words": 1,
    "punctuation": 1,
    "letters": 1,
    "transitions": 1,
    "unique_transitions": 1,
    "tokens": 1,
    "num_tokens": 1,
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
        "<PERSON> lives in Mexico and works at Apple Inc."
    ]
    
    # Vectorize documents
    vectors = vectorizer.vectorize_documents(test_texts)
    
    # Save to CSV
    vectors.to_csv("vectorized_docs_normalized.csv", index=False)
    print(f"Vectorized {len(test_texts)} documents with {len(vectors.columns)} features")
    print(f"Features: {list(vectors.columns[:10])}...")  # Show first 10 features

