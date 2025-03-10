'''
    This module contains utility functions for counting spaCy extracted features in a document.
'''

from spacy.tokens import Doc
from typing import Dict, List, Callable, Any, Union
from collections import Counter
from pathlib import Path
from sys import stderr

# features:
# pos_tags"pos_tags", 
# "pos_verbs", # redundant
# "pos_adjectives", # redundant
# "pos_adverbs", # redundant
# "pos_proper_nouns", # redundant
# "pos_adpositions", # redundant
# "pos_interjections", # redundant
# "dep_labels", 
# "morph_tags",
# "pos_bigrams",
# "sentences", 
# "func_words", 
# "punctuation", 
# "punct_periods", # redundant
# "punct_commas", # redundant
# "punct_colons", # redundant
# "punct_semicolons", # redundant
# "punct_exclamations", # redundant
# "punct_questions", # redundant
# "letters", 
# "tokens", 
# "named_entities",
# "NEs_person", 
# "NEs_location_loc",
# "NEs_location_gpe",
# "NEs_organization",
# "NEs_date",
# "NEs_without_date",
# "token_VB", # Verb base form
# "token_VBD", # Verb past tense
# "token_VBG", # Verb gerund
# "token_VBN", # Verb past participle
# "token_VBP", # Verb present participle
# "token_VBZ", # Verb present tense
# "token_EX", # Existential there
# "token_FW", # Foreign word
# "token_PRP", # Personal pronoun
# "token_superlatives", # Superlative
# "token_comparatives", # Comparative
# "first_second_person_pronouns", # First or second person pronoun
# "third_person_pronouns", # Third person pronoun
# "pronoun_it",
# "avg_noun_chunk_length",
# "avg_verb_chunk_length",
# "avg_chars_per_token",
# "avg_tokens_per_sentence",

# Generic POS tag counter
def count_pos_tags(doc: Doc, pos_tag: str = None) -> Union[int, Dict[str, int]]:
    """
    Count POS tags in a document.
    If pos_tag is provided, count occurrences of that specific tag.
    Otherwise, return a dictionary of all POS tags and their counts.
    """
    if pos_tag:
        return sum(1 for token in doc if token.pos_ == pos_tag)
    else:
        return dict(Counter([token.pos_ for token in doc]))

# Specific POS tag counters
def count_verbs(doc: Doc) -> int:
    return count_pos_tags(doc, "VERB")

def count_adjectives(doc: Doc) -> int:
    return count_pos_tags(doc, "ADJ")

def count_adverbs(doc: Doc) -> int:
    return count_pos_tags(doc, "ADV")

def count_proper_nouns(doc: Doc) -> int:
    return count_pos_tags(doc, "PROPN")

def count_adpositions(doc: Doc) -> int:
    return count_pos_tags(doc, "ADP")

def count_interjections(doc: Doc) -> int:
    return count_pos_tags(doc, "INTJ")

def count_nouns(doc: Doc) -> int:
    return count_pos_tags(doc, "NOUN")

def count_pronouns(doc: Doc) -> int:
    return count_pos_tags(doc, "PRON")

# Dependency label counter
def count_dep_labels(doc: Doc, dep_label: str = None) -> Union[int, Dict[str, int]]:
    """
    Count dependency labels in a document.
    If dep_label is provided, count occurrences of that specific label.
    Otherwise, return a dictionary of all dependency labels and their counts.
    """
    if dep_label:
        return sum(1 for token in doc if token.dep_ == dep_label)
    else:
        return dict(Counter([token.dep_ for token in doc]))

# Morphological tag counter
def count_morph_tags(doc: Doc) -> Dict[str, int]:
    """Count morphological tags in a document."""
    morph_tags = []
    for token in doc:
        morph_tags.extend(str(token.morph).split("|"))
    return dict(Counter(morph_tags))

# POS bigram counter
def count_pos_bigrams(doc: Doc) -> Dict[str, int]:
    """Count POS tag bigrams in a document."""
    pos_bigrams = []
    for i in range(len(doc) - 1):
        pos_bigrams.append(f"{doc[i].pos_}_{doc[i+1].pos_}")
    return dict(Counter(pos_bigrams))

# Sentence counter
def count_sentences(doc: Doc) -> int:
    """Count sentences in a document."""
    return len(list(doc.sents))

# Function word counter (example - you'd need to define your function words)
def count_func_words(doc: Doc) -> int:
    """Count function words in a document using spaCy's is_stop attribute.
    
    spaCy's stop words generally correspond to function words - articles,
    prepositions, conjunctions, auxiliary verbs, etc.
    """
    return sum(1 for token in doc if token.is_stop)

# Punctuation counters
def count_punctuation(doc: Doc, punct_type: str = None) -> Union[int, Dict[str, int]]:
    """
    Count punctuation in a document.
    If punct_type is provided, count occurrences of that specific punctuation.
    Otherwise, return the total count of all punctuation.
    """
    punct_map = {
        "periods": ".",
        "commas": ",",
        "colons": ":",
        "semicolons": ";",
        "exclamations": "!",
        "questions": "?"
    }
    
    if punct_type:
        if punct_type in punct_map:
            return sum(1 for token in doc if token.is_punct and token.text == punct_map[punct_type])
        else:
            raise ValueError(f"Unknown punctuation type: {punct_type}")
    else:
        return sum(1 for token in doc if token.is_punct)

# Named entity counters
def count_named_entities(doc: Doc, entity_type: str = None) -> Union[int, Dict[str, int]]:
    """
    Count named entities in a document.
    If entity_type is provided, count occurrences of that specific entity type.
    Otherwise, return the total count of all named entities.
    """
    named_entity_map = {
        "person": "PERSON",
        "location_loc": "LOC",
        "location_gpe": "GPE",
        "organization": "ORG",
        "date": "DATE"
    }
    
    if entity_type:
        if entity_type in named_entity_map:
            return sum(1 for ent in doc.ents if ent.label_ == named_entity_map[entity_type])
        elif entity_type == "without_date":
            return sum(1 for ent in doc.ents if ent.label_ != "DATE")
        else:
            raise ValueError(f"Unknown named entity type: {entity_type}")
    else:
        return len(doc.ents)

# Token-specific counters based on fine-grained tags
def count_token_by_tag(doc: Doc, tag: str = None) -> Union[int, Dict[str, int]]:
    """
    Count tokens with specific token_tags in a document.
    If tag is provided, count occurrences of that specific tag.
    Otherwise, return a dictionary of all token_tags and their counts.
    """
    if tag:
        return sum(1 for token in doc if token.tag_ == tag)
    else:
        return dict(Counter([token.tag_ for token in doc]))

# Specific verb tag counters
def count_VB(doc: Doc) -> int:
    """Count base form verbs (VB)."""
    return count_token_by_tag(doc, "VB")

def count_VBD(doc: Doc) -> int:
    """Count past tense verbs (VBD)."""
    return count_token_by_tag(doc, "VBD")

def count_VBG(doc: Doc) -> int:
    """Count gerund or present participle verbs (VBG)."""
    return count_token_by_tag(doc, "VBG")

def count_VBN(doc: Doc) -> int:
    """Count past participle verbs (VBN)."""
    return count_token_by_tag(doc, "VBN")

def count_VBP(doc: Doc) -> int:
    """Count non-3rd person singular present verbs (VBP)."""
    return count_token_by_tag(doc, "VBP")

def count_VBZ(doc: Doc) -> int:
    """Count 3rd person singular present verbs (VBZ)."""
    return count_token_by_tag(doc, "VBZ")

def count_EX(doc: Doc) -> int:
    """Count existential 'there' (EX)."""
    return count_token_by_tag(doc, "EX")

def count_FW(doc: Doc) -> int:
    """Count foreign words (FW)."""
    return count_token_by_tag(doc, "FW")

def count_PRP(doc: Doc) -> int:
    """Count personal pronouns (PRP)."""
    return count_token_by_tag(doc, "PRP")

def count_superlatives(doc: Doc) -> int:
    """Count superlative adjectives (JJS) and adverbs (RBS)."""
    return sum(1 for token in doc if token.tag_ == "JJS" or token.tag_ == "RBS")

def count_copula_verbs(doc: Doc) -> int:
    """Count copula (be) verbs."""
    return sum(1 for token in doc if token.lemma_ == "be" and (any(child.dep_ in ["attr", "acomp"] for child in token.children))) 

# Pronoun counters
def count_first_second_person_pronouns(doc: Doc) -> int:
    """Count first and second person pronouns."""
    first_second_pronouns = ["i", "me", "my", "mine", "myself", 
                            "we", "us", "our", "ours", "ourselves",
                            "you", "your", "yours", "yourself", "yourselves"]
    return sum(1 for token in doc if token.text.lower() in first_second_pronouns)

def count_third_person_pronouns(doc: Doc) -> int:
    """Count third person pronouns."""
    third_person_pronouns = ["he", "him", "his", "himself", 
                            "she", "her", "hers", "herself",
                            "they", "them", "their", "theirs", "themselves"]
    return sum(1 for token in doc if token.text.lower() in third_person_pronouns)

def count_pronoun_it(doc: Doc) -> int:
    """Count occurrences of the pronoun 'it'."""
    return sum(1 for token in doc if token.text.lower() == "it")

def count_suasive_verbs(doc: Doc) -> int:
    """
    The list of suasive verbs is read from a text file.
    """
    suasive_verbs_file = Path(__file__).parent.parent / "vocab" / "suasive-verbs.txt"

    with open(suasive_verbs_file, 'r', encoding='utf-8') as f:
            suasive_verbs = [line.strip() for line in f if line.strip()]
            
    return sum(1 for token in doc if token.lemma_.lower() in suasive_verbs)

def count_stative_verbs(doc: Doc) -> int:
    """
    The list of stative verbs is read from a text file.
    
    For ambiguous verbs (marked with "(amb)" in the .txt file), we check if it's a transitive verb 
    (=with a direct object) or not. If it is, we don't count it as stative.
    For example:
    - "He weighed the bananas" - 'weighed' is transitive, not counted as stative
    - "The bananas weigh 2 lbs" - 'weigh' is intransitive, counted as stative
    """
    stative_verbs_file = Path(__file__).parent.parent / "vocab" / "stative-verbs.txt"
    
    with open(stative_verbs_file, 'r', encoding='utf-8') as f:
            
            stative_verbs = []
            ambiguous_verbs = []
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                if "(amb)" in line:
                    verb = line.replace("(amb)", "").strip()
                    ambiguous_verbs.append(verb)
                else:  
                    stative_verbs.append(line) # Add to unambiguous stative verbs list
    
    # count stative verbs
    return sum(
        1 for token in doc if (
            token.lemma_.lower() in stative_verbs or
            (token.lemma_.lower() in ambiguous_verbs and not (
                any(child.dep_ == "dobj" for child in token.children) and
                any(child.dep_ == "nsubj" and 
                    (child.tag_ == "PRP" or child.ent_type_ == "PERSON")
                    for child in token.children)
            ))
        )
    )

# Average length metrics
def avg_noun_chunk_length(doc: Doc) -> float:
    """Calculate average length of noun chunks in tokens."""
    chunks = list(doc.noun_chunks)
    if not chunks:
        return 0.0
    return sum(len(chunk) for chunk in chunks) / len(chunks)

def avg_verb_chunk_length(doc: Doc) -> float:
    """Calculate average length of verb phrases in tokens."""
    # This is a simplified approach - you might need a more sophisticated
    # method to identify verb chunks depending on your needs
    verb_chunks = []
    for token in doc:
        if token.pos_ == "VERB":
            chunk = [token] + list(token.children)
            chunk = [t for t in chunk if t.dep_ in {"aux", "xcomp", "ccomp", "advcl", "prep", "pobj", "dobj", "nsubj", "nsubjpass", "csubj", "csubjpass", "attr", "acomp"}]
            verb_chunks.append(chunk)

    total_chunk_length = sum(len(chunk) for chunk in verb_chunks)
    num_chunks = len(verb_chunks)
    avg_chunk_length = total_chunk_length / num_chunks if num_chunks > 0 else 0
    return avg_chunk_length

def avg_chars_per_token(doc: Doc) -> float:
    """Calculate average characters per token."""
    if len(doc) == 0:
        return 0.0
    return sum(len(token.text) for token in doc) / len(doc)

def avg_tokens_per_sentence(doc: Doc) -> float:
    """Calculate average tokens per sentence."""
    sentences = list(doc.sents)
    if not sentences:
        return 0.0
    return len(doc) / len(sentences)

def normalize_features(doc: Doc, feature_counts: Dict[str, Any]) -> Dict[str, float]:
    """
    Normalize feature counts by the total number of tokens in the document.
    """
    # Get total number of tokens
    total_tokens = len(doc)
    
    # If document is empty, return zeros
    if total_tokens == 0:
        return {feature: 0.0 for feature in feature_counts}
    
    # Normalize each feature count
    normalized_counts = {}
    for feature, count in feature_counts.items():
        # Skip features that are already normalized (averages, etc.)
        if feature.startswith("avg_"):
            normalized_counts[feature] = count
        # Skip features that are dictionaries (like pos_tags when no specific tag is provided)
        elif isinstance(count, dict):
            normalized_counts[feature] = count
        # Normalize count features
        else:
            normalized_counts[feature] = count / total_tokens
    
    return normalized_counts

# Create a feature extractor dictionary for easy access to all functions
FEATURE_EXTRACTORS = {
    "pos_tags": count_pos_tags,
    "token_tags": count_token_by_tag,
    "pos_verbs": count_verbs,
    "pos_adjectives": count_adjectives,
    "pos_adverbs": count_adverbs,
    "pos_proper_nouns": count_proper_nouns,
    "pos_adpositions": count_adpositions,
    "pos_interjections": count_interjections,
    "pos_nouns": count_nouns,
    "pos_pronouns": count_pronouns,
    "dep_labels": count_dep_labels,
    "morph_tags": count_morph_tags,
    "pos_bigrams": count_pos_bigrams,
    "sentences": count_sentences,
    "func_words": count_func_words,
    "punctuation": count_punctuation,
    "punct_periods": lambda doc: count_punctuation(doc, "periods"),
    "punct_commas": lambda doc: count_punctuation(doc, "commas"),
    "punct_colons": lambda doc: count_punctuation(doc, "colons"),
    "punct_semicolons": lambda doc: count_punctuation(doc, "semicolons"),
    "punct_exclamations": lambda doc: count_punctuation(doc, "exclamations"),
    "punct_questions": lambda doc: count_punctuation(doc, "questions"),
    "tokens": lambda doc: len(doc),
    "named_entities": count_named_entities,
    "NEs_person": lambda doc: count_named_entities(doc, "person"),
    "NEs_location_loc": lambda doc: count_named_entities(doc, "location_loc"),
    "NEs_location_gpe": lambda doc: count_named_entities(doc, "location_gpe"),
    "NEs_organization": lambda doc: count_named_entities(doc, "organization"),
    "NEs_date": lambda doc: count_named_entities(doc, "date"),
    "NEs_without_date": lambda doc: count_named_entities(doc, "without_date"),
    "token_VB": count_VB,
    "token_VBD": count_VBD,
    "token_VBG": count_VBG,
    "token_VBN": count_VBN,
    "token_VBP": count_VBP,
    "token_VBZ": count_VBZ,
    "token_EX": count_EX,
    "token_FW": count_FW,
    "token_PRP": count_PRP,
    "token_superlatives": count_superlatives,
    "first_second_person_pronouns": count_first_second_person_pronouns,
    "third_person_pronouns": count_third_person_pronouns,
    "pronoun_it": count_pronoun_it,
    "avg_noun_chunk_length": avg_noun_chunk_length,
    "avg_verb_chunk_length": avg_verb_chunk_length,
    "avg_chars_per_token": avg_chars_per_token,
    "avg_tokens_per_sentence": avg_tokens_per_sentence,
    "copula_verbs": count_copula_verbs,
    "suasive_verbs": count_suasive_verbs,
    "stative_verbs": count_stative_verbs,
}

def extract_all_features(doc: Doc, normalize: bool = False) -> Dict[str, Any]:
    """
    Extract all features from a document.
    Returns a dictionary with feature names as keys and their values.
    """
    feature_counts = {feature_name: extractor(doc) for feature_name, extractor in FEATURE_EXTRACTORS.items()}
    
    if normalize:
        return normalize_features(doc, feature_counts)
    else:
        return feature_counts

def extract_features(doc: Doc, feature_names: List[str], normalize: bool = False) -> Dict[str, Any]:
    """
    Extract specified features from a document.
    Returns a dictionary with feature names as keys and their values.
    """
    feature_counts = {feature_name: FEATURE_EXTRACTORS[feature_name](doc) 
            for feature_name in feature_names if feature_name in FEATURE_EXTRACTORS}
    
    if normalize:
        return normalize_features(doc, feature_counts)
    else:
        return feature_counts
