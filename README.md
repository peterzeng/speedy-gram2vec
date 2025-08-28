# Speedy Gram2Vec

A fast and simplified implementation of [Gram2Vec](https://github.com/eric-sclafani/gram2vec), a grammatical style embedding algorithm that embeds documents into a higher dimensional space based on grammatical style features.

## 🚀 Overview

`Speedy Gram2Vec` is a rewrite of the original Gram2Vec package that simplifies and speeds up the code while maintaining full functionality. It produces document embeddings based on grammatical style features such as:

- **POS tags** (unigrams and bigrams)
- **Dependency labels**
- **Morphological tags**
- **Sentence types** (declarative, interrogative, imperative, etc.)
- **Emojis** and punctuation patterns
- **Function words** and specialized vocabulary
- **Named entities** (persons, locations, organizations)
- **Specialized verb types** (suasive, stative, copula)
- **Pronoun patterns** (first/second/third person)
- **Token-level features** (verb forms, superlatives, etc.)

Each position in the resulting vector corresponds to a tangible stylistic feature, making the embeddings interpretable and explainable.

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Quick Install

1. **Clone the repository:**
```bash
git clone <repository-url>
cd speedy-gram2vec
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download the required spaCy model:**
```bash
python -m spacy download en_core_web_lg
```

### Using Conda Environment

```bash
# Create and activate conda environment
conda create -n gram2vec python=3.11
conda activate gram2vec

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

## 🎯 Quick Start

### Basic Usage

```python
from src.vectorizer import Gram2VecVectorizer

# Initialize the vectorizer
vectorizer = Gram2VecVectorizer("en", normalize=True)

# Vectorize documents
documents = [
    "Hello, world! I like to kick things and punch the wall.",
    "This dark chocolate is the best chocolate I have ever had!🥳",
    "What is the capital of France??!"
]

# Get feature vectors
vectors = vectorizer.vectorize_documents(documents)
print(f"Vectorized {len(documents)} documents with {len(vectors.columns)} features")

# Save to CSV
vectors.to_csv("output_vectors.csv", index=False)
```

### Single Document Vectorization

```python
# Vectorize a single document
text = "This is a sample document for analysis."
vector = vectorizer.vectorize_document(text)
print(f"Document has {len(vector)} features")
```

### With Metadata

```python
# Vectorize documents with author and document IDs
documents = ["Document 1 text...", "Document 2 text...", "Document 3 text..."]
author_ids = ["author1", "author1", "author2"]
document_ids = ["doc1", "doc2", "doc3"]

df = vectorizer.from_documents(documents, author_ids, document_ids)
```

## 🔧 Features

### Core Grammatical Features

| Feature Type | Description | Example Features |
|--------------|-------------|------------------|
| **POS Unigrams** | Individual part-of-speech tags | `pos_unigrams:VERB`, `pos_unigrams:NOUN`, `pos_unigrams:ADJ` |
| **POS Bigrams** | Consecutive POS tag pairs | `pos_bigrams:VERB NOUN`, `pos_bigrams:ADJ NOUN` |
| **Dependency Labels** | Syntactic dependency relationships | `dep_labels:nsubj`, `dep_labels:dobj`, `dep_labels:prep` |
| **Morphological Tags** | Grammatical features | `morph_tags:Tense=Pres`, `morph_tags:Number=Sing` |
| **Sentence Types** | Sentence construction patterns | `sentences:declarative`, `sentences:interrogative` |

### Lexical and Stylistic Features

| Feature Type | Description | Example Features |
|--------------|-------------|------------------|
| **Function Words** | Common function words | `func_words:the`, `func_words:and`, `func_words:but` |
| **Punctuation** | Punctuation mark usage | `punctuation:!`, `punctuation:?`, `punctuation:,` |
| **Emojis** | Emoji usage patterns | `emojis:😀`, `emojis:🥳`, `emojis:❤️` |
| **Letters** | Individual letter frequency | `letters:a`, `letters:e`, `letters:t` |
| **Tokens** | Total token count (normalized) | `tokens:count` |

### Advanced Linguistic Features

| Feature Type | Description | Example Features |
|--------------|-------------|------------------|
| **Named Entities** | Person, location, organization mentions | `named_entities:PERSON`, `named_entities:LOC`, `named_entities:ORG` |
| **Specialized Verbs** | Suasive and stative verb usage | `suasive_verbs:suggest`, `stative_verbs:know` |
| **Pronoun Patterns** | Personal pronoun usage | `first_second_person_pronouns`, `third_person_pronouns` |
| **Verb Forms** | Specific verb tense/aspect forms | `token_VB`, `token_VBD`, `token_VBG` |

## 📚 API Reference

### Gram2VecVectorizer

```python
class Gram2VecVectorizer(language="en", normalize=True)
```

**Parameters:**
- `language` (str): Language code (currently supports "en")
- `normalize` (bool): Whether to normalize feature counts by document length

**Methods:**

#### `vectorize_document(text: str) -> Dict[str, float]`
Vectorize a single document.

**Parameters:**
- `text` (str): Input text to vectorize

**Returns:**
- `Dict[str, float]`: Feature vector as dictionary

#### `vectorize_documents(texts: List[str]) -> pd.DataFrame`
Vectorize multiple documents.

**Parameters:**
- `texts` (List[str]): List of input texts

**Returns:**
- `pd.DataFrame`: DataFrame with documents as rows and features as columns

#### `from_documents(documents, author_ids=None, document_ids=None) -> pd.DataFrame`
Vectorize documents with optional metadata.

**Parameters:**
- `documents` (List[str]): List of input texts
- `author_ids` (List[str], optional): Author identifiers
- `document_ids` (List[str], optional): Document identifiers

**Returns:**
- `pd.DataFrame`: DataFrame with documents and metadata

### Feature Class

```python
class Feature(name: str, counter: Counter, vocab: Optional[List[str]] = None)
```

**Methods:**

#### `get_vector() -> Dict[str, float]`
Convert counter to normalized vector with zero counts for missing vocab items.

**Returns:**
- `Dict[str, float]`: Normalized feature vector

## 🎨 Advanced Usage

### Custom Feature Configuration

```python
# Modify which features are extracted
from src.vectorizer import default_config

# Enable only specific features
custom_config = {
    "pos_unigrams": 1,
    "dep_labels": 1,
    "emojis": 1,
    "pos_bigrams": 0,  # Disable POS bigrams
    "morph_tags": 0,   # Disable morphological tags
}

# Use custom configuration
vectorizer = Gram2VecVectorizer("en", normalize=True)
# Note: Configuration is currently handled internally, but can be extended
```

### Working with Large Document Collections

```python
# Process documents in batches
def process_large_collection(texts, batch_size=1000):
    vectorizer = Gram2VecVectorizer("en", normalize=True)
    all_vectors = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_vectors = vectorizer.vectorize_documents(batch)
        all_vectors.append(batch_vectors)
    
    return pd.concat(all_vectors, ignore_index=True)
```

### Feature Analysis

```python
# Analyze feature importance
def analyze_features(vectors_df):
    # Calculate feature variance (higher variance = more discriminative)
    feature_variance = vectors_df.var().sort_values(ascending=False)
    
    # Top 10 most variable features
    print("Top 10 most variable features:")
    print(feature_variance.head(10))
    
    # Feature correlations
    correlations = vectors_df.corr()
    
    return feature_variance, correlations
```

## 📁 Vocabulary Files

The package uses vocabulary files in the `vocab/` directory for feature extraction:

| File | Description | Example Content |
|------|-------------|-----------------|
| `pos_unigrams.txt` | Universal Dependencies POS tags | `ADJ`, `NOUN`, `VERB`, `ADV` |
| `pos_bigrams.txt` | All POS tag combinations | `VERB NOUN`, `ADJ NOUN`, `NOUN VERB` |
| `dep_labels.txt` | Dependency labels | `nsubj`, `dobj`, `prep`, `det` |
| `morph_tags.txt` | Morphological tags | `Tense=Pres`, `Number=Sing`, `Case=Nom` |
| `sentences.txt` | Sentence types | `declarative`, `interrogative`, `imperative` |
| `emojis.txt` | Common emojis | `😀`, `🥳`, `❤️`, `👍` |
| `func_words.txt` | Function words | `the`, `and`, `but`, `in`, `on` |
| `punctuation.txt` | Punctuation marks | `!`, `?`, `,`, `.`, `;` |
| `letters.txt` | Alphabet letters | `a`, `b`, `c`, ..., `A`, `B`, `C` |

### Adding Custom Vocabulary

You can add custom vocabulary files by placing them in the `vocab/` directory with the same naming convention. The vectorizer will automatically load all `.txt` files from this directory.

## 🔍 Example Output

### Feature Vector Sample

```python
# Sample feature vector (first 15 features)
{
    'pos_unigrams:VERB': 0.12,
    'pos_unigrams:NOUN': 0.18,
    'pos_unigrams:ADJ': 0.08,
    'pos_bigrams:VERB NOUN': 0.05,
    'pos_bigrams:ADJ NOUN': 0.03,
    'dep_labels:nsubj': 0.06,
    'dep_labels:dobj': 0.04,
    'morph_tags:Tense=Pres': 0.08,
    'morph_tags:Number=Sing': 0.15,
    'sentences:declarative': 0.75,
    'sentences:interrogative': 0.25,
    'emojis:🥳': 0.02,
    'func_words:the': 0.07,
    'punctuation:!': 0.03,
    'letters:a': 0.08
}
```

### DataFrame Output

```python
# Output DataFrame structure
print(vectors.head())
# Output:
#    pos_unigrams:VERB  pos_unigrams:NOUN  pos_unigrams:ADJ  ...  letters:z
# 0              0.12               0.18              0.08  ...       0.00
# 1              0.15               0.22              0.10  ...       0.01
# 2              0.08               0.14              0.06  ...       0.00
```

## ⚡ Performance

This implementation is optimized for speed and memory efficiency:

- **Efficient NLP Pipeline**: Uses spaCy's optimized processing
- **Vectorized Operations**: Fast feature extraction using pandas/numpy
- **Memory Management**: Optimized vocabulary loading and feature storage
- **Batch Processing**: Support for processing large document collections

### Performance Benchmarks

```python
import time

# Benchmark vectorization speed
def benchmark_speed(texts):
    vectorizer = Gram2VecVectorizer("en", normalize=True)
    
    start_time = time.time()
    vectors = vectorizer.vectorize_documents(texts)
    end_time = time.time()
    
    docs_per_second = len(texts) / (end_time - start_time)
    print(f"Processed {len(texts)} documents in {end_time - start_time:.2f} seconds")
    print(f"Speed: {docs_per_second:.2f} documents/second")
    
    return vectors
```

## 🔧 Troubleshooting

### Common Issues

1. **Import Error**: Make sure you're running from the correct directory and have activated the right environment.

2. **spaCy Model Not Found**: Download the required model:
   ```bash
   python -m spacy download en_core_web_lg
   ```

3. **Memory Issues**: For large document collections, process in batches:
   ```python
   # Process in smaller batches
   batch_size = 100
   for i in range(0, len(documents), batch_size):
       batch = documents[i:i + batch_size]
       vectors = vectorizer.vectorize_documents(batch)
   ```

4. **Feature Count Mismatch**: Ensure vocabulary files are properly formatted (one item per line).

### Debug Mode

```python
# Enable debug output
import logging
logging.basicConfig(level=logging.DEBUG)

# Check loaded vocabularies
vectorizer = Gram2VecVectorizer("en")
print("Loaded vocabularies:", list(vectorizer.vocabs.keys()))
```

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd speedy-gram2vec

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/
```

## 📄 License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## 🙏 Acknowledgments

This implementation is based on the original [Gram2Vec](https://github.com/eric-sclafani/gram2vec) by Eric Sclafani. The research is supported in part by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA), via the HIATUS Program contract #2022-22072200005.

## 📚 References

- Sclafani, E. (2023). Gram2Vec: A grammatical style embedding algorithm. GitHub repository.
- spaCy: Industrial-strength Natural Language Processing in Python. https://spacy.io/
- Universal Dependencies: A cross-linguistic typology. https://universaldependencies.org/

## 📞 Support

If you encounter any issues or have questions, please:

1. Check the troubleshooting section above
2. Search existing issues on GitHub
3. Create a new issue with detailed information about your problem

For feature requests or general questions, feel free to open a discussion on GitHub.
