# Biber Genre Vectorizer

A Python tool for analyzing and vectorizing text documents based on Biber's linguistic features. This tool extracts various linguistic features from text documents and represents them as numerical vectors, which can be used for genre analysis, text classification, and other NLP tasks.

## Features

- Extract over 30 linguistic features from text documents
- Process individual texts or entire directories
- Support for English language (with potential for other languages)
- Features include:
  - Part-of-speech tags (nouns, verbs, adjectives, etc.)
  - Verb tenses (VB, VBD, VBG, VBN, VBP, VBZ)
  - Pronouns (first/second person, third person, "it")
  - Named entities (person, location, organization, date)
  - Punctuation patterns
  - Sentence and token metrics
  - Superlatives and comparatives
  - And more!

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/biber-genre-features.git
cd biber-genre-features

# Create and activate a conda environment
conda create -n gram2vec python=3.8
conda activate gram2vec

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_lg
```

## Usage

```python
from src.vectorizer import BiberGenreVectorizer

# Initialize the vectorizer
vectorizer = BiberGenreVectorizer("en")

# Process a list of texts
texts = [
    "This is a sample text. It contains various linguistic features.",
    "Another example with different features and structures."
]
features_df = vectorizer.process_texts(texts)

# Save the features to a CSV file
features_df.to_csv("vectorized_texts.csv", index=False)

# Process all text files in a directory
from pathlib import Path
directory_features = vectorizer.vectorize_directory(Path("path/to/text/files"))
```

## Requirements

- Python 3.8+
- pandas
- numpy
- spaCy
- en_core_web_lg (spaCy model)

## License

[MIT License](LICENSE) 