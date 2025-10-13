"""
Creates filtered vocabulary files containing only the top N features.
This allows gram2vec to extract only those specific features, making it more efficient.
"""

import pandas as pd
import argparse
from pathlib import Path
from collections import defaultdict


def parse_feature_name(feature: str):
    """
    Parse a feature name into (type, value).
    
    Examples:
        "punctuation:." -> ("punctuation", ".")
        "pos_bigrams:ADJ ADJ" -> ("pos_bigrams", "ADJ ADJ")
        "types:count" -> ("types", "count")
    """
    if ":" not in feature:
        return None, None
    parts = feature.split(":", 1)
    return parts[0], parts[1]


def get_top_features_by_type(coef_file: str, n_features: int = 10):
    """
    Load coefficients and group top N features by their type.
    
    Returns:
        dict: {feature_type: [list of feature values]}
    """
    coef_df = pd.read_csv(coef_file)
    coef_df['abs_coef'] = coef_df['human'].abs()
    
    # Get top N features
    top_features = coef_df.nlargest(n_features, 'abs_coef')
    
    print(f"\n=== Top {n_features} Features ===")
    
    # Group by feature type
    features_by_type = defaultdict(list)
    for idx, row in top_features.iterrows():
        feature_name = row['Feature']
        coef = row['human']
        
        feat_type, feat_value = parse_feature_name(feature_name)
        if feat_type and feat_value:
            features_by_type[feat_type].append(feat_value)
            print(f"  {feature_name:40s} | coef: {coef:7.4f}")
    
    print(f"\n=== Features grouped by type ===")
    for feat_type, values in features_by_type.items():
        print(f"  {feat_type}: {len(values)} features")
        for val in values:
            print(f"    - {val}")
    
    return dict(features_by_type)


def create_filtered_vocab_files(features_by_type: dict, vocab_dir: Path, output_dir: Path):
    """
    Create filtered vocab files containing only the specified features.
    
    Args:
        features_by_type: Dict of {feature_type: [feature_values]}
        vocab_dir: Path to original vocab directory
        output_dir: Path to output directory for filtered vocab files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Mapping from feature type to vocab filename
    vocab_files = {
        'punctuation': 'punctuation.txt',
        'pos_unigrams': 'pos_unigrams.txt',
        'pos_bigrams': 'pos_bigrams.txt',
        'func_words': 'func_words.txt',
        'dep_labels': 'dep_labels.txt',
        'morph_tags': 'morph_tags.txt',
        'emojis': 'emojis.txt',
        'letters': 'letters.txt',
    }
    
    print(f"\n=== Creating filtered vocab files in {output_dir} ===")
    
    for feat_type, feature_values in features_by_type.items():
        if feat_type in vocab_files:
            vocab_file = vocab_files[feat_type]
            output_file = output_dir / vocab_file
            
            # Write the filtered vocab file
            with open(output_file, 'w', encoding='utf-8') as f:
                for value in feature_values:
                    f.write(f"{value}\n")
            
            print(f"  Created {vocab_file} with {len(feature_values)} entries")
        else:
            print(f"  Note: {feat_type} doesn't have a vocab file (might be a computed feature)")
    
    # Copy over any vocab files not being filtered
    # This ensures gram2vec still works if other features are enabled
    print(f"\n=== Copying unmodified vocab files ===")
    for vocab_file in vocab_dir.glob("*.txt"):
        if vocab_file.name not in [f for f in vocab_files.values()]:
            output_file = output_dir / vocab_file.name
            output_file.write_text(vocab_file.read_text(encoding='utf-8'))
            print(f"  Copied {vocab_file.name}")


def create_feature_config(features_by_type: dict):
    """
    Create a config dict that enables only the feature types present in top features.
    """
    all_feature_types = [
        "pos_unigrams", "pos_bigrams", "dep_labels", "morph_tags",
        "sentences", "emojis", "func_words", "punctuation", "letters",
        "transitions", "unique_transitions", "tokens", "num_tokens",
        "types", "sentence_count"
    ]
    
    config = {feat_type: 1 if feat_type in features_by_type else 0 
              for feat_type in all_feature_types}
    
    print(f"\n=== Recommended Feature Config ===")
    print("config = {")
    for feat_type, enabled in config.items():
        print(f"    '{feat_type}': {enabled},")
    print("}")
    
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create filtered vocabulary files for top N features'
    )
    parser.add_argument('--coef-file', '-c', required=True,
                        help='Path to coefficients.csv from a previous training run')
    parser.add_argument('--top-n', '-n', type=int, default=10,
                        help='Number of top features to use (default: 10)')
    parser.add_argument('--vocab-dir', '-v', 
                        default='../../src/gram2vec/vocab',
                        help='Path to gram2vec vocab directory')
    parser.add_argument('--output-dir', '-o',
                        default='vocab_filtered',
                        help='Output directory for filtered vocab files')
    args = parser.parse_args()
    
    vocab_dir = Path(args.vocab_dir)
    output_dir = Path(args.output_dir)
    
    if not vocab_dir.exists():
        print(f"Error: Vocab directory not found: {vocab_dir}")
        exit(1)
    
    # Get top features grouped by type
    features_by_type = get_top_features_by_type(args.coef_file, args.top_n)
    
    # Create filtered vocab files
    create_filtered_vocab_files(features_by_type, vocab_dir, output_dir)
    
    # Show recommended config
    config = create_feature_config(features_by_type)
    
    print(f"\n=== Next Steps ===")
    print(f"1. Use the filtered vocab files by setting GRAM2VEC_VOCAB_DIR environment variable:")
    print(f"   export GRAM2VEC_VOCAB_DIR={output_dir.absolute()}")
    print(f"2. Use the recommended config when training")
    print(f"3. Train your model as usual")

