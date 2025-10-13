# Logging Guide

Both training scripts now automatically save all terminal output to descriptive log files. This makes it easy to review results, compare runs, and keep a record of your experiments.

## Features

- **Automatic logging**: Output goes to both terminal AND a log file
- **Descriptive filenames**: Log names include script, parameters, and timestamp
- **Easy review**: List and view logs without hunting through files
- **No interference**: Logs don't slow down execution

## Basic Usage

### Logging is Enabled by Default

```bash
# Just run normally - logging happens automatically!
python parallel_log_regression.py
```

Output example:
```
================================================================================
Logging to: logs/full_training_20241013_143052_features_gram2vec_output_coefficients.log
Started at: 2024-10-13 14:30:52
================================================================================

=== Using gram2vec features ===
...
(all output shown in terminal AND saved to log file)
...
================================================================================
Completed at: 2024-10-13 14:45:23
Log saved to: logs/full_training_20241013_143052_features_gram2vec_output_coefficients.log
================================================================================
```

### With Top Features Script

```bash
# Logs include the top-n parameter in the filename
python train_with_top_features.py --coef-file coefficients.csv --top-n 10
```

Creates log: `logs/train_top_features_20241013_143052_top_n_10_coef_coefficients.log`

## Log File Naming

Log filenames are descriptive and include:

### parallel_log_regression.py
Format: `full_training_{timestamp}_features_{feature_type}_output_{output_basename}.log`

Example: `full_training_20241013_143052_features_gram2vec_output_coefficients.log`

### train_with_top_features.py  
Format: `train_top_features_{timestamp}_top_n_{N}_coef_{coef_basename}.log`

Example: `train_top_features_20241013_150203_top_n_10_coef_coefficients.log`

## Log Management

### List Recent Logs

```bash
# List last 10 logs
python parallel_log_regression.py --list-logs

# Or
python train_with_top_features.py -c coefficients.csv --list-logs
```

Output example:
```
================================================================================
Recent logs in logs/ (showing up to 10):
================================================================================
 1. train_top_features_20241013_150203_top_n_10_coef_coefficients.log
    Size: 2.34 MB | Modified: 2024-10-13 15:05:23
 2. full_training_20241013_143052_features_gram2vec_output_coefficients.log
    Size: 5.67 MB | Modified: 2024-10-13 14:45:23
 3. train_top_features_20241013_120156_top_n_50_coef_coefficients.log
    Size: 2.89 MB | Modified: 2024-10-13 12:15:32
```

### View a Log File

```bash
# View full log
cat logs/train_top_features_20241013_150203_top_n_10_coef_coefficients.log

# View last 50 lines (to see results)
tail -50 logs/train_top_features_20241013_150203_top_n_10_coef_coefficients.log

# Search for specific metrics
grep "accuracy" logs/*.log
```

### Clean Up Old Logs

```bash
# Remove all logs
rm -rf logs/

# Remove logs older than 7 days
find logs/ -name "*.log" -mtime +7 -delete

# Remove specific pattern
rm logs/train_top_features_*_top_n_5_*.log
```

### Custom Log Directory

```bash
# Use a different directory
python parallel_log_regression.py --log-dir my_experiments

# This creates: my_experiments/full_training_*.log
```

### Disable Logging

```bash
# Run without saving to log file
python parallel_log_regression.py --no-log
```

## Example Workflows

### Experiment Tracking

```bash
# Run multiple experiments - each gets its own log
python train_with_top_features.py -c coefficients.csv --top-n 10
python train_with_top_features.py -c coefficients.csv --top-n 20
python train_with_top_features.py -c coefficients.csv --top-n 50

# Review results
python train_with_top_features.py -c coefficients.csv --list-logs

# Compare accuracy across experiments
grep -A 3 "Test Set" logs/train_top_features_*_top_n_*.log
```

### Archive Successful Runs

```bash
# Create archive directory
mkdir -p archives/experiment_2024_10_13

# Move logs from successful runs
mv logs/full_training_20241013_*.log archives/experiment_2024_10_13/

# Add notes
echo "Testing different feature counts" > archives/experiment_2024_10_13/README.txt
```

### Share Results

```bash
# Logs contain full reproducible output
# Just share the log file!
scp logs/train_top_features_*.log colleague@server:/path/

# Or create a report from logs
tail -100 logs/train_top_features_*_top_n_*.log > experiment_summary.txt
```

## What's Logged

### Full Training Output

- Data loading progress
- Feature extraction status  
- Cache hit/miss information
- Training progress
- Classification metrics (precision, recall, F1)
- Feature coefficients
- File save confirmations
- Timing information

### Example Log Contents

```
================================================================================
Logging to: logs/train_top_features_20241013_150203_top_n_10_coef_coefficients.log
Started at: 2024-10-13 15:02:03
================================================================================

=== Top 10 Features ===
  types:count                              | coef: -1.2349
  num_tokens:num_tokens                    | coef: -0.9908
  ...

=== Loading data ===
Loading corpus files: 100%|██████████| 5/5 [00:02<00:00,  2.1it/s]
Concatenating dataframes...
Train: 1200, Dev: 400, Test: 400

=== Extracting all gram2vec features ===
Cache enabled - config hash: a1b2c3d4e5f6
Cache directory: vector_cache

=== Train data ===
  ✓ Loading cached vectors from vectors_train_a1b2c3d4e5f6.parquet

=== Filtering to top 10 features ===
Filtered from 788 to 12 columns

Training with 10 features (reduced from 786)
Feature names: ['types:count', 'num_tokens:num_tokens', ...]

Scaling features...
Training logistic regression model...

=== Evaluation ===

Test Set Classification Report:
              precision    recall  f1-score   support

         gpt       0.95      0.92      0.93       200
       human       0.92      0.95      0.94       200

    accuracy                           0.94       400
   macro avg       0.94      0.94      0.94       400
weighted avg       0.94      0.94      0.94       400

...

Coefficients saved to coefficients_top10.csv

================================================================================
Completed at: 2024-10-13 15:05:23
Log saved to: logs/train_top_features_20241013_150203_top_n_10_coef_coefficients.log
================================================================================
```

## Tips

### 1. Timestamped Logs Prevent Overwrites
Each run creates a new log file with a unique timestamp - no accidental overwrites!

### 2. Grep for Quick Comparisons
```bash
# Compare F1 scores across all runs
grep "f1-score   support" logs/*.log

# Find the best performing run
grep "accuracy" logs/*.log | sort
```

### 3. Log Directory Organization
```
logs/
├── 2024_10_13/           # Organize by date
│   ├── experiment_1/
│   └── experiment_2/
└── archived/             # Old experiments
```

### 4. Capture Errors Too
Logs capture everything including errors, making debugging easier:
```bash
# Run and don't worry about errors disappearing
python train_with_top_features.py -c coefficients.csv --top-n 100

# Review any errors later
grep -i "error\|exception\|traceback" logs/*.log
```

### 5. Keep Logs Small
```bash
# Disable cache verbose output if logs get too large
# (Future enhancement)

# Or periodically clean up
find logs/ -name "*.log" -size +100M -delete  # Remove logs > 100MB
```

## Troubleshooting

### "Permission denied" Error
```bash
# Make sure log directory is writable
chmod +w logs/

# Or use a different directory
python parallel_log_regression.py --log-dir ~/my_logs
```

### Logs Not Being Created
```bash
# Check if --no-log flag is set
# Remove it to enable logging

# Verify log directory exists and is writable
ls -ld logs/
```

### Can't Find Recent Log
```bash
# List all logs by time
ls -lt logs/*.log | head -10

# Or use the built-in command
python parallel_log_regression.py --list-logs
```

## Command Reference

### parallel_log_regression.py
```bash
--log-dir DIR      # Directory for logs (default: logs)
--no-log           # Disable logging
--list-logs        # Show recent logs
```

### train_with_top_features.py
```bash
--log-dir DIR      # Directory for logs (default: logs)
--no-log           # Disable logging
--list-logs        # Show recent logs
```

## Summary

- ✅ **Automatic**: Logging happens by default
- ✅ **Descriptive**: Filenames include parameters and timestamps
- ✅ **Non-intrusive**: Doesn't slow down execution
- ✅ **Complete**: Captures everything you see in terminal
- ✅ **Organized**: Easy to review and compare experiments
- ✅ **Shareable**: Send log files to colleagues for full reproducibility

Happy experimenting! 🎉

