
# Outlier Detection using K-Means

A Python tool for detecting outliers in 2D point data using K-Means clustering.

## Features

- Loads and cleans corrupted or malformed text files
- Normalizes data using StandardScaler
- Applies K-Means clustering (default: 5 clusters)
- Detects outliers based on distance from cluster centers
- Configurable outlier threshold (percentile-based)

## Usage

```bash
python script.py <filename>
```

Example:
```bash
python script.py data.txt
```

## Input Format

Text file with 2D points in comma-separated format:
```
x1,y1
x2,y2
x3,y3
```

Supports various formats (spaces, tabs, multiple pairs per line).

## Output

- Detected outliers with indices and coordinates
- Distance statistics
- Outlier percentage
- Execution time

## Parameters

- `n_clusters`: Number of K-Means clusters (default: 5)
- `thresholdPercentile`: Outlier threshold percentile (default: 95)

## Dependencies

```
numpy
pandas
scikit-learn
```

Install with:
```bash
pip install numpy pandas scikit-learn
```
