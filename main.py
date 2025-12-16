import sys
import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def loadCleanedData(filename):
    """
    Load and clean data from a potentially corrupted text file.
    Parses line-by-line to extract valid x,y pairs.
    """
    print(f"Loading data from: {filename}")
    
    valid_points = []
    
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        print(f"Total lines found in file: {len(lines)}")
        
        for line in lines:
            # Αφαιρούμε κενά από την αρχή και το τέλος
            line = line.strip()
            if not line:
                continue
            
            # Αν η γραμμή έχει κόμμα, δοκιμάζουμε να τη χωρίσουμε
            # Χειριζόμαστε και περιπτώσεις που μπορεί να υπάρχουν πολλά ζεύγη ή "σκουπίδια"
            # Αντικαθιστούμε τυχόν tabs με κενά και σπάμε τη γραμμή
            parts = line.replace('\t', ' ').split(' ')
            
            for part in parts:
                if ',' in part:
                    try:
                        # Διαχωρισμός με βάση το κόμμα
                        vals = part.split(',')
                        
                        # Πρέπει να υπάρχουν ακριβώς 2 στοιχεία
                        if len(vals) == 2:
                            # Καθαρισμός από τυχόν quotes ή κενά
                            val_x = vals[0].strip().strip('"').strip("'")
                            val_y = vals[1].strip().strip('"').strip("'")
                            
                            # Μετατροπή σε float
                            x = float(val_x)
                            y = float(val_y)
                            
                            # Έλεγχος ότι είναι κανονικοί αριθμοί (όχι inf/nan)
                            if np.isfinite(x) and np.isfinite(y):
                                valid_points.append([x, y])
                    except ValueError:
                        # Αν δεν μπορεί να γίνει μετατροπή σε αριθμό, αγνοούμε το συγκεκριμένο ζεύγος
                        continue

        # Δημιουργία DataFrame από τα καθαρά δεδομένα
        df=pd.DataFrame(valid_points, columns=['x', 'y'])
        
        print(f"Successfully parsed {len(df)} valid records.")
    
        # Cleaning: remove duplicates
        initial_len = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_len:
            print(f"Removed {initial_len - len(df)} duplicate records")
        
        # Convert to numpy array
        data = df.values
        
        print(f"Final number of records to process: {len(data)}")
        print()
        
        if len(data) == 0:
            raise ValueError("No valid data points found in file!")
            
        return data

    except Exception as e:
        print(f"Error reading file: {str(e)}")
        sys.exit(1)

def detectOutliersKMeans(data, n_clusters=5, thresholdPercentile=95):
    """
    Detect outliers using k-means clustering
    
    Method:
    1. Normalize data
    2. Apply k-means with k=5 clusters
    3. Calculate distance of each point from its cluster center
    4. Define outliers as points with distance above threshold
    """
    
    # Store original data (non-normalized)
    originalData = data.copy()
    
    # Normalize data for better k-means performance
    scaler = StandardScaler()
    dataScaled = scaler.fit_transform(data)
    
    # Apply k-means
    print(f"Applying K-Means with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(dataScaled)
    centers = kmeans.cluster_centers_
    
    # Calculate distance of each point from its cluster center
    distances = np.zeros(len(dataScaled))
    for i in range(len(dataScaled)):
        clusterId = labels[i]
        center = centers[clusterId]
        distances[i] = np.linalg.norm(dataScaled[i] - center)
    
    # Define threshold for outliers (using percentile)
    threshold = np.percentile(distances, thresholdPercentile)
    
    # Detect outliers
    outlierMask = distances > threshold
    outlierIndices = np.where(outlierMask)[0]
    
    print(f"Distance threshold (percentile {thresholdPercentile}): {threshold:.3f}")
    print(f"Number of outliers detected: {len(outlierIndices)}")
    print()
    
    return outlierIndices, originalData, labels, distances


def printOutliers(outlierIndices, originalData):
    """
    Print outliers (original non-normalized points)
    """
    print("=" * 60)
    print("OUTLIERS - Original Points:")
    print("=" * 60)
    
    if len(outlierIndices) == 0:
        print("No outliers detected")
    else:
        outliers = originalData[outlierIndices]

    num_outliers = len(outlierIndices)
    for k in range(num_outliers):
        # Παίρνουμε τα δεδομένα χρησιμοποιώντας τη θέση k
        currentIndex = outlierIndices[k]
        currentPoint = outliers[k]
    
        # Εκτύπωση (το k+1 είναι για να δείχνει 1, 2, 3 αντί για 0, 1, 2)
        print(f"Outlier {k+1}: Index={currentIndex}, X={currentPoint[0]:.3f}, Y={currentPoint[1]:.3f}")    


def main():
    # Check arguments
    if len(sys.argv) != 2:
        print("Usage: python script.py <filename.csv>")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    # Start timing
    start_time = time.time()
    
    try:
        # Steps 1 & 2: Load and clean data
        data = loadCleanedData(filename)
        
        # Steps 3 & 4: K-means and outlier detection
        outlierIndices, originalData, labels, distances = detectOutliersKMeans(data, n_clusters=5, thresholdPercentile=95)
        
        # Print results
        printOutliers(outlierIndices, originalData)
        
        # End timing
        end_time = time.time()
        execution_time = end_time - start_time
        
        print()
        print(f"Total execution time: {execution_time:.3f} seconds")
        print()
        
        # Additional statistics
        print("Statistics:")
        print(f"  - Total points: {len(data)}")
        print(f"  - Outliers: {len(outlierIndices)}")
        print(f"  - Outlier percentage: {100*len(outlierIndices)/len(data):.2f}%")
        print(f"  - Mean distance from centers: {np.mean(distances):.3f}")
        print(f"  - Maximum distance: {np.max(distances):.3f}")
        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()