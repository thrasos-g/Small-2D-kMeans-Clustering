# Import required libraries
import sys
import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

def get_data(fname):
    """
    Read and parse data from a file containing x,y coordinate pairs.
    Handles various formats (spaces, tabs, commas) and removes duplicates.
    
    Args:
        fname: Path to the input file
    
    Returns:
        numpy array of [x, y] coordinates
    """
    print("Reading file:", fname)
    try:
        f=open(fname, 'r')
        lines=f.readlines()
        f.close()
        print("Lines in file:", len(lines))
        points=[]
        
        # Parse each line in the file
        for line in lines:
            line=line.strip()
            # Handle both spaces and tabs as delimiters
            parts=line.replace('\t', ' ').split(' ')
            
            for p in parts:
                if ',' in p:
                        # Extract x,y values from comma-separated format
                        val=p.split(',')
                        
                        if len(val)==2:
                            # Clean up whitespace and quotes
                            x_str=val[0].strip().strip('"').strip("'")
                            y_str=val[1].strip().strip('"').strip("'")    
                            try:
                                x=float(x_str)
                                y=float(y_str)
                                points.append([x,y])
                            except ValueError:
                                # Skip invalid coordinate values
                                continue

        # Create DataFrame and remove duplicate points
        df=pd.DataFrame(points, columns=['x', 'y'])
        oldLength=len(df)
        df=df.drop_duplicates()
        if len(df) < oldLength:
            print("Dropped duplicates:", oldLength-len(df))
            
        return df.values
    except Exception as e:
        print("Couldnt read file", {e})
        sys.exit(1)


# Validate command line arguments
if len(sys.argv)!=2:
    print("Usage: python script.py <filename>")
    sys.exit(1)

# Load the data file
filename=sys.argv[1]
start=time.time()
data=get_data(filename)  # Parse coordinates from file

# Check if data was successfully loaded
if len(data)==0:
    print("No data found")
    sys.exit(1)

# Keep original data for output formatting
ogData=data.copy()

# Normalize the data for KMeans clustering (standardize to mean=0, std=1)
scaler=StandardScaler()
scaled_data=scaler.fit_transform(data)

# Configure and run KMeans clustering algorithm
k=5  # Number of clusters
model=KMeans(n_clusters=k, random_state=42)
labels=model.fit_predict(scaled_data)  # Assign points to clusters
centers=model.cluster_centers_  # Get cluster center coordinates
print(f"Running KMeans with k={k}")

dists=[]

# Calculate distance from each point to its assigned cluster center
for i in range(len(scaled_data)):
    p=scaled_data[i]  # Current point
    c=centers[labels[i]]  # Center of the cluster this point belongs to
    d=np.linalg.norm(p-c)  # Euclidean distance to cluster center
    dists.append(d)

dists = np.array(dists)

# Identify outliers using 95th percentile threshold
# Points beyond this distance are considered anomalies
limit=np.percentile(dists,95)
outlier_mask=dists > limit
outlier_idx=np.where(outlier_mask)[0]
print(f"Threshold is: {limit:.3f}")

# Display all detected outliers
print("Outliers found:")
count=1
for i in outlier_idx:
    pt = ogData[i]  # Get original (non-scaled) coordinates
    print(f"#{count} Index: {i} X: {pt[0]:.3f}, Y: {pt[1]:.3f}")
    count+=1
print("")
print("")

# Calculate and display summary statistics
totalPoints=len(data)
numberOfOutliers=len(outlier_idx)
validPoints=totalPoints - numberOfOutliers
maxDist=np.max(dists)
totalTime = time.time() - start

print(f"Total points loaded {totalPoints}")
print(f"Number of valid points {validPoints}")
print(f"Number of outliers detected {numberOfOutliers}")
print(f"Maximum distance {maxDist:.3f}")
print(f"Total time is {totalTime:.3f} seconds")