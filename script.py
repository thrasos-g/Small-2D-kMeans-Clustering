import sys
import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def get_data(fname):
    print("Reading file:", fname)
    try:
        f=open(fname, 'r')
        lines=f.readlines()
        f.close()
        print("Lines in file:", len(lines))
        points=[]
        for line in lines:
            line=line.strip()
            parts=line.replace('\t', ' ').split(' ') #handle tabs
            for p in parts:
                if ',' in p:
                        val=p.split(',') #split by comma
                        
                        if len(val)==2:
                            x_str=val[0].strip().strip('"').strip("'") #clean up quotes
                            y_str=val[1].strip().strip('"').strip("'")    
                            try:
                                x=float(x_str)
                                y=float(y_str)
                                points.append([x,y])
                            except ValueError:
                                continue

        df=pd.DataFrame(points, columns=['x', 'y']) #make dataframe
        oldLength=len(df) #remove duplicates
        df=df.drop_duplicates()
        if len(df) < oldLength:
            print("Dropped duplicates:", oldLength-len(df))
            
        return df.values
    except Exception as e:
        print("Couldnt read file", {e})
        sys.exit(1)



if len(sys.argv)!=2:
    print("Usage: python script.py <filename>")
    sys.exit(1)

filename=sys.argv[1]
start=time.time()
data=get_data(filename) #get data
if len(data)==0:
    print("No data found")
    sys.exit(1)
ogData=data.copy() #keep original for printing later
scaler=StandardScaler()
#kmeans setup
scaled_data=scaler.fit_transform(data) # normalize
k=5
model=KMeans(n_clusters=k, random_state=42)
labels=model.fit_predict(scaled_data)
centers=model.cluster_centers_
print(f"Running KMeans with k={k}")
dists=[]



for i in range(len(scaled_data)): #find outliers
    p=scaled_data[i]
    c=centers[labels[i]] #get center for this point
    d=np.linalg.norm(p-c) # calculate euclidean distance
    dists.append(d)
dists = np.array(dists)

# use 95th percentile
limit=np.percentile(dists,95)
outlier_mask=dists > limit
outlier_idx=np.where(outlier_mask)[0]
print(f"Threshold is: {limit:.3f}")

#print
print("Outliers found:")
count=1
for i in outlier_idx:
    pt = ogData[i]
    print(f"#{count} Index: {i} X: {pt[0]:.3f}, Y: {pt[1]:.3f}")
    count+=1
print("")
print("")

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