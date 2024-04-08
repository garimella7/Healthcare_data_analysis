
# Business Understanding - Segmentation of the patients with similiar medical conditions

# Business Objective - categorizing the best possible insurance for a patient at suitable rate

# Business Constraint - listing the high payable insurance policy


# import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn import metrics
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from clusteval import clusteval
from sqlalchemy import create_engine
import joblib
import pickle

#pip install sqlalchemy==1.4.16

# Read the csv file and store it in a dataframe
patientData = pd.read_csv(r'C:\RAVI\Interview Preparation\Personal Projects\Healthcare_data_analysis\Ha_dataset.csv'
                 ,header=0, delimiter=';')

############################### Connect to SQL ################################
# Specify the database name and server name
db = 'DSProjects'
server_name = 'DESKTOP-ACLNM4U\\SQLEXPRESS'

# Specify the ODBC driver name
driver_name = 'ODBC+Driver+17+for+SQL+Server'

# Construct the connection string
conn_string = f"mssql+pyodbc://{server_name}/{db}?trusted_connection=yes&driver={driver_name}"

# Create the engine
engine = create_engine(conn_string)

# Test the connection
try:
    with engine.connect() as connection:
        print("Connected successfully!")
except Exception as e:
    print("Connection failed:", e)
    
patientData.to_sql('PatientsData', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

sql = 'select * from PatientsData;'
df = pd.read_sql_query(sql, engine)

###############################################################################  

df.info()

# First moment decision
df.Age.mean()                 # 50.24
df['Length of Stay'].mean()   # 10.8
df['Amount Billing'].mean()   # 3758.28

df.Gender.mode()                # Female
df['Medical Condition'].mode()  # Asthma
df['Test Result'].mode()        # Normal
df.Medication.mode()            # Ibuprofen
df['Insurance Provider'].mode() # LiveWell
df['Admission Type'].mode()     # Urgent
df['Admission Date'].mode()     
df['Discharge Date'].mode()     # 05/02/22

# As we can see, most of the patients admitted are females and suffering from Asthma.
# Most given medication is Ibuprofen and mostly insurance provider is LiveWell

# Second moment business decision
df.Age.std()                  # 19.81
df['Length of Stay'].std()    # 7.81
df['Amount Billing'].std()    # 1434.35

# A higher variance indicates that the values in the feature are more spread out from the mean, 
# while a lower variance indicates that the values are closer to the mean. 
df.Age.var()                  # 392.44
df['Length of Stay'].var()    # 61.11
df['Amount Billing'].var()    # 2057367.12

# Third moment business decision
# Skewness can provide insights into the shape of the distribution of data.
# +ve skewness indicates that the distribution is skewed to the right
# -ve skewness indicates that the distribution is skewed to the left 
df.Age.skew()                 # 0.058
df['Length of Stay'].skew()   # -0.012
df['Amount Billing'].skew()   # 0.233

# Kurtosis can provide insights into the shape of the distribution of data, particularly 
# regarding the presence of outliers or the heaviness of the tails compared to a normal distribution. 
# A kurtosis value greater than 0 indicates heavier tails than a normal distribution (leptokurtic), 
# while a kurtosis value less than 0 indicates lighter tails than a normal distribution (platykurtic).
df.Age.kurt()                 # -1.174
df['Length of Stay'].kurt()   # -1.36
df['Amount Billing'].kurt()   # -0.456

# Box Plots
plt.boxplot(df.Age)                 # No outliers
plt.boxplot(df['Length of Stay'])   # No outliers
plt.boxplot(df['Amount Billing'])   # No outliers

# Duplicate values - No duplicates found
dup = df.duplicated()
sum(dup)

# Missing values - No missing values
df.isna().sum()

df.corr()
# Age and Length of stay has strong correlation of 0.966. Hence, one of them can be removed.


# Auto EDA
import dtale
d = dtale.show(df)
d.open_browser()

# Dates are never useful for Clustering analysis, hence it can be removed
# Removing unnecessary columns and creating a copy of the original dataframe
df1 = df.drop(['ID', 'Length of Stay', 'Admission Date', 'Discharge Date'], axis = 1)

# Rearranging the columns
df1 = df1[['Age', 'Gender', 'Medical Condition', 'Test Result', 'Medication', 'Insurance Provider',
         'Admission Type', 'Amount Billing']]

###############################################
# Univariate analysis
# Set the color palette
sns.set_palette("cool")

# Age distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df1, x='Age', bins=20, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Gender distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df1, x='Gender')
plt.title('Distribution of Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Medical Condition distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=df1, y='Medical Condition', order=df['Medical Condition'].value_counts().index)
plt.title('Distribution of Medical Condition')
plt.xlabel('Count')
plt.ylabel('Medical Condition')
plt.show()

# Test Result distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df1, x='Test Result')
plt.title('Distribution of Test Results')
plt.xlabel('Test Result')
plt.ylabel('Count')
plt.show()


################################### PIPELINE ##################################
df1.nunique()

#### separating numeric data and categorical data for column names
num = df1.select_dtypes(exclude = ['object']).columns
cat = df1.select_dtypes(include = ['object']).columns

### separating numeric data
num1 = df1.select_dtypes(exclude = ['object'])

num_pipeline = Pipeline([('scale', MinMaxScaler())])
num_pipeline

# Fit the numeric data to the pipeline.
processed = num_pipeline.fit(df1[num]) 

# Save the pipeline
joblib.dump(processed, 'processed1')

# Transform the data with pipeline on numberic columns to get clean data
num_clean = pd.DataFrame(processed.transform(df1[num]), columns = num)
num_clean

###############################################################################

d = dtale.show(num_clean)
d.open_browser()

############################# Agglomerative Clustering ########################

plt.figure(1, figsize = (16, 8))
tree_plot = dendrogram(linkage(num_clean, method = 'ward'), get_leaves = True, show_leaf_counts = True)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel('Index')
plt.ylabel('Euclidean distance')
plt.show()

# For 2 clusters
# Linkage 'ward' gives better result than complete and single
hc = AgglomerativeClustering(n_clusters = 2, metric = 'euclidean', linkage = 'ward')
hc_p = hc.fit_predict(num_clean)
hc_labels = pd.Series(hc.labels_)
metrics.silhouette_score(num_clean, hc_labels) # 0.518

# Silhouette cluster evaluation. 
ce = clusteval(evaluate = 'silhouette')
df_array = np.array(num_clean)
# Fit
ce.fit(df_array)
# Plot
ce.plot()

'''
# For 3 clusters
# Linkage 'ward' gives better result than complete and single
hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
hc_p = hc.fit_predict(num_clean)
hc_labels = pd.Series(hc.labels_)
metrics.silhouette_score(num_clean, hc_labels) # 0.0456'''

# As we can infer, taking 2 clusters gives us the best silhouette score

###############################################################################


################################## K-Means Clustering #########################

# Scree plot or elbow curve
TWSS = []
k = list(range(2,9))

for i in k:
    km = KMeans(n_clusters = i)
    km.fit(num_clean)
    TWSS.append(km.inertia_)
    
# Creating a scree plot to find out no.of cluster
plt.plot(k, TWSS, 'ro-'); plt.xlabel("No_of_Clusters"); plt.ylabel("total_within_SS")

# Using Knee Locator
from kneed import KneeLocator
kl = KneeLocator(range(2, 9), TWSS, curve = 'convex')
# kl = KneeLocator(range(2, 9), List, curve='convex', direction = 'decreasing')
kl.elbow
plt.style.use("seaborn")
plt.plot(range(2, 9), TWSS)
plt.xticks(range(2, 9))
plt.ylabel("Inertia")
plt.axvline(x = kl.elbow, color = 'r', label = 'axvline - full height', ls = '--')
plt.show() 

km = KMeans(n_clusters = 2)
km_p = km.fit_predict(num_clean)
km_labels = pd.Series(km.labels_)
metrics.silhouette_score(num_clean, km_labels) # 0.522


# Save the model
pickle.dump(km, open('Clust_PatientsData.pkl', 'wb'))

import os
os.getcwd()

# Plotting the clusters
plt.scatter(num_clean.iloc[:, 0], num_clean.iloc[:, 1], c=km_p, cmap='viridis', s=50, alpha=0.7, edgecolors='k')

# Plotting the centroids of the clusters
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids')

plt.title('KMeans Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# As we can infer, taking 2 clusters gives us the best silhouette score

###############################################################################

# We can infer from above analysis, KMeans gives us better clustering with 2 clusters

# Add the cluster labels to the original dataframe
patientData = pd.concat([patientData, km_labels], axis = 1)
patientData = patientData.rename(columns = {0 : 'cluster_id'})

# Update the SQL table with clustered data
patientData.to_sql('PatientsData', con = engine, if_exists = 'replace', chunksize = 1000, index = 'False')

# Save the clustered data into a CSV file
patientData.to_csv('PatientsData_Clust.csv', encoding = 'utf-8', index = False)
import os
os.getcwd()


"""
 conclusion:
     
The K-means algorithm defines a cost function that computes Euclidean distance (or it can be anything similar) between two numeric values. However, it is not possible to define such distance between categorical values. 
for e.g. if Euclidean distance between numeric points A and B is 25 and A and C is 10, we know A is closer to C than B. categorical values are not numbers but are enumerations such as 'banana', 'apple' and 'oranges'. 
Euclidean distance cannot be used to compute euclidean distances between the above fruits. We cannot say apple is closer to orange or banana because Euclidean distance is not meant to handle such information. Therefore, we need to change the cost function. 

Use Hamming distance instead of Euclidean distance , i.e. if we two categorical values are same then make the distance 0 or else 1.
Instead of mean, compute mode i.e the most occurring categorical value of a feature is used as its representative. That's how you compute the centers of a cluster.
Here you go, you have defined a new cost function that can perform partitional clustering of categorical data and it is called K-modes clustering. The basic steps of K-modes algorithm are the same, except for the cost function it optimizes.
 
 """





