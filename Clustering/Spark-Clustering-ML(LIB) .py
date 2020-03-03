#!/usr/bin/env python
# coding: utf-8

# # Task :->  Use of Machine Learning Library (MLlib) :- Spark’s machine learning (ML) library for Clustering
# 
# ### Model :-> K-Means
# 
# ### Dataset :-> Auto.csv
# 

# In[1]:


# For Local Machine sparkcontext creation
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()

# In cloud the sparkcontext is created as "sc" 
# ex:- IBM watson 
# python :- 3.6
# spark :- 2.3.3


# In[2]:


sc.version


# In[3]:


# The following code contains the credentials for a file in your IBM Cloud Object Storage.

import ibmos2spark
# @hidden_cell
credentials = {
    'endpoint': 'XXXX',
    'service_id': 'XXXX',
    'iam_service_endpoint': 'XXXX',
    'api_key': 'XXXX'
}

configuration_name = 'os_0a7faf2d576d4d0e985305b42aae4ce7_configs'
cos = ibmos2spark.CloudObjectStorage(sc, credentials, configuration_name, 'bluemix_cos')


# In[5]:



# @hidden_cell
# The following code contains the credentials for a file in your IBM Cloud Object Storage.
# You might want to remove those credentials before you share your notebook.
credentials_1 = {
    'IAM_SERVICE_ID': 'XXXX',
    'IBM_API_KEY_ID': 'XXXX',
    'ENDPOINT': 'XXX',
    'IBM_AUTH_ENDPOINT': 'XXXX',
    'BUCKET': 'XXXX',
    'FILE': 'Auto.csv'
}
data_link = cos.url('Auto.csv', 'sparktry-donotdelete-pr-adbnjs9sf1yuav')


# In[7]:


df_auto = spark.read.csv(data_link, header=True, inferSchema=True)


# In[8]:


df_auto.show()


# In[9]:


df_auto.printSchema()


# In[20]:


# we use only dispacement,horsepower and weight
# sampling and plotting the dataframe
import pandas as pd
import matplotlib.pyplot as plt
x='horsepower'
y='weight'
sampled_data = df_auto.select(x,y).sample(False, 0.8).toPandas()
plt.scatter(sampled_data.horsepower,sampled_data.weight)
plt.xlabel(x)
plt.ylabel(y)
plt.title('relation of horsepower and weight')
plt.show()


# In[22]:


# assembling the columns
from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols = [x,y], outputCol = 'features')
v_df = vectorAssembler.transform(df_auto)
v_df = v_df.select(['features'])
v_df.show(10)


# # K-Means Clustering

# In[23]:


from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


# In[30]:


# creating objects with four centers and fitting
kmeans = KMeans().setK(4).setSeed(1)
model = kmeans.fit(v_df)


# In[31]:


# predicting centers
predictions = model.transform(v_df)


# In[32]:


predictions.show(5)


# In[33]:


# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))


# In[34]:


# Shows the center points
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)


# In[35]:


predictions.select('features').show(5)


# In[36]:


from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType


# In[37]:



split1_udf = udf(lambda value: value[0].item(), FloatType())
split2_udf = udf(lambda value: value[1].item(), FloatType())


# In[38]:


# unpacking features column so that we can plot them 
predictions=predictions.withColumn('x', split1_udf('features')).withColumn('y', split2_udf('features'))


# In[39]:


predictions.show(4)


# In[40]:


# sampling data to plot 
data = predictions.select('x','y','prediction').sample(False, 0.8).toPandas()


# In[41]:


x_cord = [[],[],[],[]]
y_cord = [[],[],[],[]]


# In[42]:


cx = []
cy = []
for center in model.clusterCenters():
    cx.append(center[0])
    cy.append(center[1])


# In[43]:


for i in range(len(data)):
    x_cord[data.loc[i,'prediction']].append(data.loc[i,'x'])
    y_cord[data.loc[i,'prediction']].append(data.loc[i,'y'])


# In[45]:


# plotting the clusters
plt.scatter(x_cord[0],y_cord[0],c='yellow')
plt.scatter(x_cord[1],y_cord[1],c='orange')
plt.scatter(x_cord[2],y_cord[2],c='cyan')
plt.scatter(x_cord[3],y_cord[3],c='magenta')
plt.scatter(cx,cy,c='blue',marker='x')
plt.xlabel('Income')
plt.ylabel('Expenditure')
plt.title('KMeans Result')
plt.show()


# In[ ]:





# # Bisecting k-means

# In[53]:


from pyspark.ml.clustering import BisectingKMeans

x="horsepower"
y="weight"
sampled_data_df = df_auto.select(x,y)
# Trains a bisecting k-means model.
# bkm = BisectingKMeans().setK(4).setSeed(1)
# model = bkm.fit(sampled_data)


# In[56]:


df_auto_ms = spark.read.csv(data_link, header=True, inferSchema=True)


# In[57]:


df = df_auto_ms.drop("mpg","cylinders","displacement","acceleration","year","origin")


# In[59]:


from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols = ['horsepower','weight'], outputCol = 'features')
v_df = vectorAssembler.transform(df)
v_df = v_df.select(['features'])
v_df.show(3)


# In[60]:


from pyspark.ml.clustering import BisectingKMeans

# Trains a bisecting k-means model.
bkm = BisectingKMeans().setK(2).setSeed(1)
model = bkm.fit(v_df)


# In[63]:


# Evaluate clustering.
cost = model.computeCost(v_df)
print("Within Set Sum of Squared Errors = " + str(cost))


# In[64]:


# Shows the result.
print("Cluster Centers: ")
centers = model.clusterCenters()
for center in centers:
    print(center)


# # **<center>Thanks!</center>**
