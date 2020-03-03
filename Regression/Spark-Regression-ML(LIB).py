#!/usr/bin/env python
# coding: utf-8

# # Task :- Use of  Machine Learning Library (MLlib) :- Spark’s machine learning (ML) library for Regression
# ### Models :- Linear Regression, Decision Tree Regression,Random forest regression,Gradient-boosted tree regression.
# ### Dataset :- insurance.csv.
# 
# **MLlib is Spark’s machine learning (ML) library. Its goal is to make practical machine learning scalable and easy. At a high level, it provides tools such as:**
# 
# <ul>
# <li> ML Algorithms: common learning algorithms such as classification, regression, clustering, and collaborative filtering.</li>
# 
# <li> Featurization: feature extraction, transformation, dimensionality reduction, and selection.</li>
# 
# <li> Pipelines: tools for constructing, evaluating, and tuning ML Pipelines.</li>
# 
# <li> Persistence: saving and load algorithms, models, and Pipelines.</li>
# 
# <li> Utilities: linear algebra, statistics, data handling, etc.</li>
# </ul>    

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
# IBM WATSON PYTHON+SPARK ENVIRONMENT
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


# In[4]:



# @hidden_cell
# The following code contains the credentials for a file in your IBM Cloud Object Storage.
# You might want to remove those credentials before you share your notebook.
credentials_1 = {
    'IAM_SERVICE_ID': 'XXXX',
    'IBM_API_KEY_ID': 'XXXX',
    'ENDPOINT': 'XXXX',
    'IBM_AUTH_ENDPOINT': 'XXXX',
    'BUCKET': 'XXXX',
    'FILE': 'insurance.csv'
}
data_link = cos.url('insurance.csv', 'sparktry-donotdelete-pr-adbnjs9sf1yuav')


# In[5]:


df_insurence = spark.read.csv(data_link, header=True, inferSchema=True)


# In[6]:


df_insurence.columns


# In[7]:


df_insurence.show(10)


# In[8]:


df_insurence.printSchema()


# In[9]:


df_insurence.describe().show()


# In[10]:


# for sex columns 
from pyspark.ml.feature import StringIndexer


# In[11]:


indexer = StringIndexer(inputCol="smoker", outputCol="smokerIndex")
df_insurence = indexer.fit(df_insurence).transform(df_insurence)


# In[12]:


indexer = StringIndexer(inputCol="sex", outputCol="sexIndex")
df_insurence = indexer.fit(df_insurence).transform(df_insurence)


# In[13]:


df_insurence.show()


# In[14]:


from pyspark.ml.feature import OneHotEncoderEstimator
encoder = OneHotEncoderEstimator(inputCols=["children"],outputCols=["children_enc"])
model = encoder.fit(df_insurence)
encoded = model.transform(df_insurence)
encoded.show()


# In[15]:


df_insurence.show()


# In[16]:


df = df_insurence.drop("sex", "smoker",'region') 


# In[17]:


df.show(10)


# ### Creating the Feature vector

# In[40]:


from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols = ['age','bmi','smokerIndex','sexIndex'], outputCol = 'features')


# In[41]:


vec_df = vectorAssembler.transform(df)


# In[42]:


vec_df = vec_df.select(['features', 'charges'])


# In[43]:


vec_df.show(10)


# ## Train test split

# In[44]:


splits = vec_df.randomSplit([0.7, 0.3])
train_df = splits[0]
test_df = splits[1]


# In[45]:


train_df.show(5)


# In[46]:


test_df.show(5)


# # Linear Regression Model

# In[47]:


from pyspark.ml.regression import LinearRegression
# creating object
lr_model = LinearRegression(featuresCol = 'features', labelCol='charges',maxIter=10, regParam=0.3, elasticNetParam=0.8)


# In[48]:


# fitting of lr_model
lr_fit_model = lr_model.fit(train_df)


# In[49]:


# Print the coefficients and intercept for linear regression

print("Coefficients: %s" % str(lr_fit_model.coefficients))
print("Intercept: %s" % str(lr_fit_model.intercept))


# In[50]:


# Summarize the model over the training set and print out some metrics
trainingSummary = lr_fit_model.summary


# In[51]:


# summerizing the model
print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show(5)


# In[52]:


# model  summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)


# In[53]:


from pyspark.ml.evaluation import RegressionEvaluator
lr_predictions = lr_fit_model.transform(test_df)
lr_predictions.select("prediction","charges","features").show(5)


# In[54]:


evaluator = RegressionEvaluator(labelCol="charges", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(lr_predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)


# In[ ]:





# # Decision Tree Regression

# In[55]:


# object creation 
from pyspark.ml.regression import DecisionTreeRegressor
dt_model = DecisionTreeRegressor(featuresCol ='features', labelCol = 'charges')


# In[56]:


# fitting the decision tree model
dt_model = dt_model.fit(train_df)


# In[57]:


# Predicting the value
dt_predictions = dt_model.transform(test_df)
dt_predictions.select("prediction","charges","features").show(5)


# In[58]:


evaluator = RegressionEvaluator(labelCol="charges", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(dt_predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)


# In[ ]:





# # Random Forest Regression

# In[59]:


# Object creation
from pyspark.ml.regression import RandomForestRegressor
rf_model = RandomForestRegressor(featuresCol="features",labelCol = 'charges',maxDepth=4)


# In[61]:


# Fitting the model
rf_model_fit = rf_model.fit(train_df)


# In[63]:


# predicting the value
rf_predictions = rf_model_fit.transform(test_df)
rf_predictions.select('prediction', 'charges', 'features').show(5)


# In[64]:


# evaluating the model
rf_evaluator = RegressionEvaluator(labelCol="charges", predictionCol="prediction", metricName="rmse")
rmse = rf_evaluator.evaluate(rf_predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

rf_evaluator1 = RegressionEvaluator(labelCol="charges", predictionCol="prediction", metricName="r2")
r2 = rf_evaluator1.evaluate(rf_predictions)
print("R2 on test data = %g" % r2)


# In[ ]:





# # Gradient-boosted Tree Regression

# In[65]:


# object creation
from pyspark.ml.regression import GBTRegressor
gbt_model = GBTRegressor(featuresCol = 'features', labelCol = 'charges',maxIter=10)


# In[66]:


# fitting the model
gbt_model_fit = gbt_model.fit(train_df)


# In[68]:


# predicting the value
gbt_predictions = gbt_model_fit.transform(test_df)
gbt_predictions.select('prediction', 'charges', 'features').show(5)


# In[69]:


# evaluating 
gbt_evaluator = RegressionEvaluator(labelCol="charges", predictionCol="prediction", metricName="rmse")
rmse = gbt_evaluator.evaluate(gbt_predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

gbt_evaluator1 = RegressionEvaluator(labelCol="charges", predictionCol="prediction", metricName="r2")
r2 = gbt_evaluator1.evaluate(gbt_predictions)
print("R2 on test data = %g" % r2)


# # <center> ** Thanks! ** </center>
