#!/usr/bin/env python
# coding: utf-8

# # Task :-> Use of Machine Learning Library (MLlib) :- Spark’s machine learning (ML) library for Classification
# 
# ### Models :- Logistic Regression, Decision Tree classifier,Random forest classifier,Gradient-boosted tree classifier and Hyperparameter Tunning
# 
# ### Dataset :-> Bank.csv

# In[1]:


# For Local Machine sparkcontext creation
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()

# In cloud the sparkcontext is created as "sc" 
# ex:- IBM watson 
# python :- 3.6


# In[2]:


sc.version


# In[3]:


# The following code contains the credentials for a file in your IBM Cloud Object Storage.
import ibmos2spark
# @hidden_cell
credentials = {
    'endpoint': 'XXXX',
    'service_id': 'XXXXX',
    'iam_service_endpoint': 'XXXXX',
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
    'FILE': 'bank.csv'
}

data_link = cos.url('bank.csv', 'sparktry-donotdelete-pr-adbnjs9sf1yuav')


# In[5]:


df_bank = spark.read.csv(data_link, header=True, inferSchema=True)


# In[6]:


df_bank.show(10)


# In[8]:


df_bank.printSchema()


# In[10]:


df_bank.describe().show()


# In[11]:


numeric_features = [t[0] for t in df_bank.dtypes if t[1] == 'int']


# In[12]:


df_bank.select(numeric_features).describe().toPandas().transpose()


# In[13]:


numeric_data = df_bank.select(numeric_features).toPandas()


# In[20]:


# Plotting for correlation
import pandas as pd
from pandas.plotting import scatter_matrix
axs = scatter_matrix(numeric_data, figsize=(8, 8));

n = len(numeric_data.columns)
for i in range(n):
    v = axs[i, 0]
    v.yaxis.label.set_rotation(0)
    v.yaxis.label.set_ha('right')
    v.set_yticks(())
    h = axs[n-1, i]
    h.xaxis.label.set_rotation(90)
    h.set_xticks(())


# In[27]:


df = df_bank.select('age', 'job', 'marital', 'education', 'default','balance', 'housing', 'loan', 'contact', 'duration', 'campaign',
               'pdays', 'previous', 'poutcome', 'deposit')


# In[28]:


df.printSchema()


# In[29]:


# Code Taken From databricks.com

# https://docs.databricks.com/applications/machine-learning/mllib/binary-classification-mllib-pipelines.html

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
categoricalColumns = ['job', 'marital', 'education', 'default','housing', 'loan', 'contact', 'poutcome']
stages = [] # stages in our Pipeline
for categoricalCol in categoricalColumns:
    # Category Indexing with StringIndexer
    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index")
    # Use OneHotEncoder to convert categorical variables into binary SparseVectors
    # encoder = OneHotEncoderEstimator(inputCol=categoricalCol + "Index", outputCol=categoricalCol + "classVec")
    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    # Add stages.  These are not run here, but will run all at once later on.
    stages += [stringIndexer, encoder]

label_stringIdx = StringIndexer(inputCol = 'deposit', outputCol = 'label')
stages += [label_stringIdx]

numericCols = ['age', 'balance', 'duration', 'campaign', 'pdays','previous']
assemblerInputs = [c + "classVec" for c in categoricalColumns] +numericCols
assembler = VectorAssembler(inputCols=assemblerInputs,outputCol="features")
stages += [assembler]


# In[30]:


cols = df.columns


# In[31]:


from pyspark.ml import Pipeline
pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(df)
df = pipelineModel.transform(df)
selectedCols = ['label', 'features'] + cols
df = df.select(selectedCols)
df.printSchema()


# In[32]:


pd.DataFrame(df.take(5), columns=df.columns).transpose()


# # Train Test Split

# In[33]:


train, test = df.randomSplit([0.7, 0.3], seed = 2018)


# In[34]:


print("Training Dataset Count: " + str(train.count()))
print("Test Dataset Count: " + str(test.count()))


# In[35]:


# Feature and label columns are created
train.show()


# In[36]:


# check test data
test.show(10)


# # Logistic Regression Model

# In[39]:


# fitting the logistic regression model
from pyspark.ml.classification import LogisticRegression

# Object ctreation and fitting
lr_model = LogisticRegression(labelCol="label", featuresCol="features",maxIter=10)

#Model fitting
model=lr_model.fit(train)


# In[41]:


import matplotlib.pyplot as plt
import numpy as np
beta = np.sort(model.coefficients)
plt.plot(beta)
plt.ylabel('Beta Coefficients')
plt.show()


# In[43]:


trainingSummary = model.summary


# In[44]:


roc = trainingSummary.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


# In[45]:


print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))


# In[46]:


#Precision and recall.

pr = trainingSummary.pr.toPandas()
plt.plot(pr['recall'],pr['precision'])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.show()


# In[47]:


# Predict the data
predictions = model.transform(test)
predictions.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').show(10)


# In[48]:


# Evaluation of model
from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator()
print('Test Area Under ROC', evaluator.evaluate(predictions))


# In[ ]:





# # Decision Tree Classifier

# In[49]:


from pyspark.ml.classification import DecisionTreeClassifier

# Object creation
dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth = 3)

# Fitting of model
dt_model = dt.fit(train)

# Prediction
predictions = dt_model.transform(test)
predictions.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').show(10)


# In[50]:


# Evaluate
evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))


# In[ ]:





# # Random Forest Classifier

# In[51]:


from pyspark.ml.classification import RandomForestClassifier

# Object creation and fitting
rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')

# Fitting
rf_model = rf.fit(train)

# Prediction
predictions = rf_model.transform(test)
predictions.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').show(10)


# In[52]:


# Evaluate 
evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))


# In[ ]:





# # Gradient-Bosted Tree Classsifier

# In[53]:


from pyspark.ml.classification import GBTClassifier

# object creation and fitting
gbt = GBTClassifier(maxIter=10)

# Fitting of model
gbt_model = gbt.fit(train)

# Predicting 
predictions = gbt_model.transform(test)
predictions.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').show(10)


# In[54]:


# Evaluation
evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))


# In[ ]:





# # Hyperparameter tunning, training over 20 trees!

# In[55]:


from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

paramGrid = (ParamGridBuilder()
             .addGrid(gbt.maxDepth, [2, 4, 6])
             .addGrid(gbt.maxBins, [20, 60])
             .addGrid(gbt.maxIter, [10, 20])
             .build())

cv = CrossValidator(estimator=gbt, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

# Run cross validations.  
cvModel = cv.fit(train)
predictions = cvModel.transform(test)
evaluator.evaluate(predictions)


# In[56]:


print("The area under ROC for test set after CV  is {}".format(evaluator.evaluate(predictions)))


# # **<CENTER>THANKS!</CENTER>**
