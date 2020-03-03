#!/usr/bin/env python
# coding: utf-8

# # What is SparkContext?
# Spark comes with interactive python shell in which PySpark is already installed in it. PySpark automatically creates a SparkContext for you in the PySpark Shell. SparkContext is an entry point into the world of Spark. An entry point is a way of connecting to Spark cluster. We can use SparkContext using sc variable. In the following examples, we retrieve SparkContext version and Python version of SparkContext.

# In[1]:


# to retrieve SparkContext version
sc.version


# In[2]:


# to retriece Python version of SparkContext
sc.pythonVer


# In[5]:



import ibmos2spark
@hidden_cell
# The following code contains the credentials for a file in your IBM Cloud Object Storage.
# IBM WATSON PYTHON+SPARK ENVIRONMENT
# xxxx - PUT YOUR CREDENTIAL
credentials = {
    'endpoint': 'xxxx',
    'service_id': 'xxxx',
    'iam_service_endpoint': 'xxxx',
    'api_key': 'xxxx'
}

configuration_name = 'os_0a7faf2d576d4d0e985305b42aae4ce7_configs'
cos = ibmos2spark.CloudObjectStorage(sc, credentials, configuration_name, 'bluemix_cos')

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
df_data_1 = spark.read  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')  .option('header', 'true')  .load(cos.url('people.csv', 'sparktry-donotdelete-pr-adbnjs9sf1yuav'))
df_data_1.take(5)


# # Problem 1:->
# 
# ### Create a Spark program to read the airport data from airports.text and  find all the airports which are located in United States and output the airport's name and the city's name.

# In[6]:



@hidden_cell
# The following code contains the credentials for a file in your IBM Cloud Object Storage.
# You might want to remove those credentials before you share your notebook.
credentials_1 = {
    'IAM_SERVICE_ID': 'XXXX',
    'IBM_API_KEY_ID': 'XXXX',
    'ENDPOINT': 'XXXX',
    'IBM_AUTH_ENDPOINT': 'XXXX',
    'BUCKET': 'XXXX',
    'FILE': 'airports.text'
}
aiport_data = cos.url('airports.text', 'sparktry-donotdelete-pr-adbnjs9sf1yuav')


# In[7]:


# load the aiport_data dataset into a rdd named clusterRDD
clusterRDD = sc.textFile(aiport_data)


# In[8]:


# Data of 5 rows
clusterRDD.take(5)


# In[10]:


# For removing the commas and spaces
import re
class Utils():
    COMMA_DELIMITER = re.compile(''',(?=(?:[^"]*"[^"]*")*[^"]*$)''')


# In[11]:


# Splitting of the result
def splitComma(line: str):
    splits = Utils.COMMA_DELIMITER.split(line)
    return "{}, {}".format(splits[1], splits[2])


# In[12]:


# Applying the tranformation
airportsInUSA = clusterRDD.filter(lambda line : Utils.COMMA_DELIMITER.split(line)[3] == "\"United States\"")


# In[13]:


# After applying map to join aiport name and city
airportsNameAndCityNames = airportsInUSA.map(splitComma)


# In[15]:


airportsNameAndCityNames.take(10)


# # Problem - 2 :->
# ### Create a Spark program to read the airport data from in/airports.text,  find all the airports whose latitude are bigger than 40.

# In[26]:


airports_latitude = clusterRDD.filter(lambda line: float(Utils.COMMA_DELIMITER.split(line)[6]) > 40)


# In[27]:


# Splitting of the result
def splitCommaLat(line: str):
    splits = Utils.COMMA_DELIMITER.split(line)
    return "{}, {}".format(splits[1], splits[6])


# In[28]:


airportsNames = airports_latitude.map(splitCommaLat)


# In[30]:


airportsNames.take(20)


# In[31]:


# Number of outputs
airportsNames.count()

