#!/usr/bin/env python
# coding: utf-8

# # Task :-> Spark-RDD and Airport questions and House avg. price
# ## Dataset :-> airport.txt, RealEstate.csv

# In[1]:


sc.version


# In[2]:


# The following code contains the credentials for a file in your IBM Cloud Object Storage.
import ibmos2spark
# @hidden_cell
credentials = {
    'endpoint': 'XXXX',
    'service_id': 'XXXXXX',
    'iam_service_endpoint': 'XXXX',
    'api_key': 'XXXX'
}

configuration_name = 'os_0a7faf2d576d4d0e985305b42aae4ce7_configs'
cos = ibmos2spark.CloudObjectStorage(sc, credentials, configuration_name, 'bluemix_cos')


# In[3]:



# @hidden_cell
# The following code contains the credentials for a file in your IBM Cloud Object Storage.
# You might want to remove those credentials before you share your notebook.
credentials_1 = {
    'IAM_SERVICE_ID': 'XXXX',
    'IBM_API_KEY_ID': 'XXXX',
    'ENDPOINT': 'XXXX',
    'IBM_AUTH_ENDPOINT': 'XXXXX',
    'BUCKET': 'XXXX',
    'FILE': 'airports.text'
}

airports_data = cos.url('airports.text', 'sparktry-donotdelete-pr-adbnjs9sf1yuav')


# In[4]:


# Reduce method
inputIntegers = [1, 2, 3, 4, 5]
integerRdd = sc.parallelize(inputIntegers)
   
product = integerRdd.reduce(lambda x, y: x * y)
print("product is :{}".format(product))


# # Problem Statement
# Create a Spark program to read the airport data from in/airports.text,generate a pair RDD with airport name being the key and country name being the value.Then remove all the airports which are located in United States and then convert the country name to uppercase.

# In[5]:


import re
class Utils():
    COMMA_DELIMITER = re.compile(''',(?=(?:[^"]*"[^"]*")*[^"]*$)''')


# In[6]:


airportsRDD = sc.textFile(airports_data)


# In[7]:


airportPairRDD = airportsRDD.map(lambda line: (Utils.COMMA_DELIMITER.split(line)[1],Utils.COMMA_DELIMITER.split(line)[3]))


# In[8]:


airportsNotInUSA = airportPairRDD.filter(lambda keyValue: keyValue[1] != "\"United States\"")


# In[11]:


airportsNotInUSA.take(20)


# In[12]:


upperCase = airportPairRDD.mapValues(lambda countryName: countryName.upper())


# In[90]:


upperCase.take(30)


# # Task :-> Average_House_Price_Calculation

# In[91]:



# @hidden_cell
# The following code contains the credentials for a file in your IBM Cloud Object Storage.
# You might want to remove those credentials before you share your notebook.
credentials_2 = {
    'IAM_SERVICE_ID': 'iam-ServiceId-1c03b368-fd26-4891-a4b0-b0f383badb77',
    'IBM_API_KEY_ID': 'WMxdeLf8k6VGIhKFG2D9bklY7EmVI_nm6HIcx-cCNexe',
    'ENDPOINT': 'https://s3.eu-geo.objectstorage.service.networklayer.com',
    'IBM_AUTH_ENDPOINT': 'https://iam.eu-gb.bluemix.net/oidc/token',
    'BUCKET': 'sparktry-donotdelete-pr-adbnjs9sf1yuav',
    'FILE': 'RealEstate.csv'
}
real_data = cos.url('RealEstate.csv', 'sparktry-donotdelete-pr-adbnjs9sf1yuav')


# In[92]:


real_state_data = sc.textFile(real_data)


# In[94]:


real_state_data.take(10)


# In[96]:


cleanedLines = real_state_data.filter(lambda line: "Bedrooms" not in line)


# In[97]:


housePricePairRdd = cleanedLines.map(lambda line: (line.split(",")[3], float(line.split(",")[2])))


# In[98]:


createCombiner = lambda x: (1, x)
mergeValue = lambda avgCount, x: (avgCount[0] + 1, avgCount[1] + x)
mergeCombiners = lambda avgCountA, avgCountB: (avgCountA[0] + avgCountB[0], avgCountA[1] + avgCountB[1])


# In[99]:


housePriceTotal = housePricePairRdd.combineByKey(createCombiner, mergeValue, mergeCombiners)


# In[103]:


housePriceAvg = housePriceTotal.mapValues(lambda avgCount: avgCount[1] / avgCount[0])
print("Bedrooms","Price",end=" ")
print()
for bedrooms, avgPrice in housePriceAvg.collect():
    print("{} : {}".format(bedrooms, avgPrice))

