#!/usr/bin/env python
# coding: utf-8

# # Task:- Word Count
# ### IBM WATSON PYTHON+SPARK ENVIRONMENT
# **Spark Context is created as sc**

# In[27]:


# to retrieve SparkContext version
sc.version


# In[28]:


# The following code contains the credentials for a file in your IBM Cloud Object Storage.
import ibmos2spark
@hidden_cell
credentials = {
    'endpoint': 'XXXX',
    'service_id': 'XXXX',
    'iam_service_endpoint': 'XXXXX',
    'api_key': 'XXXXX'
}

configuration_name = 'os_0a7faf2d576d4d0e985305b42aae4ce7_configs'
cos = ibmos2spark.CloudObjectStorage(sc, credentials, configuration_name, 'bluemix_cos')


# In[29]:



# @hidden_cell
# The following code contains the credentials for a file in your IBM Cloud Object Storage.
# You might want to remove those credentials before you share your notebook.
credentials_1 = {
    'IAM_SERVICE_ID': 'XXXX',
    'IBM_API_KEY_ID': 'XXXX',
    'ENDPOINT': 'XXXX',
    'IBM_AUTH_ENDPOINT': 'XXXX',
    'BUCKET': 'XXXX',
    'FILE': 'word_count.text'
}
line = cos.url('word_count.text', 'sparktry-donotdelete-pr-adbnjs9sf1yuav')


# In[30]:


# Reading the File
lines = sc.textFile(line)


# In[31]:


lines.take(5)


# In[32]:


# Creating RDD and Flattening the result
words = lines.flatMap(lambda lines:lines.split(" "))


# In[33]:


# The words are...
words.take(10)


# In[34]:


# Words count
wordCounts = words.countByValue()


# In[35]:


# Output
for word, count in wordCounts.items():
    print("{} : {}".format(word, count))


# In[ ]:




