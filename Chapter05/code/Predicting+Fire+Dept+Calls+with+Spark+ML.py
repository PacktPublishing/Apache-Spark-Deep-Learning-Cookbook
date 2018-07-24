
# coding: utf-8

# In[1]:

from pyspark.sql import SparkSession


# In[2]:

spark = SparkSession.builder    .master("local")    .appName("Predicting Fire Dept Calls")    .config("spark.executor.memory", "6gb")    .getOrCreate()


# In[3]:

df = spark.read.format('com.databricks.spark.csv')                    .options(header='true', inferschema='true')                    .load('Fire_Department_Calls_for_Service.csv')


# In[4]:

df.show(2)


# In[5]:

df.select('Call Type Group').distinct().show()


# In[6]:

df.groupBy('Call Type Group').count().show()


# In[7]:

df2 = df.groupBy('Call Type Group').count()


# In[8]:

graphDF = df2.toPandas()
graphDF = graphDF.sort_values('count', ascending=False)


# In[9]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[10]:

graphDF.plot(x='Call Type Group', y = 'count', kind='bar')
plt.title('Call Type Group by Count')
plt.show()


# In[11]:

df.groupBy('Call Type').count().orderBy('count', ascending=False).show(100)


# In[12]:

from pyspark.sql import functions as F
fireIndicator = df.select(df["Call Type"],F.when(df["Call Type"].like("%Fire%"),1)                          .otherwise(0).alias('Fire Indicator'))
fireIndicator.show()


# In[13]:

fireIndicator.groupBy('Fire Indicator').count().show()


# In[14]:

df = df.withColumn("fireIndicator", F.when(df["Call Type"].like("%Fire%"),1).otherwise(0))


# In[15]:

df.printSchema()


# In[16]:

df.select('Call Type', 'fireIndicator').show(20)


# In[17]:

df = df.select('fireIndicator', 
               'Zipcode of Incident',
               'Battalion',
               'Station Area',
               'Box', 
               'Number of Alarms',
               'Unit sequence in call dispatch',
               'Neighborhooods - Analysis Boundaries',
               'Fire Prevention District',
               'Supervisor District')
df.show(5)


# In[18]:

print('Total Rows')
df.count()


# In[19]:

print('Rows without Null values')
df.dropna().count()


# In[20]:

print('Row with Null Values')
df.count()-df.dropna().count()


# In[21]:

df = df.dropna()


# In[22]:

df.groupBy('fireIndicator').count().orderBy('count', ascending = False).show()


# In[23]:

from pyspark.ml.feature import StringIndexer


# In[24]:

column_names = df.columns[1:]
column_names


# In[25]:

categoricalColumns = column_names
indexers = []
for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol+"_Index")
    indexers += [stringIndexer]


# In[26]:

models = []
for model in indexers:
    indexer_model = model.fit(df)
    models+=[indexer_model]
    
for i in models:
    df = i.transform(df)


# In[27]:

df.columns


# In[28]:

df.select('Neighborhooods - Analysis Boundaries', 'Neighborhooods - Analysis Boundaries_Index').show()


# In[29]:

df = df.select(
          'fireIndicator',
          'Zipcode of Incident_Index',
          'Battalion_Index',
          'Station Area_Index',
          'Box_Index',
          'Number of Alarms_Index',
          'Unit sequence in call dispatch_Index',
          'Neighborhooods - Analysis Boundaries_Index',
          'Fire Prevention District_Index',
          'Supervisor District_Index')


# In[30]:

df.printSchema()


# In[31]:

df.show(5)


# In[32]:

features = df.columns[1:]


# In[33]:

from pyspark.ml.feature import VectorAssembler

feature_vectors = VectorAssembler(
        inputCols = features,
        outputCol = "features")


# In[34]:

df = feature_vectors.transform(df)


# In[35]:

df.columns


# In[36]:

df = df.drop( 'Zipcode of Incident_Index',
              'Battalion_Index',
              'Station Area_Index',
              'Box_Index',
              'Number of Alarms_Index',
              'Unit sequence in call dispatch_Index',
              'Neighborhooods - Analysis Boundaries_Index',
              'Fire Prevention District_Index',
              'Supervisor District_Index')


# In[37]:

df = df.withColumnRenamed('fireIndicator', 'label')


# In[38]:

df.show()


# In[39]:

(trainDF, testDF) = df.randomSplit([0.75, 0.25], seed = 12345)


# In[40]:

print(trainDF.count())
print(testDF.count())


# In[41]:

from pyspark.ml.classification import LogisticRegression
logreg = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)
LogisticRegressionModel = logreg.fit(trainDF)


# In[42]:

df_predicted = LogisticRegressionModel.transform(testDF)


# In[43]:

df_predicted.printSchema()


# In[44]:

df_predicted.show(5)


# In[45]:

df_predicted.crosstab('label', 'prediction').show()


# In[46]:

from sklearn import metrics


# In[47]:

actual = df_predicted.select('label').toPandas()


# In[48]:

predicted = df_predicted.select('prediction').toPandas()


# In[49]:

metrics.accuracy_score(actual, predicted)


# In[50]:

df_predicted.groupBy('label').count().show()


# In[51]:

df_predicted.describe('label').show()

