
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
fireIndicator = df.select(df["Call Type"],F.when(df["Call Type"].like("%Fire%"),1).otherwise(0))
fireIndicator = fireIndicator.                withColumnRenamed('CASE WHEN Call Type LIKE %Fire% THEN 1 ELSE 0 END', 'Fire Indicator')
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
               'Supervisor District',
               'final priority')
df.show()


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

Neighborhoods_indexer = StringIndexer(inputCol='Neighborhooods - Analysis Boundaries', outputCol='Neighbors')
zip_indexer = StringIndexer(inputCol='Zipcode of Incident', outputCol='Zip')
batallion_indexer = StringIndexer(inputCol='Battalion', outputCol='Battalion_')
stationarea_indexer = StringIndexer(inputCol='Station Area', outputCol='StationArea')
box_indexer = StringIndexer(inputCol='Box', outputCol='Box_')
fireDistrict_indexer = StringIndexer(inputCol='Fire Prevention District', outputCol='FireDistrict')
supervisorDistrict_indexer = StringIndexer(inputCol='Supervisor District', outputCol='SupervisorDistrict')


# In[25]:

Neighborhoods_indexer_model = Neighborhoods_indexer.fit(df)
zip_indexer_model  = zip_indexer.fit(df)
batallion_indexer_model  = batallion_indexer.fit(df)
stationarea_indexer_model  = stationarea_indexer.fit(df)
box_indexer_model  = box_indexer.fit(df)
fireDistrict_model  = fireDistrict_indexer.fit(df)
supervisorDistrict_model  = supervisorDistrict_indexer.fit(df)


# In[26]:

df = Neighborhoods_indexer_model.transform(df)
df = zip_indexer_model.transform(df)
df = batallion_indexer_model.transform(df)
df = stationarea_indexer_model.transform(df)
df = box_indexer_model.transform(df)
df = fireDistrict_model.transform(df)
df = supervisorDistrict_model.transform(df)


# In[27]:

df.columns


# In[28]:

df.select('Neighborhooods - Analysis Boundaries', 'Neighbors').show()


# In[29]:

df = df.select('fireIndicator',          'Neighbors',          'Zip',          'Battalion_',          'StationArea',          'Box_',          'Number Of Alarms',          'Unit sequence in call dispatch',          'FireDistrict',          'SupervisorDistrict',          'final priority')


# In[30]:

df.printSchema()


# In[31]:

df.show()


# In[32]:

features = ['Neighbors',
 'Zip',
 'Battalion_',
 'StationArea',
 'Box_',
 'Number Of Alarms',
 'Unit sequence in call dispatch',
 'FireDistrict',
 'SupervisorDistrict',
 'final priority']


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

df = df.drop("Neighbors",
 "Zip",
 "Battalion_",
 "StationArea",
 "Box_",
 "Number Of Alarms",
 "Unit sequence in call dispatch",
 "FireDistrict",
 "SupervisorDistrict",
 "final priority")


# In[37]:

df = df.withColumnRenamed('fireIndicator', 'label')


# In[38]:

df.show()


# In[39]:

(trainDF, testDF) = df.randomSplit([0.75, 0.25], seed = 12345)


# In[40]:

print(trainDF.count())
print(testDF.count())


# In[ ]:




# In[ ]:




# In[ ]:




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


# In[ ]:

##################################################


# In[ ]:

from pyspark.ml.tuning import ParamGridBuilder
param_grid = ParamGridBuilder().    addGrid(logreg.regParam, [0, 0.5, 1, 2]).    addGrid(logreg.elasticNetParam, [0, 0.5, 1]).    build()


# In[ ]:

from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")


# In[ ]:

from pyspark.ml.tuning import CrossValidator
cv = CrossValidator(estimator=logreg, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=4)


# In[ ]:

cv_model = cv.fit(df)


# In[ ]:

pred_training_cv = cv_model.transform(trainDF)


# In[ ]:

pred_test_cv = cv_model.transform(testDF)


# In[ ]:

print('Intercept: ' + str(cv_model.bestModel.intercept) + "\n"
     'coefficients: ' + str(cv_model.bestModel.coefficients))

