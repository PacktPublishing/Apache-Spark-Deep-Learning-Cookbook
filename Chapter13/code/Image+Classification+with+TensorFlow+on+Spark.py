
# coding: utf-8

# In[1]:

spark = SparkSession.builder    .master("local")    .appName("ImageClassification")    .config("spark.executor.memory", "6gb")    .getOrCreate()


# In[2]:

import pyspark.sql.functions as f
import sparkdl as dl


# In[3]:

dfMessi = dl.readImages('football/messi/').withColumn('label', f.lit(0))
dfRonaldo = dl.readImages('football/ronaldo/').withColumn('label', f.lit(1))


# In[4]:

dfMessi.show(n=10,truncate=False)


# In[5]:

dfRonaldo.show(n=10,truncate=False)


# In[6]:

trainDFmessi, testDFmessi = dfMessi.randomSplit([66.7, 33.3], seed =12)
trainDFronaldo, testDFronaldo = dfRonaldo.randomSplit([66.7, 33.3], seed=12)


# In[7]:

print('The number of images in trainDFmessi is {}'.format(trainDFmessi.toPandas().shape[0]))
print('The number of images in testDFmessi is {}'.format(testDFmessi.toPandas().shape[0]))
print('The number of images in trainDFronaldo is {}'.format(trainDFronaldo.toPandas().shape[0]))
print('The number of images in testDFronaldo is {}'.format(testDFronaldo.toPandas().shape[0]))


# In[8]:

trainDF = trainDFmessi.unionAll(trainDFronaldo)
testDF = testDFmessi.unionAll(testDFronaldo)


# In[9]:

print('The number of images in the training data is {}' .format(trainDF.toPandas().shape[0]))
print('The number of images in the testing  data is {}' .format(testDF.toPandas().shape[0]))


# In[10]:

from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

vectorizer = dl.DeepImageFeaturizer(inputCol="image", outputCol="features", modelName='InceptionV3')
logreg = LogisticRegression(maxIter=30,labelCol = "label", featuresCol="features")
pipeline = Pipeline(stages=[vectorizer, logreg])

pipeline_model = pipeline.fit(trainDF)


# In[11]:

predictDF = pipeline_model.transform(testDF)
predictDF.select('label', 'prediction').show(n = testDF.toPandas().shape[0], truncate=False)


# In[12]:

predictDF.crosstab('prediction', 'label').show()


# In[13]:

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
scoring = predictDF.select("prediction", "label")
accuracy_score = MulticlassClassificationEvaluator(metricName="accuracy")
rate = accuracy_score.evaluate(scoring)*100
print("accuracy: {}%" .format(round(rate,2)))


# In[14]:

from pyspark.ml.evaluation import BinaryClassificationEvaluator

binaryevaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
binary_rate = binaryevaluator.evaluate(predictDF)*100
print("accuracy: {}%" .format(round(binary_rate,2)))


# In[15]:

logregFT = LogisticRegression(
    regParam=0.05, 
    elasticNetParam=0.3,
    maxIter=15,labelCol = "label", featuresCol="features")
pipelineFT = Pipeline(stages=[vectorizer, logregFT])

pipeline_model_FT = pipelineFT.fit(trainDF)


# In[16]:

predictDF_FT = pipeline_model_FT.transform(testDF)
predictDF_FT.crosstab('prediction', 'label').show()


# In[17]:

binary_rate_FT = binaryevaluator.evaluate(predictDF_FT)*100
print("accuracy: {}%" .format(round(binary_rate_FT,2)))

