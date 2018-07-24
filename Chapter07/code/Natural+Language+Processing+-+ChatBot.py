
# coding: utf-8

# In[1]:

spark = SparkSession.builder    .master("local")    .appName("Natural Language Processing")    .config("spark.executor.memory", "6gb")    .getOrCreate()


# In[2]:

df = spark.read.format('com.databricks.spark.csv')                    .options(header='true', inferschema='true')                    .load('TherapyBotSession.csv')


# In[3]:

df.show()


# In[4]:

df = df.select('id', 'label', 'chat')


# In[5]:

df.show()


# In[6]:

df.groupBy("label")     .count()     .orderBy("count", ascending = False)     .show()


# In[7]:

import pyspark.sql.functions as F
df = df.withColumn('word_count',F.size(F.split(F.col('chat'),' ')))


# In[8]:

df.show()


# In[9]:

df.groupBy('label')    .agg(F.avg('word_count').alias('avg_word_count'))    .orderBy('avg_word_count', ascending = False)     .show()


# In[10]:

df_plot = df.select('id', 'word_count').toPandas()


# In[11]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

df_plot.set_index('id', inplace=True)
df_plot.plot(kind='bar', figsize=(16, 6))
plt.ylabel('Word Count')
plt.title('Word Count distribution')
plt.show()


# In[12]:

from textblob import TextBlob
def sentiment_score(chat):
        return TextBlob(chat).sentiment.polarity


# In[13]:

from pyspark.sql.types import FloatType
sentiment_score_udf = F.udf(lambda x: sentiment_score(x), FloatType())


# In[14]:

df = df.select('id', 'label', 'chat','word_count',
                   sentiment_score_udf('chat').alias('sentiment_score'))
df.show()


# In[15]:

df.groupBy('label')    .agg(F.avg('sentiment_score').alias('avg_sentiment_score'))    .orderBy('avg_sentiment_score', ascending = False)     .show()


# In[16]:

df = df.withColumn('words',F.split(F.col('chat'),' '))
df.show()


# In[17]:

stop_words = ['i','me','my','myself','we','our','ours','ourselves',
              'you','your','yours','yourself','yourselves','he','him',
              'his','himself','she','her','hers','herself','it','its',
              'itself','they','them','their','theirs','themselves',
              'what','which','who','whom','this','that','these','those',
              'am','is','are','was','were','be','been','being','have',
              'has','had','having','do','does','did','doing','a','an',
              'the','and','but','if','or','because','as','until','while',
              'of','at','by','for','with','about','against','between',
              'into','through','during','before','after','above','below',
              'to','from','up','down','in','out','on','off','over','under',
              'again','further','then','once','here','there','when','where',
              'why','how','all','any','both','each','few','more','most',
              'other','some','such','no','nor','not','only','own','same',
              'so','than','too','very','can','will','just','don','should','now']


# In[18]:

from pyspark.ml.feature import StopWordsRemover 


# In[19]:

stopwordsRemovalFeature = StopWordsRemover(inputCol="words", 
                                           outputCol="words without stop").setStopWords(stop_words)


# In[20]:

from pyspark.ml import Pipeline
stopWordRemovalPipeline = Pipeline(stages=[stopwordsRemovalFeature])
pipelineFitRemoveStopWords = stopWordRemovalPipeline.fit(df)


# In[21]:

df = pipelineFitRemoveStopWords.transform(df)
df.select('words', 'words without stop').show(5)


# In[22]:

label = F.udf(lambda x: 1.0 if x == 'escalate' else 0.0, FloatType())
df = df.withColumn('label', label('label'))


# In[23]:

df.select('label').show()


# In[24]:

import pyspark.ml.feature as feat
TF_ = feat.HashingTF(inputCol="words without stop", 
                     outputCol="rawFeatures", numFeatures=100000)
IDF_ = feat.IDF(inputCol="rawFeatures", outputCol="features")


# In[25]:

pipelineTFIDF = Pipeline(stages=[TF_, IDF_])


# In[26]:

pipelineFit = pipelineTFIDF.fit(df)
df = pipelineFit.transform(df)


# In[27]:

df.select('label', 'rawFeatures','features').show()


# In[28]:

(trainingDF, testDF) = df.randomSplit([0.75, 0.25], seed = 1234)


# In[29]:

from pyspark.ml.classification import LogisticRegression
logreg = LogisticRegression(regParam=0.025)


# In[30]:

logregModel = logreg.fit(trainingDF)


# In[31]:

predictionDF = logregModel.transform(testDF)


# In[32]:

predictionDF.select('label', 'probability', 'prediction').show()


# In[33]:

predictionDF.crosstab('label', 'prediction').show()


# In[34]:

from sklearn import metrics
actual = predictionDF.select('label').toPandas()
predicted = predictionDF.select('prediction').toPandas()


# In[35]:

print('accuracy score: {}%'.format(round(metrics.accuracy_score(actual, predicted),3)*100))


# In[36]:

from pyspark.ml.evaluation import BinaryClassificationEvaluator

scores = predictionDF.select('label', 'rawPrediction')
evaluator = BinaryClassificationEvaluator()
print('The ROC score is {}%'.format(round(evaluator.evaluate(scores),3)*100))


# In[37]:

predictionDF.describe('label').show()


# In[ ]:



