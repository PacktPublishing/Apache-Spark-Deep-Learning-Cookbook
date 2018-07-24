
# coding: utf-8

# In[1]:

spark = SparkSession.builder    .master("local")    .appName("RecommendationEngine")    .config("spark.executor.memory", "6gb")    .getOrCreate()


# In[2]:

import os
os.listdir('ml-latest-small/')


# In[3]:

movies = spark.read.format('com.databricks.spark.csv')            .options(header='true', inferschema='true')            .load('ml-latest-small/movies.csv')
tags = spark.read.format('com.databricks.spark.csv')            .options(header='true', inferschema='true')            .load('ml-latest-small/tags.csv')
links = spark.read.format('com.databricks.spark.csv')            .options(header='true', inferschema='true')            .load('ml-latest-small/links.csv')
ratings = spark.read.format('com.databricks.spark.csv')            .options(header='true', inferschema='true')            .load('ml-latest-small/ratings.csv')


# In[4]:

ratings.columns


# In[5]:

ratings.show(truncate=False)


# In[6]:

tags.show(truncate = False)


# In[7]:

movies.select('genres').distinct().show(truncate = False)


# In[8]:

links.show()


# In[9]:

print('The number of rows in movies dataset is {}'.format(movies.toPandas().shape[0]))
print('The number of rows in ratings dataset is {}'.format(ratings.toPandas().shape[0]))
print('The number of rows in tags dataset is {}'.format(tags.toPandas().shape[0]))
print('The number of rows in links dataset is {}'.format(links.toPandas().shape[0]))


# In[10]:

for i in ratings.columns:
    ratings = ratings.withColumnRenamed(i, i+'_1')    


# In[11]:

ratings.show()


# In[12]:

temp1 = ratings.join(movies, ratings.movieId_1 == movies.movieId, how = 'inner')


# In[13]:

temp2 = temp1.join(links, temp1.movieId_1 == links.movieId, how = 'inner')


# In[14]:

mainDF = temp2.join(tags, (temp2.userId_1 == tags.userId) &
                    (temp2.movieId_1 == tags.movieId), how = 'left')


# In[15]:

print(temp1.count())
print(temp2.count())
print(mainDF.count())


# In[16]:

mainDF.groupBy(['tag']).agg({'rating_1':'count'})            .withColumnRenamed('count(rating_1)', 'Row Count').orderBy(["Row Count"],ascending=False)            .show()


# In[17]:

mainDF.columns


# In[18]:

mainDF = mainDF.select('userId_1','movieId_1','rating_1','title','genres', 'imdbId','tmdbId', 'timestamp_1')               .distinct()


# In[19]:

mainDF.count()


# In[20]:

movies.createOrReplaceTempView('movies_')
links.createOrReplaceTempView('links_')
ratings.createOrReplaceTempView('ratings_')


# In[21]:

mainDF_SQL = sqlContext.sql(
"""
select
r.userId_1
,r.movieId_1
,r.rating_1
,m.title
,m.genres
,l.imdbId
,l.tmdbId
,r.timestamp_1
from ratings_ r
inner join movies_ m on 
r.movieId_1 =  m.movieId
inner join links_ l on 
r.movieId_1 = l.movieId
"""
)


# In[22]:

mainDF_SQL.show(n =  5)


# In[23]:

mainDF_SQL.count()


# In[24]:

mainDF.describe('rating_1').show()


# In[25]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

mainDF.select('rating_1').toPandas().hist(figsize=(16, 6), grid=True)
plt.title('Histogram of Ratings')
plt.show()


# In[26]:

mainDF.groupBy(['rating_1']).agg({'rating_1':'count'})            .withColumnRenamed('count(rating_1)', 'Row Count').orderBy(["Row Count"],ascending=False)            .show()


# In[27]:

userId_frequency = mainDF.groupBy(['userId_1']).agg({'rating_1':'count'})            .withColumnRenamed('count(rating_1)', '# of Reviews')            .orderBy(["# of Reviews"],ascending=False)


# In[28]:

userId_frequency.show()


# In[29]:

userId_frequency.select('# of Reviews').toPandas().hist(figsize=(16, 6), grid=True)
plt.title('Histogram of User Ratings')
plt.show()


# In[30]:

mainDF = mainDF.withColumnRenamed('userId_1', 'userid')
mainDF = mainDF.withColumnRenamed('movieId_1', 'movieid')
mainDF = mainDF.withColumnRenamed('rating_1', 'rating')
mainDF = mainDF.withColumnRenamed('timestamp_1', 'timestamp')
mainDF = mainDF.withColumnRenamed('imdbId', 'imdbid')
mainDF = mainDF.withColumnRenamed('tmdbId', 'tmdbid')


# In[31]:

mainDF.columns


# In[32]:

import pyspark.sql.functions as F
mainDF = mainDF.withColumn("rating", F.round(mainDF["rating"], 0))


# In[33]:

from pyspark.ml.feature import StringIndexer
string_indexer = StringIndexer(inputCol="genres", outputCol="genreCount")
mainDF = string_indexer.fit(mainDF).transform(mainDF)
mainDF.show()


# In[34]:

mainDF = mainDF.select('rating', 'userid', 'movieid', 'imdbid', 'tmdbid', 'timestamp', 'genreCount')


# In[35]:

mainDF.show()


# In[36]:

trainDF, testDF = mainDF.randomSplit([0.8, 0.2], seed=1234)


# In[37]:

print('The number of rows in mainDF is {}'.format(mainDF.count()))
print('The number of rows in trainDF is {}'.format(trainDF.count()))
print('The number of rows in testDF is {}'.format(testDF.count()))


# In[38]:

import numpy as np
xtrain_array = np.array(trainDF.select('userid','movieid', 'genreCount').collect())
xtest_array = np.array(testDF.select('userid','movieid', 'genreCount').collect())


# In[39]:

ytrain_array = np.array(trainDF.select('rating').collect())
ytest_array = np.array(testDF.select('rating').collect())


# In[40]:

print(xtest_array.shape)
print(ytest_array.shape)
print(xtrain_array.shape)
print(ytrain_array.shape)


# In[41]:

import keras.utils as u
ytrain_OHE = u.to_categorical(ytrain_array)
ytest_OHE = u.to_categorical(ytest_array)


# In[42]:

print(ytrain_OHE.shape)
print(ytest_OHE.shape)


# In[43]:

from keras.models import Sequential
from keras.layers import Dense, Activation


# In[44]:

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=xtrain_array.shape[1]))
model.add(Dense(10, activation='relu'))
model.add(Dense(ytrain_OHE.shape[1], activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[45]:

accuracy_history = model.fit(xtrain_array, ytrain_OHE, epochs=20, batch_size=32)


# In[46]:

plt.plot(accuracy_history.history['acc'])
plt.title('Accuracy vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
plt.plot(accuracy_history.history['loss'])
plt.title('Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


# In[47]:

score = model.evaluate(xtest_array, ytest_OHE, batch_size=128)
accuracy_rate = score[1]*100
print('accuracy is {}%'.format(round(accuracy_rate,2)))

