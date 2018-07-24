
# coding: utf-8

# In[1]:

spark = SparkSession.builder    .master("local")    .appName("StockMarket")    .config("spark.executor.memory", "6gb")    .getOrCreate()


# In[2]:

df =spark.read.format('com.databricks.spark.csv')                    .options(header='true', inferschema='true')                    .load('AAPL.csv')


# In[3]:

df.show()


# In[4]:

import pyspark.sql.functions as f
df = df.withColumn('date', f.to_date('Date'))


# In[5]:

df.show(n=5)


# In[6]:

date_breakdown = ['year', 'month', 'day']
for i in enumerate(date_breakdown):
    index = i[0]
    name = i[1]
    df = df.withColumn(name, f.split('date', '-')[index])


# In[7]:

df.show(n=10)


# In[8]:

df_plot = df.select('year', 'Adj Close').toPandas()


# In[9]:

from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

df_plot.set_index('year', inplace=True)
df_plot.plot(figsize=(16, 6), grid=True)
plt.title('Apple stock')
plt.ylabel('Stock Quote ($)')
plt.show()


# In[10]:

df.toPandas().shape


# In[11]:

df.dropna().count()


# In[12]:

df.select('Open', 'High', 'Low', 'Close', 'Adj Close').describe().show()


# In[13]:

df.groupBy(['year']).agg({'Adj Close':'count'})            .withColumnRenamed('count(Adj Close)', 'Row Count')            .orderBy(["year"],ascending=False)            .show()


# In[14]:

trainDF = df[df.year < 2017] 
testDF = df[df.year > 2016]


# In[15]:

trainDF.toPandas().shape


# In[16]:

testDF.toPandas().shape


# In[17]:

trainDF_plot = trainDF.select('year', 'Adj Close').toPandas()
trainDF_plot.set_index('year', inplace=True)
trainDF_plot.plot(figsize=(16, 6), grid=True)
plt.title('Apple Stock 2000-2016')
plt.ylabel('Stock Quote ($)')
plt.show()


# In[18]:

testDF_plot = testDF.select('year', 'Adj Close').toPandas()
testDF_plot.set_index('year', inplace=True)
testDF_plot.plot(figsize=(16, 6), grid=True)
plt.title('Apple Stock 2017-2018')
plt.ylabel('Stock Quote ($)')
plt.show()


# In[19]:

import numpy as np
trainArray = np.array(trainDF.select('Open', 'High', 'Low', 'Close','Volume', 'Adj Close' ).collect())
testArray = np.array(testDF.select('Open', 'High', 'Low', 'Close','Volume', 'Adj Close' ).collect())


# In[20]:

print(trainArray[0])
print('-------------')
print(testArray[0])


# In[21]:

from sklearn.preprocessing import MinMaxScaler
minMaxScale = MinMaxScaler()


# In[22]:

minMaxScale.fit(trainArray)


# In[23]:

testingArray = minMaxScale.transform(testArray)
trainingArray = minMaxScale.transform(trainArray)


# In[24]:

print(testingArray[0])
print('--------------')
print(trainingArray[0])


# In[25]:

xtrain = trainingArray[:, 0:-1]
xtest = testingArray[:, 0:-1]
# ytrain = trainingArray[:, 5]
# ytest = testingArray[:, 5]
ytrain = trainingArray[:, -1:]
ytest = testingArray[:, -1:]


# In[26]:

trainingArray[0]


# In[27]:

xtrain[0]


# In[28]:

ytrain[0]


# In[29]:

print('xtrain shape = {}'.format(xtrain.shape))
print('xtest shape = {}'.format(xtest.shape))
print('ytrain shape = {}'.format(ytrain.shape))
print('ytest shape = {}'.format(ytest.shape))


# In[30]:

plt.figure(figsize=(16,6))
plt.plot(xtrain[:,0],color='red', label='open')
plt.plot(xtrain[:,1],color='blue', label='high')
plt.plot(xtrain[:,2],color='green', label='low')
plt.plot(xtrain[:,3],color='purple', label='close')
plt.legend(loc = 'upper left')
plt.title('Open, High, Low, and Close by Day')
plt.xlabel('Days')
plt.ylabel('Scaled Quotes')
plt.show()


# In[31]:

plt.figure(figsize=(16,6))
plt.plot(xtrain[:,4],color='black', label='volume')
plt.legend(loc = 'upper right')
plt.title('Volume by Day')
plt.xlabel('Days')
plt.ylabel('Scaled Volume')
plt.show()


# In[32]:

from keras import models, layers


# In[33]:

model = models.Sequential()
model.add(layers.LSTM(1, input_shape=(1,5)))
model.add(layers.Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')


# In[34]:

xtrain = xtrain.reshape((xtrain.shape[0], 1, xtrain.shape[1]))
xtest  = xtest.reshape((xtest.shape[0], 1, xtest.shape[1]))


# In[35]:

print('The shape of xtrain is {}: '.format(xtrain.shape))
print('The shape of xtest is {}: '.format(xtest.shape))


# In[36]:

loss = model.fit(xtrain, ytrain, batch_size=10, epochs=100)


# In[37]:

plt.plot(loss.history['loss'], label = 'loss')
plt.title('mean squared error by epoch')
plt.legend()
plt.show()


# In[38]:

predicted = model.predict(xtest)


# In[39]:

combined_array = np.concatenate((ytest, predicted), axis = 1)


# In[40]:

plt.figure(figsize=(16,6))
plt.plot(combined_array[:,0],color='red', label='actual')
plt.plot(combined_array[:,1],color='blue', label='predicted')
plt.legend(loc = 'lower right')
plt.title('2017 Actual vs. Predicted APPL Stock')
plt.xlabel('Days')
plt.ylabel('Scaled Quotes')
plt.show()


# In[41]:

import sklearn.metrics as metrics
np.sqrt(metrics.mean_squared_error(ytest,predicted))


# In[ ]:



