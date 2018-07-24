
# coding: utf-8

# In[1]:

from pyspark.sql import SparkSession


# In[2]:

spark = SparkSession.builder    .master("local")    .appName("Neural Network Model")    .config("spark.executor.memory", "6gb")    .getOrCreate()
   
sc = spark.sparkContext


# In[3]:

df = spark.createDataFrame([('Male', 67, 150), # insert column values
                            ('Female', 65, 135),
                            ('Female', 68, 130),
                            ('Male', 70, 160),
                            ('Female', 70, 130),
                            ('Male', 69, 174),
                            ('Female', 65, 126),
                            ('Male', 74, 188),
                            ('Female', 60, 110),
                            ('Female', 63, 125),
                            ('Male', 70, 173),
                            ('Male', 70, 145),
                            ('Male', 68, 175),
                            ('Female', 65, 123),
                            ('Male', 71, 145),
                            ('Male', 74, 160),
                            ('Female', 64, 135),
                            ('Male', 71, 175),
                            ('Male', 67, 145),
                            ('Female', 67, 130),
                            ('Male', 70, 162),
                            ('Female', 64, 107),
                            ('Male', 70, 175),
                            ('Female', 64, 130),
                            ('Male', 66, 163),
                            ('Female', 63, 137),
                            ('Male', 65, 165),
                            ('Female', 65, 130),
                            ('Female', 64, 109)], 
                           ['gender', 'height','weight']) # insert header values



# In[4]:

df.show(5)


# In[5]:

from pyspark.sql import functions 


# In[6]:

df = df.withColumn('gender',functions.when(df['gender']=='Female',0).otherwise(1))


# In[7]:

df = df.select('height', 'weight', 'gender')


# In[8]:

df.show()


# In[9]:

import numpy as np


# In[10]:

df.select("height", "weight", "gender").collect()


# In[11]:

data_array =  np.array(df.select("height", "weight", "gender").collect())
data_array #view the array


# In[12]:

data_array.shape


# In[13]:

data_array[0]


# In[14]:

data_array[28]


# In[15]:

print(data_array.max(axis=0))
print(data_array.min(axis=0))


# In[16]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[17]:

min_x = data_array.min(axis=0)[0]-10
max_x = data_array.max(axis=0)[0]+10
min_y = data_array.min(axis=0)[1]-10
max_y = data_array.max(axis=0)[1]+10

print(min_x, max_x, min_y, max_y)


# In[18]:

# formatting the plot grid, scales, and figure size
plt.figure(figsize=(9, 4), dpi= 75)
plt.axis([min_x,max_x,min_y,max_y])
plt.grid()
for i in range(len(data_array)):
    value = data_array[i]
    # assign labels values to specific matrix elements
    gender = value[2]
    height = value[0]
    weight = value[1]
    
    # filter data points by gender
    a = plt.scatter(height[gender==0],weight[gender==0], marker = 'x', c= 'b', label = 'Female')
    b = plt.scatter(height[gender==1],weight[gender==1], marker = 'o', c= 'b', label = 'Male')
    
    # plot values, title, legend, x and y axis
    plt.title('Weight vs Height by Gender')
    plt.xlabel('Height (in)')
    plt.ylabel('Weight (lbs)')
    plt.legend(handles=[a,b])
    


# In[19]:

np.random.seed(12345)


# In[20]:

w1 = np.random.randn()
w2 = np.random.randn()
b= np.random.randn()


# In[21]:

print(w1, w2, b)


# In[22]:

X = data_array[:,:2]
y = data_array[:,2]
print(X,y)


# In[23]:

x_mean = X.mean(axis=0)
x_std = X.std(axis=0)
print(x_mean, x_std)


# In[24]:

def normalize(X):
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    X = (X - X.mean(axis=0))/X.std(axis=0)
    return X


# In[25]:

X = normalize(X)
print(X)


# In[26]:

print('standard deviation')
print(round(X[:,0].std(axis=0),0))
print('mean')
print(round(X[:,0].mean(axis=0),0))


# In[27]:

data_array = np.column_stack((X[:,0], X[:,1],y))
print(data_array)


# In[28]:

# formatting the plot grid, scales, and figure size
plt.figure(figsize=(9, 4), dpi= 75)
# plt.axis([min_x,max_x,min_y,max_y])
plt.grid()
for i in range(len(data_array)):
    value_n = data_array[i]
    # assign labels values to specific matrix elements
    gender_n = value_n[2]
    height_n = value_n[0]
    weight_n = value_n[1]
    an = plt.scatter(height_n[gender_n==0.0],weight_n[gender_n==0.0], marker = 'x', c= 'b', label = 'Female')
    bn = plt.scatter(height_n[gender_n==1.0],weight_n[gender_n==1.0], marker = 'o', c= 'b', label = 'Male')    
    # plot values, title, legend, x and y axis
    plt.title('Weight vs Height by Gender (normalized)')
    plt.xlabel('Height (in)')
    plt.ylabel('Weight (lbs)')
    plt.legend(handles=[an,bn])


# In[29]:

def sigmoid(input):
    return 1/(1+np.exp(-input))


# In[30]:

X = np.arange(-10,10,1)
Y = sigmoid(X)


# In[31]:

plt.figure(figsize=(6, 4), dpi= 75)
plt.axis([-10,10,-0.25,1.2])
plt.grid()
plt.plot(X,Y)
plt.title('Sigmoid Function')
plt.show()


# In[32]:

def sigmoid_derivative(x):
    return sigmoid(x) * (1-sigmoid(x))


# In[33]:

plt.figure(figsize=(6, 4), dpi= 75)
plt.axis([-10,10,-0.25,1.2])
plt.grid()
X = np.arange(-10,10,1)
Y = sigmoid(X)
Y_Prime = sigmoid_derivative(X)
plt.plot(X, Y, label="Sigmoid",c='b')
plt.plot(X, Y_Prime, marker=".", label="Sigmoid Derivative", c='b')
plt.title('Sigmoid vs Sigmoid Derivative')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()


# In[34]:

data_array.shape


# In[35]:

for i in range(100):
    random_index = np.random.randint(len(data_array))
    point = data_array[random_index]
    print(i, point)


# In[36]:

learning_rate = 0.1

all_costs = []

for i in range(100000):
    # set the random data points that will be used to calculate the summation
    random_number = np.random.randint(len(data_array))
    random_person = data_array[random_number]
    
    # the height and weight from the random individual are selected
    height = random_person[0]
    weight = random_person[1]

    z = w1*height+w2*weight+b
    predictedGender = sigmoid(z)
    
    actualGender = random_person[2]
    
    cost = (predictedGender-actualGender)**2
    
    # the cost value is appended to the list
    all_costs.append(cost)
    
    # partial derivatives of the cost function and summation are calculated
    dcost_predictedGender = 2 * (predictedGender-actualGender)
    dpredictedGenger_dz = sigmoid_derivative(z)
    dz_dw1 = height
    dz_dw2 = weight
    dz_db = 1
    
    dcost_dw1 = dcost_predictedGender * dpredictedGenger_dz * dz_dw1
    dcost_dw2 = dcost_predictedGender * dpredictedGenger_dz * dz_dw2
    dcost_db  = dcost_predictedGender * dpredictedGenger_dz * dz_db
    
    # gradient descent calculation
    w1 = w1 - learning_rate * dcost_dw1
    w2 = w2 - learning_rate * dcost_dw2
    b  = b  - learning_rate * dcost_db


# In[37]:

plt.plot(all_costs)
plt.title('Cost Value over 100,000 iterations')
plt.xlabel('Iteration')
plt.ylabel('Cost Value')
plt.show()


# In[38]:

print('The final values of w1, w2, and b')
print('---------------------------------')
print('w1 = {}'.format(w1))
print('w2 = {}'.format(w2))
print('b  = {}'.format(b))


# In[39]:

for i in range(len(data_array)):
    random_individual = data_array[i]
    height = random_individual[0]
    weight = random_individual[1]
    z = height*w1 + weight*w2 + b
    predictedGender=sigmoid(z)
    print("Individual #{} actual score: {} predicted score: {}"
          .format(i+1,random_individual[2],predictedGender))


# In[40]:

def input_normalize(height, weight):
    inputHeight = (height - x_mean[0])/x_std[0]
    inputWeight = (weight - x_mean[1])/x_std[1]
    return inputHeight, inputWeight


# In[41]:

score = input_normalize(70, 180)


# In[42]:

def predict_gender(raw_score):
    gender_summation = raw_score[0]*w1 + raw_score[1]*w2 + b
    gender_score = sigmoid(gender_summation)
    if gender_score <= 0.5:
        gender = 'Female'
    else:
        gender = 'Male'
    return gender, gender_score


# In[43]:

predict_gender(score)


# In[44]:

score = input_normalize(50,120)


# In[45]:

predict_gender(score)


# In[46]:

x_min = min(data_array[:,0])-0.1
x_max = max(data_array[:,0])+0.1
y_min = min(data_array[:,1])-0.1
y_max = max(data_array[:,1])+0.1
increment= 0.05
print(x_min, x_max, y_min, y_max)


# In[47]:

x_data= np.arange(x_min, x_max, increment)


# In[48]:

y_data= np.arange(y_min, y_max, increment)


# In[49]:

xy_data = [[x_all, y_all] for x_all in x_data for y_all in y_data]


# In[50]:

for i in range(len(xy_data)):
    data = (xy_data[i])
    height = data[0]
    weight = data[1]  
    z_new = height*w1 + weight*w2 + b
    predictedGender_new=sigmoid(z_new)
    # print(height, weight, predictedGender_new)
    ax = plt.scatter(height[predictedGender_new<=0.5],
                     weight[predictedGender_new<=0.5], 
                     marker = 'o', c= 'r', label = 'Female')
    bx = plt.scatter(height[predictedGender_new > 0.5],
                     weight[predictedGender_new>0.5], 
                     marker = 'o', c= 'b', label = 'Male')    
    # plot values, title, legend, x and y axis
    plt.title('Weight vs Height by Gender')
    plt.xlabel('Height (in)')
    plt.ylabel('Weight (lbs)')
    plt.legend(handles=[ax,bx])

