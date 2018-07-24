learning_rate = 0.1

all_costs = []

# w1 = np.random.randn()
# w2 = np.random.randn()
# b= np.random.randn()

for i in range(100000):
    random_number = np.random.randint(len(data_array))
    random_person = data_array[random_number]
    
    height = random_person[0]
    weight = random_person[1]

    z = w1*height+w2*weight+b
    predictedGender = sigmoid(z)
    
    actualGender = random_person[2]
    
    cost = (predictedGender-actualGender)**2
    
    
    ##############################
    all_costs.append(cost)
    ##############################
    
    dcost_prediction = 2 * (predictedGender-actualGender)
    dprediction_dz = sigmoid_derivative(z)
    dz_dw1 = random_person[0]
    dz_dw2 = random_person[1]
    dz_db = 1
    
    dcost_dw1 = dcost_prediction * dprediction_dz * dz_dw1
    dcost_dw2 = dcost_prediction * dprediction_dz * dz_dw2
    dcost_db  = dcost_prediction * dprediction_dz * dz_db
    
    w1 = w1 - learning_rate * dcost_dw1
    w2 = w2 - learning_rate * dcost_dw2
    b  = b  - learning_rate * dcost_db

plt.plot(all_costs)
plt.title('Cost Value over 100,000 iterations')
plt.xlabel('Iteration')
plt.ylabel('Cost Value')
plt.show()
print(w1, w2, b)
# print(all_costs)