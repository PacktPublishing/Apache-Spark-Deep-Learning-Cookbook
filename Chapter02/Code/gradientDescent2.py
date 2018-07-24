for i in range(100000):
    # set the random data points that will be used to calculate the summation
    random_number = np.random.randint(len(data_array))
    random_person = data_array[random_number]
    # the height and weight from the random individual are selected
    height = random_person[0]
    weight = random_person[1]
    z = w1 * height + w2 * weight + b
    predictedGender = sigmoid(z)
    actualGender = random_person[2]
    cost = (predictedGender - actualGender)**2
    # the cost value is appended to the list
    all_costs.append(cost)
    # partial derivatives of the cost function and summation are calculated
    dcost_predictedGender = 2 * (predictedGender - actualGender)
    dpredictedGenger_dz = sigmoid_derivative(z)
    dz_dw1 = height
    dz_dw2 = weight
    dz_db = 1
    dcost_dw1 = dcost_predictedGender * dpredictedGenger_dz * dz_dw1
    dcost_dw2 = dcost_predictedGender * dpredictedGenger_dz * dz_dw2
    dcost_db = dcost_predictedGender * dpredictedGenger_dz * dz_db
    # gradient descent calculation
    w1 = w1 - learning_rate * dcost_dw1
    w2 = w2 - learning_rate * dcost_dw2
    b = b - learning_rate * dcost_db
