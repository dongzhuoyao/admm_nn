import matplotlib.pyplot as plt
import numpy as np

def generate_toy_data():

    mean_a = [0.2, 0.2]
    cov_a = [[0.01, 0], [0, 0.01]]  # diagonal covariance

    mean_b = [0.8, 0.8]
    cov_b = [[0.01, 0], [0, 0.01]]  # diagonal covariance


    class_a_x = np.random.multivariate_normal(mean_a, cov_a, 500).T
    class_b_x = np.random.multivariate_normal(mean_b, cov_b, 500).T

    class_a_y = np.ones(np.size(class_a_x[0]))
    class_b_y = -1*np.ones(np.size(class_b_x[0]))

    """
    plt.plot(class_a_x[0], class_a_x[1], 'x')
    plt.plot(class_b_x[0], class_b_x[1], 'o')
    plt.axis('equal')
    plt.show()
    """

    data_x = np.concatenate((class_a_x, class_b_x), axis=1)
    data_y = np.concatenate((class_a_y, class_b_y), axis=0)

    num = data_y.size
    randomize = np.arange(num)
    np.random.shuffle(randomize)
    data_x = data_x[:,randomize]
    data_y = data_y[randomize]

    """
    plt.plot(data_x[0], data_x[1], 'x')
    plt.axis('equal')
    plt.show()
    """

    inter = int(0.8*num)
    data_y = data_y.reshape(1,num)
    return data_x[:,0:inter],data_y[:,0:inter],data_x[:,inter:-1],data_y[:,inter:-1]


#dd = generate_toy_data()
#print("ok")

