import numpy as np
from toy_data import generate_toy_data
feat_num = 2
layer_1_units = 10
layer_2_units = 5
beta = 10
gamma = 1
grow_rate = 5
warm_start = 1
err_tol = 1e-8

from numpy import vectorize

train_data_x, train_data_y,test_data_x, test_data_y = generate_toy_data()
data_num = train_data_y.size

a_0 = train_data_x
a_0_pinv = np.linalg.pinv(a_0)
W_1 = np.zeros((layer_1_units,feat_num))
init_var = 1
z_1 = init_var * np.random.randn(layer_1_units,data_num)
a_1 = init_var * np.random.randn(layer_1_units,data_num)

W_2 = np.zeros((layer_2_units,layer_1_units))
z_2 = init_var * np.random.randn(layer_2_units,data_num)
a_2 = init_var * np.random.randn(layer_1_units,data_num)

W_3 = np.zeros((1,layer_2_units))
z_3 = init_var * np.random.randn(1,data_num)

_lambda = np.zeros((1,data_num))
#_lambda = np.random.randn(1,data_num)

def activation(i):
    if i > 0:
        return i
    else:
        return 0

def get_z_l(a,w_a):
    def f_z(z):
        return gamma*(a-activation(z))**2 + beta*(z-w_a)**2

    z1 = max((a*gamma+w_a*beta)/(beta+gamma),0)
    result1 = f_z(z1)

    z2 = min(w_a,0)
    result2 = f_z(z2)

    if result1 <= result2:
        return z1
    else:
        return z2



def get_z_L(y,w_a,_lambda):
    if y==-1:
        def f_z(z):
            return beta*z**2 - (2*beta*w_a-_lambda)*z + max(1+z,0)

        z1 = min((2*beta*w_a - _lambda)/(2*beta),-1)
        z2 = max((2*beta*w_a-_lambda-1)/(2*beta),-1)
        if f_z(z1) < f_z(z2):
            return z1
        else:
            return z2

    if y==1:
        def f_z(z):
            return beta*z**2 - (2*beta*w_a - _lambda)*z + max(1-z,0)
        z1 = min((2*beta*w_a - _lambda+1)/(2*beta),1)
        z2 = max((2*beta*w_a - _lambda)/(2*beta),1)

        if f_z(z1) < f_z(z2):
            return z1
        else:
            return z2

    else:
        print("error class: {}".format(y))
        exit()


def get_predict(pre):
    if pre >= 0:
        return 1
    else:
        return -1

def get_loss(pre,gt):
    if gt==-1:
        return max(1+pre,0)
    elif gt==1:
        return max(1-pre,0)
    else:
        print("invalid gt..")
        exit()


vactivation = vectorize(activation)
vget_z_l = vectorize(get_z_l)
vget_z_L = vectorize(get_z_L)
vget_predict = vectorize(get_predict)
vget_loss = vectorize(get_loss)

def update(is_warm_start = False):
    global  z_1,z_2,z_3,_lambda,W_1,W_2,W_3
    # update layer 1
    old_W_1 = W_1
    old_z_1 =z_1
    W_1 = np.dot(z_1, a_0_pinv)
    a_1_left = np.linalg.inv((beta * np.dot(np.transpose(W_2), W_2) + gamma * np.eye(layer_1_units, dtype=float)))
    a_1_right = (beta * np.dot(np.transpose(W_2), z_2) + gamma * vactivation(z_1))
    a_1 = np.dot(a_1_left,a_1_right)
    z_1 = vget_z_l(a_1, np.dot(W_1, a_0))

    # update layer 2
    W_2 = np.dot(z_2, np.linalg.pinv(a_1))
    #numpy.linalg.linalg.LinAlgError: Singular matrix
    a_2_left = np.linalg.inv((beta * np.dot(np.transpose(W_3), W_3) + gamma * np.eye(layer_2_units, dtype=float)))
    a_2_right = (beta * np.dot(np.transpose(W_3), z_3) + gamma * vactivation(z_2))
    a_2 = np.dot(a_2_left , a_2_right)
    z_2 = vget_z_l(a_2, np.dot(W_2, a_1))

    # update last layer
    W_3 = np.dot(z_3, np.linalg.pinv(a_2))
    z_3 = vget_z_L(train_data_y, np.dot(W_3, a_2),_lambda)

    #print("z_3: ")
    #print(z_3)
    loss = vget_loss(z_3,train_data_y)
    #print("loss: {}".format(loss))
    #print("lambda: ")
    #print(_lambda)


    if not is_warm_start:
        _lambda = _lambda + beta * (z_3 - np.dot(W_3,a_2))

    #ret = np.linalg.norm(z_3 - np.dot(W_3,a_2),2)
    #ret = np.linalg.norm(old_W_1-W_1,2)
    ret = np.linalg.norm(old_z_1 - z_1, 2)
    return ret

def test():
    a_0 = test_data_x
    layer_1_output = vactivation(np.dot(W_1,a_0))
    layer_2_output = vactivation(np.dot(W_2,layer_1_output))
    predict = np.dot(W_3,layer_2_output)
    pre = vget_predict(predict)

    print("layer 1 value: \n")
    print(layer_1_output)
    print("layer 2 value: \n")
    print(layer_2_output)
    print("layer 3 value: \n")
    print(predict)


    hit = np.equal(pre,test_data_y)
    acc = np.sum(hit)*1.0/test_data_y.size
    print("test data predict accuracy: {}".format(acc))

def train():
    global  beta,gamma
    #warm start
    for i in range(warm_start):
        loss = update(is_warm_start=True)
        print("warm start, err :{}".format(loss))
    #real start
    i = 1
    while 1:
        loss = update(is_warm_start=False)
        print("iteration {}, err :{}".format(i,loss))
        if i%100 == 0:
            beta =  grow_rate* beta
            gamma = gamma* beta

        if i%20==0:
            test()
        i = i + 1
        if loss < err_tol:
            break





train()
test()





