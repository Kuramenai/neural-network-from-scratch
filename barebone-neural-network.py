import numpy as np
import matplotlib.pyplot as plt
import math


np.random.seed(0)

""" X = [[1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]] """

"样本数据"
def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

class Layer:
    "神经网络中某层次的初始化"
    def __init__(self,no_inputs,no_neurons):

        """初始化这一层的weights和biases,其中weights一开始是随机的
        但为了避免explosive gradient和 vanishing gradient 的出现采用 He weights 初始化方式"""

        # no_inputs : 前一个层次的神经元数目
        # no_neurons : 此层次的神经元数目
      
        self.weights = np.sqrt(2/(no_inputs + no_neurons))*np.random.randn(no_inputs,no_neurons)
        self.biases = np.zeros((1, no_neurons))  #biases 初始化为0

    def forward_Propagation(self,inputs, activation_function = None):
        "计算此层次的神经元直，如果提提供激活函数就用它来计算"

        # inputs : 前一层（或者一批样本）
        # activation_function : 本层次所使用的激活函数
       
        self.outputs = np.dot(inputs,self.weights) + self.biases

        if activation_function != None:
            self.outputs = activation_function(self.outputs)


def activation_ReLU(inputs):
    "ReLU激活函数的实现"
    output = np.maximum(0,inputs)
    return output

def activation_Softmax(inputs):
    "Softmax激活函数的实现"
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    probabilities = exp_values/np.sum(exp_values, axis=1, keepdims=True)
    outputs = probabilities
    return outputs

def calculate_Loss(network_output,y,loss_function):
    "计算神经网络输出的结果和样本之间的误差"

    # output: 神经网络输出的结果
    # y : 样本的正确结果
    # loss_function : 所使用的误差函数

    sample_losses = loss_function(network_output, y)
    data_loss = np.mean(sample_losses)
    return data_loss

def  calculate_Categorical_Cross_Entropy_Loss(network_outputs,class_targets):

    #计算样本的数目
    length_samples = len(network_outputs)
    #如果网络的输出完全错即为0,会出现log（0）的情况，则为了避免这个情况的出现先压缩其范围
    network_outputs_clipped = np.clip(network_outputs, 1e-7, 1 - 1e-7)

    loss = -np.log(network_outputs_clipped[range(length_samples),class_targets])
    return loss




# X : Training examples
# y : class targets that is correct result for each training sample

X,y = spiral_data(100,3)    

# Hidden Layer

layer1 = Layer(2,3)
layer1.forward_Propagation(X, activation_ReLU)

print(layer1.outputs[:5])

#Output Layer

layer2 = Layer(3,3)
layer2.forward_Propagation(layer1.outputs, activation_Softmax)

print(layer2.outputs[:5])

#误差

loss = calculate_Loss(layer2.outputs,y,calculate_Categorical_Cross_Entropy_Loss)

print(loss)






""" plt.scatter(X[:,0], X[:,1], c=y, cmap='brg')
plt.show() """



