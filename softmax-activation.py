import math 
import numpy as np

#Raw Python
E = math.e

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]
                ]

exp_values = [E**value for value in layer_outputs]

norm_base = sum(exp_values)
norm_values = [value/norm_base for value in exp_values]

#With NumPy

exp_values = np.exp(layer_outputs)

norm_values = exp_values/np.sum(exp_values, axis=1, keepdims=True)

print(exp_values)
print(norm_values)
print(sum(norm_values))