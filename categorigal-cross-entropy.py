import math

softmax_output = [0.7, 0.2, 0.1]
target_output = [1, 0, 0]

loss = sum([math.log(output)*target for output,target in  zip(softmax_output,target_output)])
loss = loss*(-1)
#loss += sum(loss)*(-1)

print(loss)