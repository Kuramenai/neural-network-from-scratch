inputs = [1, 2, 3, 2.5]

weight1 = [0.2, 0.8, -0.5, 1.0]
weight2 = [0.5, -0.91, 0.26, -0.5]
weight3 = [-0.26, -0.27, 0.17, 0.87]

weights = [[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]

bias = [2, 3, 0.5]

output = []

bias_index = 0
sum = 0

for weight in  weights:
    print(weight)
    for i in range(0,4):
        print(f"{inputs[i]}*{weight[i]}")
        sum+= inputs[i]*weight[i]
    print(f"sum is{sum}")
    #sum+= bias[bias_index]
    #bias_index += 1
    #print(sum)
    output.append(sum + bias[bias_index])
    bias_index += 1

""" output = [inputs[0]*weight1[0] + inputs[1]*weight1[1] + inputs[2]*weight1[2] + inputs[3]*weight1[3] + bias[0],
          inputs[0]*weight2[0] + inputs[1]*weight2[1] + inputs[2]*weight2[2] + inputs[3]*weight2[3] + bias[1],
          inputs[0]*weight3[0] + inputs[1]*weight3[1] + inputs[2]*weight3[2] + inputs[3]*weight3[3] + bias[2]

] """

print(output)