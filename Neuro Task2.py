def sigmoid(x):
    return 1 / (1 + (2.718281828459045 ** -x))  

def sigmoid_derivative(x):
    return x * (1 - x)

inputs = [0.05, 0.10]
expected_output = [0.01, 0.99]

weights_input_hidden = [[0.15, 0.20], [0.25, 0.30]]
weights_hidden_output = [[0.40, 0.45], [0.50, 0.55]]
bias_hidden = [0.35, 0.35]
bias_output = [0.60, 0.60]

hidden_layer_input = [0, 0]
hidden_layer_output = [0, 0]

for i in range(2):
    hidden_layer_input[i] = (inputs[0] * weights_input_hidden[0][i]) + (inputs[1] * weights_input_hidden[1][i]) + bias_hidden[i]
    hidden_layer_output[i] = sigmoid(hidden_layer_input[i])

output_layer_input = [0, 0]
output_layer_output = [0, 0]

for i in range(2):
    output_layer_input[i] = (hidden_layer_output[0] * weights_hidden_output[0][i]) + (hidden_layer_output[1] * weights_hidden_output[1][i]) + bias_output[i]
    output_layer_output[i] = sigmoid(output_layer_input[i])

error = [expected_output[i] - output_layer_output[i] for i in range(2)]

output_layer_delta = [error[i] * sigmoid_derivative(output_layer_output[i]) for i in range(2)]
hidden_layer_error = [0, 0]
hidden_layer_delta = [0, 0]

for i in range(2):
    hidden_layer_error[i] = (output_layer_delta[0] * weights_hidden_output[i][0]) + (output_layer_delta[1] * weights_hidden_output[i][1])
    hidden_layer_delta[i] = hidden_layer_error[i] * sigmoid_derivative(hidden_layer_output[i])

learning_rate = 0.5

for i in range(2):
    for j in range(2):
        weights_hidden_output[i][j] += learning_rate * hidden_layer_output[i] * output_layer_delta[j]
        weights_input_hidden[i][j] += learning_rate * inputs[i] * hidden_layer_delta[j]

print("Updated weights (input to hidden):", weights_input_hidden)
print("Updated weights (hidden to output):", weights_hidden_output)
