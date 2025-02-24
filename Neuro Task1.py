def tanh(x):
    return (2 / (1 + (2.71828 ** (-2 * x)))) - 1

def random_weight(seed):
    seed = (seed * 32719 + 3) % 32749  
    return (seed % 1000) / 1000 - 0.5

def forward_propagation(i1, i2):
    inputs = [i1, i2]
    
    weights = {
        'w1': random_weight(1), 'w2': random_weight(2), 
        'w3': random_weight(3), 'w4': random_weight(4),
        'w5': random_weight(5), 'w6': random_weight(6),
        'w7': random_weight(7), 'w8': random_weight(8)
    }
    
    biases = {'b1': 0.5, 'b2': 0.7}

    h1_input = (inputs[0] * weights['w1']) + (inputs[1] * weights['w3']) + biases['b1']
    h2_input = (inputs[0] * weights['w2']) + (inputs[1] * weights['w4']) + biases['b1']
    
    h1_output = tanh(h1_input)
    h2_output = tanh(h2_input)
    
    o1_input = (h1_output * weights['w5']) + (h2_output * weights['w7']) + biases['b2']
    o2_input = (h1_output * weights['w6']) + (h2_output * weights['w8']) + biases['b2']
    
    o1_output = tanh(o1_input)
    o2_output = tanh(o2_input)
    
    return o1_output, o2_output

output = forward_propagation(0.05, 0.10)
print("Output1:", output[0])
print("Output2:", output[1])













