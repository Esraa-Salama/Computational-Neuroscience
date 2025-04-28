def exp(x):
    e = 2.718281828459045
    return e ** x

def tanh(x):
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))

def dot_product(a, b):
    return sum(a_i * b_i for a_i, b_i in zip(a, b))

def matrix_multiply(A, B):
    return [[dot_product(row, col) for col in zip(*B)] for row in A]

def transpose(matrix):
    return list(zip(*matrix))

def softmax(x):
    exps = [exp(i) for i in x]
    sum_exps = sum(exps)
    return [i/sum_exps for i in exps]

def random():
    x = 1
    for _ in range(5):
        x = (x * 1103515245 + 12345) & 0x7fffffff
    return x / 2147483647

text = "Lily, After All This Time?"
words = text.split()
vocab = list(sorted(set(words)))
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}
vocab_size = len(vocab)


class SimpleRNN:
    def __init__(self, vocab_size, hidden_size):
        self.hidden_size = hidden_size
        
        # تهيئة الأوزان
        self.Wxh = [[random()*0.01 for _ in range(vocab_size)] for _ in range(hidden_size)]
        self.Whh = [[random()*0.01 for _ in range(hidden_size)] for _ in range(hidden_size)]
        self.Why = [[random()*0.01 for _ in range(hidden_size)] for _ in range(vocab_size)]
        self.bh = [0.0 for _ in range(hidden_size)]
        self.by = [0.0 for _ in range(vocab_size)]
    
    def forward(self, inputs, h_prev):
        xs, hs, ys = {}, {}, {}
        hs[-1] = h_prev.copy()
        
        for t in range(len(inputs)):
            xs[t] = [0.0]*vocab_size
            xs[t][word_to_idx[inputs[t]]] = 1.0
            
            h_raw = [
                sum(self.Wxh[i][j] * xs[t][j] for j in range(vocab_size)) +
                sum(self.Whh[i][j] * hs[t-1][j] for j in range(self.hidden_size)) +
                self.bh[i]
                for i in range(self.hidden_size)
            ]
            hs[t] = [tanh(h) for h in h_raw]
            
            ys[t] = [
                sum(self.Why[i][j] * hs[t][j] for j in range(self.hidden_size)) +
                self.by[i]
                for i in range(vocab_size)
            ]
        
        return xs, hs, ys
    
    def train(self, inputs, target_word, learning_rate):
        target = [1.0 if word == target_word else 0.0 for word in vocab]
        h_prev = [0.0]*self.hidden_size
        xs, hs, ys = self.forward(inputs, h_prev)
        

        dWxh = [[0.0]*vocab_size for _ in range(self.hidden_size)]
        dWhh = [[0.0]*self.hidden_size for _ in range(self.hidden_size)]
        dWhy = [[0.0]*self.hidden_size for _ in range(vocab_size)]
        dbh = [0.0]*self.hidden_size
        dby = [0.0]*vocab_size
        dh_next = [0.0]*self.hidden_size
        
        t = len(inputs)-1  
        dy = softmax(ys[t])
        dy[word_to_idx[target_word]] -= 1.0
        
        for i in range(vocab_size):
            for j in range(self.hidden_size):
                dWhy[i][j] += dy[i] * hs[t][j]
            dby[i] += dy[i]
        
        dh = [
            sum(self.Why[j][i] * dy[j] for j in range(vocab_size)) +
            dh_next[i]
            for i in range(self.hidden_size)
        ]
        
        dh_raw = [dh[i] * (1 - hs[t][i]**2) for i in range(self.hidden_size)]
        
        for i in range(self.hidden_size):
            for j in range(vocab_size):
                dWxh[i][j] += dh_raw[i] * xs[t][j]
            for j in range(self.hidden_size):
                dWhh[i][j] += dh_raw[i] * hs[t-1][j]
            dbh[i] += dh_raw[i]
        
        for i in range(self.hidden_size):
            for j in range(vocab_size):
                self.Wxh[i][j] -= learning_rate * dWxh[i][j]
            for j in range(self.hidden_size):
                self.Whh[i][j] -= learning_rate * dWhh[i][j]
            self.bh[i] -= learning_rate * dbh[i]
        
        for i in range(vocab_size):
            for j in range(self.hidden_size):
                self.Why[i][j] -= learning_rate * dWhy[i][j]
            self.by[i] -= learning_rate * dby[i]


hidden_size = 10
rnn = SimpleRNN(vocab_size, hidden_size)

for epoch in range(1000):
    rnn.train(["Lily,", "After", "All"], "This", learning_rate=0.1*(1-epoch/1000))
    
    if epoch % 100 == 0:
        _, _, ys = rnn.forward(["Lily,", "After", "All"], [0.0]*hidden_size)
        probs = softmax(ys[2])
        predicted_idx = max(range(vocab_size), key=lambda i: probs[i])
        predicted_word = idx_to_word[predicted_idx]
        print(f"Epoch {epoch}: Predicted '{predicted_word}'")


_, _, ys = rnn.forward(["Lily,", "After", "All"], [0.0]*hidden_size)
probs = softmax(ys[2])
predicted_idx = max(range(vocab_size), key=lambda i: probs[i])
predicted_word = idx_to_word[predicted_idx]

print(f"\nFinal prediction: '{predicted_word}'")







