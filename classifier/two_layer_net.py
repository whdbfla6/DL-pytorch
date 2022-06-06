import torch


def sample_batch(
        X: torch.tensor,
        y: torch.tensor,
        batch_size: int
        ):
    n_samples = X.shape[0]
    ind = torch.randint(low=0, high=n_samples, size = (batch_size,))
    return X[ind], y[ind]

class TwoNeuralNet:
    def __init__(
            self,
            input_size : int,
            hidden_size : int,
            output_size : int,
            dtype = torch.float32,
            std : float = 1e-4
            ):
        self.params = {}
        self.params['W1'] = std * torch.randn(input_size, hidden_size, dtype=dtype)
        self.params['b1'] = std * torch.randn(hidden_size, dtype=dtype)
        
        self.params['W2'] = std * torch.randn(hidden_size, output_size, dtype=dtype)
        self.params['b2'] = std * torch.randn(output_size, dtype=dtype)

    def nn_forward_pass(
        self,
        X):

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        hidden = torch.mm(X,W1) + b1
        hidden[hidden<0] = 0
        out = torch.mm(hidden,W2) + b2 

        return out, hidden

    def nn_forward_backward(
        self,
        X,
        y,
        reg: float = 5e-6,
        ):

        out, hidden = self.nn_forward_pass(X)
        N,_ = X.shape
        W1,W2 = self.params['W1'], self.params['W2']

        scores = torch.exp(out)
        denom = torch.sum(scores,axis=1).reshape((-1,1)).expand((-1,scores.shape[1]))
        prob = scores/denom
        loss = -torch.log(prob)
        loss = torch.sum(loss[range(N),y])/N + reg*torch.sum(torch.mul(W1,W1)) + reg*torch.sum(torch.mul(W2,W2))

        prob[range(N),y] -= 1

        self.grads = {}
        self.grads['W2'] = torch.mm(hidden.T,prob)/N + 2*reg*W2
        self.grads['b2'] = torch.sum(prob,axis=0)/N


        dhidden = torch.mm(prob,W2.T)/N 
        dhidden[hidden<0] = 0
        self.grads['W1'] = torch.mm(X.T,dhidden)/N + 2*reg*W1
        self.grads['b1'] = torch.sum(dhidden,axis=0)/N

        return loss

    def train(
        self,
        X: torch.tensor,
        y: torch.tensor,
        X_val: torch.tensor,
        y_val: torch.tensor,
        learning_rate: float = 1e-3,
        learning_rate_decay: float = 0.95,
        reg: float = 5e-6,
        num_iters: int = 100,
        batch_size: int = 1,
        ): 
        for i in range(num_iters):
            X_batch, y_batch = sample_batch(X,y,batch_size)
            loss = self.nn_forward_backward(X_batch,y_batch,reg)
            if i%100 == 0:
                print(f'[{i}/{num_iters}] training loss : {loss}')
                val_loss = self.nn_forward_backward(X_val,y_val,reg)
                print(f'[{i}/{num_iters}] validation loss : {val_loss}')
            for param in self.params:
                self.params[param] -= learning_rate*self.grads[param]

    def pred(
        self,
        X,
        ):

        pred,_ = self.nn_forward_pass(X)
        pred = torch.argmax(pred,axis=1)
        return pred
