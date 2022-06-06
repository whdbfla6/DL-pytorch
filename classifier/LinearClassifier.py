import torch

class LinearClassifier:
    def __init__(self):
        self.W = None

    def forward(
            self,
            X:torch,
            y:torch,
            lr: float = 0.01,
            reg : float = 1e-5,
            num_iters: int = 100
            ):
        n,c = X.shape
        num_class = len(torch.unique(y))
        if self.W == None: self.W = torch.rand((c,num_class))*0.001
        
        for _ in range(num_iters):
            loss,dW = self.loss(X,y,reg)
            self.W -= lr*dW

    def pred(
        self,
        X,
        ):

        pred = torch.mm(X,self.W)
        pred = torch.argmax(pred,axis=1)
        return pred

class LinearSVM(LinearClassifier):
    def __init__(self):
        super().__init__()
    def loss(
        self,
        X,
        y,
        reg : float = 1e-5,
        ):

        out = torch.mm(X,self.W)
        n,c = out.shape
        sy = out[range(n),y].reshape((-1,1)).expand((-1,c))

        margin = out-sy+1
        margin[range(n),y] = 0
        margin[margin<0] = 0
        loss = torch.sum(margin)/n + reg*torch.sum(self.W.mul(self.W))

        margin_copy = margin.clone()
        margin_copy[margin_copy>0] = 1
        margin_copy[range(n),y] = -torch.sum(margin_copy,axis=1)
        dW = torch.mm(X.T,margin_copy)/n
        dW += 2*reg*self.W
        return loss,dW

class Softmax(LinearClassifier):
    def __init__(self):
        super().__init__()
    def loss(
        self,
        X,
        y,
        reg : float = 1e-5,
        ):

        out = torch.mm(X,self.W)
        n,c = out.shape
        out_ex = torch.exp(out)
        den = torch.sum(out_ex,dim=1).reshape((-1,1)).expand((-1,c))
        prob = out_ex/den
        
        logprob = -torch.log(prob[range(n),y])
        loss = torch.sum(logprob)/n + reg*torch.sum(self.W.mul(self.W))

        prob[range(n),y] -= 1
        dW = torch.mm(X.T,prob)/n
        dW += 2*reg*self.W

        return loss,dW