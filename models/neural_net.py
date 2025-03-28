"""Neural network model."""

from typing import Sequence

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and output dimension C. 
    We train the network with a MLE loss function. The network uses a ReLU
    nonlinearity after each fully connected layer except for the last. 
    The outputs of the last fully-connected layer are passed through
    a sigmoid. 
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
        opt: str,
        loss_function: str = "mse",
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: output dimension C
            num_layers: Number of fully connected layers in the neural network
            opt: option for using "SGD" or "Adam" optimizer (Adam is Extra Credit)
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers
        self.opt = opt
        self.loss_function = loss_function 

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        
        if self.opt == "Adam":
            self.m = {}
            self.v = {}
            self.t = 0
        
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(sizes[i - 1], sizes[i]) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])
            
            # Adam optimizer parameters
            if self.opt == "Adam":
                self.m["W" + str(i)] = np.zeros((sizes[i - 1], sizes[i]))
                self.m["b" + str(i)] = np.zeros((sizes[i],))
                self.v["W" + str(i)] = np.zeros((sizes[i - 1], sizes[i]))
                self.v["b" + str(i)] = np.zeros((sizes[i],))

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output of the linear layer
        """
        return np.dot(X, W) + b
    
    def linear_grad(self, W: np.ndarray, X: np.ndarray, de_dz: np.ndarray) -> np.ndarray:
        """Gradient of linear layer.
        Parameters:
            W: the weight matrix
            X: the input data
            de_dz: the gradient of loss with respect to the layer output (dL/dz)
        Returns:
            de_dw, de_db, de_dx
            where
                de_dw: gradient of loss with respect to W
                de_db: gradient of loss with respect to b
                de_dx: gradient of loss with respect to X
        """
        de_dw = np.dot(X.T, de_dz)
        de_db = np.sum(de_dz, axis=0)
        de_dx = np.dot(de_dz, W.T)
        return de_dw, de_db, de_dx
    
    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output after applying ReLU
        """
        return np.maximum(0, X)
    
    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the gradient of ReLU with respect to X
        """
        # Derivative: 1 for positive X, 0 otherwise
        return np.where(X > 0, 1, 0)
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        # Numerically stable sigmoid:
        # For non-negative x, compute as 1 / (1 + exp(-x))
        # For negative x, compute as exp(x) / (1 + exp(x))
        return np.where(x >= 0,
                        1 / (1 + np.exp(-x)),
                        np.exp(x) / (1 + np.exp(x)))
    
    def sigmoid_grad(self, X: np.ndarray) -> np.ndarray:
        s = self.sigmoid(X)
        return s * (1 - s)
    
    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.mean((p - y) ** 2)
    
    def mse_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        n = y.shape[0]
        # return (2.0 / n) * (p - y)
        return (2.0 / (y.shape[0] * y.shape[1])) * (p - y)
    
    
    def huber_loss(self, y_true, y_pred, delta=1.0):
        diff = y_pred - y_true
        abs_diff = np.abs(diff)
        return np.mean(
            np.where(abs_diff <= delta,
                     0.5 * diff**2,
                     delta * (abs_diff - 0.5 * delta))
        )

    def huber_grad(self, y_true, y_pred, delta=1.0):
        diff = y_pred - y_true
        abs_diff = np.abs(diff)
        return np.where(abs_diff <= delta, diff, delta * np.sign(diff)) / y_true.size

    def mae_loss(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def mae_grad(self, y_true, y_pred):
        return np.sign(y_pred - y_true) / y_true.size


    
    def mse_sigmoid_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        y_pred = self.sigmoid(p)
        n = y.shape[0]
        dL_dy_pred = (2.0 / n) * (y_pred - y)
        dy_pred_dp = y_pred * (1 - y_pred)
        return dL_dy_pred * dy_pred_dp     

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the outputs for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C) 
        """
        self.outputs = {}
        # Store the output of each layer in
        # self.outputs as it will be used during back-propagation. The same keys as self.params and functions like
        # self.linear, self.relu, and self.mse can be used.
        
        for i in range(1, self.num_layers + 1):

            self.outputs["input_layer" + str(i)] = X
            
            W = self.params["W" + str(i)]
            b = self.params["b" + str(i)]
            
            # 1. Calculate linear layer:
            z = self.linear(W, X, b)
            self.outputs["linear_layer" + str(i)] = z # for backpropagation
            
            # 2. Calculate non-linear layer:
            if i < self.num_layers: # hidden layers
                X = self.relu(z)
                self.outputs["relu_layer" + str(i)] = X # for backpropagation
            
            else: # the last layer
                X = self.sigmoid(z)
                self.outputs["sigmoid_layer"] = X # for backpropagation
                
        return X

    def backward(self, y: np.ndarray) -> float:
        """Perform back-propagation and compute the gradients and losses.
        Parameters:
            y: training value targets
        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        pred = self.outputs["sigmoid_layer"]
        
        if self.loss_function.lower() == "mse":
            loss = self.mse(y, pred)
            dLoss_dPred = self.mse_grad(y, pred)
        elif self.loss_function.lower() == "huber":
            loss = self.huber_loss(y, pred, delta=1.0)
            dLoss_dPred = self.huber_grad(y, pred, delta=1.0)
        elif self.loss_function.lower() == "mae":
            loss = self.mae_loss(y, pred)
            dLoss_dPred = self.mae_grad(y, pred)
        else:
            raise ValueError(
                f"Unknown loss function '{self.loss_function}'. Choose 'mse', 'huber', or 'mae'."
            )
        # Store the gradient of each
        # parameter in self.gradients as it will be used when updating each
        # parameter and during numerical gradient checks. The same keys as self.params and functions like
        # self.linear_grad, self.relu_grad andself.softmax_grad can be used.
        
        # 1. Compute the MSE loss using the final output
        loss = self.mse(y, self.outputs["sigmoid_layer"])
        
        # 2. Compute gradients for each layer, starting from the last layer:
        for i in range(self.num_layers, 0, -1): 
            
            # 2.1. Compute activation gradient and update upstream gradient
            if i == self.num_layers:
                local = self.sigmoid_grad(self.outputs["linear_layer" + str(i)])  # dy/dz
                upstream = dLoss_dPred * local  #  chain rule: de/dz = de/dy (mse_gradient) * dy/dz (local)
            
            else:
                local = self.relu_grad(self.outputs["linear_layer" + str(i)])  # dy/dz
                upstream = upstream * local # chain rule: de/dz = de/dy (upstream from prev layer) * dy/dz (local)
            
            # 2.2. Compute weight, bias, and input gradientst
            de_dW, de_db, de_dX = self.linear_grad(self.params["W" + str(i)], 
                                                  self.outputs["input_layer" + str(i)],
                                                  upstream)
            
            # 2.3. Store computed gradients
            self.gradients["W" + str(i)] = de_dW
            self.gradients["b" + str(i)] = de_db
            
            # 2.4. Backpropagate gradient to previous layers
            upstream = de_dX
    
        return loss
        
    def update(
        self,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
        """
        if self.opt == 'SGD':
            # implement SGD optimizer here
            
            for key in self.params:
                self.params[key] -= lr * self.gradients[key]
        
        elif self.opt == 'Adam':
            # implement Adam optimizer here
            
            self.t += 1
            for key in self.params:
                # 1. Compute biased moment estimates:
                self.m[key] = b1 * self.m[key] + (1 - b1) * self.gradients[key]
                self.v[key] = b2 * self.v[key] + (1 - b2) * (self.gradients[key] ** 2)
                
                # 2. Compute bias-corrected estimates:
                m_hat = self.m[key] / (1 - b1 ** self.t)
                v_hat = self.v[key] / (1 - b2 ** self.t)
                
                # 3. Update model parameters:
                self.params[key] -= lr * m_hat / (np.sqrt(v_hat) + eps)
                
        else:
            raise NotImplementedError
        