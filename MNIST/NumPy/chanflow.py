import numpy as np

class utils:    
  def softmax(self, X_in):
    # For numerical stability, shift the layer values by their maximum
    X_in_shifted = X_in - np.max(X_in, axis=1, keepdims=True)
    X_out = np.exp(X_in_shifted) / np.sum(np.exp(X_in_shifted), axis=1, keepdims=True)
    return X_out

  def accuracy(self, y_label, y_pred):
    accuracy = np.mean(
        np.argmax(y_label, axis=1) == np.argmax(y_pred, axis=1)
    )
    return accuracy

  def cross_entropy_loss(self, logits, labels):      
      dX_out = self.softmax(logits) - labels
      return dX_out
  

utils = utils()

class layers:      
  class dense:
    def __init__(self, layer_sizes, init_weight='xavier', init_bias='zeros'):
      # Initialise weight and bias
      if init_weight == 'xavier':
        self.W = np.random.normal(
            size=layer_sizes,
            scale=2./sum(layer_sizes)
        )

      if init_bias == 'zeros':
        self.B = np.zeros(layer_sizes[1])
            
      # Group them in an array for convenience
      self.W_and_B = [self.W, self.B]

    def forward(self, X_in):
      # Store the incoming matrix for back propagation later
      self.X_in = X_in
      # Update the weights and biases
      self.W = self.W_and_B[0]
      self.B = self.W_and_B[1]

      # Calculate forward propagation
      X_out = np.matmul(self.X_in, self.W) + self.B
      return X_out

    def backward(self, dX_out):
      # Calculate weight and bias update values
      dW = np.matmul(self.X_in.T, dX_out)
      dB = np.sum(dX_out, axis=0)
      
      # Error to be passed to the next back propagation layer
      dX_in = np.matmul(dX_out, self.W.T)

      dW_and_dB = [dW, dB]
      return dX_in, dW_and_dB
    
  class relu:
    def __init__(self):
      self.W_and_B = []

    def forward(self, X_in):
      self.X_in = X_in
      X_out = np.maximum(X_in, 0)
      return X_out

    def backward(self, dX_out):
      dX_in = np.multiply(dX_out, self.X_in>=0)
      return dX_in, []
      

layers = layers()

class nn:
  def __init__(self, layers, loss_f):
    # layers is a layer of the defined layer classes
    self.layers = layers
    self.loss_f = loss_f

  def forward(self, X):
    for layer in self.layers:
      X = layer.forward(X)
    return X

  def backward(self, dX_out):
    dWs_and_dBs = []
    for layer in reversed(self.layers):
      dX_out, dW_and_dB = layer.backward(dX_out)
      dWs_and_dBs.append(dW_and_dB)
    return dWs_and_dBs

  def train_step(self, X_in, y, learning_rate):
    # Propagate forwards
    X_out = self.forward(X_in)
    
    # Calculate the loss and the difference
    dX_out = self.loss_f(X_out, y)

    # Propogate backwards and calculate the updates
    dWs_and_dBs = self.backward(dX_out)

    # Update weights and biases
    for n in range(len(self.layers)):
      for m in range(len(self.layers[n].W_and_B)):
        self.layers[n].W_and_B[m] -= np.multiply(learning_rate, dWs_and_dBs[len(self.layers)-n-1][m])

    return dX_out
  
  def predict(self, X_in):
    preds = utils.softmax(self.forward(X_in))
    return preds