# -*- coding: utf-8 -*-
"""
Created on Sat Apr 02 22:02:12 2016

@author: YI
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    

N = 1000 #the number of each class
D = 2 #dimentions
K = 3 #classes
X = np.zeros((N*K,D)) 
y = np.zeros(N*K, dtype='uint8') 
for j in xrange(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j

plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)

Num_examples=len(X)
input_dim=2
output_dim=3
learningrate = 0.001
reg = 0.001 #lambda in the regularization

def loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions
    n1 = X.dot(W1) + b1
    a1 = np.maximum(0, n1)   #ReLU
    n2 = a1.dot(W2) + b2
    exp_scores = np.exp(n2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss 
    logprobs = -np.log(probs[range(Num_examples), y])  #Softmax
    data_loss = np.sum(logprobs)
    # Add regulatization term to loss to avoid overfitting
    data_loss += reg /2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1.0 /Num_examples * data_loss
    
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    n1 = x.dot(W1) + b1
    a1 = np.maximum(0, n1)   
    n2 = a1.dot(W2) + b2
    exp_scores = np.exp(n2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)
    

def build_model(hdim, num_iterations=30000, print_loss=False):
    
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(input_dim, hdim) * np.sqrt(2/input_dim)  #for ReLu
    b1 = np.zeros((1, hdim))
    W2 = np.random.randn(hdim, output_dim) * np.sqrt(2/hdim)
    b2 = np.zeros((1, output_dim))

    # This is what we return at the end
    model = {}
    
    # Gradient descent. For each batch...
    for i in xrange(0, num_iterations):

        # Forward propagation
        n1 = X.dot(W1) + b1
        a1 = np.maximum(0, n1)
        scores = a1.dot(W2) + b2
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Backpropagation
        dscores = probs
        dscores[range(Num_examples), y] -= 1
        dW2 = (a1.T).dot(dscores)
        db2 = np.sum(dscores, axis=0, keepdims=True)
        dhidden = np.dot(dscores,W2.T)
        dhidden[a1 <= 0]=0
        dW1 = np.dot(X.T, dhidden)
        db1 = np.sum(dhidden, axis=0, keepdims=True)

        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += reg * W2
        dW1 += reg * W1

        # Gradient descent parameter update
        W1 += -learningrate * dW1
        b1 += -learningrate * db1
        W2 += -learningrate * dW2
        b2 += -learningrate * db2
        
        # Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        
        
        if print_loss and i % 1000 == 0:
          print "Loss after iteration %i: %f" %(i, loss(model))
    
    return model
    
model = build_model(5, print_loss=True)

# Plot the decision boundary
plot_decision_boundary(lambda x: predict(model, x))
plt.title("Decision Boundary for hidden layer size 3")

prediction=predict(model,X)
print 'training accuracy: %.5f' % (np.mean(prediction==y))