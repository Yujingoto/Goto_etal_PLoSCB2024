#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
# A plastic RNN based on ESN written by Yujin Goto, last edited 2024.04.22.
# This code is inspired by codes from the book "リザバーコンピューティング" by Gohei Tanaka, Ryosho Nakane, and Akira Hirose.
# ISBN: 978-4-627-85531-1
################################################################################


import numpy as np
import cupy as cp
import networkx as nx

# Identity map
def identity(x):
    return x


# Input layer
class Input:
    # Initialize
    def __init__(self, N_u, N_x, input_scale, seed=0):
        '''
        param N_u: dim (node) of the input layer
        param N_x: nodes of the reservoir
        param input_scale: scaling input signal
        '''
        
        cp.random.seed(seed=seed) # uniform dist.
        self.Win = cp.random.uniform(-input_scale, input_scale, (N_x, N_u))

    # Input-reservoir weight
    def __call__(self, u):
        '''
        param u: vector with N_u diment.
        return:  vector with N_x diment.
        '''
        
        return cp.dot(self.Win, u)


# In[33]:


# Reservoir
class Reservoir:
    # initialize recurrent network's connection matrix
    def __init__(self, N_x, density, rho, activation_func, leaking_rate, learning_rate,
                 seed):
        '''
        param N_x: node of a reservoir
        param density: connection density of the reservoir
        param rho: the spectral radius
        param activation_func: activation function
        param leaking_rate: the leak rate for the leaky integrator model
        param learning_rate: learning rate for the Oja's hebbian synaptic plasticity
        '''
        self.seed = seed
        self.N_x = N_x
        self.W = self.make_connection(N_x, density, rho)
        self.x = cp.zeros(N_x)
        self.activation_func = activation_func
        self.alpha = leaking_rate
        self.learning_rate = learning_rate
        self.W_diff = cp.zeros_like(self.W)
        self.W_temp = cp.zeros_like(self.W)

    # Generate the initial connection
    def make_connection(self, N_x, density, rho):
        # Erdos-Renyi random graph
        m = int(N_x*(N_x-1)*density/2)  # Num of connections
        G = nx.gnm_random_graph(N_x, m, self.seed)

        # convert to a matrix
        connection = nx.to_numpy_matrix(G)
        W = cp.array(connection)

        # generate non-zero connections from uniform dist.
        rec_scale = 1.0
        cp.random.seed(seed=self.seed)
        W *= cp.random.uniform(-rec_scale, rec_scale, (N_x, N_x))
        #W *= cp.random.standard_cauchy((N_x, N_x)) # Cauchy dist
        
        # estimate the spectral radius
        W = cp.asnumpy(W)
        eigv_list = np.linalg.eig(W)[0]
        sp_radius = np.max(np.abs(eigv_list))

        # scaling the spectral radius to the desired value: rho
        W *= rho / sp_radius

        return cp.asarray(W)

    # update
    def __call__(self, x_in, noise_level):
        # kernel func for Oja
        
        oja = cp.ElementwiseKernel(
            in_params = 'raw float64 W, raw float64 x, int16 width, float64 learning_rate',
            out_params = 'raw float64 W_diff',
            operation=\
            '''
            int x_idx = i%width; 
            int y_idx = i/width;
    
            W_diff[i] = learning_rate * (x[y_idx]*x[x_idx] - x[y_idx]*x[y_idx]*W[i]);
            ''',
            name='oja')
        
        '''
        param x_in: Input tiem series
        return x: state vector of the reservoir after updating
        return W: connection matrix of the reservoir after updating
        '''
      
        # Updating the reservoir's state
        self.x = (1.0 - self.alpha) * self.x                  + self.alpha * self.activation_func(cp.dot(self.W, self.x)                  + x_in)
        self.x += noise_level * cp.random.randn(self.N_x)
                
        self.W += oja(self.W, self.x, self.W.shape[0], self.learning_rate, self.W_diff, size=(self.W.shape[0] * self.W.shape[1]))
        
        return self.x, self.W
        
    # if hebbian plasticity is not required (classic ESN)    
    def nohebb_call(self, x_in, noise_level):
        self.x = (1.0 - self.alpha) * self.x                  + self.alpha * self.activation_func(cp.dot(self.W, self.x)                  + x_in)
        self.x += noise_level * cp.random.randn(self.N_x)
        
        return self.x

    # Initialize reservoir's state vector
    def reset_reservoir_state(self):
        self.x *= 0.0
        
    # Get connection matrix for some uses    
    def getW(self):
        return self.W
        
        
    # output the spectral radius
    def spectrad(self):
        self.W = cp.asnumpy(self.W)
        eigv_list = np.linalg.eig(self.W)[0]
        sp_radius = np.max(np.abs(eigv_list))
        
        return cp.asarray(sp_radius)
        


# Output layer
class Output:
    # Init
    def __init__(self, N_x, N_y, seed=0):
        '''
        param N_x: Node num of the reservoir
        param N_y: node num of the output layer
        '''
        # Normal dist.
        cp.random.seed(seed=seed)
        self.Wout = cp.random.normal(size=(N_y, N_x))

    def __call__(self, x):
        '''
        param x: vector of N_x dim
        return: vector of N_y dim
        '''
        return cp.dot(self.Wout, x)

    # Set trained output joint weight matrices.
    def setweight(self, Wout_opt):
        self.Wout = Wout_opt
    
    # update the weight reservoir-output
    def updateweight(self, grad, eta, beta):
        '''
        grad: gradient of the output weight
        eta: learning rate
        beta: Normalization param
        '''
        self.Wout -= eta * (grad + beta * self.Wout)
    
    def getWout(self):
        ave = cp.mean(cp.abs(self.Wout))
        return self.Wout, ave

# Echo state network
class ESN:
    # Initialize layers
    def __init__(self, N_u, N_y, N_x, density=0.05, input_scale=1.0,
                 rho=0.95, activation_func=cp.tanh, noise_level = 0.1, leaking_rate=1.0,
                 learning_rate=0.0001, output_func=identity, inv_output_func=identity, seed=0):
        '''
        param N_u: input dim
        param N_y: output dim
        param N_x: num of reservoir nodes
        param density: connection density of the reservoir
        param input_scale: scaling for input
        param rho: the spectral radius of the reservoir
        param activation_func: activation function of the reservoir nodes
        param leaking_rate: leak rate
        param output_func: output map (default: identity）
        param noise_level: internal noise level
        '''
        self.Input = Input(N_u, N_x, input_scale)
        self.Reservoir = Reservoir(N_x, density, rho, activation_func, 
                                   leaking_rate, learning_rate, seed)
        self.Output = Output(N_x, N_y)
        self.N_u = N_u
        self.N_y = N_y
        self.N_x = N_x
        self.y_prev = cp.zeros(N_y)
        self.output_func = output_func
        self.noise_level = noise_level

    
    # Minibatch-learning
    def train_mini(self, U, D, eta, beta, trans_len = None):
        '''
        U: input teaching data, N_u
        D: output teaching data, N_y
        '''
        
        U = cp.asarray(U)
        train_len = len(U)
        
        D = cp.asarray(D).reshape(train_len,1)
        
        self.eta = eta # learning rate
        self.beta = beta
        
        if trans_len is None:
            trans_len = 0
        Y = cp.array([])
        X = cp.array([])

        # update
        for n in range(train_len):
            x_in = self.Input(U[n])

            # Reservoir state vector
            # add noise of noise_level
            values = self.Reservoir(x_in, self.noise_level)
            x = values[0]
            W = values[1]
            
            X = cp.append(X, x)

            # model output before learning
            y = self.Output(x)
            Y = cp.append(Y, self.output_func(y))

        # output weight matrix after learning
        Y = Y.reshape(train_len,1)
        X = X.reshape(len(X)//train_len, train_len)
        
        #diff = (Y-D)/ train_len
        diff = cp.abs(Y-D) / train_len
        #E = np.std(diff*diff.T) #L2 norm
        self.grad = cp.dot(X, diff).reshape(1,len(X))
        
        self.Output.updateweight(self.grad, self.eta, self.beta)
        
        return X, Y

    
    # for hebb only
    def hebbonly(self, U, D,trans_len = None):

        U = cp.asarray(U)
        D = cp.asarray(D)
        
        train_len = len(U)
        if trans_len is None:
            trans_len = 0
        Y = []


        for n in range(train_len):
            x_in = self.Input(U[n])
            x_in = x_in.reshape(self.N_x, self.N_u)
            #x_in += self.noise_level*cp.random.randn(self.N_x, self.N_u)
            x_in = x_in.flatten()


            values = self.Reservoir(x_in, self.noise_level)
            x = values[0]
            W = values[1]


        return x,W

    # prediction after batch learning
    def predict(self, U):
        U = cp.asarray(U)
        
        test_len = len(U)
        Y_pred = cp.zeros(0)

        # update
        for n in range(test_len):
            x_in = self.Input(U[n])
            x_in = x_in.reshape(self.N_x, self.N_u)
            x_in = x_in.flatten()

            x = self.Reservoir.nohebb_call(x_in, self.noise_level)

            y_pred = self.Output(x)
            Y_pred = cp.append(Y_pred, self.output_func(y_pred))
            self.y_prev = y_pred

        return Y_pred
    
    # prediction after batch learning for only-reservoir condition
    def predict_res(self, U):
        U = cp.asarray(U)
        
        test_len = len(U)
        X_pred = cp.zeros((self.N_x, test_len))


        for n in range(test_len):
            x_in = self.Input(U[n])
            x_in = x_in.reshape(self.N_x, self.N_u)

            x_in = x_in.flatten()


            x = self.Reservoir.nohebb_call(x_in, self.noise_level)
            X_pred[:,n] = x

        return X_pred
    
    def spectrad(self):
        radius = self.Reservoir.spectrad()
        return radius
