# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 15:27:20 2021

@author: Kaike Sa Teles Rocha Alves
@email: kaike.alves@engenharia.ufjf.br
"""
# Importing libraries
import pandas as pd
import numpy as np

class KRLS_T:
    def __init__(self, M = 30, sigma = 0.5, lambda1 = 0.999, epsilon = 1e-6, sn2 = 1e-2):
        #self.hyperparameters = pd.DataFrame({})
        self.parameters = pd.DataFrame(columns = ['K', 'Q', 'Sigma', 'mu', 'm', 'Dict'])
        # Computing the output in the training phase
        self.OutputTrainingPhase = np.array([])
        # Computing the residual square in the ttraining phase
        self.ResidualTrainingPhase = np.array([])
        # Computing the output in the testing phase
        self.OutputTestPhase = np.array([])
        # Computing the residual square in the testing phase
        self.ResidualTestPhase = np.array([])
        # Hyperparameters and parameters
        # Size of the dictionary
        self.M = M
        self.sigma = sigma
        self.lambda1 = lambda1
        # Jitter noise
        self.epsilon = epsilon
        # (Sigma_n)^2
        self.sn2 = sn2
         
    def fit(self, X, y):
        
        # Compute the number of samples
        n = X.shape[0]
        
        # Initialize the first input-output pair
        x0 = X[0,].reshape(-1,1)
        y0 = y[0]
        
        # Initialize the consequent parameters
        self.Initialize_KRLS_T(x0, y0)

        for k in range(1, n):
            # if k == 6908:
            #     print(k)
            # Prepare the k-th input vector
            x = X[k,].reshape((1,-1)).T
                      
            # Update the consequent parameters
            kn1 = self.KRLS_T(x, y[k])
            
            # Compute the output
            Output = ( self.parameters.loc[0, 'Q'] @ kn1 ).T @ self.parameters.loc[0,  'mu']
            
            # Store the results
            self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output )
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(y[k]) - Output )
        return self.OutputTrainingPhase
            
    def predict(self, X):

        for k in range(X.shape[0]):
            
            # Prepare the first input vector
            x = X[k,].reshape((1,-1)).T

            # Compute k
            kn1 = np.array(())
            for ni in range(self.parameters.loc[0, 'Dict'].shape[1]):
                kn1 = np.append(kn1, [self.Kernel(self.parameters.loc[0, 'Dict'][:,ni].reshape(-1,1), x)])
            kn1 = kn1.reshape(-1,1)
            # Compute the output
            Output = ( self.parameters.loc[0, 'Q'] @ kn1 ).T @ self.parameters.loc[0, 'mu']
            # Storing the output
            self.OutputTestPhase = np.append(self.OutputTestPhase, Output )

        return self.OutputTestPhase

    def Kernel(self, x1, x2):
        k = np.exp( - ( 1/2 ) * ( (np.linalg.norm( x1 - x2 ))**2 ) / ( self.sigma**2 ) )
        return k
    
    def Initialize_KRLS_T(self, x, y):
        k = self.Kernel(x, x)
        K = np.eye(1) * (k + self.epsilon)
        Q = np.eye(1) / (k + self.epsilon)
        Sigma = np.eye(1) * k - k**2 / ( k + self.sn2 )
        mu = np.eye(1) * y * k / ( k + self.sn2 )
        NewRow = pd.DataFrame([[K, Q, Sigma, mu, 1., x]], columns = ['K', 'Q', 'Sigma', 'mu', 'm', 'Dict'])
        self.parameters = pd.concat([self.parameters, NewRow], ignore_index=True)
        # Initialize first output and residual
        self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, y)
        self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase, 0.)
        
    def KRLS_T(self, x, y):
        i = 0
        # Update Sigma and mu
        self.parameters.at[i, 'Sigma'] = self.lambda1 * self.parameters.loc[i, 'Sigma'] + ( 1 - self.lambda1 ) * self.parameters.loc[i, 'K']
        #self.parameters.at[i,  'Sigma'] = ( 1 / self.lambda1 ) * self.parameters.loc[i,  'Sigma']
        self.parameters.at[i, 'mu'] = np.sqrt( self.lambda1 ) * self.parameters.loc[i,  'mu']
        #self.parameters.at[i,  'mu'] = self.parameters.loc[i,  'mu']
        # Save the Dict and update it
        Dict_Prov = np.hstack([self.parameters.loc[i,  'Dict'], x])
        # Compute k
        k = np.array(())
        for ni in range(Dict_Prov.shape[1]):
            k = np.append(k, [self.Kernel(Dict_Prov[:,ni].reshape(-1,1), x)])
        kt = k[:-1].reshape(-1,1)
        ktt = self.Kernel(x, x)
        # Compute q, gamma^2, h, sf2, y_mean
        q = self.parameters.loc[i,  'Q'] @ kt
        gamma2 = ktt - kt.T @ q
        h = self.parameters.loc[i, 'Sigma'] @ q
        sf2 = gamma2 + q.T @ h
        y_mean = q.T @ self.parameters.loc[i,  'mu']
        sy2 = self.sn2 + sf2
        # Increase Sigma and mu
        p = np.vstack([h, sf2])
        self.parameters.at[i, 'Sigma'] = np.lib.pad(self.parameters.loc[i, 'Sigma'], ((0,1),(0,1)), 'constant', constant_values=(0))
        sizeSigma = self.parameters.loc[i,  'Sigma'].shape[0] - 1
        self.parameters.at[i, 'Sigma'][sizeSigma,sizeSigma] = sf2
        self.parameters.at[i, 'Sigma'][0:sizeSigma,sizeSigma] = h.flatten()
        self.parameters.at[i, 'Sigma'][sizeSigma,0:sizeSigma] = h.flatten()
        self.parameters.at[i, 'Sigma'] = self.parameters.loc[i,  'Sigma'] - ( 1/sy2 ) * p @ p.T
        self.parameters.at[i, 'mu'] = np.vstack([self.parameters.loc[i,  'mu'], y_mean])
        self.parameters.at[i, 'mu'] = self.parameters.loc[i,  'mu'] + ( ( y - y_mean ) / sy2 ) * p
        if gamma2 <= self.epsilon/10:
            self.parameters.at[i, 'Sigma'] = self.parameters.at[i, 'Sigma'][0:sizeSigma,0:sizeSigma]
            self.parameters.at[i, 'mu'] = np.delete(self.parameters.loc[i,  'mu'], sizeSigma, 0)
            # Update k
            k = np.delete(k, sizeSigma)
            #print("Numerical roundoff error too high, you should increase epsilon noise")
        else:
            # Save K and update it
            K_old = self.parameters.loc[i, 'K']
            self.parameters.at[i, 'K'] = np.lib.pad(self.parameters.loc[i, 'K'], ((0,1),(0,1)), 'constant', constant_values=(0))
            sizeK = self.parameters.loc[i,  'K'].shape[0] - 1
            self.parameters.at[i, 'K'][sizeK,sizeK] = ktt
            self.parameters.at[i, 'K'][0:sizeK,sizeK] = kt.flatten()
            self.parameters.at[i, 'K'][sizeK,0:sizeK] = kt.flatten()
            # Save Q and update it
            Q_old = self.parameters.at[i, 'Q']
            self.parameters.at[i, 'Q'] = np.lib.pad(self.parameters.loc[i, 'Q'], ((0,1),(0,1)), 'constant', constant_values=(0))
            p1 = np.vstack([q, -1])
            self.parameters.at[i, 'Q'] = self.parameters.loc[i, 'Q'] - ( 1/gamma2 ) * p1 @ p1.T
            if Dict_Prov.shape[1] > self.M:
                diag = np.diagonal(self.parameters.loc[i, 'Q'])
                errors = np.square( np.divide( self.parameters.loc[i, 'Q'] @ self.parameters.loc[i, 'mu'], diag.reshape(-1,1) ) )
                ind = errors.argmin()
                if ind == Dict_Prov.shape[1] - 1:
                    self.parameters.at[i, 'K'] = K_old
                    self.parameters.at[i, 'Q'] = Q_old
                    # Indexes
                    idx = np.arange(Dict_Prov.shape[1])
                    noind = np.delete(idx, ind)
                else:
                    # Indexes to keep
                    idx = np.arange(Dict_Prov.shape[1])
                    noind = np.delete(idx, ind)
                    # Reduce K
                    self.parameters.at[i, 'K'] = self.parameters.loc[0, 'K'][noind, :][:, noind]
                    # Reduce Q
                    Q = self.parameters.loc[0, 'Q'][noind, :][:, noind]
                    Qs = self.parameters.loc[i, 'Q'][noind, ind].reshape(-1,1)
                    QsT = self.parameters.loc[i, 'Q'][ind,noind].reshape(1,-1)
                    qs = self.parameters.loc[i, 'Q'][ind,ind]
                    self.parameters.at[i, 'Q'] = Q - ( Qs @ QsT ) / qs
                    # Update the Dictionary
                    self.parameters.at[i, 'Dict'] = Dict_Prov
                    self.parameters.at[i, 'Dict'] = np.delete(self.parameters.loc[i, 'Dict'], ind, 1)
                # Reduce Sigma and mu
                self.parameters.at[i, 'Sigma'] = self.parameters.loc[0, 'Sigma'][noind, :][:, noind]
                self.parameters.at[i, 'mu'] = np.delete(self.parameters.loc[i, 'mu'], ind, 0)
                # Update k
                k = np.delete(k, ind)
            else:
                # Update the Dictionary
                self.parameters.at[i, 'Dict'] = Dict_Prov
            
        return k.reshape(-1,1)
