import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from model_cp import ESN, Output, Reservoir
import a_weight
from scipy.spatial import distance
import scipy.stats as st
import seaborn as sns
import pandas as pd

fs = 44000
t = 1 # time length of input
nSegment = 2 # number of segment, repetition
lSegment = 1000

T = fs*t//nSegment  # data length
buffer = 10 # buffer for prediction length

# reservoir variables
N_x =500  # nodes of the reservoir
rho_list = np.array([0.1, 0.9, 1.0, 1.1, 1.2, 1.3,1.4,1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
learning_rate = 10**(-7)
density = 0.1
leaking_rate = 1
beta = 0.01 #0.05 # L2 parameter
eta = 10**(-1)
noise_level = 0.01 # reservoir noise
rds_transition = lSegment//5
input_scale = 1

train_num = 1
trial_num = 5
pred_num = 50
seed =0
ESN_seed = 0
delay = 1  # prediction time delay

outputs_N_hebb = cp.empty((len(rho_list), pred_num, nSegment*lSegment))
outputs_RN_hebb = cp.copy(outputs_N_hebb)
outputs_RefN_hebb = cp.copy(outputs_N_hebb)
outputs_RefRN_hebb = cp.copy(outputs_N_hebb)

outputs_N_nohebb = cp.copy(outputs_N_hebb)
outputs_RN_nohebb = cp.copy(outputs_N_hebb)
outputs_RefN_nohebb = cp.copy(outputs_N_hebb)
outputs_RefRN_nohebb = cp.copy(outputs_N_hebb)

NRMSE_N_hebb = cp.empty((len(rho_list), pred_num, nSegment*lSegment))
NRMSE_RN_hebb = cp.copy(NRMSE_N_hebb)
NRMSE_RefN_hebb = cp.copy(NRMSE_N_hebb)
NRMSE_RefRN_hebb = cp.copy(NRMSE_N_hebb)

NRMSE_N_nohebb = cp.empty((len(rho_list), pred_num, nSegment*lSegment))
NRMSE_RN_nohebb = cp.copy(NRMSE_N_nohebb)
NRMSE_RefN_nohebb = cp.copy(NRMSE_N_nohebb)
NRMSE_RefRN_nohebb = cp.copy(NRMSE_N_nohebb)

RDS_N_hebb = cp.empty((len(rho_list), pred_num, rds_transition))
RDS_RefN_hebb = cp.copy(RDS_N_hebb)
RDS_RN_hebb = cp.copy(RDS_N_hebb)
RDS_RefRN_hebb = cp.copy(RDS_N_hebb)

RDS_N_nohebb = cp.empty((len(rho_list), pred_num,rds_transition))
RDS_RefN_nohebb = cp.copy(RDS_N_nohebb)
RDS_RN_nohebb = cp.copy(RDS_N_nohebb)
RDS_RefRN_nohebb = cp.copy(RDS_N_nohebb)

corr_N_hebb = cp.empty((len(rho_list), pred_num,1))
corr_RefN_hebb = cp.empty((len(rho_list), pred_num,1))
corr_RN_hebb = cp.empty((len(rho_list), pred_num,1))
corr_RefRN_hebb = cp.empty((len(rho_list), pred_num,1))

corr_N_nohebb = cp.empty((len(rho_list), pred_num,1))
corr_RefN_nohebb = cp.empty((len(rho_list), pred_num,1))
corr_RN_nohebb = cp.empty((len(rho_list), pred_num,1))
corr_RefRN_nohebb = cp.empty((len(rho_list), pred_num,1))

def getN(length):
    seg = cp.random.randn(length*nSegment*T,1)
    long = cp.concatenate((seg[0:T*length], seg, seg[T*length::]))
    u_N_long = a_weight.A_weighting(long, fs)
    u_N_raw = u_N_long[T*length:(nSegment+1)*T]
    
    u_N_down = cp.concatenate((u_N_raw[::fs/(nSegment*lSegment)], cp.random.randn(buffer,1)))
    d_N = cp.empty((nSegment*lSegment, 1))

    for t in range(nSegment*lSegment):
        d_N[t] = u_N_down[t+delay]  # delayed N
    
    u_N = u_N_down[0:lSegment*nSegment]
    
    return u_N, d_N

def getRN(length):
    u_RN_segment = cp.random.randn(length*T,1)
    u_RN_raw = cp.concatenate((u_RN_segment, u_RN_segment, u_RN_segment, u_RN_segment))
    u_RN_raw = a_weight.A_weighting(u_RN_raw, fs)
    u_RN_raw = u_RN_raw[T*length:(nSegment+1)*T]
    u_RN_down = cp.concatenate((u_RN_raw[::fs/(nSegment*lSegment)], cp.random.randn(buffer,1)))
    d_RN = cp.empty((nSegment*lSegment, 1))
    
    for t in range(nSegment*lSegment):
        d_RN[t] = u_RN_down[t+delay]  # delayed RN
    
    u_RN = u_RN_down[0:nSegment*lSegment]
    
    return u_RN, d_RN

def sp_rad(model):
    W = cp.asnumpy(model.Reservoir.getW())
    eigv = np.linalg.eig(W)[0]
    sp_rad = np.max(np.abs(eigv))
    return sp_rad

cp.random.seed(seed)
np.random.seed(seed)
u = cp.zeros((lSegment*nSegment, trial_num*4))
d = cp.zeros((lSegment*nSegment, trial_num*4))
        
u_RefN,d_RefN = getN(1)
u_RefRN, d_RefRN = getRN(1)
                     

for ntri in range(trial_num):
    u_N, d_N = getN(1)
    u[:, ntri*4] = u_N.flatten()
    d[:, ntri*4] = d_N.flatten()
    u_RN, d_RN = getRN(1)
    u[:, ntri*4+1] = u_RN.flatten()
    d[:, ntri*4+1] = d_RN.flatten()
            
    u[:, ntri*4+2] = u_RefN.flatten()
    d[:, ntri*4+2] = d_RefN.flatten()
    u[:, ntri*4+3] = u_RefRN.flatten()
    d[:, ntri*4+3] = d_RefRN.flatten()
        
# pseudo-randomize
p = np.array([ 7, 11,  6, 15, 18,  4, 12,  2, 13,  1, 16,  0, 14, 19,  8, 10,  5,
        3, 17,  9])
        
u = u.T[p]
d = d.T[p]


u_Ntest, d_Ntest = getN(1)
u_RNtest, d_RNtest = getRN(1)
        
train_U = u[0,:].reshape(-1, 1)
train_D = d[0,:].reshape(-1, 1)

sp_hebb=np.zeros(len(rho_list))
sp_nohebb = np.zeros(len(rho_list))

########################################################################################
# 1. Generate Ns, RNs, one RefN and one RefRN
# 2. Train reservoir to predict one step ahead time series data of input signals above
# 3. Test and plot the performance of trained reservoir with RefRN,RefN, new N, new RN stim.
##########################################################################################

for irho in range(len(rho_list)):
    

    rho_r = rho_list[irho]

    model_hebb = ESN(train_U.shape[1], train_D.shape[1], N_x, density=density, 
                input_scale=input_scale, rho=rho_r, noise_level = noise_level, leaking_rate = leaking_rate, learning_rate = learning_rate, seed = ESN_seed)
    model_nohebb = ESN(train_U.shape[1], train_D.shape[1], N_x, density=density, 
                input_scale=input_scale, rho=rho_r, noise_level = noise_level, leaking_rate = leaking_rate, learning_rate = 0, seed = ESN_seed)

    for ntri in range(len(u)):
        train_U = u[ntri,:].reshape(-1,1)
        train_D = d[ntri,:].reshape(-1,1)
            
        model_hebb.train_mini(train_U, train_D, eta, beta)
        model_nohebb.train_mini(train_U, train_D, eta, beta)
    
    # get spectral radius after learning
    sp_hebb[irho] = sp_rad(model_hebb)
    sp_nohebb[irho] = sp_rad(model_nohebb)
    np.save(f'./sp_rad_hebb{N_x}', sp_hebb)
    np.save(f'./{N_x}', sp_nohebb)
        
    for ipred in range(pred_num):
        test_U_N = u_Ntest.reshape(-1, 1)
        test_U_RefN = u_RefN.reshape(-1,1)
        test_U_RN = u_RNtest.reshape(-1,1)
        test_U_RefRN = u_RefRN.reshape(-1,1)

        # Hebbian
        test_Y_N_hebb = model_hebb.predict(test_U_N)
        model_hebb.predict(cp.random.randn(10,1))
            
        test_Y_RefN_hebb = model_hebb.predict(test_U_RefN)
        model_hebb.predict(cp.random.randn(10,1))
            
        test_Y_RN_hebb = model_hebb.predict(test_U_RN)
        model_hebb.predict(cp.random.randn(10,1))
            
        test_Y_RefRN_hebb = model_hebb.predict(test_U_RefRN)
        model_hebb.predict(cp.random.randn(10,1))
            
        # Non-hebbian
        test_Y_N_nohebb = model_nohebb.predict(test_U_N)
        model_nohebb.predict(cp.random.randn(10,1))
            
        test_Y_RefN_nohebb = model_nohebb.predict(test_U_RefN)
        model_nohebb.predict(cp.random.randn(10,1))
            
        test_Y_RN_nohebb = model_nohebb.predict(test_U_RN)
        model_nohebb.predict(cp.random.randn(10,1))
            
        test_Y_RefRN_nohebb = model_nohebb.predict(test_U_RefRN)
        model_nohebb.predict(cp.random.randn(10,1))

        test_D_N = d_Ntest.reshape(-1, 1)
        test_D_RefN = d_RefN.reshape(-1, 1)
        test_D_RN = d_RNtest.reshape(-1, 1)
        test_D_RefRN = d_RefRN.reshape(-1, 1)

        outputs_N_hebb[irho, ipred,:] = test_Y_N_hebb
        outputs_RefN_hebb[irho, ipred,:] = test_Y_RefN_hebb
        outputs_RN_hebb[irho, ipred,:] = test_Y_RN_hebb
        outputs_RefRN_hebb[irho, ipred,:] = test_Y_RefRN_hebb

        outputs_N_nohebb[irho, ipred,:] = test_Y_N_nohebb
        outputs_RefN_nohebb[irho, ipred,:] = test_Y_RefN_nohebb
        outputs_RN_nohebb[irho, ipred,:] = test_Y_RN_nohebb
        outputs_RefRN_nohebb[irho, ipred,:] = test_Y_RefRN_nohebb

data_hebb = {
    'N' : outputs_N_hebb,
    'RN' : outputs_RN_hebb,
    'RefN' : outputs_RefN_hebb,
    'RefRN' : outputs_RefRN_hebb
}

long_data_hebb = []
for category, tensor in data_hebb.items():
    for plane_idx, plane in enumerate(tensor):
        for row_idx, row in enumerate(plane):
            for col_idx, value in enumerate(row):
                long_data_hebb.append({
                    'category': category,
                    'rho_idx': rho_list[plane_idx],
                    'pred_idx': row_idx,
                    'output_idx': col_idx,
                    'value': value
                })

df_hebb = pd.DataFrame(long_data_hebb)


csv_file_hebb = f"data_{pred_num}_hebb.csv"
df_hebb.to_csv(csv_file_hebb, index=False)


data_nohebb = {
    'N' : outputs_N_nohebb,
    'RN' : outputs_RN_nohebb,
    'RefN' : outputs_RefN_nohebb,
    'RefRN' : outputs_RefRN_nohebb
}


long_data_nohebb = []
for category, tensor in data_nohebb.items():
    for plane_idx, plane in enumerate(tensor):
        for row_idx, row in enumerate(plane):
            for col_idx, value in enumerate(row):
                long_data_nohebb.append({
                    'category': category,
                    'rho_idx': rho_list[plane_idx],
                    'pred_idx': row_idx,
                    'output_idx': col_idx,
                    'value': value
                })

df_nohebb = pd.DataFrame(long_data_nohebb)


csv_file_nohebb = f"data_{pred_num}_nohebb.csv"
df_nohebb.to_csv(csv_file_nohebb, index=False)


variables = {
    'fs': fs,
    't' : t,
    'nSegment' : nSegment,
    'lSegment' : lSegment,
    'T' : T,
    'buffer' : buffer,
    'N_x' : N_x,
    'learning_rate' : learning_rate,
    'density' : density,
    'leaking_rate' : leaking_rate,
    'beta' : beta,
    'eta' : eta,
    'noise_level' : noise_level,
    'rds_transition' : rds_transition,
    'input_scale' : input_scale,
    'train_num' : train_num,
    'trial_num' : trial_num,
    'pred_num' : pred_num,
    'seed' : seed,
    'ESN_seed' : ESN_seed,
    'delay' : delay,
    'rho_list' : rho_list
}


df_variable = pd.DataFrame(variables)


csv_file_variable = f"variables_{pred_num}.csv"
df_variable.to_csv(csv_file_variable, index=False)

df_test = {
    'N': cp.asnumpy(test_D_N.flatten()),
    'RN' : cp.asnumpy(test_D_RN.flatten()),
    'RefN' : cp.asnumpy(test_D_RefN.flatten()),
    'RefRN' :cp.asnumpy(test_D_RefRN.flatten())
}


df_test = pd.DataFrame(df_test)


csv_file_test = f"test_{pred_num}.csv"
df_test.to_csv(csv_file_test, index=False)

