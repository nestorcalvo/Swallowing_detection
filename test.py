import numpy as np
results = {'Fold 1':{'best_params':{'C': 0.01, 'gamma': 0.001, 'kernel': 'linear'}},
           'Fold 2':{'best_params':{'C': 0.01, 'gamma': 0.001, 'kernel': 'linear'}},
           'Fold 3':{'best_params':{'C': 0.001, 'gamma': 0.001, 'kernel': 'linear'}},
           'Fold 4':{'best_params':{'C': 0.001, 'gamma': 0.001, 'kernel': 'linear'}},
           'Fold 5':{'best_params':{'C': 0.001, 'gamma': 0.001, 'kernel': 'linear'}}}

C_array = []
gamma_array = []
kernel_array = []
for key in results:
    C_array.append(results[key]['best_params']['C'])
    gamma_array.append(results[key]['best_params']['gamma'])
    kernel_array.append(results[key]['best_params']['kernel'])
    print(results[key]['best_params'])

print(max(set(C_array), key=C_array.count))