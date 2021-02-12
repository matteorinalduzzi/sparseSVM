# -------- PART 2: Comparison between centralized code and distributed code ------
import numpy as np
import math
np.random.seed(1)

n = 20
m = 1000
beta_true = np.random.randn(n,1)
offset = np.random.randn(1)
X = np.random.normal(0, 5, size=(m,n))
Y = np.sign(X.dot(beta_true) + offset)

beta = np.append(beta_true, offset).reshape(21)
#print('True', np.around(beta,6))

# Centralized algorithm
import cvxpy as cp
A = np.hstack((X*Y,Y))
n_features = n + 1
# Parameters
rho = 1
lamda = 0.5
C = np.identity(n_features)
C[n_features-1,n_features-1] = 0

beta = cp.Variable((n_features,1))
loss = cp.sum(cp.pos(1 - A @ beta ))
reg = cp.norm(C@beta, 1)
prob = cp.Problem(cp.Minimize(loss/m + lamda*reg))

# Solving
prob.solve()
var = beta.value
var = var.reshape((var.shape[0],))
print('Centralized solution')
print(np.around(var,6))
#print(np.around(var/np.linalg.norm(var),6))


# Distributed algorithm
N = 20 # n. of agentss, every agents has a small piece of the dataset
n_iter = 500 # n. of iterations
n_samples = math.floor(A.shape[0] / N) # dataset division betweeen the N agents

X = np.zeros((n_iter, N, n_features))  # X[k,i,:] is the vector x_i at the iteration k
Z = np.zeros((n_iter, n_features))     # Z[k,:] is the vector z at the iteration k
U = np.zeros((n_iter, N, n_features))  # U[k,i,:] is the vector u_i at the iteration k
LOSS_1 = np.zeros((n_iter, N))         # vector that holds the trend of the loss function of the step 1 for the N agents

for k in range(0,n_iter-1,1): #start from k+1=1 and not from 0    
    #Step 1
    count = 0
    for i in range(N):        
        x_cp = cp.Variable(n_features)
        loss = cp.sum(cp.pos(np.ones(n_samples) - A[count:count+n_samples,:] @ x_cp))
        reg = cp.sum_squares(x_cp - Z[k,:] + U[k,i,:])
        aug_lagr = loss/m + (rho/2)*reg
        prob = cp.Problem(cp.Minimize(aug_lagr))
        prob.solve(solver=cp.ECOS) #verbose=True, adaptive_rho = False, 
        X[k+1,i,:] = x_cp.value
        # LOSS computation
        for j in range(n_samples):
            cost = 1 - np.inner(A[count+j,:], X[k+1,i,:])
            if cost >0:
                LOSS_1[k+1,i] += cost
        LOSS_1[k+1,i] += rho/2 * np.linalg.norm(X[k+1,i,:] - Z[k,:] + U[k,i,:])**2
        
        count += n_samples
    
    
    #Step 2
    mean_X = np.zeros(n_features)
    mean_U = np.zeros(n_features)
    for i in range(N):
        mean_X += X[k+1,i,:]
        mean_U += U[k,i,:]
    mean_X = 1/N * mean_X
    mean_U = 1/N * mean_U
    
    for i in range(n_features-1):
        if mean_X[i] + mean_U[i] > lamda/(N*rho):
            Z[k+1,i] = mean_X[i] + mean_U[i] - lamda/(N*rho)
        elif mean_X[i] + mean_U[i] < - lamda/(N*rho):
            Z[k+1,i] = mean_X[i] + mean_U[i] + lamda/(N*rho)
        else:
            Z[k+1,i] = 0
    Z[k+1,n_features-1] = mean_X[n_features-1] + mean_U[n_features-1] #l'ultima è un caso particolare
    
    
    #Step 3
    for i in range(N):
        U[k+1,i,:] = U[k,i,:] + X[k+1,i,:] - Z[k+1,:] 

        
print('Distributed solution')
print(Z[n_iter-1,:])


# Plot of the LOSS of step 1 for one of the N agents 
import matplotlib.pyplot as plt
plt.plot(np.linspace(0,n_iter,n_iter), LOSS_1[:,0])
plt.ylabel("LOSS", fontsize=16)
plt.xlabel("n° iterations", fontsize=16)
plt.title("Loss trend")
plt.show()

plt.plot(np.linspace(50,n_iter,450), LOSS_1[50:500,0])
plt.ylabel("LOSS", fontsize=16)
plt.xlabel("n° iterations", fontsize=16)
plt.title("Zoom loss trend")
plt.show()