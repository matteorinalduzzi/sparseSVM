# ------ PART 1: Norm L1 effect -------
import numpy as np
np.random.seed(1)

# Data generation
n = 20 #n. of features
m = 1000 #n. of examples
beta_true = np.random.randn(n,1) # hyperplane coefficients (beta)
offset = np.random.randn(1) # intercept (beta_0)

# I generate a set of linearly separable data from the hyperplane identified by (beta_true, offset)
X = np.random.normal(0, 5, size=(m,n))
Y = np.sign(X.dot(beta_true) + offset)


# Setting of the centralized problem
import cvxpy as cp
beta = cp.Variable((n,1))
v = cp.Variable()
loss = cp.sum(cp.pos(1 - cp.multiply(Y, X @ beta + v)))
reg = cp.norm(beta, 1)
lambd = cp.Parameter(nonneg=True)
prob = cp.Problem(cp.Minimize(loss/m + lambd*reg))


# Problem solved for different values of lambda
TRIALS = 100
lambda_vals = np.logspace(-2, 0, TRIALS) # lambda values from 0.01 to 1
#lambda_vals = np.linspace(0.01, 1, TRIALS) # from 0.01 to 1
beta_vals = []
for i in range(TRIALS):
    lambd.value = lambda_vals[i]
    prob.solve()
    beta_vals.append(beta.value)



# Plot beta trend as a function of lambda, normalized by the smallest coefficient
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'svg'
beta_vals_norm = np.copy(beta_vals)
for i in range(len(beta_vals)):
    if np.max(beta_vals[i]) > abs(np.min(beta_vals[i])):
        beta_vals_norm[i] = beta_vals[i]/np.max(beta_vals[i])
    else:
        beta_vals_norm[i] = beta_vals[i]/abs(np.min(beta_vals[i]))

for i in range(n):
    plt.plot(lambda_vals, [wi[i,0] for wi in beta_vals_norm])
plt.ylabel(r"$\beta$", fontsize=16).set_rotation(0)
plt.xlabel(r"$\lambda$", fontsize=16)
plt.xscale("log")
