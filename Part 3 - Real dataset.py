# ------- PART 3: Real dataset -------- 
import numpy as np
import pandas as pd
import math
import cvxpy as cp

df2 = pd.read_csv("input/sensor_readings_2.csv")

df2.columns = ['SD_front', 'SD_left', 'Label']
class_names = ['Move-Forward', 'Slight-Right-Turn', 'Sharp-Right-Turn', 'Slight-Left-Turn']
output_dictionary = {'Move-Forward': 1, 'Slight-Right-Turn': 2, 'Sharp-Right-Turn': 3, 'Slight-Left-Turn': 4}

df = df2

x1 = df['SD_front'].to_numpy() # first feature
x2 = df['SD_left'].to_numpy() # second feature
y  = df['Label'].replace(output_dictionary).to_numpy()  #class

# Train-test split (80% - 20%)
train_samples = np.int(np.around(x1.shape[0]*0.8))
x1_train = x1[0:train_samples]
x2_train = x2[0:train_samples]
y_train = y[0:train_samples]

x1_test = x1[train_samples:y.size]
x2_test = x2[train_samples:y.size]
y_test = y[train_samples:y.size]

n_iter = 500


#Split across data with L1 regularization
def svm(classe1,classe2):
    A = []
    n_rows_tot = y_train.size
    n_features = 2 + 1
    for i in range(n_rows_tot):
        if y_train[i] == classe1:
            ai_t = 1 * np.array([x1_train[i], x2_train[i], 1])
            A.append(ai_t)
        elif y_train[i] == classe2:
            ai_t = -1 * np.array([x1_train[i], x2_train[i], 1])
            A.append(ai_t)
    A = np.array(A)
        
    N = 5 # (<-> n. of agents)
    m = A[:,0].size
    n_samples = math.floor(m / N)
    rho = 1
    lamda = 0.1

    X = np.zeros((n_iter, N, n_features))
    Z = np.zeros((n_iter, n_features))
    U = np.zeros((n_iter, N, n_features))
    LOSS_1 = np.zeros(n_iter)

    mean_AX = np.zeros(n_samples)

    for k in range(0,n_iter-1,1):   
        #Step 1
        count = 0
        for i in range(N):        
            x_cp = cp.Variable(n_features)
            loss = cp.sum(cp.pos(np.ones(n_samples) - A[count:count+n_samples,:] @ x_cp))
            reg = cp.sum_squares(x_cp - Z[k,:] + U[k,i,:])
            aug_lagr = loss + (rho/2)*reg
            prob = cp.Problem(cp.Minimize(aug_lagr))
            prob.solve(solver=cp.ECOS)#verbose=True, adaptive_rho = False, 
            X[k+1,i,:] = x_cp.value

            #LOSS
            for j in range(n_samples):
                cost = 1 - np.inner(A[count+j,:], X[k+1,i,:])
                if cost >0:
                    LOSS_1[k+1] += cost
                LOSS_1[k+1] += rho/2 * np.linalg.norm(X[k+1,i,:] - Z[k,:] + U[k,i,:])**2
        
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
        Z[k+1,n_features-1] = mean_X[n_features-1] + mean_U[n_features-1]
    
    
        #Step 3
        for i in range(N):
            U[k+1,i,:] = U[k,i,:] + X[k+1,i,:] - Z[k+1,:] 

        
    print(Z[n_iter-1,:])

    return Z[n_iter-1,:];

def plot_train(beta_tilde,classe1,classe2,color1,color2):
    a = beta_tilde[0]
    b = beta_tilde[1]
    c = beta_tilde[2]
    print('-a/b',-a/b, '   -c/b',-c/b)

    ics = np.linspace(0,5,100)
    ipsilon = -a/b*ics - c/b
    e = []
    r = []
    t = []
    p = []
    for i in range(x1_train.size):
        if y_train[i] == classe1:
            e.append(x1_train[i])
            r.append(x2_train[i])
        elif y_train[i] == classe2:
            t.append(x1_train[i])
            p.append(x2_train[i])
    
    plt.plot(ics, ipsilon, '-k')
    plt.plot(e, r, color1, marker='o', linestyle="")
    plt.plot(t, p, color2, marker='o', linestyle="")


# ----------   TRAIN   --------------
beta_tilde_1 = svm(1,2)
beta_tilde_2 = svm(1,3)
beta_tilde_3 = svm(1,4)
beta_tilde_4 = svm(2,3)
beta_tilde_5 = svm(2,4)
beta_tilde_6 = svm(3,4)

import matplotlib.pyplot as plt
plt.figure(figsize = (12, 6))
plt.title('Plot of the lines associated with classifiers', fontsize=30)
plt.xlabel('x', fontsize=30)
plt.ylabel('y', fontsize=30).set_rotation(0)
plt.xlim(0.5, 3.5) 
plt.ylim(0, 1.5)

plot_train(beta_tilde_1,1,2,'-r','-g')
plot_train(beta_tilde_2,1,3,'-r','-b')
plot_train(beta_tilde_3,1,4,'-r','-m')
plot_train(beta_tilde_4,2,3,'-g','-b')
plot_train(beta_tilde_5,2,4,'-g','-m')
plot_train(beta_tilde_6,3,4,'-b','-m')


#plt.legend(loc='upper right', fontsize=20)
plt.grid()
plt.show()



# ----------   TEST   ---------------
from sklearn.metrics import confusion_matrix
# Plot confusion matrix for multiclass classification
def plot_confusion_matrix(y_test, y_pred, title, normalize=False):
    cm = confusion_matrix(y_test, y_pred)
    print(cm) # Confusion Matrix NOT normalized
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    classes = ['M-F', 'Sl-R-T', 'Sh-R-T', 'Sl-L-T']
    #classes = ['1', '2', '3', '4']
    plt.xticks(np.arange(cm.shape[1]), classes)
    plt.yticks(np.arange(cm.shape[0]), classes)
    ax.set(
        xticklabels=classes, yticklabels=classes,
        title=title,
        ylabel='True label',
        xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    
    
y_pred = np.zeros(y_test.size,dtype='int')    
for i in range(y_test.size):
    pred_count = np.zeros(7) # the ith cell i identifies the ith classifier
    #Classifier 1
    a = beta_tilde_1[0]
    b = beta_tilde_1[1]
    c = beta_tilde_1[2]
    if x1_test[i]*a + x2_test[i]*b + c > 0: # scalar product
        pred_count[1] += 1
    else:
        pred_count[2] += 1
        
    #Classifier 2
    a = beta_tilde_2[0]
    b = beta_tilde_2[1]
    c = beta_tilde_2[2]
    if x1_test[i]*a + x2_test[i]*b + c > 0:
        pred_count[1] += 1
    else:
        pred_count[3] += 1
        
    #Classifier 3
    a = beta_tilde_3[0]
    b = beta_tilde_3[1]
    c = beta_tilde_3[2]
    if x1_test[i]*a + x2_test[i]*b + c > 0:
        pred_count[1] += 1
    else:
        pred_count[4] += 1
        
    #Classifier 4
    a = beta_tilde_4[0]
    b = beta_tilde_4[1]
    c = beta_tilde_4[2]
    if x1_test[i]*a + x2_test[i]*b + c > 0:
        pred_count[2] += 1
    else:
        pred_count[3] += 1
        
    #Classifier 5
    a = beta_tilde_5[0]
    b = beta_tilde_5[1]
    c = beta_tilde_5[2]
    if x1_test[i]*a + x2_test[i]*b + c > 0:
        pred_count[2] += 1
    else:
        pred_count[4] += 1
        
    #Classifier 6
    a = beta_tilde_6[0]
    b = beta_tilde_6[1]
    c = beta_tilde_6[2]
    if x1_test[i]*a + x2_test[i]*b + c > 0:
        pred_count[3] += 1
    else:
        pred_count[4] += 1
    
    #print(pred_count)
    #print(np.argmax(pred_count))
    y_pred[i] = np.argmax(pred_count) # majority (the index with more count is the class most likely)
    
plot_confusion_matrix(y_test, y_pred, 'Confusion matrix', normalize=False)