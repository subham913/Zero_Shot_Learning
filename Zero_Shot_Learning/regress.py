import numpy as np

#Data Loading
X_seen=np.load("X_seen.npy",encoding="latin1")
X_test=np.load("Xtest.npy",encoding="latin1")
Y_test=np.load("Ytest.npy",encoding="latin1")
c_seen=np.load("class_attributes_seen.npy",encoding="latin1")
c_unseen=np.load("class_attributes_unseen.npy",encoding="latin1")

#Computing mean of 40 seen classes
mu_seen=[]
for i in range(X_seen.shape[0]):
    m=np.mean(X_seen[i],axis=0)
    mu_seen.append(m);
mu_seen=np.asarray(mu_seen)

identity=np.eye(c_seen.shape[1])#identity matrix

regularizer=[0.01,0.1,1,10,20,50,100]#different values of Regularizer

#Calculating Weight Matrix with different values of Regularizer
W=[]
for lamda in regularizer:
    w=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(c_seen),c_seen)+lamda*identity),np.transpose(c_seen)),mu_seen)
    W.append(w)

#Computing mean of unseen classes with different values of Weight Matrix found above
mu_unseen=[]
for w in W:
    mu=np.dot(c_unseen,w)
    mu_unseen.append(mu)

#Testing the Classifier for different values of Weight Matrix found above
Y_pred_list=[]
for mu in mu_unseen:
    Y_pred=[]
    for i in range(X_test.shape[0]):
        dist=[]
        for j in range(mu.shape[0]):
            temp=np.linalg.norm(mu[j]-X_test[i])
            dist.append(temp)
        dist=np.asarray(dist)
        min=np.argmin(dist)
        Y_pred.append(min+1)
    Y_pred=np.asarray(Y_pred).reshape(-1,1)
    Y_pred_list.append(Y_pred)
acc=[]
for Y_pred in Y_pred_list:
    accuracy=(Y_pred==Y_test).sum()/Y_test.shape[0]
    acc.append(accuracy)
acc=np.asarray(acc)
for i,lamda in enumerate(regularizer):
    print("accuracy with lambda = "+str(lamda)+" is "+str(acc[i]))

max=np.argmax(acc)
print("Maximum accuracy is "+str(acc[max])+" with lambda = " + str(regularizer[max]))
