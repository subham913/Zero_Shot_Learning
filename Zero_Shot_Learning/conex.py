import numpy as np

#Data Loading
X_seen=np.load("X_seen.npy",encoding="latin1")
X_test=np.load("Xtest.npy",encoding="latin1")
Y_test=np.load("Ytest.npy",encoding="latin1")
c_seen=np.load("class_attributes_seen.npy",encoding="latin1")
c_unseen=np.load("class_attributes_unseen.npy",encoding="latin1")

#Computing Mean of 40 seen classes
mu_seen=[]
for i in range(X_seen.shape[0]):
    m=np.mean(X_seen[i],axis=0)
    mu_seen.append(m);
mu_seen=np.asarray(mu_seen)

#Computing Similarity and then mean of unseen classes
s_c=np.dot(c_unseen,np.transpose(c_seen))
normalizer=np.sum(s_c,axis=1).reshape(-1,1)
s_c=s_c/normalizer
mu_unseen=np.dot(s_c,mu_seen)

#Testing the Classifier
Y_pred=[]
for i in range (X_test.shape[0]):
    dist=[]
    for j in range (mu_unseen.shape[0]):
        temp=np.linalg.norm(mu_unseen[j]-X_test[i])
        dist.append(temp)
    dist=np.asarray(dist)
    min=np.argmin(dist)
    Y_pred.append(min+1)
Y_pred=np.asarray(Y_pred).reshape(-1,1)
accuracy=(Y_pred==Y_test).sum()/Y_test.shape[0]
print("accuracy on unseen classes with Method 1 is "+str(accuracy))
