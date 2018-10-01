# Description
This tiny project used the prediction of 10 unseen classes given the other 40 seen classes in the training data.Also the attributes were given for each class.
# Datasets
The dataset can be found on this link [a link](https://drive.google.com/open?id=1AgQP9WL1SuxJOa3Jkab-JohYLLFZO32g).It has class attributes of seen and unseen classes along with Xseen.npy for training and Xtest.npy,Ytest.npy for testisng.Note that the test set only contain unseen target
# Models
The first method(convex.py) models the mean of unseen classes as convex combination of seen classes where the cofficients are chosen as similarity calculted using attributes.
The second method(regress.py) trains a linear model by considering the class attribute as input features and mean being the output(Sort of multioutput regression).
In both the cases after getting the mean can simply use prototype based classifier to predict the class.
# Requirements
1.Python3
2.numpy
