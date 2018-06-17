#importing the Libraries
import numpy as np
import matplotlib.pyplot as plt 

#importing datasets 
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST Original')

#Creating Test and Train Sets 
from sklearn.model_selection import train_test_split
train_img,test_img,train_lbl,test_lbl = train_test_split(mnist.data , mnist.target, test_size = 0.3 , random_state = 0)

# Vizualizing the data 
plt.figure()
for index ,(label , image) in enumerate(zip(train_lbl[0:6] ,
		    train_img[0:6])):
	plt.subplot(2,3,index+1)
	plt.imshow(np.reshape(image , (28,28)), cmap= plt.cm.gray)
	plt.title("Training:"+str(label))

#Logistic Regression Model From Sklear 
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver = 'lbfgs')
logreg.fit(train_img,train_lbl)

#predicting The Output & Calculating the Score
y_predict = logreg.predict(test_img)
score = logreg.score(test_img , test_lbl)
print("Score in Percent: "+str(score*100)+"%")

#Confusion Matrix
from sklearn import metrics
confmat = metrics.confusion_matrix(test_lbl , y_predict)
print("Confusion Matrix: "+str(confmat))

#Finding the mis-predicted index 
index = 0
list_index = []
for index,(label , predict) in enumerate(zip(test_lbl ,y_predict)):
	if label != predict :
		list_index.append(index)
		index += 1

#Vizualizing the mis-predicted data 
plt.figure()
for a ,index in enumerate(list_index[0:6]):
	plt.subplot(2,3,a+1)
	plt.imshow(np.reshape(test_img[index],(28,28)),cmap = plt.cm.gray)
	plt.title("Actual: "+str(test_lbl[index])+"\nPredicted: "+str(y_predict[index]),fontsize = 12)

