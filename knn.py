import numpy


trainFile=open("yeast_training.txt")
testfile=open("yeast_test.txt")

train=[]
test=[]

for line in trainFile:
    train.append(line.strip().split())

for line in testfile:
    test.append(line.strip().split())

train=numpy.array(train).astype(float)
test=numpy.array(test).astype(float)

from KNNClass  import KNNClass

for i in range(1,10):
    clf=KNNClass()
    print("K value: ",i)
    predict = clf.knn(train, test, i)
    #for i in range(len(predict)):
        #print("Predict: ",predict[i]," ","Actual: ",test[i,-1])
    correctclassified=sum([test[i,-1]==predict[i] for i in range(len(predict))])
    numberofinst=len(test)
    print("Number of correctly classified instances : ",correctclassified)
    print("Total number of instances : ", numberofinst)
    print("Accuracy: ", float(correctclassified)/numberofinst)
    print("-----------------------------------------------------------------")
 
