import statistics
import pydotplus
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz


names = ['result', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
votes = pd.read_csv("house-votes-84.csv", names=names)

# print(len(votes))
# print(votes.shape)

def decision_tree(votes, tsize):
    data=[]
    data = votes.values[:, 1:17]
    value = votes.values[:, 0]
    """print(votes.shape)
    print(data.shape)
    print(value.shape)"""

    majority = []  
    for j in range(16):
            y = 0
            n = 0 
            for i in range(435):
                if data[i, j] == 'y':
                    data[i, j] = 1
                    y += 1
                if data[i, j] == 'n': 
                    data[i, j] = 2    
                    n += 1
            if (y>n):
                majority.append(1)
            else:
                majority.append(2)

    for j in range(16):
        
            for i in range(435):
                if data[i, j] == '?':
                    if majority[j] == 1:
                        data[i, j] = 1
                    
                    if majority[j] == 2:
                        data[i, j] = 2



    #for i in range(len(data)):
    #  print("output :",data[i])

    data_train, data_test, value_train, value_test = train_test_split(data, value, train_size=tsize)
    dtree = DecisionTreeClassifier(criterion="entropy")
    dtree.fit(data_train, value_train)
    nodecount = dtree.tree_.node_count
    depth = dtree.get_depth()
    predict = dtree.predict(data_test)
    accuracy = accuracy_score(value_test, predict) * 100

    return  accuracy, nodecount , depth ,dtree

temp = decision_tree(votes, 25 / 100)    
print("training 25% = ",temp[:-1])

Meanaccuracies=[]
Meannode=[]
Meandeepth=[]
minAccuracies=[]
minNode=[]
minDeepth=[]
maxAccuracies=[]
maxNode=[]
maxDeepth=[]

tsize=[]

tree=0
dtree=0

minAcc=1000
maxAcc=0

minNo=1000
maxNo=0

minDepth=1000
maxDepth=0
for i in range(30, 71,10): 
        
    accuracies2 = []
    node2 = []
    deepth2=[]
    for j in range(5):
        
        dummy = decision_tree(votes, i / 100)
        accuracies2.append(dummy[0])
        node2.append(dummy[1])
        deepth2.append(dummy[2])
        
        if dummy[0] > maxAcc:
            maxAcc = dummy[0]
            tree=dummy[2]
            dtree=dummy[3]
            
        if dummy[1] > maxNo:
            maxNo = dummy[1]
            
        if dummy[2] > maxDepth:
            maxDepth = dummy[2]
            
        if dummy[0] < maxAcc:
            minAcc = dummy[0]
            
        if dummy[1] < maxNo:
            minNo = dummy[1]
            
        if dummy[2] < maxDepth:
            minDepth = dummy[2]
    Meanaccuracies.append(statistics.mean(accuracies2))
    Meannode.append(statistics.mean(node2))
    Meandeepth.append(statistics.mean(deepth2))

    maxAccuracies.append(maxAcc)
    maxNode.append(maxNo)
    maxDeepth.append(maxDepth)

    minAccuracies.append(minAcc)
    minNode.append(minNo)
    minDeepth.append(minDepth)

    tsize.append(i)


   
print("Meanaccuracies: ",Meanaccuracies)
print("Meannode: ",Meannode)
print("Meandeepth: ",Meandeepth)

print("minAccuracies: ",minAccuracies)
print("minNode: ",minNode)
print("minDeepth: ",minDeepth)

print("maxAccuracies: ",maxAccuracies)
print("maxNode: ",maxNode)
print("maxDeepth: ",maxDeepth)

fig,ax=plt.subplots(figsize=(7,5))
ax.plot(tsize,Meanaccuracies)
ax.set_xlabel('tsize Number')
ax.set_ylabel('Accuracy')
ax.set_title(' tsize vs Accuracy')
plt.show()
fig,ax=plt.subplots(figsize=(7,5))
ax.plot(tsize,Meannode)
ax.set_xlabel('tsize Number')
ax.set_ylabel('node')
ax.set_title(' tsize vs node')
plt.show()

fig,ax=plt.subplots(figsize=(7,5))
ax.plot(tsize, Meandeepth)
ax.set_xlabel('tsize Number')
ax.set_ylabel('depth')
ax.set_title(' tsize vs depth')
plt.show()




dot_data = export_graphviz(dtree, out_file=None,
                               filled=True, rounded=True,
                               special_characters=True, feature_names=names[1:],
                               class_names=['democrat', 'republican'])

graph =pydotplus.graph_from_dot_data(dot_data)
graph.write_png('mydecisiontree.png')

img=pltimg.imread('mydecisiontree.png')
imgplot = plt.imshow(img)
plt.show()
