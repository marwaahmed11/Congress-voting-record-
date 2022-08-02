import math
from statistics import mode


def euclidean_distance(x1,x2):
        return math.sqrt((sum(pow((x1-x2),2))))
    
class KNNClass:
   
    def knn(self,train,test,k):
        predicted_labels=[]
        for ts in test :           
          distances=[]
          k_nearest_labels=[]
          for tr in train :
            distances.append([euclidean_distance(ts[:-1],tr[:-1]),tr[-1]])
          distances=sorted(distances)
          for i in range(k) :
              k_nearest_labels.append(distances[i][1])
          predicted_labels.append(mode(k_nearest_labels))
        return predicted_labels
