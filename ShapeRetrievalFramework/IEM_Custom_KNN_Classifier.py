import math
import pandas as pd
import random
from sklearn import preprocessing
import time

def getMajorityGroup(list1, k, n): 
    vote = n*[0]

    for i in range(k): 
        min1 = list1[0]
        
        for j in range(1, len(list1)):     
            if list1[j][0] < min1[0]: 
                min1 = list1[j]; 
        print(min1)        
        list1.remove(min1); 
        vote[min1[1]] = vote[min1[1]] + 1
    
    confidence = max(vote)     
    majorityGroup = vote.index(confidence)
    
    #print("MajorityGroup = ", majorityGroup, " confidence = ", confidence)

    return majorityGroup, confidence

def euclidean_distance(feat_one, feat_two):

    squared_distance = 0

    #Assuming correct input to the function where the lengths of two features are the same

    for i in range(len(feat_one)):
        squared_distance += (feat_one[i] - feat_two[i])**2

    ed = math.sqrt(squared_distance)

    return ed;

def similarity_score(feat_one, feat_two):

    score = 0.0
    v = 4
    i = 0

    while i < len(feat_one):
        for i2 in range(0, v):
            score += abs(feat_one[i] - feat_two[i])/v
            i += 1
        v += 2
      

    return score;

def customKNN_predict(training_set, to_predict, k, n):
    if len(training_set) >= k:
        print("K cannot be smaller than the total voting groups(ie. number of training data points)")
        return
    
    distributions = []
    for group in training_set:
        for features in training_set[group]:
            #distance = euclidean_distance(features, to_predict)
            distance = similarity_score(features, to_predict)
            distributions.append([distance, group])
    
    majorityGroup, confidence = getMajorityGroup(distributions, k, n)
    print("MajorityGroup = ", majorityGroup, " Confidence = ", confidence)
    
    return majorityGroup, confidence

def customKNN_train(training_set, test_set, k, n, className):
    accurate_predictions = 0
    total_predictions = 0
    for group in test_set:
            for data in test_set[group]:
                predicted_class, confidence = customKNN_predict(training_set, data, k, n)
                total_predictions = total_predictions + 1
                if predicted_class == group:
                    accurate_predictions = accurate_predictions + 1
                print("Predicted Class = ", className[predicted_class], ", Actual Class = ", className[group])
    
    accuracy = (accurate_predictions / total_predictions) * 100
    print("KNN Classifier Accuracy = ", accuracy)
    return accuracy

def mod_data(df, className):
   
    df.replace(className[0], 0, inplace = True)
    df.replace(className[1], 1, inplace = True)
    df.replace(className[2], 2, inplace = True)
    df.replace(className[3], 3, inplace = True)
    df.replace(className[4], 4, inplace = True)


def main():
    df = pd.read_csv("./dataset.csv")
    className = ['apple', 'bell', 'bird', 'frog', 'guitar']
    mod_data(df, className)
    
    
#     #Normalize the data
#     x = df.values #returns a numpy array
#     min_max_scaler = preprocessing.MinMaxScaler()
#     x_scaled = min_max_scaler.fit_transform(x)
#     df = pd.DataFrame(x_scaled) #Replace df with normalized values
#     dataset = df.astype(float).values.tolist()
    dataset = df.values.tolist()
     
    #Shuffle the dataset
    random.shuffle(dataset)
 
    #20% of the available data will be used for testing
    test_size = 0.2
    #The keys of the dict are the classes that the data is classified into
    training_set = {0: [], 1:[], 2:[], 3:[],4:[]}
    test_set = {0: [], 1:[], 2:[], 3:[],4:[]}
     
    #Split data into training and test for cross validation
    training_data = dataset[0:(len(dataset)-int(test_size * len(dataset)))]
    test_data = dataset[(len(dataset)-int(test_size * len(dataset))): len(dataset)]
     
    #Insert data into the training set
    for record in training_data:
        training_set[record[-1]].append(record[:-1]) # Append the list in the dict will all the elements of the record except the class
 
    #Insert data into the test set
    for record in test_data:
        test_set[record[-1]].append(record[:-1]) # Append the list in the dict will all the elements of the record except the class
 
    s = time.clock()
    customKNN_train(training_set, test_set, 9,  len(className), className)
    # getMajorityGroup([[10, 1], [20, 1], [10, 0], [100, 1], [30, 0]], 3, 2)
    e = time.clock()
    print("Exec Time:" ,e-s)

if __name__ == "__main__":
    main()
