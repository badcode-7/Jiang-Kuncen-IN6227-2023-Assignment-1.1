import pandas as pd
import math
import operator
import json
from sklearn.preprocessing import StandardScaler
import time


# preprocess
def preprocess(df,labels):
    df.dropna()#Remove the na value
    #transform ? into mode
    for column in df.columns:
        # compute mode
        mode_value = df[column].mode().iloc[0]

        # Check if each element is "?"
        for index, row in df.iterrows():
            if row[column] ==" ?":
                # Replace '?' with the mode in every column
                df.at[index, column] = mode_value

    # use StandardScaler to Standardize the data
    scaler = StandardScaler()
    df[labels] = scaler.fit_transform(df[labels])
    df[labels] = pd.DataFrame(df, columns=labels).round(2)  # Set the decimal place to 2
    return df

#Decision tree generation

# Calculate the empirical entropy
def calcShannonEnt(dataset):
    numEntries = len(dataset) # Number of data set samples
    labelCounts = {} # keep the number of occurrences of each tag
    for featvec in dataset:
        #Create a dictionary for each category and count the number of times each category appears
        adu_class=featvec[-1]
        if adu_class not in labelCounts.keys():# If the tag is not in the count dictionary, add it
            labelCounts[adu_class] = 0
        labelCounts[adu_class] += 1

    shannonEnt = 0.0 # Initialize the Shannon entropy
    for key in labelCounts:  # Calculate Shannon entropy
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * math.log(prob, 2)
    return shannonEnt

# For categorical variables, split the data set
def splitDataSet(dataset, axis, value):
    retDataSet=[]
    # began to traverse the data set
    for raw in dataset:
        if raw[axis] == value:
            # Adds the value before the axis element to reducedFeatVec
            reducedFeatVec = raw[:axis]
            # The next line of content after axis+1 is added to reducedFeatVec
            reducedFeatVec.extend(raw[axis + 1:])
            retDataSet.append(reducedFeatVec)#Add it to the subset list
    # Returns the partitioned data set
    return retDataSet

# For continuous numeric variables, split the data set
def splitDataSetForSeries(dataSet, axis, value):
    eltDataSet = []# hold collections that are not larger than partition values
    gtDataSet = [] #hold collections larger than partition values
    # Partition according to the value passed in, retaining the feature value to different subsets
    for row in dataSet:
        if (row[axis] <= value):
            eltDataSet.append(row)
        else:
            gtDataSet.append(row)
    #Returns two subsets --not greater than and greater than
    return eltDataSet, gtDataSet


# calculate the information gain (ratio) of categorical features
def calcInfoGain(df, feavec, i, baseEntropy):
    uniqueVals = set(feavec)  # Weight removal
    newEntropy = 0.0  # Empirical conditions entropy initialization
    for value in uniqueVals:  # Calculate information gain
        subDataSet = splitDataSet(df, i, value)
        prob = len(subDataSet) / float(len(feavec))
        newEntropy += prob * calcShannonEnt(subDataSet)# Empirical condition entropy calculation formula
    infoGain = baseEntropy - newEntropy  # Calculate information gain

    # Compute split information
    intrinsic_info = 0  # Split information initialization
    # Iterate over the total number of different values
    for value in uniqueVals:
        pf=0
        for j in range(len(df)):
            if feavec[j] == value:
                pf=pf+1
        probability = pf / len(df)
        intrinsic_info -= probability * math.log2(probability)
    # Calculate the information gain rate
    gain_ratio_value = infoGain / intrinsic_info if intrinsic_info != 0 else 0
    return infoGain

# calculate the information gain (ratio) of continuous numeric features
def calcInfoGainForSeries(dataSet, feavec,i, baseEntropy):
    # Get a list of all the current feature values in the dataset
    featList = [example[i] for example in dataSet]
    # Get a list of income tags
    classList = [example[-1] for example in dataSet]

    bestMid = -1# Best point to split now （-1 at beginning）
    maxInfoGain = 0.0  # Current maximum information gain(not used in final model)
    best_gain_ratio = 0# Current maximum information gain rate (0 at beginning)
    best_infogain=0
    dictList = dict(zip(featList, classList)) #Integrate feature values and label values

    # Sort from smallest to largest, by the size of the consecutive values
    sortedFeatList = sorted(dictList.items(), key=operator.itemgetter(0))
    # Calculate the number of consecutive values
    numberForFeatList = len(dictList)

    # Calculate partition points and keep three decimals
    midFeatList = [round((sortedFeatList[i][0] + sortedFeatList[i + 1][0]) / 2.0, 3) for i in
                   range(numberForFeatList - 1)]

    # Traverse and calculate the information gain of each partition point
    for mid in midFeatList:
        # Partition the continuous value into two parts that are not greater than the current partition point and greater than the current partition point
        eltDataSet, gtDataSet = splitDataSetForSeries(dataSet, i, mid)

        # Calculate the sum of the product of eigenvalue entropy and weight of the two parts
        weight1=len(eltDataSet) / len(feavec)
        weight2=len(gtDataSet) / len(feavec)
        newEntropy = weight1 * calcShannonEnt(eltDataSet) + weight2* calcShannonEnt(gtDataSet)

        infoGain = baseEntropy - newEntropy# Calculate information gain
        intrinsic_info = - (weight1 * math.log2(weight1) + weight2 * math.log2(weight2))# Compute split information
        # Calculate the information gain rate
        gain_ratio_value = infoGain / intrinsic_info if intrinsic_info != 0 else 0

        #Determine whether the information  gain rate of the current partition point is higher than the best information gain rate, if it is replaced and recorded
        #if gain_ratio_value > best_gain_ratio:
            #best_gain_ratio = gain_ratio_value
            #bestMid = mid
        if infoGain>best_infogain:
            best_infogain=infoGain
            bestMid = mid
    #Returns the best information gain rate and partition point
    return best_infogain, bestMid

#Select the most appropriate features (and possible split points) to split the dataset based on the information gain ratio
def chooseBestFeatureToSplit(dataset,labels):
    numFeatures = len(dataset[0]) - 1# Feature number
    baseEntropy = calcShannonEnt(dataset)# Empirical entropy
    bestInfoGain = 0.0  # Information gain
    bestFeature = -1  # Optimal feature index value
    # Indicates whether the current best eigenvalue is continuous
    flagSeries = 0
    #Record partition points for consecutive values
    bestSeriesMid = 0.0
    for i in range(numFeatures):
        # Gets all features in column i of the dataSet
        featList = [example[i] for example in dataset]# Feature column vector

        # Determine the variable type by determining whether the element in the feature list is a string
        if isinstance(featList[0], str):
            infoGain = calcInfoGain(dataset, featList, i, baseEntropy)

        else:
            infoGain, bestMid = calcInfoGainForSeries(dataset, featList,i, baseEntropy)
            # If the current information gain is greater than the original
            if infoGain > bestInfoGain:
                # New best information gain
                bestInfoGain = infoGain
                # New best index for partitioning eigenvalues
                bestFeature = i
                flagSeries = 0
                #If the current eigenvalue is a continuous variable, change flagSeries and record the best cut-off point
                if not isinstance(dataset[0][bestFeature], str):
                    flagSeries = 1
                    bestSeriesMid = bestMid

    #Returns different values based on different variable types
    if flagSeries:
        return bestFeature, bestSeriesMid
    else:
        return bestFeature

# Find the most tags
def majorityCnt(classList):
    classCount = {}
    #Iterate over the count and save to the dictionary
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

#Create decision tree
def createTree(dataset, labels,max_depth=None):
    # Get all the classification tags of the dataset
    classList = [example[-1] for example in dataset]
    # If maximum depth is reached, return most categories
    if max_depth is not None and max_depth <= 0:
        return majorityCnt(classList)
    # Returns this type when all types are the same
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # Calculate the number of data in the first row, if there is only one, it means that all the characteristics have been traversed, and the remaining one is the category label
    if len(dataset[0]) == 1:  # When there is no feature set that can be classified
        return majorityCnt(classList) # Returns the most frequent occurrence of the remaining tags

        #Initialization parameter
        bestFeatLabel = ''
        flagSeries = 0
        midSeries = 0.0

    #Select the best partition feature and get the subscript of that feature
    bestFeat = chooseBestFeatureToSplit(dataset,labels)
    # Determines whether it is a continuous value based on whether it returns a tuple
    if isinstance(bestFeat, tuple):
        bestFeatLabel = str(labels[bestFeat[0]]) + '小于' + str(bestFeat[1]) + '?'# Modify the fork point information
        midSeries = bestFeat[1]# Get the current partition point
        bestFeat = bestFeat[0]# Get the subscript value
        flagSeries = 1# Continuous value flag
    else:
        bestFeatLabel = labels[bestFeat]# Get bifurcation point information
        flagSeries = 0# Discrete value flag

    # Use a dictionary to store the tree structure, where forks are divided into feature names
    myTree = {bestFeatLabel: {}}
    # Get all possible values for the current feature tag
    featValues = [example[bestFeat] for example in dataset]

    # Continuous value processing
    if flagSeries:
        # Partition the continuous value into two parts that are not greater than the current partition point and greater than the current partition point
        eltDataSet, gtDataSet = splitDataSetForSeries(dataset, bestFeat, midSeries)
        # Get the remaining feature tags
        subLabels = labels[:]
        # Recursively handles subtrees that are smaller than partition points
        subTree = createTree(eltDataSet, subLabels,max_depth - 1 if max_depth is not None else None)
        myTree[bestFeatLabel]['小于'] = subTree
        # Recursively handles subtrees that are larger than the current partition point
        subTree = createTree(gtDataSet, subLabels,max_depth - 1 if max_depth is not None else None)
        myTree[bestFeatLabel]['大于'] = subTree

        return myTree

    # Discrete value processing
    else:
        # Delete the feature value of this partition from the list
        del (labels[bestFeat])
        # Remove duplicate feature value
        uniqueVals = set(featValues)
        # Iterate over all eigenvalues
        for value in uniqueVals:
            # Get the remaining feature tags
            subLabels = labels[:]
            # Recursive call to divide all data in the data set whose feature is equal to the current feature value under the current node
            subTree = createTree(splitDataSet(dataset, bestFeat, value), subLabels,max_depth - 1 if max_depth is not None else None)
            # Put the subtree under the fork
            myTree[bestFeatLabel][value] = subTree
        return myTree



# Column tag naming
columns_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                 'income']
# Continuous feature column tags
process_label = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
traindf = pd.read_csv('adult.data', names=columns_names)# read data
# Data Preprocessing
traindf = preprocess(traindf, process_label)
shanno=calcShannonEnt(traindf)
labels=columns_names
dataset_list=traindf.to_numpy().tolist()

# Start generating the decision tree
start_time = time.time()  # Record the start time
mytree=createTree(dataset_list,labels,max_depth=10) #According to the optimal depth to generate the decision tree
end_time = time.time()  # Record the end time
training_time = end_time - start_time  # Calculate the time spent training
print(f"train time：{training_time:.2f} s")
print(mytree)

# Save the decision tree in json format using the JSON module
file_name = 'my_tree.json'
with open(file_name, 'w') as json_file:
    json.dump(mytree, json_file)



