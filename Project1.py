import numpy as np
import pprint
import operator
import math

pp = pprint.PrettyPrinter(indent=2)


def preprocess(file):
    # Function loading and processing csv file to workable numpy array, with train/test split functionality
    #
    # parameters:
    #   file:           String containing path to csv file
    #

    # specify train/test split
    split = 0.0

    # read and process csv file to numpy matrix
    csv = np.genfromtxt(file, delimiter=",", dtype='unicode')

    # amount of instances
    n = len(csv[:, 0])
    test_n = round(n * split)

    # split dataset in test and training data
    test_data = csv[range(test_n), :]
    training_data = csv[range(test_n, n), :]

    return training_data, test_data


def train(training_data, classes, attrNames):
    # Function returning all probabilities necessary to make predictions according to the Naive Bayes model
    #
    # parameters:
    #   training_data:  numpy matrix containing rows of instances with attributes on each column
    #                   and the last column containing the class
    #   classes:        array of possible class values
    #   attrNames:      array of attribute names

    # calculate nr. of attributes and instances
    numberofattr = len(training_data[1, :]) - 1
    n = len(training_data[:, 0])

    # initialize object for conditional probabilities using the same structure of dictionaries specified in the lecture
    count = {}
    for i in attrNames:
        classObject = {}
        for ii in classes:
            classObject[ii] = {}
        count[i] = classObject

    # initialize object for unconditional probabilities
    classCount = {}
    for ii in classes:
        classCount[ii] = 0

    # Count instances
    for ii in range(n):
        # last column of the data always contains the class
        classValue = training_data[ii, -1]
        # counting class values for the unconditional probablities
        classCount[classValue] += 1

        # Counting attribute values per class value
        for i in range(numberofattr):
            if training_data[ii, i] in count[attrNames[i]][classValue]:
                count[attrNames[i]][classValue][training_data[ii, i]] += 1
            else:
                count[attrNames[i]][classValue][training_data[ii, i]] = 1

    # Absolute numbers -> Probablities:
    for i in attrNames:
        for ii in classes:
            # calculating the conditional probabilities while being inspired by:
            # https://stackoverflow.com/questions/30964577/divide-each-python-dictionary-value-by-total-value
            count[i][ii] = {k: v / total for total in (sum(count[i][ii].values()),) for k, v in count[i][ii].items()}
    # Same for unconditional probabilities
    for i in classes:
        classCount[i] = classCount[i] / n

    return {"conditional": count, "unconditional": classCount}


def predict(model, test_data):
    # Function returning an array of prediction according to Naive Bayes model given certain observations
    #
    # parameters:
    #   model:              dictionary containing all probabilities necessary to make predictions
    #                       according to the Naive Bayes model
    #   test_data:          numpy matrix containing rows of instances with attributes on each column
    #                       and the last column containing the class

    # initializing array for the pridictions
    predictions = []

    # Calculate which class is most likely for every observation given the model
    for observation in test_data:
        likelihood = {}
        # implement naive bayes formula specified in the lectures
        for i in model["unconditional"]:
            likelihood[i] = model["unconditional"][i]
            iii = 0
            for ii in model["conditional"]:
                if observation[iii] in model["conditional"][ii][i]:
                    likelihood[i] = likelihood[i] * model["conditional"][ii][i][observation[iii]]
                else:
                    # swap 0 with other value for smoothing
                    likelihood[i] = likelihood[i] * 0
                iii += 1
        # append class that is most likely to the array of predictions
        predictions.append(max(likelihood.items(), key=operator.itemgetter(1))[0])

    return predictions


def evaluate(model, test_data):
    # Function returning the accuracy of an naive bayes model given test data
    #
    # parameters:
    #   model:              dictionary containing all probabilities necessary to make predictions
    #                       according to the Naive Bayes model
    #   test_data:          numpy matrix containing rows of instances with attributes on each column
    #                       and the last column containing the class

    # array of class values
    classVector = test_data[:, -1]
    # array of predicted values
    predictionVector = predict(model, test_data)
    # formula specified in lecture slides
    accuracy = (classVector == predictionVector).mean()
    return accuracy


def info_gain(classVector, attrVector):
    # Function returning the information gain between to vectors
    #
    # parameters:
    #   classVector:         array containing class values
    #   attrVector:          array containing attribute values

    # amount of instances
    n = len(classVector)

    # retrieve class names and frequency from the array
    classNames, classFreq = np.unique(classVector, return_counts=True)
    classProb = classFreq / n

    # calculate entropy of the root node
    classEntropy = entropy(classProb)

    # retrieve class names and frequency from the array
    attrNames, attrFreq = np.unique(attrVector, return_counts=True)
    attrProb = attrFreq / n

    conditional_entropy = {}

    # calculate entropy of all the branching nodes
    for i in attrNames:
        split = np.where(attrVector == i)
        condClassNames, condClassFreq = np.unique(classVector[split], return_counts=True)
        condClassProb = condClassFreq / len(classVector[split])
        conditional_entropy[i] = entropy(condClassProb)

    # calculate information gain using formula specified in lectures
    information_gain = classEntropy
    for i in range(len(attrNames)):
        information_gain -= attrProb[i] * conditional_entropy[attrNames[i]]

    return information_gain


def entropy(prob):
    # Function returning the entropy of a discreet random variable
    #
    # parameters:
    #   prob:               array containing probabilities of all possible outcomes (should sum up to 1)
    #

    # calculate entropy as specified in the lectures
    res = 0
    for i in prob:
        res = res - i * math.log(i, 2)
    return res


def stumpAccuracy(data):
    # Function calculating the accuracy of a 1-r model when it is tested on the full training set
    #
    # parameters:
    #   data:         numpy matrix containing rows of instances with attributes on each column
    #                 and the last column containing the class
    #

    number_of_attr = len(data[1, :]) - 1
    classVector = data[:, -1]
    attrVectors = data[:, range(number_of_attr)]

    # Find attribute with maximum information gain to base 1-r model on
    igs = []
    for i in range(number_of_attr):
        igs.append(info_gain(classVector, attrVectors[:, i]))

    best_attr = np.where(igs == np.amax(igs))
    attrVector = data[:, best_attr[0][0]]

    # retrieve attribute names
    attrNames = np.unique(attrVector, return_counts=False)
    # number of instances
    n = len(classVector)

    # Go through each attribute. Then for each attribute, find most
    # popular class and consider any value other than that class to be an
    # error.
    mistakes = 0
    for i in attrNames:
        split = np.where(attrVector == i)
        classNames, classFreq = np.unique(classVector[split], return_counts=True)
        mistakes += sum(classFreq) - max(classFreq)
    accuracy = 1 - mistakes / n
    return accuracy


# --------------------------------------------------------------------------------------------
# Calculations performed for the report:

# Car dataset
car_file = "data/car.csv"
# array of class values
car_classes = ["unacc", "acc", "good", "vgood"]
# Array of attributes
car_attr = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]

training_data, test_data = preprocess(car_file)
model = train(training_data, car_classes, car_attr)
print("Car:")
print("Naive bayes accuracy:")
print(evaluate(model, training_data))
print("Stump accuracy: ")
print(stumpAccuracy(training_data))

# Information gain matrix
# Can be found in Information_Gain_Car.xlsx
ig_matrix = np.empty((7, 7))
for i in range(7):
    for ii in range(7):
        ig_matrix[i][ii] = info_gain(training_data[:, i], training_data[:, ii])
print("Information Gain Matrix:")
pp.pprint(ig_matrix)
print()

# Cmc dataset
cmc_file = "data/cmc.csv"
cmc_attr = ['w-education', 'h-education', 'n-child', 'w-relation', 'w-work', 'h-occupation', 'standard-of-living',
            'media-exposure']
cmc_classes = ['No-use', 'Short-term', 'Long-term']

training_data, test_data = preprocess(cmc_file)
model = train(training_data, cmc_classes, cmc_attr)
print("Cmc:")
print("Naive bayes accuracy:")
print(evaluate(model, training_data))
print("Stump accuracy: ")
print(stumpAccuracy(training_data))

ig_matrix = np.empty((9, 9))
for i in range(9):
    for ii in range(9):
        ig_matrix[i][ii] = info_gain(training_data[:, i], training_data[:, ii])

print("Information Gain Matrix:")
pp.pprint(ig_matrix)
print()

# Breast cancer dataset
bc_file = "data/breast-cancer.csv"
bc_classes = ['recurrence-events', 'no-recurrence-events', "?"]
bc_attr = ['age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']

training_data, test_data = preprocess(bc_file)
model = train(training_data, bc_classes, bc_attr)
print("Breast Cancer dataset (without imputation):")
print("Naive bayes accuracy:")
print(evaluate(model, training_data))
print("Stump accuracy dataset: ")
print(stumpAccuracy(training_data))

ig_matrix = np.empty((10, 10))
for i in range(10):
    for ii in range(10):
        ig_matrix[i][ii] = info_gain(training_data[:, i], training_data[:, ii])

print("Information Gain Matrix:")
pp.pprint(ig_matrix)

print()

bc_file = "data/breast-cancer-imputation.csv"
training_data, test_data = preprocess(bc_file)
model = train(training_data, bc_classes, bc_attr)
print("Breast Cancer (imputation):")
print("Naive bayes accuracy:")
print(evaluate(model, training_data))
print("Stump accuracy: ")
print(stumpAccuracy(training_data))

ig_matrix = np.empty((10, 10))
for i in range(10):
    for ii in range(10):
        ig_matrix[i][ii] = info_gain(training_data[:, i], training_data[:, ii])

print("Information Gain Matrix:")
pp.pprint(ig_matrix)
