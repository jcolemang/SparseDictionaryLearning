
from dictionary_classifier import DictionaryClassifier
import data_parsing

import pdb
import random
import time
import string


def main1():

    f = open('text')
    text = f.read()
    f.close()

    domain = string.printable
    grams = 10

    # removing bad characters
    for i in range(len(text)):
        if not text[i] in domain:
            text.replace(text[i], '')

    # turning the string data into vector data
    data = data_parsing.encode_string_into_vectors(text, grams=grams, domain=domain)

    # creating the classifier
    ords = [ord(l) for l in domain]
    vec_size = max(ords) - min(ords)
    classifier = DictionaryClassifier(vec_size * grams, use_all_data=True)
    for vec, label in data:
        classifier.add_vector_to_class(vec, label)
    classifier.create_dictionary()


def main():

    # classifier parameters
    num_in_training_set = 40000
    num_in_testing_set = 2000
    small_classifier = DictionaryClassifier(
            28*28, # vector size
            use_all_data=False, 
            svd_threshold=500,
            pursuit_threshold=500,
            max_pursuit_iterations=5
            )

    # Opening data
    print('Setting up data')
    data_file = open('Kaggle Competition MINST train.csv')
    lines = data_file.readlines()
    data_file.close()
    lines = lines[1:]

    print(len(lines))

    # converting to usable format
    print('Converting data')
    training = []
    testing = []
    counter = 0
    for line in lines:

        vals = line.split(',')
        vector = []

        vector_class = vals[0]
        for i in range(1, len(vals)):
            vector.append(float(vals[i]))

        if counter < num_in_training_set:
            training.append((vector, vector_class))
        elif counter < num_in_training_set + num_in_testing_set:
            testing.append((vector, vector_class))
        else:
            break
        counter += 1
    
    # inserting into the classifier
    print('Inserting data into classifier')
    for vec in training:
        small_classifier.add_vector_to_class(vec[0], vec[1])

    # setting up the classifier
    print('Creating dictionary')
    small_classifier.create_dictionary()

    # classifying
    print('Classifying data')
    small_correct = 0
    start = time.time()
    for vec in testing:
        small_guess = small_classifier.classify(vec[0])
        if small_guess == vec[1]:
            small_correct += 1
    print('Total time:', time.time() - start)

    print()
    print('Total correct small: {0}'.format(small_correct)) 
    print('Percentage small: {0}'.format(small_correct/len(testing))) 
        

if __name__ == "__main__":
    main()
