
from dictionary_classifier import DictionaryClassifier
import data_parsing

from PIL import Image
import numpy
import pdb
import random
import time
import string


# text
def main1():

    f = open('timeline')
    text = f.read()
    f.close()

    text = text[:100000]

    print('File length: {0}'.format(len(text)))

    domain = string.printable
    grams = 10

    # removing bad characters
    print('Removing unknown characters')
    i = 0
    while i < len(text):
        if not text[i] in domain:
            text = text.replace(text[i], '')
        else:
            i += 1

    # turning the string data into vector data
    print('Encoding data into vectors')
    data = data_parsing.encode_string_into_vectors(text, grams=grams, domain=domain)

    # creating the classifier
    print('Creating classifier')
    ords = [ord(l) for l in domain]
    vec_size = max(ords) - min(ords) + 1

    classifier = DictionaryClassifier(
            use_all_data=False,
            svd_threshold=10,
            vector_length=vec_size * grams, 
            pursuit_improvement_threshold=0.01)

    for vec, label in data:
        classifier.add_vector_to_class(vec, label)
    classifier.create_dictionary()

    print(classifier.dictionary_shape())


    while True:
        starting_point = input('>>> ')
        response = starting_point[len(starting_point)-grams:]
        guess = ''
        while guess != '\n':
            vec = data_parsing.encode_string_into_vector(
                    response[len(response)-grams:], 
                    domain=domain, 
                    min_size=grams)
            guess = classifier.classify(vec)
            response += guess
            print(response)
        

# iris
def main3():
    num_in_training_set = 50
    num_in_testing_set = 100
    classifier = DictionaryClassifier(
            vector_length=4, # vector size
            use_all_data=True, 
            pursuit_threshold=0,
            )

    print('Setting up data')
    data_file = open('iris.txt')
    lines = data_file.readlines()
    data_file.close()
    random.seed(0)
    random.shuffle(lines)

    testing = []
    
    counter = 0
    for line in lines:
        arr = line.split(',')
        classification = arr[-1][:-1]
        str_vec = arr[:-1]
        vec = [float(x) for x in str_vec]
        if counter < num_in_training_set:
            classifier.add_vector_to_class(vec, classification)
        else:
            testing.append( (classification, vec) )
        counter += 1

    correct = 0
    total = len(testing)
    for classification, vec in testing:
        if classifier.classify(vec) == classification:
            correct += 1
    print('Correct: {0}'.format(correct))
    print('Total: {0}'.format(total))
    print('Percent: {0}'.format(correct/total))


# minst 
def main():

    # classifier parameters
    num_in_training_set = 32000
    num_in_testing_set = 500
    small_classifier = DictionaryClassifier(
            28*28, # vector size
            use_all_data=False, 
            svd_threshold=1000,
            pursuit_accuracy_threshold=50,
            pursuit_improvement_threshold=0.1
            )

    # Opening data
    print('Setting up data')
    data_file = open('Kaggle Competition MINST train.csv')
    lines = data_file.readlines()
    data_file.close()
    lines = lines[1:]
    random.shuffle(lines)

    # converting to usable format
    print('Converting data')
    training = []
    testing = []
    counter = 0

#    classes_used = {}
#    num_in_class = 100

    for line in lines:

        vals = line.split(',')
        vector = []

        vector_class = vals[0]
        for i in range(1, len(vals)):
            vector.append(float(vals[i]))

#        if not vector_class in classes_used:
#            training.append((vector, vector_class))
#            classes_used[vector_class] = 1
#        elif classes_used[vector_class] < num_in_class:
#            training.append((vector, vector_class))
#            classes_used[vector_class] += 1


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
    print('Shape: {0}'.format(small_classifier.dictionary_shape()))

    pdb.set_trace()

    incorrect = []

    # classifying
    print('Classifying data')
    small_correct = 0
    start = time.time()
    for vec in testing:
        small_guess = small_classifier.classify(vec[0])
        if small_guess == vec[1]:
            small_correct += 1
        else:
            class_name = vec[1]
            bad_vec = vec[0]
            print('Real:', class_name)
            print('Guess:', small_guess)
            Image.fromarray(numpy.array(bad_vec).reshape(28, 28)).show()
            pdb.set_trace()


    print('Total time:', time.time() - start)

    print('Total correct small: {0}'.format(small_correct)) 
    print('Percentage small: {0}'.format(small_correct/len(testing))) 
        

if __name__ == "__main__":
    main()
