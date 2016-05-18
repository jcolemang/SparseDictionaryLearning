
# Necessary for matching pursuit
from numpy import linalg as la
import numpy

# just generally useful
from math import sqrt
import pdb
import random

import time


def coherence(A):
    for i in range(A.shape[1]):
        A[:,i] /= la.norm(A[:,i]) 

    dot_prods = numpy.dot(A.T, A)
    size = dot_prods.shape[0] # symmetric

    max_val = 0
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            if dot_prods[i][j] > max_val:
                max_val = dot_prods[i][j]
    return max_val


def matching_pursuit( sensing_matrix, target, max_iterations=10, threshold=0.1 ):

    size = sensing_matrix.shape
    rows = size[0]
    columns = size[1]

    # normalizing columns just cuz
    for i in range(columns):
        sensing_matrix[:,i] /= la.norm(sensing_matrix[:,i]) 
        
    approximation = numpy.zeros(columns)
    residual = target.copy()

    for i in range(max_iterations):

        # Getting the largest dot product of A with r
        dot_products = numpy.dot(sensing_matrix.T, residual)
        max_index = dot_products.argmax()
        max_val = dot_products[max_index]

        # Updating x
        approximation[max_index] += max_val

        # Updating r
        residual = target - numpy.dot(sensing_matrix, approximation)

        # See how r compares to b
        if la.norm(residual) < threshold:
            break
    
    return approximation 


def orthogonal_matching_pursuit( sensing_matrix, target, max_iterations=10, threshold=0.1 ):

    size = sensing_matrix.shape
    rows = size[0]
    columns = size[1]
    indices = []
    residual = target 
    residual_norm = la.norm(residual)

    for i in range(max_iterations):

        # reseting the estimation
        estimation = numpy.zeros(columns)
        
        # getting the index of the max dot product
        dot_products = numpy.dot(sensing_matrix.T, residual)
        max_index = dot_products.argmax()
        indices.append(max_index)

        # finding least squares solution
        significant_columns = sensing_matrix[:, indices]
        least_squares = numpy.dot(la.pinv(significant_columns), target)

        # updating the estimation
        for idx in range(len(least_squares)):
            estimation[indices[idx]] = least_squares[idx]

        # updating and checking the residual
        prev_residual_norm = residual_norm
        residual = target - numpy.dot(sensing_matrix, estimation)
        residual_norm = la.norm(residual)

        if residual_norm < threshold or abs(residual_norm - prev_residual_norm) < 0.001:
            break

    return estimation


def stagewise_orthogonal_matching_pursuit( sensing_matrix, target, max_iterations=10, threshold=0.1, vecs_per_stage=2):
    """
    http://sparselab.stanford.edu/SparseLab_files/local_files/StOMP.pdf

    TODO implement the method given in the paper.
    """

    size = sensing_matrix.shape
    rows = size[0]
    columns = size[1]
    indices = []
    residual = target 
    residual_norm = la.norm(residual)

    for i in range(max_iterations):

        # reseting the estimation
        estimation = numpy.zeros(columns)
        
        # getting the index of the max dot product
        dot_products = numpy.absolute(numpy.dot(sensing_matrix.T, residual))
#        dot_products = numpy.dot(sensing_matrix.T, residual)

#        print('\n\nA bunch of fun stuff')
#        for j in range(len(dot_products)):
#            print(dot_products[j])

        for j in range(vecs_per_stage):
            max_index = dot_products.argmax()
            if not max_index in indices:
                indices.append(max_index)
            dot_products[max_index] = 0

        # finding least squares solution
        significant_columns = sensing_matrix[:, indices]
        least_squares = numpy.dot(la.pinv(significant_columns), target)

        # updating the estimation
        for idx in range(len(least_squares)):
            estimation[indices[idx]] = least_squares[idx]

        # updating and checking the residual
        prev_residual_norm = residual_norm
        residual = target - numpy.dot(sensing_matrix, estimation)
        residual_norm = la.norm(residual)

        if residual_norm < threshold or abs(residual_norm - prev_residual_norm) < 0.001:
            break

    return estimation


def sparsity_of_vector(vec, threshold=0.000001):
    s = 0
    for i in range(vec.size):
        if abs(vec[i]) > threshold:
            s += 1
    return s


def test_matching_pursuit(A, b, max_pursuit_iterations=10, success=0.00001):
    return _test_pursuit(matching_pursuit, A, b, max_pursuit_iterations, success)


def test_orthogonal_matching_pursuit(A, b, max_pursuit_iterations=10, success=0.00001):
    return _test_pursuit(orthogonal_matching_pursuit, A, b, max_pursuit_iterations, success)


def _test_pursuit(function, A, b, max_iterations, success_threshold):
    print(function.__name__)
    result = function(A, b, max_iterations=max_iterations, threshold=success_threshold)
    if la.norm(numpy.dot(A, result) - b) < success_threshold:
        print('Success!') 
    else:
        print('Failed norm: {0}'.format(la.norm(numpy.dot(A, result) - b)))


def main():
    
    # example 1
#    A = numpy.array( [ 
#        [1/sqrt(2), 0, -1/sqrt(2), 1], 
#        [1/sqrt(2), 1,  1/sqrt(2), 0] ] )
#    b = numpy.array( [[1/sqrt(2)], [1/sqrt(2)]] )

    # my test
    # sensing matrix
    print('Generating matrix')
    measurements = 500
    unknowns = 10000
    sparsity = 5 
    A = numpy.random.randn( measurements, unknowns )

    print('Normalizing columns')
    for i in range(A.shape[1]):
        A[:,i] /= la.norm(A[:,i]) 

    print('Coherence: {0}'.format(coherence(A)))

    print('Creating sparse solution')
    x = numpy.zeros(unknowns)
    for i in range(sparsity):
        x[random.randint(0, unknowns)] = random.randint(0, 100)

    # observation
    b = numpy.dot(A, x)

    test_orthogonal_matching_pursuit(A, b, max_pursuit_iterations=sparsity)
    test_matching_pursuit(A, b, max_pursuit_iterations=sparsity)



if __name__ == "__main__":
    main()
