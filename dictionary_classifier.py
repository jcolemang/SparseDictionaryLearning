
import numpy
from compressed_sensing import orthogonal_matching_pursuit

import pdb
from math import sqrt

import time


class DictionaryClassifier:
    """
    A classifier that uses compressed sensing to classify
    solutions.

    @author Coleman Gibson 
    """

    def __init__(self, vector_length, use_all_data=False, svd_threshold=1000, pursuit_threshold=0, max_pursuit_iterations=None):
        self._classes = {}
        self._vector_length = vector_length
        self._dictionary = None
        self._class_range = {}
        self._shrink = not use_all_data
        self._svd_threshold = svd_threshold
        self._ortho_matching_pursuit_threshold = pursuit_threshold
        self._max_class_size = 0

        if max_pursuit_iterations == None:
            self._max_ortho_iterations = len(self._classes)
        else:
            self._max_ortho_iterations = max_pursuit_iterations


    def add_class(self, class_name):
        if class_name in self._classes:
            raise KeyError('Class already exists')
        self._classes[class_name] = []
        self._class_range[class_name] = [0, 0]


    def add_vector_to_class(self, vector, class_name):

        if type(vector) == list:
            vector = numpy.array(vector)

        if not self._vector_is_valid(vector):
            raise ValueError('Vector is not the correct size')

        if class_name in self._classes:
            self._classes[class_name].append(vector)
        else:
            self.add_class(class_name)
            self.add_vector_to_class(vector, class_name)

        if len(self._classes[class_name]) > self._max_class_size:
            self._max_class_size = len(self._classes[class_name])


    def create_dictionary(self):
        """
        Add in some principal component analysis, training, so on
        """
        if self._shrink:
            print('Shrinking')
            self._shrink_classes()
        self._base_dictionary()


    def find_sparse_solution_to(self, new_vector):
        """
        Uses orthogonal matching pursuit to find a sparse solution
        to Dx = new_vector
        """

        if not self._vector_is_valid(new_vector):
            raise ValueError('Vector is not the correct size')

        if self._dictionary is None:
            self.create_dictionary()

        solution = orthogonal_matching_pursuit(
                self._dictionary, 
                new_vector, 
                max_iterations=self._max_ortho_iterations, 
                threshold=self._ortho_matching_pursuit_threshold)

        return solution


    def classify(self, new_vector):
        """
        Classifies a new incoming vector by looking at which class
        best estimates the new_vector
        """

        if type(new_vector) == list:
            new_vector = numpy.array(new_vector)

        sparse_solution = self.find_sparse_solution_to(new_vector)

        class_norms = {}
        for key in self._classes:
            class_vec = self._zero_all_but_class(key, sparse_solution)
            approximation = numpy.dot(self._dictionary, class_vec)
            class_norms[key] = numpy.linalg.norm(approximation - new_vector)

        random_key = list(class_norms.keys())[0]
        min_val = class_norms[random_key]
        for key in class_norms:
            if class_norms[key] <= min_val:
                best_class = key
                min_val = class_norms[key]

        return best_class


    def _zero_all_but_class(self, key, vector_to_zero):
        zeroed = numpy.zeros(vector_to_zero.shape)
        for i in range(self._class_range[key][0], self._class_range[key][1] + 1):
            zeroed[i] = vector_to_zero[i]
        return zeroed


    def _base_dictionary(self):
        vectors = []
        current_first = 0
        for key in self._classes:
            self._class_range[key][0] = current_first
            current_first += len(self._classes[key])
            for v in self._classes[key]:
                vectors.append(v)
                self._class_range[key][1] = len(vectors) - 1
        self._dictionary = self._to_array(vectors) 

        for i in range(self._dictionary.shape[1]):
            self._dictionary[:,i] /= numpy.linalg.norm(self._dictionary[:,i]) 

        print(self._dictionary.shape)


    def _shrink_classes(self):
        """
        Pulls the significant singular values away from the SVD 
        of each class. Really just PCA on the vectors of each 
        class given
        """

        for class_name in self._classes:
            class_array = self._to_array(self._classes[class_name])
            U, singular_values, Vt = numpy.linalg.svd(class_array, full_matrices=True)
            S = numpy.diag(singular_values)
            num_to_keep = 0
            for val in singular_values:
                if val > self._svd_threshold:
                    num_to_keep += 1

            U_to_keep = U[:, :num_to_keep]
            singular_values_to_keep = singular_values[:num_to_keep]
            S_to_keep = numpy.diag(singular_values_to_keep)
            Vt_to_keep = Vt[:num_to_keep,:]

            # contains the columns I care about
            result = numpy.dot(U_to_keep, numpy.dot(S_to_keep, Vt_to_keep)).T
            to_keep = []
            for i in range(num_to_keep):
                to_keep.append(result[i])
            self._classes[class_name] = to_keep 


    def _vector_is_valid(self, vector):
        return vector.shape[0] == self._vector_length


    def _to_array(self, list_of_vecs):
        return numpy.vstack(list_of_vecs).T


        

def main():
    d = DictionaryClassifier(2)

    d.add_vector_to_class('a', numpy.array([1/sqrt(2), 1/sqrt(2)]))
    d.add_vector_to_class('b', numpy.array([0, 1]))
    d.add_vector_to_class('c', numpy.array([-1/sqrt(2), 1/sqrt(2)]))
    d.add_vector_to_class('d', numpy.array([1, 0]))
    d.create_dictionary()
    print(d.classify(numpy.array([0, 1])))



if __name__ == '__main__':
    main()

