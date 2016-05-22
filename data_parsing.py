

import string
from PIL import Image

import pdb


def encode_string_into_vector(string_to_parse, domain=string.printable, min_size=0):
    offset = min([ord(l) for l in domain])
    vec_offset = 0
    if len(string_to_parse) < min_size:
        vec_offset =  min_size - len(string_to_parse)
    letter_vec_size = max([ord(l) for l in domain]) - offset + 1
    vec = [0 for i in range((len(string_to_parse)+vec_offset)*letter_vec_size)]
    for i in range(len(string_to_parse)):
        letter_num = (i + vec_offset) * letter_vec_size
        character_num = ord(string_to_parse[i]) - offset
        try:
            vec[letter_num + character_num] = 1.0
        except:
            pdb.set_trace()

    return vec



def encode_string_into_vectors(string_to_parse, grams=10, maps_to_grams=1, domain=string.printable):
    vectors_with_label = []

    for i in range(grams, len(string_to_parse)):
        letters = string_to_parse[i-grams:i]
        maps_to = string_to_parse[i:i+maps_to_grams]
        vectors_with_label.append((letters, maps_to))

    offset = min([ord(l) for l in domain])
    vec_size = max([ord(l) for l in domain]) - offset + 1
    encoded_string_vectors = []

    for vec, label in vectors_with_label:
        new_vec = [0 for i in range(vec_size*grams)]
        for i in range(len(vec)):
            letter = vec[i]
            try:
                new_vec[ (i * vec_size) + ord(letter) - offset ] = 1.0
            except:
                pdb.set_trace()
                raise Exception
        encoded_string_vectors.append((new_vec, label))

    return encoded_string_vectors


def decode_vector_into_string(vector, grams=1, domain=string.printable):
    offset = min([ord(l) for l in domain])
    result = ''
    for i in range(grams):
        subvec = vector[i*len(domain):(i+1)*len(domain)]
        result += chr(subvec.index(max(subvec)) + offset)
    return result


def vector_to_image(image_size, vector):
    Image.fromarray(vector.reshape(image_size[0], image_size[1]))



def main():
    grams = 2 
    domain = 'abcd'
    to_encode = 'abcba'
    print(to_encode)
    results = encode_string_into_vectors(to_encode, grams=grams, domain=domain)
    print(results)
    for foo in results:
        print(decode_vector_into_string(foo[0], grams=grams, domain=domain), foo[1])

if __name__ == '__main__':
    main()



