

import string

import pdb



def encode_string_into_vectors(string_to_parse, grams=10, maps_to_grams=1, domain=string.printable):
    vectors_with_label = []

    for i in range(grams, len(string_to_parse)):
        letters = string_to_parse[i-grams:i]
        maps_to = string_to_parse[i:i+maps_to_grams]
        vectors_with_label.append((letters, maps_to))

    offset = min([ord(l) for l in domain])
    vec_size = max([ord(l) for l in domain]) - offset
    encoded_string_vectors = []

    for vec, label in vectors_with_label:
        new_vec = [0 for i in range(vec_size*grams)]
        new_label = [0 for i in range(vec_size*maps_to_grams)]
        for i in range(len(vec)):
            letter = vec[i]
            try:
                new_vec[ (i * vec_size) + ord(letter) - offset ] = 1
            except:
                print( len(new_vec),  (i * vec_size) + ord(letter) - offset)
#        for i in range(len(label)):
#            letter = label[i]
#            new_label[ (i * vec_size) + ord(letter) - offset ] = 1
#        encoded_string_vectors.append((new_vec, new_label))
        encoded_string_vectors.append((new_vec, label))

    return encoded_string_vectors


def decode_vector_into_string(vector, grams=1, domain=string.printable):
    offset = min([ord(l) for l in domain])
    result = ''
    for i in range(grams):
        subvec = vector[i*len(domain):(i+1)*len(domain)]
        result += chr(subvec.index(max(subvec)) + offset)
    return result


def main():
    grams = 2 
    domain = 'abcd'
    results = encode_string_into_vectors('aabbc', grams=grams, domain=domain)
    print(results)
    for foo in results:
        print()
        print(decode_vector_into_string(foo[0], grams=grams, domain=domain))
        print(decode_vector_into_string(foo[1], grams=grams, domain=domain))

if __name__ == '__main__':
    main()



