import numpy as np  
import random
import csv


'''
n_examples = {
    'os-dramatic-rest': 871239,
    'osed-dramatic': 217440,
    'os-calm-rest': 869538,
    'osed-calm': 261304,
    'ed': 7951,
}


for key, value in n_examples.items():
    arr = np.random.randint(value, size=500)
    arr = np.sort(arr)
    print(key, arr)
    with open('prediction-listener/'+key+'_500.npy', 'wb') as f:
        np.save(f, arr)'''


for dataset in ['red']:

    indices_arr = []
    count = 0
    with open('emotions/model-os-dramatic-rest-red/prob-sampled/'+dataset+'_test_int.csv', 'r') as infile:
        readCSV = csv.reader(infile, delimiter=',')
        for row in readCSV:
            if count >= 1:
                context = row[0] # you don't have the context column in the ed_test_int.csv how do you do this?!
                arr = context.split('\n')
                #print(arr)
                if len(arr) % 2 == 1:
                    if len(arr) - 1 >= 4:
                        indices_arr.append(count-1)
            count += 1
    #print(indices_arr)
    print("No. of listener utterances in test set: ", dataset, len(indices_arr))

    arr = np.array(random.sample(indices_arr, 500))
    arr = np.sort(arr)

    print(len(arr))

    with open('prediction-listener/'+dataset+'_500.npy', 'wb') as f:
        np.save(f, arr)
