import numpy as np
from tqdm import tqdm
from os.path import join


version_name = 'os-dramatic-rest'
file_name = 'test'
max_length = 100

encoded_path = './datasets/'+version_name+'/'+file_name+'/encoded.txt'
f = open(encoded_path, 'r', encoding = 'utf-8')
print('Reading encoded data from \"{}\"...'.format(encoded_path))
lines = f.read().splitlines()

uttr_emots_path = './datasets/'+version_name+'/'+file_name+'/uttr_emots.npy'
print('Reading emotions from \"{}\"...'.format(uttr_emots_path))
uttr_emots = np.load(uttr_emots_path)
uttr_emots = np.array(uttr_emots)
print("uttr_emots length = ", len(uttr_emots))             # uttr_emots length =  1170216
#uttr_emots = np.argsort(uttr_emots, axis = 1)

'''

Last: 1170276,1170277

871188 ==> 1170209  ,  1170214
871189 ==> 1170209  ,  1170215
871190 ==> 1170209  ,  1170216
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉| 871190/871239 [01:34<00:00, 9176.39it/s]
Traceback (most recent call last):
  File "pre-testing.py", line 53, in <module>
    target_emots[i] = uttr_emots[k]
IndexError: index 1170216 is out of bounds for axis 0 with size 1170216
'''

# RoBERTa uses 1 as the padding value
inputs = np.ones((len(lines), max_length), dtype = np.int32)
input_segments = np.ones((len(lines), max_length), dtype = np.int32)
input_emots = np.zeros((len(lines), max_length), dtype = np.int32)

targets_i = np.ones((len(lines), max_length), dtype = np.int32)
targets_r = np.ones((len(lines), max_length), dtype = np.int32)
target_segments = np.ones((len(lines), max_length), dtype = np.int32)
target_emots = np.zeros(len(lines), dtype = np.int32)

n = 0
for i, line in tqdm(enumerate(lines), total = len(lines)):
    # 0,8976,2156,2085,52,32,11,761,9,10,910,1182,479,2 0,0,0,0,0,0,0,0,0,0,0,0,0,0 0,8976,2156,38,619,101,38,697,10,1256,455,301,479,38,1266,2156,38,128,119,1819,117,810,326,4218,479,256,119,479,2   1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1   0,1
    inp_str, inp_seg_str, tar_str, tar_seg_str, idx_str = line.split('\t')
    j, k = [int(s) for s in idx_str.split(',')]

    print(i, "==>", j, " , ", k)

    inp_ids = [int(s) for s in inp_str.split(',')]
    inp_seg_ids = [int(s) for s in inp_seg_str.split(',')]
    tar_ids = [int(s) for s in tar_str.split(',')]
    tar_seg_ids = [int(s) for s in tar_seg_str.split(',')]

    seg_id = 0
    for x in range(len(inp_seg_ids)):
        if inp_seg_ids[x] != seg_id:
            j += 1
            seg_id = inp_seg_ids[x]
        #input_emots[i,x] = uttr_emots[j,-1]
        input_emots[i,x] = uttr_emots[j]
    #target_emots[i] = uttr_emots[k,-1]
    target_emots[i] = uttr_emots[k]

    inputs[i,:len(inp_ids)] = inp_ids
    input_segments[i,:len(inp_seg_ids)] = inp_seg_ids
    targets_i[i,:len(tar_ids)-1] = tar_ids[:-1]
    targets_r[i,:len(tar_ids)-1] = tar_ids[1:]
    target_segments[i,:len(tar_seg_ids)-1] = tar_seg_ids[:-1]

f.close()

print("end")
