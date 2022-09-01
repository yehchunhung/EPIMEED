import numpy as np
import tensorflow as tf
from tqdm import tqdm
from os.path import join

def create_dataset_from_encoded(read_path, max_length, index = None, contain_comm = False, contain_mask = False):
    if contain_comm and contain_mask: # has both comm and mask
        encoded_path = join(read_path, 'encoded.txt')
        f = open(encoded_path, 'r', encoding = 'utf-8')
        print('Reading encoded data from \"{}\"...'.format(encoded_path))
        lines = f.read().splitlines()

        uttr_emots_path = join(read_path, 'uttr_emots.npy')
        uttr_comms_path = join(read_path, 'uttr_comms.npy')
        uttr_masks_path = join(read_path, 'uttr_masks.npy')
        print('Reading emotions from \"{}\"...'.format(uttr_emots_path))
        print('Reading emotions from \"{}\"...'.format(uttr_comms_path))
        print('Reading emotions from \"{}\"...'.format(uttr_masks_path))
        uttr_emots = np.load(uttr_emots_path)
        uttr_comms = np.load(uttr_comms_path)
        uttr_masks = np.load(uttr_masks_path, allow_pickle = True)
        #uttr_emots = np.argsort(uttr_emots, axis = 1)

        # RoBERTa uses 1 as the padding value
        inputs = np.ones((len(lines), max_length), dtype = np.int32)
        input_segments = np.ones((len(lines), max_length), dtype = np.int32)
        input_emots = np.zeros((len(lines), max_length), dtype = np.int32)
        input_comms = np.zeros((len(lines), max_length), dtype = np.int32)

        targets_i = np.ones((len(lines), max_length), dtype = np.int32)
        targets_r = np.ones((len(lines), max_length), dtype = np.int32)
        target_segments = np.ones((len(lines), max_length), dtype = np.int32)
        targets_mask = np.ones((len(lines), max_length), dtype = np.int32)
        target_emots = np.zeros(len(lines), dtype = np.int32)
        target_comms = np.zeros(len(lines), dtype = np.int32)

        n = 0
        for i, line in tqdm(enumerate(lines), total = len(lines)):
            # 0,8976,2156,2085,52,32,11,761,9,10,910,1182,479,2 0,0,0,0,0,0,0,0,0,0,0,0,0,0 0,8976,2156,38,619,101,38,697,10,1256,455,301,479,38,1266,2156,38,128,119,1819,117,810,326,4218,479,256,119,479,2   1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1   0,1
            inp_str, inp_seg_str, tar_str, tar_seg_str, idx_str = line.split('\t')
            j, k = [int(s) for s in idx_str.split(',')]

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
                input_comms[i,x] = uttr_comms[j]
            #target_emots[i] = uttr_emots[k,-1]
            target_emots[i] = uttr_emots[k]
            target_comms[i] = uttr_comms[k]

            inputs[i,:len(inp_ids)] = inp_ids
            input_segments[i,:len(inp_seg_ids)] = inp_seg_ids
            targets_i[i,:len(tar_ids)-1] = tar_ids[:-1]
            targets_r[i,:len(tar_ids)-1] = tar_ids[1:]
            target_segments[i,:len(tar_seg_ids)-1] = tar_seg_ids[:-1]
            
            if len(uttr_masks[k]) > max_length:
                targets_mask[i,:max_length] = uttr_masks[k][:max_length]
            else:
                targets_mask[i,:len(uttr_masks[k])] = uttr_masks[k]
            

        f.close()

        if index is not None:
            inputs = inputs[index]
            input_segments = input_segments[index]
            input_emots = input_emots[index]
            input_comms = input_comms[index]
            targets_i = targets_i[index]
            targets_r = targets_r[index]
            targets_mask = targets_mask[index]
            target_segments = target_segments[index]
            target_emots = target_emots[index]
            target_comms = target_comms[index]

        return (tf.data.Dataset.from_tensor_slices(inputs),
                tf.data.Dataset.from_tensor_slices(input_segments),
                tf.data.Dataset.from_tensor_slices(input_emots),
                tf.data.Dataset.from_tensor_slices(input_comms),
                tf.data.Dataset.from_tensor_slices(targets_i),
                tf.data.Dataset.from_tensor_slices(targets_r),
                tf.data.Dataset.from_tensor_slices(target_segments),
                tf.data.Dataset.from_tensor_slices(target_emots),
                tf.data.Dataset.from_tensor_slices(target_comms),
                tf.data.Dataset.from_tensor_slices(targets_mask)), inputs.shape[0]
    
    elif contain_mask: # no comm
        encoded_path = join(read_path, 'encoded.txt')
        f = open(encoded_path, 'r', encoding = 'utf-8')
        print('Reading encoded data from \"{}\"...'.format(encoded_path))
        lines = f.read().splitlines()

        uttr_emots_path = join(read_path, 'uttr_emots.npy')
        uttr_masks_path = join(read_path, 'uttr_masks.npy')
        print('Reading emotions from \"{}\"...'.format(uttr_emots_path))
        print('Reading emotions from \"{}\"...'.format(uttr_masks_path))
        uttr_emots = np.load(uttr_emots_path)
        uttr_masks = np.load(uttr_masks_path, allow_pickle = True)
        #uttr_emots = np.argsort(uttr_emots, axis = 1)

        # RoBERTa uses 1 as the padding value
        inputs = np.ones((len(lines), max_length), dtype = np.int32)
        input_segments = np.ones((len(lines), max_length), dtype = np.int32)
        input_emots = np.zeros((len(lines), max_length), dtype = np.int32)

        targets_i = np.ones((len(lines), max_length), dtype = np.int32)
        targets_r = np.ones((len(lines), max_length), dtype = np.int32)
        target_segments = np.ones((len(lines), max_length), dtype = np.int32)
        targets_mask = np.zeros((len(lines), max_length), dtype = np.int32)
        target_emots = np.zeros(len(lines), dtype = np.int32)

        n = 0
        for i, line in tqdm(enumerate(lines), total = len(lines)):
            # 0,8976,2156,2085,52,32,11,761,9,10,910,1182,479,2 0,0,0,0,0,0,0,0,0,0,0,0,0,0 0,8976,2156,38,619,101,38,697,10,1256,455,301,479,38,1266,2156,38,128,119,1819,117,810,326,4218,479,256,119,479,2   1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1   0,1
            inp_str, inp_seg_str, tar_str, tar_seg_str, idx_str = line.split('\t')
            j, k = [int(s) for s in idx_str.split(',')]

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
            
            if len(uttr_masks[k]) > max_length:
                targets_mask[i,:max_length] = uttr_masks[k][:max_length]
            else:
                targets_mask[i,:len(uttr_masks[k])] = uttr_masks[k]
            
            
            # create a mask to distinguish no comm and at least weak comm
            # targets_mask[i, :] = (targets_mask[i, :] > 0) * 1
            
        f.close()

        if index is not None:
            inputs = inputs[index]
            input_segments = input_segments[index]
            input_emots = input_emots[index]
            targets_i = targets_i[index]
            targets_r = targets_r[index]
            targets_mask = targets_mask[index]
            target_segments = target_segments[index]
            target_emots = target_emots[index]

        return (tf.data.Dataset.from_tensor_slices(inputs),
                tf.data.Dataset.from_tensor_slices(input_segments),
                tf.data.Dataset.from_tensor_slices(input_emots),
                tf.data.Dataset.from_tensor_slices(targets_i),
                tf.data.Dataset.from_tensor_slices(targets_r),
                tf.data.Dataset.from_tensor_slices(target_segments),
                tf.data.Dataset.from_tensor_slices(target_emots),
                tf.data.Dataset.from_tensor_slices(targets_mask)), inputs.shape[0]
    
    elif contain_comm: # no mask
        encoded_path = join(read_path, 'encoded.txt')
        f = open(encoded_path, 'r', encoding = 'utf-8')
        print('Reading encoded data from \"{}\"...'.format(encoded_path))
        lines = f.read().splitlines()

        uttr_emots_path = join(read_path, 'uttr_emots.npy')
        uttr_comms_path = join(read_path, 'uttr_comms.npy')
        print('Reading emotions from \"{}\"...'.format(uttr_emots_path))
        print('Reading emotions from \"{}\"...'.format(uttr_comms_path))
        uttr_emots = np.load(uttr_emots_path)
        uttr_comms = np.load(uttr_comms_path)
        #uttr_emots = np.argsort(uttr_emots, axis = 1)

        # RoBERTa uses 1 as the padding value
        inputs = np.ones((len(lines), max_length), dtype = np.int32)
        input_segments = np.ones((len(lines), max_length), dtype = np.int32)
        input_emots = np.zeros((len(lines), max_length), dtype = np.int32)
        input_comms = np.zeros((len(lines), max_length), dtype = np.int32)

        targets_i = np.ones((len(lines), max_length), dtype = np.int32)
        targets_r = np.ones((len(lines), max_length), dtype = np.int32)
        target_segments = np.ones((len(lines), max_length), dtype = np.int32)
        target_emots = np.zeros(len(lines), dtype = np.int32)
        target_comms = np.zeros(len(lines), dtype = np.int32)

        n = 0
        for i, line in tqdm(enumerate(lines), total = len(lines)):
            # 0,8976,2156,2085,52,32,11,761,9,10,910,1182,479,2 0,0,0,0,0,0,0,0,0,0,0,0,0,0 0,8976,2156,38,619,101,38,697,10,1256,455,301,479,38,1266,2156,38,128,119,1819,117,810,326,4218,479,256,119,479,2   1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1   0,1
            inp_str, inp_seg_str, tar_str, tar_seg_str, idx_str = line.split('\t')
            j, k = [int(s) for s in idx_str.split(',')]

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
                input_comms[i,x] = uttr_comms[j]
            #target_emots[i] = uttr_emots[k,-1]
            target_emots[i] = uttr_emots[k]
            target_comms[i] = uttr_comms[k]

            inputs[i,:len(inp_ids)] = inp_ids
            input_segments[i,:len(inp_seg_ids)] = inp_seg_ids
            targets_i[i,:len(tar_ids)-1] = tar_ids[:-1]
            targets_r[i,:len(tar_ids)-1] = tar_ids[1:]
            target_segments[i,:len(tar_seg_ids)-1] = tar_seg_ids[:-1]

        f.close()

        if index is not None:
            inputs = inputs[index]
            input_segments = input_segments[index]
            input_emots = input_emots[index]
            input_comms = input_comms[index]
            targets_i = targets_i[index]
            targets_r = targets_r[index]
            target_segments = target_segments[index]
            target_emots = target_emots[index]
            target_comms = target_comms[index]

        return (tf.data.Dataset.from_tensor_slices(inputs),
                tf.data.Dataset.from_tensor_slices(input_segments),
                tf.data.Dataset.from_tensor_slices(input_emots),
                tf.data.Dataset.from_tensor_slices(input_comms),
                tf.data.Dataset.from_tensor_slices(targets_i),
                tf.data.Dataset.from_tensor_slices(targets_r),
                tf.data.Dataset.from_tensor_slices(target_segments),
                tf.data.Dataset.from_tensor_slices(target_emots),
                tf.data.Dataset.from_tensor_slices(target_comms)), inputs.shape[0]
    
    else: # neither comm nor mask
        encoded_path = join(read_path, 'encoded.txt')
        f = open(encoded_path, 'r', encoding = 'utf-8')
        print('Reading encoded data from \"{}\"...'.format(encoded_path))
        lines = f.read().splitlines()

        uttr_emots_path = join(read_path, 'uttr_emots.npy')
        print('Reading emotions from \"{}\"...'.format(uttr_emots_path))
        uttr_emots = np.load(uttr_emots_path)
        #uttr_emots = np.argsort(uttr_emots, axis = 1)

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

        if index is not None:
            inputs = inputs[index]
            input_segments = input_segments[index]
            input_emots = input_emots[index]
            targets_i = targets_i[index]
            targets_r = targets_r[index]
            target_segments = target_segments[index]
            target_emots = target_emots[index]

        return (tf.data.Dataset.from_tensor_slices(inputs),
                tf.data.Dataset.from_tensor_slices(input_segments),
                tf.data.Dataset.from_tensor_slices(input_emots),
                tf.data.Dataset.from_tensor_slices(targets_i),
                tf.data.Dataset.from_tensor_slices(targets_r),
                tf.data.Dataset.from_tensor_slices(target_segments),
                tf.data.Dataset.from_tensor_slices(target_emots)), inputs.shape[0]


def create_test_dataset(tokenizer, path, batch_size, max_length, index = None, contain_comm = False, contain_mask = False):
    print('Vocabulary size is {}.'.format(tokenizer.vocab_size))
    test_dataset, N = create_dataset_from_encoded(path, max_length, index, contain_comm, contain_mask)
    test_dataset = tf.data.Dataset.zip(test_dataset).batch(batch_size)
    return test_dataset, N

def create_datasets(tokenizer, path, buffer_size, batch_size, max_length, contain_comm = False, contain_mask = False):
    print('Vocabulary size is {}.'.format(tokenizer.vocab_size))

    train_dataset, _ = create_dataset_from_encoded(join(path, 'train'), max_length, None, contain_comm, contain_mask)
    val_dataset, _ = create_dataset_from_encoded(join(path, 'valid'), max_length, None, contain_comm, contain_mask)
    test_dataset, _ = create_dataset_from_encoded(join(path, 'test'), max_length, None, contain_comm, contain_mask)

    train_dataset = tf.data.Dataset.zip(train_dataset).shuffle(buffer_size).batch(batch_size)
    val_dataset = tf.data.Dataset.zip(val_dataset).batch(batch_size)
    test_dataset = tf.data.Dataset.zip(test_dataset).batch(batch_size)

    return train_dataset, val_dataset, test_dataset

'''def create_os_test_dataset(tokenizer, path, batch_size, max_length, index = None):
    print('Vocabulary size is {}.'.format(tokenizer.vocab_size))
    test_dataset, N = create_dataset_from_encoded(path, max_length, index)
    test_dataset = tf.data.Dataset.zip(test_dataset).batch(batch_size)
    return test_dataset, N


def create_ed_datasets(tokenizer, path, buffer_size, batch_size, max_length, index = None):
    print('Vocabulary size is {}.'.format(tokenizer.vocab_size))

    def create_dataset(read_path, cascade = True):
        print('Reading data from \"{}\"...'.format(read_path))

        if not cascade:
            with open(join(read_path, 'prompts.txt'), 'r', encoding = 'utf-8') as f:
                all_prompts = f.read().splitlines()
            with open(join(read_path, 'context_emots.txt'), 'r', encoding = 'utf-8') as f:
                all_context_emots = f.read().splitlines()
        else:
            all_prompts = []
            all_context_emots = []

        with open(join(read_path, 'uttrs.txt'), 'r', encoding = 'utf-8') as f:
            uttrs = f.read().splitlines()

        # For test set, we randomly choose a turn to be target.
        dialogs_file = 'dialogs_partial.txt' if not cascade else 'dialogs.txt'
        with open(join(read_path, dialogs_file), 'r', encoding = 'utf-8') as f:
            dialogs = [(int(i) for i in line.split(',')) for line in f.read().splitlines()]

        uttr_emots = np.load(join(read_path, 'uttr_emots.npy'))
        assert len(uttrs) == uttr_emots.shape[0]
        uttr_emots = np.argsort(uttr_emots, axis = 1)

        SOS_ID = tokenizer.encode('<s>')[0]
        EOS_ID = tokenizer.encode('</s>')[0]

        # RoBERTa uses 1 as the padding value
        # For RoBERTa style input: <s> u1 </s> </s> u2 </s> </s> u3 </s> ...
        inputs = np.ones((len(uttrs), max_length), dtype = np.int32)

        # These three are always associated with MEED (aka RoBERTa style input).
        input_segments = np.zeros((len(uttrs), max_length), dtype = np.int32)
        input_emots = np.zeros((len(uttrs), max_length), dtype = np.int32)
        target_segments = np.zeros((len(uttrs), max_length), dtype = np.int32)
        target_emots = np.zeros(len(uttrs), dtype = np.int32)

        # These two are always the same for any input style: <s> target </s>
        targets_i = np.ones((len(uttrs), max_length), dtype = np.int32)
        targets_r = np.ones((len(uttrs), max_length), dtype = np.int32)

        n = 0
        indices = []
        for ind, (s, t) in tqdm(enumerate(dialogs), total = len(dialogs)):
            if t - s < 2:
                continue

            if cascade:
                uttr_ids = tokenizer.encode(uttrs[s])

                inp_ids = [SOS_ID] + uttr_ids + [EOS_ID]
                inp_seg_ids = [0] * (len(uttr_ids) + 2)
                inp_emots = [uttr_emots[s,-1]] * (len(uttr_ids) + 2)

                for i in range(s + 1, t):
                    u = ' '.join(uttrs[s:i])
                    if len(u.split()) > max_length: break

                    uttr_ids = tokenizer.encode(uttrs[i])
                    tar_ids = [SOS_ID] + uttr_ids + [EOS_ID]
                    tar_seg_ids = [(i - s) % 2] * (len(uttr_ids) + 2)

                    if (len(inp_ids) <= max_length and len(tar_ids) - 1 <= max_length):
                        inputs[n,:len(inp_ids)] = inp_ids
                        input_segments[n,:len(inp_seg_ids)] = inp_seg_ids
                        input_emots[n,:len(inp_ids)] = inp_emots
                        target_emots[n] = uttr_emots[i,-1]
                        targets_i[n,:len(tar_ids)-1] = tar_ids[:-1]
                        targets_r[n,:len(tar_ids)-1] = tar_ids[1:]
                        target_segments[n,:len(tar_seg_ids)] = tar_seg_ids
                        n += 1

                    inp_ids += ([EOS_ID] + uttr_ids + [EOS_ID])
                    inp_seg_ids += [(i - s) % 2] * (len(uttr_ids) + 2)
                    inp_emots += [uttr_emots[i,-1]] * (len(uttr_ids) + 2)
            else:
                u = ' '.join(uttrs[s:t-1])
                if len(u.split()) > max_length: continue

                uttr_ids = tokenizer.encode(uttrs[s])
                inp_ids = [SOS_ID] + uttr_ids + [EOS_ID]
                inp_seg_ids = [0] * (len(uttr_ids) + 2)
                inp_emots = [uttr_emots[s,-1]] * (len(uttr_ids) + 2)
                for i in range(s + 1, t - 1):
                    uttr_ids = tokenizer.encode(uttrs[i])
                    inp_ids += ([EOS_ID] + uttr_ids + [EOS_ID])
                    inp_seg_ids += [(i - s) % 2] * (len(uttr_ids) + 2)
                    inp_emots += [uttr_emots[i,-1]] * (len(uttr_ids) + 2)

                uttr_ids = tokenizer.encode(uttrs[t - 1])
                tar_ids = [SOS_ID] + uttr_ids + [EOS_ID]
                tar_seg_ids = [(t - s - 1) % 2] * (len(uttr_ids) + 2)

                if (len(inp_ids) <= max_length and len(tar_ids) - 1 <= max_length):
                    inputs[n,:len(inp_ids)] = inp_ids
                    input_segments[n,:len(inp_seg_ids)] = inp_seg_ids
                    input_emots[n,:len(inp_ids)] = inp_emots
                    target_emots[n] = uttr_emots[t-1,-1]
                    targets_i[n,:len(tar_ids)-1] = tar_ids[:-1]
                    targets_r[n,:len(tar_ids)-1] = tar_ids[1:]
                    target_segments[n,:len(tar_seg_ids)] = tar_seg_ids
                    indices.append(ind)
                    n += 1

        print('Created dataset with {} examples.'.format(n))
        print('Number of indices: {}'.format(len(indices)))

        prompts = []
        for ind in indices:
            prompts.append((all_context_emots[ind], all_prompts[ind]))

        inputs = inputs[:n,:]
        input_segments = input_segments[:n,:]
        input_emots = input_emots[:n,:]
        targets_i = targets_i[:n,:]
        targets_r = targets_r[:n,:]
        target_segments = target_segments[:n,:]
        target_emots = target_emots[:n]

        if not cascade and index is not None:
            inputs = inputs[index]
            input_segments = input_segments[index]
            input_emots = input_emots[index]
            targets_i = targets_i[index]
            targets_r = targets_r[index]
            target_segments = target_segments[index]
            target_emots = target_emots[index]

        return (tf.data.Dataset.from_tensor_slices(inputs),
                tf.data.Dataset.from_tensor_slices(input_segments),
                tf.data.Dataset.from_tensor_slices(input_emots),
                tf.data.Dataset.from_tensor_slices(targets_i),
                tf.data.Dataset.from_tensor_slices(targets_r),
                tf.data.Dataset.from_tensor_slices(target_segments),
                tf.data.Dataset.from_tensor_slices(target_emots)), prompts, inputs.shape[0]

    train_dataset, _, _ = create_dataset(join(path, 'train'))
    val_dataset, _, _ = create_dataset(join(path, 'validation'))
    test_dataset, prompts, N = create_dataset(join(path, 'test'), cascade = False)

    train_dataset = tf.data.Dataset.zip(train_dataset).shuffle(buffer_size).batch(batch_size)
    val_dataset = tf.data.Dataset.zip(val_dataset).batch(batch_size)
    test_dataset = tf.data.Dataset.zip(test_dataset).batch(batch_size)

    return train_dataset, val_dataset, test_dataset, prompts, N'''
