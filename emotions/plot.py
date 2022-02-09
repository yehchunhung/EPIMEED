import csv
import pandas as pd
import numpy as np


ED_emotions = ['afraid', 'angry','annoyed',
            'anticipating','anxious','apprehensive','ashamed','caring','confident','content','devastated','disappointed',
            'disgusted','embarrassed','excited','faithful','furious','grateful','guilty','hopeful','impressed','jealous',
            'joyful','lonely','nostalgic','prepared','proud','sad','sentimental','surprised','terrified','trusting',
            'agreeing','acknowledging','encouraging','consoling','sympathizing','suggesting','questioning','wishing','neutral']

ED_emotions_ordered = ['prepared', 'anticipating', 'hopeful', 'proud', 'excited', 'joyful', 'content', 'caring', 'grateful', 'trusting', 'confident', 'faithful', 'impressed', 'surprised', 'terrified', 'afraid', 'apprehensive', 'anxious', 'embarrassed', 'ashamed', 'devastated', 'sad', 'disappointed', 'lonely', 'sentimental', 'nostalgic', 'guilty', 'disgusted', 'furious', 'angry', 'annoyed', 'jealous', 'agreeing', 'acknowledging', 'encouraging', 'consoling', 'sympathizing', 'suggesting', 'questioning', 'wishing', 'neutral']


f_out = open('./performance-scores/plot-data-heldout-all.log', 'a')

method_list = ['argmax', 'prob-sampled']

for dataset in ['os-dramatic-rest', 'ed', 'osed-dramatic']:

    for model_name in ['os-dramatic-rest', 'os-dramatic-rest-ed', 'os-dramatic-rest-osed-dramatic']:

            dictionary = {}

            for version in ['ground', 'meed2', 'simple', 'complex']:

                
                    
                if version == 'simple' or version == 'complex':
                    #for method in ['argmax', 'prob-sampled']:
                    for method in method_list:

                        ED_arr = []
                        for i in range(41):
                            ED_arr.append(0)

                        file_path = './model-{}/{}-heldout-all/{}_test_int.csv'.format(model_name, method, dataset)

                        pred_df = pd.read_csv(file_path)
                    
                        emo_list = pred_df[''+version].tolist()

                        for emo in emo_list:
                            emo_label = ED_emotions[emo]
                            emo_ind_mapped = ED_emotions_ordered.index(emo_label)
                            ED_arr[emo] += 1

                        dictionary[''+version+' ('+method+')'] = ED_arr

                else:

                    ED_arr = []
                    for i in range(41):
                        ED_arr.append(0)

                    if version == 'meed2':

                        file_path = './model-{}/meed2/{}_test_int.csv'.format(model_name, dataset)

                    elif version == 'ground':

                        file_path = './model-{}/argmax-heldout-all/{}_test_int.csv'.format(model_name, dataset)

                    pred_df = pd.read_csv(file_path)
                    
                    emo_list = pred_df[''+version].tolist()

                    for emo in emo_list:
                        emo_label = ED_emotions[emo]
                        emo_ind_mapped = ED_emotions_ordered.index(emo_label)
                        ED_arr[emo] += 1

                    dictionary[''+version] = ED_arr
                
            print("===")
            f_out.write("===\n")
            print(dataset, model_name)
            f_out.write("Dataset: "+dataset+", Model:"+model_name+"\n")
            print("===")
            f_out.write("===\n\n")
            print()

            for key, value in dictionary.items():
                print(key, " = ", str(value))
                f_out.write(""+key+" = "+str(value)+"\n")

            f_out.write("\n\n\n")

            print()


f_out.close()    



    