import boto3
import csv
import json
from datetime import datetime
import numpy as np
import json

# --------------------- MAIN CODE --------------------- 

total_agreed = 0
total_agreed_all = 0

folder_name = "HITs"

#model_list = ['meed', 'blender', 'heal-ranked', 'heal-ranked-cosine']
model_list = ['emoprepend', 'meed2', 'meed2+', 'meed2++']

infile = './'+folder_name+'/Answers/answers_1.csv'
outfile = './'+folder_name+'/Answers/answers_2.csv'

with open(infile, 'r') as datafile:

    with open(outfile, 'a') as csvfile:
        
        readCSVdata = csv.reader(datafile, delimiter=',')
        writer = csv.writer(csvfile, delimiter=str(','), lineterminator='\n')

        #header = ["HITId", "Part_no", "Question_no"] + workers
        count = 0

        for row in readCSVdata:
            if count == 0:
                writer.writerow(["HITId", "Part_no", "Dialogue_no", "Dialogue_ID", "model", 'Worker 1 answer', 'Worker 2 answer', 'Worker 3 answer'])
            else:
                
                pre_row = [row[0], row[1], row[2], row[3]]

                for model in model_list:
                    #print("===")
                    #print(model)
                    new_row = pre_row + [model]

                    worker_ans = []
                    for j in range(4, 7):
                        if ""+row[j] != "-1":
                            #print(row[j])
                            worker_answer = json.loads(row[j])
                            model_found = False 
                            for category in ['good', 'okay', 'bad']:
                                for obj in worker_answer[category]:
                                    if model in obj["model"]:
                                        worker_ans.append(category)
                                        model_found = True
                                        break
                                if model_found:
                                    break
                        else:
                            worker_ans.append(-1)

                    write_row = new_row + worker_ans
                    writer.writerow(write_row)

            count += 1

print(count)


