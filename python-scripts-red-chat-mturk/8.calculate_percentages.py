import boto3
import csv
import json
from datetime import datetime
import numpy as np

# --------------------- MAIN CODE --------------------- 

folder_name = "HITs"

answer_dict = {}

#model_list = ['meed', 'blender', 'heal-ranked', 'heal-ranked-cosine']
model_list = ['emoprepend', 'meed2', 'meed2+', 'meed2++']

for model in model_list:
    answer_dict[model] = {
        "good": 0,
        "okay": 0,
        "bad": 0,
        "agreed": 0,
        "total": 0,
        "agreed_percentage": 0
    }

infile = './'+folder_name+'/Answers/answers_3.csv'

with open(infile, 'r') as datafile:

    #writer.writerow(["HITId", "Part_no", "Dialogue_no", "Dialogue_ID", "Model", "Agreed rating", 'Worker 1 answer', 'Worker 2 answer', 'Worker 3 answer'])
        
    readCSVdata = csv.reader(datafile, delimiter=',')
    count = 0

    for row in readCSVdata:
        if count == 0:
            print()
        else:
            model = row[4]
            agreed_rating = "-"
            agreed_rating = row[5].strip()
            if agreed_rating == "-":
                answer_dict[model]["total"] += 1
            else:
                answer_dict[model][agreed_rating] += 1
                answer_dict[model]["agreed"] += 1
                answer_dict[model]["total"] += 1

            answer_dict[model]["agreed_percentage"] = (answer_dict[model]["agreed"] / answer_dict[model]["total"])*100

        count += 1

print("model\tgood\tokay\tbad\tagreed\ttoal\tagreed_percentage")
for key, value in answer_dict.items():
    good_percentage = (value["good"] / value["agreed"]) * 100
    okay_percentage = (value["okay"] / value["agreed"]) * 100
    bad_percentage = (value["bad"] / value["agreed"]) * 100
    print(key, "\t", round(good_percentage,4), "\t", round(okay_percentage,4), "\t", round(bad_percentage, 4), "\t", value["agreed"], "\t", value["total"], "\t", value["agreed_percentage"])

