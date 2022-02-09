import csv
import pandas as pd
import numpy as np
import json

ED_emotions_ordered = ['prepared', 'anticipating', 'hopeful', 'proud', 'excited', 'joyful', 'content', 'caring', 'grateful', 'trusting', 'confident', 'faithful', 'impressed', 'surprised', 'terrified', 'afraid', 'apprehensive', 'anxious', 'embarrassed', 'ashamed', 'devastated', 'sad', 'disappointed', 'lonely', 'sentimental', 'nostalgic', 'guilty', 'disgusted', 'furious', 'angry', 'annoyed', 'jealous', 'agreeing', 'acknowledging', 'encouraging', 'consoling', 'sympathizing', 'suggesting', 'questioning', 'wishing', 'neutral']



for dataset in ['red+_mask']: 

    with open('original/all-' + dataset + '.csv', 'r') as infile:
        csv_reader = csv.reader(infile, delimiter = ',')
        next(csv_reader)

        count = 0
        dial_count = 1

        dialogs = []

        for row in csv_reader:
            if count == 0:
                id_ = dataset + " | " + str(dial_count)
                dial_count += 1
                turns = []
                responses = []

                dialog_arr = list(filter(None, row[0].split("\n")))

                for i in range(0, len(dialog_arr)):
                    turn_text = dialog_arr[i]
                    turns.append({"text": turn_text})

                real_response = {
                    "text": row[3].strip(),
                    "emotion": row[2].strip(),
                    "model": "real"
                }

            elif count in [1, 2, 3, 4]:

                responses.append({
                    "text": row[3].strip(),
                    "model": row[1].strip(),
                    "emotion": row[2].strip(),
                    "added": False
                    })


            count += 1

            if count == 5:

                new_responses = []

                for i in range(len(responses)):
                    res1 = responses[i]
                    if res1["added"] == False:

                        new_res = {
                            "text": res1["text"],
                            "model": res1["model"],
                            "emotion": res1["emotion"] 
                        }

                        for j in range(i+1, len(responses)):
                            res2 = responses[j]
                            if res2["added"] == False:
                                if res1["text"] == res2["text"] and res1["emotion"] == res2["emotion"]:
                                    new_res["model"] = new_res["model"] + " | " + res2["model"]
                                    res2["added"] = True

                        new_responses.append(new_res)
                        res1["added"] = True


                # append object
                dialogs.append({

                    "id": id_,
                    "turn_count": len(turns),
                    "turns": turns,
                    "real_response": real_response,
                    "responses": new_responses,
                    "contain_gold_turn": False,

                })

                count = 0
            

        json_obj = {
            "data": []
        }

        block_10 = {
            "dialogs": []
        }

        dialog_counter = 0
        for dialog in dialogs:
            dialog_counter += 1
            if dialog_counter % 3 == 0:
                dialog["contain_gold_turn"] = True

            block_10["dialogs"].append(dialog)


            if dialog_counter == 10:
                json_obj["data"].append(block_10)
                block_10 = {
                    "dialogs": []
                }
                dialog_counter = 0

        #print(json_obj)

        with open("json/" + dataset + '.json', 'w') as f:
            json.dump(json_obj, f)

        print("written: " + dataset)
                   


    