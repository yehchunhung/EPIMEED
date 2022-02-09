import boto3
import csv
import json
from datetime import datetime
import numpy as np
import os.path
from operator import *
import plotly.graph_objects as go

import xml.etree.ElementTree as ET

# --------------------- HELPER METHODS --------------------- 

# Quick method to encode url parameters
def encode_get_parameters(baseurl, arg_dict):
    queryString = baseurl + "?"
    for indx, key in enumerate(arg_dict):
        queryString += str(key) + "=" + str(arg_dict[key])
        if indx < len(arg_dict)-1:
            queryString += "&"
    return queryString

# --------------------- MAIN CODE --------------------- 

aws_access_key_id = 'YOUR_ACCESS_ID'
aws_secret_access_key = 'YOUR_SECRET_KEY'

with open('./rootkey.csv', 'r') as infile:
  readCSV = csv.reader(infile, delimiter=',')
  count = 0
  for row in readCSV:
    if count == 0:
      aws_access_key_id = row[0].strip().split('=')[1]
    elif count == 1:
      aws_secret_access_key = row[0].strip().split('=')[1]
    count += 1

config = json.load(open("config.json"))
HIT = config["HIT"]
params_to_encode = {}
folder_name = ""

if HIT["USE_SANDBOX"]:
    print("review HIT on sandbox")
    endpoint_url = "https://mturk-requester-sandbox.us-east-1.amazonaws.com"
    mturk_form_action = "https://workersandbox.mturk.com/mturk/externalSubmit"
    mturk_url = "https://workersandbox.mturk.com/"
    external_submit_endpoint = "https://workersandbox.mturk.com/mturk/externalSubmit"
    folder_name = "SandboxHITs"
else:
    print("review HIT on mturk")
    endpoint_url = "https://mturk-requester.us-east-1.amazonaws.com"
    mturk_form_action = "https://www.mturk.com/mturk/externalSubmit"
    mturk_url = "https://worker.mturk.com/"
    external_submit_endpoint = "https://www.mturk.com/mturk/externalSubmit"
    folder_name = "HITs"

client = boto3.client(
    'mturk',
    endpoint_url=endpoint_url,
    region_name=HIT["REGION_NAME"],
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)

# This will return $10,000.00 in the MTurk Developer Sandbox
print(client.get_account_balance()['AvailableBalance'])

# create HITs
print("Reviewing HITs...")

HITIds = [
    '36GJS3V78VRTCCNO3HA5QZAELECGJQ',
    '30U1YOGZGAXJZVWNOLIBCFY4459DST',
    '3QE4DGPGBRC39U430AN1KXLV3HM4GN',
    '3OLZC0DJ8JGGZYTDAUOREZYOVEBIV5',
    '3QQUBC64ZEF0HO9Z7P217SRQXODNXT',
    '3L2OEKSTW9B4EMC56JZK8984AZ88YW',
    '3BPP3MA3TCL1NOFX07WU6B2UCN2LEG',
    '388FBO7JZRUPHTZ9CLO3Q6YZZU7NYZ',
    '3S8A4GJRD3406EYC8TF2HW8JMNG6VZ',
    '3HEM8MA6H9DGBCK01QK1H90S00RQPV',
    '3TL87MO8CMQTT2FTVIG0OTNUOOHLFY',
    '38RHULDV9YGTENTI04TN01H22NDIWF',
    '3QTFNPMJC6JPMJ9E87QM4D715OBNZQ',
    '3CESM1J3EI4DL3YHY3KY3YRXTWI6W9',
    '33NOQL7T9O04JCMA2513MGH6GTC8ZN',
    '3WPCIUYH1A9KEV92DLN1MKUQ6ONDTI',
    '304QEQWKZPLA2J305SD7D34KDRXO0E',
    '3IWA71V4TIHSA33788GK5H542YO6X2',
    '3UDTAB6HH607VWZU33UPGD9VISB90L',
    '3P4C70TRMRIZ72BO62MI6EN6P3HLGX'
]

    
total = 0
avg_time = 0
no_of_bonus_earned = 0

worker_dict = {}

for hit in HITIds:

    response = client.list_assignments_for_hit(
        HITId=hit,
        MaxResults=10,
        AssignmentStatuses=[
            #'Submitted','Approved','Rejected',
            'Submitted','Approved'
        ]
    )

    total += int(response['NumResults'])

    duration_arr = []

    for assignment in response['Assignments']:

        duration = ( assignment['SubmitTime'].timestamp() - assignment['AcceptTime'].timestamp() ) / 60

        answer_obj = {}

        root = ET.fromstring(assignment['Answer'])
        for child in root:
            key = ''
            for sub in child:
                if 'QuestionIdentifier' in sub.tag:
                    key = sub.text
                if 'FreeText' in sub.tag:
                    answer_obj[key] = sub.text

        duration_ours = float(answer_obj['elapsed_time'])
        avg_time += duration_ours

        bonus_earned = False
        if float(answer_obj['earnings']) > 0.4:
            bonus_earned = True
            no_of_bonus_earned += 1

        if assignment['WorkerId'] in worker_dict:
            worker_dict[assignment['WorkerId']].append({
                "AssignmentId": assignment['AssignmentId'],
                "Duration": duration,
                "Duration_ours": duration_ours,
                "Bonus": bonus_earned,
                })
        else:
            worker_dict[assignment['WorkerId']] = []
            worker_dict[assignment['WorkerId']].append({
                "AssignmentId": assignment['AssignmentId'],
                "Duration": duration,
                "Duration_ours": duration_ours,
                "Bonus": bonus_earned,
                })

#print(worker_dict)
print("Total so far: " + str(total)) # Need 30
if (total != 0):
    print("Average time per HIT: " + str(avg_time/total))
    print("Percentage of bonuses earned: " + str(no_of_bonus_earned/total))


list_of_blocked_workers = []
if os.path.isfile('./'+folder_name+'/Blocked/blocked_worker_ids.txt'):
    f = open('./'+folder_name+'/Blocked/blocked_worker_ids.txt', "r")
    txt = f.read()
    list_of_blocked_workers = txt.split(",")
    print("No. of blocked workers: ", len(list_of_blocked_workers)-1)
else:
    print("File doesn't exist!")
    print("No. of blocked workers: ", len(list_of_blocked_workers))

print("No. of workers: ", len(worker_dict))

# ============ Check and block workers ============

min_time_per_HIT = 100
max_time_per_HIT = -1

worker_statistics_dict = {}
assignment_statistics_dict = {}

for key, value in worker_dict.items():
    
    worker_id = key
    no_of_hits = len(value)

    avg_duration_per_worker = 0

    no_of_bonus_earned = 0

    for assignment in value:
        assignment_id = assignment["AssignmentId"]
        duration_ours = assignment["Duration_ours"]
        if duration_ours >= 2 and duration_ours < min_time_per_HIT:
            min_time_per_HIT = duration_ours
        if duration_ours > max_time_per_HIT:
            max_time_per_HIT = duration_ours
        avg_duration_per_worker += duration_ours
        
        bonus_earned = 0
        if assignment["Bonus"] == True:
            no_of_bonus_earned += 1
            bonus_earned = 1
        
        assignment_statistics_dict[assignment_id] = {
          "time_taken": duration_ours,
          "bonus_earned": bonus_earned
        }
        
        
    avg_duration_per_worker = (avg_duration_per_worker/len(value))
    percentage_of_bonus_earned = (no_of_bonus_earned/len(value))*100

    #print(worker_id, no_of_hits, avg_duration_per_worker, percentage_of_bonus_earned)
    #print()

    worker_statistics_dict[worker_id] = {
        "no_of_hits": no_of_hits,
        "avg_duration": avg_duration_per_worker,
        "bonus_percentage": percentage_of_bonus_earned
    }

print("Minimum time taken per HIT: ", min_time_per_HIT)
print("Maximum time taken per HIT: ", max_time_per_HIT)

worker_arr = []
num_hits_arr = []
avg_duration_arr = []
bonus_arr = []

for element in sorted(worker_statistics_dict.items(),key=lambda x:-getitem(x[1],'no_of_hits')):
    #print(element[0])
    #print(element[1])
    worker_arr.append(element[0])
    num_hits_arr.append(element[1]["no_of_hits"])
    avg_duration_arr.append(element[1]["avg_duration"])
    bonus_arr.append(element[1]["bonus_percentage"])

fig1 = go.Figure()
fig1.add_trace(go.Bar(
    x=worker_arr,
    y=num_hits_arr,
    name='No. of assignments per worker',
    #marker_color=ED_color
))

# Here we modify the tickangle of the xaxis, resulting in rotated labels.
fig1.update_layout(barmode='group', 
                  xaxis_tickangle=-90, 
                  #xaxis={'categoryorder':'total descending'},
                  font = {'family': "Times", 'size': 16, 'color': "Black"},
                  xaxis_title="WorkerID",
                  yaxis_title="No. of assignments submitted",
                  #xaxis={'categoryorder':'total ascending'}
                  #xaxis={
                  #          "tickfont": {'size':4},
                  #      }
                  )
fig1.show()

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=worker_arr,
    y=avg_duration_arr,
    name='Avg. duration per HIT for each worker',
    #marker_color=ED_color
))

# Here we modify the tickangle of the xaxis, resulting in rotated labels.
fig2.update_layout(barmode='group', 
                  xaxis_tickangle=-90, 
                  #xaxis={'categoryorder':'total descending'},
                  font = {'family': "Times", 'size': 16, 'color': "Black"},
                  xaxis_title="WorkerID",
                  yaxis_title="Avg. duration per HIT (min.)",
                  #xaxis={'categoryorder':'total ascending'}
                  #xaxis={
                  #          "tickfont": {'size':4},
                  #      }
                  )
fig2.show()

fig3 = go.Figure()
fig3.add_trace(go.Bar(
    x=worker_arr,
    y=bonus_arr,
    name='Percentage of bonuses earned by each worker',
    #marker_color=ED_color
))

# Here we modify the tickangle of the xaxis, resulting in rotated labels.
fig3.update_layout(barmode='group', 
                  xaxis_tickangle=-90, 
                  #xaxis={'categoryorder':'total descending'},
                  font = {'family': "Times", 'size': 16, 'color': "Black"},
                  xaxis_title="WorkerID",
                  yaxis_title="Percentage of bonuses earned",
                  #xaxis={'categoryorder':'total ascending'}
                  #xaxis={
                  #          "tickfont": {'size':4},
                  #      }
                  )
fig3.show()


assignment_id_arr = []
time_arr = []
bonus_assignment_arr = []

for element in sorted(assignment_statistics_dict.items(),key=lambda x:getitem(x[1],'time_taken')):
  assignment_id_arr.append(element[0])
  time_arr.append(element[1]["time_taken"])
  bonus_assignment_arr.append(element[1]["bonus_earned"])


fig4 = go.Figure()
fig4.add_trace(go.Bar(
    x=assignment_id_arr,
    y=time_arr,
    name='Time taken per each assignment',
    #marker_color=ED_color
))

# Here we modify the tickangle of the xaxis, resulting in rotated labels.
fig4.update_layout(barmode='group', 
                  xaxis_tickangle=-90, 
                  #xaxis={'categoryorder':'total descending'},
                  font = {'family': "Times", 'size': 16, 'color': "Black"},
                  xaxis_title="AssignmentID",
                  yaxis_title="Time taken (min.)",
                  #xaxis={'categoryorder':'total ascending'}
                  #xaxis={
                  #          "tickfont": {'size':4},
                  #      }
                  )
fig4.show()


fig5 = go.Figure()
fig5.add_trace(go.Bar(
    x=assignment_id_arr,
    y=bonus_assignment_arr,
    name='Bonus earned per each assignment',
    #marker_color=ED_color
))

# Here we modify the tickangle of the xaxis, resulting in rotated labels.
fig5.update_layout(barmode='group', 
                  xaxis_tickangle=-90, 
                  #xaxis={'categoryorder':'total descending'},
                  font = {'family': "Times", 'size': 16, 'color': "Black"},
                  xaxis_title="AssignmentID",
                  yaxis_title="Bonus earned",
                  #xaxis={'categoryorder':'total ascending'}
                  #xaxis={
                  #          "tickfont": {'size':4},
                  #      }
                  )
fig5.show()

fig6 = go.Figure(data=[go.Histogram(x=time_arr)])
fig6.update_layout(
    xaxis_title="Assignment duration",
    yaxis_title="No. of assignments",
    font = {'family': "Times", 'size': 16, 'color': "Black"},
    )
fig6.show()

