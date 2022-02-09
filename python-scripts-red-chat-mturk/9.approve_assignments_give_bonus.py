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
approved_assignments = 0
bonus_given = 0

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

    for assignment in response['Assignments']:

        assignment_id = assignment['AssignmentId']
        worker_id = assignment['WorkerId']

        answer_obj = {}

        root = ET.fromstring(assignment['Answer'])
        for child in root:
            key = ''
            for sub in child:
                if 'QuestionIdentifier' in sub.tag:
                    key = sub.text
                if 'FreeText' in sub.tag:
                    answer_obj[key] = sub.text

        bonus_earned = False
        if float(answer_obj['earnings']) > 0.4:
            bonus_earned = True

        if assignment['AssignmentStatus'] == 'Submitted':
          response = client.approve_assignment(
              AssignmentId=assignment_id,
              RequesterFeedback='Thank you for your work!',
              OverrideRejection=False
          )
          print("Assignment " + assignment_id + "by worker "+ worker_id+" is approved!")
          print(response)
          print()
          approved_assignments += 1

        if bonus_earned == True:
          response = client.send_bonus(
              WorkerId=worker_id,
              BonusAmount="0.1",
              AssignmentId=assignment_id,
              Reason='Bonus for getting at least 2 out of 3 bonus tasks correct!',
          )
          print("Assignment " + assignment_id + "by worker "+ worker_id+" is given bonus!")
          print(response)
          print()
          bonus_given += 1

print("Bonus percentage: ", ((bonus_given/total)*100))
print("Approved assignments: ", approved_assignments)
print("Total: ", total)
    