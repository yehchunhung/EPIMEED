import boto3
import csv
import json
from datetime import datetime
import numpy as np
import json

import xml.etree.ElementTree as ET

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

    
avg_time = 0
workers = []


'''
# response: 
{
    'NextToken': 'p1:2Gr85dbqp34nrRRjUzzAVOYbkafpceMmCKLnqNDDUzW0yESWDIBMpeJxyK5wkw==', 
    'NumResults': 1,
    'Assignments': [

        {'AssignmentId': '34S6N1K2ZVKFTI4TGA8HNNPV756HLZ', 
        'WorkerId': 'A134T7UJ7WTSL5', 
        'HITId': '3VI0PC2ZAYLIZ99B841UU6M3RYEOX1', 
        'AssignmentStatus': 'Submitted', 
        'AutoApprovalTime': datetime.datetime(2021, 2, 14, 21, 38, 41, tzinfo=tzlocal()), 
        'AcceptTime': datetime.datetime(2021, 2, 11, 21, 36, 26, tzinfo=tzlocal()), 
        'SubmitTime': datetime.datetime(2021, 2, 11, 21, 38, 41, tzinfo=tzlocal()), 
        'Answer': '<?xml version="1.0" encoding="ASCII"?><QuestionFormAnswers xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2005-10-01/QuestionFormAnswers.xsd"><Answer><QuestionIdentifier>part_no</QuestionIdentifier><FreeText>0</FreeText></Answer><Answer><QuestionIdentifier>elapsed_time</QuestionIdentifier><FreeText>1.8236333333333334</FreeText></Answer><Answer><QuestionIdentifier>dialogID_0</QuestionIdentifier><FreeText>ed | ed | 1</FreeText></Answer><Answer><QuestionIdentifier>answer_0</QuestionIdentifier><FreeText>{"good":[{"text":"That is so true. I wish you the best of luck.","model":"plain"}],"okay":[{"text":"I am sorry to hear that.","model":"alexa"},{"text":"That is a great feeling to have.","model":"meed2 | complex (argmax)"}],"bad":[{"text":"I am not sure what you are talking about.","model":"complex (prob-sampled)"}]}</FreeText></Answer><Answer><QuestionIdentifier>dialogID_1</QuestionIdentifier><FreeText>ed | ed | 2</FreeText></Answer><Answer><QuestionIdentifier>answer_1</QuestionIdentifier><FreeText>{"good":[{"text":"That\'s great. I\'m happy for you.","model":"complex (prob-sampled)"}],"okay":[{"text":"That\'s great. I hope you have a great family.","model":"alexa"},{"text":"I\'m so sorry to hear that.","model":"plain"}],"bad":[{"text":"I\'m glad you were able to help.","model":"meed2"},{"text":"That\'s really nice of them.","model":"complex (argmax)"}]}</FreeText></Answer><Answer><QuestionIdentifier>dialogID_2</QuestionIdentifier><FreeText>ed | ed | 3</FreeText></Answer><Answer><QuestionIdentifier>answer_2</QuestionIdentifier><FreeText>{"good":[{"text":"That is true, I hope it works out for you.","model":"alexa"}],"okay":[{"text":"I believe in you","model":"meed2"},{"text":"I agree, I agree with you.","model":"plain"}],"bad":[{"text":"Exactly, I get that entirely!","model":"real"},{"text":"That is true. I don\'t know what you\'re talking about.","model":"complex (argmax) | complex (prob-sampled)"}]}</FreeText></Answer><Answer><QuestionIdentifier>dialogID_3</QuestionIdentifier><FreeText>ed | ed | 4</FreeText></Answer><Answer><QuestionIdentifier>answer_3</QuestionIdentifier><FreeText>{"good":[{"text":"Do you have a favorite place to go?","model":"plain"}],"okay":[{"text":"That sounds like a good plan","model":"alexa"}],"bad":[{"text":"Did you have a good time?","model":"meed2"},{"text":"Oh, no! I\'ve never been there before.","model":"complex (argmax) | complex (prob-sampled)"}]}</FreeText></Answer><Answer><QuestionIdentifier>dialogID_4</QuestionIdentifier><FreeText>ed | ed | 5</FreeText></Answer><Answer><QuestionIdentifier>answer_4</QuestionIdentifier><FreeText>{"good":[{"text":"What kind of car is it?","model":"meed2"}],"okay":[{"text":"That\'s good to hear.  I\'m glad it worked out for you.","model":"alexa"},{"text":"What kind of car is it?","model":"plain"}],"bad":[{"text":"Oh, no!  What kind of car was it?","model":"complex (prob-sampled)"},{"text":"That\'s so sweet.  It\'s always nice to have a new car.","model":"complex (argmax)"}]}</FreeText></Answer><Answer><QuestionIdentifier>dialogID_5</QuestionIdentifier><FreeText>ed | ed | 6</FreeText></Answer><Answer><QuestionIdentifier>answer_5</QuestionIdentifier><FreeText>{"good":[{"text":"Oh, no! I\'m so sorry to hear that!","model":"plain"}],"okay":[{"text":"Oh, no! I\'m so sorry to hear that!","model":"alexa | complex (argmax)"},{"text":"Oh, no! Are you going to tell me about it?","model":"meed2"}],"bad":[{"text":"Wow, that was a long time ago!","model":"complex (prob-sampled)"},{"text":"Wow, what happened?","model":"real"}]}</FreeText></Answer><Answer><QuestionIdentifier>dialogID_6</QuestionIdentifier><FreeText>ed | ed | 7</FreeText></Answer><Answer><QuestionIdentifier>answer_6</QuestionIdentifier><FreeText>{"good":[{"text":"That sounds like a lot of fun! I hope you have a great time!","model":"plain"}],"okay":[{"text":"That sounds like so much fun! Where are you going?","model":"meed2 | complex (argmax) | complex (prob-sampled)"}],"bad":[{"text":"Oh, wow! I\'m so sorry to hear that! I hope you have a great time!","model":"alexa"}]}</FreeText></Answer><Answer><QuestionIdentifier>dialogID_7</QuestionIdentifier><FreeText>ed | ed | 8</FreeText></Answer><Answer><QuestionIdentifier>answer_7</QuestionIdentifier><FreeText>{"good":[{"text":"No, they didn\'t give me any feedback","model":"plain"}],"okay":[{"text":"No, they didn\'t give me anything","model":"meed2"}],"bad":[{"text":"No, I don\'t think so","model":"alexa | complex (argmax) | complex (prob-sampled)"}]}</FreeText></Answer><Answer><QuestionIdentifier>dialogID_8</QuestionIdentifier><FreeText>ed | ed | 9</FreeText></Answer><Answer><QuestionIdentifier>answer_8</QuestionIdentifier><FreeText>{"good":[{"text":"That is so awesome!  Congratulations!","model":"meed2"}],"okay":[{"text":"Ha!  What a neat day to announce it &#8212; did they do any special kind of announcement?  I see so many videos of interesting and neat ways to make that announcement.","model":"real"},{"text":"That is so exciting!  I am so happy for you!","model":"plain"}],"bad":[{"text":"That is awesome!  What kind of baby is it?","model":"alexa"},{"text":"Oh, wow!  That is such a great surprise!  How old is she?","model":"complex (argmax) | complex (prob-sampled)"}]}</FreeText></Answer><Answer><QuestionIdentifier>dialogID_9</QuestionIdentifier><FreeText>ed | ed | 10</FreeText></Answer><Answer><QuestionIdentifier>answer_9</QuestionIdentifier><FreeText>{"good":[{"text":"What did they come out for?","model":"complex (argmax)"}],"okay":[{"text":"Maybe next time.","model":"alexa"}],"bad":[{"text":"I would have been scared too.","model":"meed2 | complex (prob-sampled)"},{"text":"What movie did you see?","model":"plain"}]}</FreeText></Answer><Answer><QuestionIdentifier>earnings</QuestionIdentifier><FreeText>0.4</FreeText></Answer></QuestionFormAnswers>'
        }
    ], 
    'ResponseMetadata': {
        'RequestId': 'b344e6ae-9c7a-4083-a985-730aee0d74d9', 
        'HTTPStatusCode': 200, 
        'HTTPHeaders': {
                'x-amzn-requestid': 'b344e6ae-9c7a-4083-a985-730aee0d74d9', 
                'content-type': 'application/x-amz-json-1.1', 
                'content-length': '6593', 
                'date': 'Thu, 11 Feb 2021 20:39:34 GMT'
        }, 
        'RetryAttempts': 0
    }
}
'''

for part_no in range(0, len(HITIds)):

    hit = HITIds[part_no]

    response = client.list_assignments_for_hit(
        HITId=hit,
        MaxResults=100,
        AssignmentStatuses=[
            #'Submitted','Approved','Rejected',
            'Submitted','Approved',
        ]
    )

    #print(response)

    for assignment in response['Assignments']:

        #print(assignment)

        answer_obj = {}

        root = ET.fromstring(assignment['Answer'])
        for child in root:
            key = ''
            for sub in child:
                if 'QuestionIdentifier' in sub.tag:
                    key = sub.text
                if 'FreeText' in sub.tag:
                    answer_obj[key] = sub.text

        if assignment['WorkerId'] not in workers:
            workers.append(assignment['WorkerId'])


print(workers)

final_arr = []

num_HITs = len(HITIds)
per_HIT = 10

header_pre = ["HITId", "Part_no", "Dialogue_no", "Dialogue_ID"]
header = header_pre + workers

for i in range(0, per_HIT*num_HITs):
    arr = [-1, -1, -1, -1]
    for j in range(0, len(workers)):
        arr.append(-1)
    final_arr.append(arr)


for part_no in range(0, len(HITIds)):

    hit = HITIds[part_no]

    for k in range(0, per_HIT):
        final_arr[per_HIT*part_no + k][0] = hit
        final_arr[per_HIT*part_no + k][1] = part_no
        final_arr[per_HIT*part_no + k][2] = k

    response = client.list_assignments_for_hit(
        HITId=hit,
        MaxResults=100,
        AssignmentStatuses=[
            #'Submitted','Approved','Rejected',
            'Submitted','Approved',
        ]
    )

    for assignment in response['Assignments']:


        answer_obj = {}

        root = ET.fromstring(assignment['Answer'])
        for child in root:
            key = ''
            for sub in child:
                if 'QuestionIdentifier' in sub.tag:
                    key = sub.text
                if 'FreeText' in sub.tag:
                    answer_obj[key] = sub.text



        # if float(answer_obj['earnings']) > 0.4:
            
        #   worker_index = workers.index(assignment['WorkerId'])

        #   for question_no in range(0, 10):
        #     final_arr[per_HIT*part_no + question_no][3] = answer_obj['dialogID_'+str(question_no)]
        #     final_arr[per_HIT*part_no + question_no][worker_index + len(header_pre)] = answer_obj['answer_'+str(question_no)]

        
        worker_index = workers.index(assignment['WorkerId'])

        for question_no in range(0, 10):
            final_arr[per_HIT*part_no + question_no][3] = answer_obj['dialogID_'+str(question_no)]
            final_arr[per_HIT*part_no + question_no][worker_index + len(header_pre)] = answer_obj['answer_'+str(question_no)]

        '''root = ET.fromstring(assignment['Answer'])
        
        for child in root:
            key = ''
            for sub in child:
                if 'QuestionIdentifier' in sub.tag:
                    key = sub.text
                if 'FreeText' in sub.tag:
                    answer = sub.text

                    #append to array:
                    worker_index = workers.index(assignment['WorkerId'])
                    if "dialogID" in key and answer != None:
                        key_arr = key.split("_")
                        question_no = int(key_arr[1])
                        final_arr[per_HIT*part_no + question_no][3] = answer

                    
                    elif "answer" in key and answer != None: 
                        key_arr = key.split("_")
                        question_no = int(key_arr[1])

                        final_arr[per_HIT*part_no + question_no][worker_index + len(header_pre)] = answer'''


with open('./'+folder_name+'/Answers/answers_0.csv', 'a') as csvfile:
    
    writer = csv.writer(csvfile, delimiter=str(','), lineterminator='\n')
    writer.writerow(header)
    print(header)

    for arr in final_arr:
        print(arr)
        writer.writerow(arr)


