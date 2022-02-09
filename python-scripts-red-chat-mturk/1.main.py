import boto3
import csv
import json
from boto.mturk.question import ExternalQuestion
import datetime

# --------------------- HELPER METHODS --------------------- 

# Quick method to encode url parameters
def encode_get_parameters(baseurl, arg_dict):
    queryString = baseurl + "?"
    for indx, key in enumerate(arg_dict):
        queryString += str(key) + "=" + str(arg_dict[key])
        if indx < len(arg_dict)-1:
            queryString += "&"
    return queryString

def myconverter(o):
    if isinstance(o, datetime.datetime):
        return o.__str__()

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
    print("create HIT on sandbox")
    endpoint_url = "https://mturk-requester-sandbox.us-east-1.amazonaws.com"
    mturk_form_action = "https://workersandbox.mturk.com/mturk/externalSubmit"
    mturk_url = "https://workersandbox.mturk.com/"
    external_submit_endpoint = "https://workersandbox.mturk.com/mturk/externalSubmit"
    folder_name = "SandboxHITs"
else:
    print("create HIT on mturk")
    endpoint_url = "https://mturk-requester.us-east-1.amazonaws.com"
    mturk_form_action = "https://www.mturk.com/mturk/externalSubmit"
    mturk_url = "https://worker.mturk.com/"
    external_submit_endpoint = "https://www.mturk.com/mturk/externalSubmit"
    folder_name = "HITs"

params_to_encode['host'] = external_submit_endpoint

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
print("Creating HITs...")

for i in range(0, 20):

    part = i
    base_url = "https://lia.epfl.ch/emotionrecog/" + str(part)
    encoded_url = encode_get_parameters(base_url, params_to_encode)
    print(encoded_url)
    question = ExternalQuestion(encoded_url, HIT["FrameHeight"])
    question_xml = question.get_as_xml()
    print(question_xml)

    new_hit = client.create_hit(
        MaxAssignments=HIT["MaxAssignments"],
        AutoApprovalDelayInSeconds=HIT["AutoApprovalDelayInSeconds"],
        LifetimeInSeconds=HIT["LifetimeInSeconds"],
        AssignmentDurationInSeconds=HIT["AssignmentDurationInSeconds"],
        Reward=HIT["Reward"],
        Title=HIT['Title'],
        Keywords=HIT["Keywords"],
        Description=HIT["Description"],
        Question=question_xml,
        RequesterAnnotation='PART:'+str(part),
        QualificationRequirements=[
            {
                'QualificationTypeId': '00000000000000000040', #no. of hits approved
                'Comparator': 'GreaterThanOrEqualTo',
                'IntegerValues': [
                    500,
                ],
                'RequiredToPreview': False,
                'ActionsGuarded': 'Accept'
            },
            {
                'QualificationTypeId': '000000000000000000L0', #percentage hits approved
                'Comparator': 'GreaterThanOrEqualTo',
                'IntegerValues': [
                    98,
                ],
                'RequiredToPreview': False,
                'ActionsGuarded': 'Accept'
            },
            {
                'QualificationTypeId': '00000000000000000060', #adult content
                'Comparator': 'EqualTo',
                'IntegerValues': [
                    1,
                ],
                'RequiredToPreview': False,
                'ActionsGuarded': 'Accept'
            },
            {
                'QualificationTypeId': '00000000000000000071', #country
                'Comparator': 'In',
                'LocaleValues': [
                    {
                        'Country': 'US',
                    },
                    {
                        'Country': 'AU',
                    },
                    {
                        'Country': 'CH',
                    },
                    {
                        'Country': 'GB',
                    },
                    {
                        'Country': 'NZ',
                    },
                    {
                        'Country': 'CA',
                    },
                ],
                'RequiredToPreview': False,
                'ActionsGuarded': 'Accept'
            },
        ],
    )

    with open('./'+folder_name+'/Descriptions/hit_description.txt', 'a', encoding='utf8') as outfile:
      outfile.write("HITId: " + new_hit['HIT']['HITId'] + '\n')
      outfile.write("A new HIT has been created. You can preview it here:"+ '\n')
      outfile.write(mturk_url + "mturk/preview?groupId=" + new_hit['HIT']['HITGroupId']+ '\n')

    with open('./'+folder_name+'/IDs/hit_ids.txt', 'a', encoding='utf8') as outfile:
      outfile.write("'" + new_hit['HIT']['HITId'] + "',\n")

    print("HITId: " + new_hit['HIT']['HITId'])
    print("A new HIT has been created. You can preview it here:")
    print(mturk_url + "mturk/preview?groupId=" + new_hit['HIT']['HITGroupId'])
    
    with open('./'+folder_name+'/FullHITs/part_'+str(part)+'.json', 'w', encoding='utf8') as outfile:
      json.dump(new_hit, outfile, ensure_ascii=False, default=myconverter)

    print(i)



