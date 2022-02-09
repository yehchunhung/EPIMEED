import csv
import pandas as pd
import numpy as np

'''
dataset = 'os-dramatic-rest'
model = 'os-dramatic-rest'

ground = [7744, 5208, 8857, 15201, 2227, 12624, 4987, 7331, 14192, 10020, 2787, 4264, 2206, 5997, 7878, 3806, 20713, 5526, 6314, 9620, 4190, 6165, 4834, 2660, 3755, 6156, 5528, 10567, 3361, 10005, 4499, 21591, 25132, 33183, 4022, 6228, 7089, 21223, 149282, 15843, 19272]
meed2 = [1088, 85, 26, 6416, 8, 768, 41, 353, 1493, 5112, 8, 1, 93, 21, 150, 722, 8480, 180, 423, 587, 123, 99, 141, 20, 331, 500, 300, 1568, 103, 70, 573, 11084, 2657, 9335, 0, 13, 312, 138, 455896, 10197, 2572]
simple_argmax = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 522087, 0, 0]
simple_prob_sampled = [8328, 5626, 9015, 15699, 2346, 12913, 5296, 7654, 14713, 10108, 2997, 4403, 2391, 5834, 8644, 4148, 21131, 5533, 6629, 9826, 4556, 6570, 4959, 2706, 3872, 6395, 6143, 11295, 3578, 10171, 5239, 21976, 25779, 34528, 4251, 6080, 6416, 20270, 138005, 15762, 20302]
complex_argmax = [214, 125, 294, 1218, 64, 472, 172, 253, 669, 523, 55, 111, 49, 202, 613, 230, 1672, 289, 168, 649, 155, 91, 329, 79, 90, 387, 438, 316, 90, 325, 134, 1501, 1061, 2270, 106, 188, 127, 827, 504053, 1014, 464]
complex_prob_sampled = [8476, 5766, 10160, 15578, 2374, 13489, 5933, 7193, 15394, 9755, 3345, 4902, 2310, 6531, 9133, 4730, 23563, 5374, 7021, 10085, 4720, 6945, 5276, 2842, 4257, 6495, 6642, 11402, 4032, 10015, 5144, 23093, 23849, 32051, 4064, 6340, 5850, 19839, 133235, 14704, 20180]
'''
# heldout
# =======

'''
dataset = 'ed' 
model = 'os-dramatic-rest-ed'

ground = [141, 61, 132, 31, 40, 41, 21, 75, 93, 57, 28, 58, 58, 84, 105, 27, 42, 138, 40, 134, 89, 102, 163, 25, 88, 34, 129, 126, 97, 107, 30, 80, 217, 329, 287, 167, 228, 99, 972, 161, 63]
meed2 = [222, 56, 137, 13, 34, 5, 2, 32, 84, 44, 0, 18, 79, 84, 76, 13, 0, 201, 70, 104, 18, 34, 87, 42, 134, 19, 103, 68, 98, 31, 12, 37, 86, 266, 251, 124, 260, 41, 1937, 65, 12]
simple_argmax = [283, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 85, 0, 0, 0, 0, 153, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 79, 0, 0, 0, 4399, 0, 0]
simple_prob_sampled = [139, 56, 114, 48, 31, 55, 30, 70, 95, 59, 22, 58, 45, 56, 89, 15, 38, 152, 33, 160, 67, 87, 180, 23, 92, 44, 109, 92, 68, 87, 23, 83, 209, 354, 290, 177, 217, 135, 1088, 140, 69]
complex_argmax = [242, 13, 45, 48, 9, 28, 11, 34, 57, 34, 10, 34, 71, 48, 43, 11, 13, 129, 28, 143, 29, 50, 152, 7, 35, 30, 100, 53, 66, 37, 18, 67, 78, 172, 177, 59, 86, 34, 2545, 30, 123]
complex_prob_sampled = [137, 60, 112, 58, 33, 59, 20, 55, 121, 66, 18, 48, 43, 62, 94, 17, 48, 134, 57, 138, 62, 115, 187, 16, 86, 35, 112, 103, 84, 98, 28, 94, 207, 396, 307, 162, 224, 136, 833, 141, 193]
'''

'''
dataset = 'osed-dramatic' 
model = 'os-dramatic-rest-osed-dramatic'

ground = [4655, 3334, 4001, 8616, 1024, 4726, 2303, 3414, 6911, 4727, 1800, 1834, 956, 3347, 3668, 2241, 15949, 3315, 3448, 5001, 1562, 2519, 1945, 1184, 1397, 4133, 2553, 5292, 1264, 4392, 2046, 13230, 1491, 1400, 355, 435, 469, 1298, 5830, 469, 1543]
meed2 = [3380, 512, 1310, 10487, 303, 997, 329, 810, 3901, 3434, 306, 413, 583, 618, 4274, 2265, 48924, 1664, 2729, 2975, 609, 1976, 1097, 409, 1013, 3627, 1357, 6732, 545, 1910, 3892, 24939, 0, 0, 0, 0, 0, 0, 1757, 0, 0]
simple_argmax = [4844, 0, 0, 8133, 0, 0, 0, 0, 6793, 4807, 0, 0, 0, 0, 4619, 2307, 47143, 3364, 0, 5889, 2531, 2363, 1769, 0, 1814, 5149, 3277, 5857, 1493, 4698, 2035, 21192, 0, 0, 0, 0, 0, 0, 0, 0, 0]
simple_prob_sampled = [5172, 4047, 3628, 9101, 1214, 4191, 2530, 3419, 7302, 3363, 2461, 1858, 1284, 2451, 4649, 3458, 15518, 2501, 3535, 4448, 1947, 2728, 1963, 1387, 1652, 4330, 3534, 5861, 1639, 3812, 3392, 12184, 1243, 1308, 316, 348, 224, 899, 3219, 425, 1536]
complex_argmax = [4363, 248, 493, 9962, 138, 540, 406, 683, 6644, 4490, 252, 222, 99, 391, 5057, 2376, 41017, 3179, 351, 6203, 2310, 1940, 1704, 107, 1567, 5967, 3827, 5614, 1227, 4042, 1959, 21508, 94, 98, 24, 25, 18, 68, 727, 29, 108]
complex_prob_sampled = [4947, 3782, 3852, 8905, 1168, 4148, 2663, 3226, 7221, 3386, 2431, 1958, 1208, 2462, 4650, 3544, 16394, 2341, 3294, 4391, 1854, 2668, 1955, 1398, 1585, 4104, 3486, 5785, 1610, 3537, 3100, 11998, 1375, 1479, 344, 455, 258, 1055, 3905, 518, 1637]
'''

# ============ LISTENER 4TURNS ==============

'''
dataset = 'os-dramatic-rest'
model = 'os-dramatic-rest'

ground = [3295, 2415, 4832, 6500, 1072, 5942, 2844, 2803, 6488, 4054, 1346, 2289, 872, 2965, 3834, 2139, 11327, 2320, 2944, 4204, 1954, 3013, 2348, 1222, 1997, 2575, 3050, 4811, 1768, 4167, 1925, 9784, 9270, 12093, 1648, 3100, 2389, 8340, 58881, 5627, 8663]
alexa = [630, 470, 695, 1199, 166, 942, 415, 590, 1165, 787, 249, 336, 202, 499, 715, 283, 1656, 472, 524, 714, 420, 523, 415, 224, 332, 536, 519, 871, 299, 904, 424, 1645, 22632, 22485, 22677, 22631, 22654, 22434, 22690, 22525, 22561]
meed2 = [549, 29, 26, 2653, 7, 541, 39, 131, 665, 1689, 6, 1, 38, 14, 111, 550, 7076, 86, 308, 225, 64, 69, 81, 19, 223, 218, 239, 1042, 90, 35, 326, 7795, 692, 3413, 0, 11, 14, 62, 188793, 3617, 1563]
simple_argmax = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 223110, 0, 0]
simple_prob_sampled = [3431, 2408, 3926, 6566, 949, 5455, 2381, 3132, 6252, 4269, 1300, 1847, 1039, 2507, 3636, 1798, 9056, 2297, 2902, 4200, 1900, 2789, 2109, 1182, 1651, 2667, 2539, 4727, 1497, 4376, 2197, 9227, 11084, 14845, 1819, 2637, 2733, 8701, 59793, 6412, 8874]
complex_argmax = [214, 125, 294, 1218, 64, 472, 172, 253, 669, 523, 55, 111, 49, 202, 613, 230, 1672, 289, 168, 649, 155, 91, 329, 79, 90, 387, 438, 316, 90, 325, 134, 1501, 1061, 2270, 106, 188, 127, 827, 205076, 1014, 464]
complex_prob_sampled = [3553, 2546, 4960, 6514, 1043, 5961, 2946, 2846, 6719, 4044, 1493, 2394, 895, 3016, 4167, 2284, 11438, 2219, 3156, 4307, 2095, 3175, 2406, 1264, 1993, 2698, 3074, 5091, 1963, 4208, 2164, 10308, 9252, 12185, 1702, 2878, 2197, 8036, 55862, 5466, 8592]
'''


dataset = 'ed' 
model = 'os-dramatic-rest-ed'

ground = [51, 14, 73, 14, 12, 22, 10, 52, 55, 37, 9, 29, 19, 33, 28, 19, 28, 79, 19, 70, 43, 45, 88, 16, 47, 19, 76, 50, 56, 35, 14, 52, 145, 187, 220, 119, 89, 76, 277, 92, 38]
alexa = [11, 5, 11, 5, 8, 3, 7, 6, 7, 2, 3, 12, 6, 9, 7, 1, 4, 12, 13, 8, 7, 9, 15, 1, 8, 2, 11, 14, 7, 7, 6, 7, 228, 242, 281, 225, 241, 250, 246, 257, 253]
meed2 = [101, 9, 83, 5, 1, 3, 0, 13, 67, 25, 0, 10, 36, 52, 14, 6, 0, 131, 44, 64, 9, 17, 56, 13, 78, 9, 49, 34, 77, 4, 8, 25, 68, 231, 248, 123, 86, 41, 567, 45, 5]
simple_argmax = [119, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 0, 0, 0, 0, 93, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 73, 0, 0, 0, 2134, 0, 0]
simple_prob_sampled = [66, 31, 57, 19, 15, 32, 12, 32, 54, 35, 9, 32, 19, 27, 39, 9, 19, 80, 20, 80, 33, 40, 107, 7, 39, 22, 48, 44, 35, 44, 10, 47, 103, 180, 146, 89, 105, 61, 513, 71, 26]
complex_argmax = [78, 13, 45, 48, 9, 28, 11, 34, 57, 34, 10, 34, 24, 48, 43, 11, 13, 69, 28, 143, 29, 50, 152, 7, 35, 30, 100, 53, 66, 37, 18, 67, 78, 172, 171, 59, 86, 34, 280, 30, 123]
complex_prob_sampled = [57, 29, 66, 25, 19, 39, 12, 26, 68, 33, 7, 23, 18, 33, 40, 9, 32, 65, 40, 82, 33, 67, 111, 6, 44, 16, 61, 59, 45, 40, 16, 59, 113, 178, 172, 99, 74, 71, 258, 60, 152]


'''
dataset = 'osed-dramatic' 
model = 'os-dramatic-rest-osed-dramatic'

ground = [1187, 941, 1411, 1902, 303, 1261, 819, 839, 1767, 1074, 541, 614, 241, 965, 1146, 795, 5462, 733, 972, 1086, 417, 746, 556, 337, 424, 988, 845, 1373, 417, 962, 576, 3505, 514, 477, 136, 185, 135, 471, 2208, 201, 545]
alexa = [141, 116, 120, 232, 37, 136, 92, 98, 210, 129, 74, 62, 34, 102, 131, 70, 516, 88, 105, 139, 47, 89, 66, 46, 42, 117, 99, 170, 37, 137, 66, 367, 4025, 3983, 3962, 3993, 3971, 4043, 4049, 4016, 4120]
meed2 = [718, 51, 495, 1589, 97, 227, 98, 76, 679, 733, 49, 105, 84, 146, 1120, 659, 19517, 334, 602, 455, 72, 335, 293, 85, 164, 742, 282, 1194, 135, 167, 708, 6335, 0, 0, 0, 0, 0, 0, 1731, 0, 0]
simple_argmax = [1309, 0, 0, 2400, 0, 0, 0, 0, 1799, 1170, 0, 0, 0, 0, 1156, 658, 16704, 754, 0, 1115, 483, 673, 629, 0, 408, 969, 754, 1349, 449, 1187, 543, 5568, 0, 0, 0, 0, 0, 0, 0, 0, 0]
simple_prob_sampled = [1438, 1225, 1052, 2585, 337, 1227, 759, 971, 2083, 931, 734, 498, 399, 707, 1301, 1021, 4566, 685, 1059, 1225, 560, 836, 541, 410, 429, 1159, 994, 1691, 489, 1064, 927, 3462, 343, 381, 100, 108, 67, 262, 915, 120, 416]
complex_argmax = [828, 248, 493, 4229, 138, 540, 406, 683, 1650, 853, 252, 222, 99, 391, 1594, 727, 10578, 569, 351, 1429, 262, 250, 564, 107, 161, 1787, 1304, 1106, 183, 531, 467, 5884, 94, 98, 24, 25, 18, 68, 727, 29, 108]
complex_prob_sampled = [1205, 1013, 1306, 2211, 369, 1151, 911, 802, 1916, 922, 673, 618, 288, 763, 1337, 1016, 5381, 582, 929, 1102, 461, 750, 553, 409, 438, 986, 1015, 1510, 461, 856, 754, 3346, 434, 506, 133, 190, 93, 377, 1518, 213, 579]
'''



ED_emotions_ordered = ['prepared', 'anticipating', 'hopeful', 'proud', 'excited', 'joyful', 'content', 'caring', 'grateful', 'trusting', 'confident', 'faithful', 'impressed', 'surprised', 'terrified', 'afraid', 'apprehensive', 'anxious', 'embarrassed', 'ashamed', 'devastated', 'sad', 'disappointed', 'lonely', 'sentimental', 'nostalgic', 'guilty', 'disgusted', 'furious', 'angry', 'annoyed', 'jealous', 'agreeing', 'acknowledging', 'encouraging', 'consoling', 'sympathizing', 'suggesting', 'questioning', 'wishing', 'neutral']

ED_emotions_ordered_cap = []
for emo in ED_emotions_ordered:
    ED_emotions_ordered_cap.append(emo.capitalize())

ED_emotions_cap = []

import plotly.graph_objects as go
#models=['Ground truth', 'Neural predictor', 'Simple DT (argmax)', 'Simple DT (prob. sampled)', 'Complex DT (argmax)', 'Complex DT (prob. sampled)', ]

'''
fig = go.Figure(data=[
    go.Bar(name='Ground truth', x=ED_emotions_ordered_cap, y=ground, marker_color='#636efa'),
    go.Bar(name='Alexa', x=ED_emotions_ordered_cap, y=alexa, marker_color='#ef553b'),
    go.Bar(name='Neural predictor', x=ED_emotions_ordered_cap, y=meed2, marker_color='#40cc96'),
    go.Bar(name='Simple DT (argmax)', x=ED_emotions_ordered_cap, y=simple_argmax, marker_color='#ab63fa'),
    go.Bar(name='Complex DT (argmax)', x=ED_emotions_ordered_cap, y=complex_argmax, marker_color='#f8a15a'),
    go.Bar(name='Simple DT (prob. sampled)', x=ED_emotions_ordered_cap, y=simple_prob_sampled, marker_color='#47d3f3'),
    go.Bar(name='Complex DT (prob. sampled)', x=ED_emotions_ordered_cap, y=complex_prob_sampled, marker_color='#f66693'),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.show()


fig = go.Figure(data=[
    go.Bar(name='Ground truth', x=ED_emotions_ordered_cap, y=ground, marker_color='#636efa'),
    go.Bar(name='Alexa', x=ED_emotions_ordered_cap, y=alexa, marker_color='#ef553b'),
    go.Bar(name='Neural predictor', x=ED_emotions_ordered_cap, y=meed2, marker_color='#40cc96'),
    go.Bar(name='Simple DT (argmax)', x=ED_emotions_ordered_cap, y=simple_argmax, marker_color='#ab63fa'),
    go.Bar(name='Complex DT (argmax)', x=ED_emotions_ordered_cap, y=complex_argmax, marker_color='#f8a15a'),
    go.Bar(name='Simple DT (prob. sampled)', x=ED_emotions_ordered_cap, y=simple_prob_sampled, marker_color='#47d3f3'),
    go.Bar(name='Complex DT (prob. sampled)', x=ED_emotions_ordered_cap, y=complex_prob_sampled, marker_color='#f66693'),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_yaxes(type="log")
fig.show()
'''
# =======================================

fig = go.Figure(data=[
    go.Bar(name='', x=ED_emotions_ordered_cap, y=ground, marker_color='#636efa'),
    #go.Bar(name='Equally sampled', x=ED_emotions_ordered_cap, y=alexa, marker_color='#ef553b'),
    go.Bar(name='', x=ED_emotions_ordered_cap, y=alexa, marker_color='#f66693'),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_yaxes(type="log")
fig.update_xaxes(tickangle = -45)
fig.update_layout(
    font=dict(
        family="Times New Roman",
        size=16,
        color="Black"
    ))
fig.show()

# =======================================

fig = go.Figure(data=[
    go.Bar(name='', x=ED_emotions_ordered_cap, y=ground, marker_color='#636efa'),
    #go.Bar(name='Neural predictor', x=ED_emotions_ordered_cap, y=meed2, marker_color='#40cc96'),
    go.Bar(name='', x=ED_emotions_ordered_cap, y=meed2, marker_color='#f66693'),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_yaxes(type="log")
fig.update_xaxes(tickangle = -45)
fig.update_layout(
    font=dict(
        family="Times New Roman",
        size=16,
        color="Black"
    ))
fig.show()

# =======================================

fig = go.Figure(data=[
    go.Bar(name='', x=ED_emotions_ordered_cap, y=ground, marker_color='#636efa'),
    #go.Bar(name='Simple DT (argmax)', x=ED_emotions_ordered_cap, y=simple_argmax, marker_color='#ab63fa'),
    go.Bar(name='', x=ED_emotions_ordered_cap, y=simple_argmax, marker_color='#f66693'),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_yaxes(type="log")
fig.update_xaxes(tickangle = -45)
fig.update_layout(
    font=dict(
        family="Times New Roman",
        size=16,
        color="Black"
    ))
fig.show()

# =======================================

fig = go.Figure(data=[
    go.Bar(name='', x=ED_emotions_ordered_cap, y=ground, marker_color='#636efa'),
    #go.Bar(name='Complex DT (argmax)', x=ED_emotions_ordered_cap, y=complex_argmax, marker_color='#f8a15a'),
    go.Bar(name='', x=ED_emotions_ordered_cap, y=complex_argmax, marker_color='#f66693'),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_yaxes(type="log")
fig.update_xaxes(tickangle = -45)
fig.update_layout(
    font=dict(
        family="Times New Roman",
        size=16,
        color="Black"
    ))
fig.show()

# =======================================  

fig = go.Figure(data=[
    go.Bar(name='', x=ED_emotions_ordered_cap, y=ground, marker_color='#636efa'),
    #go.Bar(name='Simple DT (prob. sampled)', x=ED_emotions_ordered_cap, y=simple_prob_sampled, marker_color='#47d3f3'),
    go.Bar(name='', x=ED_emotions_ordered_cap, y=simple_prob_sampled, marker_color='#f66693'),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_yaxes(type="log")
fig.update_xaxes(tickangle = -45)
fig.update_layout(
    font=dict(
        family="Times New Roman",
        size=16,
        color="Black"
    ))
fig.show()

# =======================================

fig = go.Figure(data=[
    go.Bar(name='', x=ED_emotions_ordered_cap, y=ground, marker_color='#636efa'),
    go.Bar(name='', x=ED_emotions_ordered_cap, y=complex_prob_sampled, marker_color='#f66693'),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_yaxes(type="log")
fig.update_xaxes(tickangle = -45)
fig.update_layout(
    font=dict(
        family="Times New Roman",
        size=16,
        color="Black"
    ))
fig.show()

# =======================================

print(sum(ground))
print(sum(meed2))
print(sum(simple_argmax))
print(sum(simple_prob_sampled))
print(sum(complex_argmax))
print(sum(complex_prob_sampled))