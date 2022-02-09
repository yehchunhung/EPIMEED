import csv
import pandas as pd
import numpy as np

'''
dataset = 'os-dramatic-rest'
model = 'os-dramatic-rest'

ground = [12977, 8787, 15787, 25062, 3802, 21541, 9023, 12037, 23964, 16434, 4919, 7449, 3655, 10395, 13591, 6667, 36754, 9136, 10879, 16162, 7102, 10609, 8379, 4528, 6455, 10271, 9720, 18099, 5961, 16457, 7675, 36634, 40551, 54004, 6656, 10568, 11022, 34696, 244846, 25258, 32727]
meed2 = [1917, 124, 46, 10605, 14, 1560, 95, 580, 2610, 7702, 21, 4, 156, 33, 292, 1390, 16671, 346, 840, 939, 211, 209, 236, 40, 675, 839, 643, 3010, 246, 108, 1108, 21286, 3941, 15729, 0, 31, 347, 214, 755775, 15868, 4778]
simple_argmax = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 871239, 0, 0]
simple_prob_sampled = [13918, 9304, 15172, 26066, 3804, 21600, 8918, 12556, 24680, 16297, 5308, 7398, 3950, 10135, 14418, 6703, 35292, 8976, 11108, 16433, 7585, 10804, 8168, 4562, 6348, 10726, 9853, 18553, 6043, 17061, 8629, 36702, 42528, 57557, 7175, 10167, 10724, 33859, 232028, 26065, 34066]
complex_argmax = [341, 203, 495, 1958, 113, 789, 283, 415, 1124, 865, 90, 197, 84, 335, 999, 375, 2773, 490, 282, 1062, 270, 166, 551, 109, 145, 640, 724, 498, 143, 519, 215, 2455, 1767, 3618, 175, 315, 201, 1368, 841807, 1533, 747]
complex_prob_sampled = [13781, 9351, 17575, 25302, 4043, 22056, 10156, 12091, 25429, 15875, 5517, 8200, 3749, 10902, 15216, 7694, 39278, 8897, 11419, 16409, 7706, 11758, 8964, 4816, 7098, 10398, 11111, 18807, 6850, 16869, 8590, 37105, 38749, 52344, 6518, 10336, 9306, 32487, 231082, 23526, 33879]
'''

'''
dataset = 'ed'
model = 'os-dramatic-rest-ed'

ground = [230, 110, 241, 106, 112, 119, 58, 142, 182, 115, 45, 150, 97, 191, 192, 43, 82, 261, 128, 252, 141, 207, 313, 51, 187, 74, 281, 278, 167, 200, 80, 127, 298, 414, 315, 235, 241, 123, 1020, 196, 147]
meed2 = [387, 101, 291, 91, 79, 46, 6, 60, 194, 88, 5, 87, 131, 200, 203, 29, 1, 347, 258, 258, 47, 80, 202, 82, 265, 44, 306, 342, 155, 85, 74, 76, 146, 334, 261, 133, 266, 59, 2004, 85, 43]
simple_argmax = [404, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 125, 0, 0, 0, 0, 222, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 244, 0, 0, 0, 6956, 0, 0]
simple_prob_sampled = [197, 86, 181, 74, 49, 93, 42, 87, 192, 99, 22, 107, 107, 102, 163, 24, 53, 203, 69, 239, 110, 144, 296, 40, 106, 64, 172, 171, 153, 134, 42, 124, 355, 616, 465, 272, 288, 222, 1647, 223, 118]
complex_argmax = [5, 8, 9, 27, 2, 18, 7, 9, 24, 18, 3, 3, 2, 13, 27, 5, 26, 20, 3, 36, 6, 7, 38, 5, 20, 13, 35, 13, 30, 12, 4, 57, 40, 102, 22, 11, 6, 35, 7192, 25, 13]
complex_prob_sampled = [100, 76, 163, 167, 40, 162, 72, 101, 177, 128, 57, 72, 34, 106, 148, 64, 231, 101, 86, 149, 62, 92, 126, 38, 84, 76, 112, 156, 75, 139, 74, 239, 347, 531, 100, 137, 110, 306, 2391, 249, 273]
'''

'''
dataset = 'osed'
model = 'os-dramatic-rest-osed-dramatic'

ground = [7066, 5132, 6440, 12745, 1734, 7265, 3725, 5050, 10480, 7079, 2866, 3011, 1449, 5178, 5696, 3721, 25239, 4813, 5415, 7401, 2505, 3960, 3072, 1862, 2253, 6125, 4144, 8010, 2121, 6497, 3122, 20204, 2459, 2374, 619, 753, 736, 2187, 9459, 858, 2615]
meed2 = [5040, 670, 2132, 14808, 484, 1651, 523, 1062, 5569, 5042, 439, 653, 819, 955, 6467, 3627, 81158, 2405, 4184, 4157, 782, 2868, 1663, 588, 1433, 5299, 1990, 9806, 854, 2444, 5724, 38612, 0, 4, 0, 0, 0, 0, 3528, 0, 0]
simple_argmax = [7601, 0, 0, 12864, 0, 0, 0, 0, 10362, 7049, 0, 0, 0, 0, 6633, 3362, 76987, 5018, 0, 8568, 3411, 3711, 2823, 0, 2520, 7247, 4673, 8899, 2150, 7310, 3289, 32963, 0, 0, 0, 0, 0, 0, 0, 0, 0]
simple_prob_sampled = [8122, 6312, 5838, 14295, 1788, 6619, 3980, 5320, 11220, 5173, 3685, 2934, 2129, 3709, 7105, 5277, 24300, 3825, 5530, 7000, 2878, 4209, 3213, 2066, 2588, 6751, 5296, 9222, 2457, 5821, 5314, 18887, 1897, 1897, 479, 548, 367, 1529, 4936, 604, 2320]
complex_argmax = [294, 149, 307, 1612, 79, 325, 213, 360, 746, 459, 134, 103, 53, 228, 852, 471, 8652, 369, 178, 716, 166, 141, 333, 45, 105, 775, 609, 414, 100, 279, 238, 2182, 511, 855, 53, 65, 45, 413, 193353, 214, 244]
complex_prob_sampled = [4278, 3428, 4421, 7924, 1080, 4738, 2748, 3097, 6998, 3896, 1893, 1991, 1155, 2739, 4534, 2902, 15182, 2513, 3334, 4263, 1888, 2698, 2140, 1325, 1520, 3641, 3228, 5034, 1535, 4008, 2648, 11205, 8050, 10574, 1572, 2354, 1931, 6918, 50106, 5124, 6827]
'''

# heldout
# =======

'''
dataset = 'ed' 
model = 'os-dramatic-rest-ed'

ground  =  [230, 110, 241, 106, 112, 119, 58, 142, 182, 115, 45, 150, 97, 191, 192, 43, 82, 261, 128, 252, 141, 207, 313, 51, 187, 74, 281, 278, 167, 200, 80, 127, 298, 414, 315, 235, 241, 123, 1020, 196, 147]
meed2  =  [387, 101, 291, 91, 79, 46, 6, 60, 194, 88, 5, 87, 131, 200, 203, 29, 1, 347, 258, 258, 47, 80, 202, 82, 265, 44, 306, 342, 155, 85, 74, 76, 146, 334, 261, 133, 266, 59, 2004, 85, 43]
simple_argmax  =  [404, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 125, 0, 0, 0, 0, 222, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 244, 0, 0, 0, 6956, 0, 0]
simple_prob_sampled  =  [217, 89, 178, 72, 50, 101, 48, 111, 181, 96, 30, 101, 78, 89, 135, 21, 55, 220, 59, 241, 103, 161, 307, 41, 145, 61, 147, 162, 126, 133, 37, 139, 399, 592, 477, 264, 306, 232, 1612, 211, 124]
complex_argmax  =  [396, 69, 168, 97, 85, 65, 17, 89, 159, 82, 23, 103, 115, 182, 265, 22, 34, 226, 145, 322, 59, 116, 330, 37, 132, 70, 305, 309, 150, 145, 38, 133, 120, 209, 190, 64, 88, 38, 2570, 33, 151]
complex_prob_sampled  =  [226, 120, 210, 126, 77, 156, 70, 105, 203, 123, 52, 151, 80, 151, 189, 40, 94, 221, 148, 241, 103, 192, 340, 50, 164, 82, 243, 245, 146, 204, 71, 135, 341, 521, 343, 204, 240, 169, 900, 172, 303]
'''


dataset = 'osed-dramatic' 
model = 'os-dramatic-rest-osed-dramatic'

ground  =  [7066, 5132, 6440, 12745, 1734, 7265, 3725, 5050, 10480, 7079, 2866, 3011, 1449, 5178, 5696, 3721, 25239, 4813, 5415, 7401, 2505, 3960, 3072, 1862, 2253, 6125, 4144, 8010, 2121, 6497, 3122, 20204, 2459, 2374, 619, 753, 736, 2187, 9459, 858, 2615]
meed2  =  [5040, 670, 2132, 14808, 484, 1651, 523, 1062, 5569, 5042, 439, 653, 819, 955, 6467, 3627, 81158, 2405, 4184, 4157, 782, 2868, 1663, 588, 1433, 5299, 1990, 9806, 854, 2444, 5724, 38612, 0, 4, 0, 0, 0, 0, 3528, 0, 0]
simple_argmax  =  [7601, 0, 0, 12864, 0, 0, 0, 0, 10362, 7049, 0, 0, 0, 0, 6633, 3362, 76987, 5018, 0, 8568, 3411, 3711, 2823, 0, 2520, 7247, 4673, 8899, 2150, 7310, 3289, 32963, 0, 0, 0, 0, 0, 0, 0, 0, 0]
simple_prob_sampled  =  [8099, 6271, 5734, 14163, 1829, 6610, 4020, 5271, 11300, 5230, 3834, 2951, 2009, 3749, 7120, 5310, 24276, 3865, 5496, 6857, 3009, 4252, 3089, 2099, 2568, 6516, 5472, 9165, 2567, 5952, 5285, 18777, 1917, 2001, 467, 547, 354, 1418, 4987, 667, 2337]
complex_argmax  =  [6290, 455, 1053, 15918, 258, 929, 711, 1173, 10188, 5352, 440, 330, 229, 802, 7003, 3449, 71631, 4282, 699, 7906, 2521, 2227, 2252, 297, 1866, 7999, 4821, 8181, 1405, 5027, 2802, 35436, 159, 179, 36, 37, 30, 94, 2763, 48, 162]
complex_prob_sampled  =  [7660, 5952, 6203, 13036, 1777, 6340, 4356, 4876, 10854, 5071, 3730, 3037, 1893, 4001, 7130, 5381, 26013, 3589, 5088, 6544, 2758, 3995, 3071, 2242, 2347, 6067, 5415, 8593, 2565, 5483, 4614, 17759, 2361, 2599, 612, 786, 442, 1837, 7611, 918, 2834]


# heldout - all
# =============

'''
dataset = 'os-dramatic-rest'
model = 'os-dramatic-rest'

ground  =  [12977, 8787, 15787, 25062, 3802, 21541, 9023, 12037, 23964, 16434, 4919, 7449, 3655, 10395, 13591, 6667, 36754, 9136, 10879, 16162, 7102, 10609, 8379, 4528, 6455, 10271, 9720, 18099, 5961, 16457, 7675, 36634, 40551, 54004, 6656, 10568, 11022, 34696, 244846, 25258, 32727]
meed2  =  [1917, 124, 46, 10605, 14, 1560, 95, 580, 2610, 7702, 21, 4, 156, 33, 292, 1390, 16671, 346, 840, 939, 211, 209, 236, 40, 675, 839, 643, 3010, 246, 108, 1108, 21286, 3941, 15729, 0, 31, 347, 214, 755775, 15868, 4778]
simple_argmax  =  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 871239, 0, 0]
simple_prob_sampled  =  [13950, 9259, 16092, 26245, 3856, 22108, 9435, 12301, 24916, 16349, 5320, 7831, 3826, 10335, 14507, 7265, 37535, 8930, 11306, 16193, 7626, 11273, 8495, 4672, 6680, 10752, 10593, 18701, 6432, 16675, 8431, 37663, 41267, 56396, 6911, 10439, 10167, 33610, 227429, 25151, 34317]
complex_argmax  =  [231, 120, 325, 1256, 49, 476, 187, 297, 758, 598, 74, 109, 59, 212, 685, 318, 2110, 369, 203, 693, 176, 99, 328, 99, 79, 495, 484, 320, 106, 349, 165, 1648, 1116, 2702, 95, 217, 134, 963, 850165, 1307, 1063]
complex_prob_sampled  =  [13795, 9455, 18289, 25711, 4382, 22693, 10663, 11972, 26006, 16166, 5561, 8899, 3703, 11416, 15786, 8161, 42360, 9123, 11752, 16723, 7875, 11820, 8875, 4840, 7401, 10771, 11788, 19095, 7188, 16314, 8568, 38969, 37253, 50676, 6715, 11250, 9110, 32118, 220041, 23711, 34245]
'''

'''
dataset = 'ed' 
model = 'os-dramatic-rest-ed'

ground  =  [230, 110, 241, 106, 112, 119, 58, 142, 182, 115, 45, 150, 97, 191, 192, 43, 82, 261, 128, 252, 141, 207, 313, 51, 187, 74, 281, 278, 167, 200, 80, 127, 298, 414, 315, 235, 241, 123, 1020, 196, 147]
meed2  =  [387, 101, 291, 91, 79, 46, 6, 60, 194, 88, 5, 87, 131, 200, 203, 29, 1, 347, 258, 258, 47, 80, 202, 82, 265, 44, 306, 342, 155, 85, 74, 76, 146, 334, 261, 133, 266, 59, 2004, 85, 43]
simple_argmax  =  [404, 0, 252, 0, 0, 63, 0, 0, 0, 0, 0, 0, 125, 249, 0, 0, 0, 222, 0, 372, 0, 0, 0, 0, 238, 0, 0, 1034, 0, 0, 0, 115, 0, 233, 120, 0, 0, 0, 4424, 100, 0]
simple_prob_sampled  =  [227, 130, 181, 128, 91, 129, 56, 128, 190, 135, 41, 128, 108, 162, 207, 34, 79, 238, 143, 278, 113, 165, 287, 60, 179, 76, 206, 276, 143, 199, 77, 147, 347, 495, 333, 212, 259, 153, 1083, 184, 144]
complex_argmax  =  [400, 59, 240, 55, 70, 50, 6, 84, 141, 63, 14, 77, 129, 239, 250, 14, 22, 224, 129, 403, 50, 103, 338, 37, 237, 63, 282, 314, 141, 103, 29, 185, 162, 261, 250, 92, 93, 45, 2152, 79, 266]
complex_prob_sampled  =  [195, 97, 217, 114, 99, 122, 56, 106, 209, 122, 49, 128, 87, 156, 186, 38, 81, 217, 145, 270, 107, 181, 303, 44, 163, 89, 231, 264, 161, 166, 68, 176, 385, 526, 383, 223, 206, 160, 818, 197, 406]
'''


ED_emotions_ordered = ['prepared', 'anticipating', 'hopeful', 'proud', 'excited', 'joyful', 'content', 'caring', 'grateful', 'trusting', 'confident', 'faithful', 'impressed', 'surprised', 'terrified', 'afraid', 'apprehensive', 'anxious', 'embarrassed', 'ashamed', 'devastated', 'sad', 'disappointed', 'lonely', 'sentimental', 'nostalgic', 'guilty', 'disgusted', 'furious', 'angry', 'annoyed', 'jealous', 'agreeing', 'acknowledging', 'encouraging', 'consoling', 'sympathizing', 'suggesting', 'questioning', 'wishing', 'neutral']

ED_emotions_ordered_cap = []
for emo in ED_emotions_ordered:
    ED_emotions_ordered_cap.append(emo.capitalize())

ED_emotions_cap = []

import plotly.graph_objects as go
#models=['Ground truth', 'Neural predictor', 'Simple DT (argmax)', 'Simple DT (prob. sampled)', 'Complex DT (argmax)', 'Complex DT (prob. sampled)', ]

fig = go.Figure(data=[
    go.Bar(name='Ground truth', x=ED_emotions_ordered_cap, y=ground, marker_color='#636efa'),
    go.Bar(name='Neural predictor', x=ED_emotions_ordered_cap, y=meed2, marker_color='#ef553b'),
    go.Bar(name='Simple DT (argmax)', x=ED_emotions_ordered_cap, y=simple_argmax, marker_color='#40cc96'),
    go.Bar(name='Complex DT (argmax)', x=ED_emotions_ordered_cap, y=complex_argmax, marker_color='#ab63fa'),
    go.Bar(name='Simple DT (prob. sampled)', x=ED_emotions_ordered_cap, y=simple_prob_sampled, marker_color='#f8a15a'),
    go.Bar(name='Complex DT (prob. sampled)', x=ED_emotions_ordered_cap, y=complex_prob_sampled, marker_color='#47d3f3'),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.show()


fig = go.Figure(data=[
    go.Bar(name='Ground truth', x=ED_emotions_ordered_cap, y=ground, marker_color='#636efa'),
    go.Bar(name='Neural predictor', x=ED_emotions_ordered_cap, y=meed2, marker_color='#ef553b'),
    go.Bar(name='Simple DT (argmax)', x=ED_emotions_ordered_cap, y=simple_argmax, marker_color='#40cc96'),
    go.Bar(name='Complex DT (argmax)', x=ED_emotions_ordered_cap, y=complex_argmax, marker_color='#ab63fa'),
    go.Bar(name='Simple DT (prob. sampled)', x=ED_emotions_ordered_cap, y=simple_prob_sampled, marker_color='#f8a15a'),
    go.Bar(name='Complex DT (prob. sampled)', x=ED_emotions_ordered_cap, y=complex_prob_sampled, marker_color='#47d3f3'),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_yaxes(type="log")
fig.show()

# =======================================

fig = go.Figure(data=[
    go.Bar(name='Ground truth', x=ED_emotions_ordered_cap, y=ground, marker_color='#636efa'),
    go.Bar(name='Neural predictor', x=ED_emotions_ordered_cap, y=meed2, marker_color='#ef553b'),
    #go.Bar(name='Simple DT (argmax)', x=ED_emotions_ordered_cap, y=simple_argmax, marker_color='#636efa'),
    #go.Bar(name='Complex DT (argmax)', x=ED_emotions_ordered_cap, y=complex_argmax, marker_color='#ab63fa'),
    #go.Bar(name='Simple DT (prob. sampled)', x=ED_emotions_ordered_cap, y=simple_prob_sampled, marker_color='#f8a15a'),
    #go.Bar(name='Complex DT (prob. sampled)', x=ED_emotions_ordered_cap, y=complex_prob_sampled, marker_color='#47d3f3'),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_yaxes(type="log")
fig.show()

# =======================================

fig = go.Figure(data=[
    go.Bar(name='Ground truth', x=ED_emotions_ordered_cap, y=ground, marker_color='#636efa'),
    #go.Bar(name='Neural predictor', x=ED_emotions_ordered_cap, y=meed2, marker_color='#ef553b'),
    go.Bar(name='Simple DT (argmax)', x=ED_emotions_ordered_cap, y=simple_argmax, marker_color='#40cc96'),
    #go.Bar(name='Complex DT (argmax)', x=ED_emotions_ordered_cap, y=complex_argmax, marker_color='#ab63fa'),
    #go.Bar(name='Simple DT (prob. sampled)', x=ED_emotions_ordered_cap, y=simple_prob_sampled, marker_color='#f8a15a'),
    #go.Bar(name='Complex DT (prob. sampled)', x=ED_emotions_ordered_cap, y=complex_prob_sampled, marker_color='#47d3f3'),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_yaxes(type="log")
fig.show()

# =======================================

fig = go.Figure(data=[
    go.Bar(name='Ground truth', x=ED_emotions_ordered_cap, y=ground, marker_color='#636efa'),
    #go.Bar(name='Neural predictor', x=ED_emotions_ordered_cap, y=meed2, marker_color='#ef553b'),
    #go.Bar(name='Simple DT (argmax)', x=ED_emotions_ordered_cap, y=simple_argmax, marker_color='#636efa'),
    go.Bar(name='Complex DT (argmax)', x=ED_emotions_ordered_cap, y=complex_argmax, marker_color='#ab63fa'),
    #go.Bar(name='Simple DT (prob. sampled)', x=ED_emotions_ordered_cap, y=simple_prob_sampled, marker_color='#f8a15a'),
    #go.Bar(name='Complex DT (prob. sampled)', x=ED_emotions_ordered_cap, y=complex_prob_sampled, marker_color='#47d3f3'),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_yaxes(type="log")
fig.show()

# =======================================  

fig = go.Figure(data=[
    go.Bar(name='Ground truth', x=ED_emotions_ordered_cap, y=ground, marker_color='#636efa'),
    #go.Bar(name='Neural predictor', x=ED_emotions_ordered_cap, y=meed2, marker_color='#ef553b'),
    #go.Bar(name='Simple DT (argmax)', x=ED_emotions_ordered_cap, y=simple_argmax, marker_color='#636efa'),
    #go.Bar(name='Complex DT (argmax)', x=ED_emotions_ordered_cap, y=complex_argmax, marker_color='#ab63fa'),
    go.Bar(name='Simple DT (prob. sampled)', x=ED_emotions_ordered_cap, y=simple_prob_sampled, marker_color='#f8a15a'),
    #go.Bar(name='Complex DT (prob. sampled)', x=ED_emotions_ordered_cap, y=complex_prob_sampled, marker_color='#47d3f3'),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_yaxes(type="log")
fig.show()

# =======================================

fig = go.Figure(data=[
    go.Bar(name='Ground truth', x=ED_emotions_ordered_cap, y=ground, marker_color='#636efa'),
    #go.Bar(name='Neural predictor', x=ED_emotions_ordered_cap, y=meed2, marker_color='#ef553b'),
    #go.Bar(name='Simple DT (argmax)', x=ED_emotions_ordered_cap, y=simple_argmax, marker_color='#636efa'),
    #go.Bar(name='Complex DT (argmax)', x=ED_emotions_ordered_cap, y=complex_argmax, marker_color='#ab63fa'),
    #go.Bar(name='Simple DT (prob. sampled)', x=ED_emotions_ordered_cap, y=simple_prob_sampled, marker_color='#f8a15a'),
    go.Bar(name='Complex DT (prob. sampled)', x=ED_emotions_ordered_cap, y=complex_prob_sampled, marker_color='#47d3f3')
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_yaxes(type="log")
fig.show()

# =======================================

print(sum(ground))
print(sum(meed2))
print(sum(simple_argmax))
print(sum(simple_prob_sampled))
print(sum(complex_argmax))
print(sum(complex_prob_sampled))