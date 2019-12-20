import argparse
import pysparkling
import scipy.io
import json
import numpy as np 
from operator import itemgetter
import pickle 

from trajnettools import TrackRow
## read file ke trackrows 
def trackrows(line):
    line = json.loads(line)
    track = line.get('track')
    if track is not None:
        return [track['f'], track['p'], track['x'], track['y']]
    return None

def get_trackrows(sc, input_file):
    print('processing ' + input_file)
    return (sc
            .textFile(input_file)
            .map(trackrows)
            .filter(lambda r: r is not None)
            .cache())

def get_dest(rows):
	L = rows.collect()  
	dict_frames = {}
	for ind in range(len(L)):
		if L[ind][1] in dict_frames:
			dict_frames[L[ind][1]].append([L[ind][0], L[ind][2], L[ind][3]])
		else:
			dict_frames[L[ind][1]] = [[L[ind][0], L[ind][2], L[ind][3]]] 

	dict_dest = {}
	for entry in dict_frames:
		dict_dest[entry] = (dict_frames[entry][-1][-2:])

	return dict_dest

def generate_dest(sc, input_file): 
    rows = get_trackrows(sc, input_file)
    dict_dest = get_dest(rows)
    dataset = input_file.replace('./DATA_BLOCK/data/train/real_data/', '').replace('.ndjson', '')
    # dataset = input_file.replace('./DATA_BLOCK/data/groundtruth/real_data/', '').replace('.ndjson', '')
    print(dataset)
    print(dict_dest)
    with open('dest_new/' + dataset + '.pkl', 'wb') as f:
        pickle.dump(dict_dest, f)



sc = pysparkling.Context()
input_file = './DATA_BLOCK/data/train/real_data/biwi_hotel.ndjson'
generate_dest(sc, input_file)
input_file = './DATA_BLOCK/data/train/real_data/crowds_zara01.ndjson'
generate_dest(sc, input_file)
input_file = './DATA_BLOCK/data/train/real_data/crowds_zara03.ndjson'
generate_dest(sc, input_file)
input_file = './DATA_BLOCK/data/train/real_data/crowds_students001.ndjson'
generate_dest(sc, input_file)
input_file = './DATA_BLOCK/data/train/real_data/crowds_students003.ndjson'
generate_dest(sc, input_file)
input_file = './DATA_BLOCK/data/train/real_data/lcas.ndjson'
generate_dest(sc, input_file)
input_file = './DATA_BLOCK/data/train/real_data/wildtrack.ndjson'
generate_dest(sc, input_file)

# input_file = './DATA_BLOCK/data/groundtruth/real_data/biwi_eth.ndjson'
# generate_dest(sc, input_file)
# input_file = './DATA_BLOCK/data/groundtruth/real_data/crowds_uni_examples.ndjson'
# generate_dest(sc, input_file)
# input_file = './DATA_BLOCK/data/groundtruth/real_data/crowds_zara02.ndjson'
# generate_dest(sc, input_file)