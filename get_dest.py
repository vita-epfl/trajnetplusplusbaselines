import argparse
import pysparkling
import scipy.io
import json
import numpy as np 
from operator import itemgetter
import pickle 

from trajnetplusplustools import TrackRow

## read file trackrows 
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
    """ Ensure that the last frame is the GOAL """
    L = rows.collect()  
    dict_frames = {}
    for ind in range(len(L)):
        f, ped_id, x, y = L[ind]
        if ped_id in dict_frames:
            dict_frames[ped_id].append([f, x, y])
        else:
            dict_frames[ped_id] = [[f, x, y]] 

    dict_dest = {}
    for ped_id in dict_frames:
        ped_presence = dict_frames[ped_id]
        ped_presence_sorted = sorted(ped_presence, key=lambda x: x[0])
        if ped_presence[-1][-2:] != ped_presence_sorted[-1][-2:]:
            import pdb
            pdb.set_trace()
        dict_dest[ped_id] = (ped_presence_sorted[-1][-2:])

    return dict_dest

def generate_dest(sc, input_file): 
    rows = get_trackrows(sc, input_file)
    dict_dest = get_dest(rows)
    # dataset = input_file.replace('./DATA_BLOCK/data/train/real_data/', '').replace('.ndjson', '')
    dataset_type = input_file.split('/')[-2] 
    dataset = input_file.split('/')[-1].replace('.ndjson', '')
    # dataset = input_file.replace('./DATA_BLOCK/data/groundtruth/real_data/', '').replace('.ndjson', '')
    print(dataset)
    print(dict_dest)
    with open('goal_files/' + dataset_type + '/' + dataset + '.pkl', 'wb') as f:
        pickle.dump(dict_dest, f)


sc = pysparkling.Context()
input_file = './DATA_BLOCK/trajdata/train/biwi_hotel.ndjson'
generate_dest(sc, input_file)
input_file = './DATA_BLOCK/trajdata/train/crowds_zara01.ndjson'
generate_dest(sc, input_file)
input_file = './DATA_BLOCK/trajdata/train/crowds_zara03.ndjson'
generate_dest(sc, input_file)
input_file = './DATA_BLOCK/trajdata/train/crowds_students001.ndjson'
generate_dest(sc, input_file)
input_file = './DATA_BLOCK/trajdata/train/crowds_students003.ndjson'
generate_dest(sc, input_file)
input_file = './DATA_BLOCK/trajdata/train/lcas.ndjson'
generate_dest(sc, input_file)
input_file = './DATA_BLOCK/trajdata/train/wildtrack.ndjson'
generate_dest(sc, input_file)
input_file = './DATA_BLOCK/trajdata/val/biwi_eth.ndjson'
generate_dest(sc, input_file)
input_file = './DATA_BLOCK/trajdata/val/crowds_uni_examples.ndjson'
generate_dest(sc, input_file)
input_file = './DATA_BLOCK/trajdata/val/crowds_zara02.ndjson'
generate_dest(sc, input_file)
# input_file = './DATA_BLOCK/data/train/real_data/biwi_hotel.ndjson'
# generate_dest(sc, input_file)
# input_file = './DATA_BLOCK/data/train/real_data/crowds_zara01.ndjson'
# generate_dest(sc, input_file)
# input_file = './DATA_BLOCK/data/train/real_data/crowds_zara03.ndjson'
# generate_dest(sc, input_file)
# input_file = './DATA_BLOCK/data/train/real_data/crowds_students001.ndjson'
# generate_dest(sc, input_file)
# input_file = './DATA_BLOCK/data/train/real_data/crowds_students003.ndjson'
# generate_dest(sc, input_file)
# input_file = './DATA_BLOCK/data/train/real_data/lcas.ndjson'
# generate_dest(sc, input_file)
# input_file = './DATA_BLOCK/data/train/real_data/wildtrack.ndjson'
# generate_dest(sc, input_file)
# input_file = './DATA_BLOCK/synth_single/train/orca_circle_crossing_6ped_single.ndjson'
# generate_dest(sc, input_file)
# input_file = './DATA_BLOCK/synth_single/val/orca_circle_crossing_6ped_single.ndjson'
# generate_dest(sc, input_file)
# input_file = './DATA_BLOCK/synth_small/train/orca_circle_crossing_6ped_small.ndjson'
# generate_dest(sc, input_file)
# input_file = './DATA_BLOCK/synth_small/val/orca_circle_crossing_6ped_small.ndjson'
# generate_dest(sc, input_file)
# input_file = './DATA_BLOCK/synth_clean/synth_medium/train/orca_circle_crossing_6ped_medium.ndjson'
# generate_dest(sc, input_file)
# input_file = './DATA_BLOCK/synth_clean/synth_medium/val/orca_circle_crossing_6ped_medium.ndjson'
# generate_dest(sc, input_file)
# input_file = './DATA_BLOCK/synth_circle_data/train/orca_circle_crossing_5ped.ndjson'
# generate_dest(sc, input_file)
# input_file = './DATA_BLOCK/synth_circle_data/val/orca_circle_crossing_5ped.ndjson'
# generate_dest(sc, input_file)
# input_file = './DATA_BLOCK/synth_circle_data/val_original/orca_circle_crossing_5ped.ndjson'
# generate_dest(sc, input_file)
# input_file = './DATA_BLOCK/synth_circle_all/val/orca_circle_crossing_synth_again.ndjson'
# generate_dest(sc, input_file)
# input_file = './DATA_BLOCK/test_synth_circle/test_private/orca_circle_crossing_5ped.ndjson'
# generate_dest(sc, input_file)
# input_file = './DATA_BLOCK/synth_huge/train/orca_circle_crossing_synth_big.ndjson'
# generate_dest(sc, input_file)
# input_file = './DATA_BLOCK/synth_huge/val/orca_circle_crossing_synth_big.ndjson'
# generate_dest(sc, input_file)
# input_file = './DATA_BLOCK/synth_traj/synth_traj_medium/train/orca_circle_crossing_11ped_small.ndjson'
# generate_dest(sc, input_file)
# input_file = './DATA_BLOCK/synth_traj/synth_traj_medium/val/orca_circle_crossing_11ped_small.ndjson'
# generate_dest(sc, input_file)
# input_file = './DATA_BLOCK/data/groundtruth/real_data/biwi_eth.ndjson'
# generate_dest(sc, input_file)
# input_file = './DATA_BLOCK/data/groundtruth/real_data/crowds_uni_examples.ndjson'
# generate_dest(sc, input_file)
# input_file = './DATA_BLOCK/data/groundtruth/real_data/crowds_zara02.ndjson'
# generate_dest(sc, input_file)
# input_file = './DATA_BLOCK/test_synth_circle/test_private/orca_circle_crossing_5ped.ndjson'
# generate_dest(sc, input_file)