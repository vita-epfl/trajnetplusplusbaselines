import argparse
from trajnettools.reader import Reader
from trajnettools import show
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os.path

def convert_to_pixel(path, rect=False):
    
    # ## Highway: Old
    # h_gtv = np.array([[ 1.35580796e+00, -9.84949024e-01,  4.67320007e+02],
    #                  [ 9.95875424e-02,  4.93216277e-01,  2.09817495e+02],
    #                  [-6.29994003e-05, -3.71870402e-04,  1.00000000e+00]], np.float32)

    # if rect:
    #     g = 73*np.array(path[np.newaxis, :, :], np.float32)
    #     print(g.shape)
    # else:
    #     primary = np.array([[r.x, r.y] for r in path])
    #     g = 73*np.array(primary[np.newaxis, :, :], np.float32)
    #     print(g.shape)
    # v = cv2.perspectiveTransform(g, h_gtv)    
    # x_list = [x for x in v[0, :, 0]]
    # y_list = [y for y in v[0, :, 1]]    

    # ## Highway: New
    x_list = [t.x*73 for t in path]
    y_list = [t.y*73 for t in path] 

    # ## Round
    # x_list = [t.x/0.0044 for t in path]
    # y_list = [t.y/0.0044 for t in path] 

    if rect:
        return np.array([y_list, x_list]).transpose()

    return x_list, y_list

def plot_scene(frame, input_paths, i, pred_paths={}):
    # On-screen, things will be displayed at 80dpi regardless of what we set here
    # This is effectively the dpi for the saved figure. We need to specify it,
    # otherwise `savefig` will pick a default dpi based on your local configuration
    dpi = 80
    stride = 12
    ##### We Will Load Figure according to frame here #####
    # # highway
    img_file = '/data/parth-data/convertPics/out{}.jpg'.format(int(frame/3)) 
    # roundabout
    # img_file = '/data/parth-data/RoundAbout/UC_davic_DJI_001/out{}.jpg'.format(int(frame)) 
    print("Img_File: ", img_file)
    if not os.path.isfile(img_file):
        return
    im = plt.imread(img_file) 
    #######################################################
    # implot = plt.imshow(im)
    height, width, nbands = im.shape
    print(im.shape)

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    ax.imshow(im, interpolation='nearest')

    ##### Primary GT #####
    primary = input_paths[0]
    x_list, y_list = convert_to_pixel(primary)
    ax.plot(x_list[:9], y_list[:9], linewidth=2, c='b')   
    ax.plot(x_list[9:9+stride], y_list[9:9+stride], linewidth=2, c='b', label='GT')   
    ax.scatter(x_list[9:9+stride], y_list[9:9+stride], linewidth=2, c='b')   
    # ax.scatter(x=[1462], y=[821], c='r', s=40)

    # Roundabout
    # 12*0.5/0.0044
    # Create a Rectangle patch
    # rect = patches.Rectangle((y_list[9] - 684, x_list[9] - 684), 1363, 1363, linewidth=1, edgecolor='r', facecolor='none')

    # Highway 
    # 12*1.0*73
    # t = 6
    rect = patches.Rectangle((x_list[9] - 438, y_list[9] - 438), 876, 876, linewidth=1, edgecolor='r', facecolor='none')
    # rect_edges = np.array([[primary[9].x - t, primary[9].y - t], 
    #               [primary[9].x - t, primary[9].y + t], 
    #               [primary[9].x + t, primary[9].y + t], 
    #               [primary[9].x + t, primary[9].y - t]])  
    # point_list = convert_to_pixel(rect_edges, rect=True)
    # # Create a Polygon patches
    # rect = patches.Polygon(np.array(point_list), linewidth=1, edgecolor='r', facecolor='none')


    # Add the patch to the Axes
    ax.add_patch(rect)

    ##### Neighbours GT #####
    # for path in input_paths[1:]:
    #     x_list, y_list = convert_to_pixel(path)
    #     ax.scatter(y_list[-1], x_list[-1], c='r', s=80)

    ##### Predictions ####
    for name, primary in pred_paths.items():
        if 'vanilla' in name:
            color = 'r'
        if 'occupancy' in name:
            color = 'g'
        if 'directional' in name:
            color = 'g'
        x_list, y_list = convert_to_pixel(primary)
        ax.scatter(x_list[:stride], y_list[:stride], s=50, label=name, c=color)
        ax.plot(x_list[:stride], y_list[:stride], linewidth=2, c=color)   

    # put a red dot, size 40, at 2 locations:
    # ax.scatter(x=[30, 40], y=[50, 60], c='r', s=40)
    ##### We Will Do Plotting Here #####
    
    ax.legend(prop={'size': 40})
    fig.savefig('fig_hw/test_{}.jpg'.format(i), dpi=dpi, transparent=True)
    plt.show()

    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_files', nargs='+',
                        help='Trajnet dataset file(s).')
    parser.add_argument('--n', type=int, default=5,
                        help='sample n trajectories')
    parser.add_argument('--id', type=int, nargs='*',
                        help='plot a particular scene')
    parser.add_argument('-o', '--output', default=None,
                        help='specify output prefix')
    parser.add_argument('--random', default=True, action='store_true',
                        help='randomize scenes')
    args = parser.parse_args()

    ## TODO Configure Writing images
    # if args.output is None:
    #     args.output = args.dataset_file

    ## Read GT Scenes
    reader = Reader(args.dataset_files[0], scene_type='paths')
    interaction_type = 5  
    if interaction_type in [1,2,3,4]:
        type_ids = [scene_id for scene_id in reader.scenes_by_id.keys() if interaction_type in reader.scenes_by_id[scene_id].tag[1]]
    else:
        type_ids = [scene_id for scene_id in reader.scenes_by_id.keys() if 4 in reader.scenes_by_id[scene_id].tag]        

    # type_ids = type_ids[:args.n]
    scenes = [[s_id, s] for s_id, s in reader.scenes() if s_id in type_ids]

# for type_id in type_ids:
        # scene = scenes[type_id]    

    # if args.id:
    #     scenes = reader.scenes(ids=args.id, randomize=args.random)
    # elif args.n:
    #     scenes = reader.scenes(limit=args.n, randomize=args.random)
    # else:
    #     scenes = reader.scenes(randomize=args.random)

    ## Reader Predictions 
    reader_list = {}
    for dataset_file in args.dataset_files[1:]:
        name = dataset_file.split('/')[-2]
        reader_list[name] = Reader(dataset_file, scene_type='paths')

    ## Visualize
    pred_paths = {}
    for scene_id, paths in scenes[1:]:
        frame = paths[0][9].frame
        for dataset_file in args.dataset_files[1:]:
            name = dataset_file.split('/')[-2]
            scenes_pred = reader_list[name].scenes(ids=[scene_id])
            for scene_id, preds in scenes_pred:
                primary = preds[0]
                primary_path = [t for t in primary if t.scene_id == scene_id]
            pred_paths[name] = primary_path
        ##TODO
        # with show.predicted_paths(paths, pred_paths):
            # pass
        print(scene_id)
        print(frame)
        plot_scene(frame, paths, scene_id, pred_paths)

if __name__ == '__main__':
    main()
