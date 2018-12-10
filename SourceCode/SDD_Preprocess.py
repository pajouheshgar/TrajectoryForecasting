import numpy as np
from collections import defaultdict
import pandas as pnd


def entity_type_to_int(x):
    if x == "Pedestrian":
        return 0
    if x == "Biker":
        return 1
    if x == "Skater":
        return 2
    if x == "Car":
        return 3
    if x == "Bus":
        return 4
    if x == "Cart":
        return 5

def split_object_trajectory_to_intervals(object_trajectory, fps=30, interval_length=10, min_length=6):
    interval_frames = fps * interval_length
    splitted_trajectories_list = []

    starting_frame = int(object_trajectory[0][4])
    interval_end = interval_frames - (starting_frame % interval_frames) + starting_frame
    trajectory = []
    for i in range(len(object_trajectory)):
        frame = int(object_trajectory[i][4])
        if frame < interval_end:
            trajectory.append(object_trajectory[i])
        else:
            starting_frame, interval_end = interval_end, interval_end + interval_frames
            trajectory = select_good_part(trajectory, fps, min_length)
            if len(trajectory) > 0:
                splitted_trajectories_list.append(trajectory)
            trajectory = []

    return splitted_trajectories_list

def select_good_part(trajectory, fps=30, min_length=6):
    min_frames = fps * min_length
    # selects a part of sequence which is not lost and not occluded
    good_part = []
    for snapshot in trajectory:
        if len(good_part) == min_frames:
            return good_part
        if not snapshot[5] and not snapshot[6]:
            good_part.append(snapshot)
        else:
            good_part = []

    if len(good_part) == min_frames:
        return good_part

    if len(good_part) < min_frames:
        return []

def extract_objects(input_data):
    object_trajectory_dict = defaultdict(lambda: [[], [], {'Occluded': False, 'Lost': False}])
    for row in input_data:
        row[9] = entity_type_to_int(row[9])
        index = row[0]
        object_trajectory_dict[index][0].append(row[1:].astype(np.int32))
        if row[6] == 1:
            object_trajectory_dict[index][2]["Lost"] = True
        if row[7] == 1:
            object_trajectory_dict[index][2]["Occluded"] = True

    for object in object_trajectory_dict.keys():
        splitted_trajectory = split_object_trajectory_to_intervals(object_trajectory_dict[object][0])
        object_trajectory_dict[object][1] = splitted_trajectory
    return dict(object_trajectory_dict)

def extract_frames(input_data):
    frame_dict = defaultdict(lambda: [])
    for row in input_data:
        frame = int(row[5])
        frame_dict[frame].append(row)
    return dict(frame_dict)


if __name__ == "__main__":
    unstable_videos = [['nexus', 3], ['nexus', 4], ['nexus', 5], ['bookstore', 1], ['deathCircle', 3], ['hyang', 3]]

    split = dict(np.load("Dataset/SDD/sdd_data_split.npy").item())

    videos = split['videos']
    splits = split['splits']

    videos += unstable_videos
    converter = {9: entity_type_to_int}
    annotations_dir = "Dataset/SDD/annotations/"
    video_names = set([s[0] for s in videos])

    total_objects = 0
    total_unlost_objects = 0
    total_unoccluded_objects = 0
    total_examples = 0
    for v in videos:
        # data = np.loadtxt(annotations_dir + v[0] + "/video{}/annotations.txt".format(v[1]), converters=converter)
        data = pnd.read_csv(annotations_dir + v[0] + "/video{}/annotations.txt".format(v[1]), sep=" ").as_matrix()

        object_dict = extract_objects(data)
        frame_dict = extract_frames(data)
        np.save(annotations_dir + v[0] + "/video{}/object_dict.npy".format(v[1]), object_dict)
        np.save(annotations_dir + v[0] + "/video{}/frame_dict.npy".format(v[1]), frame_dict)
        n_object = len(object_dict.keys())
        n_examples = sum([len(s[1]) for s in object_dict.values()])
        n_unlost_object = len([s for s in object_dict.values() if s[2]["Lost"] == False])
        n_unoccluded_object = len([s for s in object_dict.values() if s[2]["Occluded"] == False])

        average_length = np.mean([len(s[0]) for s in object_dict.values()])
        print(v[0] + "/video{} : ".format(v[1]))
        print("\t Number of objects = {}".format(n_object))
        print("\t Number of trajectories = {}".format(n_examples))
        print("\t Number of unlost objects = {}".format(n_unlost_object))
        print("\t Number of unoccluded objects = {}".format(n_unoccluded_object))
        print("\t Average length oh sequence = {}".format(average_length))
        total_objects += n_object
        total_examples += n_examples
        total_unlost_objects += n_unlost_object
        total_unoccluded_objects += n_unoccluded_object
    print("Total objects found = {}".format(total_objects))
    print("Total trajectories found = {}".format(total_examples))
    print("Total unlost objects found = {}".format(total_unlost_objects))
    print("Total unoccluded objects found = {}".format(total_unoccluded_objects))

# nexus : 3 4 5

# bookstore : 1

# deathCircle : 3

# hyang : 3
