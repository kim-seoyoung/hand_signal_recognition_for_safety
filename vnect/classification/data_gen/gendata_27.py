import argparse
import pickle
from tqdm import tqdm
import sys

sys.path.extend(['../'])
from data_gen.preprocess import pre_normalization

training_subjects = [1,2,3,4,5,6]
training_cameras = [1,2]
max_body_true = 2
max_body_kinect = 4
num_joint = 21
max_frame = 100

import numpy as np
import os


def read_skeleton_filter(file):
    np_data = np.load(file)
    skeleton_sequence = {}
    skeleton_sequence['numFrame'] = np_data.shape[0]
    skeleton_sequence['frameInfo'] = []
    # num_body = 0
    for t in range(skeleton_sequence['numFrame']):
        frame_info = {}
        frame_info['numBody'] = 1
        frame_info['bodyInfo'] = []

        for m in range(frame_info['numBody']):
            body_info = {}
            body_info['numJoint'] = np_data.shape[1]
            body_info['jointInfo'] = []
            for v in range(body_info['numJoint']):
                body_info['jointInfo'].append(np_data[t][v])
            frame_info['bodyInfo'].append(body_info)
        skeleton_sequence['frameInfo'].append(frame_info)

    return skeleton_sequence


def get_nonzero_std(s):  # tvc
    index = s.sum(-1).sum(-1) != 0  # select valid frames
    s = s[index]
    if len(s) != 0:
        s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()  # three channels
    else:
        s = 0
    return s


def read_xyz(file, max_body=4, num_joint=25):  # 取了前两个body
    seq_info = read_skeleton_filter(file)
    data = np.zeros((max_body, seq_info['numFrame'], num_joint, 3))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[m, n, j, :] = v[:]
                else:
                    pass

    # select two max energy body
    energy = np.array([get_nonzero_std(x) for x in data])
    index = energy.argsort()[::-1][0:max_body_true]
    data = data[index]

    data = data.transpose(3, 1, 2, 0) # 0: xyz 1: frame 2: joint 3: body
    return data


def gendata(data_path, out_path, ignored_sample_path=None, benchmark='xview', part='eval'):
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []
    sample_name = []
    sample_label = []
    for filename in os.listdir(data_path):
        if filename in ignored_samples:
            continue
        action_class = int(
            filename[filename.find('a') + 1:filename.find('s') - 1])
        subject_id = int(
            filename[filename.find('s') + 1:filename.find('t') - 1])
        camera_id = int(
            filename[filename.find('t') + 1:filename.find('color') - 1])

        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
            # issample = istraining ###########################
        else:
            raise ValueError()

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    fp = np.zeros((len(sample_label), 3, max_frame, num_joint, max_body_true), dtype=np.float32)

    for i, s in enumerate(tqdm(sample_name)):
        data = read_xyz(os.path.join(data_path, s), max_body=max_body_kinect, num_joint=num_joint)
        fp[i, :, 0:data.shape[1], :, :] = data

    fp = pre_normalization(fp)
    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mydata Data Converter.')
    parser.add_argument('--data_path', default='/dataset/RGB/numpy/')
    parser.add_argument('--ignored_sample_path',
                        default='../data/nturgbd_raw/samples_with_missing_skeletons.txt')
    parser.add_argument('--out_folder', default='../data/mydata/gen/')

    benchmark = ['xsub', 'xview']
    part = ['train', 'val']
    arg = parser.parse_args()

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            print(b, p)
            gendata(
                arg.data_path,
                out_path,
                # arg.ignored_sample_path,
                benchmark=b,
                part=p)
