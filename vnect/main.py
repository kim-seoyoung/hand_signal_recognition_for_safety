import run_estimator_ps
import classification.predict as predict
from src import utils
import multiprocessing as mp
import os
import yaml
import classification.data_gen.gendata_4_predict as gen_data


parser = predict.get_parser()
args = parser.parse_args()

video = args.input
filename, _ = os.path.splitext(os.path.basename(video))
savepath = os.path.join(args.output_dir, filename)

T = False
box_size = 368

pose = run_estimator_ps.pose_estimation
joint_parents = [16, 15, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, 14, 14, 1, 4, 7, 10, 13]

if __name__ == '__main__':

    ################## get config #####################################
    parser = predict.parser.parse_args()

    if parser.config is not None:
        with open(parser.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(parser).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    #
    # video = arg.input
    # filename, _ = os.path.splitext(os.path.basename(video))
    # savepath = os.path.join(arg.output_dir, filename)

    ############## pose estimation ####################
    q_start3d = mp.Queue()
    q_joints = mp.Queue()
    ps_main = mp.Process(target=pose, args=(q_start3d, q_joints))
    ps_plot3d = mp.Process(target=utils.plot_3d, args=(q_start3d, q_joints, joint_parents, savepath, arg.savegif), daemon=True)

    ps_main.start()
    ps_plot3d.start()
    ps_main.join()

    #################### data generation for AGCN ###################
    out_path = os.path.join(arg.out_folder, 'predict')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    gen_data.gendata(
        arg.data_path,
        out_path,
        # arg.ignored_sample_path,
    )


    ######################## prediction ####################3
    # load arg form config file

    predict.init_seed(0)
    processor = predict.Processor(arg)
    processor.start()



