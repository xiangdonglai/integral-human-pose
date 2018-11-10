import os
import pprint
import copy
import time
import logging

# define project dependency
import _init_paths

# pytorch
import torch
from torch.utils.data import DataLoader

# import from common_pytorch
# from common_pytorch.dataset.__init__ import *
from common_pytorch.dataset.all_dataset import *

from common_pytorch.config_pytorch import update_config_from_file, update_config_from_args, s_args, s_config, \
    s_config_file
from common_pytorch.common_loss.balanced_parallel import DataParallelModel, DataParallelCriterion

from common_pytorch.net_modules import trainNet, validNet, evalNetChallenge

# import dynamic config
exec('from common_pytorch.blocks.' + s_config.pytorch.block + \
     ' import get_default_network_config, get_pose_net, init_pose_net')
exec('from common_pytorch.loss.' + s_config.pytorch.loss + \
     ' import get_default_loss_config, get_loss_func, get_label_func, get_result_func, get_merge_func')
from core.loader import hm36_eccv_challenge_Dataset, hm36_Dataset, inthewild_Dataset


def main():
    # parsing specific config
    config = copy.deepcopy(s_config)
    config.network = get_default_network_config()  # defined in blocks
    config.loss = get_default_loss_config()

    config = update_config_from_file(config, s_config_file, check_necessity=True)
    config = update_config_from_args(config, s_args)  # config in argument is superior to config in file

    # create log and path
    final_log_path = os.path.dirname(s_args.model)
    log_name = os.path.basename(s_args.model)
    logging.basicConfig(filename=os.path.join(final_log_path, '{}_test.log'.format(log_name)),
                        format='%(asctime)-15s %(message)s', level=logging.INFO)
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    logger.info('training config:{}\n'.format(pprint.pformat(config)))

    # define devices create multi-GPU context
    os.environ["CUDA_VISIBLE_DEVICES"] = config.pytorch.gpus  # a safer method
    devices = [int(i) for i in config.pytorch.gpus.split(',')]
    logger.info("Using Devices: {}".format(str(devices)))

    # lable, loss, metric, result and flip function
    logger.info("Defining lable, loss, metric, result and flip function")
    label_func = get_label_func(config.loss)
    loss_func = get_loss_func(config.loss)
    loss_func = DataParallelCriterion(loss_func)
    result_func = get_result_func(config.loss)
    merge_flip_func = get_merge_func(config.loss)

    # dataset, --detector=maskRCNN_R50-FPN
    logger.info("Creating dataset")
    target_id = config.dataiter.target_id
    test_imdbs = []
    test_imdbs.append(
        eval(config.dataset.name[target_id])(config.dataset.test_image_set[target_id], config.dataset.path[target_id],
         config.train.patch_width, config.train.patch_height,
         config.train.rect_3d_width, config.train.rect_3d_height))

    train_imdbs = []
    train_imdbs.append(  # H36M
        eval(config.dataset.name[2])(config.dataset.train_image_set[2], config.dataset.path[2],
         config.train.patch_width, config.train.patch_height,
         config.train.rect_3d_width, config.train.rect_3d_height))

    batch_size = len(devices) * config.dataiter.batch_images_per_ctx

    dataset_test = eval(config.dataset.name[target_id] + "_Dataset")(
        [test_imdbs[0]], False, s_args.detector, config.train.patch_width,
        config.train.patch_height, config.train.rect_3d_width, config.train.rect_3d_height, batch_size,
        config.dataiter.mean, config.dataiter.std, config.aug, label_func, config.loss)

    test_data_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False,
                                  num_workers=config.dataiter.threads, drop_last=False)

    dataset_train = eval(config.dataset.name[2] + "_Dataset")(   # H36M
        [train_imdbs[0]], False, s_args.detector, config.train.patch_width,
        config.train.patch_height, config.train.rect_3d_width, config.train.rect_3d_height, batch_size,
        config.dataiter.mean, config.dataiter.std, config.aug, label_func, config.loss)

    test_imdbs[0].mean_bone_length = train_imdbs[0].mean_bone_length

    # prepare network
    assert os.path.exists(s_args.model), 'Cannot find model!'
    logger.info('Load checkpoint from {}'.format(s_args.model))
    # joint_num = dataset_test.joint_num
    joint_num = dataset_train.joint_num
    net = get_pose_net(config.network, joint_num)
    net = DataParallelModel(net).cuda()  # claim multi-gpu in CUDA_VISIBLE_DEVICES
    ckpt = torch.load(s_args.model)  # or other path/to/model
    net.load_state_dict(ckpt['network'])
    logger.info("Net total params: {:.2f}M".format(sum(p.numel() for p in net.parameters()) / 1000000.0))

    # test
    logger.info("Test DB size: {}.".format(int(len(dataset_test))))
    print("Now Time is:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    beginT = time.time()
    preds_in_patch = None
    preds_in_patch, _ = validNet(test_data_loader, net, config.loss, result_func, loss_func, merge_flip_func,
                                 config.train.patch_width, config.train.patch_height, devices,
                                 test_imdbs[0].flip_pairs, flip_test=False, flip_fea_merge=False)
    # evalNetChallenge(0, preds_in_patch, test_data_loader, test_imdbs[0], final_log_path)
    print('Testing %.2f seconds.....' % (time.time() - beginT))

if __name__ == "__main__":
    main()
