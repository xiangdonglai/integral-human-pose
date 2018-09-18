import os
import pprint
import shutil
import copy
import time
import matplotlib

matplotlib.use('Agg')  # avoid error on philly

# define project dependency
import _init_paths
# common
from common.speedometer import Speedometer
from common.tensorboard import TensorboardCallback
from common.utility.logger import create_logger
from common.utility.folder import make_folder
from common.utility.visualization import plot_LearningCurve

# pytorch
import torch
from torch.utils.data import DataLoader

# import from common_pytorch
# from common_pytorch.dataset.__init__ import *
from common_pytorch.dataset.all_dataset import *

from common_pytorch.config_pytorch import update_config_from_file, update_config_from_args, s_args, s_config \
    , s_config_file
from common_pytorch.optimizer import get_optimizer
from common_pytorch.io_pytorch import save_model, save_latest_model, save_lowest_vloss_model
from common_pytorch.common_loss.balanced_parallel import DataParallelModel, DataParallelCriterion
from common_pytorch.net_modules import trainNet, validNet, evalNet

# import dynamic config
exec('from blocks.' + s_config.pytorch.block + \
     ' import get_default_network_config, get_pose_net, init_pose_net')
exec('from loss.' + s_config.pytorch.loss + \
     ' import get_default_loss_config, get_loss_func, get_label_func, get_result_func, get_merge_func')

from core.loader import mpii_Dataset, hm36_Dataset, mpii_hm36_Dataset, coco_Dataset, posetrack_Dataset \
    , coco_posetrack_Dataset

def main():
    # parsing specific config
    config = copy.deepcopy(s_config)
    config.network = get_default_network_config()  # defined in blocks
    config.loss = get_default_loss_config()

    config.imdb = None
    if 'posetrack' in config.dataset.name:
        config.imdb = get_default_posetrack_config()

    # TODO(xiao): check sufficiency
    config = update_config_from_file(config, s_config_file, check_necessity=True)
    config = update_config_from_args(config, s_args)  # config in argument is superior to config in file

    # create log and path
    final_output_path, final_log_path, logger = create_logger(s_config_file, config.dataset.train_image_set,
                                                              config.pytorch.output_path, config.pytorch.log_path)
    logger.info('training config:{}\n'.format(pprint.pformat(config)))
    # shutil.copy2(os.path.join(os.path.dirname(__file__), 'blocks', config.pytorch.block + '.py'), final_output_path)

    # tensorboard
    tensorboard_train = tensorboard_valid = None
    if config.pytorch.use_tensorboard:
        log_tf = ['train_tf', 'valid_tf']
        log_tf = [os.path.join(final_log_path, item) for item in log_tf]
        [make_folder(item) for item in log_tf]
        tensorboard_train = TensorboardCallback(log_tf[0])
        tensorboard_valid = TensorboardCallback(log_tf[1])

    # define devices create multi-GPU context
    os.environ["CUDA_VISIBLE_DEVICES"] = config.pytorch.gpus  # a safer method
    devices = [int(i) for i in config.pytorch.gpus.split(',')]
    logger.info("Using Devices: {}".format(str(devices)))

    # lable, loss, metric and result
    logger.info("Defining lable, loss, metric and result")
    label_func = get_label_func(config.loss)
    loss_func = get_loss_func(config.loss)
    loss_func = DataParallelCriterion(loss_func)
    result_func = get_result_func(config.loss)
    merge_flip_func = get_merge_func(config.loss)

    # dataset, basic imdb
    logger.info("Creating dataset")
    train_imdbs = []
    valid_imdbs = []
    for n_db in range(0, len(config.dataset.name)):
        train_imdbs.append(
            eval(config.dataset.name[n_db])(config.dataset.train_image_set[n_db], config.dataset.path[n_db],
                                            config.train.patch_width, config.train.patch_height,
                                            config.train.rect_3d_width, config.train.rect_3d_height,
                                            config.pytorch.use_philly, config.imdb))
        valid_imdbs.append(
            eval(config.dataset.name[n_db])(config.dataset.test_image_set[n_db], config.dataset.path[n_db],
                                            config.train.patch_width, config.train.patch_height,
                                            config.train.rect_3d_width, config.train.rect_3d_height,
                                            config.pytorch.use_philly, config.imdb))

    batch_size = len(devices) * config.dataiter.batch_images_per_ctx

    # basic data_loader unit
    dataset_name = ""
    for n_db in range(0, len(config.dataset.name)):
        dataset_name = dataset_name + config.dataset.name[n_db] + "_"
    dataset_train = \
        eval(dataset_name + "Dataset")(
            train_imdbs, True, '', config.train.patch_width, config.train.patch_height, config.train.rect_3d_width, config.train.rect_3d_height, batch_size,
            config.dataiter.mean, config.dataiter.std, config.aug, label_func, config.loss, config.pytorch.use_philly
        )

    dataset_valid = \
        eval(config.dataset.name[config.dataiter.target_id] + "_Dataset")(
            [valid_imdbs[config.dataiter.target_id]], False, config.dataiter.det_bbox_src, config.train.patch_width,
            config.train.patch_height, config.train.rect_3d_width, config.train.rect_3d_height, batch_size, config.dataiter.mean, config.dataiter.std, config.aug, label_func,
            config.loss, config.pytorch.use_philly
        )

    train_data_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True,
                                   num_workers=config.dataiter.threads, drop_last=True)
    valid_data_loader = DataLoader(dataset=dataset_valid, batch_size=batch_size, shuffle=False,
                                   num_workers=config.dataiter.threads, drop_last=False)

    # prepare network
    logger.info("Creating network")
    joint_num = dataset_train.joint_num
    assert dataset_train.joint_num == dataset_valid.joint_num
    net = get_pose_net(config.network, joint_num)
    init_pose_net(net, config.network)
    net = DataParallelModel(net).cuda()  # claim multi-gpu in CUDA_VISIBLE_DEVICES
    model_prefix = os.path.join(final_output_path, config.train.model_prefix)

    # Optimizer
    logger.info("Creating optimizer")
    optimizer, scheduler = get_optimizer(config.optimizer, net)

    # resume
    train_loss = []
    valid_loss = []
    latest_model = '{}_latest.pth.tar'.format(model_prefix)
    if s_args.autoresume and os.path.exists(latest_model):
        model_path = latest_model if os.path.exists(latest_model) else s_args.model
        assert os.path.exists(model_path), 'Cannot find model!'
        logger.info('Load checkpoint from {}'.format(model_path))

        # load state from ckpt
        ckpt = torch.load(model_path)
        config.train.begin_epoch = ckpt['epoch'] + 1
        net.load_state_dict(ckpt['network'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        train_loss.extend(ckpt['train_loss'])
        valid_loss.extend(ckpt['valid_loss'])

        assert config.train.begin_epoch >= 2, 'resume error. begin_epoch should no less than 2'
        logger.info('continue training the {0}th epoch, init from the {1}th epoch'.format(config.train.begin_epoch,
                                                                                          config.train.begin_epoch - 1))
    # net = nn.DataParallel(net, device_ids=devices).cuda()
    logger.info("Net total params: {:.2f}M".format(sum(p.numel() for p in net.parameters()) / 1000000.0))

    # train and valid
    vloss_min = 10000000.0
    logger.info("Train DB size: {}; Valid DB size: {}.".format(int(len(dataset_train)), int(len(dataset_valid))))
    for epoch in range(config.train.begin_epoch, config.train.end_epoch + 1):
        scheduler.step()
        logger.info(
            "Working on {}/{} epoch || LearningRate:{} ".format(epoch, config.train.end_epoch, scheduler.get_lr()[0]))
        speedometer = Speedometer(train_data_loader.batch_size, config.pytorch.frequent, auto_reset=False)

        beginT = time.time()
        tloss = trainNet(epoch, train_data_loader, net, optimizer, config.loss, loss_func, speedometer,
                         tensorboard_train)
        endt1 = time.time() - beginT

        beginT = time.time()
        preds_in_patch_with_score, vloss = \
            validNet(epoch, valid_data_loader, net, config.loss, result_func, loss_func, merge_flip_func,
                     config.train.patch_width, config.train.patch_height, devices, valid_imdbs[config.dataiter.target_id].flip_pairs,
                     tensorboard_valid, flip_test=False)
        endt2 = time.time() - beginT

        beginT = time.time()
        evalNet(epoch, preds_in_patch_with_score, valid_data_loader, valid_imdbs[config.dataiter.target_id],
                config.train.patch_width, config.train.patch_height, config.train.rect_3d_width, config.train.rect_3d_height, final_output_path, config.test, is_save=False)
        endt3 = time.time() - beginT
        logger.info('One epoch training %.1fs, validation %.1fs, evaluation %.1fs ' % (endt1, endt2, endt3))

        train_loss.append(tloss)
        valid_loss.append(vloss)

        if vloss < vloss_min:
            vloss_min = vloss
            save_lowest_vloss_model({
                'epoch': epoch,
                'network': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss
            }, model_prefix, logger)

        save_latest_model({
            'epoch': epoch,
            'network': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss
        }, model_prefix, logger)

        if (epoch % (config.train.end_epoch // 30) == 0 and epoch > 0.8*config.train.end_epoch) \
                or epoch == config.train.begin_epoch \
                or epoch == config.train.end_epoch:
            save_model({
                'epoch': epoch,
                'network': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss
            }, model_prefix, logger, epoch)

        jobName = os.path.basename(s_args.cfg).split('.')[0]
        plot_LearningCurve(train_loss, valid_loss, config.pytorch.log_path, jobName)
        plot_LearningCurve(train_loss, valid_loss, final_log_path, "Learning Curve")


if __name__ == "__main__":
    main()
