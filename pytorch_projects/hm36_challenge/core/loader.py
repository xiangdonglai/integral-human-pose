import numpy as np
import os
import cv2
import json
import torch

import torch.utils.data as data

from common.utility.image_processing_cv import get_single_patch_sample

from common_pytorch.dataset.hm36 import from_mpii_to_hm36, from_coco_to_hm36


class single_patch_Dataset(data.Dataset):
    def __init__(self, db, is_train, det_bbox_src, patch_width, patch_height, rect_3d_width, rect_3d_height, batch_size,
                 mean, std, aug_config, label_func, label_config):

        if det_bbox_src:
            if det_bbox_src == 'org_img':
                self.db = db[0].org_db()
            else:
                self.db = db[0].dt_db(det_bbox_src)
        else:
            self.db = db[0].jnt_bbox_db()

        self.num_samples = len(self.db)

        self.joint_num = db[0].joint_num

        self.is_train = is_train
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.rect_3d_width = rect_3d_width
        self.rect_3d_height = rect_3d_height
        self.mean = mean
        self.std = std
        self.aug_config = aug_config
        self.label_func = label_func
        self.label_config = label_config

        if self.is_train:
            self.do_augment = True
        else:
            self.do_augment = False
            # padding samples to match input_batch_size
            extra_db = len(self.db) % batch_size
            for i in range(0, batch_size - extra_db):
                self.db.append(self.db[i])

        self.depth_in_image = True

        if det_bbox_src == 'org_img':
            self.do_augment = False
            self.depth_in_image = 'org_img'

        self.db_length = len(self.db)

    def __getitem__(self, index):
        the_db = self.db[index]

        img_patch, label, label_weight = \
            get_single_patch_sample(the_db['image'], the_db['center_x'], the_db['center_y'],
                                    the_db['width'], the_db['height'],
                                    the_db['joints_3d'].copy(), the_db['joints_3d_vis'].copy(),
                                    the_db['flip_pairs'].copy(), the_db['parent_ids'].copy(),
                                    self.patch_width, self.patch_height,
                                    self.rect_3d_width, self.rect_3d_height, self.mean, self.std,
                                    self.do_augment, self.aug_config, self.label_func, self.label_config,
                                    depth_in_image=self.depth_in_image)

        return img_patch.astype(np.float32), label.astype(np.float32), label_weight.astype(np.float32)

    def __len__(self):
        return self.db_length


class hm36_Dataset(single_patch_Dataset):
    def __init__(self, db, is_train, det_bbox_src, patch_width, patch_height, rect_3d_width, rect_3d_height, batch_size,
                 mean, std, aug_config, label_func, label_config):
        single_patch_Dataset.__init__(self, db, is_train, det_bbox_src, patch_width, patch_height, rect_3d_width,
                                      rect_3d_height, batch_size, mean, std, aug_config,
                                      label_func, label_config)


class hm36_eccv_challenge_Dataset(single_patch_Dataset):
    def __init__(self, db, is_train, det_bbox_src, patch_width, patch_height, rect_3d_width, rect_3d_height, batch_size,
                 mean, std, aug_config, label_func, label_config):
        single_patch_Dataset.__init__(self, db, is_train, det_bbox_src, patch_width, patch_height, rect_3d_width,
                                      rect_3d_height, batch_size, mean, std, aug_config,
                                      label_func, label_config)


class mpii_hm36_eccv_challenge_Dataset(data.Dataset):
    def __init__(self, db, is_train, det_bbox_src, patch_width, patch_height, rect_3d_width, rect_3d_height, batch_size,
                 mean, std, aug_config, label_func, label_config):

        assert det_bbox_src == ''

        self.db0 = db[0].jnt_bbox_db()
        self.db1 = db[1].jnt_bbox_db()

        self.num_samples0 = len(self.db0)
        self.num_samples1 = len(self.db1)

        from_mpii_to_hm36(self.db0)

        self.joint_num = db[1].joint_num

        self.is_train = is_train
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.rect_3d_width = rect_3d_width
        self.rect_3d_height = rect_3d_height
        self.mean = mean
        self.std = std
        self.aug_config = aug_config
        self.label_func = label_func
        self.label_config = label_config

        if self.is_train:
            self.do_augment = True
        else:
            assert 0, "testing not supported for mpii_hm36_Dataset"

        self.db_length = self.num_samples0 * 2

        self.count = 0
        self.idx = np.arange(self.num_samples1)
        np.random.shuffle(self.idx)

    def __getitem__(self, index):
        if index < self.num_samples0:
            the_db = self.db0[index]
        else:
            the_db = self.db1[self.idx[index - self.num_samples0]]

        img_patch, label, label_weight = \
            get_single_patch_sample(the_db['image'], the_db['center_x'], the_db['center_y'],
                                    the_db['width'], the_db['height'],
                                    the_db['joints_3d'].copy(), the_db['joints_3d_vis'].copy(),
                                    self.db1[0]['flip_pairs'].copy(), self.db1[0]['parent_ids'].copy(),
                                    self.patch_width, self.patch_height,
                                    self.rect_3d_width, self.rect_3d_height, self.mean, self.std,
                                    self.do_augment, self.aug_config,
                                    self.label_func, self.label_config,
                                    depth_in_image=True)

        self.count = self.count + 1
        if self.count >= self.db_length:
            self.count = 0
            self.idx = np.arange(self.num_samples1)
            np.random.shuffle(self.idx)

        return img_patch.astype(np.float32), label.astype(np.float32), label_weight.astype(np.float32)

    def __len__(self):
        return self.db_length


class mpii_coco_hm36_eccv_challenge_Dataset(data.Dataset):
    def __init__(self, db, is_train, det_bbox_src, patch_width, patch_height, rect_3d_width, rect_3d_height, batch_size,
                 mean, std, aug_config, label_func, label_config):

        assert det_bbox_src == ''

        self.db0 = db[0].jnt_bbox_db()
        self.db1 = db[1].jnt_bbox_db()
        self.db2 = db[2].jnt_bbox_db()

        self.num_samples0 = len(self.db0)
        self.num_samples1 = len(self.db1)
        self.num_samples2 = len(self.db2)

        from_mpii_to_hm36(self.db0)
        from_coco_to_hm36(self.db1)

        self.joint_num = db[2].joint_num

        self.is_train = is_train
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.rect_3d_width = rect_3d_width
        self.rect_3d_height = rect_3d_height
        self.mean = mean
        self.std = std
        self.aug_config = aug_config
        self.label_func = label_func
        self.label_config = label_config

        if self.is_train:
            self.do_augment = True
        else:
            assert 0, "testing not supported for mpii_hm36_Dataset"

        self.db_length = self.num_samples0 * 3

        self.count = 0
        self.idx1 = np.arange(self.num_samples1)
        np.random.shuffle(self.idx1)
        self.idx2 = np.arange(self.num_samples2)
        np.random.shuffle(self.idx2)

    def __getitem__(self, index):
        if index < self.num_samples0:
            the_db = self.db0[index]
        elif index < self.num_samples0 * 2:
            the_db = self.db1[self.idx1[index - self.num_samples0]]
        else:
            the_db = self.db2[self.idx2[index - self.num_samples0 * 2]]

        img_patch, label, label_weight = get_single_patch_sample(the_db['image'], the_db['center_x'],
                                                                 the_db['center_y'],
                                                                 the_db['width'], the_db['height'],
                                                                 the_db['joints_3d'].copy(),
                                                                 the_db['joints_3d_vis'].copy(),
                                                                 self.db2[0]['flip_pairs'].copy(),
                                                                 self.db2[0]['parent_ids'].copy(), self.patch_width,
                                                                 self.patch_height, self.rect_3d_width,
                                                                 self.rect_3d_height, self.mean,
                                                                 self.std, self.do_augment, self.aug_config,
                                                                 self.label_func, self.label_config, depth_in_image=True)

        self.count = self.count + 1
        if self.count >= self.db_length:
            self.count = 0
            self.idx1 = np.arange(self.num_samples1)
            np.random.shuffle(self.idx1)
            self.idx2 = np.arange(self.num_samples2)
            np.random.shuffle(self.idx2)

        return img_patch.astype(np.float32), label.astype(np.float32), label_weight.astype(np.float32)

    def __len__(self):
        return self.db_length


def read_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    kps = []
    for people in data['people']:
        # kp = np.array(people['pose_keypoints']).reshape(-1, 3)
        kp = np.array(people['pose_keypoints_2d']).reshape(-1, 3)  # the new OpenPose version
        kps.append(kp)
    return kps


def get_bbox(json_path, vis_thr=0.2):
    kps = read_json(json_path)
    # Pick the most confident detection.
    scores = [np.mean(kp[kp[:, 2] > vis_thr, 2]) for kp in kps]
    kp = kps[np.argmax(scores)]
    vis = kp[:, 2] > vis_thr
    vis_kp = kp[vis, :2]
    min_pt = np.min(vis_kp, axis=0)
    max_pt = np.max(vis_kp, axis=0)
    person_height = np.linalg.norm(max_pt - min_pt)
    if person_height == 0:
        print('bad!')
        import ipdb
        ipdb.set_trace()
    center = (min_pt + max_pt) / 2.
    scale = 315. / person_height

    return scale, center


def resize_img(img, scale_factor):
    new_size = (np.floor(np.array(img.shape[0:2]) * scale_factor)).astype(int)
    new_img = cv2.resize(img, (new_size[1], new_size[0]))
    # This is scale factor of [height, width] i.e. [y, x]
    actual_factor = [
        new_size[0] / float(img.shape[0]), new_size[1] / float(img.shape[1])
    ]
    return new_img, actual_factor


def scale_and_crop(image, scale, center, img_size):
    assert len(img_size) == 2
    image_scaled, scale_factors = resize_img(image, scale)
    # Swap so it's [x, y]
    scale_factors = [scale_factors[1], scale_factors[0]]
    center_scaled = np.round(center * scale_factors).astype(np.int)

    margin_y = int(img_size[1] / 2)
    margin_x = int(img_size[0] / 2)
    margin = np.array([margin_x, margin_y])
    image_pad = np.pad(
        image_scaled, ((margin_y, ), (margin_x, ), (0, )), mode='edge')
    center_pad = center_scaled + margin
    # figure out starting point
    start_pt = center_pad - margin
    end_pt = center_pad + margin
    # crop:
    crop = image_pad[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0], :]
    proc_param = {
        'scale': scale,
        'start_pt': start_pt,
        'end_pt': end_pt,
        'img_size': img_size
    }

    return crop, proc_param


class inthewild_Dataset(data.Dataset):
    seqName = 'dslr_dance1'

    def __init__(self, db, is_train, det_bbox_src, patch_width, patch_height, rect_3d_width, rect_3d_height, batch_size,
                 mean, std, aug_config, label_func, label_config):
        # super(inthewild_Dataset, self).__init__()
        self.patch_width = patch_width
        self.patch_height = patch_height

        self.mean = mean
        self.std = std

        self.joint_num = 0
        self.image_root = '/media/posefs1b/Users/donglaix/siggasia018/{}/openpose_image/'.format(self.seqName)
        self.json_root = '/media/posefs1b/Users/donglaix/siggasia018/{}/openpose_result/'.format(self.seqName)

        assert batch_size == 1

    def __len__(self):
        if self.seqName == 'dslr_dance1':
            return 360
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        imgName = os.path.join(self.image_root, '{}_{:012d}_rendered.png'.format(self.seqName, idx))
        img = cv2.imread(imgName)[:, :, ::-1]

        jsonName = os.path.join(self.json_root, '{}_{:012d}_keypoints.json'.format(self.seqName, idx))
        scale, center = get_bbox(jsonName)

        crop, params = scale_and_crop(img, scale, center, np.array([288, 384]))

        # import matplotlib.pyplot as plt
        # plt.imshow(crop)
        # plt.show()

        crop = ((crop - self.mean) / self.std)
        crop = np.transpose(crop, (2, 0, 1))  # C, H, W
        return (torch.from_numpy(crop).float(), )
