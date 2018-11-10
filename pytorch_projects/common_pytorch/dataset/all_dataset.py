from .imdb import IMDB
from .mpii import mpii
from .hm36 import hm36
from .hm36_eccv_challenge import hm36_eccv_challenge


class inthewild(IMDB):
    def __init__(self, image_set_name, dataset_path, patch_width, patch_height, rect_3d_width, rect_3d_height):
        super(inthewild, self).__init__('inthewild', image_set_name, dataset_path, patch_width, patch_height)
        self.flip_pairs = None   # dummy value

    @property
    def cache_path(self):
        """
        make a directory to store all caches
        :return: cache path
        """
        return ''
