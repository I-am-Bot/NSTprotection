from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        # Set the default = 5000 to test the whole test set.
        parser.add_argument('--num_test', type=int, default=5000, help='how many test images to run')
        parser.add_argument('--calc_mean_std', action='store_true', help='calculate mean and standard variance')
        parser.add_argument('--adv', action='store_true', help='generate adversary samples')
        parser.add_argument('--result_dir', default = 'type1')
        parser.add_argument('--content_dataroot')
        parser.add_argument('--style_dataroot')
        parser.add_argument('--style_num')

        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))

        self.isTrain = False
        return parser
