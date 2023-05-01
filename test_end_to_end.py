import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util
from torchvision.utils import save_image
from pathlib import Path
from data.base_dataset import get_transform
from PIL import Image


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    if not os.path.exists("/egr/research-dselab/liyaxin1/unlearnable/CAST_pytorch/results/" + opt.result_dir):
        os.makedirs("/egr/research-dselab/liyaxin1/unlearnable/CAST_pytorch/results/" + opt.result_dir)
    # train_dataset = create_dataset(util.copyconf(opt, phase="train"))
    model = create_model(opt)      # create a model given opt.model and other options
    # create a webpage for viewing the results
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    content_for_calc_adv = Path("/egr/research-dselab/liyaxin1/unlearnable/pytorch-AdaIN/content_endtoend")
    content_for_calc_adv = content_for_calc_adv.glob('*')
    modified_opt = util.copyconf(opt, load_size=opt.load_size)
    transform = get_transform(modified_opt)
    content = [transform(Image.open(A_path).convert('RGB')) for A_path in content_for_calc_adv]
    for i, data in enumerate(dataset):

        print(Path(data['B_paths'][0]).stem)
        if i == 0:
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.parallelize()
        if opt.eval:
            model.eval()
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break

        model.set_input(data)  # unpack data from data loader

        # import ipdb
        # ipdb.set_trace()
    
        if opt.adv == True:
            model.adv_end_to_end(content, opt.result_dir)           # run adv
            
        else:
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()     # get image paths
            # if i % 5 == 0:  # save images to an HTML file
            #     print('processing (%04d)-th image... %s' % (i, img_path))
            # save_images(webpage, visuals, img_path, width=opt.display_winsize)
            if opt.adv == False and opt.calc_mean_std == False:
                output_dir = Path("/egr/research-dselab/liyaxin1/unlearnable/CAST_pytorch/results/" + opt.result_dir)
                output_fake_B = output_dir / '{:s}_{:s}{:s}'.format(Path(data['A_paths'][0]).stem, Path(data['B_paths'][0]).stem,'.png')
                save_image(visuals['fake_B'], output_fake_B)
    # webpage.save()  # save the HTML
