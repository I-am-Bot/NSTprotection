import torch
import torch.nn as nn

PATH = "/egr/research-dselab/liyaxin1"
import torchvision
from pathlib import Path
from torchvision import transforms
from PIL import Image
import os
import numpy as np


from options.test_options import TestOptions
from models import create_model

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),

    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),

    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),

    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


class Net(nn.Module):
    def __init__(self, encoder, decoder=None):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        # self.decoder = decoder
        self.mse_loss = nn.MSELoss()

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)


    """
    1/24/2023 added: non-targeted attack
    """
    def adv_loss(self, input_mean, input_std, target_mean, target_std):
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def calc_adv_loss(self, style_feats, x_adv):
        x_adv.requires_grad_()
        with torch.enable_grad():
            adv_feats = self.encode_with_intermediate(x_adv)
            # import ipdb
            # ipdb.set_trace()

            loss_adv = 0

            adv_mean, adv_std = calc_mean_std(adv_feats[3])
            target_mean, target_std = calc_mean_std(style_feats[3])
            # import ipdb
            # ipdb.set_trace()
            return self.adv_loss(target_mean.detach(), target_std.detach(), adv_mean, adv_std)


# #AdaAttn
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


class AdaAttN(nn.Module):
    def __init__(self, in_planes, max_sample=256 * 256, key_planes=None):
        super(AdaAttN, self).__init__()
        if key_planes is None:
            key_planes = in_planes
        self.f = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.g = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.max_sample = max_sample

    def forward(self, content, style, content_key, style_key, seed=None):
        F = self.f(content_key)
        G = self.g(style_key)
        H = self.h(style)
        b, _, h_g, w_g = G.size()
        G = G.view(b, -1, w_g * h_g).contiguous()
        if w_g * h_g > self.max_sample:
            if seed is not None:
                torch.manual_seed(seed)
            index = torch.randperm(w_g * h_g).to(content.device)[:self.max_sample]
            G = G[:, :, index]
            style_flat = H.view(b, -1, w_g * h_g)[:, :, index].transpose(1, 2).contiguous()
        else:
            style_flat = H.view(b, -1, w_g * h_g).transpose(1, 2).contiguous()
        b, _, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        S = torch.bmm(F, G)
        # S: b, n_c, n_s
        S = self.sm(S)
        # mean: b, n_c, c
        mean = torch.bmm(S, style_flat)
        # std: b, n_c, c
        std = torch.sqrt(torch.relu(torch.bmm(S, style_flat ** 2) - mean ** 2))
        # mean, std: b, c, h, w
        mean = mean.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        std = std.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return std * mean_variance_norm(content) + mean
    def adv_forward(self, content, style, content_key, style_key, seed=None):
        F = self.f(content_key)
        G = self.g(style_key)
        H = self.h(style)
        b, _, h_g, w_g = G.size()
        G = G.view(b, -1, w_g * h_g).contiguous()
        if w_g * h_g > self.max_sample:
            if seed is not None:
                torch.manual_seed(seed)
            index = torch.randperm(w_g * h_g).to(content.device)[:self.max_sample]
            G = G[:, :, index]
            style_flat = H.view(b, -1, w_g * h_g)[:, :, index].transpose(1, 2).contiguous()
        else:
            style_flat = H.view(b, -1, w_g * h_g).transpose(1, 2).contiguous()
        b, _, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        S = torch.bmm(F, G)
        # S: b, n_c, n_s
        S = self.sm(S)
        # mean: b, n_c, c
        mean = torch.bmm(S, style_flat)
        # std: b, n_c, c
        std = torch.sqrt(torch.relu(torch.bmm(S, style_flat ** 2) - mean ** 2))
        # mean, std: b, c, h, w
        mean = mean.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        std = std.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return mean, std

class Transformer(nn.Module):

    def __init__(self, in_planes, key_planes=None):
        super(Transformer, self).__init__()
        self.attn_adain_4_1 = AdaAttN(in_planes=in_planes, key_planes=key_planes)
        self.attn_adain_5_1 = AdaAttN(in_planes=in_planes, key_planes=key_planes + 512)
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))

    def forward(self, content4_1, style4_1, content5_1, style5_1, content4_1_key, style4_1_key, content5_1_key, style5_1_key, seed=None, content_mask=None, style_mask=None):
        return self.merge_conv(self.merge_conv_pad(
            self.attn_adain_4_1(
                content4_1, style4_1, content4_1_key, style4_1_key, seed) +
            self.upsample5_1(self.attn_adain_5_1(
                content5_1, style5_1, content5_1_key, style5_1_key, seed))))
    
    def adv_forward(self, content4_1, style4_1, content5_1, style5_1, content4_1_key, style4_1_key, content5_1_key, style5_1_key, seed=None, content_mask=None, style_mask=None):
        mean_4, std_4 = self.attn_adain_4_1.adv_forward(
                content4_1, style4_1, content4_1_key, style4_1_key, seed)
        mean_5, std_5 = self.attn_adain_5_1.adv_forward(
                content5_1, style5_1, content5_1_key, style5_1_key, seed)
        return mean_4, std_4, mean_5, std_5

class AdaAttNModel(nn.Module):

    def __init__(self, image_encoder):
        super(AdaAttNModel, self).__init__()
        enc_layers = list(image_encoder.children())
        enc_1 = nn.Sequential(*enc_layers[:4])
        enc_2 = nn.Sequential(*enc_layers[4:11])
        enc_3 = nn.Sequential(*enc_layers[11:18])
        enc_4 = nn.Sequential(*enc_layers[18:31])
        enc_5 = nn.Sequential(*enc_layers[31:44])
        self.image_encoder_layers = [enc_1, enc_2, enc_3, enc_4, enc_5]
        self.seed = 6666
        self.shallow_layer = True
        self.mse_loss = nn.MSELoss()

        self.max_sample = 64 * 64

        transformer_path = '/egr/research-dselab/liyaxin1/unlearnable/AdaAttN/checkpoints/AdaAttN/latest_net_transformer.pth'
        net_adaattn_3_path = '/egr/research-dselab/liyaxin1/unlearnable/AdaAttN/checkpoints/AdaAttN/latest_net_adaattn_3.pth'
        self.transformer = Transformer(in_planes=512, key_planes=512 + 256 + 128 + 64).to(device)
        self.net_adaattn_3 = AdaAttN(in_planes=256, key_planes=256 + 128 + 64, max_sample=256 * 256).to(device)
        self.transformer.load_state_dict(torch.load(transformer_path))
        self.net_adaattn_3.load_state_dict(torch.load(net_adaattn_3_path))
        self.net_adaattn_3.eval()
        self.transformer.eval()
    
    def zero_grad(self):
        for param in self.net_adaattn_3.parameters():
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()
        
        for param in self.transformer.parameters():
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

        for enc in self.image_encoder_layers:
            # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
            for param in enc.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad.zero_()
        

    def encode_with_intermediate(self, input_img):
        results = [input_img]
        for i in range(5):
            func = self.image_encoder_layers[i]
            results.append(func(results[-1]))
        return results[1:]

    @staticmethod
    def get_key(feats, last_layer_idx, need_shallow=True):
        if need_shallow and last_layer_idx > 0:
            results = []
            _, _, h, w = feats[last_layer_idx].shape
            for i in range(last_layer_idx):
                results.append(mean_variance_norm(nn.functional.interpolate(feats[i], (h, w))))
            results.append(mean_variance_norm(feats[last_layer_idx]))
            return torch.cat(results, dim=1)
        else:
            return networks.mean_variance_norm(feats[last_layer_idx])
    
    def adv_loss(self, a, b):
        return self.mse_loss(a.mean(dim=(2, 3)), b.mean(dim=(2, 3)))

    def calc_adv_loss(self, c_feats, s_feats, x_adv):
        x_adv.requires_grad_()
        mean_3, std_3 = self.net_adaattn_3.adv_forward(c_feats[2], s_feats[2], self.get_key(c_feats, 2, True),
                                                self.get_key(s_feats, 2, True), self.seed)
        mean_4, std_4, mean_5, std_5 = self.transformer.adv_forward(c_feats[3], s_feats[3], c_feats[4], s_feats[4], self.get_key(c_feats, 3), self.get_key(s_feats, 3),
                     self.get_key(c_feats, 4), self.get_key(s_feats, 4), None, None, None)
        with torch.enable_grad():
            adv_s_feats = self.encode_with_intermediate(x_adv)

            adv_mean_3, adv_std_3 = self.net_adaattn_3.adv_forward(c_feats[2], adv_s_feats[2], self.get_key(c_feats, 2, True),
                                                self.get_key(adv_s_feats, 2, True), self.seed)
            adv_mean_4, adv_std_4, adv_mean_5, adv_std_5  = self.transformer.adv_forward(c_feats[3], adv_s_feats[3], c_feats[4], adv_s_feats[4], self.get_key(c_feats, 3), self.get_key(adv_s_feats, 3),
                     self.get_key(c_feats, 4), self.get_key(adv_s_feats, 4), None, None, None)      

        return (self.mse_loss(mean_3, adv_mean_3) + self.mse_loss(std_3, adv_std_3) + \
                self.mse_loss(mean_4, adv_mean_4) + self.mse_loss(std_4, adv_std_4) + \
                self.mse_loss(mean_5, adv_mean_5) + self.mse_loss(std_5, adv_std_5)) / 3                           
        # return (self.adv_loss(mean_3, adv_mean_3) + self.adv_loss(std_3, adv_std_3) + \
        #         self.adv_loss(mean_4, adv_mean_4) + self.adv_loss(std_4, adv_std_4) + \
        #         self.adv_loss(mean_5, adv_mean_5) + self.adv_loss(std_5, adv_std_5)) / 3                                                                   
    
    def calc_adv_loss_2(self, style_feature, x_adv):
        x_adv.requires_grad_()
        # import ipdb
        # ipdb.set_trace()
        with torch.enable_grad():
            adv_s_feats = self.encode_with_intermediate(x_adv) 
            adv_mean_3, adv_std_3 = calc_mean_std(adv_s_feats[2])
            adv_mean_4, adv_std_4 = calc_mean_std(adv_s_feats[3])
            adv_mean_5, adv_std_5 = calc_mean_std(adv_s_feats[4])
            
            mean_3, std_3 = calc_mean_std(style_feature[2])
            mean_4, std_4 = calc_mean_std(style_feature[3])
            mean_5, std_5 = calc_mean_std(style_feature[4])            
        return (10 * self.mse_loss(mean_3, adv_mean_3) + self.mse_loss(std_3, adv_std_3) + \
                self.mse_loss(mean_4, adv_mean_4) + self.mse_loss(std_4, adv_std_4) + \
                self.mse_loss(mean_5, adv_mean_5) + self.mse_loss(std_5, adv_std_5)) / 3 



device = 'cuda'
vgg_AdaIN = vgg

vgg_AdaIN.load_state_dict(torch.load("/egr/research-dselab/liyaxin1/unlearnable/pytorch-AdaIN/models/vgg_normalised.pth"))
vgg_AdaIN.to(device)
net_AdaIN = Net(vgg_AdaIN).to(device)
net_AdaIN.eval()


net_AdaAttN = AdaAttNModel(vgg_AdaIN).to(device)    


opt = TestOptions().parse()  # get test options
# hard-code some parameters for test
opt.num_threads = 0   # test code only supports num_threads = 1
opt.batch_size = 1    # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)


convert_tensor = transforms.ToTensor()
# img_s = Image.open("/egr/research-dselab/liyaxin1/unlearnable/style_images_small/2/charing-cross-bridge-03.jpg")

style_dir = Path('/egr/research-dselab/liyaxin1/unlearnable/resized_style_images/' + opt.style_num)
result_dir = '/egr/research-dselab/liyaxin1/unlearnable/ensemble/{:s}/{:s}/'.format("CAST_attack", opt.style_num)
if os.path.exists(result_dir) == False:
    os.makedirs(result_dir)
style_paths = sorted([f for f in style_dir.glob('*')])

# content_dir = Path('/egr/research-dselab/liyaxin1/unlearnable/select_contents/')
# content_paths = [f for f in content_dir.glob('*')]

# contents = []
# content_feats = []
# for i, content_path in enumerate(content_paths):
#     contents.append(style_tf(Image.open(str(content_path))).to(device).unsqueeze(0))

import time
start_time = time.time()
os.makedirs('/egr/research-dselab/liyaxin1/unlearnable/ensemble/ensemble_attack_resize/{:s}/'.format(opt.style_num), exist_ok=True)

for i, style_path in enumerate(style_paths):
    # import ipdb
    # ipdb.set_trace()
    if i % 10 == 0:
        end_time = time.time()
        running_time = end_time - start_time
        print("Running time before", i, " : ", running_time, "seconds")

    if Path(style_path).exists():
        style_name = Path(style_path).stem
        img_s = Image.open(str(style_path))
        style_AdaIN = convert_tensor(img_s).unsqueeze(0).cuda()
        style_cast = convert_tensor(img_s).unsqueeze(0).cuda() * 2 - 1

        epsilon_0 = 8.0 / 255.0
        alpha_0 = 0.8 / 255.0 

        epsilon_1 = 16.0 / 255.0
        alpha_1 = 1.6 / 255.0

        org_style_feats = model.netAE.encode_with_intermediate(style_cast)
        org_style_mean, org_style_std = model.netAE.calc_mean_std(org_style_feats[-1])
        # org_mean, org_std = model.netAE.encode_with_intermediate(style)

        style_feats_AdaIN = net_AdaIN.encode_with_intermediate(style_AdaIN)
        style_feats_AdaAttN = net_AdaAttN.encode_with_intermediate(style_AdaIN)

        x_adv_AdaIN = style_cast.detach() + 0.001 * torch.randn(style_cast.shape).cuda().detach()
        x_adv_cast = x_adv_AdaIN.clone().detach() * 2 - 1

        for _step in range(50):
            x_adv_cast.requires_grad_()
            x_adv_AdaIN.requires_grad_()
            loss_cast = model.netAE.adv_forward_loss(x_adv_cast, org_style_mean, org_style_std)
            loss_AdaIN = net_AdaIN.calc_adv_loss(style_feats_AdaIN, x_adv_AdaIN)
            loss_AdaAttN = net_AdaAttN.calc_adv_loss_2(style_feats_AdaAttN, x_adv_AdaIN)
            
            # print(loss)
            grad_cast = torch.autograd.grad(loss_cast, [x_adv_cast])[0].detach()
            grad_AdaIN = torch.autograd.grad(loss_AdaIN, [x_adv_AdaIN])[0].detach()
            grad_AdaAttN = torch.autograd.grad(loss_AdaAttN, [x_adv_AdaIN])[0].detach()
            # grad = grad / len(contents)

            x_adv_average = (x_adv_cast.detach() + alpha_0 * torch.sign(grad_cast)) * 0.5 + 0.5 + \
                            (x_adv_AdaIN.detach() + alpha_1 * torch.sign(grad_AdaIN)) + \
                            (x_adv_AdaIN.detach() + alpha_1 * torch.sign(grad_AdaAttN))
            x_adv_average = x_adv_average / 3
            # x_adv_CAST = x_adv_CAST.detach() + alpha * torch.sign(torch.tensor(grad))
            
            x_adv_average = torch.min(torch.max(x_adv_average, style_AdaIN - epsilon_0), style_AdaIN + epsilon_0)
            x_adv_average = torch.clamp(x_adv_average, 0, 1.0)
            
            x_adv_AdaIN = x_adv_average.clone().detach()
            x_adv_cast = x_adv_average.clone().detach() * 2 - 1

            # model.netAE.zero_grad()
            for param in model.netAE.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad.zero_()

        # torchvision.utils.save_image(x_adv_CAST[0] * 0.5 + 0.5, result_dir + '{:s}.png'.format(style_name))
        torchvision.utils.save_image(x_adv_AdaIN[0], '/egr/research-dselab/liyaxin1/unlearnable/ensemble/ensemble_attack_resize/{:s}/{:s}.png'.format(opt.style_num, style_name))
        # torchvision.utils.save_image(style[0] - x_adv[0], result_dir + 'noise_{:s}.png'.format(style_name))
        # torch.save(style[0] - x_adv[0], '.results/noise_test1.pt')

        print(i, "adv done.")
        del style_AdaIN
        del style_cast
        del org_style_feats
        del x_adv_average
        del x_adv_AdaIN
        del x_adv_cast
        img_s.close()

# import ipdb
# loss_3 = torch.tensor(0)
# for i, style_path in enumerate(style_paths):
#     # import ipdb
#     # ipdb.set_trace()
#     if i % 10 == 0:
#         end_time = time.time()
#         running_time = end_time - start_time
#         print("Running time before", i, " : ", running_time, "seconds")

#     if Path(style_path).exists():
#         style_name = Path(style_path).stem
#         style = style_tf(Image.open(str(style_path)))
#         style = style.to(device).unsqueeze(0)

#         epsilon = 8.0 / 255.0
#         alpha = 0.8 / 255.0
        
#         # epsilon = 16.0 / 255.0
#         # alpha = 16.0 / 255.0 
#         style_feats_AdaIN = net_AdaIN.encode_with_intermediate(style)
        
#         style_CAST = (style - 0.5) * 2
#         style_feats_CAST = net_CAST.encode_with_intermediate(style_CAST)

#         content_feats = []
#         style_feats_AdaAttN = net_AdaAttN.encode_with_intermediate(style)
#         # for content in contents:
#         #     content_feats.append(net_AdaAttN.encode_with_intermediate(content))

#         x_adv = style.detach() + 0.001 * torch.randn(style.shape).cuda().detach()
#         x_adv_CAST = style_CAST.detach() + 0.001 * torch.randn(style_CAST.shape).cuda().detach()

#         for _step in range(50):
#             # loss_1 = net_AdaIN.calc_adv_loss(style_feats_AdaIN, x_adv)
#             loss_2 = net_CAST.calc_adv_loss(style_feats_CAST, x_adv_CAST) #[TODO]x_adv * 2
#             # for j in range(5):
#             #     loss_3 = net_AdaAttN.calc_adv_loss(content_feats[j], style_feats_AdaAttN, x_adv)
#             #     # loss = loss_1 + loss_2 * args.alpha + loss_3
#             #     loss = loss_3
#             #     # print(loss)
#             #     # print(loss_1.item(), loss_2.item(), loss_3.item())
#             #     if j == 0:
#             #         grad = torch.autograd.grad(loss, [x_adv])[0].detach()
#             #     else:
#             #         grad += torch.autograd.grad(loss, [x_adv])[0].detach()
#             # loss_3 = net_AdaAttN.calc_adv_loss_2(style_feats_AdaAttN, x_adv)
#             # print(loss_2.item())
#             # loss = loss_3
#             # print(loss_1, loss_2, loss_3)

#             grad = torch.autograd.grad(loss_2, [x_adv_CAST])[0].detach()
#             # grad = torch.autograd.grad(loss_3, [x_adv])[0].detach()
#             # grad = grad / len(contents)

#             # x_adv = x_adv.detach() + alpha * torch.sign(torch.tensor(grad))
#             x_adv_CAST = x_adv_CAST.detach() + alpha * torch.sign(torch.tensor(grad))

#             # x_adv = torch.min(torch.max(x_adv, style - epsilon), style + epsilon)
#             # x_adv = torch.clamp(x_adv, 0, 1.0)

#             x_adv_CAST = torch.min(torch.max(x_adv_CAST, style_CAST - epsilon), style_CAST + epsilon)
#             x_adv_CAST = torch.clamp(x_adv_CAST, -1.0, 1.0)
            
#             # if _step == 49:
#             #     print(loss.item())
#             net_AdaAttN.zero_grad()
#             # for content in contents:
#             #     content = content.detach()
#             style = style.detach()

#         torchvision.utils.save_image(x_adv_CAST[0] * 0.5 + 0.5, result_dir + '{:s}.png'.format(style_name))
#         # torchvision.utils.save_image(x_adv[0], result_dir + '{:s}.png'.format(style_name))
#         # torchvision.utils.save_image(style[0] - x_adv[0], result_dir + 'noise_{:s}.png'.format(style_name))
#         torch.save(style[0] - x_adv[0], result_dir + 'noise_{:s}.pt'.format(style_name))
        
#         print("{:s}: save attack_{:s}.png".format(str(i), style_name))
#         del style
#         del content_feats
#         del style_feats_AdaAttN
#         del x_adv

    