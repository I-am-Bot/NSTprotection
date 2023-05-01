import torch.nn as nn
import torch
import os
PATH = "/egr/research-dselab/liyaxin1"
import torchvision

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


class ADAIN_Encoder(nn.Module):
    def __init__(self, encoder, gpu_ids=[]):
        super(ADAIN_Encoder, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1 64
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1 128
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1 256
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1 512
        
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

    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def adain(self, content_feat, style_feat, calc_mean_std = False, style_name = "1"):
        assert (content_feat.size()[:2] == style_feat.size()[:2])
        size = content_feat.size()
        style_mean, style_std = self.calc_mean_std(style_feat)
        content_mean, content_std = self.calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)

        # print(style_std.shape)
        # print(style_mean.shape)
        # print((normalized_feat * style_std.expand(size) + style_mean.expand(size)).shape)

        if calc_mean_std == True:
        # normalized_feat * style_std.expand(size) + style_mean.expand(size) + 0.01
            if not os.path.exists("/egr/research-dselab/liyaxin1/unlearnable/CAST_pytorch/results/target_mean_std/"):
                os.mkdir("/egr/research-dselab/liyaxin1/unlearnable/CAST_pytorch/results/target_mean_std/")
            torch.save(style_mean, "/egr/research-dselab/liyaxin1/unlearnable/CAST_pytorch/results/target_mean_std/style_mean_{:s}.pt".format(style_name))
            torch.save(style_std, "/egr/research-dselab/liyaxin1/unlearnable/CAST_pytorch/results/target_mean_std/style_std_{:s}.pt".format(style_name))
        
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

    def adv_adain(self, style_feat):
        style_mean, style_std = self.calc_mean_std(style_feat)

        return style_mean, style_std

    def forward(self, content, style, encoded_only = False, adv = False, calc_mean_std = False, style_name="", artist_name=""):
        # print(style.shape)
        # print(content.shape)
        # input("check")
        # style = torch.load("/egr/research-dselab/liyaxin1/unlearnable/CAST_pytorch/results/demo/attack1.pt")
        if adv == True:
            self.adv_forward(content, style, style_name, artist_name)

        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)
        if calc_mean_std == True:
            adain_feat = self.adain(content_feats[-1], style_feats[-1], calc_mean_std = True, style_name = style_name)
            return 
        if encoded_only:
            return content_feats[-1], style_feats[-1]
        else:
            adain_feat = self.adain(content_feats[-1], style_feats[-1])
            return adain_feat

    def adv_forward(self, content, style, style_name, artist_name, encoded_only = False):

        # target_mean = torch.load("/egr/research-dselab/liyaxin1/unlearnable/CAST_pytorch/results/target_mean_std/style_mean_{:s}.pt".format(style_name))
        # target_std = torch.load("/egr/research-dselab/liyaxin1/unlearnable/CAST_pytorch/results/target_mean_std/style_std_{:s}.pt".format(style_name))
        criterion = torch.nn.MSELoss()

        epsilon = 16.0 / 255.0
        alpha = 1.6 / 255.0
 
        # print(torch.max(style))
        # print(torch.min(style))
        import ipdb
        ipdb.set_trace()

        style_feats = self.encode_with_intermediate(style)
        style_mean, style_std = self.adv_adain(style_feats[-1])
        target_mean, target_std = style_mean.clone(), style_std.clone()
        x_adv = style.detach() + 0.001 * torch.randn(style.shape).cuda().detach()
        for _step in range(50):
            x_adv.requires_grad_()
            with torch.enable_grad():
                style_feats = self.encode_with_intermediate(x_adv)
                content_feats = self.encode_with_intermediate(content)

                style_mean, style_std = self.adv_adain(style_feats[-1])
                loss = criterion(target_mean, style_mean) + criterion(target_std, style_std)
                print(loss)
            # import ipdb
            # ipdb.set_trace()    
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + alpha * torch.sign(grad.detach())
            # x_adv = x_adv.detach() - alpha * torch.sign(grad.detach())

            x_adv = torch.min(torch.max(x_adv, style - epsilon), style + epsilon)
            x_adv = torch.clamp(x_adv, -1.0, 1.0)
            # if _step == 49:
            #     print(loss.item())
        import ipdb
        ipdb.set_trace()
        diff = (x_adv[0] * 0.5 + 0.5) - (style[0] * 0.5 + 0.5)
        dir_name = artist_name
        if not os.path.exists("/egr/research-dselab/liyaxin1/unlearnable/CAST_pytorch/results/" + dir_name + "/"):
            os.mkdir("/egr/research-dselab/liyaxin1/unlearnable/CAST_pytorch/results/"+ dir_name + "/")
        # torch.save(x_adv, "/egr/research-dselab/liyaxin1/unlearnable/CAST_pytorch/results/{:s}/attack_{:s}.pt".format(dir_name, style_name))
        torchvision.utils.save_image(x_adv[0] * 0.5 + 0.5, "/egr/research-dselab/liyaxin1/unlearnable/CAST_pytorch/results/{:s}/attack_{:s}.png".format(dir_name, style_name))
        torchvision.utils.save_image(diff, "/egr/research-dselab/liyaxin1/unlearnable/CAST_noise/noise_{:s}.png".format(style_name))
        torch.save(diff, "/egr/research-dselab/liyaxin1/unlearnable/CAST_noise/noise_{:s}.pt".format(style_name))
        print("save attack_{:s}.png".format(style_name))
        # return  style_mean, style_std
        # input("attack done")
    def adv_forward_loss(self, x_adv, org_style_mean, org_style_std):
        criterion = torch.nn.MSELoss()
        with torch.enable_grad():
            style_feats = self.encode_with_intermediate(x_adv)
            style_mean, style_std = self.calc_mean_std(style_feats[-1])
            loss = criterion(org_style_mean, style_mean) + criterion(org_style_std, style_std)
        return loss

class Decoder(nn.Module):
    def __init__(self, gpu_ids=[]):
        super(Decoder, self).__init__()
        decoder = [
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(), # 256
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),# 128
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),# 64
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3))
            ]
        self.decoder = nn.Sequential(*decoder)

    def forward(self, adain_feat):
        fake_image = self.decoder(adain_feat)

        return fake_image
