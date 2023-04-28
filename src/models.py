import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .networks import Discriminator, HazeRemovalNet,HazeProduceNet, DepthEstimationNet
from .loss import ContrastLoss, WeightedL1Loss, AdversarialLoss, PerceptualLoss, TVLoss

# torch.autograd.set_detect_anomaly(True)

class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()

        if config.MODEL == 1:
            self.name = 'reconstruct'


        self.config = config
        self.iteration = 0

        self.gen_weights_path = os.path.join(config.PATH, 'weights.pth')
        self.gen_optimizer_path = os.path.join(config.PATH, 'optimizer_'+self.name + '.pth')
        self.dis_weights_path = os.path.join(config.PATH, self.name + '_dis.pth')
        self.transformer_weights_path = os.path.join(config.PATH, self.name + '.pth')
        self.transformer_discriminator_weights_path = os.path.join(config.PATH, self.name + '_dis.pth')
        self.reconstructor_weights_path = os.path.join(config.PATH, self.name + '.pth')

    def load(self):
        pass

    def save(self, save_best, psnr, iteration):
        pass



class Model(BaseModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.l2_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.adversarial_loss = AdversarialLoss(type='lsgan')
        self.weighted_l1loss = WeightedL1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.contrastive_loss = ContrastLoss()
        self._mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1).cuda()
        self._std = torch.tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1).cuda()
        self.use_dc_A = True if config.USE_DC_A == 1 else False


        self.depth_estimator = DepthEstimationNet(config.BASE_CHANNEL_NUM // 2, min_d=config.MIN_D, max_d=config.MAX_D, path=self.gen_weights_path[:-4]+'_'+self.name+'.pth', is_real_model=config.IS_REAL_MODEL)
        self.net_h2c = HazeRemovalNet(config.BASE_CHANNEL_NUM //2 , min_beta=config.MIN_BETA, max_beta=config.MAX_BETA, min_d=config.MIN_D, max_d=config.MAX_D, path=self.gen_weights_path[:-4]+'_'+self.name+'.pth', use_dc_A=config.USE_DC_A, r=config.GF_R, is_real_model=config.IS_REAL_MODEL)
        self.net_c2h = HazeProduceNet(config.BASE_CHANNEL_NUM // 2, in_channels=3, out_channels=3, min_beta=config.MIN_BETA, max_beta=config.MAX_BETA)


        self.epoch = 0

        if config.MODE == 1:
            self.discriminator_h2c = Discriminator(in_channels=3, use_spectral_norm=True, use_sigmoid=True)
            self.discriminator_c2h = Discriminator(in_channels=3, use_spectral_norm=True, use_sigmoid=True)

            self.optimizer = optim.Adam(
                [
                    {'params': self.net_c2h.parameters()},
                    {'params': self.net_h2c.parameters()},
                ],

                lr=float(config.LR),
                betas=(config.BETA1, config.BETA2),
                weight_decay=config.WEIGHT_DECAY
            )

            self.optimizer_depth = optim.Adam(
                [
                    {'params': self.depth_estimator.parameters()},
                ],

                lr=float(config.LR),
                betas=(config.BETA1, config.BETA2),
                weight_decay=config.WEIGHT_DECAY
            )

            self.optimizer_dis = optim.Adam(
                [
                    {'params': self.discriminator_h2c.parameters()},
                    {'params': self.discriminator_c2h.parameters()},
                ],

                lr=float(config.LR * config.D2G_LR),
                betas=(config.BETA1, config.BETA2)
            )

            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=800, last_epoch=self.epoch-1)



    def forward_depth(self, clean_imgs):
        n,c,h,w = clean_imgs.shape
        input = F.interpolate(clean_imgs, mode='bilinear', size=[self.config.CROP_SIZE,self.config.CROP_SIZE])
        result = self.depth_estimator(input)
        result = F.interpolate(result, mode='bilinear', size=[h,w])
        return result

    def forward_depth_see(self, clean_imgs):
        depth = self.depth_estimator(clean_imgs)
        n, c, h, w = depth.shape
        D_max = torch.max(depth.view(n,c,-1), dim=2, keepdim=True)[0].unsqueeze(3)
        D_min = torch.min(depth.view(n,c,-1), dim=2, keepdim=True)[0].unsqueeze(3)
        depth = (depth-D_min)/(D_max-D_min)
        return depth

    def forward_c2h_given_parameters(self, clean_imgs, ex, beta):
        x = self.net_c2h.forward(clean_imgs,ex,beta)
        return x


    def forward_c2h_random_parameters(self, clean_imgs, ex, requires_paras=False): # random beta
        x, beta = self.net_c2h.forward_random_parameters(clean_imgs, ex)
        if requires_paras:
            return x, beta
        else:
            return x


    def forward_h2c(self, hazy_imgs, require_paras=False, times=1.0, use_guided_filter=False):
        # if requires_paras: return clean, ed, beta
        return self.net_h2c(hazy_imgs, require_paras, times=times, use_guided_filter=use_guided_filter)


    def process(self, clean_images, hazy_images, clean_2, hazy_2):
        self.iteration += 1

        self.optimizer_dis.zero_grad()
        self.discriminator_h2c.zero_grad()
        self.discriminator_c2h.zero_grad()

        clean_images_h2c, gt_ed_h2c, pred_beta_h2c = self.forward_h2c(hazy_images, require_paras=True)
        pred_ed_h2c = self.forward_depth(clean_images_h2c)
        hazy_images_h2c2h = self.forward_c2h_given_parameters(clean_images_h2c, pred_ed_h2c, pred_beta_h2c)


        pred_ed_clean = self.forward_depth(clean_images)
        hazy_images_c2h, beta_gt_c2h = self.forward_c2h_random_parameters(clean_images, pred_ed_clean, requires_paras=True)

        clean_images_c2h2c, ed_pred_c2h, beta_pred_c2h = self.forward_h2c(hazy_images_c2h,require_paras=True)


        gen_loss = 0
        dis_loss = 0

        #### dis loss ####
        dis_real_clean, _ = self.discriminator_h2c(clean_images)
        dis_fake_clean, _ = self.discriminator_h2c(
            clean_images_h2c.detach())

        dis_clean_real_loss = self.adversarial_loss((dis_real_clean), is_real=True, is_disc=True)
        dis_clean_fake_loss = self.adversarial_loss((dis_fake_clean), is_real=False, is_disc=True)

        dis_clean_loss = (dis_clean_real_loss + dis_clean_fake_loss) / 2
        dis_clean_loss.backward()

        dis_real_haze, _ = self.discriminator_c2h(
            (hazy_images))
        dis_fake_haze, _ = self.discriminator_c2h(
            hazy_images_c2h.detach())

        dis_haze_real_loss = self.adversarial_loss((dis_real_haze), is_real=True, is_disc=True)
        dis_haze_fake_loss = self.adversarial_loss((dis_fake_haze), is_real=False, is_disc=True)
        dis_haze_loss = (dis_haze_real_loss + dis_haze_fake_loss) / 2
        dis_haze_loss.backward()

        dis_loss += (dis_clean_fake_loss + dis_clean_real_loss + dis_haze_real_loss + dis_haze_fake_loss) / 4

        self.optimizer_dis.step()

        ### gen loss ####
        self.optimizer.zero_grad()
        self.net_h2c.zero_grad()
        self.net_c2h.zero_grad()

        ### cycle reconstruction loss###
        cycle_loss_c2h2c = self.l1_loss(clean_images,
                                        clean_images_c2h2c)  # + self.depth_aware_l1loss(clean_images, clean_images_c2h2c, clean_images_depth)
        cycle_loss_h2c2h = self.l1_loss(hazy_images, hazy_images_h2c2h)
        cycle_loss = (cycle_loss_c2h2c + cycle_loss_h2c2h)

        ### Contrast loss ###
        contrastive_loss_h2c2h = self.contrastive_loss(p=clean_2, a=clean_images_h2c, n=hazy_2)
        contrastive_loss_c2h2c = self.contrastive_loss(p=hazy_2, a=hazy_images_c2h, n=clean_2)
        contrastive_loss = (contrastive_loss_c2h2c + contrastive_loss_h2c2h) / 2

        ### para loss ###
        para_beta_loss = self.l2_loss(beta_pred_c2h, beta_gt_c2h.detach()) / (self.config.MAX_BETA - self.config.MIN_BETA)
        para_loss = para_beta_loss

        ### global ###
        gen_fake_haze, _ = self.discriminator_c2h(
            (hazy_images_c2h))
        gen_fake_clean, _ = self.discriminator_h2c(
            clean_images_h2c)

        gen_fake_haze_ganloss = self.adversarial_loss((gen_fake_haze), is_real=True, is_disc=False)
        gen_fake_clean_ganloss = self.adversarial_loss((gen_fake_clean), is_real=True, is_disc=False)
        gen_gan_loss = (gen_fake_clean_ganloss + gen_fake_haze_ganloss) / 2

        ### total loss ###

        gen_loss += self.config.GAN_LOSS_WEIGHT * gen_gan_loss
        gen_loss += self.config.CYCLE_LOSS_WEIGHT * cycle_loss
        gen_loss += self.config.CONTRAST_LOSS_WEIGHT * contrastive_loss
        gen_loss += self.config.PARA_LOSS_WEIGHT * para_loss


        gen_loss.backward()
        self.optimizer.step()

        self.optimizer_depth.zero_grad()
        self.depth_estimator.zero_grad()



        depth_net_loss = 0
        pred_ed_h2c = self.forward_depth(clean_images_h2c.detach())
        if self.config.USE_DCP == 0:
            depth_loss = self.l1_loss(gt_ed_h2c.detach(), pred_ed_h2c) / (self.config.MAX_D-self.config.MIN_D)
        else:
            A_dcp = self.net_h2c.dcgf_tool.get_atmosphere_light_new(hazy_images)
            t_dcp = self.net_h2c.dcgf_tool.get_transmission(hazy_images, A_dcp)
            t_dcp = self.net_h2c.dcgf_tool.get_refined_transmission(hazy_images, t_dcp).clamp(0.05, 0.95)

            pseudo_depth_dcp = (torch.log(t_dcp)/(-pred_beta_h2c)).detach()

            pred_ed_h2c = self.forward_depth(clean_images_h2c.detach())
            depth_loss = self.l1_loss(pseudo_depth_dcp.detach(), pred_ed_h2c)/(self.config.MAX_D-self.config.MIN_D)
        depth_net_loss += depth_loss
        depth_net_loss.backward()

        self.optimizer_depth.step()


        self.optimizer_depth.step()

        logs = [
            ("g_cyc", cycle_loss.item()),
            ("g_para", para_loss.item()),
            ("g_contrast", contrastive_loss.item()),
            ("g_depth", depth_net_loss.item()),
            ("g_gan", gen_gan_loss.item()),
            ("g_total", gen_loss.item()),
            ("d_dis", dis_loss.item()),
            ("lr", self.get_current_lr()),
        ]
        return clean_images_c2h2c, gen_loss, dis_loss, logs


    def get_current_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def save(self, save_best, psnr, iteration):


        if self.config.MODEL == 1:
            torch.save({
                'net_h2c':self.net_h2c.state_dict(),
                'net_c2h':self.net_c2h.state_dict(),
                'net_depth':self.depth_estimator.state_dict(),
            },self.gen_weights_path[:-4]+'_'+self.name+'.pth' if not save_best else self.gen_weights_path[
                                                                   :-4] +'_'+self.name+ "_%.2f" % psnr + "_%d" % iteration + '.pth', _use_new_zipfile_serialization=False)
            torch.save({'discriminator_c2h': self.discriminator_c2h.state_dict(),
                        'discriminator_h2c': self.discriminator_h2c.state_dict(),
                        }, self.gen_weights_path[
                           :-4] + '_' + self.name + '_dis.pth' if not save_best else self.gen_weights_path[
                                                                                     :-4] + '_' + self.name + "_dis_%.2f" % psnr + "_%d" % iteration + '.pth')

            torch.save({
                'iteration': self.iteration,
                'epoch': self.epoch,
                'scheduler': self.scheduler.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'optimizer_dis': self.optimizer_dis.state_dict(),
                'optimizer_depth':self.optimizer_depth.state_dict()

            }, self.gen_optimizer_path if not save_best else self.gen_optimizer_path[
                                                           :-4] + "_%.2f" % psnr + "_%d" % iteration + '.pth', _use_new_zipfile_serialization=False)


    def load(self):
        if os.path.exists(self.gen_weights_path[:-4] + '_reconstruct' + '.pth'):
            print('Loading %s weights...' % 'reconstruct')

            if torch.cuda.is_available():
                weights = torch.load(self.gen_weights_path[:-4] + '_reconstruct' + '.pth')
            else:
                weights = torch.load(self.gen_weights_path[:-4] + '_reconstruct' + '.pth',
                                     lambda storage, loc: storage)

            self.net_h2c.load_state_dict(weights['net_h2c'])
            self.net_c2h.load_state_dict(weights['net_c2h'])
            self.depth_estimator.load_state_dict(weights['net_depth'])

            print('Loading %s weights...' % 'reconstruct complete!')

        if os.path.exists(self.gen_weights_path[:-4] + '_' + self.name + '_dis.pth') and self.config.MODE == 1:
            print('Loading discriminator weights...')

            if torch.cuda.is_available():
                weights = torch.load(self.gen_weights_path[:-4] + '_' + self.name + '_dis.pth')
            else:
                weights = torch.load(self.gen_weights_path[:-4] + '_' + self.name + '_dis.pth',
                                     lambda storage, loc: storage)

            self.discriminator_c2h.load_state_dict(weights['discriminator_c2h'])
            self.discriminator_h2c.load_state_dict(weights['discriminator_h2c'])


        if os.path.exists(self.gen_optimizer_path) and self.config.MODE == 1:
            print('Loading %s optimizer...' % self.name)
            if torch.cuda.is_available():
                data = torch.load(self.gen_optimizer_path)
            else:
                data = torch.load(self.gen_optimizer_path, lambda storage, loc: storage)

            self.optimizer.load_state_dict(data['optimizer'])
            self.scheduler.load_state_dict(data['scheduler'])
            self.epoch = data['epoch']
            self.iteration = data['iteration']
            self.optimizer_dis.load_state_dict(data['optimizer_dis'])
            self.optimizer_depth.load_state_dict(data['optimizer_depth'])

    def backward(self, gen_loss):
        gen_loss.backward()
        self.optimizer.step()


    def update_scheduler(self):
        self.scheduler.step()








