from models_old.integrated import IntegratedModel
from torch import optim
import torch

from models.base_model import BaseModel

class GAN_model(BaseModel):
    def __init__(self, args, character_names, dataset):
        super(GAN_model, self).__init__(args)

        self.character_names = character_names
        self.dataset = dataset
        self.n_topology = len(character_names)
        self.models = []
        self.D_para = []
        self.G_para = []
        self.args = args

        for i in range(self.n_topology):
            model = IntegratedModel(args, dataset.joint_topologies[i], None, self.device, character_names[i])
            self.models.append(model)
            self.D_para += model.D_parameters()
            self.G_para += model.G_parameters()

        if self.is_train:
            self.fake_pools = []
            self.optimizerD = optim.Adam(self.D_para, args.learning_rate, betas=(0.9, 0.999))
            self.optimizerG = optim.Adam(self.G_para, args.learning_rate, betas=(0.9, 0.999))
            self.optimizers = [self.optimizerD, self.optimizerG]
            self.criterion_rec = torch.nn.MSELoss()
            self.criterion_gan = GAN_loss(args.gan_mode).to(self.device)
            self.criterion_cycle = torch.nn.L1Loss()
            self.criterion_ee = Criterion_EE(args, torch.nn.MSELoss())
            for i in range(self.n_topology):
                self.fake_pools.append(ImagePool(args.pool_size))


    def set_input(self, motions):
        self.motions_input = motions


    def discriminator_requires_grad_(self, requires_grad):
        for model in self.models:
            for para in model.discriminator.parameters():
                para.requires_grad = requires_grad


    def forward(self):
        self.latents = []
        self.offset_repr = []
        self.pos_ref = []
        self.ee_ref = []
        self.res = []
        self.res_denorm = []
        self.res_pos = []
        self.fake_res = []
        self.fake_res_denorm = []
        self.fake_pos = []
        self.fake_ee = []
        self.fake_latent = []
        self.motions = []
        self.motion_denorm = []
        self.rnd_idx = []

        for i in range(self.n_topology):
            self.offset_repr.append(self.models[i].static_encoder(self.dataset.offsets[i]))

        # reconstruct
        for i in range(self.n_topology):
            motion, offset_idx = self.motions_input[i]
            motion = motion.to(self.device)
            self.motions.append(motion)

            motion_denorm = self.dataset.denorm(i, offset_idx, motion)
            self.motion_denorm.append(motion_denorm)
            offsets = [self.offset_repr[i][p][offset_idx] for p in range(self.args.num_layers + 1)]
            latent, res = self.models[i].auto_encoder(motion, offsets)
            res_denorm = self.dataset.denorm(i, offset_idx, res)
            res_pos = self.models[i].fk.forward_from_raw(res_denorm, self.dataset.offsets[i][offset_idx])
            self.res_pos.append(res_pos)
            self.latents.append(latent)
            self.res.append(res)
            self.res_denorm.append(res_denorm)

            pos = self.models[i].fk.forward_from_raw(motion_denorm, self.dataset.offsets[i][offset_idx]).detach()
            ee = get_ee(pos, self.dataset.joint_topologies[i], self.dataset.ee_ids[i],
                        velo=self.args.ee_velo, from_root=self.args.ee_from_root)
            height = self.models[i].height[offset_idx]
            height = height.reshape((height.shape[0], 1, height.shape[1], 1))
            ee /= height
            self.pos_ref.append(pos)
            self.ee_ref.append(ee)

    def load(self):
        pass

    def save(self):
        pass






        


        



        



