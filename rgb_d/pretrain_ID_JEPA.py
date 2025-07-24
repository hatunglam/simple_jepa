import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    ModelSummary,
)
#from pytorch_lightning.loggers import WandbLogger
from model_rgbd import IDJEPA_base


'''Dummy Dataset'''
class IJEPADataset(Dataset):
    def __init__(self,
                 dataset_path,
                 stage='train',
                 ):
        super().__init__()
        rgb =torch.randn(3, 100, 100)
        depth = torch.randn(1, 100, 100)
        
        self.rgb_data = rgb.repeat(100, 1, 1, 1)
        self.depth_data = depth.repeat(100, 1, 1, 1)
        
    def __len__(self):
        return len(self.rgb_data)
    
    def __getitem__(self, index): 
        data = {
            "rgb_image" : self.rgb_data[index],
            "depth_image" : self.depth_data[index]
        }
        
        return data
        # return self.rgb_data[index], self.depth_data[index]


'''Placeholder for datamodule in pytorch lightning'''
'''
Placeholder for datamodule in pytorch lightning
'''
class D2VDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_path,
                 batch_size=10,
                 num_workers=0, # Set to 0 for no additional workers from 4
                 pin_memory=False, # set to True if using GPU
                 shuffle=True
                 ):
        super().__init__()
        
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        
    def setup(self, stage=None):
        self.train_dataset = IJEPADataset(dataset_path=self.dataset_path, stage='train')
        self.val_dataset = IJEPADataset(dataset_path=self.dataset_path, stage='val')
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

'''
pytorch lightning model
'''
class IDJEPA(pl.LightningModule):
    def __init__(
            self,
            img_size=100,
            patch_size=10,
            in_chans_rgb=3,
            in_chans_depth=1, 
            embed_dim=64,
            enc_heads=8,
            enc_depth=8,
            decoder_depth=6,
            lr=1e-6,
            weight_decay=0.05,
            target_aspect_ratio = (0.75,1.5),
            target_scale = (0.15, .2),
            context_aspect_ratio = 1,
            context_scale = (0.85,1.0),
            M = 4, #number of different target blocks
            m=0.996, #momentum
            m_start_end = (.996, 1.)

    ):
        super().__init__()
        self.save_hyperparameters()
        
        #define models
        self.model = IDJEPA_base(img_size=img_size, patch_size=patch_size, in_chans_rgb=in_chans_rgb,
                                in_chans_depth=in_chans_depth, embed_dim=embed_dim, 
                                enc_depth = enc_depth, num_heads=enc_heads, pred_depth=decoder_depth, M=M)

        #define hyperparameters
        self.M = M
        self.lr = lr
        self.weight_decay = weight_decay
        self.m = m
        self.target_aspect_ratio = target_aspect_ratio
        self.target_scale = target_scale
        self.context_aspect_ratio = context_aspect_ratio
        self.context_scale = context_scale
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_tokens = (img_size // patch_size) ** 2
        self.m_start_end = m_start_end

        #define loss
        self.criterion = nn.MSELoss()
    
    def forward(self,data, target_aspect_ratio, target_scale, context_aspect_ratio, context_scale):
        return self.model(data, target_aspect_ratio, target_scale, context_aspect_ratio, context_scale)
    
    '''Update momentum for teacher encoder'''
    def update_momentum(self, m):
        student_model = self.model.student_encoder.eval()
        teacher_model = self.model.teacher_encoder.eval()
        with torch.no_grad():
            for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()):
                teacher_param.data.mul_(other=m).add_(other=student_param.data, alpha=1 - m)


    def training_step(self, batch, batch_idx):
        #generate random target and context aspect ratio and scale
        target_aspect_ratio = np.random.uniform(self.target_aspect_ratio[0], self.target_aspect_ratio[1])
        target_scale = np.random.uniform(self.target_scale[0], self.target_scale[1])
        context_aspect_ratio = self.context_aspect_ratio
        context_scale = np.random.uniform(self.context_scale[0], self.context_scale[1])

        y_student, y_teacher = self(batch, target_aspect_ratio, target_scale, context_aspect_ratio, context_scale)
        loss = self.criterion(y_student, y_teacher)
        self.log('train_loss', loss)
        print(f"Epoch {self.current_epoch}: Train loss = {loss.item():.4f}")
                    
        return loss
    
    def validation_step(self, batch, batch_idx):
        target_aspect_ratio = np.random.uniform(self.target_aspect_ratio[0], self.target_aspect_ratio[1])
        target_scale = np.random.uniform(self.target_scale[0], self.target_scale[1])
        context_aspect_ratio = self.context_aspect_ratio
        context_scale = np.random.uniform(self.context_scale[0], self.context_scale[1])

        y_student, y_teacher = self(batch, target_aspect_ratio, target_scale, context_aspect_ratio, context_scale)
        loss = self.criterion(y_student, y_teacher)
        self.log('val_loss', loss)
        
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx):
        target_aspect_ratio = np.random.uniform(self.target_aspect_ratio[0], self.target_aspect_ratio[1])
        target_scale = np.random.uniform(self.target_scale[0], self.target_scale[1])
        context_aspect_ratio = self.context_aspect_ratio
        context_scale = 1
        self.model.mode = "test"

        return self(batch, target_aspect_ratio, target_scale, context_aspect_ratio, context_scale) #just get teacher embedding

    def on_after_backward(self):
        self.update_momentum(self.m)
        self.m += (self.m_start_end[1] - self.m_start_end[0]) / self.trainer.estimated_stepping_batches


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


if __name__ == '__main__':
    dataset = D2VDataModule(dataset_path='data')

    model = IDJEPA(img_size=100, patch_size=10, in_chans_rgb=3, in_chans_depth=1, embed_dim=64, enc_heads=8, enc_depth=8, decoder_depth=6, lr=1e-3)
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)

    trainer = pl.Trainer(
        accelerator='cpu',
        devices=1,
        #precision=16,
        max_epochs=15,
        callbacks=[lr_monitor, model_summary],
        gradient_clip_val=.1,
    )

    trainer.fit(model, dataset)

    from predict_rgbd import predict_test

    predict_test(model)


