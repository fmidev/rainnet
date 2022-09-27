"""
RainNet model definition.
Bent Harnist (FMI) 2022
"""
import numpy as np
import torch
import pytorch_lightning as pl
from networks import RainNet as RN
from costfunctions import *

class RainNet(pl.LightningModule):
    """Model for the RainNet neural network."""
    
    def __init__(self, config): 
        
        super().__init__()
        self.save_hyperparameters()
        
        self.personal_device = torch.device(config.train_params.device)
        self.network = RN(
            kernel_size = config.model.kernel_size,
            mode = config.model.mode,
            conv_shape= config.model.conv_shape)
        
        if config.train_params.loss.name == "log_cosh":
            self.criterion = LogCoshLoss()
        elif config.train_params.loss.name == "ssim":
            self.criterion = SSIM(**config.train_params.loss.kwargs)
        elif config.train_params.loss.name == "ms_ssim":
            self.criterion = MS_SSIM(**config.train_params.loss.kwargs)
        elif config.train_params.loss.name == "mix":
            self.criterion = MixLoss(**config.train_params.loss.kwargs)
        elif config.train_params.loss.name == "gaussian_nll":
            self.criterion = GaussianNLL(**config.train_params.loss.kwargs)
        else:
            raise NotImplementedError(f"Loss {config.train_params.loss.name} not implemented!")
            
        # leadtime parameters
        self.verif_leadtimes = config.train_params.verif_leadtimes        
        self.predict_leadtimes = config.train_params.predict_leadtimes
        self.train_leadtimes = config.train_params.train_leadtimes

        # 1.0 corresponds to harmonic loss weight decrease, 0.0 to no decrease at all, less than 1.0 is sub-harmonic, more is super-harmonic
        discount_rate = config.train_params.loss.discount_rate 
        # equal weighting for each lt, sum to one.
        if discount_rate == 0:
            self.train_loss_weights = np.ones(self.train_leadtimes) / self.train_leadtimes
            self.verif_loss_weights = np.ones(self.verif_leadtimes) / self.verif_leadtimes
        # Diminishing weight by n_lt^( - discount_rate), sum to one.
        else:
            train_t = np.arange(1,self.train_leadtimes+1)
            self.train_loss_weights = (train_t**(- discount_rate) / (train_t**(- discount_rate)).sum())
            verif_t = np.arange(1,self.verif_leadtimes+1)
            self.verif_loss_weights = (verif_t**(- discount_rate) / (verif_t**(- discount_rate)).sum())
        
        # optimization parameters
        self.lr = float(config.train_params.lr)
        self.lr_sch_params = config.train_params.lr_scheduler
        self.automatic_optimization = False

    def forward(self, x):
        return self.network(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.lr_sch_params.name is None:
            return optimizer
        elif self.lr_sch_params.name == "reduce_lr_on_plateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, **self.lr_sch_params.kwargs)
            return [optimizer],[lr_scheduler]
        else:
            raise NotImplementedError("Lr scheduler not defined.")
                
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        y_hat, total_loss = self._iterative_prediction(batch=batch, stage="train")
        opt.step()
        opt.zero_grad()
        self.log("train_loss", total_loss)
        return {"prediction" : y_hat, "loss" : total_loss}
    
    def validation_step(self, batch, batch_idx):
        y_hat, total_loss = self._iterative_prediction(batch=batch, stage="valid")
        self.log("val_loss", total_loss)
        return {"prediction" : y_hat, "loss" : total_loss}

    def validation_epoch_end(self, outputs):
        torch.cuda.empty_cache()
        sch = self.lr_schedulers()
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val_loss"])

    def test_step(self, batch, batch_idx):
        y_hat, total_loss = self._iterative_prediction(batch=batch, stage="test")
        self.log("test_loss", total_loss)
        return {"prediction" : y_hat, "loss" : total_loss}


    def predict_step(self, batch, batch_idx: int, dataloader_idx = None):
        y_seq = self._iterative_prediction(batch=batch, stage="predict")
        return self.trainer.datamodule.predict_dataset.scaled_to_dbz(y_seq)


    def _iterative_prediction(self, batch, stage):

        if stage == "train":
            n_leadtimes = self.train_leadtimes
            calculate_loss = True
            loss_weights = self.train_loss_weights
        elif stage == "valid" or stage == "test":
            n_leadtimes = self.verif_leadtimes
            calculate_loss = True
            loss_weights = self.verif_loss_weights
        elif stage == "predict":
            n_leadtimes = self.predict_leadtimes
            calculate_loss = False
        else:
            raise ValueError(f"Stage {stage} is undefined. \n choices: 'train', 'valid', test', 'predict'")

        x,y,_ = batch
        y = y[:,:n_leadtimes]
        y_seq = torch.empty(
            (x.shape[0], n_leadtimes, *x.shape[-2:]),
            device=self.device
            )
        if calculate_loss : 
            total_loss = 0

        for i in range(n_leadtimes):
            y_hat = self(x)
            if calculate_loss:
                y_i = y[:,None,i,:,:].clone()
                loss = self.criterion(y_hat, y_i) * loss_weights[i]
                total_loss += loss.detach()
                if stage == "train":
                    self.manual_backward(loss)
            y_seq[:, i, :, :] = y_hat.detach().squeeze()
            if i != n_leadtimes - 1 :
                x = torch.roll(x, -1, dims=1)
                x[:,3,:,:] = y_hat.detach().squeeze()
        if calculate_loss:
            return y_seq, total_loss
        else:
            return y_seq
