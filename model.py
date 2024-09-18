import config
from lightning import LightningModule
import torch
from torch.optim import SGD
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def calculate_loss(model, imgs, annotations):
    # Temporarily switch model to train mode only to compute the loss, but disable gradient calculations using torch.no_grad()
    # Revert back to eval mode after computing the loss
    with torch.no_grad():
        model.train()
        loss_dict = model(imgs, annotations)
        loss = sum(loss for loss in loss_dict.values())
        model.eval()
    return loss


class Model(LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.model = fasterrcnn_resnet50_fpn(pretrained=False)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        self.lr = config.LEARNING_RATE
        self.momentum = config.MOMENTUM
        self.weight_decay = config.WEIGHT_DECAY

    def forward(self, imgs, annotations):
        return self.model(imgs, annotations)

    def _common_step(self, batch, batch_idx):
        imgs, annotations = batch
        imgs = list(img.to(self.device) for img in imgs)
        annotations = [{k: v.to(self.device) for k, v in t.items()} for t in annotations]
        return imgs, annotations

    def training_step(self, batch, batch_idx):
        imgs, annotations = self._common_step(batch, batch_idx)
        loss_dict = self(imgs, annotations)
        losses = sum(loss for loss in loss_dict.values())
        self.log('train_loss', losses)
        return losses

    def validation_step(self, batch, batch_idx):
        imgs, annotations = self._common_step(batch, batch_idx)
        loss = calculate_loss(self.model, imgs, annotations)
        self.log('val_loss', loss)
    
    def test_step(self, batch, batch_idx):
        imgs, annotations = self._common_step(batch, batch_idx)
        loss = calculate_loss(self.model, imgs, annotations)
        self.log('test_loss', loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        return SGD(params, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
