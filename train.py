import config
from dataset import DataModule
from lightning import Trainer
from model import Model
from torch import save


data_module = DataModule(data_dir=config.DATA_DIR)
model = Model(num_classes=10)

trainer = Trainer(
    max_epochs=config.EPOCHS,
    accelerator=config.ACCELERATOR,
    devices=config.DEVICES,
    enable_checkpointing=config.CHECKPOINTING
)

trainer.fit(model, data_module)
trainer.validate(model, data_module)
trainer.test(model, data_module)

# Save model
optimizer = model.configure_optimizers()
saved_model_path = 'saved_model.pth'
save({
    'epoch': config.EPOCHS,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, saved_model_path)
