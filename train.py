try:
    # Attempt relative import (works when run as part of a package)
    from . import config
    from .dataset import DataModule
    from .model import Model
except ImportError:
    # Fallback to absolute import (works when run independently)
    import config
    from dataset import DataModule
    from model import Model

from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import torch


data_module = DataModule(data_dir=config.DATA_DIR)
model = Model(num_classes=config.NUM_CATEGORIES)

early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    mode='min'
)

trainer = Trainer(
    max_epochs=config.EPOCHS,
    accelerator=config.ACCELERATOR,
    devices=config.DEVICES,
    enable_checkpointing=config.CHECKPOINTING,
    log_every_n_steps=config.LOG_N_STEPS,
    default_root_dir=config.DATA_DIR,
    callbacks=[early_stopping_callback]
)

trainer.fit(model, data_module)
trainer.validate(model, data_module)
trainer.test(model, data_module)

# Save model
optimizer = model.configure_optimizers()
torch.save(model.state_dict(), config.MODEL_PATH)