from transformers import TrainerCallback

class FreezeThenUnfreeze(TrainerCallback):
    def __init__(self, freeze_epochs=2):
        self.freeze_epochs = freeze_epochs

    def _set_encoder_trainable(self, model, trainable):
        for p in model.base_model.parameters():   # the encoder, without the head
            p.requires_grad = trainable
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Encoder trainable={trainable} | trainable params: {n:,}")

    def on_train_begin(self, args, state, control, model=None, **kw):
        self._set_encoder_trainable(model, False)

    def on_epoch_begin(self, args, state, control, model=None, **kw):
        if int(state.epoch) == self.freeze_epochs:
            self._set_encoder_trainable(model, True)

