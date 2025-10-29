from utils.torch_utils import de_parallel, ModelEMA


class QuantizableModelEMA(ModelEMA):

    def update(self, model):
        """Updates the Exponential Moving Average (EMA) parameters based on the current model's parameters."""
        self.updates += 1
        d = self.decay(self.updates)

        msd = de_parallel(model).state_dict()  # model state_dict
        for k, v in self.ema.state_dict().items():
            
            if k.split('.')[-1] not in {'weight', 'bias'}:
                continue

            if v.dtype.is_floating_point:  # true for FP16 and FP32
                v *= d
                v += (1 - d) * msd[k].detach()
        # assert v.dtype == msd[k].dtype == torch.float32, f'{k}: EMA {v.dtype} and model {msd[k].dtype} must be FP32'
