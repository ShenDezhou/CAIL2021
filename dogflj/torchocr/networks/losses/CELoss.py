from jittor import nn
import jittor


class CELoss(nn.Module):

    def __init__(self, ignore_index=None):
        super().__init__()
        self.loss_func = jittor.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def execute(self, pred, args):
        batch_size = pred.shape[0]
        label = args['targets']
        pred = pred.log_softmax(2)
        pred = pred.permute(1, 0, 2)
        # preds_lengths = torch.tensor([pred.shape[0]] * batch_size, dtype=torch.long)
        loss = self.loss_func(pred, label)
        return {'loss': loss}