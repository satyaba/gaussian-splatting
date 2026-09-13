import torch
from torch import nn

class SegmentationDecoder(nn.Module):
  def __init__(self, seg_encoding_dim, num_classes):
    super().__init__()
    self.linear = nn.Linear(seg_encoding_dim, num_classes)

  def forward(self, encoding):
    return self.linear(encoding)

def save_decoder_checkpoint(decoder, decoder_optimizer, iteration, path):
    torch.save({
        'iteration': iteration,
        'model_state_dict': decoder.state_dict(),
        'optimizer_state_dict': decoder_optimizer.state_dict(),
    }, path)

def load_decoder_checkpoint(decoder, decoder_optimizer, path):
    ckpt = torch.load(path, weights_only=False)
    decoder.load_state_dict(ckpt['model_state_dict'])
    decoder_optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt['iteration']
