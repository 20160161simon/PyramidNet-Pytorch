import torch
import torch.nn as nn

class ShakeDrop(nn.Module):
    def __init__(self, p_drop=0.5, alpha_range=(-1, 1)):
        super().__init__()
        self.p_drop = p_drop
        self.alpha_range = alpha_range

    def forward(self, x):
        if not self.training:
            return x * (1 - self.p_drop)
        
        batch_size = x.size(0)
        gate = torch.bernoulli(torch.ones(batch_size, device=x.device) * (1 - self.p_drop))
        alpha = torch.rand(batch_size, device=x.device) * (self.alpha_range[1] - self.alpha_range[0]) + self.alpha_range[0]
        
        mask = (gate + (1 - gate) * alpha).view(-1, 1, 1, 1)
        output = x * mask
        
        beta = torch.rand(batch_size, device=x.device).view(-1, 1, 1, 1)
        return BackwardShake.apply(output, beta, mask)

class BackwardShake(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, output, beta, mask):
        ctx.save_for_backward(beta, mask)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        beta, mask = ctx.saved_tensors
        grad_input = grad_output * (mask + (1 - mask) * beta)
        return grad_input, None, None