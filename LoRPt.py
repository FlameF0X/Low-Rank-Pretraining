class LoRPtLinear(nn.Module):
    """Low-rank factorized linear layer for memory efficiency"""
    def __init__(self, in_features, out_features, rank=64):
        super().__init__()
        self.rank = rank
        self.lora_A = nn.Parameter(torch.randn(out_features, rank) * 0.02)
        self.lora_B = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        weight = self.lora_A @ self.lora_B
        return F.linear(x, weight, self.bias)

# This is all :D
