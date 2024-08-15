import numpy as np
import torch
import pytorch_lightning as pl


# ----------------------------------------------------------------
# ----------------------------------------------------------------
# ----------------------------------------------------------------
class Positional_Encoding(torch.nn.Module):
    """
    Returns a tensor of shape: [1, seq_len, embed_len]
    """

    def __init__(self, seq_len, embed_len):
        super(Positional_Encoding, self).__init__()
        pe = torch.zeros(seq_len, embed_len)
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_len, 2)
            * (-np.log(10000.0) / embed_len)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    #@torch.autocast(device_type="cuda")
    def forward(self, x_len: int) -> torch.Tensor:
        return self.pe[:x_len,:].unsqueeze(0)


# ----------------------------------------------------------------
# ----------------------------------------------------------------
if __name__ == "__main__":
    p_e_ = Positional_Encoding(80, 256)
    print(p_e_(80))
