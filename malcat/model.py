import torch
import torch.nn as nn


def relative_time_encoding(delta_t, dim_model):
    positions = delta_t.unsqueeze(-1)
    div_term = torch.exp(
        torch.arange(0, dim_model, 2, device=delta_t.device, dtype=delta_t.dtype)
        * (-torch.log(torch.tensor(10000.0, device=delta_t.device, dtype=delta_t.dtype)) / dim_model)
    )
    encoding = torch.zeros(*delta_t.shape, dim_model, device=delta_t.device, dtype=delta_t.dtype)
    encoding[..., 0::2] = torch.sin(positions * div_term)
    encoding[..., 1::2] = torch.cos(positions * div_term[: encoding[..., 1::2].shape[-1]])
    return encoding


class Malcat(nn.Module):
    def __init__(
        self,
        input_dim=2,
        dim_model=512,
        n_head=8,
        n_encoder_layers=6,
        dropout=0.1,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, dim_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)
        self.output_projection = nn.Linear(dim_model, input_dim)

    def forward(self, lc, padding_mask=None):
        h = self.input_projection(lc)
        h = h + relative_time_encoding(lc[..., 0], h.shape[-1])
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        return self.output_projection(h)


malcat = Malcat
