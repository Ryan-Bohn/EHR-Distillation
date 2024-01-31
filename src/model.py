import torch
import torch.nn as nn
import math

import torch
import torch.nn as nn
import math

class TransformerEncoderForRegression(nn.Module):
    def __init__(self, num_features, max_seq_len=320, embed_dim=64, num_heads=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.max_seq_len = max_seq_len

        # feature embedding layer
        self.feature_embedding = nn.Linear(num_features, embed_dim)

        # transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # regression Layer
        self.regressor = nn.Linear(embed_dim, 1)

        # sinusoidal positional encoding
        position = torch.arange(self.max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(self.max_seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_seq_len, embed_dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, num_features]

        # truncate or pad the input sequence
        seq_len = x.size(1)
        if seq_len > self.max_seq_len:
            x = x[:, -self.max_seq_len:, :]  # keep the last max_seq_len time steps
        elif seq_len < self.max_seq_len:
            pad = torch.zeros(x.size(0), self.max_seq_len - seq_len, x.size(2)).to(x.device)
            x = torch.cat([x, pad], dim=1)

        # feature embedding
        x = self.feature_embedding(x)

        # add positional embeddings to feature embeddings
        pos_encoding = self.pe[:, :seq_len, :].expand(x.size(0), -1, -1)
        x = x + pos_encoding

        # passing the input through the transformer encoder
        encoded = self.transformer_encoder(x)

        # use the transformed [CLS] token for regression
        cls_token_output = encoded[:, 0, :]  # extract the [CLS] token's output
        output = self.regressor(cls_token_output)
        return output
    

class TransformerEncoderForTimeStepWiseRegression(nn.Module):
    def __init__(self, num_features, max_seq_len=320, embed_dim=64, num_heads=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.max_seq_len = max_seq_len

        # Feature embedding layer
        self.feature_embedding = nn.Linear(num_features, embed_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Regression Layer for each time step
        self.regressor = nn.Linear(embed_dim, 1)

        # Sinusoidal positional encoding
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_seq_len, embed_dim]
        self.register_buffer('pe', pe)

    @staticmethod
    def create_causal_mask(size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask

    def forward(self, x, padding_mask):
        # x: [batch_size, seq_len, num_features]
        seq_len = x.size(1)

        # Feature embedding
        x = self.feature_embedding(x)

        # Add positional embeddings to feature embeddings
        pos_encoding = self.pe[:, :seq_len, :].expand(x.size(0), -1, -1)
        x = x + pos_encoding

        # Create causal mask
        causal_mask = self.create_causal_mask(seq_len).to(x.device)

        # Passing the input through the transformer encoder with masks
        # print(f"causal mask is {causal_mask}")
        # print(f"padding mask is {padding_mask}")
        encoded = self.transformer_encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask, is_causal=True)
        # print(f"encoded (with causal mask) is {encoded}")
        # encoded_padding_mask_only = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        # print(f"encoded (with only padding mask) is {encoded_padding_mask_only}")
        # encoded_no_mask = self.transformer_encoder(x)
        # print(f"encoded (no mask) is {encoded_no_mask}")
        # encoded_no_mask = self.transformer_encoder(x)
        # print(f"Do it again to see {encoded_no_mask}")
        # print(f"encoded shape is {encoded.shape}")

        # Use the output of each time step for regression
        output = self.regressor(encoded).squeeze(-1)  # Shape: [batch_size, seq_len]
        return output
    

class TransformerEncoderForTimeStepWiseClassification(nn.Module):
    def __init__(self, num_features, num_classes, max_seq_len=320, embed_dim=64, num_heads=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.max_seq_len = max_seq_len

        # Feature embedding layer
        self.feature_embedding = nn.Linear(num_features, embed_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classifier Layer for each time step
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Sinusoidal positional encoding
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_seq_len, embed_dim]
        self.register_buffer('pe', pe)

    @staticmethod
    def create_causal_mask(size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask

    def forward(self, x, padding_mask):
        # x: [batch_size, seq_len, num_features]
        seq_len = x.size(1)

        # Feature embedding
        x = self.feature_embedding(x)

        # Add positional embeddings to feature embeddings
        pos_encoding = self.pe[:, :seq_len, :].expand(x.size(0), -1, -1)
        x = x + pos_encoding

        # Create causal mask
        causal_mask = self.create_causal_mask(seq_len).to(x.device)

        # Passing the input through the transformer encoder with masks
        encoded = self.transformer_encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)

        # Use the output of each time step for classification
        output = self.classifier(encoded)  # Shape: [batch_size, seq_len, num_classes]
        return output