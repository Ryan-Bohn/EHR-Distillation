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
    

class TransformerEncoderForTimeStepWiseClassificationLegacy(nn.Module):
    def __init__(self, num_features, num_classes, max_seq_len=320, embed_dim=64, num_heads=4, num_layers=3, dropout=0.1, do_embed_stamp=False):
        """
        If do_embed_stamp is set, the model will try to embed timestamp as a positional encoding and sum up with feature embedding.
        In this case, forward pass expects the first column of x being timestamps.
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.do_embed_stamp = do_embed_stamp

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

class TransformerEncoderForTimeStepWiseClassification(nn.Module):
    def __init__(self, num_features, num_classes, max_seq_len=320, embed_dim=64, num_heads=4, num_layers=3, dropout=0.1, do_embed_stamp=False):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.do_embed_stamp = do_embed_stamp
        self.embed_dim = embed_dim

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

    def forward(self, x, padding_mask): # TODO: dynamic tape is not what I thought it was!! It assumes timestamps to be equally distributed.
        seq_len = x.size(1)

        # Feature embedding
        x = self.feature_embedding(x)

        # Positional encoding
        if self.do_embed_stamp:
            pos_encoding = self.dynamic_tape(x.size(0), seq_len, padding_mask)
        else:
            pos_encoding = self.static_pe(seq_len)
        
        x = x + pos_encoding.to(x.device)

        # Create causal mask
        causal_mask = self.create_causal_mask(seq_len).to(x.device)

        # Transformer encoder
        encoded = self.transformer_encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)

        # Classification
        output = self.classifier(encoded)
        return output

    def dynamic_tape(self, batch_size, seq_len, padding_mask): # arxiv 2305.16642
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * (-math.log(10000.0) / self.embed_dim))
        pe = torch.zeros(seq_len, self.embed_dim)
        for i in range(batch_size):
            length = seq_len - padding_mask[i].sum().item()  # Actual sequence length
            adjusted_freq = div_term * length / self.embed_dim
            pe[:, 0::2] = torch.sin(position * adjusted_freq)
            pe[:, 1::2] = torch.cos(position * adjusted_freq)
        pe = pe.unsqueeze(0).expand(batch_size, -1, -1)  # Adjust shape for batch
        return pe

    def static_pe(self, seq_len):
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * (-math.log(10000.0) / self.embed_dim))
        pe = torch.zeros(1, seq_len, self.embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    @staticmethod
    def create_causal_mask(size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask



class TransformerEncoderPlusMimic3BenchmarkMultitaskHeads(nn.Module):
    def __init__(self, num_features, max_seq_len=320, embed_dim=32, num_heads=4, num_layers=3, dropout=0.3):
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

        # Classifier for IHM task (2-class classification)
        self.ihm_classifier = nn.Linear(embed_dim, 2)

        # Classifier for LOS task (regression)
        self.los_regressor = nn.Linear(embed_dim, 1)

        # Classifier for Pheno task (25 binary classifications)
        self.pheno_classifier = nn.Linear(embed_dim, 50)  # 25 tasks * 2 classes each

        # Classifier for Decomp task (binary classification)
        self.decomp_classifier = nn.Linear(embed_dim, 2)

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

        # Task-specific heads

        # IHM Classifier
        ihm_output = self.ihm_classifier(encoded)

        # LOS Regressor
        los_output = self.los_regressor(encoded)

        # Pheno Classifier
        pheno_output = self.pheno_classifier(encoded)
        # Reshape the pheno_output to [batch_size, seq_len, 25, 2]
        pheno_output = pheno_output.view(pheno_output.size(0), pheno_output.size(1), 25, 2)

        # Decomp Classifier
        decomp_output = self.decomp_classifier(encoded)

        # Return outputs for each task
        return {
            "ihm": ihm_output,
            "los": los_output,
            "pheno": pheno_output,
            "decomp": decomp_output
        }