from model import TransformerEncoderPlusMimic3BenchmarkMultitaskHeads
from torchinfo import summary

model = TransformerEncoderPlusMimic3BenchmarkMultitaskHeads(num_features=17, embed_dim=32, max_seq_len=320, num_heads=4, num_layers=3)
batch_size = 1
summary(model, input_size=((batch_size, 320, 17), (batch_size, 320)))