# Time Series Distillation with Transformers

Experiment on mimic3 benchmark multitask set.

Run these preprocessing scripts first:

```shell
cd src
python preprocess.py -n mimic3benchmark -d ../data/mimic3/benchmark/multitask/
```

Some output statistics:

| split | dataset size (#episodes) | valid size | max ts length | avg ts length | std ts length |
| ----- | ------------------------ | ---------- | ------------- | ------------- | ------------- |
| train | 35621                    | 35571      | 2804          | 87.27         | 124.23        |
| test  | 6281                     | 6273       | 1993          | 88.85         | 127.70        |

## Length of stay regression preliminary

Do a time-step-wise regression on remaining length of stay.

Data collate manner:

- A batch consists of N sample features, key_padding_masks, masks and labels
- Key_padding_masks: for each sample, positions (time steps) marked as a padding (True in boolTensor) will not attend attention
- Masks: for each sample, only positions marked as 1 will contribute to the learning / eval loss (in other words, a valid data point). This mainly comes from mimic3 benchmark preprocessings

Learner model arch:

Transformer (encoder-only):

- Max_seq_len = 320 (~mu+2sigma)
- Mimic3 benchmark has 17 features, so input dim = 17
- Each feature token will then be embedded into embed_dim = 64
- 3 layers, with 4 heads, possibly dropout = 0.1
- A **regressor to map latent vector into a number**
- Every attention layer is futher masked by a triu causal matrix that prevent previous time steps from attending later positions, so that **the model learns a latent representation at each time step that encodes all previous information**

Training criterion: MSE loss; Eval metric: MSE loss averaged over each time step

![Figure_1](assets/Figure_1.png)

## In hospital mortality classification preliminary

All same as previous, except in transformer, instead of a regressor there's a forward layer mapping into 2-dim logit vector for binary classification.

Training criterion: CE loss; Eval metric: CE loss and AUROC score

Observed overfitting. Best eval AUROC ~0.82.

Efforts trying to make the performance align with reported in benchmark paper (unitask ~0.86, multitask ~0.87)

Firstly, **make sure the test set is normalized using the same stats as in training set.**

Then, do a hyperparameters search:

| slurm id     | num_layers | num_heads | embed_dim | dropout | lr   | wd   | best epoch | best test auroc |
| ------------ | ---------- | --------- | --------- | ------- | ---- | ---- | ---------- | --------------- |
| 19134073     | 3          | 4         | 32        | 0.1     | 1e-3 | 1e-3 | 60         | 0.8555          |
| 19134218     | 3          | 4         | 64        | 0.1     | 1e-3 | 5e-4 | 60         | 0.8578          |
| 19134259     | 3          | 4         | 32        | 0.1     | 1e-3 | 5e-4 | 77         | **0.8605**      |
| 19134296     | 2          | 2         | 32        | 0.1     | 1e-3 | 5e-4 | 80         | 0.8596          |
| 19134511     | 4          | 4         | 32        | 0.1     | 1e-3 | 5e-4 | 85         | **0.8607**      |
| 19134537     | 3          | 4         | 32        | 0.1     | 1e-3 | 1e-4 | 21         | 0.8596          |
| 19135203     | 3          | 4         | 32        | 0.2     | 1e-3 | 1e-4 | 38         | **0.8613**      |
| 19159827     | 3          | 4         | 32        | 0.2     | 1e-3 | 1e-4 | 34         | **0.8616**      |
| **19286460** | 3          | 4         | 32        | 0.3     | 1e-3 | 1e-4 | 53         | **0.8630!!**    |
| 19287365     | 3          | 4         | 32        | 0.5     | 1e-3 | 1e-4 | 61         | **0.8612**      |
| 19289511     | 3          | 4         | 32        | 0.2     | 1e-3 | 5e-4 | 89         | **0.8616**      |
| 19313189     | 3          | 4         | 32        | 0.3     | 1e-3 | 5e-4 | 95         | 0.8580          |

## Vanilla (multitask) dataset distillation

LOS task loss is extremely large and when its weight is not 0 nan will occur in both model parameters and learnable synthetic data. So temporarily except this task, considering only the rest 3.

![Figure_1](assets/Figure_2.png)

Evaluation (by train-on-synthetic, eval-on-real) at some distillation epochs:

Train epoch 10:

---------Results----------
Eval samples: 3235
Eval loss: 0.0027
Eval AUROC score: **0.7516**

## Distillation: single task (ihm) (2/28~)

slurm-a100.sh:

```
python -u main.py distill --method vanilla --tasks ihm
```

2 experiments:

- 19875321: on A100

  ```
  @dataclass
  class VanillaDistillConfig:
      n_samples: int = 1 # number of synthetic samples
      batch_size_syn: int = 1
      batch_size_real: int = 256 # minibatch size of real datasets
      max_seq_len: int = 320
      num_heads: int = 4
      num_layers: int = 3
      embed_dim: int = 32
      n_inner_steps: int = 50
      n_epochs: int = 100
      lr_data: float = 1e-3
      wd_data: float = 1e-4
      init_lr_model: float = 1e-3
      lr_lr_model: float = 1e-3
      min_lr_model: float = 1e-5
      ihm_w: float = 1
      los_w: float = 0 # TODO
      pheno_w: float = 0
      decomp_w: float = 0
  ```

- 19875323: inner loops 50 -> 10 (make it possible to run on gpu with less memory)



Timestamp encoding: TODO















## Appendix: Knowledge from ChatGPT

------------------

**You:**

{'key': '86018_episode5_timeseries.csv', 'feature':           capillary_refill_rate  diastolic_blood_pressure  fraction_inspired_oxygen  glascow_coma_scale_eye_opening  ...  systolic_blood_pressure  temperature    weight        ph
hour_bin                                                                                                             ...                                                          
0                     -0.063595                 -0.012875                 -0.492104                        0.594025  ...                -0.029213    -0.103468 -0.122331  0.110522
1                     -0.063595                  0.136659                 -0.492104                        0.594025  ...                 0.111480     0.321327 -1.053237  0.110522
2                     -0.063595                  0.077751                 -0.492104                        0.594025  ...                 0.098690     0.068473 -1.053237  0.110522
3                     -0.063595                  0.077751                 -0.492104                        0.594025  ...                 0.098690     0.119044 -1.053237  0.110522
4                     -0.063595                  0.100408                 -0.492104                        0.594025  ...                 0.105085     0.119044 -1.053237  0.110522
5                     -0.063595                  0.059626                 -0.492104                        0.594025  ...                 0.060319     0.119044 -1.053237  0.110522
6                     -0.063595                  0.059626                 -0.492104                        0.594025  ...                -0.022817     0.119044 -1.053237  0.110522
7                     -0.063595                  0.136659                 -0.492104                        0.594025  ...                 0.137060     0.119044 -1.053237  0.110522
8                     -0.063595                  0.136659                 -0.492104                        0.594025  ...                 0.188221     0.119044 -1.053237  0.110522
9                     -0.063595                  0.073220                 -0.492104                        0.594025  ...                 0.047529     0.119044 -1.053237  0.110522
10                    -0.063595                  0.086814                 -0.492104                        0.594025  ...                -0.010027     0.119044 -1.053237  0.110522
11                    -0.063595                  0.100408                 -0.492104                        0.594025  ...                 0.085899     0.119044 -1.053237  0.110522
12                    -0.063595                  0.109471                 -0.492104                        0.594025  ...                 0.105085     0.034759 -1.053237  0.110522
13                    -0.063595                  0.082283                 -0.492104                        0.594025  ...                 0.149851     0.034759 -1.053237  0.110522
14                    -0.063595                  0.046032                 -0.492104                        0.594025  ...                -0.029213     0.034759 -1.053237  0.110522
15                    -0.063595                  0.046032                 -0.492104                        0.594025  ...                 0.047529     0.034759 -1.053237  0.110522
16                    -0.063595                  0.077751                 -0.492104                        0.594025  ...                -0.022817     0.169615 -1.053237  0.110522
17                    -0.063595                  0.114002                 -0.492104                        0.594025  ...                 0.098690     0.169615 -1.053237  0.110522
18                    -0.063595                  0.114002                 -0.492104                        0.594025  ...                 0.073109     0.169615 -1.053237  0.110522
19                    -0.063595                  0.123065                 -0.492104                        0.594025  ...                 0.124270     0.169615 -1.053237  0.110522
20                    -0.063595                  0.141190                 -0.492104                        0.594025  ...                 0.143456     0.169615 -1.053237  0.110522
21                    -0.063595                  0.141190                 -0.492104                        0.594025  ...                 0.143456     0.169615 -1.053237  0.110522
22                    -0.063595                  0.141190                 -0.492104                        0.594025  ...                 0.143456     0.169615 -1.053237  0.110522

[23 rows x 17 columns], 'label': {'ihm': {'pos': 0, 'mask': 0, 'label': 0}, 'los': {'masks': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'labels': [22.3488, 21.3488, 20.3488, 19.3488, 18.3488, 17.3488, 16.3488, 15.3488, 14.3488, 13.3488, 12.3488, 11.3488, 10.3488, 9.3488, 8.3488, 7.3488, 6.3488, 5.3488, 4.3488, 3.3488, 2.3488, 1.3488, 0.3488]}, 'pheno': {'labels': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]}, 'decomp': {'masks': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'labels': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}}}

If I want to train a transformer with such feature data to do a regression task called "los": you can see the lengths of "masks" and "labels" are equal to the length of the feature time series. And each mask / label corresponds to a time bin (a row in the feature dataframe). Whenever mask is 1, all data before (and including) the current row is used to predict a single floating point number called "length of stay (los)", which is the corresponding value in "labels". How do I achieve this using transformer? All necessary preprocessings have already been done (normalization and stuff). Break this down step by step, including how and where to do the final pre-processings of the data (data frame to tensor, padding, positional encoding etc), how to form a training batch (do I have to make every pair of <feature_before_and_including_current_time_bin, current_los_ground_truth> as a sample in a batch), and how to adjust a transformer which is designed for seq2seq tasks to do a regression task

------------------

**ChatGPT:**

To train a Transformer model on your dataset for a regression task like "length of stay (los)", where each sample's input is all data up to a certain time bin and the output is a single floating-point number, you'll need to carefully structure your data preprocessing, batching, and model architecture. Here's a step-by-step breakdown:

### Step 1: Preprocessing Data

- **DataFrame to Tensor**: Convert each DataFrame in your dataset to a PyTorch tensor. This can be done in the `__getitem__` method of your dataset class.

```python
Copy code
import torch

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data  # data is a list of dictionaries like the one you provided

    def __getitem__(self, idx):
        item = self.data[idx]
        features_tensor = torch.tensor(item['feature'].values, dtype=torch.float32)
        labels = item['label']['los']['labels']
        masks = item['label']['los']['masks']
        return features_tensor, labels, masks

```

- **Padding and Positional Encoding**:
  - Since Transformer models require inputs of the same length, pad all sequences to the maximum length in your dataset. This is often done in a custom `collate_fn`.
  - Apply positional encoding to the padded sequences. This can be done either in the dataset class or in the `collate_fn`.

### Step 2: Forming Training Batches

- **Custom `collate_fn`**:
  - Implement a custom `collate_fn` to handle the padding of your sequences and the formation of input-target pairs.
  - For each sequence, you will need to create sub-sequences of increasing length (up to the current time bin where the mask is 1) and pair each with the corresponding LOS value.

```python
Copy code
def custom_collate_fn(batch):
    # batch is a list of tuples: (features_tensor, labels, masks)
    max_length = max([features.size(0) for features, _, _ in batch])
    batch_features = []
    batch_labels = []
    for features, labels, masks in batch:
        for t in range(len(masks)):
            if masks[t] == 1:
                # Pad features up to the current time bin
                padded_features = torch.nn.functional.pad(features[:t+1], (0, 0, 0, max_length - (t+1)))
                batch_features.append(padded_features)
                batch_labels.append(labels[t])
    # Stack all sequences and labels
    batch_features = torch.stack(batch_features)
    batch_labels = torch.tensor(batch_labels, dtype=torch.float32)
    # Apply positional encoding here if not done in __getitem__
    return batch_features, batch_labels

```

- **DataLoader**:
  - Use this `collate_fn` in your DataLoader.

```python
Copy code
dataset = MyDataset(data)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate_fn)

```

### Step 3: Adjusting Transformer Model for Regression Task

- **Model Architecture**:
  - Modify the output layer of the Transformer to produce a single continuous value for regression, instead of a sequence.
  - You might use only the encoder part of the Transformer, as you are not generating sequences but rather predicting a single value.

```python
Copy code
class TransformerRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_encoder = # Define the Transformer encoder
        self.regressor = torch.nn.Linear(transformer_output_size, 1)

    def forward(self, x):
        transformer_output = self.transformer_encoder(x)
        # You might want to use only the output of the last time step
        regression_output = self.regressor(transformer_output[:, -1, :])
        return regression_output

```

- **Loss Function**:
  - Use a suitable loss function for regression, such as Mean Squared Error (MSE).

```python
Copy code
criterion = torch.nn.MSELoss()

```

- **Training Loop**:
  - In your training loop, feed the input sequences into the model and calculate the loss against the LOS values.

### Summary

This approach involves significant custom handling of data, especially in the `collate_fn`, to create the appropriate input-target pairs for training the Transformer. The Transformer model itself needs to be adapted for a regression task, outputting a single continuous value rather than a sequence. This task is complex due to the nature of your data and the specific requirements of the regression task with a Transformer model.



------------------

**You:**

Replying to:For each sequence, you will need to create sub-sequences of increasing length (up to the current time bin where the mask is 1) and pair each with the corresponding LOS value.Is this a common practice for typical NLP tasks, say translation? Do I have to break a single sentence - translated sentence pair into smaller parts?

------------------

**ChatGPT:**

No, the method of breaking a single sequence into sub-sequences of increasing length and pairing each with a target value is not a common practice for typical NLP tasks like translation. The approach I described earlier is specific to your regression task, where the objective is to predict a single value (length of stay) based on all the data up to each time point.

In typical NLP tasks such as machine translation:


- **Sequence-to-Sequence Model**: The task is treated as a sequence-to-sequence problem, where an entire input sequence (e.g., a sentence in the source language) is mapped to an entire output sequence (e.g., the translated sentence in the target language).
- **Full Sequences**: Both the input sentence and the target sentence are used in their entirety for each training example. There's no need to break them into smaller parts.
- **Model Architecture**: Models like the Transformer use an encoder-decoder structure for these tasks. The encoder processes the entire input sequence, and the decoder generates the entire output sequence.
- **Training**: During training, the entire input sequence is passed to the encoder, and the decoder is trained to generate the output sequence, often using techniques like Teacher Forcing (where the true output sequence is used as part of the input to the decoder during training).
- **Attention Mechanisms**: Transformers and similar models use attention mechanisms to focus on different parts of the input sequence when generating each word of the output sequence. This is done automatically by the model and does not require manual segmentation of the input sequence.

### Example: Machine Translation

In a machine translation task using a Transformer model, you would:


- Train the model on pairs of complete sentences (one in the source language and one in the target language).
- Use the encoder to process the entire source sentence.
- Use the decoder to generate the entire target sentence, either in a training loop with Teacher Forcing or in an inference mode (e.g., using beam search).

There's no standard requirement to break down the sentence pairs into smaller parts in typical sequence-to-sequence models for tasks like translation.



------------------

**You:**

If I'm getting this right, teacher forcing means that I always input some prefix of the ground truth translated text into decoder, and let it predict the next token?

------------------

**ChatGPT:**

Yes, you've got the concept of Teacher Forcing correctly. In the context of sequence-to-sequence models, such as those used in machine translation, Teacher Forcing is a training strategy where the ground truth token from the target sequence (translated text) at a previous time step is provided as input to the decoder at the current time step. The decoder then predicts the next token in the sequence.

Here's a more detailed explanation:


- **Sequence Initialization**: The decoder is initially fed a start token to begin generating the output sequence.
- **Using Ground Truth as Input**: At each subsequent time step during training, instead of using the decoder's own previous predictions, the ground truth token from the target sequence is used as input to the decoder. This means the model is always conditioned on the correct sequence up to that point.
- **Predicting the Next Token**: The decoder then predicts the next token based on this input.
- **Advantages**: Teacher Forcing can speed up training and help the model converge more quickly, as it always guides the model with the correct sequence. It can be particularly helpful in the early stages of training when the model's predictions are likely to be inaccurate.
- **Drawbacks**: One potential drawback of Teacher Forcing is that it can create a discrepancy between training and inference (also known as "exposure bias"). During inference, the model won't have access to the ground truth and will have to rely on its own predictions. To mitigate this, techniques like scheduled sampling or using a mix of Teacher Forcing and free-running modes (feeding model's own predictions back into the decoder) can be employed.
- **Example**: In machine translation, if the target sequence (translated sentence) is "The quick brown fox", at the first time step after the start token, "The" is used as input to the decoder. At the next time step, "quick" is used, and so on. The decoder learns to predict the next word ("quick" after "The", "brown" after "The quick", etc.) during training.

Teacher Forcing is a widely used technique in various sequence generation tasks, including machine translation, text summarization, and more, wherever sequence-to-sequence models are applicable.



------------------

**You:**

Why this can be done in parallel instead of in sequence

------------------

**ChatGPT:**

In sequence-to-sequence models like those used in machine translation, Teacher Forcing allows for parallel processing of the target sequence during training, which is a significant advantage for computational efficiency. Here's why and how this parallelization works:


- **Entire Target Sequence Known**: During training, the entire target sequence is known in advance (the ground truth translation of the sentence). This allows each token in the sequence to be processed simultaneously rather than sequentially.
- **Masked Attention in Decoders**: Transformer-based models use masked self-attention mechanisms in the decoder. This ensures that the prediction for a particular token can only depend on previously known tokens in the sequence (including the current one). During training with Teacher Forcing, since all the correct tokens up to a certain point in the sequence are known, the model can compute the attention and subsequent layers for all these tokens in parallel.
- **Batch Processing**: The model can process an entire batch of target sequences simultaneously. Each token position across all sequences in the batch is processed in parallel, which is much more efficient than processing each token one after the other.
- **Efficient Backpropagation**: Since the computations for different tokens and different examples in a batch can be done in parallel, backpropagation (the process of updating model weights) is also parallelized, leading to faster training.

### Training vs. Inference

- **Training**: During training with Teacher Forcing, the model can look at the entire target sequence (up to the current token) at once. This parallel processing significantly speeds up training.
- **Inference**: During inference, the model does not have access to the future ground truth tokens. Therefore, it generates the sequence one token at a time, and each next token prediction depends on the previously generated tokens. This sequential process cannot be parallelized in the same way as training.

In summary, Teacher Forcing during training leverages the known target sequence to enable parallel processing, significantly speeding up the training of sequence-to-sequence models. However, this parallelization is specific to the training phase and does not apply to inference, where the model generates the sequence token by token.



------------------

**You:**

So transformer translation model will see the entire input first, encode it, and with reference to the encoded feature vector, it'll be fed with a start token in the beginning, or the previous results it has output, to predict the next token? Instead of see a word, output a word?

------------------

**ChatGPT:**

Yes, your understanding is correct. In a Transformer model for translation, the process involves seeing the entire input sequence first, encoding it, and then generating the output sequence either token-by-token (during inference) or with the entire sequence available (during training with Teacher Forcing). Here's a breakdown of the process:

### Encoding the Input Sequence

- **Entire Input Sequence**: The Transformer's encoder processes the entire input sequence (e.g., a sentence in the source language) at once. Each word (or subword/token) in the input sequence is converted into a vector using embeddings.
- **Contextual Encoding**: The encoder uses self-attention mechanisms to understand the context around each word and generates a set of feature vectors that represent the entire input sequence in context.

### Decoding the Output Sequence

- **Start Token**: Decoding begins with a start token. The Transformer's decoder then generates the output sequence one token at a time.
- **Reference to Encoded Input**: For each token it generates, the decoder uses the encoded input (from the encoder) and all previously generated tokens to predict the next token. This process involves a combination of self-attention (to focus on previously generated tokens) and encoder-decoder attention (to focus on relevant parts of the input sequence).

### During Training (Teacher Forcing)

- The decoder is provided with the correct target sequence up to the current token. This allows the model to be trained on predicting the next token given the correct sequence so far. All positions in the target sequence can be processed in parallel because the model knows the correct token for each position up to that point.

### During Inference

- The model does not have access to the target sequence. It starts with a start token and generates each subsequent token based on the previously generated tokens and the encoded input. This process is sequential, with each new token depending on the ones generated before it.

### Example: Machine Translation

- **Training**: The input might be a sentence in English, and the entire corresponding German sentence is used for training with Teacher Forcing. The model learns to predict each word of the German sentence given the English sentence and the correct German words leading up to that point.
- **Inference**: The model sees the entire English sentence, encodes it, and then begins to generate the German translation one word at a time, using the previously generated words and the encoded input to predict the next word.

In summary, Transformer models in translation tasks process the entire input sequence to understand its context and then generate the output sequence, leveraging the context from the input and the partially generated output. The method of generation differs between training (parallel, with Teacher Forcing) and inference (sequential).



------------------

