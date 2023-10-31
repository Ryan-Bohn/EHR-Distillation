# Experiment Log 10.11~

## 1 Overall plan

### 1.1 Current stage

- Choose a well studied downstream task (mortality prediction, length-of-stay prediction), select features, form a sub-dataset by joining tables and filtering (refer to [MIMIC docs](https://mimic.mit.edu/docs/iv/))
- Build an NN for it (better easy to perform DD on, e.g. temporal convolutional network)
- Get a distilled dataset **that has the same structure as the original selected sub-dataset**
- Evaluate the DD on traditional classifiers as well as NN on the same objective

### 1.2 Future work

- Try different DD strategies
- Explore how to perform DD with traditional classifiers

## 2 Preliminary verification

### 2.1 Problem setup

- **Objective**: In-hospital mortality prediction based on the first 48hr of an ICU stay

- **Data**: ~20 selected features (variables), all in tabular format, from MIMIC (III or IV)

- **Motivation**:

  - Mainly inspired by the foundamental benchmark study on MIMIC-III: [H. Harutyunyan et al. - Multitask learning and benchmarking with clinical time series data (2019)](https://www.nature.com/articles/s41597-019-0103-9)

  - Mortality is a primary outcome of interest in acute care: ICU mortality rates are the highest among hospital units (10% to 29% depending on age and illness), and early detection of at-risk patients is key to improving outcomes

  - The study selected out only **17** variables for all the 4 tasks, including mortality prediction, which is a relatively simple selected sub-dataset

    ![image-20231018005720825](assets/image-20231018005720825.png)

  - For MIMIC-III, H. Harutyunyan et al. provided the code base; doing the similar thing on MIMIC-IV should not be too hard

### 2.2 Data processing

Useing the exact same pipeline of H. Harutyunyan et al., we have:

- **Size**

  - ~18k training subjects / stays

  - ~3k evaluating subjects / stays

- **Format**

  - Episodes (ICU stays) of **time series** of 48hr events, without a fixed sample rate (new timestamp is added each time a new lab/chart event happens)

    ![image-20231018034236127](assets/image-20231018034236127.png)

  - Episode-level information (patient age, gender, ethnicity, height, weight) and outcomes (mortality, length of stay, diagnoses) are also available

#### Question now

How to approach the time series?

#### My intuitions

1. Resample: just like in the original paper, **resample** the timeseries to a fixed sample rate, so that the length is unified
2. Recover missing variables: recover by **imputation**
3. **Treat them as if they are images**: for each column (representing a type of variable), there are constraints on the values, say pH is within 0~14, just like pixels are within 0~255
4. Distill

### 2.3 Model #TODO

2 models worth trying:

- Adapting models from the vanilla Data Distillation paper
- Adapting models (random NN) from DD by Matching Features

### 2.4 Experiments #TODO

