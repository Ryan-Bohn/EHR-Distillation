# Previous works on MIMIC 10.11~10.17

## MIMIC-IV

### An Extensive Data Processing Pipeline for MIMIC-IV (2022) ([link](https://paperswithcode.com/paper/an-extensive-data-processing-pipeline-for))

> An increasing amount of research is being devoted to applying machine learning methods to electronic health record (EHR) data for various clinical purposes. This growing area of research has exposed the challenges of the accessibility of EHRs. MIMIC is a popular, public, and free EHR dataset in a raw format that has been used in numerous studies. The absence of standardized pre-processing steps can be, however, a significant barrier to the wider adoption of this rare resource. Additionally, this absence can reduce the reproducibility of the developed tools and limit the ability to compare the results among similar studies. **In this work, we provide a greatly customizable pipeline to extract, clean, and pre-process the data available in the fourth version of the MIMIC dataset (MIMIC-IV)**. The pipeline also presents an end-to-end wizard-like package supporting predictive model creations and evaluations. The pipeline covers a range of clinical prediction tasks which can be broadly classified into four categories - readmission, length of stay, mortality, and phenotype prediction. The tool is publicly available at https://github.com/healthylaife/MIMIC-IV-Data-Pipeline.

#### Highlights

- Preprocessing MIMIC-IV, focusing on both feature selection and data cleaning
- Has some integrated introduction to previous MIMIC-III benchmarks

### Integrated multimodal artificial intelligence framework for healthcare applications (2022) ([link](https://www.nature.com/articles/s41746-022-00689-4))

> Artificial intelligence (AI) systems hold great promise to improve healthcare over the next decades. Specifically, AI systems leveraging multiple data sources and input modalities are poised to become a viable method to deliver more accurate results and deployable pipelines across a wide range of applications. In this work, **we propose and evaluate a unified Holistic AI in Medicine (HAIM) framework to facilitate the generation and testing of AI systems that leverage multimodal inputs. Our approach uses generalizable data pre-processing and machine learning modeling stages that can be readily adapted for research and deployment in healthcare environments. We evaluate our HAIM framework by training and characterizing 14,324 independent models based on HAIM-MIMIC-MM, a multimodal clinical database** (*N* = 34,537 samples) containing 7279 unique hospitalizations and 6485 patients, spanning all possible input combinations of 4 data modalities (i.e., tabular, time-series, text, and images), 11 unique data sources and 12 predictive tasks. We show that this framework can consistently and robustly produce models that outperform similar single-source approaches across various healthcare demonstrations (by 6–33%), including 10 distinct chest pathology diagnoses, along with length-of-stay and 48 h mortality predictions. We also quantify the contribution of each modality and data source using Shapley values, which demonstrates the heterogeneity in data modality importance and the necessity of multimodal inputs across different healthcare-relevant tasks. The generalizable properties and flexibility of our Holistic AI in Medicine (HAIM) framework could offer a promising pathway for future multimodal predictive systems in clinical and operational healthcare settings.

The latest research adapted this as baseline: MultiModN—Multimodal, Multi-Task, Interpretable Modular Networks (2023) ([link](https://paperswithcode.com/paper/multimodn-multimodal-multi-task-interpretable))

#### Hightlights

- Construction of a large MM dataset by joining MIMIC-IV and MIMIC-CXR-JPG, store each patient's all related data in a single pickle file

## MIMIC-III

### MIMIC-III Benchmarks (2017) ([link](https://github.com/YerevaNN/mimic3-benchmarks))

> Python suite to construct benchmark machine learning datasets from the MIMIC-III clinical database. Currently, the benchmark datasets cover four key inpatient clinical prediction tasks that map onto core machine learning problems: prediction of mortality from early admission data (**classification**), real-time detection of decompensation (**time series classification**), forecasting length of stay (**regression**), and phenotype classification (**multilabel sequence classification**).

Also there's a corresponding paper: 

#### Highlights

- Benchmarks for 4 well studied tasks
- Along with baseline models (both linear and neural)

### Early hospital mortality prediction using vital signals (2018) ([link](An Extensive Data Processing Pipeline for MIMIC-IV))

> Early hospital mortality prediction is critical as intensivists strive to make efficient medical decisions about the severely ill patients staying in intensive care units. As a result, various methods have been developed to address this problem based on clinical records. However, some of the laboratory test results are time-consuming and need to be processed. **In this paper, we propose a novel method to predict mortality using features extracted from the heart signals of patients within the first hour of ICU admission**. In order to predict the risk, quantitative features have been computed based on the heart rate signals of ICU patients. Each signal is described in terms of 12 statistical and signal-based features. The extracted features are fed into eight classifiers: decision tree, linear discriminant, logistic regression, support vector machine (SVM), random forest, boosted trees, Gaussian SVM, and K-nearest neighborhood (K-NN). To derive insight into the performance of the proposed method, several experiments have been conducted using the well-known clinical dataset named Medical Information Mart for Intensive Care III (MIMIC-III). The experimental results demonstrate the capability of the proposed method in terms of precision, recall, F1-score, and area under the receiver operating characteristic curve (AUC). The decision tree classifier satisfies both accuracy and interpretability better than the other classifiers, producing an F1-score and AUC equal to 0.91 and 0.93, respectively. It indicates that heart rate signals can be used for predicting mortality in patients in the ICU, achieving a comparable performance with existing predictions that rely on high dimensional features from clinical records which need to be processed and may contain missing information.

#### Highlights

- SOTA on mortality prediction on MIMIC-III ([link](https://paperswithcode.com/sota/mortality-prediction-on-mimic-iii))
- 9 types of non-NN classifiers: Random forest, Hard Gaussian, SVM, Hard Decision tree, Easy Boosted trees, Hard K-NN, Hard Logistic regression, Easy Linear discriminant, Easy Linear SVM

## Summary

### Plan for current stage

- Choose a well studied downstream task (mortality prediction, length-of-stay prediction), select features, form a dataset by joining tables and filtering (refer to [MIMIC docs](https://mimic.mit.edu/docs/iv/))
- Build an NN for it (better easy to perform DD on, e.g. temporal convolutional network)
- Get a distilled dataset
- Evaluate the DD on traditional classifiers as well as NN

### Futurework

- Try different DD strategies
- Explore how to perform DD with traditional classifiers