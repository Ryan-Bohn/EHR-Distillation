Construct MIMIC-III Benchmarks
=========================

## Data Preparation

1. The following command takes MIMIC-III CSVs, generates one directory per `SUBJECT_ID` and writes ICU stay information to `data/{SUBJECT_ID}/stays.csv`, diagnoses to `data/{SUBJECT_ID}/diagnoses.csv`, and events to `data/{SUBJECT_ID}/events.csv`. This step might take around an hour.
```bash
       python -m mimic3benchmark.scripts.extract_subjects {PATH TO MIMIC-III CSVs} data/root/
```

2. The following command attempts to fix some issues (ICU stay ID is missing) and removes the events that have missing information. About 80% of events remain after removing all suspicious rows (more information can be found in [`mimic3benchmark/scripts/more_on_validating_events.md`](mimic3benchmark/scripts/more_on_validating_events.md)).
```bash
       python -m mimic3benchmark.scripts.validate_events data/root/
```

3. The next command breaks up per-subject data into separate episodes (pertaining to ICU stays). Time series of events are stored in ```{SUBJECT_ID}/episode{#}_timeseries.csv``` (where # counts distinct episodes) while episode-level information (patient age, gender, ethnicity, height, weight) and outcomes (mortality, length of stay, diagnoses) are stores in ```{SUBJECT_ID}/episode{#}.csv```. This script requires two files, one that maps event ITEMIDs to clinical variables and another that defines valid ranges for clinical variables (for detecting outliers, etc.). **Outlier detection is disabled in the current version**.
```bash
       python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root/
```

4. The next command splits the whole dataset into training and testing sets. Note that the train/test split is the same of all tasks.
```bash
       python -m mimic3benchmark.scripts.split_train_and_test data/root/
```

5. Now we can creat the multitask dataset, which which will later be used in our models. These commands are independent, if you are going to work only on one benchmark task, you can run only the corresponding command.
```bash
       python -m mimic3benchmark.scripts.create_multitask data/root/ data/multitask/
```


## Running experiments on Mimic-III Benchmark

First run this preprocessing script to preprocess the data into uniform 48-hour tensors, replacing the multitask dataset directory with your directory:

```shell
cd src
python preprocess.py -n mimic3benchmark -d {PATH TO MULTITASK DATASET} --sr 1.0
```
