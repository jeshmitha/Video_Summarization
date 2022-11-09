# Video_Summarization 
## Training

For training, evaluating and testing the model for a particular dataset(SumMe, TVSum) and setting(Canonical, Augmented and Transfer) using single split, run:<br/>
Example: <br/>
```
python main_model.py  --setting "tvsum_canon" --split_index 0
```
Syntax: <br/>
```
python main_model.py  --setting ["tvsum_canon"|"tvsum_aug"|"tvsum_trans"|"summe_canon"|"summe_aug"|"summe_trans"] --split_index [0|1|2|3|4]
```
Alternatively, to train the model on both datasets for all settings with all the 5 splits, use the 'run_splits.sh' script according to the following:
```
chmod +x run_splits.sh
sh run_splits.sh
```
## Train & Test splits

The JSON files under the "data/splits" directory contain the split information(80% of videos for training and 20% videos for testing) for each setting and dataset that was used in our experiments. 

## Evaluation
After each training epoch the algorithm performs an evaluation step, and uses the trained model to compute and store the following values:

* The importance scores for the frames of each test video. 
* These scores are then used by the provided evaluation scripts to assess the overall performance of the model (in F-Score, Kendall, Spearman scores with both Ground Truth Scores/Summaries and User Annotations/Summaries).
* All the defined losses.

## Setup before Training & Testing:

### Installations:
* python=3.9.7
* numpy=1.21.2
* pandas=1.3.4
* pytorch=1.10.1
* h5py=3.7.0
* hdf5storage=0.1.16
* ortools=9.2.9972
* scikit-learn=1.0.1
* scipy=1.7.3
* matplotlib=3.5.0
* json: 2.0.9

### Data:
Upload the h5 files into their respective folders (SumMe, TVSum, OVP, YouTube) present under the 'data' folder. 
The h5 files for TVSum, SumMe, OVP and YouTube can be downloaded [here](https://kingston.box.com/shared/static/zefb0i17mx3uvspgx70hovt9rr2qnv7y.zip).
