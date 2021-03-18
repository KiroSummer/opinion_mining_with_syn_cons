# opinion_mining_with_syn_cons
This repositry contains our code, configurations, and model for our work on "A Unified Span-Based Approach for Opinion Mining with Syntactic Constituents", which is published on NAACL-2021.
The src directory contains our code and the exp-4.1-baseline contains our experiment for "Baseline+BERT" (data0, the first data of the five fold cross-validation).

![model](https://github.com/KiroSummer/opinion_mining_with_syn_cons/blob/main/figures/model.jpg)

## Environment
Python3, Pytorch, Transformers 2.1.1 (for BERT)

### Training
Please reset and check the files in the train.sh and config.json when you want to run the code.

```
sh train.sh GPU\_ID
```

### Test
To test the performance of the trained model, you should run the following script.

```
sh predict.sh GPU\_ID
```
We release the sample model of the "exp-4.1-baseline" on the Google Drive, [url](https://drive.google.com/file/d/17u8ofyaBThb66qYPZe-60A2lyEnWCNil/view?usp=sharing).
Important, use the offline evaluation script to eval the output file.
