# opinion_mining_with_syn_cons
This repositry contains our code, configurations, and model for our work on "A Unified Span-Based Approach for Opinion Mining with Syntactic Constituents", which is published on NAACL-2021.

## Environment
Python3, Pytorch, Transformers 2.1.1 (for BERT)

### Training
Please reset and check the files in the train.sh and config.json when you want to run the code.

```
sh train.sh GPU\_ID
```

### Test

```
sh predict.sh GPU\_ID
```
