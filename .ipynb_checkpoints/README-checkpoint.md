# GenomicLLM
Code for paper **Exploring Genomic Large Language Models: Bridging the Gap between Natural Language and Gene Sequences**.


# Requirements
```
matplotlib
numpy
pandas
biopython==1.79
rouge==1.0.1
tokenizers==0.11.6
torch==2.0.1+cu117
torchaudio==2.0.2+cu117
torchvision==0.15.2+cu117
transformers==4.18.0
```


## Materials
 Download data sets and the trained models from https://zenodo.org/records/10695802

<!-- #region -->
## Training：

Set all the parameters in configurator.py before training:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```
<!-- #endregion -->

## Test：

<!-- #region -->
Test the overall test set:
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --custom=False --BeckyGRCh38Data_num_samples=-1 --GUEData_num_samples=-1 --BeckyData_num_samples=-1 --HyenaData_num_samples=-1
```

Test part of the test set:
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --custom=False --BeckyGRCh38Data_num_samples=-1 --BeckyGRCh38_data_name=['enhancer', 'splice site']
```

Test a custom data set:
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --custom=True --file_name='./data/Genomic/custom_data/human_enhancers_cohn_test.txt'
```
<!-- #endregion -->

```python

```
