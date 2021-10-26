# SeerForcing-NMT
Source code for the ACL 2021 long paper Guiding Teacher Forcing with Seer Forcing for Neural Machine Translation.

## Related code

Implemented based on [Fairseq-py](https://github.com/pytorch/fairseq), an open-source toolkit released by Facebook which was implemented strictly referring to [Vaswani et al. (2017)](https://arxiv.org/pdf/1706.03762.pdf).

## Requirements
This system has been tested in the following environment.
+ OS: Ubuntu 16.04.1 LTS 64 bits
+ Python version \>=3.6
+ Pytorch version \>=1.4

## Get started
- Build
```
pip install --editable .
```

- Preprocess the training data. Pretrain the general-domain model with the general-domain data. Read [here](https://fairseq.readthedocs.io/en/latest/getting_started.html#training-a-new-model) for more instructions.

- Train the model

```
# save dir
save=
CUDA_VISIBLE_DEVICES=0,1,2,3  python3  train.py --ddp-backend=no_c10d  data-bin/{training-data}\
    --arch transformer_wmt_en_de  --fp16 --alpha 0.25 \
      --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
        --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
          --lr 0.0007 --min-lr 1e-09 --dropout 0.3 --seer_dropout 0.2 \
           --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0\
        --max-tokens  4096  --save-dir checkpoints/$save  --save-interval-updates 1000 \
        --update-freq 2 --no-progress-bar --log-format json --log-interval 25  
```
Add `--share-decoder-input-output-embed` for the WMT17 Zh-En.
Add `--share-all-embeddings`  for the WMT16 En-De and En-Ro.
Set `--dropout` 0.3 for NIST Zh-En and 0.1 for other datasets.
`--seer_dropout` is the dropout probablity for the seer decoder. 0.2 for the NIST Zh-En and WMT16 En-Ro, 0.1 for the WMT16 En-De and WMT17 Zh-En.
`--alpha` is the hyperparameter in the total loss function. 0.25 for the NIST Zh-En and 0.5 for other datasets.

- Generate the translation 

```
python generate.py {training-data} --path $MODEL \
    --gen-subset test --beam 4 --batch-size 128 \
    --remove-bpe --lenpen {float} \
```

The length penalty is set as 1.4 for the WMT17 Zh-En experiments and 0.6 for other datasets.


## Citation
```
@inproceedings{FengGGYS20,
  author    = {Yang Feng and
               Shuhao Gu and
               Dengji Guo and
               Zhengxin Yang and
               Chenze Shao},
  title     = {Guiding Teacher Forcing with Seer Forcing for Neural Machine Translation},
  booktitle = {Proceedings of the 59th Annual Meeting of the Association for Computational
               Linguistics and the 11th International Joint Conference on Natural
               Language Processing, {ACL/IJCNLP} 2021, (Volume 1: Long Papers), Virtual
               Event, August 1-6, 2021},
  pages     = {2862--2872},
  year      = {2021},
  url       = {https://doi.org/10.18653/v1/2021.acl-long.223},
}
```























