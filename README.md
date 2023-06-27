# LLM-QA4RE

Data and code for ACL 2023 Findings: [Aligning Instruction Tasks Unlocks Large Language Models as Zero-Shot Relation Extractors](https://arxiv.org/pdf/2305.11159.pdf).

We present LLM-QA4RE, which aligns underrepresented tasks in the instruction-tuning dataset (relation extraction) to a common task (question answering) to unlock instruction-tuned LLMs' abilities on relation extraction. 

QA4RE achieves significant and consistent performance gains over 6 LLMs across 4 datasets. In addition, it shows strong transferability to model sizes from 175B (GPT-3.5 series) to even 80M (FLAN-T5 Small).

<!-- ![QA4RE-main figure.jpeg](https://s2.loli.net/2023/05/15/Lk1saYNjni3yqWP.jpg) -->

<a href="https://sm.ms/image/Lk1saYNjni3yqWP" target="_blank"><img src="https://s2.loli.net/2023/05/15/Lk1saYNjni3yqWP.jpg" width="75%" height="75%" ></a>


#### TODO
- [ ] Organize and release CODE for FLAN T5 Series
- [x] Release output results of GPT-3.5 Series
- [x] Organize and release Code for GPT-3.5 Series

## Installation

Run the following commands to create a conda environment with the required packages.

```shell
conda create -n QA4IE python=3.9 pip
conda activate QA4IE
pip install -r requirements.txt
# same env with few-shot-bioIE
```


## Data and Launch
Download data and subsets via [Google Drive](https://drive.google.com/file/d/1tAB7V4_bV76FiPGMsoOWWJnZPtpePtwe/view?usp=sharing)
<!-- are prepared in `./data` dir -->
Unzip directly in `./` and then the root folder should organize like this:
```
.
├───  data
│    ├───  RETACRED
│    ├───  TACRED
│    ├───  TACREV
│    ├───  semeval
├───  outputs
│    ├───  RETACRED
│    ├───  TACRED
│    ├───  TACREV
│    ├───  semeval
├───  projs
│    ├───  QA4RE
│    ├───  vanillaRE
│    ├───  README.md
│    ├───  re_templates.py
│    └───  re_utils.py
├───  utils
│   ...
```

For running, please refer to the [README](./projs/README.md) in `./projs` dir.

## Results
#### QA4RE works on GPT-3.5 Series and FLAN-T5 Series, 6 LLMs in total
<!-- ![QA4RE-table 1.jpeg](https://s2.loli.net/2023/05/15/is8XGo71lODm3Bq.jpg) -->
<a href="https://sm.ms/image/is8XGo71lODm3Bq" target="_blank"><img src="https://s2.loli.net/2023/05/15/is8XGo71lODm3Bq.jpg" width="75%" height="75%" ></a>


#### QA4RE works on smaller instruction-tuned models.
<!-- ![QA4RE-table 8.jpeg](https://s2.loli.net/2023/05/15/IEUrGuBWn9FNmb8.jpg) -->
<a href="https://sm.ms/image/IEUrGuBWn9FNmb8" target="_blank"><img src="https://s2.loli.net/2023/05/15/IEUrGuBWn9FNmb8.jpg" width="75%" height="75%"></a>

## Cite

If you find our paper, code, or data  helpful, please cite it.

```
@inproceedings{Zhang2023LLM-QA4RE,
  title={Aligning Instruction Tasks Unlocks Large Language Models as Zero-Shot Relation Extractors},
  author={Kai Zhang, Bernal Jiménez Gutiérrez, Yu Su},
  booktitle={Findings of ACL 2023},
  year={2023}
}
```

## Question

If you have any questions, please feel free to contact `drogozhang[AT]gmail[DOT]com` or open an issue so we can help you better and quicker :)
