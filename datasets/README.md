# QA Datasets

Track 2 of the DADC shared task requires you to submit training data (in SQuAD v1.1 JSON format, see https://huggingface.co/datasets/adversarial_qa#dataset-structure). These examples can be selected from existing datasets, expert-annotated, crowdsourced, or synthetically-generated.

We provide a few dataset resources to help you with example selection below. Please make sure to cite the original resource creators if you make use of any of the below-mentioned resources.



## SQuAD v1.1

#### Overview
The Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.

For further details see: https://rajpurkar.github.io/SQuAD-explorer/.


#### Using the Dataset
To use this dataset with the 🤗 HuggingFace datasets or transformers libraries:

```
from datasets import load_dataset
dataset = load_dataset("squad")
```

#### Citation Information
```
@article{2016arXiv160605250R,
       author = {{Rajpurkar}, Pranav and {Zhang}, Jian and {Lopyrev},
                 Konstantin and {Liang}, Percy},
        title = "{SQuAD: 100,000+ Questions for Machine Comprehension of Text}",
      journal = {arXiv e-prints},
         year = 2016,
          eid = {arXiv:1606.05250},
        pages = {arXiv:1606.05250},
archivePrefix = {arXiv},
       eprint = {1606.05250},
}
```



## AdversarialQA

#### Overview
AdversarialQA consists of three Reading Comprehension datasets constructed using an adversarial model-in-the-loop.

Three different models aree used; BiDAF ([Seo et al., 2016](https://arxiv.org/abs/1611.01603)), BERT-Large ([Devlin et al., 2018](https://arxiv.org/abs/1810.04805)), and RoBERTa-Large ([Liu et al., 2019](https://arxiv.org/abs/1907.11692)) in the annotation loop and construct three datasets; D(BiDAF), D(BERT), and D(RoBERTa), each with 10,000 training examples, 1,000 validation, and 1,000 test examples.

The adversarial human annotation paradigm ensures that these datasets consist of questions that current state-of-the-art models (at least the ones used as adversaries in the annotation loop) find challenging.

For further details see: https://adversarialqa.github.io/.


#### Using the Dataset
To use this dataset with the 🤗 HuggingFace datasets or transformers libraries:

```
from datasets import load_dataset
dataset = load_dataset("adversarial_qa")
```

#### Citation Information
```
@article{bartolo2020beat,
    author = {Bartolo, Max and Roberts, Alastair and Welbl, Johannes and Riedel, Sebastian and Stenetorp, Pontus},
    title = {Beat the AI: Investigating Adversarial Human Annotation for Reading Comprehension},
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {8},
    number = {},
    pages = {662-678},
    year = {2020},
    doi = {10.1162/tacl\_a\_00338},
    URL = { https://doi.org/10.1162/tacl_a_00338 },
    eprint = { https://doi.org/10.1162/tacl_a_00338 },
    abstract = { Innovations in annotation methodology have been a catalyst for Reading Comprehension (RC) datasets and models. One recent trend to challenge current RC models is to involve a model in the annotation process: Humans create questions adversarially, such that the model fails to answer them correctly. In this work we investigate this annotation methodology and apply it in three different settings, collecting a total of 36,000 samples with progressively stronger models in the annotation loop. This allows us to explore questions such as the reproducibility of the adversarial effect, transfer from data collected with varying model-in-the-loop strengths, and generalization to data collected without a model. We find that training on adversarially collected samples leads to strong generalization to non-adversarially collected datasets, yet with progressive performance deterioration with increasingly stronger models-in-the-loop. Furthermore, we find that stronger models can still learn from datasets collected with substantially weaker models-in-the-loop. When trained on data collected with a BiDAF model in the loop, RoBERTa achieves 39.9F1 on questions that it cannot answer when trained on SQuAD—only marginally lower than when trained on data collected using RoBERTa itself (41.0F1). }
}
```



## MRQA

#### Overview
The MRQA 2019 Shared Task focuses on generalization in question answering. An effective question answering system should do more than merely interpolate from the training set to answer test examples drawn from the same distribution: it should also be able to extrapolate to out-of-distribution examples — a significantly harder challenge.

The dataset is a collection of 18 existing QA dataset (carefully selected subset of them) and converted to the same format (SQuAD format). Among these 18 datasets, six datasets were made available for training, six datasets were made available for development, and the final six for testing. The dataset is released as part of the MRQA 2019 Shared Task.

For further details see: https://github.com/mrqa/MRQA-Shared-Task-2019.


#### Using the Dataset
To use this dataset with the 🤗 HuggingFace datasets or transformers libraries:

```
from datasets import load_dataset
dataset = load_dataset("mrqa")
```

#### Citation Information
```
@inproceedings{fisch2019mrqa,
    title={{MRQA} 2019 Shared Task: Evaluating Generalization in Reading Comprehension},
    author={Adam Fisch and Alon Talmor and Robin Jia and Minjoon Seo and Eunsol Choi and Danqi Chen},
    booktitle={Proceedings of 2nd Machine Reading for Reading Comprehension (MRQA) Workshop at EMNLP},
    year={2019},
}
```



## SynQA

#### Overview
SynQA is a Reading Comprehension dataset created in the work "Improving Question Answering Model Robustness with Synthetic Adversarial Data Generation" (https://aclanthology.org/2021.emnlp-main.696/). It consists of 314,811 synthetically generated questions on the passages in the SQuAD v1.1 (https://arxiv.org/abs/1606.05250) training set.

This work uses a synthetic adversarial data generation to make QA models more robust to human adversaries. The authors develop a data generation pipeline that selects source passages, identifies candidate answers, generates questions, then finally filters or re-labels them to improve quality. Using this approach, they amplify a smaller human-written adversarial dataset to a much larger set of synthetic question-answer pairs. Incorporating the synthetic data improves on the previous SOTA on AdversarialQA (https://adversarialqa.github.io/) by 3.7F1 and improves model generalisation on nine of the twelve MRQA datasets. Models are also considerably more robust to new human-written adversarial examples in a human-in-the-loop evaluation setting: crowdworkers can fool the best model only 8.8% of the time on average, compared to 17.6% for a model trained without synthetic data.

For further details see: https://github.com/maxbartolo/improving-qa-model-robustness.


#### Using the Dataset
To use this dataset with the 🤗 HuggingFace datasets or transformers libraries:

```
from datasets import load_dataset
dataset = load_dataset("mbartolo/synQA")
```

#### Citation Information
```
@inproceedings{bartolo-etal-2021-improving,
    title = "Improving Question Answering Model Robustness with Synthetic Adversarial Data Generation",
    author = "Bartolo, Max  and
      Thrush, Tristan  and
      Jia, Robin  and
      Riedel, Sebastian  and
      Stenetorp, Pontus  and
      Kiela, Douwe",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.696",
    doi = "10.18653/v1/2021.emnlp-main.696",
    pages = "8830--8848",
    abstract = "Despite recent progress, state-of-the-art question answering models remain vulnerable to a variety of adversarial attacks. While dynamic adversarial data collection, in which a human annotator tries to write examples that fool a model-in-the-loop, can improve model robustness, this process is expensive which limits the scale of the collected data. In this work, we are the first to use synthetic adversarial data generation to make question answering models more robust to human adversaries. We develop a data generation pipeline that selects source passages, identifies candidate answers, generates questions, then finally filters or re-labels them to improve quality. Using this approach, we amplify a smaller human-written adversarial dataset to a much larger set of synthetic question-answer pairs. By incorporating our synthetic data, we improve the state-of-the-art on the AdversarialQA dataset by 3.7F1 and improve model generalisation on nine of the twelve MRQA datasets. We further conduct a novel human-in-the-loop evaluation and show that our models are considerably more robust to new human-written adversarial examples: crowdworkers can fool our model only 8.8{\%} of the time on average, compared to 17.6{\%} for a model trained without synthetic data.",
}
```


