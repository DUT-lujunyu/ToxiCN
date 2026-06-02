# Facilitating Fine-grained Detection of Chinese Toxic Language: Hierarchical Taxonomy, Resources, and Benchmark


🎉 **2026.5** Recently, we organized ToxiCN and a selection of its representative citing papers. If you are interested, please see [ToxiCN-family](https://dut-lujunyu.github.io/ToxiCN-Family/). We hope this collection can offer a clearer view of Chinese toxicity research.

_This repository will be updated from time to time with the latest research progress of the DUTIR team in Chinese hate speech detection. Welcome to follow! If you have any questions, feel free to contact us at: [dut_ljy@foxmail.com](dut_ljy@foxmail.com)_.

## Contents
- [News](#news)
- [Monitor Toxic Frame](#monitor-toxic-frame)
- [ToxiCN](#toxicn)
- [Insult Lexicon](#insult-lexicon)
- [Baseline](#baseline)
- [Follow-up Research](#follow-up-research)
- [Poster](#poster)
- [Cite](#cite)


---

**The paper has been accepted in ACL 2023 (main conference, long paper).** [Paper](https://aclanthology.org/2023.acl-long.898/)

☠️ ***Warning: The samples presented in this paper may be considered offensive or vulgar.***

## Monitor Toxic Frame

We introduce a hierarchical taxonomy, **Monitor Toxic Frame**. Based on this taxonomy, posts are progressively divided into multiple granularities: **(I) Whether Toxic**, **(II) Toxic Type** (general offensive language or hate speech), **(III) Targeted Group**, and **(IV) Expression Category** (explicitness, implicitness, or reporting).

## ToxiCN

We conduct fine-grained annotation on posts crawled from _Zhihu_ and _Tieba_, including both direct and indirect toxic samples. The resulting **ToxiCN** dataset contains 12k comments covering **Sexism**, **Racism**, **Regional Bias**, **Anti-LGBTQ**, and **Others**. The dataset is released as ***ToxiCN_1.0.csv***. Below we briefly describe each fine-grained label.

| Label           | Description |
|-----------------|-------------|
| toxic           | Identify whether a comment is toxic (1) or non-toxic (0). |
| toxic_type      | non-toxic: 0, general offensive language: 1, hate speech: 2 |
| expression      | non-hate: 0, explicit hate speech: 1, implicit hate speech: 2, reporting: 3 |
| target (a list) | LGBTQ: Index 0, Region: Index 1, Sexism: Index 2, Racism: Index 3, Others: Index 4, non-hate: Index 5 |

## Insult Lexicon

See: https://github.com/DUT-lujunyu/ToxiCN/tree/main/ToxiCN_ex/ToxiCN/lexicon

## Baseline

We present a migratable baseline of **Toxic Knowledge Enhancement (TKE)** for enriching text representations. The implementation is provided in **_modeling_bert.py_**, based on **transformers 3.1.0**.

## Licenses and Ethics Statement

This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (**CC BY-NC-ND 4.0**).

The opinions and findings contained in the dataset samples should not be interpreted as representing the views expressed or implied by the authors. We acknowledge the risk of malicious actors attempting to reverse-engineer comments. **We sincerely hope that users will employ the dataset responsibly and appropriately, avoiding misuse or abuse.** We believe the benefits of our proposed resources outweigh the associated risks. **All resources are intended solely for scientific research and are prohibited from commercial use.**

## Follow-up Research

_Here we list some of our team's work on toxic language detection. Feel free to follow!_

* **Towards Comprehensive Detection of Chinese Harmful Meme** (NeurIPS2024). In this paper, we present the definition of  **Chinese Harmful Meme Detection** to align with the Chinese online environment. and present **ToxiCN MM**, the first Chinese harmful meme dataset. [paper](https://arxiv.org/abs/2410.02378) [repo](https://github.com/DUT-lujunyu/ToxiCN_MM)
* **PclGPT: A Large Language Model for Patronizing and Condescending Language Detection** (EMNLP2024 findings). In this paper, we focus on a specific type of implicit toxic bias, **patronizing and condescending language (PCL)**, and leverage LLMs to detect it. [paper](https://arxiv.org/abs/2410.00361) [repo](https://github.com/dut-laowang/emnlp24-PclGPT)
* **Towards Patronizing and Condescending Language in Chinese Videos: A Multimodal Dataset and Detector** (ICASSP2025). In this paper, we introduce the **PCL MM** dataset, the first Chinese multimodal dataset for PCL, and propose the **MultiPCL** framework for detection. [paper](https://arxiv.org/abs/2409.05005) [repo](https://github.com/dut-laowang/PCLMM)
___ 


## Poster
![CCAC_2](https://github.com/DUT-lujunyu/ToxiCN/assets/53985277/8e26c649-0952-4d04-a562-b971f441df07)



## Cite
If you want to use the resources, please cite the following paper:
~~~
@inproceedings{lu-etal-2023-facilitating,
    title = "Facilitating Fine-grained Detection of {C}hinese Toxic Language: Hierarchical Taxonomy, Resources, and Benchmarks",
    author = "Lu, Junyu  and
      Xu, Bo  and
      Zhang, Xiaokun  and
      Min, Changrong  and
      Yang, Liang  and
      Lin, Hongfei",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.898",
    doi = "10.18653/v1/2023.acl-long.898",
    pages = "16235--16250",
}
~~~
