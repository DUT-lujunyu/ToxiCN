# Facilitating Fine-grained Detection of Chinese Toxic Language: Hierarchical Taxonomy, Resources, and Benchmark

🎉**2024.9 Our related study, titled " ", has been accepted to NeurIPS 2024! In this paper, we present ToxiCN_MM, the first Chinese harmful meme dataset. Welcome to star or fork it at [https://github.com/DUT-lujunyu/ToxiCN](https://github.com/DUT-lujunyu/ToxiCN)!**
🎉**2024.5 Our proposed dataset, ToxiCN, has been adopted by the international evaluation CLEF 2024: [Multilingual Text Detoxification](https://pan.webis.de/clef24/pan24-web/text-detoxification.html) as the sole Chinese data source. [Report](https://ceur-ws.org/Vol-3740/paper-223.pdf)**
___ 



**The paper has been accepted in ACL 2023 (main conference, long paper).** [Paper](https://aclanthology.org/2023.acl-long.898/)

☠️ ***Warning: The samples presented by this paper may be considered offensive or vulgar.***

## 📜 Monitor Toxic Frame
we introduce a hierarchical taxonomy **Monitor Toxic Frame**. Based on the taxonomy, the posts are progressively divided into diverse granularities as follows: **_(I) Whether Toxic_**, ***(II) Toxic Type*** (general offensive language or hate speech), ***(III) Targeted Group***, ***(IV) Expression Category*** (explicitness, implicitness, or reporting). 

## 📜 ToxiCN
We conduct a fine-grained annotation of posts crawled from _Zhihu_ and _Tieba_, including both direct and indirect toxic samples. And ToxiCN dataset is presented, which has 12k comments containing **_Sexism_**, **_Racism_**, **_Regional Bias_**, **_Anti-LGBTQ_**, and **_Others_**. The dataset is presented in ***ToxiCN_1.0.csv***. Here we simply describe each fine-grain label.

| Label           | Description                                                  |
| --------------- | ------------------------------------------------------------ |
| toxic           | Identify if a comment is toxic (1) or non-toxic (0).         |
| toxic_type      | non-toxic: 0, general offensive language: 1, hate speech: 2  |
| expression      | non-hate: 0, explicit hate speech: 1, implicit hate speech: 2, reporting: 3|
| target (a list) | LGBTQ: Index 0, Region: Index 1, Sexism: Index 2, Racism: Index 3,  others: Index 4, non-hate: Index 5 |

## 📜 Insult Lexicon
See https://github.com/DUT-lujunyu/ToxiCN/tree/main/ToxiCN_ex/ToxiCN/lexicon
## 📜 Benchmark
We present a migratable benchmark of **Toxic Knowledge Enhancement** (**TKE**), enriching the text representation. The code is shown in **_modeling_bert.py_**, which is based on **transformers 3.1.0**.

## ❗️ Ethics Statement
The opinions and findings contained in the samples of our presented dataset should not be interpreted as representing the views expressed or implied by the authors. We hope that the benefits of our proposed resources outweigh their risks. **All resources are for scientific research only.**

## ❗️ Licenses
This work is licensed under a Creative Commons Attribution- NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0). 

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
