# Facilitating Fine-grained Detection of Chinese Toxic Language: Hierarchical Taxonomy, Resources, and Benchmark

**The paper has been accepted in ACL 2023.**

☠️ ***Warning: The samples presented by this paper may be considered offensive or vulgar.***

## 📜 Monitor Toxic Frame
we introduce a hierarchical taxonomy **Monitor Toxic Frame**. Based on the taxonomy, the posts are progressively divided into diverse granularities as follows: **_(I) Whether Toxic_**, ***(II) Toxic Type*** (general offensive language or hate speech), ***(III) Targeted Group***, ***(IV) Expression Category*** (explicitness, implicitness, or reporting). 

## 📜 ToxiCN
We conduct a fine-grained annotation of posts crawled from _Zhihu_ and _Tieba_, including both direct and indirect toxic samples. And ToxiCN dataset is presented, which has 12k comments containing **_Sexism_**, **_Racism_**, **_Regional Bias_**, **_Anti-LGBTQ_**, and **_Others_**. The dataset is presented in ***ToxiCN_1.0.csv***.

## 📜 Insult Lexicon
The resource is still being collated. We will upload it as soon as possible.

## 📜 Benchmark
We present a migratable benchmark of **Toxic Knowledge Enhancement** (**TKE**), enriching the text representation. The code is shown in **_modeling_bert.py_**, which is based on **transformers 3.1.0**.

## ❗️ Ethics Statement
The opinions and findings contained in the samples of our presented dataset should not be interpreted as representing the views expressed or implied by the authors. We hope that the benefits of our proposed resources outweigh their risks. **All resources are for scientific research only.**

## ❗️ Licenses
This work is licensed under a Creative Commons Attribution- NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0). 

## Poster

![Poster](https://github.com/DUT-lujunyu/ToxiCN/assets/53985277/545440ad-6abf-4cde-83b8-4e4cae948c56)


## Cite
If you want to use the resources, please cite the following paper:
~~~
@misc{lu2023facilitating,
      title={Facilitating Fine-grained Detection of Chinese Toxic Language: Hierarchical Taxonomy, Resources, and Benchmarks}, 
      author={Junyu Lu and Bo Xu and Xiaokun Zhang and Changrong Min and Liang Yang and Hongfei Lin},
      year={2023},
      eprint={2305.04446},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
~~~

