---
license: apache-2.0
task_categories:
- text-classification
language:
- id
tags:
- Hate Speech Classification
- Toxicity Classification
- Demographic Information
size_categories:
- 10K<n<100K
configs:
  - config_name: main
    data_files:
    - split: main
      path: 
      - "indotoxic2024_annotated_data_v2_final.jsonl"
  - config_name: annotator
    data_files:
    - split: annotator
      path:
      - "indotoxic2024_annotator_demographic_data_v2_final.jsonl"
---

```
Notice: We added new data and restructured the dataset on 31st October 2024 (GMT+7)
Changes:
- Group unique texts together
- The annotators of a text are now set as a list of annotator_id. Each respective column is a list of the same size of annotators_id.
- Added Polarized column

Notice 2: We rename the dataset from IndoToxic2024 to IndoDiscourse
```

# A Multi-Labeled Dataset for Indonesian Discourse: Examining Toxicity, Polarization, and Demographics Information

## Dataset Overview

IndoToxic2024 is a multi-labeled dataset designed to analyze online discourse in Indonesia, focusing on **toxicity, polarization, and annotator demographic information**. This dataset provides insights into the growing political and social divisions in Indonesia, particularly in the context of the **2024 presidential election**. Unlike previous datasets, IndoToxic2024 offers a **multi-label annotation** framework, enabling nuanced research on the interplay between toxicity and polarization.

## Dataset Statistics

- **Total annotated texts:** **28,477**  
- **Platforms:** X (formerly Twitter), Facebook, Instagram, and news articles  
- **Timeframe:** September 2023 – January 2024  
- **Annotators:** 29 individuals from diverse demographic backgrounds  

### Label Distribution - For Experiments

| Label        | Count |
|-------------|-------|
| **Toxic**   | 2,156 (balanced) |
| **Non-Toxic** | 6,468 (balanced) |
| **Polarized** | 3,811 (balanced) |
| **Non-Polarized** | 11,433 (balanced) |

## Dataset Structure

The dataset consists of texts labeled for **toxicity and polarization**, along with **annotator demographics**. Each text is annotated by at least one coder, with **44.6% of texts receiving multiple annotations**. Annotations were aggregated using majority voting, excluding texts with perfect disagreement.

### Features:
- `text`: The Indonesian social media or news text
- `toxicity`: List of toxicity annotations (1 = Toxic, 0 = Non-Toxic)
- `polarization`: List of polarization annotations (1 = Polarized, 0 = Non-Polarized)
- `annotators_id`: List of annotator_id that annotate the text (anonymized) -- Refer to `annotator` subset for each annotator_id's demographic informatino

## Baseline Model Performance

![image/png](https://cdn-uploads.huggingface.co/production/uploads/61e997f66735d3a73e291055/mWNkGL_RdqdKQE7rzK59a.png)

### Experiment Code

[Notebook for Toxicity Related Experiment](https://huggingface.co/datasets/Exqrch/IndoDiscourse/blob/main/IndoDiscourse%20-%20Toxicity%20Related%20Experiment%20Code.ipynb)


### Key Results:

We benchmarked IndoDiscourse using **BERT-based models** and **large language models (LLMs)**. The results indicate that:

- **BERT-based models outperform 0-shot LLMs**, with **IndoBERTweet** achieving the highest accuracy.
- **Polarization detection is harder than toxicity detection**, as evidenced by lower recall scores.
- **Demographic information improves classification**, especially for polarization detection.

### Additional Findings:
- **Polarization and toxicity are correlated**: Using polarization as a feature improves toxicity detection, and vice versa.
- **Demographic-aware models perform better for polarization detection**: Including coder demographics boosts classification performance.
- **Wisdom of the crowd**: Texts labeled by multiple annotators lead to higher recall in toxicity detection.

## Ethical Considerations

- **Data Privacy**: All annotator demographic data is anonymized.
- **Use Case**: This dataset is released **for research purposes only** and should not be used for surveillance or profiling.

## Citation

If you use IndoDiscourse, please cite:

```bibtex
@misc{susanto2025multilabeleddatasetindonesiandiscourse,
      title={A Multi-Labeled Dataset for Indonesian Discourse: Examining Toxicity, Polarization, and Demographics Information}, 
      author={Lucky Susanto and Musa Wijanarko and Prasetia Pratama and Zilu Tang and Fariz Akyas and Traci Hong and Ika Idris and Alham Aji and Derry Wijaya},
      year={2025},
      eprint={2503.00417},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.00417}, 
}```
