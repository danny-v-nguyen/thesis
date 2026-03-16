# Teaming LLMs to Detect and Mitigate Hallucinations
*Authors:* Demian Till, 
           John Smeaton, 
           Peter Haubrick, 
           Gouse Saheb, 
           Florian Graef, 
           David Berman

*Source URL:* [https://arxiv.org/pdf/2510.19507]()

## Abstract
- Recent work demonstrated results in LLM hallucination detection and mitigation
  through consistency-based approach through aggregation of responses from a single
  LLM for a prompt
- Extending these single-model consistency methods between multiple LLMs with 
  different training data, schemes, and architectures can result in substantial 
  further improvements in hallucination detection and mitigation.
- Consortium consistency approach evaluated across pool of 15 LLMs
- Explore conditions where it is beneficial to team LLMs in this manner
- Performance improvements come with reduced inference costs

## Introduction
> When information relevant to deployment-time performance is under-represented or 
> misrepresented in the pre-training corpus, the model is less likely to be able to
> provide accurate responses [1,2]

> Moreover, instruction fine-tuning can incentivize models to make educated guesses 
> in the absence of reliable knowledge on a given topic. [3]

**Note** Look into [3] if there's quantifiable measurement

> Self-consistency [5] effectively mitigates a class of hallucinations by sampling 
> multiple generations from an LLM in response to a given prompt and a final 
> answer is selected by taking a majority vote over the responses. This approach, 
> and follow up work [6] demonstrated improvement over alternative methods such as 
> debate [7] and self-reflection [8], and that smaller models using this 
> approach can match or surpass the accuracy of substantially stronger models.

**Note** Look into debate and self-reflection.

> Semantic entropy [9,10] uses a similar consistency-based approach to detect 
> likely hallucinations by grouping together similar generations sampled in 
> response to a given prompt and computing the entropy over the resulting 
> clusters of responses.

> The authors showed that semantic entropy achieves greater hallucination 
> detection accuracy than alternative methods, including ones requiring 
> white-box model access and model training [10].

> [11] extend this idea to compute consistency across multiple responses based on
> semantic information within internal model embeddings rather than output text, 
> achieving state-of-the-art (SOTA) hallucination detection results.

> **However these single-model consistency approaches naturally fail in cases where** 
> **models produce relatively consistent hallucinations in response to a given prompt.**

Expected.

> Other consistency-based hallucination detection methods such as [11–14] could 
> similarly be extended to use multiple different LLMs, however we leave this to 
> future work, and in the case of [11] this would require aligning the embedding 
> spaces of different LLMs.

Need to look into [11]

## Experimental Setup

### Metrics

Accuracy, AUROC, AURAC.

### Baselines

**Hard:** highest of the M single-model consistency score on a metric. Requires 
*a priori* knowledge of what the best M is

**Standard:**  median of M scores. Average performance of single model consistency
methods.

**Worst-case:** the lowest of M

### Sampling 

Generated 40 responses per prompts. Adjust prompts when M is not a factor of 40.

### Models

15 LLMs, 6B to 141B params. Check Appendix C for list.

### Datasets

11 tasks covering reasoning and general domain knowledge:

- **Reasoning** GSM8K [17] (200 random sample questions), GPQA-Diamond [18]
- **General/Domain Knowledge** 8 MMLU [19] subsets covering virology, world religions,
  jurisprudence, astronomy, public relations, anatomy, college chemistry, global facts.
  TruthfulQA [2], common misconceptions.

### Separate tasks for model selection

Strategy to select which models to team consider relative and absolute capability
levels of the candidate models. Had to estimate benchmark scores for models used
described in their Appendix D.

### Compute costs

Sampling 40 responses per question across 11 datasets and 15 models cost ~$1000.

## Results

### Performance with well-matched models

*Table 1:*

| Metric           | Baseline   | Accuracy         | AUROC              | AURAC             |
|------------------|------------|------------------|--------------------|-------------------|
| Mean score ∆ (%) | Hard       | $+1.33 \pm 1.03$ | $+1.84 \pm 1.48$   | $+2.75 \pm 0.69$  |
|                  | Standard   | $+3.70 \pm 1.20$ | $+5.63 \pm 1.46$   | $+5.39 \pm 1.09$  |
|                  | Worst-case | $+9.67 \pm 3.44$ | $+18.80 \pm 10.41$ | $+16.20 \pm 7.22$ |

TL;DR marginal improvement over any single model.

### Impact of model strength

Stronger group, stronger Hard baseline improvement. Possibly due to more intelligent
guesses. Weaker models generate varied responses on hallucinations, more difficult to
detect using single-model semantic entropy. Consortium entropy benefits from
different models due to less likelihood of all hallucinating in the same way.

### Impact of variance in model capability

As variance of model capability goes down, advantage of consortium over hard baseline
grows more reliable across all metrics. Bundling weak models into consortiums 
increases response entropy ("dissenting opinions").

### Cost-performance tradeoffs

Costs money yo.

## Related works

### Hallucination detection

Whitebox methods using output probabilities [20-22] to calculate uncertainty scores;
training hallucination detection using internal embeddings [23,24].

Blackbox methods prompt LLMs to provide confidence scores [25] and sampling responses
for consistency [12,13,9,10]. [26] uses verifier model to check answers of target
model, but is limited to teaming two models.

Other methods combine consistency with white-box model access [14]; [11] achieves 
SOTA (state of the art).

This method combines [9,10], could be applied to [11-14], would probably see
similar improvement. Future work.

### Hallucination mitigation

[5,27,6] can be seen as mitigating a subset of hallucinations.

Combining models before responses [28-31], during generation of responses [32-34],
and after generation [7,35-37] can be seen as mitigating hallucinations from 
limitations of individual models.

Note these are all focusing on increasing accuracy rather than hallucination 
detection.

Another line of approach is RAG [38-40]. Hey, this is another thing I need to
read on.