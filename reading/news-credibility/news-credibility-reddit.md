# News Source Credibility Assessment: A Reddit Case Study

*Authors:* Arash Amini, Yigit Ege Bayiz, Ashwin Ram, Radu Marculescu, Ufuk Topcu

University of Texas Austin

## Questions
- What the hell is a Siamese network model
  - A Siamese neural network (sometimes called a twin neural network) is an 
    artificial neural network that uses the same weights while working in tandem 
    on two different input vectors to compute comparable output vectors. 
    Often one of the output vectors is precomputed, thus forming a baseline 
    against which the other output vector is compared. This is similar to 
    comparing fingerprints but can be described more technically as a distance 
    function for locality-sensitive hashing.

## Abstract

- CREDiBERT; CREDibility assessment using Bi-directional Encoder Representations 
  from Transformers
- Tuned for Reddit submissions focusing on political discourse
- Encoding content with CREDiBERT and integrating with classification neural net,
  improves Reddit credibility assessments by 3% F1 score over existing methods
- New version of post-to-post network in Reddit that encodes user interactions
  enhances credibility assessment by 8% F1 score.
- Demonstrates CREDiBERT applicability by evaluating Reddit community susceptibility
  to different topics and assessing credibility score of unseen sources

## Automated Fact Checking

- NLP techniques demonstrated high precision in differentiating fake news 
  (Raza and Ding 2022)
- Despite advances in detection, reputation of article source is the key component
- Credibility ~ perceived believability
- Devlin et al 2018; Jwa et al. 2019; exBAKE, BERT based architecture for fake
  news identification
- BERT tends to overfit and may underperform on large texts compared to Stylometry
  and BiLSTM; Pryzbyla (2020)
- Raza and Ding 2022 propose enhanced transformer model; leveraging content and 
  social media traces.
- Look into Chen and Shy 2024, Leite et al. 2023 for developments in combating
  misinformation

## Credibility Assessment

- Historical patterns in source reporting accuracy good indicator of propaganda
- User recognition and reaction
- CREDiBERT model; semi supervised, utilize Siamese network architecture designed
  to mitigate overfitting issues in standard BERT models for this task.

## Results

- TL;DR, improved over S-BERT/BERT classification