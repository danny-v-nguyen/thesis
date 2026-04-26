# Applying Machine Learning and Corpus Linguistics to Open-Source Intelligence Validation
## A Consistency Determination-Based Approach to Comparing Verified and Unverified Text Data

*Author:* Anthony Alix Jacques

Dissertation, 2024; Pardee RAND Graduate School

# Intro

- Language agnostic solution for determining validity of OSINT
- Fake News Detection methods
- Method uses ChatGPT to compare self-content with other portions to 
  determine consistency instead of finding a truth source or whatever
- Stance, Syntax, Source, Source-Corroboration
    - Stance: emotion or opinion
    - Syntax: grammar, spelling, style, etc.
    - Source: source reputability
    - Source-Corroboration: connections to other sources
- FNDs use some combo of the above to determine truthfulness
- Generative models can accurately mimic 4S categories
- Fact checking verification involves Knowledge Graphs and Recognizing
  Textual Entailment techniques
    - KGs compare nodal connections between textual concepts with
      unverified text
    - RTE predicts if evidence in reference text supports unverified 
      claim, using ML or lexical similarity.
- Application of similarity metrics to machine learning for OSINT 
  validation purposes
- Application of transformer machine learning methodology for doc-to-doc
  similarity comparison in the OSINT validation context
- Consistency determination created via ML and similarity metrics
  which can be used as another assistive measure by intel analysts
- Analyzing doc-to-tweet on individual level rather than aggregate
  in portuguese
- Echo effect pollutes data collection
    - News media reprinting articles from others
    - People repeating each other's opinions on Twitter
    - Multiple articles that are all based on one single source
    - etc.

## Data

- MINT: Mainstream and Independent News Text Corpus
    - Hard News: mainstream media
    - Soft News: opinion section of mainstream media and independent 
      news/magazines
    - Opinion: celebrity, fashion, beauty, family, lifestyle; magazines,
      tabloids, newspaper supplements.
    - Satirical: satire websites
    - Conspiracy: Websites that published 5 or more COVID conspiracy 
      theories

*Note: does not distinguish between fake and true news*

- Manual labeling of consistency
- There were mistakes lol

## Method

- Trained a neural net to binary classify two statements
- Also used ChatGPT to compare the HCPM to the unverified statements
  and classify accordingly
- False positives (consistent when not) deeply punished by neural net 
  trainings; caused model to strongly prefer classifying as inconsistent

## Conclusion

- Did ok. ChatGPT-4 also did ok.
