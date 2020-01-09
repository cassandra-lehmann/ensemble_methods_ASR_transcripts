## Project File Structure:

- **Data:** Storage of project data.
- In the root directory you can find all the scripts (.py or .ipynb):

00_explore.ipynb

10_clean.ipynb

20_premodeling.ipynb

30_modelling.ipynb

40_analysis.ipynb

functions.py.
 


# Title:

Better than the Best - Ensemble methods for optimizing Automated Speech Recognition transcripts
(Propulsion Academy Final Project DSWD-2019-09)

Authors: Cassandra Danielle Lehmann & Oana Serban

## Install

To install the environment:

- `git clone` this repo
- Set up the provided Anaconda virtual environment which includes the needed packages with the right versions: `conda env create -f spinning_bytes.yaml`
- Activate the spinning_bytes environment: `conda activate spinning_bytes`
- Run `jupyter lab` and open the provided notebooks.


## Project description
Automatic Speech Recognition (ASR) has many applications, for instance transcribing interviews and meetings, chatbots, automatic answering for phonecalls etc. The speech-to-text quality has significantly increased in the last years and so has the demand for ASR-based solutions. Tech companies developing ASR systems are in constant pursuit of reaching the best possible transcription quality, and in some experiments even reach human performance. 

However, in many cases the output of a particular ASR system contains large amounts of errors. One potential solution would be to go beyond one ASR system and use/combine the output of several systems to produce the transcript. In other contexts (e.g. sentiment analysis), it has been shown that such "ensemble methods" outperform each single participating system. Is the same possible for ASR? 


## Goals
- Define a method for comparing ASR system transcriptions on a word level and retrieving the most accurate ones.
- Investigate discrepancies between system transcriptions on a word level.
- Develop a tool for generating an optimized transcription using the outputs from multiple ASR systems.
- Generate optimized transcriptions for a provided set of utterances and calculate the quality  of the meta-system per utterance. 


### Data set description
70875 manual transcriptions with metadata coming from 9 English corpora. 
70875 machine transcriptions produced by at least 7 ASR systems.
Data is provided in a unified format as a JSON file.


### Data exploration

 - explored files received and file structure; 
 - average wer per corpus and wer distribution per, configuration, machine; 
 - volume of data per corpusa and configuration; 
 - are there empty reference texts?; 
 - are there empty hypothesis texts but not reference texts?;
 - reference text duplicates; 
 - do references contain special characters?
 - how many references with less than 5 words
 - upper cases in reference or hypothesis


### Data cleaning

Cleaning steps:
 - lowercase reference and hypothesis
 - replace t_v_, t_v_s,  i_d_ with tv, tvs, id
 - remove from reference:, ", [, ], {, }
 - remove leading, trailing, multiple spaces from hypothesis and reference
 - remove sentences with <5 words
 - drop switchboard corpus
 - drop reference texts with more than 1 speaker
 - keep only columns of interest

 From where did we loose data after cleaning?
 Recompute WER after the cleaning.
 Order data by mean wer by configuration.

### Pre-modeling 
 - keeps machines 2, 3, 4, 5
 - split hypoithesis into words

ALIGNMENT:
 - inspired by paper: http://my.fit.edu/~vkepuska/ece5527/sctk-2.3-rc1/doc/rover/rover.htm
 - aign identical words using biopython package (pairwise2.align.globalmx(ref,hyp,10,1, gap_char=["\n_"]), documentation under https://biopython.org/DIST/docs/api/Bio.pairwise2-module.html.
 - important: the order of the hypothsis when creating the wtn matters, you should start witht the best machine/configuration. How do you define best? We 'hardcoded' it based on average wer.

VOTING:
- majority voting after alignment, using a modified mode function (if there is a tie between a word and the gap character, take the word).
- important: improving on breaking the ties (now done randomly) would improve the model.

SPLITTING the references in to train, validation, test. In train and validation the alignment is done to reference. In test we assume we don't have a reference, therefore the alignment is done to the hypothesis of the best machine.


### Feature engineering

A machine tracker is implemented as additional feature for model training and testing.
Machine-tracker keeps track on ASR machines that have previously predicted the correct word. (one instance lookback)
When an ASR machine outputs a correct word, the tracker value will equal 1, otherwise the tracker value will equal 1.
During testing, since there is no reference of a correct word, the model considers the previously predicted word as a correct word
although this is not always the case. 
With this approach, there is a disconnect between training and testing data, however model performance is still relatively accurate.

Note:
During testing, feature engineering and model prediction are implemented using a for loop.
Using 2802 sentences (43316 words) as test data, runtime approximates to 40Hours. However, only a small amount of RAM and CPU's 
are needed to run the script.

### Modelling
Mainly used: Random Forest Classifier model
Also tried, but not included in notebooks: Random Forest Regressor (performance lower than Random Forest Classifier), CatBoost (performance lower than Random Forest Classifier).

Note:
Random Forest Classifier requires a minimum of 180GB of RAM. Total training-time of model is approximately 5 minutes with 32CPU.

How does it work?
Random Forest model is trained on over 400,000 words that are encoded into numerical values. 
Classification model is used; hence each word is considered a class.
The model attempts to classify the correct word based on previous training and new input data consisting of words
captured by multiple ASR machines, as well as trackers indicating correct past predictions (one instance lookback) to add weights to ASR machines that perform well.
The model outputs predicted class(word) and probability of predictions being correct.


### Outlook

We have a voting mechanism and a basline model that can be improved upon, for example by:
 - Enriching model’s vocabulary
 - Combining majority voting and model output
 - Using Word2Vec (giving context to words) instead of with LabelEncoding
 - Adding extra features (types of words)
 - Using probability calculation of Random Forest
 - Taking confidence as a factor 
 - Deep Learning and pre-trained models (have a look at this paper: https://arxiv.org/pdf/1802.02607.pdf)

