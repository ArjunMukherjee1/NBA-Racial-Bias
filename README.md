# Racial Bias in NBA Commentary

A NLP study of whether NBA commentators describe players differently based on skin tone. Transcripts from 50+ broadcasts (2023–2025) are turned into 28,756 labeled player mentions and analyzed with sentiment analysis and machine learning models.

*Code for the paper [Computationally investigating racial bias in NBA commentary](https://doi.org/10.64336/001c.144055) (Journal of High School Science, 2025).*

## Data

Each mention is the 15 words on either side of a player's name. The player's name becomes `<mentioned_player>` and other players' names become `<teammate>` or `<opponent>`. Skin tone labels come from matching NBA.com headshots to the Fitzpatrick scale.

## Analysis (`AnalyzeData.py`)

- **Preprocessing** Darker-skinned mentions are downsampled to match the lighter-skinned count. Each mentions is converted into TF-IDF vectors of single words and word pairs, which helps the models understand what words are important. Data is divided into an 80/20 train/test split.
- **Model Training:** Each model predicts skin tone from a mention alone. Since indicators of skin tone were removed, model accuracy significantly above 50% indicates racial bias. Logistic regression uses the `saga` solver with L2 regularization. The random forest has 200 trees with a max depth of 20. BERT is `bert-base-uncased`, fine-tuned for 3 epochs. Only logistic regression runs by default (had the highest accuracy in tests). The other models are commented out.
- **Descriptors:** Logistic regression coefficients show which of ~100 stereotype-related words (athleticism, intellect, work ethic, etc.) the model links to each group. Proportion z-tests check which of those links are significant.
- **Sentiment:** VADER scores each mention from −1 to 1. Scores are compared across racial groups to determine whether commentators discuss players more positive/negatively based on race.
## Findings

- Sentiment is nearly identical for both groups.
- Logistic regression reached 60% accurac which is significantly above chance. Indicates that the model was able to reliably identify differences in the ways commentators discuss players of different skin tones.
- Athleticism words (`athletic`, `fast`, `machine`) are linked to darker-skinned players. Intellect and effort words (`smart`, `hustling`, `disciplined`) are linked to lighter-skinned players which reinforces existing stereotypes in athletics.

## Structure

```
ExtractMentions.py   Builds the mention dataset
AnalyzeData.py       Classifiers, descriptor coefficients, sentiment analysis
cleanTranscript.py   Cleans the raw YouTube transcripts
convert.py           Converts roster CSV to a JSON file
train_data_15.json   Dataset used in the paper
Data/                Transcripts, rosters, datasets with other context window sizes
```

## Running

```bash
pip install numpy pandas nltk scikit-learn xgboost vaderSentiment scipy matplotlib seaborn
```

```bash
python AnalyzeData.py
```

To rebuild the dataset, set the window size at the bottom of `ExtractMentions.py` and run it.
