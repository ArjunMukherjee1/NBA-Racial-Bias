# NBA-Racial-Bias

The language that sports commentators use has a major impact on the perception of athletes.
Prior social science research shows that live sports commentary often contains racial bias.
However, many of these studies were predicated on manual categorizations of statements and
small sets of data. This study investigated racial bias in basketball commentary by using
computational techniques to analyze a large sample of data. Transcripts of NBA games from
2023-2025 were processed to create a dataset of over 20,000 player mentions tagged with the
player’s skin tone (lighter-skinned or darker-skinned). The data was analyzed using two
approaches: sentiment analysis and machine learning classification. The results of sentiment
analysis showed that mentions describing darker-skinned players and lighter-skinned players
were very similar in tone. Conversely, a logistic regression model was able to predict a player’s
skin tone from a mention with an accuracy of 60%, signifying that there were marginal
differences in the ways commentators discussed players of different races. Inspection of the
model’s coefficients found that terms related to athleticism (e.g., athletic, speed) were associated
with darker-skinned players while words related to character and intellect (e.g., smart, mature)
were linked to lighter-skinned players. This disparity in speech promotes harmful stereotypes
about race, physicality and cognitive ability in sports. However, humans – regardless of ethnicity
- harbor ethnocentric biases which, in this study, were detected by ML models as being
marginally greater (~ 60% accuracy) than chance alone would dictate. Therefore, there is a
significant risk of incorrectly attributing these universal human traits with deliberate malice and
resentment against other ethnic groups or races; especially in the absence of any sentiment bias.

This repository contains all code and data used for this project.
