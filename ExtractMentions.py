#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Sun Aug 25 21:52:19 2024

@author: arjunmukherjee
'''
import sys
sys.path.append('/Users/arjunmukherjee/Library/Python/3.12/lib/python/site-packages')
from nltk.tokenize import word_tokenize
import json
import re
import os
from collections import Counter
with open("ASI Data/NBA_rosters.json", "r") as file:
    rosters = json.load(file)

players = []
tokens = []
year = 0
teams = []
all_mentions = []
def readFile(fileName, team1, team2, date):
    global players, tokens, year, teams
    transcriptFile = open(fileName, "r")
    transcriptText = transcriptFile.read()
    transcriptText = transcriptText.lower()
    #transcriptText = replaceContractions(transcriptText)
    transcriptFile.close()
    tokens = word_tokenize(transcriptText)
    players = rosters[team1] | rosters[team2]
    year = date
    teams = [team1, team2]
    #checkNameOverlap(players)

def extract_name_variants(p):
    possibilities = []
    full_name = p.lower()
    possibilities.append(full_name)
    split = full_name.split()
    first = split[0]
    last = split[1]
    middle = ""
    if('-' in last):
        specialName = last.split('-')
        middle = specialName[0]
        last = specialName[1]
        possibilities.append(first + " " + middle + " " + last)
        possibilities.append(middle + " " + last)
        possibilities.append(first + " " + last)
    elif(len(split) == 3):
        suffix = split[2]
        possibilities.append(first + " " + last)
        possibilities.append(last + " " + suffix)

    possibilities.append(first)
    possibilities.append(last)
    return possibilities
    
def checkNameOverlap(players):
    allNames = []
    for p in players:
        full_name = p.lower()
        split = full_name.split()
        allNames.append(split[0])
        allNames.append(split[1])
        if('-' in split[1]):
            specialNames = split[1].split('-')
            allNames.append(specialNames[0])
            allNames.append(specialNames[1])
    count = Counter(allNames)
    duplicates = [item for item, count in count.items() if count > 1]
    print(duplicates)

# Function to replace names in the context that belong to other players.
def replaceOtherPlayerNames(context, current_player, all_player_names, teams):
    # Extract the variants for the current player (we don't want to replace these).
    current_variants = extract_name_variants(current_player)

    if current_player in [k.lower() for k in rosters[teams[0]].keys()]:
        current_team, opponent_team = teams[0], teams[1]
    else:
        current_team, opponent_team = teams[1], teams[0]
    # Loop over all players in the dataset
    for player in all_player_names:
        player_lower = player.lower()

        # Skip replacing the current player's own mentions (already replaced separately)
        for variant in current_variants:
            # Use a regex to find whole-word matches of the variant.
            # This prevents accidental replacement of substrings.
            pattern = r'\b' + re.escape(variant) + r'\b'
            # Replace the matched variant with the placeholder "player".
            context = re.sub(pattern, '<mentioned_player>', context)
        # Generate name variants for the other player
        player_variants = extract_name_variants(player_lower)

        for variant in player_variants:
            pattern = r'\b' + re.escape(variant) + r'\b'

            # Determine if the player is a teammate or an opponent
            if player in rosters[current_team]:
                replacement = '<teammate>'
            elif player in rosters[opponent_team]:
                replacement = '<opponent>'
            context = re.sub(pattern, replacement, context, flags=re.IGNORECASE)
    return context


def replaceContractions(context):
    contractions = {
        "couldn't": "could not",
        "aren't": "are not",
        "didn't": "did not",
        "doesn't": "does not",
        "hadn't": "had not",
        "i'm": "i am",
        "it'll": "it will",
        "it's": "it is",
        "let's": "let us",
        "here's": "here is",
        "that's": "that is",
        "he's": "he is",
        "you're": "you are",
        "don't": "do not",
        "he'll": "he will",
        "hasn't": "has not",
        "there's": "there is",
        "wasn't": "was not",
        "can't": "can not",
        "isn't": "is not",
        "won't": "will not",
        "haven't": "have not"
    }
    contractions_re = re.compile('(%s)' % '|'.join(map(re.escape, contractions.keys())), flags=re.IGNORECASE)
    
    def replace(match):
        contraction = match.group(0)
        expanded = contractions.get(contraction.lower())
        return expanded if expanded else contraction
    
    newContext = contractions_re.sub(replace, context)
    return newContext

def captureWordBubble(numWords, p, window_len, i, full_name, teams):
    start = max(0, i- window_len)
    before = " ".join(tokens[start:i])
    end = min(i+window_len + numWords, len(tokens))
    after = " ".join(tokens[i+numWords:end])
    context = before + " <mentioned_player> " +after
    context = replaceContractions(context)
    context = replaceOtherPlayerNames(context, full_name,players, teams)
    label = {"Player": full_name, "Skin Tone": players[p]["Skin Tone"], "Teams": teams, "Year": year}
    mention = {"Label" : label, "Mention": context}
    all_mentions.append(mention)

def extractSpecialNames(i, p, first, middle, last, full_name, window_len, teams):
    fullLast = middle + '-' + last
    if(tokens[i] == first and tokens[i+1] == middle and tokens[i+2] == last):
        captureWordBubble(3, p,window_len, i, full_name, teams)
    elif(tokens[i]== first and tokens[i+1] == fullLast):
        captureWordBubble(2, p,window_len, i, full_name, teams)
    elif((tokens[i] == first and tokens[i+1] == last)):
        captureWordBubble(2, p,window_len, i, full_name, teams)
    elif(tokens[i] == first):
        captureWordBubble(1,p,window_len, i, full_name, teams)
    elif(tokens[i] == middle and tokens[i+1] == last and tokens[i-1] != first):
        captureWordBubble(2,p,window_len, i, full_name, teams)
    elif(tokens[i] == last and tokens[i-1] != middle and tokens[i-1] != first):
        captureWordBubble(1,p,window_len, i, full_name, teams)
    elif(tokens[i] == fullLast and tokens[i-1] != first):
        captureWordBubble(1,p,window_len, i, full_name, teams)

def extractMention(i, p, first, last, full_name, window_len, teams):
    if((tokens[i] == first and tokens[i+1] == last)):
        captureWordBubble(2,p,window_len, i, full_name, teams)
    elif(tokens[i] == first):
        captureWordBubble(1,p,window_len, i, full_name, teams)
    elif(tokens[i] == last and tokens[i-1] != first):
        captureWordBubble(1,p,window_len, i, full_name, teams)

def extractJuniors(i, p, first, last, suffix, full_name, window_len, teams):
    if(tokens[i] == first and tokens[i+1] == last and tokens[i+2] == suffix):
        captureWordBubble(3,p,window_len, i, full_name, teams)
    elif(tokens[i] == first and tokens[i+1] == last):
        captureWordBubble(2,p,window_len, i, full_name, teams)
    elif(tokens[i] == first):
        captureWordBubble(1,p,window_len, i, full_name, teams)
    elif(tokens[i] == last and tokens[i+1] == suffix and tokens[i-1] != first):
        captureWordBubble(2,p,window_len, i, full_name, teams)
    elif(tokens[i] == last and tokens[i-1] != first):
        captureWordBubble(1,p,window_len, i, full_name, teams)

def findMentions(players, window_len, teams, year):
    for i in range(len(tokens)):
        for p in players:
            full_name = p.lower()
            split = full_name.split()
            first = split[0]
            last = split[1]
            if('-' in last):
                specialName = last.split('-')
                middle = specialName[0]
                last = specialName[1]
                extractSpecialNames(i, p, first, middle, last, full_name, window_len, teams)
            elif(len(split) == 3):
                suffix = split[2]
                extractJuniors(i, p, first, last, suffix, full_name, window_len, teams)
            else:
                extractMention(i, p, first, last, full_name, window_len, teams)
            
            


def collectAllTranscripts():
    folder_path = "ASI Data/Train/"

    # Regular expression pattern to extract teams and year from filenames
    pattern = r"(\d+:\d+:\d+) - (\w+) vs (\w+).txt"

    # Loop through each file in the folder
    #count = 0
    for file_name in os.listdir(folder_path):
        match = re.match(pattern, file_name)
        if match:
            date, team1, team2 = match.groups()
            #count +=1
            year = int(date.split(":")[-1]) + 2000  # Converts "25" to "2025"

            # Construct the full file path
            file_path = os.path.join(folder_path, file_name)
            #print(file_path, team1, team2, year)
            # Process the file
            readFile(file_path, team1, team2, year)
            findMentions(players, 5, teams, year)
collectAllTranscripts()

#readFile("newTranscript.txt", "Suns", "Grizzlies", 2024)
#findMentions(players, 5, teams, year)

output_file = "train_data_5.json"
# Write the dictionary to a JSON file
with open(output_file, "w") as json_file:
    json.dump(all_mentions, json_file, indent=4)

'''
for i in range(len(tokens)):
    for p in players:
        full_name = p.lower()
        split = full_name.split()
        first = split[0]
        last = split[1]
        if((tokens[i] == first and tokens[i+1] == last)):
            start = max(0, i- window_len)
            before = " ".join(tokens[start:i])
            end = min(i+window_len + 2, len(tokens))
            after = " ".join(tokens[i+2:end])
            mention = before + " <player> " +after
            mentions.append(mention)
        elif(tokens[i] == first):
            start = max(0, i- window_len)
            before = " ".join(tokens[start:i])
            end = min(i+window_len + 1, len(tokens))
            after = " ".join(tokens[i+1:end])
            mention = before + " <player> " +after
            mentions.append(mention)
        elif(tokens[i] == last and tokens[i-1] != first):
            start = max(0, i- window_len)
            before = " ".join(tokens[start:i])
            end = min(i+window_len + 1, len(tokens))
            after = " ".join(tokens[i+1:end])
            mention = before + " <player> " + after
            mentions.append(mention)
print(mentions)

if((tokens[i] == first and tokens[i+1] == last)):
                start = max(0, i- window_len)
                before = " ".join(tokens[start:i])
                end = min(i+window_len + 2, len(tokens))
                after = " ".join(tokens[i+2:end])
                context = before + " <player> " +after
                label = {"Player": full_name, "Skin Tone": players[p]["Skin Tone"], "Teams": teams, "Year": year}
                mention = {"Label" : label, "Mention": context}
                all_mentions.append(mention)
            elif(tokens[i] == first):
                start = max(0, i- window_len)
                before = " ".join(tokens[start:i])
                end = min(i+window_len + 1, len(tokens))
                after = " ".join(tokens[i+1:end])
                context = before + " <player> " +after
                label = {"Player": full_name, "Skin Tone": players[p]["Skin Tone"], "Teams": teams, "Year": year}
                mention = {"Label" : label, "Mention": context}
                all_mentions.append(mention)
            elif(tokens[i] == last and tokens[i-1] != first):
                start = max(0, i- window_len)
                before = " ".join(tokens[start:i])
                end = min(i+window_len + 1, len(tokens))
                after = " ".join(tokens[i+1:end])
                context = before + " <player> " +after
                label = {"Player": full_name, "Skin Tone": players[p]["Skin Tone"], "Teams": teams, "Year": year}
                mention = {"Label" : label, "Mention": context}
                all_mentions.append(mention)
                    
    readFile("ASI Data/NBA_Transcripts/1:13:25 - Lakers vs Spurs.txt", "Spurs", "Lakers", 2025)
    findMentions(players, 5, teams, year)
    readFile("ASI Data/NBA_Transcripts/1:2:24 - Warriors vs 76ers.txt", "Warriors", "76ers", 2024)
    findMentions(players, 5, teams, year)
    readFile("ASI Data/NBA_Transcripts/1:2:25 - Lakers vs Blazers.txt", "Lakers", "Blazers", 2025)
    findMentions(players, 5, teams, year)
    readFile("ASI Data/NBA_Transcripts/1:3:25 - Knicks vs Thunder.txt", "Knicks", "Thunder", 2025)
    findMentions(players, 5, teams, year)
    readFile("ASI Data/NBA_Transcripts/1:3:25 - Rockets vs Celtics.txt", "Rockets", "Celtics", 2025)
    findMentions(players, 5, teams, year)
    readFile("ASI Data/NBA_Transcripts/1:7:25 - Lakers vs Mavericks.txt", "Lakers", "Mavericks", 2025)
    findMentions(players, 5, teams, year)
    readFile("ASI Data/NBA_Transcripts/1:7:25 - Lakers vs Mavericks.txt", "Lakers", "Mavericks", 2025)
    findMentions(players, 5, teams, year)

    "could n't": "could not",
        "are n't": "are not",
        "did n't": "did not",
        "does n't": "does not",
        "had n't": "had not",
        "i 'm": "i am",
        "it 'll": "it will",
        "it 's": "it is",
        "let 's": "let us",
        "here's": "here is",
        "that 's": "that is",
        "he 's": "he is",
        "you 're": "you are",
        "do n't": "do not",
        "he 'll": "he will",
        "has n't": "has not",
        "there 's": "there is",
        "was n't": "was not",
        "ca n't": "can not",
        "is n't": "is not",
        "wo n't": "will not"
'''


