#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Sun Aug 25 21:52:19 2024

@author: arjunmukherjee
'''

from nltk.tokenize import word_tokenize

import json
import re
import os
from collections import Counter

#import nltk
#nltk.download('punkt_tab')

with open("Data/NBA_Rosters.json", "r") as file:
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

#Replaces other names in the mention with <teamate> or <oponent>
def replaceOtherPlayerNames(context, current_player, all_player_names, teams):
    current_variants = extract_name_variants(current_player)

    if current_player in [k.lower() for k in rosters[teams[0]].keys()]:
        current_team, opponent_team = teams[0], teams[1]
    else:
        current_team, opponent_team = teams[1], teams[0]

    for player in all_player_names:
        player_lower = player.lower()

        #Replaces the player mentioned's name
        for variant in current_variants:
            pattern = r'\b' + re.escape(variant) + r'\b'
            context = re.sub(pattern, '<mentioned_player>', context)
        
        player_variants = extract_name_variants(player_lower)

        #Replaces other players names
        for variant in player_variants:
            pattern = r'\b' + re.escape(variant) + r'\b'

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
            
            


def collectAllTranscripts(window_length):
    folder_path = "Data/Transcripts/"

    pattern = r"(\d+:\d+:\d+) - (\w+) vs (\w+).txt"
    for file_name in os.listdir(folder_path):
        match = re.match(pattern, file_name)
        if match:
            date, team1, team2 = match.groups()
            year = int(date.split(":")[-1]) + 2000  #Converts "25" to "2025"
            file_path = os.path.join(folder_path, file_name)
            readFile(file_path, team1, team2, year)
            findMentions(players, window_length, teams, year)

collectAllTranscripts(10)


output_file = "Data/train_data_10.json"
with open(output_file, "w") as json_file:
    json.dump(all_mentions, json_file, indent=4)
