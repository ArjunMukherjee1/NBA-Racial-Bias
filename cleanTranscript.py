#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 20:37:39 2024

@author: arjunmukherjee
"""
import re
transcriptFile = open("Transcript.txt", "r")
transcriptText = transcriptFile.read()
cleanTranscript = re.sub('\\d*:*\\d+:\\d+', ' ', transcriptText)
cleanTranscript = re.sub('\\n', ' ', cleanTranscript)
cleanTranscript = re.sub('\\[[^\\]]*\\]', '', cleanTranscript)
cleanTranscript = re.sub('\\s+', ' ', cleanTranscript)
transcriptFile.close()
newTranscript = open("newTranscript.txt", "w")
newTranscript.write(cleanTranscript)