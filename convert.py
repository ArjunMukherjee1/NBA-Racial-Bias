import pandas as pd
import json
# Load the CSV
'''df = pd.read_csv("ASI Data/NBA_rosters.csv")

#Allow players with multiple teams
df["Team"] = df["Team"].str.split("/")
df = df.explode("Team")
# Convert to JSON grouped by teams
result = df.groupby("Team").apply(
    lambda x: x.set_index("Player Name")[["Skin Tone"]].to_dict("index")
).to_dict()

# Save to JSON
with open("ASI Data/NBA_Rosters.json", "w") as f:
    json.dump(result, f, indent=4)'''
# Load the CSV file
df = pd.read_csv("ASI Data/NBA_rosters.csv")

# If any player belongs to multiple teams (e.g., "76ers/Clippers"),
# split the 'Team' column on "/" and create separate rows
df["Team"] = df["Team"].str.split("/")
df = df.explode("Team")
df["Team"] = df["Team"].str.strip()  # Clean any extra whitespace

# Create a dictionary with teams as keys and nested dictionaries for players
teams_dict = {}
for _, row in df.iterrows():
    team = row["Team"]
    player = row["Player Name"]
    skin_tone = row["Skin Tone"]
    
    if team not in teams_dict:
        teams_dict[team] = {}
    
    teams_dict[team][player] = {"Skin Tone": skin_tone}

# Save the dictionary to a JSON file with pretty formatting
with open("ASI Data/NBA_Rosters.json", "w") as f:
    json.dump(teams_dict, f, indent=4)




