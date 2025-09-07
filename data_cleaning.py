
import numpy as np
import pandas as pd
import glob

def create_filenames(path):
    """
    Create and locate all necessary filenames for DraftKings fantasy football analysis.
    
    This function constructs file paths for required CSV files and automatically searches
    for DFN (Daily Fantasy Nerd) projection files using glob patterns. It handles both
    offense and defense projection files by searching for files that match the expected
    naming conventions.
    
    Parameters
    ----------
    path : str
        The directory path where all the CSV files are located.
        
    Returns
    -------
    tuple
        A tuple containing five file paths in the following order:
        - fname_salary (str): Path to DKSalaries.csv file
        - fname_entries (str): Path to DKEntries.csv file  
        - fname_lineups (str): Path to lineups.csv file
        - fname_proj_offense (str): Path to DFN NFL Offense projection file
        - fname_proj_defense (str): Path to DFN NFL Defense projection file
        
    Raises
    ------
    FileNotFoundError
        If either the DFN NFL Offense or DFN NFL Defense projection files
        cannot be found in the specified directory.
        
    Notes
    -----
    The function uses glob patterns to find projection files:
    - Offense files: "DFN NFL Offense*.csv"
    - Defense files: "DFN NFL Defense*.csv"
    
    If multiple files match the pattern, the first one found is used.
    """
    
    fname_entries = f"{path}/DKEntries.csv"  #file where your lineup entries will be saved
    fname_lineups = f"{path}/lineups.csv"

    # Automatically find DFN projection files
    # Find DFN NFL Offense file
    offense_files = glob.glob(f"{path}/DFN NFL Offense*.csv")
    if offense_files:
        fname_proj_offense = offense_files[0]
        print(f"Found offense projections: {fname_proj_offense}")
    else:
        fname_proj_offense = None
        raise FileNotFoundError(f"No DFN NFL Offense file found in {path}")

    # Find DFN NFL Defense file
    defense_files = glob.glob(f"{path}/DFN NFL Defense*.csv")
    if defense_files:
        fname_proj_defense = defense_files[0]
        print(f"Found defense projections: {fname_proj_defense}")
    else:
        fname_proj_defense = None
        raise FileNotFoundError(f"No DFN NFL Defense file found in {path}")
    
    salary_files = glob.glob(f"{path}/DKSalaries*.csv")
    if salary_files:
        fname_salary = salary_files[0]
        print(f"Found salary file: {fname_salary}")
    else:
        fname_salary = None
        raise FileNotFoundError(f"No DKSalaries file found in {path}")
        
    return fname_salary, fname_entries, fname_lineups, fname_proj_offense, fname_proj_defense
        

def load_data_showdown(fname_salary, fname_proj_offense, fname_proj_defense):
    ''' Load data for NFL Showdown contests '''
    df_salary = pd.read_csv(fname_salary, skiprows=7, usecols=range(10, 19))
    df_proj_off = pd.read_csv(fname_proj_offense)
    df_proj_def = pd.read_csv(fname_proj_defense)

    df_salary = df_salary[df_salary['Roster Position']=='FLEX']

    df = pd.concat([df_proj_off, df_proj_def])
    df = df[df['Proj FP']>0]
    df['Team'] =  df['Team'].str.replace("@","")
    df['Opp'] =  df['Opp'].str.replace("@","")
    df = df.reset_index(drop=True)
    df = df.rename(columns={"Player Name": "Name"})

    # Function to create the sorted game name
    def create_sorted_game(row):
        teams = [row['Team'], row['Opp']]
        teams.sort()  # Sort the team names alphabetically
        return f"{teams[0]}@{teams[1]}"

    # Apply the function to create the "Game" column
    df['Game'] = df.apply(create_sorted_game, axis=1)

    #replace team name with abbrev for Defenses
    df_salary.loc[df_salary['Position'] == 'DST', 'Name'] = df_salary.loc[df_salary['Position'] == 'DST', 'TeamAbbrev']
    df.loc[df['Pos']=='D','Pos'] = 'DST'
    df.loc[df['Pos'] == 'DST', 'Name'] = df.loc[df['Pos'] == 'DST', 'Team']

    df_salary = df_salary.rename(columns={"Position": "Pos"})

    df_merge = pd.merge(df, df_salary, on=["Name",'Pos'])
    cols = ['Name','Name + ID', 'Pos','Roster Position','Salary_y','Team','Opp','Game','Proj FP']
    df_merge = df_merge[cols]
    df_merge = df_merge.rename(columns={"Salary_y": "Salary"})

    return df_merge


def load_data(fname_salary, fname_proj_offense, fname_proj_defense):
    ''' Load data for standard NFL contests '''
    df_salary = pd.read_csv(fname_salary, skiprows=7, usecols=range(10, 19))
    df_proj_off = pd.read_csv(fname_proj_offense)
    df_proj_def = pd.read_csv(fname_proj_defense)


    df = pd.concat([df_proj_off, df_proj_def])
    df = df[df['Proj FP']>0]
    df['Team'] =  df['Team'].str.replace("@","")
    df['Opp'] =  df['Opp'].str.replace("@","")
    df = df.reset_index(drop=True)

    # Function to create the sorted game name
    def create_sorted_game(row):
        teams = [row['Team'], row['Opp']]
        teams.sort()  # Sort the team names alphabetically
        return f"{teams[0]}@{teams[1]}"

    # Apply the function to create the "Game" column
    df['Game'] = df.apply(create_sorted_game, axis=1)
    df = df.rename(columns={"Player Name": "Name"})

    #replace team name with abbrev for Defenses
    df_salary.loc[df_salary['Position'] == 'DST', 'Name'] = df_salary.loc[df_salary['Position'] == 'DST', 'TeamAbbrev']
    df.loc[df['Pos']=='D','Pos'] = 'DST'
    df.loc[df['Pos'] == 'DST', 'Name'] = df.loc[df['Pos'] == 'DST', 'Team']

    # Filter df to only include teams that are in df_salary (Sunday games only)
    # Get unique team abbreviations from df_salary
    salary_teams = set(df_salary['TeamAbbrev'].unique())
    
    # Filter df to only include teams that are in the salary file
    df = df[df['Team'].isin(salary_teams) | df['Opp'].isin(salary_teams)]

    df_merge = pd.merge(df, df_salary, on=["Name", "Salary",])
    cols = ['Name','Name + ID', 'Pos','Salary','Team','Opp','Game','Proj FP']
    df_merge = df_merge[cols]

    return df_merge

def load_data_actual(fname_salary, fname_proj_offense, fname_proj_defense):
    ''' Load data with projections and actual points (after the contest) '''
    df_salary = pd.read_csv(fname_salary)
    df_proj_off = pd.read_csv(fname_proj_offense)
    df_proj_def = pd.read_csv(fname_proj_defense)

    df = pd.concat([df_proj_off, df_proj_def])
    df = df[df['Proj FP']>0]
    df['Team'] =  df['Team'].str.replace("@","")
    df['Opp'] =  df['Opp'].str.replace("@","")
    df = df.reset_index(drop=True)

    # Function to create the sorted game name
    def create_sorted_game(row):
        teams = [row['Team'], row['Opp']]
        teams.sort()  # Sort the team names alphabetically
        return f"{teams[0]}@{teams[1]}"

    # Apply the function to create the "Game" column
    df['Game'] = df.apply(create_sorted_game, axis=1)
    df = df.rename(columns={"Player Name": "Name"})

    #replace team name with abbrev for Defenses
    df_salary.loc[df_salary['Position'] == 'DST', 'Name'] = df_salary.loc[df_salary['Position'] == 'DST', 'TeamAbbrev']
    df.loc[df['Pos']=='D','Pos'] = 'DST'
    df.loc[df['Pos'] == 'DST', 'Name'] = df.loc[df['Pos'] == 'DST', 'Team']


    df_merge = pd.merge(df, df_salary, on=["Name", "Salary",])
    cols = ['Name + ID', 'Pos','Salary','Team','Opp','Game','Proj FP','Actual FP']
    df_merge = df_merge[cols]

    df = df_merge
    return df

def lineup_pts(df_lineups, df, pts_type = "Actual FP"):
    ''' Calculate the actual or projected points for lineups in df_linueps using fantasy points in df'''
    assert (pts_type == 'Actual FP') or (pts_type == 'Proj FP')
    name_to_fp = df.set_index('Name + ID')['Actual FP'].to_dict()

    # Step 2: Replace "Name + ID" with "Actual FP" in df_lineups
    df_lineups_pts = df_lineups.replace(name_to_fp)

    # Step 3: Add the sum of each row to a new column "Actual FP"
    df_lineups_pts[pts_type] = df_lineups_pts.sum(axis=1)
    return df_lineups_pts


def lineup_string(x,df):
    ''' Filter players based on the binary array x to get their Name + ID and put in a string'''
    selected_players = df[pd.Series(x) == 1]

    # Sort players based on the sequence
    sequence = ['QB', 'RB', 'RB', 'WR', 'WR', 'WR', 'TE', 'DST']
    QB = selected_players[selected_players.Pos=='QB']
    DST = selected_players[selected_players.Pos=='DST']
    RB = selected_players[selected_players.Pos=='RB'].iloc[0:2]
    WR = selected_players[selected_players.Pos=='WR'].iloc[0:3]
    TE = selected_players[selected_players.Pos=='TE'].iloc[[0]]

    sorted_players = pd.concat([QB, RB,WR,TE, DST ])
    merged_df = pd.merge(sorted_players, selected_players, on='Name + ID', how='outer', indicator=True)
    result = merged_df[merged_df['_merge'] == 'right_only'].drop(columns=['_merge'])

    FLEX = df[df['Name + ID']==result['Name + ID'].values[0]]

    lineup = pd.concat([QB, RB,WR,TE, FLEX,DST, ])

    lineup_str = ",".join(lineup['Name + ID'].values)
    
    return lineup_str

def write_lineups(X, df, fname_lineups):
    ''' save lineups in binary matrix X, data in dataframe df, to file fname_lineups'''
    output_str = "QB,RB,RB,WR,WR,WR,TE,FLEX,DST\n"
    for i in range(X.shape[0]):
        x = X[i,:]
        lineup_str = lineup_string(x,df)
        output_str += lineup_str + '\n'

    with open(fname_lineups, 'w') as file:
        file.write(output_str)    

    l = pd.read_csv(fname_lineups)
    return l 

def update_entries(fname_entries, df_lineups):
    ''' Update DraftKings entry file with new lineups '''
    nlineups = len(df_lineups)
    df_entries = pd.read_csv(fname_entries, skiprows=0, usecols=range(0, 12))
    print(f"df_entries: {df_entries.head()}")
    df_entries = df_entries.iloc[0:nlineups]
    df_entries = df_entries[df_entries.columns[0:13]]
    cols =  ['QB', 'RB', 'RB', 'WR', 'WR', 'WR', 'TE', 'FLEX','DST']
    cols1 =  ['QB', 'RB', 'RB.1', 'WR', 'WR.1','WR.2', 'TE', 'FLEX','DST']
    df_entries[cols1] = df_lineups
    df_entries.columns = ['Entry ID','Contest Name','Contest ID','Entry Fee', 'QB', 'RB', 'RB', 'WR', 'WR', 'WR', 'TE', 'FLEX', 'DST']
    df_entries.to_csv(fname_entries, index=False)
    return df_entries

def lineup_string_showdown(x,cpt,df):
    ''' Filter players based on the binary arrays x and cpt to get their Name + ID and put in a string'''
    selected_flex = df[(pd.Series(x) == 1) & (pd.Series(cpt) == 0)]
    selected_captain = df[pd.Series(cpt) == 1]
    # Sort players based on the sequence
    sequence = ['CPT', 'FLEX', 'FLEX','FLEX','FLEX','FLEX']   
    lineup = pd.concat((selected_captain, selected_flex))
    lineup_str = ",".join(lineup['Name + ID'].values)
    return lineup_str


def write_lineups_showdown(X,CPT, df, fname_lineups):
    ''' save lineups in binary matrixes X,CPT in dataframe df, to file fname_lineups'''
    output_str = "CPT,FLEX,FLEX,FLEX,FLEX,FLEX\n"
    for i in range(X.shape[0]):
        x = X[i,:]
        cpt = CPT[i,:]
        lineup_str = lineup_string_showdown(x, cpt, df)
        output_str += lineup_str + '\n'

    with open(fname_lineups, 'w') as file:
        file.write(output_str)    

    l = pd.read_csv(fname_lineups)
    return l 

def update_entries_showdown(fname_entries, df_lineups):
    ''' Update DraftKings Showdown entry file with new lineups '''
    nlineups = len(df_lineups)
    df_entries = pd.read_csv(fname_entries)
    df_entries = df_entries.iloc[0:nlineups]
    df_entries = df_entries[df_entries.columns[0:13]]
    cols =  ['QB', 'RB', 'RB', 'WR', 'WR', 'WR', 'TE', 'FLEX','DST']
    cols1 =  ['CPT', 'FLEX', 'FLEX.1', 'FLEX.2', 'FLEX.3', 'FLEX.4']
    df_entries[cols1] = df_lineups
    df_entries.columns = ['Entry ID','Contest Name','Contest ID','Entry Fee', 'CPT', 'FLEX', 'FLEX', 'FLEX', 'FLEX', 'FLEX']
    df_entries.to_csv(fname_entries, index=False)
    return df_entries
