#Games matrix
import numpy as np
import pulp
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from plotly.offline import plot


def games_matrix(df):
    """
    Create binary constraint matrices for DraftKings lineup optimization.
    
    This function converts a DataFrame of players into several binary matrices that can be used
    as constraints in a linear programming optimization problem for creating optimal fantasy
    football lineups. Each matrix represents different constraints or player attributes.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing player data with columns:
        - 'Game': Game identifier (e.g., "CIN@CLE")
        - 'Team': Player's team abbreviation
        - 'Opp': Opponent team abbreviation  
        - 'Pos': Player position (QB, RB, WR, TE, DST)
        - 'Salary': Player salary
        - 'Proj FP': Projected fantasy points
        
    Returns
    -------
    tuple
        A tuple containing 10 numpy arrays:
        
        Games : numpy.ndarray, shape (nplayers, ngames)
            Binary matrix where Games[i,j] = 1 if player i is in game j, 0 otherwise.
            Used to enforce game stacking constraints.
            
        Opps : numpy.ndarray, shape (nplayers, nteams) 
            Binary matrix where Opps[i,j] = 1 if player i's opponent is team j, 0 otherwise.
            Used to enforce opponent-based constraints.
            
        Teams : numpy.ndarray, shape (nplayers, nteams)
            Binary matrix where Teams[i,j] = 1 if player i is on team j, 0 otherwise.
            Used to enforce team-based constraints.
            
        RB : numpy.ndarray, shape (nplayers,)
            Binary vector where RB[i] = 1 if player i is a running back, 0 otherwise.
            Used to enforce position constraints.
            
        WR : numpy.ndarray, shape (nplayers,)
            Binary vector where WR[i] = 1 if player i is a wide receiver, 0 otherwise.
            Used to enforce position constraints.
            
        QB : numpy.ndarray, shape (nplayers,)
            Binary vector where QB[i] = 1 if player i is a quarterback, 0 otherwise.
            Used to enforce position constraints.
            
        TE : numpy.ndarray, shape (nplayers,)
            Binary vector where TE[i] = 1 if player i is a tight end, 0 otherwise.
            Used to enforce position constraints.
            
        DST : numpy.ndarray, shape (nplayers,)
            Binary vector where DST[i] = 1 if player i is a defense/special teams, 0 otherwise.
            Used to enforce position constraints.
            
        Salary : numpy.ndarray, shape (nplayers,)
            Vector containing each player's salary.
            Used in salary cap constraints.
            
        Proj : numpy.ndarray, shape (nplayers,)
            Vector containing each player's projected fantasy points.
            Used as the objective function to maximize.
    
    Notes
    -----
    The function creates dictionaries to map team names, opponent names, and game names
    to integer indices for efficient matrix operations. It assumes that the number of
    unique teams equals the number of unique opponents (nteams == nopp).
    
    These matrices are typically used with linear programming solvers to find optimal
    lineups that satisfy DraftKings roster construction rules while maximizing projected points.
    """
    ngames = len(df['Game'].unique())
    nteams = len(df['Team'].unique())
    nopp = len(df['Opp'].unique())
    nplayers = len(df)
    assert nteams == nopp
    print(f"{ngames} games between {nteams} teams, {nplayers} players")

    teams_dict = {}
    for cnt, team in enumerate(df['Team'].unique()):
        teams_dict[team] = cnt
    opps_dict = {}
    for cnt, team in enumerate(df['Opp'].unique()):
        opps_dict[team] = cnt
    games_dict = {}   
    for cnt, game in enumerate(df['Game'].unique()):
        games_dict[game] = cnt
        
    Games, Opps, Teams = np.zeros((nplayers, ngames)), np.zeros((nplayers, nteams)),np.zeros((nplayers, nteams))
    RB, WR, QB, TE, DST  = np.zeros(nplayers),np.zeros(nplayers), np.zeros(nplayers), np.zeros(nplayers), np.zeros(nplayers)
    Salary, Proj = np.zeros(nplayers), np.zeros(nplayers)

    for index,row in df.iterrows():
        RB[index] = int(row.Pos=='RB')
        WR[index] = int(row.Pos=='WR')
        QB[index] = int(row.Pos=='QB')
        TE[index] = int(row.Pos=='TE')
        DST[index] = int(row.Pos=='DST')
        Salary[index] = row.Salary
        Proj[index] = row['Proj FP']
        ind_game = games_dict[row.Game]
        ind_opp =  teams_dict[row.Opp]
        ind_team = teams_dict[row.Team]
        
        Games[index,ind_game] =1 
        Teams[index,ind_team] =1 
        Opps[index,ind_opp] =1 
    return Games, Opps, Teams, RB, WR, QB, TE, DST, Salary, Proj

def compute_lineups(df, nlineups, noverlap, max_use, qb_stack):
    """
    Generate multiple DraftKings fantasy football lineups using linear programming optimization.
    
    This function iteratively solves for optimal lineups by maximizing projected fantasy points
    while satisfying DraftKings roster construction rules and diversity constraints. Each new
    lineup is constrained to have limited overlap with previously generated lineups.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing player data with columns:
        - 'Game': Game identifier (e.g., "CIN@CLE")
        - 'Team': Player's team abbreviation
        - 'Opp': Opponent team abbreviation  
        - 'Pos': Player position (QB, RB, WR, TE, DST)
        - 'Salary': Player salary
        - 'Proj FP': Projected fantasy points
        
    nlineups : int
        Number of lineups to generate.
        
    noverlap : int
        Maximum number of players that can overlap between any two lineups.
        Used to ensure lineup diversity.
        
    max_use : int
        Maximum number of times any individual player can be used across all lineups.
        Prevents over-exposure to specific players.
        
    qb_stack : int
        Parameter controlling QB-receiver stacking strategy. Higher values encourage
        more aggressive stacking of QB with WR/TE from the same team.
        
    Returns
    -------
    numpy.ndarray, shape (nlineups, nplayers)
        Binary matrix where X[i,j] = 1 if player j is in lineup i, 0 otherwise.
        Each row represents one complete lineup.
        
    Notes
    -----
    The optimization enforces the following DraftKings roster rules:
    - Exactly 9 players total
    - 1 QB, 2 RB, 4 WR, 1 TE, 1 DST
    - Salary cap of $50,000
    - At least 2 different games represented
    - At least 2 different teams represented
    - QB-receiver stacking constraints
    - No DST vs opposing offense
    - No QB and RB on same team
    - No more than 2 RB/WR/TE on same team
    
    The function uses PuLP linear programming solver to find optimal solutions.
    Each lineup is solved independently with constraints based on previously
    generated lineups to ensure diversity.
    
    Examples
    --------
    >>> df = load_data(salary_file, offense_file, defense_file)
    >>> lineups = compute_lineups(df, nlineups=10, noverlap=6, max_use=3, qb_stack=8)
    >>> print(f"Generated {lineups.shape[0]} lineups with {lineups.shape[1]} players each")
    """
    Games, Opps, Teams, RB, WR, QB, TE, DST, Salary, Proj = games_matrix(df)
    nplayers = Games.shape[0]
    ngames = Games.shape[1]
    nteams = Teams.shape[1]
    X = np.zeros((nplayers, nlineups))  #matrix of lineups

    for lineup in range(nlineups):
        prob = pulp.LpProblem(f"DK_Lineup_{lineup}", pulp.LpMaximize)

        # Player variables
        xplayer = [pulp.LpVariable(f'player_{i:03}',cat="Binary") for i in range(nplayers)]
        #game variables
        xgame = [pulp.LpVariable(f'game_{i:03}',cat="Binary") for i in range(ngames)]
        #team variables
        xteam = [pulp.LpVariable(f'team_{i:03}',cat="Binary") for i in range(nteams)]
        #qb-receiver stack variables
        xqb_rec = [pulp.LpVariable(f'qb_rec_{i:03}', cat="Binary") for i in range(nteams)]
        
        # Define the objective function: Proj FP * x
        objective = pulp.lpSum([Proj[i] * xplayer[i] for i in range(nplayers)])
        prob += objective

        #Define Position constraints: Pos* x= pi
        prob += (pulp.lpSum([RB[i] * xplayer[i] for i in range(nplayers)]) == 2, "RB")
        #prob += (pulp.lpSum([RB[i] * xplayer[i] for i in range(nplayers)]) <= 2, "RB_upper")
        prob += (pulp.lpSum([WR[i] * xplayer[i] for i in range(nplayers)]) == 4, "WR")    
        #prob += (pulp.lpSum([WR[i] * xplayer[i] for i in range(nplayers)]) <= 4, "WR_UPPER")   
        prob += (pulp.lpSum([TE[i] * xplayer[i] for i in range(nplayers)]) == 1, "TE")    
        #prob += (pulp.lpSum([TE[i] * xplayer[i] for i in range(nplayers)]) <= 1, "TE_UPPER")       
        prob += (pulp.lpSum([QB[i] * xplayer[i] for i in range(nplayers)]) == 1, "QB")
        prob += (pulp.lpSum([DST[i] * xplayer[i] for i in range(nplayers)]) == 1, "DST")
        
        #Define total players constraint: 1*x = 9
        prob += (pulp.lpSum([xplayer[i] for i in range(nplayers)]) == 9, "9_players")

        #Define salary constriant: Salary*x <=50000
        prob += (pulp.lpSum([Salary[i] * xplayer[i] for i in range(nplayers)]) <= 50000, "Salary")
        
        # 2 different games constraints
        for j in range(ngames):
            prob += (pulp.lpSum([Games[i,j] * xplayer[i] for i in range(nplayers)]) >= xgame[j], f"Game_{j}")
        prob += (pulp.lpSum([xgame[i] for i in range(ngames)]) >= 2, f"Games")
        
    # 2 different teams constraints
        for j in range(nteams):
            prob += (pulp.lpSum([Teams[i,j] * xplayer[i] for i in range(nplayers)]) >= xteam[j], f"Team_{j}")
        prob += (pulp.lpSum([xteam[i] for i in range(nteams)]) >= 2, f"Teams")  
        
        #QB - WR Stack - Put 1 QB with 2 WR 
        for j in range(nteams):
            prob += (pulp.lpSum([((8-qb_stack)*QB[i] + WR[i])*Teams[i,j] * xplayer[i] for i in range(nplayers)]) >= 8*xqb_rec[j]
                    , f"qb_rec_stack_{j}")
        prob += (pulp.lpSum([xqb_rec[i] for i in range(nteams)]) >= 1, f"qb_rec_stacks")
        
        #NO DST VS Offense  constraint
        for j in range(nteams):
            prob += (pulp.lpSum([((1-DST[i])*Teams[i,j] + 8*DST[i]*Opps[i,j]) * xplayer[i] for i in range(nplayers)]) <=8, 
                    f"No_DST_Offense_{j}")
        
        #No QB and RB on same team, no RB and RB on same team
        for j in range(nteams):
            prob += (pulp.lpSum([(QB[i] + RB[i])*Teams[i,j] * xplayer[i] for i in range(nplayers)]) <=1, 
                    f"No_DST_Offense_Team_{j}")
        
        #No WR/TE and RB on same team
        for j in range(nteams):
            prob += (pulp.lpSum([(2*RB[i] + WR[i] + TE[i])*Teams[i,j] * xplayer[i] for i in range(nplayers)]) <=2, 
                    f"No_RB_Rec_Team_{j}")
        
        #CAP PlAYER USAGE IN LINEUPS    
        if lineup >=1:        
            for j in range(nplayers):
                z = X[j,:]
                prob += (pulp.lpSum([z[i]  for i in range(lineup)])* xplayer[j] <= max_use, f"Max_use_player_{j}")
        
        #Overlap constraints: X[:,j]*x <=noverlap
        if lineup >=1:        
            for j in range(lineup):
                #print(f"\tLineup {lineup}:{j} overlap constraint")
                z = X[:,j]
                prob += (pulp.lpSum([z[i] * xplayer[i] for i in range(nplayers)]) <= noverlap, f"Overlap_{lineup}_{j}")
        
        ########################################################################
        # Solve the problem
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        new_lineup = np.array( [v.varValue for v in prob.variables() if 'player_' in str(v)])
    
        X[:,lineup] = new_lineup 
        lineup_salary = df.iloc[new_lineup==1]['Salary'].sum()

        # Print the optimal objective value
        print(f"Lineup {lineup+1}: {pulp.value(prob.objective):.2f} points, ${lineup_salary:,}")
    X=X.T
    return X

def create_lineup_dashboard(df_lineups, output_file="draftkings_lineup_dashboard.html"):
    """
    Create an interactive HTML dashboard with separate tabs for each position showing player usage across lineups for DraftKings NFL Millionaire Maker.
    
    Parameters
    ----------
    df_lineups : pandas.DataFrame
        DataFrame containing lineup data with columns for each position (QB, RB, RB.1, WR, WR.1, WR.2, TE, DST, FLEX)
        
    output_file : str, optional
        Name of the output HTML file. Default is "draftkings_lineup_dashboard.html"
        
    Returns
    -------
    str
        Path to the generated HTML file
    """
    
    # Define the black and pink theme colors
    colors = {
        'background': '#0a0a0a',
        'text': '#ff69b4',
        'grid': '#333333',
        'bar': '#ff1493',
        'bar_secondary': '#ff69b4',
        'tab_active': '#ff1493',
        'tab_inactive': '#333333'
    }
    
    # Position mapping for display
    position_mapping = {
        'QB': 'Quarterback',
        'RB': 'Running Back 1', 
        'RB.1': 'Running Back 2',
        'WR': 'Wide Receiver 1',
        'WR.1': 'Wide Receiver 2', 
        'WR.2': 'Wide Receiver 3',
        'TE': 'Tight End',
        'DST': 'Defense/Special Teams',
        'FLEX': 'Flex'
    }
    
    positions = ['QB', 'RB', 'RB.1', 'WR', 'WR.1', 'WR.2', 'TE', 'DST', 'FLEX']
    
    # Create individual plots for each position
    plot_data = {}
    for pos in positions:
        value_counts = df_lineups[pos].value_counts()
        
        # Debug: print value counts for first position
        if pos == positions[0]:
            print(f"Debug - {pos} value counts:")
            print(value_counts.head())
        
        # Sort by values (highest first) for proper y-axis ordering
        value_counts_sorted = value_counts.sort_values(ascending=True)  # ascending=True puts highest at top
        
        fig = go.Figure(data=[
            go.Bar(
                x=value_counts_sorted.values.tolist(),  # Convert to list
                y=value_counts_sorted.index.tolist(),   # Convert to list
                orientation='h',
                marker_color=colors['bar'],
                hovertemplate=f'<b>{position_mapping[pos]}</b><br>' +
                             'Player: %{y}<br>' +
                             'Usage: %{x} lineups<br>' +
                             '<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title={
                'text': f'{position_mapping[pos]} - Player Usage',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': colors['text']}
            },
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font=dict(color=colors['text'], size=12),
            height=600,
            width=1000,
            margin=dict(l=50, r=50, t=80, b=50),
            xaxis=dict(
                gridcolor=colors['grid'],
                zerolinecolor=colors['grid'],
                showgrid=True,
                color=colors['text'],
                title_font=dict(color=colors['text']),
                tickfont=dict(color=colors['text']),
                title="Number of Lineups"
            ),
            yaxis=dict(
                gridcolor=colors['grid'],
                zerolinecolor=colors['grid'],
                showgrid=True,
                color=colors['text'],
                title_font=dict(color=colors['text']),
                tickfont=dict(color=colors['text']),
                title="Players"
            )
        )
        
        # Convert to JSON string for JavaScript
        plot_data[pos] = fig.to_json()
    
    # Create individual HTML files for each position and combine them
    import json
    
    # Create the main HTML with tabs
    html_string = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>DraftKings NFL Millionaire Maker - Lineup Analysis</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                background-color: {colors['background']};
                color: {colors['text']};
                font-family: 'Arial', sans-serif;
                margin: 0;
                padding: 0;
            }}
            .header {{
                text-align: center;
                padding: 20px;
                background-color: {colors['background']};
                border-bottom: 2px solid {colors['text']};
            }}
            .tab-container {{
                display: flex;
                background-color: {colors['background']};
                border-bottom: 1px solid {colors['grid']};
                padding: 0 20px;
                overflow-x: auto;
            }}
            .tab {{
                background-color: {colors['tab_inactive']};
                color: {colors['text']};
                border: none;
                padding: 15px 25px;
                cursor: pointer;
                font-size: 14px;
                font-weight: bold;
                border-radius: 8px 8px 0 0;
                margin-right: 5px;
                transition: all 0.3s ease;
                white-space: nowrap;
            }}
            .tab:hover {{
                background-color: {colors['grid']};
            }}
            .tab.active {{
                background-color: {colors['tab_active']};
                color: {colors['background']};
            }}
            .tab-content {{
                display: none;
                padding: 20px;
                background-color: {colors['background']};
                min-height: 70vh;
            }}
            .tab-content.active {{
                display: block;
            }}
            .plot-container {{
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 60vh;
            }}
            .plotly-graph-div {{
                background-color: {colors['background']} !important;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1 style="color: {colors['text']}; font-size: 28px; margin-bottom: 10px;">
                🏈 DraftKings NFL Millionaire Maker - Lineup Analysis 🏈
            </h1>
            <p style="color: {colors['text']}; font-size: 16px; margin-bottom: 0;">
                Interactive dashboard showing player usage across {len(df_lineups)} generated lineups
            </p>
        </div>
        
        <div class="tab-container">
            {''.join([f'<button class="tab" onclick="showTab(\'{pos}\')" id="tab-{pos}">{position_mapping[pos]}</button>' for pos in positions])}
        </div>
        
        {''.join([f'''
        <div class="tab-content" id="content-{pos}">
            <div class="plot-container">
                <div id="plot-{pos}"></div>
            </div>
        </div>''' for pos in positions])}
        
        <script>
            // Plot data for each position
            const plotData = {json.dumps(plot_data)};
            
            // Show first tab by default
            showTab('{positions[0]}');
            
            function showTab(position) {{
                // Hide all tab contents
                document.querySelectorAll('.tab-content').forEach(content => {{
                    content.classList.remove('active');
                }});
                
                // Remove active class from all tabs
                document.querySelectorAll('.tab').forEach(tab => {{
                    tab.classList.remove('active');
                }});
                
                // Show selected tab content
                document.getElementById('content-' + position).classList.add('active');
                document.getElementById('tab-' + position).classList.add('active');
                
                // Render plot if not already rendered
                const plotDiv = document.getElementById('plot-' + position);
                if (!plotDiv.hasAttribute('data-rendered')) {{
                    try {{
                        const data = JSON.parse(plotData[position]);
                        Plotly.newPlot(plotDiv, data.data, data.layout, {{responsive: true}});
                        plotDiv.setAttribute('data-rendered', 'true');
                    }} catch (error) {{
                        console.error('Error rendering plot for', position, ':', error);
                        plotDiv.innerHTML = '<p style="color: #ff69b4; text-align: center;">Error loading plot data</p>';
                    }}
                }}
            }}
        </script>
    </body>
    </html>
    """
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_string)
    
    print(f"Interactive dashboard with tabs saved as: {output_file}")
    return output_file