# Daily-Fantasy-Football-DraftKings

🏈 **Advanced lineup optimization system for DraftKings NFL Millionaire Maker contests** 🏈

This project uses linear programming optimization to generate multiple diverse, high-scoring fantasy football lineups for DraftKings NFL Millionaire Maker contests. The system maximizes projected fantasy points while enforcing DraftKings roster rules and strategic constraints.

## 🎯 What This Code Does

### **Core Functionality:**
- **Generates 150+ optimized lineups** using linear programming (PuLP solver)
- **Maximizes projected fantasy points** while staying under $50,000 salary cap
- **Enforces DraftKings roster rules**: 1 QB, 2 RB, 4 WR, 1 TE, 1 DST
- **Ensures lineup diversity** with overlap constraints between lineups
- **Implements advanced strategies** like QB-receiver stacking and game correlation
- **Creates interactive dashboards** to analyze player usage patterns

### **Key Features:**
- **Smart Data Integration**: Automatically finds and processes DFN projection files
- **Sunday Games Only**: Filters projections to only include DraftKings contest games
- **Anti-Stacking Rules**: Prevents conflicting player combinations
- **Usage Limits**: Caps individual player exposure across all lineups
- **Visual Analytics**: Interactive HTML dashboard with black/pink theme
- **Auto-Export**: Generates DraftKings entry files for easy upload

## 🚀 Quick Start

### **Installation:**
```bash
pip install -r requirements.txt
```

### **Basic Usage:**
```python
from data_cleaning import *
from lineup_optimizer import *

# Set parameters
path = '2025-09-07'  # Directory with your data files
nlineups = 150       # Number of lineups to generate
noverlap = 4         # Max overlap between lineups
max_use = 15         # Max times any player can be used
qb_stack = 1         # QB-receiver stacking parameter

# Generate lineups
fname_salary, fname_entries, fname_lineups, fname_proj_offense, fname_proj_defense = create_filenames(path)
df_merge = load_data(fname_salary, fname_proj_offense, fname_proj_defense)
X = compute_lineups(df_merge, nlineups, noverlap, max_use, qb_stack)
df_lineups = write_lineups(X, df_merge, fname_lineups)

# Create interactive dashboard
create_lineup_dashboard(df_lineups)
```

## 📁 Required Files

Place these files in your date directory (e.g., `2025-09-07/`):
- **`DKSalaries.csv`** - DraftKings salary file (Sunday games only)
- **`DFN NFL Offense*.csv`** - Daily Fantasy Nerd offensive projections
- **`DFN NFL Defense*.csv`** - Daily Fantasy Nerd defensive projections
- **`DKEntries.csv`** - DraftKings entry template (optional)

## 🎮 Optimization Strategy

### **Roster Construction:**
- **Position Requirements**: Exactly 1 QB, 2 RB, 4 WR, 1 TE, 1 DST
- **Salary Cap**: $50,000 maximum
- **Game Diversity**: At least 2 different games represented
- **Team Diversity**: At least 2 different teams represented

### **Advanced Constraints:**
- **QB-Stacking**: Encourages QB + WR/TE from same team
- **Anti-Conflicts**: No QB+RB same team, no DST vs opposing offense
- **Usage Limits**: Prevents over-exposure to individual players
- **Overlap Control**: Limits similarity between generated lineups

### **Diversity Features:**
- **Player Usage Caps**: Max 15 uses per player across all lineups
- **Lineup Overlap**: Max 4 players shared between any two lineups
- **Iterative Generation**: Each lineup considers previous lineups

## 📊 Interactive Dashboard

The system generates a beautiful interactive HTML dashboard featuring:
- **Tabbed Interface**: Separate tabs for each position (QB, RB1, RB2, WR1, WR2, WR3, TE, DST, FLEX)
- **Usage Analytics**: Bar charts showing player frequency across lineups
- **Black/Pink Theme**: Professional DraftKings-inspired design
- **Hover Tooltips**: Detailed player information and usage counts
- **Auto-Open**: Dashboard automatically opens in your browser

## 🔧 File Structure

```
Daily-Fantasy-Football-DraftKings/
├── data_cleaning.py          # Data processing and file management
├── lineup_optimizer.py       # Linear programming optimization
├── test.py                   # Main execution script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── 2025-09-07/              # Date-specific data directory
    ├── DKSalaries.csv
    ├── DFN NFL Offense*.csv
    ├── DFN NFL Defense*.csv
    └── DKEntries.csv
```

## 📈 Output Files

- **`lineups.csv`** - Generated lineups in DraftKings format
- **`DKEntries.csv`** - Updated entry file ready for DraftKings upload
- **`lineups_plot.html`** - Interactive dashboard (auto-opens)

## 🎯 Perfect For

- **DraftKings NFL Millionaire Maker** contests
- **Mass multi-entry** strategies
- **GPP (Guaranteed Prize Pool)** tournaments
- **Lineup optimization** and analysis
- **Player exposure** management

## 📚 Dependencies

- **numpy** - Numerical computations
- **pandas** - Data manipulation
- **pulp** - Linear programming solver
- **plotly** - Interactive visualizations
- **matplotlib/seaborn** - Additional plotting

## 🏆 Strategy Notes

This system is designed for **high-volume GPP play** where you want to:
- Generate many diverse lineups quickly
- Avoid over-exposure to popular players
- Implement advanced stacking strategies
- Analyze player usage patterns
- Maximize projected fantasy points

Perfect for DraftKings NFL Millionaire Maker contests where you need to balance upside potential with lineup diversity!

## Notebooks
DraftKings NFL Lineups: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zlisto/Daily-Fantasy-Football-DraftKings/blob/main/DraftKingsNFLLineups.ipynb)
