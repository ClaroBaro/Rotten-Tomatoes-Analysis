"""
Fitting the beta prior model on historical movie data.
"""

import sys
from pathlib import Path

import pandas as pd

# anchor paths to this file's location, not the working directory
_HERE         = Path(__file__).parent          # src/usage/
_PROJECT_ROOT = _HERE.parent.parent            # Rotten Tomatoes/

sys.path.insert(0, str(_HERE.parent))          # puts src/ on the path
from beta_prior_model import BetaPriorModel    # noqa: E402

movies = pd.read_csv(_PROJECT_ROOT / 'data' / 'movies.csv')
model = BetaPriorModel()
model.fit(movies)
model.save(str(_HERE / 'beta_prior_model.pkl'))