import pandas as pd
import numpy as np


def predict_reg(f):
    # Load new new data path, number of variable must match the regression
    dat_new = pd.read_csv(f)
    # Initialise Regression coefficients list to data-frame
    dat = pd.DataFrame([{'Var2': 0.01, 'Var3': 0.022, 'Var7': 0.44,
                       'Var4': 0.55, 'Var5': 0.012, 'Var6': 0.089,
                        'Var1': 0.30}])   # Match variables in a regression model

    # Convert to array with sorted column name

    def array_c(x):
        ar_x = np.array(x.reindex(columns=sorted(x.columns)))
        return ar_x
    # Calculated weighted scores (Vectorised multiplication)
    weighted = pd.DataFrame(np.multiply(array_c(dat_new), array_c(dat)))
    # Convert to array
    # Calculate row sums as model score
    score = pd.DataFrame(weighted.sum(axis=1, skipna=True))
    # Name the column
    score.columns = ["Score"]

    # Define the Score groups

    def score_group(x):
        if x['Score'] <= -1:
            return 'Low'
        elif x['Score'] >= 1:
            return 'High'
        elif x['Score'] > -1 < 1:
            return 'Intermediate'
    #
    score['Risk'] = score.apply(score_group, axis=1)
    return score
# Remove the print after testing


if __name__ == "__main__":
    Res = predict_reg('Teatr.csv')
    Res.to_csv('Res.csv')


