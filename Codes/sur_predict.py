"""
Created on Mon Apr  1 20:10:16 2019

@author: caanene
"""
import numpy as np
from pandas import DataFrame as Dff
from pandas import read_csv
import argparse


def sur_predict(weight, nexp, outputf):
    """
    Calculates risk groups with features weights from regression type models,
    writes outputf as csv.
    weight: feature weights from a model
    nexp: datafile of new values or cases, must contain column "ID"
    outputf: name of the results file
    Usage: python sur_predict.py -w weight.csv -n nexp.csv -o test.csv
    """
    def data_process(x):
        # Function to load and sort csv
        xd = read_csv(x)
        xd = xd.reindex(columns=sorted(xd.columns))
        return xd

    def risk_group(x):
        # Function to assign the risk groups
        if x['Score'] <= -1.07510:
            return 'Low'
        elif x['Score'] >= 1.05089:
            return 'High'
        else:
            return 'Intermediate'
    #
    w = data_process(weight)
    n = data_process(nexp)
    nc = n.drop(["ID"], axis="columns")

    # Product
    wn = Dff(np.multiply(np.array(w), np.array(nc)))
    res = Dff(wn.sum(axis=1, skipna=True))
    res.columns = ["Score"]
    res["Risk"] = res.apply(risk_group, axis=1)
    res["ID"] = n["ID"]
    #
    res.to_csv(outputf, index=True)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Calculate patient risk scores and group", add_help=False)
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                        help="Check the defaults")
    parser.add_argument("-w", "--weight", type=str, dest="weight",
                        help="Feature weights", default="weight.csv")
    parser.add_argument("-n", "--nexp", type=str, dest="nexp",
                        help="New feature measurements", default="nexp.csv")
    parser.add_argument('-o', '--output', type=str, dest='outputf',
                        help='output file name', default='Risk.csv')

    args = parser.parse_args()
    #
    sur_predict(args.weight, args.nexp, args.outputf)
