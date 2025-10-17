# =======================================================================
# Turbulence framework, plotting part. From:
# Gustavo Deco, Morten L. Kringelbach, Turbulent-like Dynamics in the Human Brain,
# Cell Reports, Volume 33, Issue 10, 2020, 108471, ISSN 2211-1247,
# https://doi.org/10.1016/j.celrep.2020.108471.
# (https://www.sciencedirect.com/science/article/pii/S2211124720314601)
#
# Part of the Thermodynamics of Mind framework:
# Kringelbach, M. L., Sanz Perl, Y., & Deco, G. (2024). The Thermodynamics of Mind.
# Trends in Cognitive Sciences (Vol. 28, Issue 6, pp. 568–581). Elsevier BV.
# https://doi.org/10.1016/j.tics.2024.03.009
#
# Code by Gustavo Deco, 2020.
# Translated by Marc Gregoris, May 21, 2024
# Refactored by Gustavo Patow, June 9, 2024
# =======================================================================
import os
import numpy as np
import neuronumba.tools.hdf as sio
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
from WholeBrain.WholeBrain.Utils.p_values import plotComparisonAcrossLabels2Ax, fontSize
# import Utils.numTricks as nTricks  # my numerical tricks
import neuronumba.tools.matlab_tricks as mTricks  # matlab compatibility tricks

# ------------------------------ Data Loader
import LibBrain.DataLoaders.ADNI_A as ADNI
DL = ADNI.ADNI_A(use360=True, cutTimeSeries=True)
ADNI_version = 'N238rev' #
# ------------------------------

dataPath = './_Data_Produced/' + DL.name() + '/' + ADNI_version + '/'
fullDataPath = dataPath + f'_Information_cascade/'

def calculate_stats(datas):
     means = np.nanmean(datas, axis=0)
     stds = np.nanstd(datas, axis=0)
     return means, stds


def plotTurbu_lambda(ax, turbuRes, observ_name, lambda_val, rev_lambdas):
    index = rev_lambdas.index(lambda_val)
    # --------------------------------------------------------------------------------------------
    # Comparisons of Amplitude Turbulence (D) across groups
    # --------------------------------------------------------------------------------------------
    classific = DL.get_classification()
    subjects = DL.get_allStudySubjects()
    groups = DL.get_groupLabels()
    BOX_R_SPA = {group: [] for group in groups}
    observ = observ_name if observ_name in turbuRes[subjects[0]] else observ_name + f'-{lambda_val}'
    for subj in subjects:
        obs_lista = np.squeeze(turbuRes[subj][observ])
        elem = (obs_lista[index]
                if isinstance(obs_lista, list) or isinstance(obs_lista, np.ndarray)
                else obs_lista)
        if observ_name == 'Transfer': elem = 1-elem
        BOX_R_SPA[classific[subj]].append(elem)
    for group in groups:
        BOX_R_SPA[group] = mTricks.reject_outliers(BOX_R_SPA[group])
    plotComparisonAcrossLabels2Ax(ax, BOX_R_SPA,
                                  custom_test='Mann-Whitney',
                                  columnLables=["HC", "MCI", "AD"],
                                  graphLabel=f"λ={lambda_val:.2f}")


def plotTurbuAttr(range, data_emp, observ, title):
    fig, axs = plt.subplots(2, int(len(range)/2), figsize=(18, 10))
    for ax, lambda_val in zip(axs.reshape(-1), range):
        print(f'\n\nPlotting Turbu lambda: {lambda_val}')
        plotTurbu_lambda(ax, data_emp, observ, lambda_val, range)
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# =======================================================================
# main plot organization routines
# =======================================================================
def plotTurbu(rev_lambdas, turbus, observations):
    for obs in observations:
        print('\n\n############################################')
        print(f'#    Turbulence: {observations[obs]} #')
        print('############################################')
        plotTurbuAttr(rev_lambdas, turbus, obs, observations[obs])


def plotMeta(turbus):
    print('\n\n############################################')
    print('#    Turbulence: Metastability             #')
    print('############################################')
    fig, ax = plt.subplots()
    plotTurbu_lambda(ax, turbus, 'Meta', lambdas[0])
    fig.suptitle('Metastability')
    plt.show()


def plotInfoCascadeFlow(range, turbus):
    print('\n\n############################################')
    print('#    Turbulence: Information Cascade Flow  #')
    print('############################################')
    plotTurbuAttr(range, turbus, 'TransferLambda', 'Information Cascade Flow')


def plotInfoCascade(turbus):
    print('\n\n############################################')
    print('#    Turbulence: Information Cascade       #')
    print('############################################')
    fig, ax = plt.subplots()
    plotTurbu_lambda(ax, turbus, 'InformationCascade', lambdas[0])
    fig.suptitle('Information Cascade')
    plt.show()

# =======================================================================
# load results
# =======================================================================
# def loadTurbu(datapath, sufix):
#     turbus = {}
#     for subj in DL.get_classification():
#         turbus[subj] = sio.loadmat(datapath + f'{sufix}_{subj}.mat')
#     return turbus

def load_turbu2(datapath):
    turbus = {}
    for subj in DL.get_classification():
        turbus[subj] = sio.loadmat(datapath + f'turbu_{subj}.mat')
    return turbus


# =======================================================================
# ==========================================================================
if __name__=="__main__":
    _observations = {'Rspatime': 'amplitude turbulence (D)',
                     'Transfer': 'Information Transfer'}
    lambdas = [0.01, 0.03, 0.06, 0.09, 0.12,
               0.15, 0.18, 0.21, 0.24, 0.27]
    rev_lambdas = list(reversed(lambdas))  # To have the same order as in Matlab
    # turbus_ = loadTurbu(dataPath, 'turbu_emp')  # regular turbulence

    # ------------- Information Cascade and Information Cascade Flow
    if os.path.exists(fullDataPath):
        turbus_ = load_turbu2(fullDataPath)
        plotTurbu(rev_lambdas, turbus_, _observations)
        plotMeta(turbus_)
        plotInfoCascadeFlow(rev_lambdas, turbus_)
        plotInfoCascade(turbus_)
    print("done")

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF