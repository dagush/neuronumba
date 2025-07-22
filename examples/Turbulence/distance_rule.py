# =======================================================================
# Exemple d’ús del DataLoader ADNI_A + Turbulence amb EDRLongDistance
# i comparació de la turbulència cerebral entre grups (HC, MCI, AD)
# =======================================================================
import sys
import os

from neuronumba.observables import Information_transfer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'LibBrain')))
sys.path.append("C:/Users/LisaHaz/PycharmProjects/neuronumba/WholeBrain/WholeBrain")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from LibBrain.DataLoaders.ADNI_A import ADNI_A
from neuronumba.observables.turbulence import Turbulence
from neuronumba.observables.distance_rule import *
from WholeBrain.WholeBrain.Utils.p_values import plotComparisonAcrossLabels2
from LibBrain.DataLoaders.ADNI_A import computeAvgSC_HC_Matrix, classification, correctSC
from plot_turbu2 import *


def main():
    # 1. Carregar DataLoader i SC
    DL = ADNI_A()
    classifications = DL.get_classification()

    # Calcular SC mitjana dels controls sans (HC)
    raw_SC = computeAvgSC_HC_Matrix(classifications, "E:\lisa\WorkBrain\_Data_Raw\ADNI-A\connectomes")
    SC = correctSC(raw_SC)  # Normalització opcional

    # 2. Carregar parcellació i coordenades
    parcellation = DL.get_parcellation()
    cog_dist = parcellation.get_coords()
    cog_dist = cog_dist[:360]  # Només 360 primers

    # 3. Lambda values to test
    lambdas = [0.01, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27]

    # 4. Inicialitzar resultats per lambda
    turbus_EDR =  {}
    turbus_EDR_LR = {}
    # Iterar per tots els subjectes abans de lambdas
    for subject in classifications:
        turbus_EDR[subject] = {"Rspatime": [], "Meta": [], "gKoP": [], "Transfer": []}
        turbus_EDR_LR[subject] = {"Rspatime": [], "Meta": [], "gKoP": [], "Transfer": []}

    # 5. Iterar per valors de lambda
    for lambda_val in lambdas:
        print(f"\n🔁 Lambda = {lambda_val}")
        # Inicialitzar turbulència
        turbulence_EDR = Turbulence(
            cog_dist=cog_dist,
            distance_rule=ExponentialDistanceRule(lambda_val=lambda_val),
        )
        turbulence_EDR._init_dependant()

        turbulence_EDR_LR = Turbulence(
            cog_dist=cog_dist,
            distance_rule=EDRLongDistance(lambda_val=lambda_val, sc=SC)
        )
        turbulence_EDR_LR._init_dependant()

        infoTransfer_EDR = Information_transfer(
            cog_dist=cog_dist,
            distance_rule = ExponentialDistanceRule(lambda_val=lambda_val),
        )

        infoTransfer_EDR._init_dependant()

        infoTransfer_LR = Information_transfer(
            cog_dist=cog_dist,
            distance_rule = EDRLongDistance(lambda_val=lambda_val, sc=SC)
        )
        infoTransfer_LR._init_dependant()




        # 5. Iterar per tots els subjectes
        for subject in classifications:
            try:
                print(f"\n Processant subjecte: {subject}")
                data = DL.get_subjectData(subject)
                timeseries = data[subject]['timeseries']
                if timeseries.shape[0] < 360:
                    print(f"Subjecte amb només {timeseries.shape[0]} regions. Saltant...")
                    continue

                print("c_exp:", infoTransfer_EDR.c_exp.shape)
                print("bold_signal:", timeseries[:360, :].shape)

                # 7. Classificar i emmagatzemar el resultat
                result_EDR = turbulence_EDR.compute_turbulence(timeseries[:360, :])
                result_EDR_LR = turbulence_EDR_LR.compute_turbulence(timeseries[:360, :])

                resultInfoTransfer_EDR = infoTransfer_EDR.compute_information_transfer(timeseries[:360, :])
                resultInfoTransfer_LR = infoTransfer_LR.compute_information_transfer(timeseries[:360, :])

                turbus_EDR[subject]["Rspatime"].append(result_EDR["Rspatime"])
                turbus_EDR[subject]["Meta"].append(result_EDR["Meta"])
                turbus_EDR[subject]["gKoP"].append(result_EDR["gKoP"])
                turbus_EDR[subject]["Transfer"].append(resultInfoTransfer_EDR["Transfer"])

                turbus_EDR_LR[subject]["Rspatime"].append(result_EDR_LR["Rspatime"])
                turbus_EDR_LR[subject]["Meta"].append(result_EDR_LR["Meta"])
                turbus_EDR_LR[subject]["gKoP"].append(result_EDR_LR["gKoP"])
                turbus_EDR_LR[subject]["Transfer"].append(resultInfoTransfer_LR["Transfer"])

            except Exception as e:
                print(f" Error amb subjecte {subject}: {e}")

    # 8. Mostrar gràfica comparativa
    print("\n📈 Mostrant comparació de la turbulència entre grups...")

    rev_lambdas = list(reversed(lambdas))
    _observations_EDR  = {
        'Rspatime': 'Amplitude Turbulence (D) using EDR',
        'Meta': 'Global metastability using EDR',
        'gKoP': 'Global Kuramoto Parameters using EDR',
        'Transfer': 'Information transfer using EDR',
    }
    _observations_EDR_LR = {
        'Rspatime': 'Amplitude Turbulence (D) using EDR + LR',
        'Meta': 'Global metastability using EDR + LR',
        'gKoP': 'Global Kuramoto Parameters using EDR + LR',
        'Transfer': 'Information transfer using EDR + LR',
    }


    plotTurbu(rev_lambdas, turbus_EDR, _observations_EDR)
    plotTurbu(rev_lambdas, turbus_EDR_LR, _observations_EDR_LR)



if __name__ == "__main__":
    main()