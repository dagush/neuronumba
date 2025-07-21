# =======================================================================
# Exemple d’ús del DataLoader ADNI_A + Turbulence amb EDRLongDistance
# i comparació de la turbulència cerebral entre grups (HC, MCI, AD)
# =======================================================================
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'LibBrain')))
sys.path.append("C:/Users/LisaHaz/PycharmProjects/neuronumba/WholeBrain/WholeBrain")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from LibBrain.DataLoaders.ADNI_A import ADNI_A
from neuronumba.observables.turbulence import Turbulence
from neuronumba.observables.distance_rule import *
from WholeBrain.WholeBrain.Utils.p_values import plotComparisonAcrossLabels2


def main():
    # 1. Carregar DataLoader
    DL = ADNI_A()

    # 2. Carregar parcellació i coordenades
    parcellation = DL.get_parcellation()
    cog_dist = parcellation.get_coords()
    cog_dist = cog_dist[:360]  # Només 360 primers

    # 3. Inicialitzar turbulència
    turbulence_EDR = Turbulence(
        cog_dist=cog_dist,
        distance_rule= ExponentialDistanceRule()
    )
    turbulence_EDR._init_dependant()

    turbulence_EDR_LR = Turbulence(
        cog_dist=cog_dist,
        distance_rule=EDRLongDistance(lambda_val=0.18, lr_fraction=0.05, seed=42)
    )
    turbulence_EDR_LR._init_dependant()

    # 4. Inicialitzar diccionaris per guardar resultats
    results_EDR = {'HC': [], 'MCI': [], 'AD': []}
    results_EDR_LR = {'HC': [], 'MCI': [], 'AD': []}

    classifications = DL.get_classification()

    # 5. Iterar per tots els subjectes
    for subject in classifications:
        try:
            print(f"\n🔍 Processant subjecte: {subject}")
            data = DL.get_subjectData(subject)
            timeseries = data[subject]['timeseries']
            if timeseries.shape[0] < 360:
                print(f"⚠️  Subjecte amb només {timeseries.shape[0]} regions. Saltant...")
                continue

            # 6. Calcular turbulència
            result_EDR = turbulence_EDR.compute_turbulence(timeseries[:360, :])
            mean_turbulence_EDR = result_EDR["Rspatime"]

            result_EDR_LR = turbulence_EDR_LR.compute_turbulence(timeseries[:360, :])
            mean_turbulence_EDR_LR = result_EDR_LR["Rspatime"]

            # 7. Classificar i emmagatzemar el resultat segons grup
            group = classifications.get(subject)
            if group in results_EDR:
                results_EDR[group].append(mean_turbulence_EDR)
                results_EDR_LR[group].append(mean_turbulence_EDR_LR)

        except Exception as e:
            print(f"❌ Error amb subjecte {subject}: {e}")

    # 8. Mostrar gràfica comparativa
    print("\n📈 Mostrant comparació de la turbulència entre grups...")

    plotComparisonAcrossLabels2(
        results_EDR,
        columnLables=['HC', 'MCI', 'AD'],
        graphLabel='Group mean turbulence using EDR'
    )

    plotComparisonAcrossLabels2(
        results_EDR_LR,
        columnLables=['HC', 'MCI', 'AD'],
        graphLabel='Group mean turbulence using EDR + LR'
    )


if __name__ == "__main__":
    main()