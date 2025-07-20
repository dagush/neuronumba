# examples/adni_distance_example.py
# =======================================================================
# Exemple d’ús del DataLoader ADNI_A + Turbulence amb EDR i EDRLongDistance
# =======================================================================
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'LibBrain')))


# Defineix la ruta absoluta on està el teu projecte (el directori que conté 'DataLoaders')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from LibBrain.DataLoaders.ADNI_A import ADNI_A
from neuronumba.observables.turbulence import Turbulence
from neuronumba.observables.distance_rule import *

def main():
    # 1. Carregar dades
    DL = ADNI_A()
    subject = '002_S_0413'  # Subject d'exemple, canvia'l per un que tinguis

    # Obtenir dades específiques del subjecte
    data = DL.get_subjectData(subject)
    timeseries_379 = data[subject]['timeseries']
    timeseries = timeseries_379[:360, :]
    print(timeseries.shape)
    # Obtenir centres ROI des de la parcel·lació
    parcellation = DL.get_parcellation()
    cog_dist = parcellation.get_coords()  # coordenades
    print(cog_dist.shape)



    # 3. Crear la classe Turbulence amb la regla de distància i les coordenades
    turbulence = Turbulence(
        cog_dist=cog_dist,
        distance_rule = EDRLongDistance(lambda_val=0.18, lr_fraction=0.05, seed=42)
    )
    turbulence._init_dependant()

    # 4. Calcular la turbulència a partir de la sèrie temporal fMRI
    result = turbulence.compute_turbulence(timeseries)

    # 5. Mostrar resultats
    print("Resultats turbulència:")
    for key, value in result.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
