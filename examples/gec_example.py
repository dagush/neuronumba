import os
import numpy as np

from neuronumba.tools.filters import BandPassFilter
from neuronumba.observables import FC, HFreq
from neuronumba.fitting.gec import FitGEC
from neuronumba.simulator.models import Hopf
from neuronumba.observables.linear.linearfc import LinearFC
from neuronumba.tools.loader import load_2d_matrix

if __name__ == '__main__':
    # Lets load some fMRI data for the example
    subj_bolds_raw = {}
    fmri_path = "./Data_Raw/ebrains_popovych"
    for path in os.listdir(fmri_path):
        subject_path = os.path.join(fmri_path, path)
        if os.path.isdir(subject_path):
            fmri_file = os.path.join(subject_path, 'rfMRI_REST1_LR_BOLD.csv')
            if not os.path.isfile(fmri_file):
                raise FileNotFoundError(f"fMRI file <{fmri_file}> not found!")
            subj_bolds_raw[path] = load_2d_matrix(fmri_file) # Must be in (time, node) format

    tr = 720 # fMRI TR in milliseconds
    n_nodes = subj_bolds_raw[next(iter(subj_bolds_raw))].shape[1] # Number of nodes

    # We create the bandpass filter we will use for the signals
    bpf = BandPassFilter(
        k=2,
        flp=0.01,
        fhi=0.09,
        tr=tr,
        apply_detrend=True,
        apply_demean=True,
        remove_artifacts=False
    )

    # Lets apply the filter to all bold signals of the subjects
    subj_bolds = {subj: bpf.filter(subj_bolds_raw[subj]) for subj in subj_bolds_raw.keys()}

    # We compute the h_freq for each node using the "healthy control" filtered bold signals.
    # In our partiicular example, we use all the subjects to compute the frequencies.
    h_freq_ob = HFreq()
    h_freq_ob.tr = tr
    h_freq_ob.group_fmri = subj_bolds
    h_freq = h_freq_ob.compute()

    # Build model
    linear_hopf = Hopf()
    linear_hopf.a = -0.02
    linear_hopf.omega = h_freq

    # We will need to compute the empirical FC
    fc = FC()

    # For each subject, let's compute its GEC
    for subj in subj_bolds.keys():
        filtered_ts = subj_bolds[subj]

        FC_emp  = fc.from_fmri(filtered_ts)['FC']

        # Lets initilaize the FitGEC class
        fit_gec = FitGEC()
        fit_gec.tau = 2
        fit_gec.max_iters = 5000
        fit_gec.convergence_epsilon = 0.001
        fit_gec.sigma = 0.01
        fit_gec.eps_fc = 0.0004
        fit_gec.eps_cov = 0.0001
        fit_gec.convergence_test_iters = 100

        # Before computing the GEC, we need to give it an initial seed. For example,
        # idially we would give the SC. If we don't have it, we can still seed with different
        # options. It is important to make it sure that the seed it is similar to a SC, so
        # it have this requisits:
        #   - Zeros on the diagonal
        #   - Non-zeros otherwise
        #   - Values are positivie
        #   - And "normalized" to 0.2
        #
        # Some examples for the GEC seed:
        #   - With a constant value:
        #       gec_seed = np.full((n_nodes, n_nodes), 0.1)
        #   - With random values:
        #       gec_seed = np.random.uniform(0.05, 0.2, size=(n_nodes, n_nodes))
        #   - With an average of the SC of the controls:
        #       gec_seed = DL.get_AvgSC_ctrl(ctrl_label='HC', normalized='maxSC')
        #   - With the empirical FC transformed:
        gec_seed = FC_emp.copy() - np.min(FC_emp)
        np.fill_diagonal(gec_seed, 0)
        gec_seed = gec_seed / np.max(gec_seed) * 0.2

        # Now we can compute the GEC
        GEC = fit_gec.fitGEC(
            filtered_ts,
            FC_emp,
            gec_seed,
            linear_hopf,
            tr
        )

        # We can also print some debug information about the GEC computation. Or we can
        # access to the debug information stored in the class (e.g. last_run_... properites)
        fit_gec.last_run_debug_printing()
