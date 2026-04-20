import os
import sys
import pickle
import contextlib
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm
from tc_python import *

# ============================
# Path and runtime configuration
# ============================

# Base directory of the current script. All other paths are defined relative to it
# so the project can be moved without editing absolute Windows-specific locations.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input/output layout expected by the script:
#   project_root/
#       sfe_tc_commented_generic.py
#       data/
#           input.xlsx
#       results/
#       checkpoints/
INPUT_FILE = os.path.join(BASE_DIR, "data", "input.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "sfe_checkpoint.pkl")

# Save a checkpoint every N successfully processed compositions.
CHECKPOINT_EVERY = 5

# Temperature increment used for Gibbs-energy sampling (in °C).
STEP = 10

# Discrete austenitisation temperatures evaluated in aus_max() (in °C).
AUSTENITISATION_TEMPERATURES = [1075, 1200, 950, 1000, 1125, 1100, 975]

# Number of worker processes. Adjust to match license/core availability.
N_PROCS = 5

# Database name used by TC-Python.
TC_DATABASE = "TCFE12"

# Output filenames.
GIBBS_CSV_NAME = "gibbs_results.csv"
GIBBS_XLSX_NAME = "gibbs_results.xlsx"
PHASES_CSV_NAME = "phases_results.csv"
PHASES_XLSX_NAME = "phases_results.xlsx"

# Ensure output directories exist before any computation starts.
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


@contextlib.contextmanager
def suppress_stdout():
    """
    Redirect stdout and stderr to os.devnull.

    TC-Python can emit verbose messages during system construction and equilibrium
    evaluation. This context manager keeps the console output restricted to the
    progress bar and explicit user-facing prints.
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def load_input_dataframe(input_file):
    """
    Load the composition table from Excel.

    Expected input convention:
    - Each row corresponds to one nominal alloy composition.
    - Alloying elements are stored in contiguous columns from 'C' to 'Co'.
    - Values are given in wt.% for the nominal alloy composition.

    Returns
    -------
    df : pandas.DataFrame
        Numeric dataframe restricted to the alloying-element columns.
    rows_as_lists : list[list[float]]
        Plain-Python representation of the rows for multiprocessing.
    elements_full : list[str]
        Global element order used to map indices to element symbols.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Input file not found: {input_file}\n"
            f"Create the file at 'data/input.xlsx' relative to this script."
        )

    df = pd.read_excel(input_file)

    # Restrict the dataframe to the alloying-element block. This assumes that the
    # spreadsheet contains contiguous columns labelled from C to Co.
    df = df.loc[:, 'C':'Co']

    # Force numeric dtype to avoid downstream issues when setting TC conditions.
    df = df.astype(float)

    rows_as_lists = df.values.tolist()
    elements_full = df.columns.tolist()
    return df, rows_as_lists, elements_full


def sfe_calculation(individual, elements, calc_equilibrium_hcp, calc_equilibrium_fcc, calc_equilibrium_bcc):
    """
    Evaluate constrained equilibria for one austenitised composition.

    For each temperature in the sampling grid, the function computes for three
    phase-restricted systems (HCP, FCC and BCC):
      1. The Gibbs free energy of the dominant stable phase.
      2. The name of the dominant stable phase.
      3. The volume-fraction distribution of all stable phases.

    Parameters
    ----------
    individual : list[float]
        Composition of the austenite/matrix used as input for the calculation,
        expressed as mass fraction for the active elements.
    elements : list[str]
        Active alloying elements for the current composition.
    calc_equilibrium_hcp, calc_equilibrium_fcc, calc_equilibrium_bcc :
        Pre-built single-equilibrium calculators associated with phase-restricted
        TC-Python systems.

    Returns
    -------
    tuple
        Lists containing Gibbs energies, dominant phases and phase-fraction
        dictionaries for the HCP, FCC and BCC calculations.

    Notes
    -----
    - This is a thermodynamic equilibrium sweep over temperature.
    - No kinetic effects are modelled here.
    - Global minimisation is disabled in the restricted systems to probe the
      intended phase basin.
    """
    composition = individual

    # Set composition conditions for every active component in all three systems.
    for idx, element in enumerate(elements):
        mass_fraction = composition[idx]
        calc_equilibrium_hcp.set_condition(
            ThermodynamicQuantity.mass_fraction_of_a_component(element),
            mass_fraction
        )
        calc_equilibrium_fcc.set_condition(
            ThermodynamicQuantity.mass_fraction_of_a_component(element),
            mass_fraction
        )
        calc_equilibrium_bcc.set_condition(
            ThermodynamicQuantity.mass_fraction_of_a_component(element),
            mass_fraction
        )

    temperatures = list(range(0, 901, STEP))

    # ---------------- HCP-restricted system ----------------
    g_maxphase_hcp_list = []
    hcp_phase_dom_list = []
    hcp_vf_list = []

    for temp_c in temperatures:
        calc_equilibrium_hcp.set_condition(
            ThermodynamicQuantity.temperature(),
            temp_c + 273.15
        )
        try:
            result = calc_equilibrium_hcp.calculate()
            stable_phases = result.get_stable_phases()

            vf_dict = {}
            for phase in stable_phases:
                vf_dict[phase] = result.get_value_of(
                    ThermodynamicQuantity.volume_fraction_of_a_phase(phase)
                )

            # Select the dominant phase by maximum volume fraction at the current T.
            if vf_dict:
                phase_max = max(vf_dict, key=vf_dict.get)
                g_max = result.get_value_of(f"G({phase_max})")
            else:
                phase_max = None
                g_max = np.nan

            hcp_phase_dom_list.append(phase_max)
            g_maxphase_hcp_list.append(g_max)
            hcp_vf_list.append(vf_dict)
            result.invalidate()

        except Exception:
            # Failed equilibrium states are recorded explicitly as missing.
            hcp_phase_dom_list.append(None)
            g_maxphase_hcp_list.append(np.nan)
            hcp_vf_list.append({})

    # ---------------- FCC-restricted system ----------------
    g_maxphase_fcc_list = []
    fcc_phase_dom_list = []
    fcc_vf_list = []

    for temp_c in temperatures:
        calc_equilibrium_fcc.set_condition(
            ThermodynamicQuantity.temperature(),
            temp_c + 273.15
        )
        try:
            result = calc_equilibrium_fcc.calculate()
            stable_phases = result.get_stable_phases()

            vf_dict = {}
            for phase in stable_phases:
                vf_dict[phase] = result.get_value_of(
                    ThermodynamicQuantity.volume_fraction_of_a_phase(phase)
                )

            if vf_dict:
                phase_max = max(vf_dict, key=vf_dict.get)
                g_max = result.get_value_of(f"G({phase_max})")
            else:
                phase_max = None
                g_max = np.nan

            fcc_phase_dom_list.append(phase_max)
            g_maxphase_fcc_list.append(g_max)
            fcc_vf_list.append(vf_dict)
            result.invalidate()

        except Exception:
            fcc_phase_dom_list.append(None)
            g_maxphase_fcc_list.append(np.nan)
            fcc_vf_list.append({})

    # ---------------- BCC-restricted system ----------------
    g_maxphase_bcc_list = []
    bcc_phase_dom_list = []
    bcc_vf_list = []

    for temp_c in temperatures:
        calc_equilibrium_bcc.set_condition(
            ThermodynamicQuantity.temperature(),
            temp_c + 273.15
        )
        try:
            result = calc_equilibrium_bcc.calculate()
            stable_phases = result.get_stable_phases()

            vf_dict = {}
            for phase in stable_phases:
                vf_dict[phase] = result.get_value_of(
                    ThermodynamicQuantity.volume_fraction_of_a_phase(phase)
                )

            if vf_dict:
                phase_max = max(vf_dict, key=vf_dict.get)
                g_max = result.get_value_of(f"G({phase_max})")
            else:
                phase_max = None
                g_max = np.nan

            bcc_phase_dom_list.append(phase_max)
            g_maxphase_bcc_list.append(g_max)
            bcc_vf_list.append(vf_dict)
            result.invalidate()

        except Exception:
            bcc_phase_dom_list.append(None)
            g_maxphase_bcc_list.append(np.nan)
            bcc_vf_list.append({})

    return (
        g_maxphase_hcp_list,
        g_maxphase_fcc_list,
        g_maxphase_bcc_list,
        hcp_phase_dom_list,
        fcc_phase_dom_list,
        bcc_phase_dom_list,
        hcp_vf_list,
        fcc_vf_list,
        bcc_vf_list,
    )



def aus_max(individual, elements, system_aus):
    """
    Select an austenitisation condition from a discrete candidate set.

    The selection rule is purely thermodynamic:
      - Reject conditions with liquid present.
      - Among liquid-free states, maximise the FCC_A1 volume fraction.
      - If a nearly single-phase austenite state is found (FCC_A1 >= 0.99),
        return immediately.

    Parameters
    ----------
    individual : list[float]
        Nominal alloy composition in wt.% for the active elements.
    elements : list[str]
        Active alloying elements.
    system_aus : tc_python.System
        Unrestricted thermodynamic system used to search for austenitisation.

    Returns
    -------
    list_matrix : list[float]
        Composition of FCC_A1 as weight fraction for each active element.
    list_phases : list[str]
        Stable phases at the selected austenitisation condition.
    aus_fraction : float or bool
        FCC_A1 volume fraction at the selected condition. Returns False when no
        valid liquid-free condition is found.

    Important limitation
    --------------------
    This routine does not optimise over a continuous temperature interval. The
    solution is restricted to the predefined candidate temperatures.
    """
    list_matrix = []
    list_phases = []
    list_aus_fraction = []
    list_t_aus = []

    if len(elements) != len(individual):
        return list_matrix, list_phases, False

    calc_equilibrium_aus = system_aus.with_single_equilibrium_calculation()

    # Nominal alloy composition is interpreted here as wt.%; TC-Python requires
    # mass fractions, hence division by 100.
    for idx, element in enumerate(elements):
        mass_fraction = individual[idx] / 100.0
        calc_equilibrium_aus.set_condition(
            ThermodynamicQuantity.mass_fraction_of_a_component(element),
            mass_fraction
        )

    for t_aus_c in AUSTENITISATION_TEMPERATURES:
        calc_equilibrium_aus.set_condition(
            ThermodynamicQuantity.temperature(),
            t_aus_c + 273.15
        )

        result = calc_equilibrium_aus.calculate()

        phase_fractions = {
            "LIQ": result.get_value_of(
                ThermodynamicQuantity.volume_fraction_of_a_phase('LIQUID')
            ),
            "FCC_A1": result.get_value_of(
                ThermodynamicQuantity.volume_fraction_of_a_phase('FCC_A1')
            )
        }

        # Early exit for a nearly single-phase austenite state without liquid.
        if phase_fractions["FCC_A1"] >= 0.99 and phase_fractions["LIQ"] <= 0:
            list_matrix = [
                result.get_value_of(
                    ThermodynamicQuantity.composition_of_phase_as_weight_fraction(
                        'FCC_A1', element
                    )
                )
                for element in elements
            ]
            list_phases = result.get_stable_phases()
            aus_frac = phase_fractions["FCC_A1"]
            result.invalidate()
            return list_matrix, list_phases, aus_frac

        # Store all liquid-free candidate states and choose the best one later.
        if phase_fractions["LIQ"] <= 0:
            list_aus_fraction.append(phase_fractions["FCC_A1"])
            list_t_aus.append(t_aus_c)

        result.invalidate()

    if list_aus_fraction:
        best_idx = int(np.argmax(list_aus_fraction))
        best_t_c = list_t_aus[best_idx]
        best_frac = list_aus_fraction[best_idx]

        calc_equilibrium_aus.set_condition(
            ThermodynamicQuantity.temperature(),
            best_t_c + 273.15
        )
        result = calc_equilibrium_aus.calculate()

        list_matrix = [
            result.get_value_of(
                ThermodynamicQuantity.composition_of_phase_as_weight_fraction(
                    'FCC_A1', element
                )
            )
            for element in elements
        ]
        list_phases = result.get_stable_phases()
        result.invalidate()

        return list_matrix, list_phases, best_frac

    return list_matrix, list_phases, False


# These globals are populated in main and accessed inside worker processes.
ROWS_AS_LISTS = None
ELEMENTS_FULL = None


def process_composition(idx_row):
    """
    Worker routine for a single nominal composition.

    Workflow
    --------
    1. Detect active elements (non-zero nominal composition).
    2. Build an unrestricted system for austenitisation selection.
    3. Build three phase-restricted systems (HCP/FCC/BCC).
    4. Evaluate Gibbs energy and phase information on the temperature grid.

    Returns
    -------
    tuple or None
        (idx, combined_row, dic_entry) if successful, otherwise None for the
        degenerate case with no active alloying elements.

    Notes
    -----
    Each worker creates its own TC-Python session. This is deliberate because
    TC-Python objects are not shared across processes.
    """
    idx, row = idx_row

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"openpyxl.worksheet._reader"
    )

    active_indices = [i for i, value in enumerate(row) if value != 0.0]
    active_elements = [ELEMENTS_FULL[i] for i in active_indices]
    active_comp_nominal = [row[i] for i in active_indices]

    if not active_elements:
        return None

    with suppress_stdout():
        with TCPython() as session:
            # Unrestricted system used to identify the best austenitised state.
            system_builder_aus = session.select_database_and_elements(
                TC_DATABASE, ["Fe"] + active_elements
            )
            system_aus = system_builder_aus.get_system()

            element_aus, phases, aus_fraction = aus_max(
                active_comp_nominal,
                active_elements,
                system_aus
            )

            # HCP-restricted system.
            system_builder_hcp = session.select_database_and_elements(
                TC_DATABASE, ["Fe"] + active_elements
            )
            system_builder_hcp.without_default_phases()
            system_builder_hcp.select_phase("HCP_A3")
            system_hcp = system_builder_hcp.get_system()
            calc_equilibrium_hcp = (
                system_hcp.with_single_equilibrium_calculation()
                .disable_global_minimization()
            )

            # FCC-restricted system.
            system_builder_fcc = session.select_database_and_elements(
                TC_DATABASE, ["Fe"] + active_elements
            )
            system_builder_fcc.without_default_phases()
            system_builder_fcc.select_phase("FCC_A1")
            system_fcc = system_builder_fcc.get_system()
            calc_equilibrium_fcc = (
                system_fcc.with_single_equilibrium_calculation()
                .disable_global_minimization()
            )

            # BCC-restricted system.
            system_builder_bcc = session.select_database_and_elements(
                TC_DATABASE, ["Fe"] + active_elements
            )
            system_builder_bcc.without_default_phases()
            system_builder_bcc.select_phase("BCC_A2")
            system_bcc = system_builder_bcc.get_system()
            calc_equilibrium_bcc = (
                system_bcc.with_single_equilibrium_calculation()
                .disable_global_minimization()
            )

            (
                g_hcp_list,
                g_fcc_list,
                g_bcc_list,
                hcp_phase_dom_list,
                fcc_phase_dom_list,
                bcc_phase_dom_list,
                hcp_vf_list,
                fcc_vf_list,
                bcc_vf_list,
            ) = sfe_calculation(
                element_aus,
                active_elements,
                calc_equilibrium_hcp,
                calc_equilibrium_fcc,
                calc_equilibrium_bcc,
            )

            # Concatenate the temperature-resolved Gibbs energies in a fixed order:
            # HCP block + FCC block + BCC block.
            combined_row = g_hcp_list + g_fcc_list + g_bcc_list

            dic_entry = {
                'aus_comp': element_aus,
                'phases': phases,
                'aus_fraction': aus_fraction,
                'phase_hcp': hcp_phase_dom_list,
                'phase_fcc': fcc_phase_dom_list,
                'phase_bcc': bcc_phase_dom_list,
                'vf_hcp': hcp_vf_list,
                'vf_fcc': fcc_vf_list,
                'vf_bcc': bcc_vf_list,
            }

    return idx, combined_row, dic_entry



def save_checkpoint(path, completed_indices, combined_results_dict, dic_phases_dict):
    """
    Persist partial results to disk.

    The checkpoint is written atomically through a temporary file followed by
    os.replace(), which reduces the risk of a corrupted checkpoint if execution
    is interrupted during writing.
    """
    data = {
        'completed_indices': list(completed_indices),
        'combined_results': combined_results_dict,
        'dic_phases': dic_phases_dict,
    }
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as file_handle:
        pickle.dump(data, file_handle)
    os.replace(tmp_path, path)



def load_checkpoint(path):
    """
    Load previously saved results if a checkpoint exists.

    Returns empty data structures when no checkpoint is present.
    """
    if not os.path.exists(path):
        return set(), {}, {
            'aus_comp': {},
            'phases': {},
            'aus_fraction': {},
            'phase_hcp': {},
            'phase_fcc': {},
            'phase_bcc': {},
            'vf_hcp': {},
            'vf_fcc': {},
            'vf_bcc': {},
        }

    with open(path, "rb") as file_handle:
        data = pickle.load(file_handle)

    completed_indices = set(data.get('completed_indices', []))
    combined_results_dict = data.get('combined_results', {})
    dic_phases_dict = data.get('dic_phases', {})

    # Ensure all expected keys exist even when loading an older checkpoint.
    for key in [
        'aus_comp',
        'phases',
        'aus_fraction',
        'phase_hcp',
        'phase_fcc',
        'phase_bcc',
        'vf_hcp',
        'vf_fcc',
        'vf_bcc',
    ]:
        dic_phases_dict.setdefault(key, {})

    return completed_indices, combined_results_dict, dic_phases_dict



def build_output_tables(valid_indices, combined_results_dict, dic_phases_dict):
    """
    Assemble the final pandas DataFrames in the original processing order.
    """
    temperatures = list(range(0, 901, STEP))

    column_names = (
        [f'HCP_{temp}' for temp in temperatures] +
        [f'FCC_{temp}' for temp in temperatures] +
        [f'BCC_{temp}' for temp in temperatures]
    )

    combined_results = [combined_results_dict[i] for i in valid_indices]
    result_df = pd.DataFrame(combined_results, columns=column_names)

    dic_phases = {
        key: [dic_phases_dict[key][i] for i in valid_indices]
        for key in dic_phases_dict.keys()
    }
    phases_df = pd.DataFrame(dic_phases)

    return result_df, phases_df



def save_output_tables(result_df, phases_df):
    """
    Save the Gibbs-energy table and the phase/auxiliary table to CSV and Excel.
    """
    gibbs_csv_path = os.path.join(OUTPUT_DIR, GIBBS_CSV_NAME)
    gibbs_xlsx_path = os.path.join(OUTPUT_DIR, GIBBS_XLSX_NAME)
    phases_csv_path = os.path.join(OUTPUT_DIR, PHASES_CSV_NAME)
    phases_xlsx_path = os.path.join(OUTPUT_DIR, PHASES_XLSX_NAME)

    result_df.to_csv(gibbs_csv_path, index=False)
    result_df.to_excel(gibbs_xlsx_path, index=False)

    phases_df.to_csv(phases_csv_path, index=False)
    phases_df.to_excel(phases_xlsx_path, index=False)

    return gibbs_csv_path, gibbs_xlsx_path, phases_csv_path, phases_xlsx_path



def main():
    global ROWS_AS_LISTS, ELEMENTS_FULL

    _, ROWS_AS_LISTS, ELEMENTS_FULL = load_input_dataframe(INPUT_FILE)

    completed_indices, combined_results_dict, dic_phases_dict = load_checkpoint(
        CHECKPOINT_PATH
    )

    indexed_rows = list(enumerate(ROWS_AS_LISTS))
    remaining_tasks = [
        (idx, row) for idx, row in indexed_rows if idx not in completed_indices
    ]

    processed_since_checkpoint = 0

    if remaining_tasks:
        with Pool(processes=N_PROCS) as pool:
            for result in tqdm(
                pool.imap(process_composition, remaining_tasks),
                total=len(remaining_tasks),
                desc="Processing compositions",
                unit="row",
            ):
                if result is None:
                    continue

                idx, combined_row, dic_entry = result
                completed_indices.add(idx)
                combined_results_dict[idx] = combined_row

                for key in dic_phases_dict.keys():
                    dic_phases_dict[key][idx] = dic_entry[key]

                processed_since_checkpoint += 1
                if processed_since_checkpoint >= CHECKPOINT_EVERY:
                    save_checkpoint(
                        CHECKPOINT_PATH,
                        completed_indices,
                        combined_results_dict,
                        dic_phases_dict,
                    )
                    processed_since_checkpoint = 0

        # Final save after the remaining work has completed.
        save_checkpoint(
            CHECKPOINT_PATH,
            completed_indices,
            combined_results_dict,
            dic_phases_dict,
        )

    # Keep only rows for which the full calculation returned a result.
    valid_indices = sorted(combined_results_dict.keys())

    result_df, phases_df = build_output_tables(
        valid_indices,
        combined_results_dict,
        dic_phases_dict,
    )

    gibbs_csv_path, gibbs_xlsx_path, phases_csv_path, phases_xlsx_path = save_output_tables(
        result_df,
        phases_df,
    )

    print("Saved files:")
    print(f"  - {gibbs_csv_path}")
    print(f"  - {gibbs_xlsx_path}")
    print(f"  - {phases_csv_path}")
    print(f"  - {phases_xlsx_path}")
    print("\nPreview of Gibbs-energy table:")
    print(result_df.head())
    print("\nPreview of phase-information table:")
    print(phases_df.head())


if __name__ == "__main__":
    main()
