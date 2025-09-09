import os
from pathlib import Path
import pandas as pd
import enum
from scipy.stats import shapiro
import seaborn as sns
import matplotlib.pyplot as plt


class Metrics(enum.Enum):
    FV = "FV"
    PIC = "PIC"
    CR = "CR"
    AC = "AC"
    NV = "NV"
    DIVEX = "DIVEX"
    FD = "FD"
    SV = "SVC"

class Methods(enum.Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MAX = "max"
    MIN = "min"

class CorrelationContents():
    def __init__(self):
        self.module : str = ""
        self.direct_mean = 2.0
        self.direct_median = 2.0
        self.direct_max = 2.0
        self.direct_min = 2.0
        self.div_mean = 2.0
        self.div_median = 2.0
        self.div_max = 2.0
        self.div_min = 2.0
        self.early_mean = 2.0
        self.early_median = 2.0
        self.early_max = 2.0
        self.early_min = 2.0

    def set_module(self, value: str):
        self.module = value

    def set_direct_mean(self, value: float):
        self.direct_mean = value

    def set_direct_median(self, value: float):
        self.direct_median = value

    def set_direct_max(self, value: float):
        self.direct_max = value

    def set_direct_min(self, value: float):
        self.direct_min = value

    def set_div_mean(self, value: float):
        self.div_mean = value

    def set_div_median(self, value: float):
        self.div_median = value

    def set_div_max(self, value: float):
        self.div_max = value

    def set_div_min(self, value: float):
        self.div_min = value

    def set_early_mean(self, value: float):
        self.early_mean = value

    def set_early_median(self, value: float):
        self.early_median = value

    def set_early_max(self, value: float):
        self.early_max = value

    def set_early_min(self, value: float):
        self.early_min = value

FV_values = []
PIC_values = []
CR_values = []
AC_values = []
NV_values = []
DIVEX_values = []
FD_values = []
SV_values = []


def get_filename(metric: Metrics):
    match metric:
        case Metrics.FV:
            return "fitness_variance.csv"
        case Metrics.PIC:
            return "population_information_content.csv"
        case Metrics.CR:
            return "change_rate.csv"
        case Metrics.AC:
            return "autocorrelation.csv"
        case Metrics.NV:
            return "neutrality_volume.csv"
        case Metrics.DIVEX:
            return "diversity.csv"
        case Metrics.FD:
            return "function_dispersion.csv"
        case Metrics.SV:
            return "state_variance.csv"

def determine_distribution(metric: Metrics, data, module: str, project: str):
    """
    Berechnet alle Korrelationen für eine gegebene Metrik und einen Modul.
    """
    result = CorrelationContents()
    result.set_module(project + "." + module)

    for method in Methods:
        data_to_explore = data[data['metric'] == metric.value + method.value]
        statistic, p_value = shapiro(data_to_explore['result'])

        match metric.value:
            case "FV":
                FV_values.append(p_value)
            case "PIC":
                PIC_values.append(p_value)
            case "CR":
                CR_values.append(p_value)
            case "AC":
                AC_values.append(p_value)
            case "NV":
                NV_values.append(p_value)
            case "DIVEX":
                DIVEX_values.append(p_value)
            case "FD":
                FD_values.append(p_value)
            case "SVC":
                SV_values.append(p_value)
            case _:
                None


def process_module(module_path):
    module_name = os.path.basename(module_path)
    project_name = os.path.basename(Path(module_path).parent)
    all_dataframes = []
    i = 1
    for filename in os.listdir(module_path):
        # Überprüfe, ob die Datei eine CSV-Datei ist
        if filename.endswith(".csv") and filename.startswith("metric"):
            file_path = os.path.join(module_path, filename)

            df = pd.read_csv(file_path)
            df['run'] = i

            all_dataframes.append(df)
            i += 1

    print(f"Read {len(all_dataframes)} files for {module_path}")
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    for metric in Metrics:
        print(f"  -> Berechne Korrelationen für Metrik: {metric.name}")
        determine_distribution(metric, combined_df, module_name, project_name)

def find_and_process_projects():
    projects_dir = os.path.join(os.getcwd(), 'projects')

    for project_name in os.listdir(projects_dir):
        project_path = os.path.join(projects_dir, project_name)

        if os.path.isdir(project_path):
            for subdir in os.listdir(project_path):
                subdir_path = os.path.join(project_path, subdir)
                process_module(subdir_path)

    print("AC distribution p values:\n")
    print(f"{min(AC_values)} --- {max(AC_values)}")
    print("PIC distribution p values:\n")
    print(f"{min(PIC_values)} --- {max(PIC_values)}")
    print("NV distribution p values:\n")
    print(f"{min(NV_values)} --- {max(NV_values)}")
    print("FV distribution p values:\n")
    print(f"{min(FV_values)} --- {max(FV_values)}")
    print("SV distribution p values:\n")
    print(f"{min(SV_values)} --- {max(SV_values)}")
    print("DIV distribution p values:\n")
    print(f"{min(DIVEX_values)} --- {max(DIVEX_values)}")
    print("FD distribution p values:\n")
    print(f"{min(FD_values)} --- {max(FD_values)}")
    print("CR distribution p values:\n")
    print(f"{min(CR_values)} --- {max(CR_values)}")

    for value in AC_values:
        print(value)

    print("Next")

    for value in FD_values:
        print(value)
   
    print("DIV VALUES")
    for value in DIVEX_values:
        print(value)

if __name__ == "__main__":
    find_and_process_projects()
