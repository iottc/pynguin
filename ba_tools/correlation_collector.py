import os
from pathlib import Path
import pandas as pd
import enum
from scipy.stats.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt


class DataKeeper():
    def __init__(self):
        base_data : pd.DataFrame = []
        internet_data : pd.DataFrame = []
        choice_data : pd.DataFrame = []

    def set_base_data(self, data):
        self.base_data = data

    def set_internet_data(self, data):
        self.internet_data = data

    def set_choice_data(self, data):
        self.choice_data = data

    def get_data(self):
        return self.base_data, self.internet_data, self.choice_data

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

def write_csv(metric, measure: CorrelationContents):
    output_dir = Path("./corrs")
    output_file = output_dir / get_filename(metric)
    file_header = "module,correlation_direct_mean,correlation_direct_median,correlation_direct_max,correlation_direct_min,correlation_delta_mean,correlation_delta_median,correlation_delta_max,correlation_delta_min,correlation_early_mean,correlation_early_median,correlation_early_max,correlation_early_min\n"
    header_necessary: bool

    try:
        output_file.resolve(strict=True)
    except FileNotFoundError:
        header_necessary = True
    else:
        header_necessary = False

    with open(output_file, "a") as file:
        if header_necessary:
            file.write(file_header)
        file.write(f"{measure.module},{measure.direct_mean},{measure.direct_median},{measure.direct_max},{measure.direct_min},{measure.div_mean},{measure.div_median},{measure.div_max},{measure.div_min},{measure.early_mean},{measure.early_median},{measure.early_max},{measure.early_min}\n")


def calculate_correlations(metric: Metrics, data, module: str, project: str):
    """
    Berechnet alle Korrelationen für eine gegebene Metrik und einen Modul.
    """
    result = CorrelationContents()
    result.set_module(project + "." + module)

    for method in Methods:
        data_to_explore = data[data['metric'] == metric.value + method.value]
        corr_direct = data_to_explore["result"].corr(data_to_explore["coverage"], method="spearman")

        results_list = data_to_explore["result"].to_list()
        coverage_list = data_to_explore["coverage"].to_list()

        delta_result = [results_list[i + 1] - results_list[i] for i in range(len(results_list) - 1)]
        delta_coverage = [coverage_list[i + 1] - coverage_list[i] for i in range(len(coverage_list) - 1)]

        try:
            corr_div_result = spearmanr(delta_result, delta_coverage)
            corr_div = corr_div_result.correlation
        except ValueError:
            corr_div = None

        grouped_df = data_to_explore.groupby("run")
        # Head(5) auf jede Gruppe anwenden, dann den Median pro Gruppe berechnen
        early_results = grouped_df["result"].head(20).groupby(data["run"]).median()
        final_coverage = grouped_df["coverage"].last()

        summary_df = pd.DataFrame({
            'Early_Results': early_results,
            'Final_Coverage': final_coverage
        })


        corr_early_matrix = summary_df.corr(method='spearman')
        corr_early = corr_early_matrix.iloc[0, 1]

        if method == Methods.MEAN:
            result.set_direct_mean(corr_direct)
            result.set_div_mean(corr_div)
            result.set_early_mean(corr_early)
        elif method == Methods.MEDIAN:
            result.set_direct_median(corr_direct)
            result.set_div_median(corr_div)
            result.set_early_median(corr_early)
        elif method == Methods.MAX:
            result.set_direct_max(corr_direct)
            result.set_div_max(corr_div)
            result.set_early_max(corr_early)
        elif method == Methods.MIN:
            result.set_direct_min(corr_direct)
            result.set_div_min(corr_div)
            result.set_early_min(corr_early)

    write_csv(metric, result)

def get_coverages(data, module: str, project: str):
    fig, axes = plt.subplots()
    sns.lineplot(data=data, x="iteration", y="coverage")
    axes.set_title(f"Coverage of {project}.{module}")
    axes.set_xlabel("iteration")
    axes.set_ylabel("coverage")

    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig(f"./plots/fix_scale/coverage_{project}_{module}.png")

def get_thesis_coverage(keeper):
    base_data, internet_data, choice_data = keeper.get_data()

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

    sns.lineplot(data=base_data, x="iteration", y="coverage", ax=axes[0, 0])
    axes[0, 0].set_title("Early converging coverage of mimesis.providers.base")
    axes[0, 0].set_xlabel("iteration")
    axes[0, 0].set_ylabel("coverage")
    axes[0, 0].set_ylim(0, 1)

    sns.lineplot(data=internet_data, x="iteration", y="coverage", ax=axes[0, 1])
    axes[0, 1].set_title("Medium converging coverage of mimesis.providers.internet")
    axes[0, 1].set_xlabel("iteration")
    axes[0, 1].set_ylabel("coverage")
    axes[0, 1].set_ylim(0, 1)


    sns.lineplot(data=choice_data, x="iteration", y="coverage", ax=axes[1, 0])
    axes[1, 0].set_title("Late converging coverage of mimesis.providers.choice")
    axes[1, 0].set_xlabel("iteration")
    axes[1, 0].set_ylabel("coverage")
    axes[1, 0].set_ylim(0, 1)

    fig.delaxes(axes[1, 1])

    plt.tight_layout()

    # Speichert die gesamte Figur mit den drei Plots
    plt.savefig("./plots/combined_metrics_thesis.png")

def process_module(module_path, keeper):
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

    if module_name == "base":
        keeper.set_base_data(combined_df)
    if module_name == "choice":
        keeper.set_choice_data(combined_df)
    if module_name == "internet":
        keeper.set_internet_data(combined_df)

    # for metric in Metrics:
    #     print(f"  -> Berechne Korrelationen für Metrik: {metric.name}")
    #     calculate_correlations(metric, combined_df, module_name, project_name)
    #get_coverages(combined_df, module_name, project_name)

def find_and_process_projects():
    keeper = DataKeeper()
    projects_dir = os.path.join(os.getcwd(), 'projects')

    for project_name in os.listdir(projects_dir):
        project_path = os.path.join(projects_dir, project_name)

        if os.path.isdir(project_path):
            for subdir in os.listdir(project_path):
                subdir_path = os.path.join(project_path, subdir)
                process_module(subdir_path, keeper)

    get_thesis_coverage(keeper)


if __name__ == "__main__":
    find_and_process_projects()
