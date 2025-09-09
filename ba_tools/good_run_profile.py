import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import enum

stagnation_window = 10
analysis_results = []
min_coverage_spread_threshold = 0.2

# Enum-Klassen sind unverändert und können so bleiben
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


def rename_metrics(df):

    print(df.head())

    metric_prefix_mapping = {
        'DIVEX': 'Diversity',
        'SVC': 'State Variance',
        'FD': 'Function Dispersion',
        'FV': 'Fitness Variance',
        'CR': 'Change Rate',
        'AC': 'Autocorrelation',
        'NV': 'Neutrality Volume',
        'PIC': 'Population Information Content'
    }

    suffixes = ['min', 'max', 'mean', 'median']

    df['Metrik_Name'] = None
    df['FitnessObservationType'] = None

    for index, row in df.iterrows():
        original_metric = row['Metric']
        found_suffix = None
        found_metric = None

        print(original_metric)

        for sfx in sorted(suffixes, key=len, reverse=True):
            if original_metric.endswith(sfx):
                found_suffix = sfx
                prefix = original_metric[:-len(sfx)]
                print(prefix)
                if prefix in metric_prefix_mapping:
                    found_metric = metric_prefix_mapping[prefix]
                    found_suffix = sfx
                    break
        df.loc[index, 'Metric'] = found_metric
        df.loc[index, 'FitnessObservationType'] = found_suffix

    resultOrder = [
        "Module",
        "Metric",
        "FitnessObservationType",
        "p_value",
        "Rank_Biserial_Correlation"
    ]

    return df[resultOrder].copy()

def process_all_modules_and_combine_data(projects_dir: str):
    all_combined_dataframes = []

    for project_name in os.listdir(projects_dir):
        project_path = os.path.join(projects_dir, project_name)

        if os.path.isdir(project_path):
            for subdir in os.listdir(project_path):
                subdir_path = os.path.join(project_path, subdir)

                if os.path.isdir(subdir_path):
                    module_name = subdir
                    all_dataframes_for_module = []
                    i = 1

                    for filename in os.listdir(subdir_path):
                        if filename.endswith(".csv") and filename.startswith("metric"):
                            file_path = os.path.join(subdir_path, filename)

                            df = pd.read_csv(file_path)
                            df['run'] = i
                            # Die entscheidende Änderung: Kombiniere Projekt- und Modulnamen
                            df['module'] = f"{project_name}.{module_name}"
                            all_dataframes_for_module.append(df)
                            i += 1

                    if all_dataframes_for_module:
                        combined_df_for_module = pd.concat(all_dataframes_for_module, ignore_index=True)
                        all_combined_dataframes.append(combined_df_for_module)

    return pd.concat(all_combined_dataframes, ignore_index=True)

def analyze_combined_data(metric_df: pd.DataFrame):
    analysis_results = []
    top_quantile = 0.75
    bottom_quantile = 0.25
    modules = []
    for module_name, df_module in metric_df.groupby('module'):

        final_coverage = df_module.groupby('run')['coverage'].last()

        top_threshold = final_coverage.quantile(top_quantile)
        bottom_threshold = final_coverage.quantile(bottom_quantile)

        golden_runs = final_coverage[final_coverage >= top_threshold].index
        dud_runs = final_coverage[final_coverage <= bottom_threshold].index

        mean_golden_coverage = final_coverage[golden_runs].median()
        mean_dud_coverage = final_coverage[dud_runs].median()
        coverage_diff = mean_golden_coverage - mean_dud_coverage
        modules.append((module_name, coverage_diff))

        if coverage_diff >= min_coverage_spread_threshold:
            for metric_name, df_metric in df_module.groupby('metric'):

                golden_group = df_metric[df_metric['run'].isin(golden_runs)]['result']
                dud_group = df_metric[df_metric['run'].isin(dud_runs)]['result']

                p_value = 1.0
                rank_biserial_corr = 0.0

                if not golden_group.empty and not dud_group.empty:
                    try:
                        u_statistic, p_value = mannwhitneyu(golden_group, dud_group, alternative='two-sided')
                        n1 = len(golden_group)
                        n2 = len(dud_group)
                        print(f"ustatistic {metric_name}: {u_statistic}\n")
                        print(f"nenner {metric_name}: {n1 * n2}\n")
                        print(f"n1 {metric_name}: {n1}\n")
                        print(f"n2 {metric_name}: {n2}\n")

                        rank_biserial_corr = 1 - (2 * u_statistic) / (n1 * n2)
                    except ValueError:
                        pass

                analysis_results.append({
                    'Module': module_name,
                    'Metric': metric_name,
                    'p_value': p_value,
                    'Rank_Biserial_Correlation': rank_biserial_corr
                })

    return rename_metrics(pd.DataFrame(analysis_results)), modules

if __name__ == "__main__":
    projects_dir = os.path.join(os.getcwd(), 'projects')

    all_modules_df = process_all_modules_and_combine_data(projects_dir)
    all_modules_df.to_csv("./good_bad_run/all_modules.csv", index=False, sep=',', decimal='.')
    all_modules_df = all_modules_df[~all_modules_df['metric'].str.startswith('DIVm')]


    summary_df, modules = analyze_combined_data(all_modules_df)
    summary_df.to_csv('./good_bad_run/profile_analysis_summary_all_values.csv', index=False, sep=',', decimal='.')
    significant_results = summary_df[summary_df['p_value'] <= 0.05]
    significant_results.to_csv('./good_bad_run/profile_analyses_significant_results.csv', index=False, sep=',', decimal='.')

    strong_effects = summary_df[
        (summary_df['p_value'] <= 0.05) &
        (summary_df['Rank_Biserial_Correlation'].abs() >= 0.5)
    ]

    print(strong_effects.to_string(index = False))

    strong_effects.to_csv('./good_bad_run/profile_analysis_strong_effects.csv', index=False, sep=',', decimal='.')
    print("-- all modules with coverage diff -- ")
    for module in modules:
        print(module)
        print("\n")
    print("-- alll modules with coverage diff >= threshold -- ")
    for module in modules:
        if module[1] >= min_coverage_spread_threshold:
            print(module[0])
            print("\n")
