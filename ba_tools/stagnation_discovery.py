import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import enum

stagnation_window = 10
analysis_results = []

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

    for module_name, module_df in metric_df.groupby('module'):
        print(f"Verarbeite Modul: {module_name}...")
        for metric_name, df_metric in module_df.groupby('metric'):
            df_copy = df_metric.copy()
            df_copy['coverage_diff'] = df_copy.groupby('run')['coverage'].diff().fillna(0)
            is_stagnating = df_copy.groupby('run')['coverage_diff'].transform(
                lambda x: x.rolling(window=stagnation_window).sum()
            ) <= 1e-6
            df_copy['stagnates'] = is_stagnating

            stagnating_group = df_copy[df_copy['stagnates'] == True]['result']
            non_stagnating_group = df_copy[df_copy['stagnates'] == False]['result']


            u_statistic, p_value = mannwhitneyu(stagnating_group, non_stagnating_group, alternative='two-sided')
            n1 = len(stagnating_group)
            n2 = len(non_stagnating_group)
            rank_biserial_corr = 1 - (2 * u_statistic) / (n1 * n2)

            analysis_results.append({
                'Module': module_name,
                'Metric': metric_name,
                'p_value': p_value,
                'Rank_Biserial_Correlation': rank_biserial_corr,
            })

    return pd.DataFrame(analysis_results)

if __name__ == "__main__":
    projects_dir = os.path.join(os.getcwd(), 'projects')

    # Schritt 1: Alle Daten sammeln
    all_modules_df = process_all_modules_and_combine_data(projects_dir)
    all_modules_df.to_csv("./stagnation_stuff/all_modules.csv", index=False, sep=',', decimal='.')

    if all_modules_df is not None:
        summary_df = analyze_combined_data(all_modules_df)
        summary_df = summary_df[~summary_df['Metric'].str.startswith('DIVm')]
        summary_df.to_csv('./stagnation_stuff/stagnation_analysis_summary_all_values.csv', index=False, sep=',', decimal='.')
        significant_results = summary_df[summary_df['p_value'] <= 0.05]
        significant_results.to_csv('./stagnation_stuff/stagnation_significant_results.csv', index=False, sep=',', decimal='.')

        strong_effects = summary_df[
            (summary_df['p_value'] <= 0.05) &
            (summary_df['Rank_Biserial_Correlation'].abs() >= 0.5)
        ]

        strong_effects.to_csv('./stagnation_stuff/stagnation_analysis_strong_effects.csv', index=False, sep=',', decimal='.')
