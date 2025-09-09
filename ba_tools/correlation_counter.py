import pandas as pd
import glob

def count_correlations_per_module(df):
    id_vars = ['metric_name', 'module']

    correlation_columns = [col for col in df.columns if col.startswith('correlation_')]

    df_unpivoted = df.melt(id_vars = id_vars,
                            value_vars=correlation_columns,
                            var_name='korrelation_volltyp',
                            value_name='korrelationswert')

    df_unpivoted[['korrelation_typ', 'beschreibung']] = df_unpivoted['korrelation_volltyp'].str.split('_', expand=True).iloc[:, 1:3]

    df_filtered = df_unpivoted[df_unpivoted['korrelationswert'].abs() >= 0.5].copy()

    ergebnis = df_filtered.groupby(
            ['metric_name', 'korrelation_typ']
    )['module'].nunique().reset_index(name='Anzahl der Module')

    return ergebnis


def zaehle_korrelationen_pro_feature(df):
    id_vars = ['metric_name']
    correlation_columns = [col for col in df.columns if col.startswith('correlation_')]

    df_unpivoted = df.melt(id_vars=id_vars,
                           value_vars=correlation_columns,
                           var_name='korrelation_volltyp',
                           value_name='korrelationswert')

    df_unpivoted[['korrelation_typ', 'beschreibung']] = df_unpivoted['korrelation_volltyp'].str.split('_', expand=True).iloc[:, 1:3]

    df_filtered = df_unpivoted[df_unpivoted['korrelationswert'].abs() >= 0.5].copy()
    ergebnis_df = df_filtered.groupby(
        ['metric_name', 'korrelation_typ', 'beschreibung']
    ).size().reset_index(name='anzahl_treffer')

    ergebnis_df = ergebnis_df.sort_values(
        by=['metric_name', 'korrelation_typ', 'beschreibung']
    ).reset_index(drop=True)

    treffer_pro_modul = df_filtered['metric_name'].value_counts().reset_index()
    treffer_pro_modul.columns = ['metric_name', 'anzahl_treffer']

    ergebnis_df = ergebnis_df.sort_values(by='metric_name', ascending=False)

    return ergebnis_df

def zaehle_korrelationen_pro_modul(df):
    id_vars = ['module']
    correlation_columns = [col for col in df.columns if col.startswith('correlation_')]

    df_unpivoted = df.melt(id_vars=id_vars,
                           value_vars=correlation_columns,
                           var_name='korrelation_volltyp',
                           value_name='korrelationswert')

    df_unpivoted[['korrelation_typ', 'beschreibung']] = df_unpivoted['korrelation_volltyp'].str.split('_', expand=True).iloc[:, 1:3]

    df_filtered = df_unpivoted[df_unpivoted['korrelationswert'].abs() >= 0.5].copy()
    ergebnis_df = df_filtered.groupby(
        ['module', 'korrelation_typ', 'beschreibung']
    ).size().reset_index(name='anzahl_treffer')

    ergebnis_df = ergebnis_df.sort_values(
        by=['module', 'korrelation_typ', 'beschreibung']
    ).reset_index(drop=True)

    treffer_pro_modul = df_filtered['module'].value_counts().reset_index()
    treffer_pro_modul.columns = ['module', 'anzahl_treffer']


all_files = glob.glob("corrs/*.csv")
dataframes = []
for filename in all_files:
    if filename != "starke_korrelationen.csv":
        df = pd.read_csv(filename, index_col=None, header=0)

        metrik_name = filename.replace('.csv', '').replace("corrs/", '')
        df['metric_name'] = metrik_name

        dataframes.append(df)
merged_df = pd.concat(dataframes, axis=0, ignore_index=True)

ergebnis_df = zaehle_korrelationen_pro_modul(merged_df)
feature_df = zaehle_korrelationen_pro_feature(merged_df)
feature_module_df = count_correlations_per_module(merged_df)

# print(ergebnis_df.to_string(index = False))
# print(feature_df.to_string(index = False))
feature_module_df = feature_module_df[feature_module_df['korrelation_typ'] == 'delta']
print(feature_module_df.to_string(index = False))
