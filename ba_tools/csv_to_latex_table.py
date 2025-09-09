import csv
from string import Template

def mark_correlation(values: list):
    result = []
    if "_" in values[0]:
        values[0] = values[0].replace("_", r"\_")
    result.append(values[0])
    for value in values[1:]:
        if value.lower() != "nan":
            floated_value = round(float(value), 2)
            if abs(floated_value) >= 0.5:
                result.append(rf"\textbf{{{floated_value}}}")
            else:
                result.append(floated_value)
        else:
            result.append(value)
    return result


metrics = [
    "diversity"
]

header = Template(r"""
\begin{table}[htbp]
    \centering
    \caption{\textbf{Results of the correlation analysis for ${metric} }}
    \label{tab:${metric}}
    \scalebox{0.6}{
        \begin{tabular}{|l|*{12}{c|}}\hline
        \textbf{Module} & \multicolumn{4}{c|}{\textbf{Direct Correlation}} & \multicolumn{4}{c|}{\textbf{Delta Value Correlation}} & \multicolumn{4}{c|}{\textbf{Early Correlation}} \\
        \cline{2-13}
        & \textbf{mean} & \textbf{median} & \textbf{max} & \textbf{min} & \textbf{mean} & \textbf{median} & \textbf{max} & \textbf{min} & \textbf{mean} & \textbf{median} & \textbf{max} & \textbf{min} \\
        \hline""")


footer ="\t\t" + r'''\end{tabular}
    }
\end{table}
'''
for metric in metrics:
    template_data = {
        "metric": metric,
        "sign": "$"
    }
    resulting_table = [header.substitute(template_data)]
    with open(f"./corrs/{metric}.csv", 'r') as csvfile:
        reader = csv.reader(csvfile)
        print(reader.line_num)
        next(reader, None)
        for row in reader:
            result = mark_correlation(row)
            resulting_table.append("\t\t"
                                    + fr"\texttt{{{result[0]}}}"
                                    + "\t&\t" + f"{result[1]}"
                                    + "\t&\t" + f"{result[2]}"
                                    + "\t&\t" + f"{result[3]}"
                                    + "\t&\t" + f"{result[4]}"
                                    + "\t&\t" + f"{result[5]}"
                                    + "\t&\t" + f"{result[6]}"
                                    + "\t&\t" + f"{result[7]}"
                                    + "\t&\t" + f"{result[8]}"
                                    + "\t&\t" + f"{result[9]}"
                                    + "\t&\t" + f"{result[10]}"
                                    + "\t&\t" + f"{result[11]}"
                                    + "\t&\t" + f"{result[12]}"
                                    + r"\\")
            resulting_table.append("\t\t" + r"\hline")

        resulting_table.append(footer)
        final_string = "\n".join(resulting_table)
        print(final_string)
