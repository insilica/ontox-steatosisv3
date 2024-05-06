import matplotlib.pyplot as plt, seaborn as sns, pandas as pd

metrics = pd.read_csv('cache/02_analyze/metrics.csv')

#%% AGGREGATE PLOT ====================================================================
correct_cols = ['TP', 'TN', 'FP', 'FN', 'P', 'N']
tot_metrics = metrics.groupby('short_label')[correct_cols].sum()
tot_metrics['Sensitivity'] = tot_metrics['TP'] / tot_metrics['P']
tot_metrics['Specificity'] = tot_metrics['TN'] / tot_metrics['N']
tot_metrics['Accuracy'] = (tot_metrics['TP'] + tot_metrics['TN']) / (tot_metrics['P'] + tot_metrics['N'])
tot_metrics['Balanced Accuracy'] = (tot_metrics['Sensitivity'] + tot_metrics['Specificity']) / 2
tot_metrics = tot_metrics.sort_values('Balanced Accuracy', ascending=False)
tot_metrics = tot_metrics[['Sensitivity', 'Specificity', 'Accuracy', 'Balanced Accuracy','TP', 'TN', 'FP', 'FN', 'P', 'N']]
tot_metrics.to_csv('cache/04_build_figures/tot_metrics.csv')

colors = {
    "Location": "#377538",
    "Disease": "#99C9EC",
    "Weather Variable": "#D9CD7F",
    "Weather Disease Impact": "#9D4C97",
    "Include": "#322185",
    "Measures of Disease": "#BE6E78"
}

# Build a damn awesome plot for the balanced accuracy
df_sorted = tot_metrics.sort_values(by="Balanced Accuracy", ascending=False)
plt.figure(figsize=(7, 10))
barplot = sns.barplot(
    x='Balanced Accuracy', y='short_label', data=df_sorted,
    palette=colors.values()  # Use the first N colors from the solarized palette
)

plt.title('Balanced Accuracy by Label')
plt.xlabel('Balanced Accuracy')
plt.ylabel('')
plt.grid(True, linestyle='--', alpha=0.6)  # Add grid lines for better readability
plt.gca().set_facecolor((1, 1, 1, 0))  # Transparent background on the plot area
plt.gcf().set_facecolor((1, 1, 1, 0))  # Transparent background on the figure
plt.tight_layout()
plt.savefig('balanced_accuracy.png')

#%% CATEGORY SPECIFIC PLOT =================================================================
# sort metrics in order of aggregated balanced accuracy and then by balanced accuracy
pdf = metrics.merge(tot_metrics[['Balanced Accuracy']], on='short_label', suffixes=('', '_aggregate'))
pdf = pdf.sort_values(by=['Balanced Accuracy_aggregate','Balanced Accuracy'], ascending=False)
pdf['colors'] = pdf['short_label'].map(colors)
pdf['label'] = pdf['short_label'] + ' = ' + pdf['answer'].astype(str)

pdf.to_csv('cache/04_build_figures/by_category_metrics.csv')
plt.figure(figsize=(7, 10))
colors = list(pdf['colors'])
barplot = sns.barplot(x='Balanced Accuracy', y='label', data=pdf, palette=colors)

plt.title('Balanced Accuracy by Label')
plt.xlabel('Balanced Accuracy')
plt.ylabel('')
plt.grid(True, linestyle='--', alpha=0.6)  # Add grid lines for better readability
plt.gca().set_facecolor((1, 1, 1, 0))  # Transparent background on the plot area
plt.gcf().set_facecolor((1, 1, 1, 0))  # Transparent background on the figure
plt.tight_layout()
plt.savefig('balanced_accuracy_2.png')