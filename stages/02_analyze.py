import pandas as pd, sqlite3, json, os, dotenv
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
from sysrev.client import Synchronizer, Client

conn = sqlite3.connect('.sr/sr.sqlite')
project_id = 125267
labeled_articles = pd.read_sql('SELECT article_id FROM article_label', conn)

dotenv.load_dotenv()
client = Client(os.getenv('SR_ADMIN_TOKEN'))
synchronizer = Synchronizer()
synchronizer.sync_article_info(client, project_id, labeled_articles['article_id'].values)
synchronizer.sync_labels(client, project_id)

labels = pd.read_sql('SELECT label_id, short_label, value_type FROM labels WHERE enabled = 1', conn)

# automatic answers
autolabel = pd.read_sql('SELECT article_id, short_label, answer FROM auto_labels al INNER JOIN labels lbl ON lbl.label_id = al.label_id', conn)
autolabel['answer'] = autolabel['answer'].apply(json.loads)
autolabel = autolabel.explode('answer')
autolabel['prediction'] = autolabel['answer']
autolabel = autolabel[autolabel['answer'].notnull()].drop_duplicates()

# user answers
completions = pd.read_sql("""SELECT article_id, short_label, user_id, answer FROM article_label as al 
                          inner join labels as lbl on lbl.label_id = al.label_id""", conn)
completions['answer'] = completions['answer'].apply(json.loads)
completions = completions.explode('answer')
completions['tot_users'] = completions.groupby(['article_id', 'short_label'])['user_id'].transform('nunique')
completions['ans_users'] = completions.groupby(['article_id', 'short_label', 'answer'])['user_id'].transform('nunique')
completions = completions[completions['ans_users'] == completions['tot_users']]
completions = completions.drop(columns=['tot_users', 'ans_users','user_id']).drop_duplicates()

# filter to article_id + short_label in autolabel
art_lbl = autolabel[['article_id', 'short_label']].drop_duplicates().set_index(['article_id', 'short_label'])
completions = completions.join(art_lbl, on=['article_id', 'short_label'], how='inner')
completions['user_answer'] = completions['answer']

# find all combinations of unique vlues of article_id, short_label, answer
lbl_answers = completions[['short_label', 'answer']].drop_duplicates().reset_index(drop=True)
lbl_answers = completions[['article_id']].merge(lbl_answers, how='cross')
art_lbl_ans = lbl_answers.merge(completions, on=['article_id','short_label','answer'], how='left')
art_lbl_ans = art_lbl_ans[['article_id', 'short_label', 'answer', 'user_answer']]
art_lbl_ans = art_lbl_ans.merge(autolabel, on=['article_id', 'short_label', 'answer'], how='inner')

# user_answer = 1 if user_answer == answer else 0
art_lbl_ans['user_answer'] = (art_lbl_ans['user_answer'] == art_lbl_ans['answer']).astype(int)
art_lbl_ans['prediction'] = (art_lbl_ans['prediction'] == art_lbl_ans['answer']).astype(int)

df = art_lbl_ans

# For boolean labels, reverse user_answer and prediction where answer is False and set value to True
bool_lbl = labels[labels['value_type'] == 'boolean']['short_label'].values.tolist()
df.loc[df['short_label'].isin(bool_lbl) & (df['answer'] == False), 'user_answer'] = 1 - df['user_answer']
df.loc[df['short_label'].isin(bool_lbl) & (df['answer'] == False), 'prediction'] = 1 - df['prediction']
df.loc[df['short_label'].isin(bool_lbl) & (df['answer'] == False), 'answer'] = True

# write df to csv
df.to_csv('cache/02_analyze/evaluate.csv', index=False)

def calc_metrics(x):
    # Calculate the confusion matrix
    cm = confusion_matrix(x['user_answer'], x['prediction'], labels=[1, 0])
    TP, FN, FP, TN = cm.ravel()
    
    # Calculate metrics based on the confusion matrix
    Sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
    Specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    Accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
    Balanced_Accuracy = (Sensitivity + Specificity) / 2
    articles = x['article_id'].nunique()
    
    # Calculate totals for positives and negatives
    Total_Positives = TP + FN  # Total actual positives
    Total_Negatives = TN + FP  # Total actual negatives

    return pd.Series({
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN, 'P': Total_Positives, 'N': Total_Negatives,
        'Articles': articles,  # Number of articles with this label
        'Sensitivity': Sensitivity, 'Specificity': Specificity, 
        'Accuracy': Accuracy, 'Balanced Accuracy': Balanced_Accuracy, 
    })

# Group by 'short_label' and 'answer' and apply the function
metrics = df.groupby(['short_label', 'answer']).apply(calc_metrics).reset_index()
metrics.to_csv('cache/02_analyze/metrics.csv', index=False)
print(metrics.iloc[0])

# short_label           Include
# answer                   True
# TP                       57.0
# TN                       36.0
# FP                       10.0
# FN                        8.0
# P                        65.0
# N                        46.0
# Articles                 95.0
# Sensitivity          0.876923
# Specificity          0.782609
# Accuracy             0.837838
# Balanced Accuracy    0.829766
# Name: 0, dtype: object