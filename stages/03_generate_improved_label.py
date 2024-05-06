import pandas as pd, sqlite3, json, dotenv, os, re
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score

conn = sqlite3.connect('.sr/sr.sqlite')

evaldf = pd.read_csv('cache/02_analyze/evaluate.csv')
labels = pd.read_sql("SELECT * FROM labels WHERE short_label = 'Include'", conn)
articles = pd.read_sql('SELECT * FROM csl_citations', conn)
articles['text'] = articles['title'] + '\n' + articles['abstract']
articles = articles[['article_id', 'text']]

# get all the inclusion misclassified articles
incdf = evaldf[evaldf['short_label'] == 'Include']
incdf = incdf[['article_id', 'user_answer', 'prediction']]
incdf = incdf.merge(articles, on='article_id', how='left').drop_duplicates()
incdf.groupby(['user_answer','prediction']).count()

## get the current inclusion criteria
criteria = labels['question'][0]

good_inclusions = incdf[(incdf['user_answer'] == 1) & (incdf['prediction'] == 1)]
bad_inclusions = incdf[(incdf['user_answer'] == 0) & (incdf['prediction'] == 1)]
good_exclusions = incdf[(incdf['user_answer'] == 0) & (incdf['prediction'] == 0)]
bad_exclusions = incdf[(incdf['user_answer'] == 1) & (incdf['prediction'] == 0)]

# Apply sample limit
bad_inclusions_sample = bad_inclusions.sample(min(50, len(bad_inclusions)))
good_exclusions_sample = good_exclusions.sample(min(10, len(good_exclusions)))
bad_exclusions_sample = bad_exclusions.sample(min(50, len(bad_exclusions)))
good_inclusions_sample = good_inclusions.sample(min(5, len(good_inclusions)))

# Convert to formatted text blocks
bad_inclusions_text = '\n\n----------------\n\n'.join(bad_inclusions_sample['text'].tolist())
good_exclusions_text = '\n\n----------------\n\n'.join(good_exclusions_sample['text'].tolist())
bad_exclusions_text = '\n\n----------------\n\n'.join(bad_exclusions_sample['text'].tolist())
good_inclusions_text = '\n\n----------------\n\n'.join(good_inclusions_sample['text'].tolist())

# get metrics
metrics = pd.read_csv('cache/02_analyze/metrics.csv').query('short_label == "Include"').iloc[0]
prompt = f"""
The current inclusion criteria is:

{criteria}

In total there were {len(good_inclusions)} correctly included articles, {len(bad_inclusions)} incorrectly included articles (that should have been excluded), {len(good_exclusions)} good exclusions, and {len(bad_exclusions)} bad exclusions (that should have been included).

The following is a sample of labeled articles that are grouped by whether they were correctly included. 
\n
```good inclusions - articles were correctly included with our criteria
{good_inclusions_text}
```
\n
```bad inclusions - articles were incorrectly included with our criteria
{bad_inclusions_text}
```
\n
```good exclusions - articles were correctly excluded with our criteria
{good_exclusions_text}
```
\n
```bad exclusions - articles were incorrectly excluded with our criteria
{bad_exclusions_text}
```
\n
Update the inclusion criteria to include articles that were bad exclusions and to exclude articles that were bad inclusions. Make sure the new criteria does not create any new errors on existing good includes and good excludes. The new criteria should be in a set of bullets giving reasons to include and a set of bullets giving reasons to exclude. output the updated criteria then tell me about what you changed and why. Give the sections headers that explain that articles need to meet all of the inclusion criteria and that articles should be excluded if they meet any of the exclusion criteria.
"""

# write prompt to file cache/02_analyze/improve_inclusion_criteria.txt
os.makedirs('cache/03_generate_improved_label', exist_ok=True)
with open('cache/03_generate_improved_label/improve_inclusion_criteria.txt', 'w') as f:
    f.write(prompt)