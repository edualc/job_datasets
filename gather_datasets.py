import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm


dataset_paths = {
    'job': 'balsa/queries/join-order-benchmark',
    'job_extended': 'balsa/queries/join-order-benchmark-extended',
    'job_light': 'learnedcardinalities/workloads',
    'job_light_ranges': 'neurocard/neurocard/queries',
    'job_m': 'neurocard/neurocard/queries'
}


def generate_dataframe(dataset_name, data):
    df = pd.DataFrame(pd.DataFrame(data[dataset_name], index=data[dataset_name].keys()).transpose().iloc[:,0])
    df.columns = ['query']
    df['query_id'] = df.index
    df['dataset'] = dataset_name

    # reorder columns
    df = df[['query_id', 'dataset', 'query']]

    return df

# SELECT COUNT(*)
# FROM cast_info AS ci,movie_info AS mi,movie_info_idx AS mi_idx,title AS t
# WHERE t.production_year BETWEEN 2008 AND 2014 AND mi_idx.info > '8.0' AND ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)') AND mi.info IN ('Horror', 'Thriller') AND mi.note IS NULL AND t.id = mi.movie_id AND t.id = mi_idx.movie_id AND t.id = ci.movie_id AND mi.movie_id = ci.movie_id AND mi.movie_id = mi_idx.movie_id AND mi_idx.movie_id = ci.movie_id 


# SELECT MIN(t.title) AS movie_title
# FROM keyword AS k, movie_info AS mi, movie_keyword AS mk, title AS t
# WHERE k.keyword LIKE '%sequel%' AND mi.info IN ('Sweden', 'Norway', 'Germany', 'Denmark', 'Swedish', 'Denish', 'Norwegian', 'German') AND t.production_year > 2005 AND t.id = mi.movie_id AND t.id = mk.movie_id AND mk.movie_id = mi.movie_id AND k.id = mk.keyword_id  

def cleanup_query(query):
    new_query = " ".join(map(lambda x: x.strip(), query.split('\n'))).replace(';','')

    # equalize formatting of from part
    select_part = new_query.split('FROM')[0].strip()
    from_part = new_query.split('FROM')[1].split('WHERE')[0].split(',')
    from_part = list(map(lambda x: x.strip(), from_part))
    new_from_part = ", ".join(from_part)
    where_part = new_query.split('WHERE')[1].strip()
    new_query = f"{select_part} FROM {new_from_part} WHERE {where_part}"

    return new_query.strip()

def check_overlap(df, datasets, column_to_compare='query'):
    print('=================================================================')
    print(f'  CHECKING OVERLAP FOR {column_to_compare}')
    print('=================================================================')
    for ds_a in datasets:
        for ds_b in datasets:
            if ds_a == ds_b:
                continue

            if datasets.index(ds_a) > datasets.index(ds_b):
                continue

            df_a = df[df.dataset==ds_a]
            df_b = df[df.dataset==ds_b]

            print(f"{df_a.shape[0]} total queries\t({ds_a})")
            print(f"{df_b.shape[0]} total queries\t({ds_b})")

            num_queries_overlap = np.intersect1d(df_a[column_to_compare], df_b[column_to_compare]).shape[0]

            print(f"share {num_queries_overlap} identical queries.")
            print('')


def generate_all_queries_df():
    if os.path.isfile('all_job_queries.csv'):
        return pd.read_csv('all_job_queries.csv')

    data = {}

    # Read JOB
    # ==========================================================================================
    data['job'] = {}
    for file in os.listdir(dataset_paths['job']):
        if not file.endswith('.sql'):
            continue

        file_path = os.path.join(dataset_paths['job'], file)
        lines = open(file_path, 'r').readlines()

        data['job'][file.split('.sql')[0]] = cleanup_query("".join(lines))

    # Read JOB-Extended
    # ==========================================================================================
    data['job_extended'] = {}
    for file in os.listdir(dataset_paths['job_extended']):
        if not file.endswith('.sql'):
            continue

        file_path = os.path.join(dataset_paths['job_extended'], file)
        lines = open(file_path, 'r').readlines()

        data['job_extended'][file.split('.sql')[0]] = cleanup_query("".join(lines))

    # Read JOB-Light
    # ==========================================================================================
    data['job_light'] = {}
    file_path = os.path.join(dataset_paths['job_light'], 'job-light.sql')
    line_num = 0
    with open(file_path, 'r') as f:
        for line in f:
            if 'SELECT' in line:
                data['job_light'][str(line_num)] = cleanup_query(line)
                line_num += 1

    # Read JOB-Light-Ranges
    # ==========================================================================================
    data['job_light_ranges'] = {}
    file_path = os.path.join(dataset_paths['job_light_ranges'], 'job-light-ranges.sql')
    line_num = 0
    with open(file_path, 'r') as f:
        for line in f:
            if 'SELECT' in line:
                data['job_light_ranges'][str(line_num)] = cleanup_query(line)
                line_num += 1

    # Read JOB-M
    # ==========================================================================================
    data['job_m'] = {}
    file_path = os.path.join(dataset_paths['job_m'], 'job-m.sql')
    line_num = 0
    with open(file_path, 'r') as f:
        for line in f:
            if 'SELECT' in line:
                data['job_m'][str(line_num)] = cleanup_query(line)
                line_num += 1


    df_job = generate_dataframe('job', data)
    df_job_extended = generate_dataframe('job_extended', data)
    df_job_light = generate_dataframe('job_light', data)
    df_job_light_ranges = generate_dataframe('job_light_ranges', data)
    df_job_m = generate_dataframe('job_m', data)

    df = pd.concat([df_job, df_job_extended, df_job_light, df_job_light_ranges, df_job_m], axis=0)
    df = df.reset_index(drop=True)

    df['select'] = df['query'].str.split('FROM', expand=True)[0].str.strip()
    df['from'] = 'FROM ' + df['query'].str.split('FROM', expand=True)[1].str.split('WHERE', expand=True)[0].str.strip()
    df['where'] = 'WHERE ' + df['query'].str.split('FROM', expand=True)[1].str.split('WHERE', expand=True)[1].str.strip()
    df['select_from'] = (df['select'] + ' ' + df['from']).str.strip()


    datasets = list(dataset_paths.keys())
    check_overlap(df, datasets, 'query')
    check_overlap(df, datasets, 'select_from')

    for d in datasets:
        df['in_' + str(d)] = (df.dataset == d).astype(int)

    # Set 1/0 whether this query also occurs in other datasets
    for i, row in tqdm(df.iterrows(), total=len(df)):
        for d in set(df[df['query'] == row['query']].dataset):
            df['in_' + str(d)][i] = 1

        # set 0.5 in case of identical SELECT and FROM part (=where difference)
        for d in set(df[df['select_from'] == row['select_from']].dataset):
            if df['in_' + str(d)][i] != 1:
                df['in_' + str(d)][i] = 0.5

    df['has__not_like'] = df['query'].str.lower().str.contains('not like').astype(int)
    df['has__like'] = df['query'].str.lower().str.replace('not like', 'XNLX').str.contains('like').astype(int)
    df['has__between'] = df['query'].str.lower().str.contains('between').astype(int)
    df['has__not_in'] = df['query'].str.lower().str.contains("not in \(").astype(int)
    df['has__in'] = df['query'].str.lower().str.replace('not in \(', 'XNIX').str.contains('in \(').astype(int)
    df['has__or'] = df['query'].str.lower().str.contains(' or ').astype(int)
    df['has__group_by'] = df['query'].str.lower().str.contains(' group by ').astype(int)
    df['has__inequality'] = df['query'].str.lower().str.contains('!=').astype(int)
    df['has__range'] = df['query'].str.lower().str.contains('<').astype(int)
    df['has__range'] += df['query'].str.lower().str.contains('>').astype(int)
    df['has__range'] = np.minimum(1, df['has__range'])
    df['has__is_null'] = df['query'].str.lower().str.contains('is null').astype(int)
    df['has__is_not_null'] = df['query'].str.lower().str.contains('is not null').astype(int)

    df.to_csv('all_job_queries.csv', header=True, index=False, encoding='utf-8')
    return df

if __name__ == '__main__':
    df = generate_all_queries_df()

    num_filters = []
    filters = []
    # for query in df[df.dataset=='job_light_ranges'][:55]['query']:
    for query in df['query']:
        select_part = query.split('FROM')[0].strip()
        from_part = query.split('FROM')[1].split('WHERE')[0].split(',')
        where_part = query.split('WHERE')[1].strip()

        filters += where_part.split(' AND ')
        num_filters.append(len(where_part.split(' AND ')))
    
    print('')
    for f in sorted(list(set(filters))):
        print(f)
    print('')
    # print(num_filters)
    # print('')

    import code; code.interact(local=dict(globals(), **locals()))



