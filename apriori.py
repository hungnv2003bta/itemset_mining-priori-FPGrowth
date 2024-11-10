import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

def preprocessData(df):
    df.dropna(inplace=True)
    df = df[~df["Invoice"].str.contains("C", na=False)]
    
    num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"] and "ID" not in col]
    for col in num_cols:
        replace_with_thresholds(df, col)

    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    df_germany = df[df["Country"] == "Germany"]
    gr_inv_pro_df = create_invoice_product_df(df_germany, id=True)
    
    return gr_inv_pro_df

def runApriori(data, minSupport=0.01, minConfidence=0.5):
    # gr_inv_pro_df = preprocessData(data)

    frequent_itemsets = apriori(data, min_support=minSupport, use_colnames=True)
    rules = association_rules(frequent_itemsets, num_itemsets=len(frequent_itemsets), metric="confidence", min_threshold=minConfidence)

    sorted_rules = rules.sort_values("confidence", ascending=False)
    return frequent_itemsets, sorted_rules

def to_str_results(frequent_itemsets, rules):
    itemsets_str = []
    rules_str = []

    for _, row in frequent_itemsets.iterrows():
        itemsets_str.append(f"itemset: {set(row['itemsets'])}, support: {row['support']:.3f}")

    for _, row in rules.iterrows():
        antecedents = set(row['antecedents'])
        consequents = set(row['consequents'])
        confidence = row['confidence']
        rules_str.append(f"Rule: {antecedents} ==> {consequents}, confidence: {confidence:.3f}")

    return itemsets_str, rules_str

def recommendation_system(product_id, support_val, num_of_products, sorted_rules):
    recommendation_list = []
    for idx, product in enumerate(sorted_rules['antecedents']):
        for j in list(product):
            if j == product_id: 
                recommendation_list.append(list(sorted_rules.iloc[idx]["consequents"])[0])
                recommendation_list = list( dict.fromkeys(recommendation_list))
    return (recommendation_list[:num_of_products])
