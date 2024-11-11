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
    # Delete if the product name contains "POST":
    df = df[~df["Description"].str.contains("POST", na=False)]
    df = df[~df["Invoice"].str.contains("C", na=False)]
    
    num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"] and "ID" not in col]
    for col in num_cols:
        replace_with_thresholds(df, col)

    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    df_germany = df[df["Country"] == "Germany"]
    gr_inv_pro_df = create_invoice_product_df(df_germany, id=True)
    
    return gr_inv_pro_df

def check_id(data, product_id):
    # Find Product name with Stock Code
    product_name = data[data["StockCode"] == product_id]["Description"].values[0]
    return product_id, product_name

def runApriori(data, minSupport, minConfidence):
    # gr_inv_pro_df = preprocessData(data)

    frequent_itemsets = apriori(data, min_support=minSupport, use_colnames=True)
    
    rules = association_rules(frequent_itemsets, metric="support",num_itemsets=len(frequent_itemsets), min_threshold=minConfidence)

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
        support = row['support']
        rules_str.append(f"Rule: {antecedents} ==> {consequents},support : {support:.3f}, confidence: {confidence:.3f}")

    return itemsets_str, rules_str

# def recommendation_system(product_ids, num_of_products, sorted_rules):
#     # Validate input
#     if not sorted_rules.empty and 'antecedents' in sorted_rules.columns and 'consequents' in sorted_rules.columns:
#         # Initialize recommendation set to avoid duplicates
#         recommendation_list = []

#         # Iterate through the rules
#         for idx, row in sorted_rules.iterrows():
#             # antecedents = row['antecedents']
#             antecedents = set(row['antecedents'])
#             consequents = row['consequents']
#             confidence = row['confidence']

#             if antecedents.issubset(set(product_ids)):
#                 for consequent in consequents:
#                     recommendation_list.append((consequent, confidence))
                    
#         recommendation_list = sorted(recommendation_list, key=lambda x: x[1], reverse=True)
#         return recommendation_list[0:num_of_products] if recommendation_list else ["No recommendations found"]
#     else:
#         return ["Invalid sorted_rules data"]

def recommendation_system(input_products, num_of_products, rules_df):    
    recommendations = []

    for _, rule in rules_df.iterrows():
        antecedents = set(rule['antecedents'])
        consequents = rule['consequents']
        confidence = rule['confidence']
        
        # Check if all input products are in the antecedents of the rule
        if antecedents.issubset(set(input_products)):
            for consequent in consequents:
                recommendations.append((consequent, confidence))
                
    # Sort by confidence and return the top N recommendations
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    return recommendations[:num_of_products]

def recommendation_system_func(df, input_products, num_of_products, sorted_rules):    
    recommendations = recommendation_system(input_products, num_of_products, sorted_rules)
    result = []

    for product_id, confidence in recommendations:
        if product_id in df['StockCode'].values:
            product_name = df[df['StockCode'] == product_id]['Description'].values[0]
            result.append({'ProductID': product_id, 'ProductName': product_name, 'Confidence': confidence})
    
    result_df = pd.DataFrame(result)
    return result_df if not result_df.empty else "No recommendations found"