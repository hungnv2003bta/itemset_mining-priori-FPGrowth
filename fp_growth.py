import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Xác định và xử lý ngoại lệ
def detect_outliers(dataframe, column):
    q1, q3 = dataframe[column].quantile([0.01, 0.99])
    iqr = q3 - q1
    low_limit, up_limit = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return low_limit, up_limit

def adjust_outliers(dataframe, column):
    low_limit, up_limit = detect_outliers(dataframe, column)
    dataframe[column] = dataframe[column].clip(lower=low_limit, upper=up_limit)

# Xây dựng ma trận sản phẩm-hóa đơn
def build_invoice_product_matrix(dataframe, use_ids=False):
    group_cols = ['Invoice', "StockCode"] if use_ids else ['Invoice', 'Description']
    product_matrix = dataframe.groupby(group_cols)['Quantity'].sum().unstack().fillna(0)
    return product_matrix.applymap(lambda qty: 1 if qty > 0 else 0)

# Tiền xử lý dữ liệu
def preprocess_data(df):
    df = df.dropna()
    df = df[~df["Invoice"].str.contains("C", na=False)]
    numeric_cols = [col for col in df.select_dtypes(include=['int64', 'float64']) if "ID" not in col]
    
    for col in numeric_cols:
        adjust_outliers(df, col)
    
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    return build_invoice_product_matrix(df[df["Country"] == "Germany"], use_ids=True)

# Thuật toán FP-Growth và tính độ tin cậy
def perform_fpgrowth(data, min_support=0.01, min_confidence=0.5):
    frequent_itemsets = fpgrowth(data, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return frequent_itemsets, rules.sort_values("confidence", ascending=False)

# Chuyển đổi kết quả thành chuỗi
def format_results(frequent_itemsets, rules):
    itemsets = [f"Itemset: {set(row['itemsets'])}, Support: {row['support']:.3f}" for _, row in frequent_itemsets.iterrows()]
    formatted_rules = [f"Rule: {set(row['antecedents'])} => {set(row['consequents'])}, Confidence: {row['confidence']:.3f}" for _, row in rules.iterrows()]
    return itemsets, formatted_rules

# Hệ thống gợi ý sản phẩm
def generate_recommendations(product_id, rules, num_recommendations=5):
    recommendations = [list(rule["consequents"])[0] 
                       for _, rule in rules[rules['antecedents'].apply(lambda x: product_id in x)].iterrows()]
    return list(dict.fromkeys(recommendations))[:num_recommendations]

