import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from fp_growth import perform_fpgrowth, format_results, preprocess_data, generate_recommendations

# Bước 1: Tải lên file CSV và xử lý
# uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
# if uploaded_file is not None:
    # Chuyển đổi file đã tải thành danh sách các giao dịch
    # data = pd.read_csv(uploaded_file, header=None)  # Đọc file CSV mà không cần tiêu đề
    # transactions = data.apply(lambda x: x.dropna().tolist(), axis=1).tolist()  # Chuyển về danh sách các giao dịch

try:
    dataset = pd.read_excel('online_retail_II.xlsx', sheet_name='Year 2010-2011')
    dataset['StockCode'] = dataset['StockCode'].astype(str)
    st.write(dataset.head())

    dataset = preprocess_data(dataset)
except FileNotFoundError:
    st.error("The dataset file 'online_retail_II.xlsx' was not found. Please check the file path.")
    # Tiền xử lý dữ liệu
    # te = TransactionEncoder()
    # te_ary = te.fit(transactions).transform(transactions)
    # df = pd.DataFrame(te_ary, columns=te.columns_)
    # st.write("Dataframe của các giao dịch:")
    # st.write(df)

    # Chọn min_support và tìm tập phổ biến
min_support = st.slider('Select min_support value', min_value=0.0, max_value=1.0, step=0.01)
frequent_itemsets = fpgrowth(dataset, min_support=min_support, use_colnames=True)
st.write("Tập phổ biến:")
st.table(frequent_itemsets)

    # Bước 2: Chọn các item cho nhóm A và nhóm B
items = list(dataset.columns)
group_A = st.multiselect("Select items for Group A", items)
group_B = st.multiselect("Select items for Group B", items)

if group_A and group_B:
        # Tính support của A và B
    set_A = set(group_A)
    set_B = set(group_B)

        # Tính Support(A ∪ B) và Support(A)
    support_A = dataset[dataset[group_A].all(axis=1)].shape[0] / dataset.shape[0]
    support_A_union_B = dataset[dataset[group_A + group_B].all(axis=1)].shape[0] / dataset.shape[0]

        # Tính Confidence(A ⇒ B) = Support(A ∪ B) / Support(A)
    if support_A > 0:
        confidence_A_to_B = support_A_union_B / support_A
        st.write(f"Độ tin cậy (confidence) của luật A ⇒ B là: {confidence_A_to_B:.2f}")
    else:
        st.write("Support của nhóm A bằng 0 nên không thể tính độ tin cậy.")
