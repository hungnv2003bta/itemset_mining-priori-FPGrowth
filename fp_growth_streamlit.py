import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from fp_growth import run_fpgrowth, to_str_results, recommendation_system_func, preprocessData, check_id


# Initialize session state variables
# Initialize session state variables
if 'clicks' not in st.session_state:
    st.session_state['clicks'] = {}

if 'sorted_rules' not in st.session_state:
    st.session_state['sorted_rules'] = None
    st.session_state['items'] = None
    st.session_state['rules'] = None

# Helper functions to handle button clicks
def run_fp_growth_click():
    st.session_state.clicks['run_fpgrowth'] = True

def reset_fp_growth_click():
    st.session_state.clicks['run_fpgrowth'] = False

def run_recommendation_click():
    st.session_state.clicks['run_recommendation'] = True

def reset_recommendation_click():
    st.session_state.clicks['run_recommendation'] = False

def main():
    st.markdown("# FP Growth Algorithm")

    # add info using side bar
    st.sidebar.markdown("**ABOUT DATASET**")
    st.sidebar.markdown("*Dataset Story*")
    st.sidebar.markdown("""
    - Bộ dữ liệu Online Retail II, bao gồm dữ liệu bán hàng của cửa hàng bán hàng trực tuyến tại Vương quốc Anh, đã được sử dụng.
    - Dữ liệu bán hàng từ 01/12/2009 - 09/12/2011 được bao gồm trong bộ dữ liệu.
    - Danh mục sản phẩm của công ty này bao gồm quà lưu niệm.
    """)

    st.sidebar.markdown("*INFO ABOUT THE VARIABLES*")
    st.sidebar.markdown("""
    - **InvoiceNo**: Số hóa đơn -> Nếu mã này bắt đầu bằng C, điều đó có nghĩa là giao dịch đã bị hủy.
    - **StockCode**: Mã sản phẩm -> Số duy nhất cho mỗi sản phẩm.
    - **Description**: Tên sản phẩm.
    - **Quantity**: Số lượng sản phẩm -> số lượng sản phẩm trên hóa đơn đã được bán.
    - **InvoiceDate**: Ngày lập hóa đơn.
    - **UnitPrice**: Giá mỗi đơn vị sản phẩm.
    - **CustomerID**: Số khách hàng duy nhất.
    - **Country**: Quốc gia nơi khách hàng cư trú.
    """)


    # Load and display the dataset
    try:
        df = pd.read_excel('online_retail_II.xlsx', sheet_name='Year 2010-2011')
        df['StockCode'] = df['StockCode'].astype(str)
        df['Invoice'] = df['Invoice'].astype(str)
        st.write("### Sample Data")
        st.write(df.head(15))

        # Preprocess the dataset
        dataset = preprocessData(df)
    except FileNotFoundError:
        st.error("The dataset file 'online_retail_II.xlsx' was not found. Please check the file path.")
        st.stop()

    st.markdown('---')

    # Input sliders for support and confidence inside a form
    with st.form(key='fp_growth_form'):
        support = st.slider("Minimum Support Value", min_value=0.0, max_value=0.99, value=0.05, key='support_slider')
        confidence = st.slider("Minimum Confidence Value", min_value=0.0, max_value=0.99, value=0.01, key='confidence_slider')
        submit_button = st.form_submit_button(label='Run FP Growth ', on_click=run_fp_growth_click)

    # Run the FP Growth algorithm if the button was clicked
    if st.session_state.clicks.get('run_fpgrowth', False):
        try:
            items, sorted_rules = run_fpgrowth(dataset, support, confidence)
            i, r = to_str_results(items, sorted_rules)

            # Store results in session state
            st.session_state['sorted_rules'] = sorted_rules
            st.session_state['items'] = i
            st.session_state['rules'] = r

            st.success("FP Growth Algorithm completed successfully.")
        except Exception as e:
            st.error(f"An error occurred during FP Growth processing: {e}")

    # Display results if FP Growth was run
    if st.session_state.clicks.get('run_fpgrowth', False):
        if st.session_state['items'] is not None and st.session_state['rules'] is not None:
            st.markdown("## LIST OF FREQUENT ITEMSETS AND ASSOCIATION RULES")
            st.markdown(f"### min_support: {support}  min_confidence: {confidence}")
            st.markdown("### Frequent Itemsets")
            st.write(f"Number of frequent itemsets: {len(st.session_state['items'])}")
            st.write(pd.DataFrame(st.session_state['items']).head(10))
          

            st.markdown("### Association Rules")
            st.write(f"Number of association rules: {len(st.session_state['rules'])}")
            st.write(pd.DataFrame(st.session_state['rules']).head(10))

    # Product Recommendation Section
    st.markdown('---')
    st.markdown("### Recommend Products based on a Product ID")
    
    # Input Section in Streamlit App
    input_products = st.multiselect("Select products for recommendation", list(df["StockCode"].unique()))
    num_of_products = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)

# Fetch recommendations and display
    if input_products:
        recommendations_df = recommendation_system_func(df, input_products, num_of_products, st.session_state['sorted_rules'])
        st.write("Recommendations based on selected products:")
        st.table(recommendations_df)
    else:
        st.write("Please select at least one product to get recommendations.")


    # with st.form(key='recommendation_form'):
    #     product_id = st.text_input("Enter a Product ID (e.g., '22326')", "22326").strip()
    #     num_of_products_input = st.text_input("Enter the Number of Products to Recommend", "5")
    #     submit_recommendation = st.form_submit_button(label='Run Recommendation', on_click=run_recommendation_click)

    # try:
    #     num_of_products = int(num_of_products_input)
    #     if num_of_products <= 0:
    #         st.error("Number of products must be a positive integer.")
    #         num_of_products = 5
    # except ValueError:
    #     st.error("Please enter a valid integer for the number of products.")
    #     num_of_products = 5

    # if st.session_state.clicks.get('run_recommendation', False):
    #     # Generate recommendations if Apriori was run
    #     if st.session_state.clicks.get('run_fpgrowth', False):
    #         try:
    #             recommended_products = recommendation_system_func(df, product_id, num_of_products, st.session_state['sorted_rules'])
    #             if recommended_products:
    #                 if recommended_products == "Invalid Product ID":
    #                     st.warning("Invalid Product ID. Please enter a valid Product ID.")
    #                 else:
    #                     product_name = check_id(df, product_id)[1]
    #                     st.markdown(f"#### Recommended Products with id: {product_id} - name: {product_name}")
    #                     recommended_df = pd.DataFrame(recommended_products, columns=["product_id", "name"])
    #                     st.write(recommended_df)
    #             else:
    #                 st.warning("No recommendations found for the given Product ID.")
    #         except Exception as e:
    #             st.error(f"An error occurred while generating recommendations: {e}")
    #     else:
    #         st.warning("Please run the FP Growth algorithm first to get recommendations.")

if __name__ == "__main__":
    main()








# try:
#     dataset = pd.read_excel('online_retail_II.xlsx', sheet_name='Year 2010-2011')
#     dataset['StockCode'] = dataset['StockCode'].astype(str)
#     st.write(dataset.head())

#     dataset = preprocess_data(dataset)
# except FileNotFoundError:
#     st.error("The dataset file 'online_retail_II.xlsx' was not found. Please check the file path.")
#     # Tiền xử lý dữ liệu
#     # te = TransactionEncoder()
#     # te_ary = te.fit(transactions).transform(transactions)
#     # df = pd.DataFrame(te_ary, columns=te.columns_)
#     # st.write("Dataframe của các giao dịch:")
#     # st.write(df)

#     # Chọn min_support và tìm tập phổ biến
# with st.form(key='apriori_form'):
#         min_support = st.slider("Minimum Support Value", min_value=0.0, max_value=0.99, value=0.05, key='support_slider')
#         min_confidence = st.slider("Minimum Confidence Value", min_value=0.0, max_value=0.99, value=0.01, key='confidence_slider')
#         submit_button = st.form_submit_button(label='Run Apriori', on_click=run_pf_growth_click)

# frequent_itemsets = perform_fpgrowth(dataset, min_support=min_support)
# st.write("Tập phổ biến:")
# st.table(frequent_itemsets)

#     # Bước 2: Chọn các item cho nhóm A và nhóm B
# items = list(dataset.columns)
# group_A = st.multiselect("Select items for Group A", items)
# group_B = st.multiselect("Select items for Group B", items)

# if group_A and group_B:
#         # Tính support của A và B
#     set_A = set(group_A)
#     set_B = set(group_B)

#         # Tính Support(A ∪ B) và Support(A)
#     support_A = dataset[dataset[group_A].all(axis=1)].shape[0] / dataset.shape[0]
#     support_A_union_B = dataset[dataset[group_A + group_B].all(axis=1)].shape[0] / dataset.shape[0]

#         # Tính Confidence(A ⇒ B) = Support(A ∪ B) / Support(A)
#     if support_A > 0:
#         confidence_A_to_B = support_A_union_B / support_A
#         st.write(f"Độ tin cậy (confidence) của luật A ⇒ B là: {confidence_A_to_B:.2f}")
#     else:
#         st.write("Support của nhóm A bằng 0 nên không thể tính độ tin cậy.")
