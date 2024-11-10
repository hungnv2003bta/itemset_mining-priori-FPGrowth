import streamlit as st
import pandas as pd
from apriori import runApriori, to_str_results, recommendation_system_func, preprocessData, check_id

# Initialize session state variables
if 'clicks' not in st.session_state:
    st.session_state['clicks'] = {}

if 'sorted_rules' not in st.session_state:
    st.session_state['sorted_rules'] = None
    st.session_state['items'] = None
    st.session_state['rules'] = None

# Helper functions to handle button clicks
def run_apriori_click():
    st.session_state.clicks['run_apriori'] = True

def reset_apriori_click():
    st.session_state.clicks['run_apriori'] = False

def run_recommendation_click():
    st.session_state.clicks['run_recommendation'] = True

def reset_recommendation_click():
    st.session_state.clicks['run_recommendation'] = False

def main():
    st.markdown("# Apriori Algorithm")

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
    with st.form(key='apriori_form'):
        support = st.slider("Minimum Support Value", min_value=0.0, max_value=0.99, value=0.05, key='support_slider')
        confidence = st.slider("Minimum Confidence Value", min_value=0.0, max_value=0.99, value=0.01, key='confidence_slider')
        submit_button = st.form_submit_button(label='Run Apriori', on_click=run_apriori_click)

    # Run the Apriori algorithm if the button was clicked
    if st.session_state.clicks.get('run_apriori', False):
        try:
            items, sorted_rules = runApriori(dataset, support, confidence)
            i, r = to_str_results(items, sorted_rules)

            # Store results in session state
            st.session_state['sorted_rules'] = sorted_rules
            st.session_state['items'] = i
            st.session_state['rules'] = r

            st.success("Apriori Algorithm completed successfully.")
        except Exception as e:
            st.error(f"An error occurred during Apriori processing: {e}")

    # Display results if Apriori was run
    if st.session_state.clicks.get('run_apriori', False):
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

    with st.form(key='recommendation_form'):
        product_id = st.text_input("Enter a Product ID (e.g., '22326')", "22326").strip()
        num_of_products_input = st.text_input("Enter the Number of Products to Recommend", "5")
        submit_recommendation = st.form_submit_button(label='Run Recommendation', on_click=run_recommendation_click)

    try:
        num_of_products = int(num_of_products_input)
        if num_of_products <= 0:
            st.error("Number of products must be a positive integer.")
            num_of_products = 5
    except ValueError:
        st.error("Please enter a valid integer for the number of products.")
        num_of_products = 5

    if st.session_state.clicks.get('run_recommendation', False):
        # Generate recommendations if Apriori was run
        if st.session_state.clicks.get('run_apriori', False):
            try:
                recommended_products = recommendation_system_func(df, product_id, num_of_products, st.session_state['sorted_rules'])
                if recommended_products:
                    if recommended_products == "Invalid Product ID":
                        st.warning("Invalid Product ID. Please enter a valid Product ID.")
                    else:
                        product_name = check_id(df, product_id)[1]
                        st.markdown(f"#### Recommended Products with id: {product_id} - name: {product_name}")
                        recommended_df = pd.DataFrame(recommended_products, columns=["product_id", "name"])
                        st.write(recommended_df)
                else:
                    st.warning("No recommendations found for the given Product ID.")
            except Exception as e:
                st.error(f"An error occurred while generating recommendations: {e}")
        else:
            st.warning("Please run the Apriori algorithm first to get recommendations.")

if __name__ == "__main__":
    main()
