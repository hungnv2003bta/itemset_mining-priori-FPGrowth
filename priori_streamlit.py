import streamlit as st
import pandas as pd
from apriori import runApriori, to_str_results, recommendation_system, preprocessData

st.markdown("# Apriori Algorithm")

st.markdown("Here are some sample rows from the dataset")
try:
    dataset = pd.read_excel('online_retail_II.xlsx', sheet_name='Year 2010-2011')
    dataset['StockCode'] = dataset['StockCode'].astype(str)
    st.write(dataset.head())

    dataset = preprocessData(dataset)
except FileNotFoundError:
    st.error("The dataset file 'online_retail_II.xlsx' was not found. Please check the file path.")

st.markdown('---')
st.markdown('### Inputs')

support = st.slider("Minimum Support Value", min_value=0.01, max_value=0.5, value=0.15)
confidence = st.slider("Minimum Confidence Value", min_value=0.1, max_value=0.9, value=0.6)

if st.button("Run Apriori"):
    st.markdown("## Running Apriori Algorithm...")

    try:
        items, sorted_rules = runApriori(dataset, support, confidence)
        i, r = to_str_results(items, sorted_rules)

        st.markdown("## Results")

        st.markdown("### Frequent Itemsets")
        st.write(i)

        st.markdown("### Association Rules")
        st.write(r)

        st.markdown("### Recommend Products based on a Product ID")
        product_id = st.text_input("Enter a Product ID (e.g., '21987')", "21987").strip()

        num_of_products_input = st.text_input("Enter the Number of Products to Recommend", "5")
        try:
            num_of_products = int(num_of_products_input)
            if num_of_products <= 0:
                st.error("Number of products must be a positive integer.")
                num_of_products = 5  # Default value
        except ValueError:
            st.error("Please enter a valid integer for the number of products.")
            num_of_products = 5  # Default value

        if st.button("Recommend Products"):
            st.markdown("#### Recommended Products")
            try:
                # Convert product_id to string for consistent comparison
                product_id = str(product_id)
                
                # Call the recommendation function with the correct arguments
                recommendations = recommendation_system(product_id, support, num_of_products, sorted_rules)
                
                st.write(f"Recommendations for Product ID {product_id}:")
                if recommendations:
                    st.write(recommendations)
                else:
                    st.write("No recommendations found for the given Product ID.")
            except Exception as e:
                st.error(f"An error occurred while generating recommendations: {e}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
