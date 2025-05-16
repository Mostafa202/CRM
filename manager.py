
import pandas as pd
from preprocess import preprocess_train
from preprocess import preprocess_test
from Modeling import training_models
from predictions import predict
import streamlit as st
from PIL import Image



# Insert icon of web app
icon = Image.open("img.jpg")
#  Page Layout
st.set_page_config(page_title="Root Cause Analysis", page_icon=icon)
st.markdown('# Root Cause Analysis')

# CSS codes to improve the design of the web app
st.markdown(
"""
<style>
h1 {text-align: center;
}
body {background-image: url('./dt.png');
      width: 1400px;
      margin: 15px auto;
}
</style>""",
    unsafe_allow_html=True,
)

# Insert image
logo = Image.open("crm.png")
st.image(logo, width=500)


# Insert image into left side section
img = Image.open("img.jpg")
st.sidebar.image(img)
# Sidebar - collects user input features into dataframe
with st.sidebar.header("Test Data: 1. Upload the csv data"):
    accounts = st.sidebar.file_uploader("Upload your csv accounts", type=["csv"])
    sales = st.sidebar.file_uploader("Upload your csv sales", type=["csv"])
    products = st.sidebar.file_uploader("Upload your csv products", type=["csv"])


if st.button("Training Models"):
    st.header("**Training Models**")
    st.write("---")
    accounts_df = pd.read_csv('accounts.csv')
    sales_df = pd.read_csv('sales_pipeline.csv')
    products_df = pd.read_csv('products.csv')
    df = preprocess_train(accounts_df,products_df,sales_df)
    st.write('Preprocessing Done.')
    st.header("*DataSet Sample*")
    st.write(df.sample(3))
    st.write('Training ML Models...')
    training_models(df)
    st.write('Training Done.')

if st.button("Analyzing"):
    st.header("**Analyzing**")
    st.write("---")
    if accounts is not None and sales is not None and products is not None:
        # @st.cache
        def load_csv():
            accounts_data = pd.read_csv(accounts)
            sales_data = pd.read_csv(sales)
            products_data = pd.read_csv(products)

            return accounts_data,sales_data,products_data

        accounts_data_get,sales_data_get,products_data_get = load_csv()

        processed_data = preprocess_test(accounts_data_get,products_data_get,sales_data_get)
        st.write('Preprocessing Done.')
        for i in range(len(processed_data)):
            st.header("*DataSet Sample*")
            st.write(processed_data.loc[i:i])
            predictions,final_data = predict(processed_data.loc[i:i])

            if len(predictions)==1:
                st.write('Won Deal')
            else:
                st.write('Loss Deal')
                st.write('The Cause of Loss Deal is::: ')
                st.write(predictions[0],'::',final_data[str(predictions[0])][i])
                st.write(predictions[1],'::',final_data[str(predictions[1])][i])


   
      





    







