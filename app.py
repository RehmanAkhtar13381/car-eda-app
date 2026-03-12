import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="Car EDA App",
    page_icon="car",
    layout="wide"
)

data = {
    "Manufacturer": ["Ford","VW","VW","VW","Toyota","Ford","Toyota",
                     "Toyota","Ford","VW","Toyota","Ford","Ford","VW",
                     "BMW","Toyota","Toyota","Toyota","Porsche"],
    "Model":        ["Fiesta","Polo","Polo","Golf","Yaris","Focus","Prius",
                     "Yaris","Mondeo","Golf","RAV4","Mondeo","Focus","Golf",
                     "Z4","RAV4","RAV4","RAV4","718 Cayman"],
    "Engine_Size":  [1.0,1.0,1.2,1.2,1.2,1.4,1.4,1.4,1.6,1.6,
                     1.8,1.8,2.0,2.0,2.0,2.0,2.2,2.4,4.0],
    "Fuel_Type":    ["Petrol","Petrol","Petrol","Diesel","Petrol","Petrol",
                     "Hybrid","Petrol","Diesel","Diesel","Hybrid","Diesel",
                     "Diesel","Diesel","Petrol","Hybrid","Petrol","Hybrid","Petrol"],
    "Year":         [2002,2006,2012,2007,1992,2018,2015,1998,2014,1989,
                     1988,2010,1992,2014,1990,2018,2007,2003,2016],
    "Mileage":      [127300,127869,73470,92697,245990,33603,30663,97286,
                     39190,222390,210814,86686,262514,83047,293666,28381,
                     79393,117425,57850],
    "Price":        [3074,4101,9977,7792,720,29204,30297,4046,24072,933,
                     1705,14350,1049,17173,719,52671,16026,11667,49704]
}

df = pd.DataFrame(data)

st.sidebar.title("Car EDA App")
st.sidebar.markdown("---")

menu = st.sidebar.selectbox("Select Section", [
    "Home",
    "Dataset",
    "Basic Info",
    "EDA Analysis",
    "Visualizations",
    "Correlation"
])

manufacturer_filter = st.sidebar.multiselect(
    "Filter by Manufacturer",
    options=df["Manufacturer"].unique(),
    default=list(df["Manufacturer"].unique())
)

fuel_filter = st.sidebar.multiselect(
    "Filter by Fuel Type",
    options=df["Fuel_Type"].unique(),
    default=list(df["Fuel_Type"].unique())
)

df_filtered = df[
    (df["Manufacturer"].isin(manufacturer_filter)) &
    (df["Fuel_Type"].isin(fuel_filter))
]

if menu == "Home":
    st.title("Car Dataset EDA App")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Cars",    len(df))
    col2.metric("Manufacturers", df["Manufacturer"].nunique())
    col3.metric("Avg Price",     f"{df['Price'].mean():,.0f}")
    col4.metric("Avg Mileage",   f"{df['Mileage'].mean():,.0f}")

    st.markdown("---")
    st.subheader("Dataset Preview")
    st.dataframe(df, use_container_width=True)

elif menu == "Dataset":
    st.title("Full Dataset")
    st.markdown("---")
    st.dataframe(df_filtered, use_container_width=True)
    col1, col2 = st.columns(2)
    col1.metric("Filtered Rows",  len(df_filtered))
    col2.metric("Total Columns",  len(df_filtered.columns))

elif menu == "Basic Info":
    st.title("Basic Information")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Types")
        st.dataframe(pd.DataFrame(df.dtypes,
                     columns=["Data Type"]),
                     use_container_width=True)
    with col2:
        st.subheader("Missing Values")
        st.dataframe(pd.DataFrame(df.isnull().sum(),
                     columns=["Missing"]),
                     use_container_width=True)

    st.markdown("---")
    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Manufacturer Count")
        st.dataframe(df["Manufacturer"].value_counts(),
                     use_container_width=True)
    with col2:
        st.subheader("Fuel Type Count")
        st.dataframe(df["Fuel_Type"].value_counts(),
                     use_container_width=True)

elif menu == "EDA Analysis":
    st.title("EDA Analysis")
    st.markdown("---")

    st.subheader("Avg Price by Manufacturer")
    st.dataframe(
        df.groupby("Manufacturer")["Price"]
          .mean().round(2)
          .reset_index()
          .rename(columns={"Price": "Avg Price"}),
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("Avg Price by Fuel Type")
    st.dataframe(
        df.groupby("Fuel_Type")["Price"]
          .mean().round(2)
          .reset_index()
          .rename(columns={"Price": "Avg Price"}),
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("Outlier Detection IQR Method")
    for col in ["Price", "Mileage", "Engine_Size"]:
        Q1      = df[col].quantile(0.25)
        Q3      = df[col].quantile(0.75)
        IQR     = Q3 - Q1
        outliers = df[
            (df[col] < Q1 - 1.5 * IQR) |
            (df[col] > Q3 + 1.5 * IQR)
        ]
        st.write(f"{col} Outliers Found: {len(outliers)}")
        if len(outliers) > 0:
            st.dataframe(outliers, use_container_width=True)

elif menu == "Visualizations":
    st.title("Visualizations")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(data=df_filtered, x="Price",
                     kde=True, color="#3b82f6", ax=ax)
        ax.set_title("Price Distribution")
        ax.grid(True)
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(data=df_filtered, x="Mileage",
                     kde=True, color="#f97316", ax=ax)
        ax.set_title("Mileage Distribution")
        ax.grid(True)
        st.pyplot(fig)
        plt.close()

    with col3:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(x="Fuel_Type", data=df_filtered,
                      color="#2ecc71", ax=ax)
        ax.set_title("Fuel Type Count")
        ax.grid(True)
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(x="Manufacturer", y="Price",
                    data=df_filtered,
                    color="#e74c3c", ax=ax)
        ax.set_title("Avg Price by Manufacturer")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True)
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(df_filtered["Mileage"],
                   df_filtered["Price"],
                   color="#8b5cf6", alpha=0.8, s=80)
        ax.set_title("Mileage vs Price")
        ax.set_xlabel("Mileage")
        ax.set_ylabel("Price")
        ax.grid(True)
        st.pyplot(fig)
        plt.close()

    with col3:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(df_filtered["Year"],
                   df_filtered["Price"],
                   color="#f59e0b", alpha=0.8, s=80)
        ax.set_title("Year vs Price")
        ax.set_xlabel("Year")
        ax.set_ylabel("Price")
        ax.grid(True)
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(x="Fuel_Type", y="Price",
                    data=df_filtered,
                    hue="Fuel_Type", legend=False, ax=ax)
        ax.set_title("Price by Fuel Type")
        ax.grid(True)
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.violinplot(x="Fuel_Type", y="Price",
                       data=df_filtered,
                       hue="Fuel_Type", legend=False, ax=ax)
        ax.set_title("Violin Price by Fuel Type")
        ax.grid(True)
        st.pyplot(fig)
        plt.close()

    with col3:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(df_filtered["Engine_Size"],
                   df_filtered["Price"],
                   color="#10b981", alpha=0.8, s=80)
        ax.set_title("Engine Size vs Price")
        ax.set_xlabel("Engine Size")
        ax.set_ylabel("Price")
        ax.grid(True)
        st.pyplot(fig)
        plt.close()

elif menu == "Correlation":
    st.title("Correlation Heatmap")
    st.markdown("---")

    corr = df[["Engine_Size","Year","Mileage","Price"]].corr()

    st.subheader("Correlation Matrix")
    st.dataframe(corr.round(2), use_container_width=True)

    st.markdown("---")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f",
                cmap="coolwarm", ax=ax,
                linewidths=0.5)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)
    plt.close()