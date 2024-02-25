import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pandas_datareader.data as web

# Function to fetch bond market data from FRED
def fetch_bond_market_data(start_date, end_date):
    try:
        data = web.DataReader(['DGS2', 'DGS5', 'DGS10', 'DGS30'], 'fred', start=start_date, end=end_date)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Function to perform Monte Carlo simulation for bond prices
def monte_carlo_bond_pricing(mean_yield, volatility, num_simulations=1000, maturity_period=10, face_value=1000, coupon_rate=0.05):
    simulated_yield_rates = np.random.normal(mean_yield, volatility, num_simulations)
    bond_prices = []
    for yield_rate in simulated_yield_rates:
        cash_flows = [coupon_rate * face_value] * maturity_period
        cash_flows[-1] += face_value
        discounted_cash_flows = [cf / ((1 + yield_rate) ** (i + 1)) for i, cf in enumerate(cash_flows)]
        bond_prices.append(sum(discounted_cash_flows))
    return bond_prices

# Function to generate daily bond market report text and visualizations
def generate_daily_bond_market_report(bond_data, bond_types, num_simulations):
    st.title('Daily Bond Market Report')
    for bond_type in bond_types:
        st.subheader(f'{bond_type} Bond Market')
        if bond_type in bond_data.columns:
            # Explanation text
            explanation_text = f"Price Range Explanation for {bond_type} Bond:\n\n"
            explanation_text += "- Bond prices are distributed around the mean yield with volatility.\n\n"
            explanation_text += "- Prices closer to the minimum may indicate low demand or risk aversion.\n\n"
            explanation_text += "- Prices closer to the maximum may indicate high demand or market optimism.\n\n"
            explanation_text += "- Prices in the middle may indicate stable market conditions.\n"
            st.text(explanation_text)

            # Check for missing values
            if bond_data[bond_type].isnull().any():
                st.warning(f"Missing data for {bond_type} bond.")
            else:
                # Calculate bond prices
                bond_prices = monte_carlo_bond_pricing(bond_data[bond_type].mean(), bond_data[bond_type].std(), num_simulations)

                # Create a DataFrame for statistical summary
                summary_data = {
                    'Statistic': ['Minimum Price', 'Maximum Price', 'Median Price', 'Standard Deviation'],
                    'Value': [min(bond_prices), max(bond_prices), np.median(bond_prices), np.std(bond_prices)]
                }
                summary_df = pd.DataFrame(summary_data)

                # Display the statistical summary table
                st.write(summary_df)

                # Plot distribution of bond prices
                st.subheader('Distribution of Bond Prices')
                fig, ax = plt.subplots()
                sns.histplot(bond_prices, bins=30, kde=True, ax=ax)
                plt.title(f'Distribution of {bond_type} Bond Prices')
                plt.xlabel('Bond Price')
                plt.ylabel('Frequency')
                st.pyplot(fig)
        else:
            st.warning(f"Data not available for {bond_type} bond.")

if __name__ == "__main__":
    # Sidebar - Input parameters
    st.sidebar.title('Simulation Parameters')
    start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input("End Date", datetime.now())
    num_simulations = st.sidebar.slider("Number of Simulations", min_value=100, max_value=5000, value=1000)

    # Fetch bond market data
    st.write("Fetching data...")
    bond_data = fetch_bond_market_data(start_date, end_date)
    if bond_data is not None:
        if bond_data.empty:
            st.warning("No data fetched.")
        else:
            st.write("Fetched data column names:")
            st.write(bond_data.columns)

            bond_types = ['DGS2', 'DGS5', 'DGS10', 'DGS30']

            # Generate daily bond market report
            generate_daily_bond_market_report(bond_data, bond_types, num_simulations)
