# Import dependencies.
import os
import tempfile
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
from fpdf import FPDF
from scipy.optimize import root_scalar


class BondApp:
    def __init__(self):
        """
        Initialize the BondApp class with default parameters.
        """
        self.bond_symbol = None
        self.face_value = None
        self.coupon_rate = None
        self.maturity_period = None
        self.num_simulations = None
        self.mean_yield = None
        self.volatility = None
        self.bond_pricer = None

    def run(self):
        """
        Run the Streamlit application.
        """  

        # Display instructions and information about the application.
        st.title("Government Bond Pricer")
        st.write("Welcome to the Government Bond Pricer! This tool allows you to analyze and price US government bonds.")
        # Additional information and instructions.
        st.markdown("## Exporting Results")
        st.write("Export analysis results to PDF or CSV format for further reference or sharing.")
        st.write("You can export scenario and sensitivity analysis results, including histograms, summary statistics, and price range explanations.")
        # User instructions and information about bond selection.
        st.markdown("## Bond Selection")
        st.write("Please select a US government bond symbol from the dropdown menu below:")

        self.select_bond()

        if self.bond_symbol:
            self.fetch_bond_data()
            self.display_bond_details()
            self.plot_last_price_chart()

            self.get_simulation_parameters()

            if self.num_simulations and self.mean_yield and self.volatility:
                self.run_scenario_analysis()
                self.run_sensitivity_analysis()

    def select_bond(self):
        """
        Display dropdown to select the US government bond symbol.
        """
        # Dropdown menu for selecting bond symbol.
        self.bond_symbol = st.selectbox("Select US Government Bond Symbol", ["^IRX", "^FVX", "^TNX", "^TYX"])

    def fetch_bond_data(self):
        """
        Fetch bond data based on the selected bond symbol.
        """
        # Fetch bond data based on selected bond symbol.
        bond_data = yf.Ticker(self.bond_symbol)
        bond_info = bond_data.info

        # Update how you access the 'Close' price to avoid the warning.
        real_price = yf.Ticker(self.bond_symbol).history(period="1d")['Close'].iloc[0]

        self.face_value = bond_info.get('previousClose', real_price)
        self.coupon_rate = bond_info.get('couponRate', 5.0) * 100
        self.maturity_period = bond_info.get('maturity', 10)


    def display_bond_details(self):
        """
        Display details of the selected bond.
        """
        # Display details of the selected bond.
        st.write("### Selected Bond Details:")
        st.write(f"**Symbol:** {self.bond_symbol}")
        st.write(f"**Face Value:** ${self.face_value:.2f}")
        st.write(f"**Coupon Rate:** {self.coupon_rate:.2f}%")
        st.write(f"**Maturity Period:** {self.maturity_period} years")

    def plot_last_price_chart(self):
        """
        Plot chart of the last price of the selected bond.
        """
        # Plot chart of the last price of the selected bond.
        bond_data = yf.Ticker(self.bond_symbol)
        history = bond_data.history(period="1y")
        if not history.empty:
            st.write("### Last Price Chart:")
            st.line_chart(history['Close'])
        else:
            st.write("Last price data not available for selected bond.")

    def calculate_duration(self, bond_prices, yield_rate):
        """
        Calculate the duration of the bond.
        """
        cash_flows = np.zeros(int(self.maturity_period))
        for t in range(1, int(self.maturity_period) + 1):
            cash_flows[t - 1] = self.face_value * self.coupon_rate / 100
        cash_flows[-1] += self.face_value  # Add the principal amount as the last cash flow.

        duration = np.sum([t * cf / ((1 + yield_rate / 100) ** t) for t, cf in enumerate(cash_flows, start=1)]) / np.sum([cf / ((1 + yield_rate / 100) ** t) for t, cf in enumerate(cash_flows, start=1)])
        return duration

    def calculate_convexity(self, bond_prices, mean_yield, volatility):
        """
        Calculate the convexity of the bond.
        """
        convexity = 0
        for i in range(len(bond_prices)):
            yield_up = self.present_value(mean_yield + volatility)
            yield_down = self.present_value(mean_yield - volatility)
            convexity += ((yield_up + yield_down - 2 * bond_prices[i]) / (bond_prices[i] * volatility ** 2))
        convexity /= (2 * np.mean(bond_prices))
        return convexity
    
    # BUGGY ZERO DIV ERROR
    # def calculate_ytm(self, bond_price):
    #     """
    #     Calculate the yield to maturity (YTM) of the bond.
    #     """
    #     def ytm_func(y):
    #         # Check if maturity period is zero
    #         if self.maturity_period == 0:
    #             return np.nan
            
    #         # Calculate YTM using the formula
    #         return bond_price - sum([cf / ((1 + y / 100) ** t) for t, cf in enumerate([self.coupon_rate / 100 * self.face_value] * int(self.maturity_period))]) - self.face_value / ((1 + y / 100) ** self.maturity_period)

    #     try:
    #         # Use root finding method to solve for YTM
    #         ytm_solution = root_scalar(ytm_func, bracket=[-100, 100])
    #         return ytm_solution.root
    #     except ValueError:
    #         return np.nan
            
    def calculate_modified_duration(self, bond_price, ytm):
        """
        Calculate the Modified Duration of the bond.
        """
        modified_duration = sum(
            [t * cf / ((1 + ytm / 100) ** t) for t, cf in enumerate([self.coupon_rate / 100 * self.face_value] * int(self.maturity_period))] +
            [self.maturity_period * self.face_value / ((1 + ytm / 100) ** self.maturity_period)]
        ) / bond_price
        return modified_duration

    def get_simulation_parameters(self):
        """
        Get simulation parameters from the user.
        """
        # Input fields for scenario analysis parameters.
        self.num_simulations = st.number_input("Number of Monte Carlo Simulations", value=10000, step=1000)
        self.mean_yield = st.number_input("Mean Yield Rate (%)", value=5.0, step=0.1)
        self.volatility = st.number_input("Volatility of Yield Rates (%)", value=1.0, step=0.1)

    def run_scenario_analysis(self):
        """
        Run scenario analysis and display results.
        """
        # Instructions for scenario analysis.
        st.header("Scenario Analysis")
        st.write("Explore different scenarios to understand how changes in yield rates and volatility affect bond prices.")
        st.write("Adjust the parameters below and click 'Run Scenario Analysis' to initiate the analysis.")

        scenarios = {
            "Base Scenario": (self.mean_yield, self.volatility),
            "High Volatility": (self.mean_yield, self.volatility + 1),
            "Low Yield": (self.mean_yield - 1, self.volatility),
        }
        # Button to run scenario analysis.
        if st.button("Run Scenario Analysis"):
            for scenario_name, (mean_yield, volatility) in scenarios.items():
                bond_prices = self.price_bond(mean_yield, volatility)
                self.display_scenario_analysis(scenario_name, bond_prices)

    def display_scenario_analysis(self, scenario_name, bond_prices):
        """
        Display scenario analysis including histogram, summary statistics, and price range.
        """
        st.subheader(scenario_name)
        st.write("### Bond Price Distribution")
        self.plot_histogram(bond_prices)
        
        # Summary statistics.
        st.write("### Summary Statistics:")
            # Perform scenario analysis and display results.
        summary_statistics = {
            "Mean": np.mean(bond_prices),
            "Median": np.median(bond_prices),
            "Minimum": np.min(bond_prices),
            "Maximum": np.max(bond_prices),
            "Standard Deviation": np.std(bond_prices),
        }

        # Calculate Duration.
        duration = self.calculate_duration(bond_prices, self.mean_yield)
        summary_statistics["Duration"] = duration
        # Calculate Convexity.
        convexity = self.calculate_convexity(bond_prices, self.mean_yield, self.volatility)
        summary_statistics["Convexity"] = convexity

        # bond_price = np.mean(bond_prices)

        # # Calculate YTM.
        # ytm = self.calculate_ytm(bond_price)
        # summary_statistics["YTM"] = ytm
        # # Calculate Modified Duration
        # modified_duration = self.calculate_modified_duration(bond_price, ytm)
        # summary_statistics["Modified_Duration"] = modified_duration

        summary_df = pd.DataFrame(summary_statistics.items(), columns=["Metric", "Value"])
        st.table(summary_df)

        # Price range explanation.
        real_price = yf.Ticker(self.bond_symbol).history(period="1d")['Close'].iloc[0]
        expected_min_price = np.percentile(bond_prices, 5)
        expected_max_price = np.percentile(bond_prices, 95)
        st.write("### Price Range Explanation:")
        st.write(f"The real-time price of the selected bond is ${real_price:.2f}.")
        st.write(f"Based on the simulations, the expected price range for today (5th to 95th percentile) is between ${expected_min_price:.2f} and ${expected_max_price:.2f}")
        st.write('---')

        # Export buttons.
        export_pdf = st.button("Export to PDF - " + scenario_name)
        export_csv = st.button("Export to CSV - " + scenario_name)
        if export_pdf:
            self.export_to_pdf(scenario_name, bond_prices, summary_statistics)
        if export_csv:
            self.export_to_csv(scenario_name, bond_prices, summary_statistics)

    def export_to_pdf(self, scenario_name, bond_prices, summary_statistics):
        """
        Export scenario analysis results to PDF.
        """
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Add scenario name.
        pdf.cell(200, 10, txt=f"Scenario: {scenario_name}", ln=True, align="C")
        pdf.ln(10)

        # Add bond price distribution chart.
        temp_chart = self.plot_histogram(bond_prices).to_json()
        pdf.image(temp_chart, x=10, y=pdf.get_y(), w=180)
        pdf.ln(180)

        # Add summary statistics table.
        pdf.cell(200, 10, txt="Summary Statistics:", ln=True, align="C")
        pdf.ln(10)
        for metric, value in summary_statistics.items():
            pdf.cell(100, 10, txt=f"{metric}:", ln=True)
            pdf.cell(100, 10, txt=f"{value:.2f}", ln=True)
        pdf.ln(10)

        # Save PDF file.
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            pdf.output(tmp_file.name)
            st.markdown(f"Download the PDF [here](/{tmp_file.name})", unsafe_allow_html=True)

    def export_to_csv(self, scenario_name, bond_prices, summary_statistics):
        """
        Export scenario analysis results to CSV.
        """
        df = pd.DataFrame({"Bond Prices": bond_prices})
        for metric, value in summary_statistics.items():
            df[metric] = value
        df.to_csv(f"{scenario_name}_results.csv", index=False)

    def run_sensitivity_analysis(self):
        """
        Run sensitivity analysis and display results.
        """
        # Instructions for sensitivity analysis.
        st.header("Sensitivity Analysis")
        st.write("Analyze how changes in coupon rates and maturity periods impact bond prices.")
        st.write("Click 'Run Sensitivity Analysis' to initiate the analysis.")
        parameters = ["coupon_rate", "maturity_period"]
        values = {
            "coupon_rate": [4, 5, 6],
            "maturity_period": [5, 10, 15],
        }
        # Button to run sensitivity analysis.
        if st.button("Run Sensitivity Analysis"):
                # Perform sensitivity analysis and display results.
            for param_name in parameters:
                for value in values[param_name]:
                    setattr(self, param_name, value)
                    bond_prices = self.price_bond(self.mean_yield, self.volatility)
                    self.display_sensitivity_analysis(f"{param_name}={value}", bond_prices)

    def display_sensitivity_analysis(self, param_value, bond_prices):
        """
        Display sensitivity analysis including histogram, summary statistics, and price range.
        """
        st.subheader(param_value)
        st.write("### Bond Price Distribution")
        self.plot_histogram(bond_prices)
        
        # Summary statistics.
        st.write("### Summary Statistics:")
        summary_statistics = {
            "Mean": np.mean(bond_prices),
            "Median": np.median(bond_prices),
            "Minimum": np.min(bond_prices),
            "Maximum": np.max(bond_prices),
            "Standard Deviation": np.std(bond_prices),
        }
        # Calculate Duration.
        duration = self.calculate_duration(bond_prices, self.mean_yield)
        summary_statistics["Duration"] = duration
        # Calculate Convexity.
        convexity = self.calculate_convexity(bond_prices, self.mean_yield, self.volatility)
        summary_statistics["Convexity"] = convexity

        # bond_price = np.mean(bond_prices)

        # # Calculate YTM.
        # ytm = self.calculate_ytm(bond_price)
        # summary_statistics["YTM"] = ytm
        # # Calculate Modified Duration
        # modified_duration = self.calculate_modified_duration(bond_price, ytm)
        # summary_statistics["Modified_Duration"] = modified_duration

        summary_df = pd.DataFrame(summary_statistics.items(), columns=["Metric", "Value"])
        st.table(summary_df)
        
        # Price range explanation.
        real_price = yf.Ticker(self.bond_symbol).history(period="1d")['Close'].iloc[0]
        expected_min_price = np.percentile(bond_prices, 5)
        expected_max_price = np.percentile(bond_prices, 95)
        st.write("### Price Range Explanation:")
        st.write(f"The real-time price of the selected bond is ${real_price:.2f}.")
        st.write(f"Based on the simulations, the expected price range for today (5th to 95th percentile) is between ${expected_min_price:.2f} and ${expected_max_price:.2f}")
        st.write('---')

        # Export buttons.
        export_pdf = st.button("Export to PDF - " + param_value)
        export_csv = st.button("Export to CSV - " + param_value)
        if export_pdf:
            self.export_to_pdf(param_value, bond_prices, summary_statistics)
        if export_csv:
            self.export_to_csv(param_value, bond_prices, summary_statistics)

    def plot_histogram(self, bond_prices):
        """
        Plot histogram of bond prices using Plotly for interactive visualization.
        """
        fig = px.histogram(x=bond_prices, nbins=30, labels={'x': 'Bond Price', 'y': 'Frequency'}, title='Bond Price Distribution')
        st.plotly_chart(fig)

    def price_bond(self, mean_yield, volatility):
        """
        Calculate bond prices based on mean yield and volatility.
        """
        simulated_yield_rates = np.random.normal(mean_yield, volatility, self.num_simulations)
        bond_prices = np.zeros(self.num_simulations)
        for i in range(self.num_simulations):
            bond_prices[i] = self.present_value(simulated_yield_rates[i])
        return bond_prices

    def present_value(self, yield_rate):
        """
        Calculate the present value of the bond using the given yield rate.
        """
        if self.maturity_period == 0:
            return np.nan  # Return NaN if maturity period is 0

        present_value = 0
        for t in range(1, int(self.maturity_period) + 1):
            present_value += self.face_value * self.coupon_rate / 100 / ((1 + yield_rate / 100) ** t)
        present_value += self.face_value / ((1 + yield_rate / 100) ** self.maturity_period)
        return present_value

if __name__ == "__main__":
    app = BondApp()
    app.run()
