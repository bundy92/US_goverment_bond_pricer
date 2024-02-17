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
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, logging, pipeline
from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException
import numba
import re
import torch
import sentencepiece

logging.set_verbosity_error()

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

class BondApp:
    def __init__(self):
        # Bond info.
        self.bond_symbol = None
        self.long_name = ""
        self.short_name = ""
        self.face_value = None
        self.coupon_rate = None
        self.maturity_period = None
        self.fifty_day_avg = None
        self.two_hundred_day_avg = None
        
        # Simulation info.
        self.num_simulations = None
        self.mean_yield = None
        self.volatility = None
        self.bond_pricer = None

        # Dictionary mapping bond symbols to maturity periods.
        self.maturity_periods = {
            "^IRX": 1,
            "^FVX": 5,
            "^TNX": 10,
            "^TYX": 30
        }

    def run(self):
        st.set_page_config(layout="wide", initial_sidebar_state="expanded")
        st.sidebar.title("US Government Bond Pricer")
        st.sidebar.write("This tool allows you to do basic analyzis of US government bonds.")

        self.select_bond()
        self.fetch_bond_data()
        self.display_bond_details()
        self.plot_last_price_chart()
        self.get_simulation_parameters()
        self.fetch_and_analyze_news()
        self.run_scenario_analysis()
        self.run_sensitivity_analysis()
        

    def select_bond(self):
        self.bond_symbol = st.sidebar.selectbox("Select US Government Bond Symbol", ["^IRX", "^FVX", "^TNX", "^TYX"])

    def fetch_bond_data(self):
        bond_data = yf.Ticker(self.bond_symbol)
        bond_info = bond_data.info

        self.long_name = bond_info.get('longName')
        self.short_name = bond_info.get('shortName')
        self.face_value = bond_info.get('previousClose', 0)
        self.coupon_rate = bond_info.get('couponRate', 0) * 100
        # Default maturity period is 10 years.
        self.maturity_period = self.maturity_periods.get(self.bond_symbol, 10)
        self.fifty_day_avg = bond_info.get('fiftyDayAverage', None)
        self.two_hundred_day_avg = bond_info.get('twoHundredDayAverage', None)
        

    def display_bond_details(self):
        st.sidebar.write("### Bond Details:")
        st.sidebar.write(f"**Symbol:** {self.bond_symbol}")
        st.sidebar.write(f"**Name:** {self.short_name}")
        st.sidebar.write(f"**Face Value:** ${self.face_value:.2f}")
        st.sidebar.write(f"**50 day average:** ${self.fifty_day_avg:.2f}")
        st.sidebar.write(f"**200 day average:** ${self.two_hundred_day_avg:.2f}")
        st.sidebar.write(f"**Coupon Rate 0 if NA:** ${self.coupon_rate:.2f}")
        st.sidebar.write(f"**Maturity Period:** {self.maturity_period} years")

    def plot_last_price_chart(self):
        """
        Plot chart of the last price of the selected bond.
        """
        st.title(self.long_name)
        # Plot chart of the last price of the selected bond.
        bond_data = yf.Ticker(self.bond_symbol)
        history = bond_data.history(period="1y")
        if not history.empty:
            st.write("### Last Price Chart:")
            st.line_chart(history['Close'])
        else:
            st.write("Last price data not available for selected bond.")

    #@numba.jit(nopython=False, cache=True)
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
    
    #@numba.jit(nopython=False, cache=True)
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

    #@numba.jit(nopython=False, cache=True)
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
        self.num_simulations = st.sidebar.number_input("Number of Monte Carlo Simulations", value=10000, step=1000)
        self.mean_yield = st.sidebar.number_input("Mean Yield Rate (%)", value=5.0, step=0.1)
        self.volatility = st.sidebar.number_input("Volatility of Yield Rates (%)", value=1.0, step=0.1)

    def run_scenario_analysis(self):
        st.sidebar.header("Scenario Analysis")
        st.sidebar.write("Explore different scenarios to understand how changes in yield rates and volatility affect bond prices.")
        st.sidebar.write("Adjust the parameters below and click 'Run Scenario Analysis' to initiate the analysis.")

        if st.sidebar.button("Run Scenario Analysis"):
            scenarios = {
                "Base Scenario": (self.mean_yield, self.volatility),
                "High Volatility": (self.mean_yield, self.volatility + 1),
                "Low Yield": (self.mean_yield - 1, self.volatility),
            }

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
        st.write(f"Based on the simulations, the expected price range for today (5th to 95th percentile) is between ${expected_min_price:.2f} and ${expected_max_price:.2f}.", unsafe_allow_html=True)
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
        st.sidebar.header("Sensitivity Analysis")
        st.sidebar.write("Analyze how changes in coupon rates and maturity periods impact bond prices.")
        st.sidebar.write("Click 'Run Sensitivity Analysis' to initiate the analysis.")

        if st.sidebar.button("Run Sensitivity Analysis"):
            parameters = ["coupon_rate", "maturity_period"]
            values = {
                "coupon_rate": [4, 5, 6],
                "maturity_period": [5, 10, 15],
            }

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
            st.write(f"Based on the simulations, the expected price range for today (5th to 95th percentile) is between ${expected_min_price:.2f} and ${expected_max_price:.2f}.", unsafe_allow_html=True)
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

    #@numba.jit(nopython=False, cache=True)
    def price_bond(self, mean_yield, volatility):
        """
        Price the bond using Monte Carlo simulations.
        """
        # Perform Monte Carlo simulations to price the bond.
        bond_prices = np.zeros(self.num_simulations)
        for i in range(self.num_simulations):
            yield_rate = np.random.normal(mean_yield, volatility)
            present_value = self.present_value(yield_rate)
            bond_prices[i] = present_value
        return bond_prices
    
    #@numba.jit(nopython=False, cache=True)
    def present_value(self, yield_rate):
        """
        Calculate the present value of the bond.
        """
        present_value = sum([cf / ((1 + yield_rate / 100) ** t) for t, cf in enumerate([self.coupon_rate / 100 * self.face_value] * int(self.maturity_period))]) + self.face_value / ((1 + yield_rate / 100) ** self.maturity_period)
        return present_value
    
    # A different matplotlibes historgram layout.
    # def plot_histogram(self, data):
    #     """
    #     Plot histogram of bond prices.
    #     """
    #     # Plot histogram using Seaborn.
    #     plt.figure(figsize=(10, 6))
    #     sns.histplot(data, kde=True, bins=30, color="blue")
    #     plt.title("Bond Price Distribution")
    #     plt.xlabel("Price ($)")
    #     plt.ylabel("Frequency")
    #     st.pyplot()

    def plot_line_chart(self, data):
        """
        Plot line chart of bond prices.
        """
        # Plot line chart using Altair.
        df = pd.DataFrame({"Simulation": np.arange(len(data)), "Bond Price": data})
        line_chart = alt.Chart(df).mark_line().encode(x="Simulation", y="Bond Price")
        st.altair_chart(line_chart, use_container_width=True)

    # News analysis part.
    def fetch_and_analyze_news(self):
        st.sidebar.write(f"### News and sentiment of {self.bond_symbol}")

        if st.sidebar.button("Get News Summary and Sentiment"):
            st.write("It takes approximately 30 seconds.")
            self.fetch_and_analyze_news_internal()
    
    def fetch_and_analyze_news_internal(self):
        sentiment_analyzer = pipeline("sentiment-analysis")

        with st.spinner("Fetching news data and performing analysis..."):
            monitored_tickers = [self.bond_symbol]
            raw_urls = {ticker: self.search_for_stock_news_urls(ticker) for ticker in monitored_tickers}
            exclude_list = ['maps', 'policies', 'preferences', 'accounts', 'support']
            cleaned_urls = {ticker: self.strip_unwanted_urls(raw_urls[ticker], exclude_list) for ticker in monitored_tickers}
            articles = {ticker: self.scrape_and_process(cleaned_urls[ticker]) for ticker in monitored_tickers}
            summaries = {ticker: self.summarize(articles[ticker]) for ticker in monitored_tickers}
            scores = {ticker: sentiment_analyzer(summaries[ticker]) for ticker in monitored_tickers}
            final_output = self.create_output_array(summaries, scores, cleaned_urls)
            df = pd.DataFrame(final_output, columns=["Ticker", "Summary", "Sentiment", "Score", "URL"])

            st.write("### News Summary and Sentiment Analysis:")
            st.table(df.style.set_properties(**{'font-size': '12px', 'text-align': 'left'}))

    #BROKEN! Below a working version.
    def search_for_stock_news_urls(self, ticker):
        """
        Search for stock news URLs using Google.
        """
        st.write("Searching for news...")
        search_url = "https://www.google.com/search?q=yahoo+finance+{}&tbm=nws".format(ticker)
        #search_url = "https://www.ecosia.org/news?q=yahoo%20finance%20{}".format(ticker)
        #headers = {"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/118.0"}
        cookies = {"CONSENT": "PENDING+900", "SOCS": "CAISHAgBEhJnd3NfMjAyMzA4MTAtMF9SQzIaAmRlIAEaBgiAo_CmBg"}
        #r = requests.get(search_url, cookies={"CONSENT": "YES+cb.20240116-07-p0.en+FX+410"})
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
        r = requests.get(search_url, headers=headers, cookies=cookies)

        soup = BeautifulSoup(r.text, 'html.parser')
        atags = soup.find_all('a')
        hrefs = [link['href'] for link in atags]
        return hrefs 
        
    # def search_for_stock_news_urls(self, ticker):
    #     search_url = "https://www.google.com/search?q=yahoo+finance+{}&tbm=nws".format(ticker)
    #     #search_url = "https://www.ecosia.org/news?q=yahoo%20finance%20{}".format(ticker)
    #     r = requests.get(search_url, cookies = {"CONSENT": "PENDING+900", "SOCS": "CAISHAgBEhJnd3NfMjAyMzA4MTAtMF9SQzIaAmRlIAEaBgiAo_CmBg"})
    #     soup = BeautifulSoup(r.text, 'html.parser')
    #     atags = soup.find_all('a')
    #     hrefs = [link['href'] for link in atags]
    #     return hrefs 

    
    def strip_unwanted_urls(self, urls, exclude_list):
        """
        Strip out unwanted URLs.
        """
        st.write("Cleaning up URLs...")
        val = []
        for url in urls: 
            if 'https://' in url and not any(exclude_word in url for exclude_word in exclude_list):
                res = re.findall(r'(https?://\S+)', url)[0].split('&')[0]
                val.append(res)
        return list(set(val))

    def scrape_and_process(self, URLs):
        """
        Scrape and process unwanted URLs.
        """
        st.write("Cleaning up articles...")
        self.chrome_options = Options()
        # Add any additional options if needed
        self.chrome_options.add_argument("--headless")
        # Initialize a Chrome WebDriver (you can use other browsers as well).
        try:
            driver = webdriver.Chrome(options=self.chrome_options)
            
            ARTICLES = []
            for url in URLs: 
                driver.get(url)
                
                try:
                    # Wait for consent popup to appear (adjust timeout as needed).
                    WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="consent-popup"]/button'))).click()
                except:
                    # If no consent popup is found, continue scraping.
                    pass

                # Now the consent popup should be dismissed, continue scraping.
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                paragraphs = soup.find_all('p')
                text = [paragraph.text for paragraph in paragraphs]
                words = ' '.join(text).split(' ')[:100]
                ARTICLE = ' '.join(words)
                ARTICLES.append(ARTICLE)

            # Close the WebDriver once scraping is done.
            driver.quit()

            return ARTICLES
        except WebDriverException as e:
            st.error("An error occurred while scraping articles. Please try again later.")
            return []
    
    def summarize(self, articles):
        """
        Summarize articles.
        """
        # Setup summarization model
        model_name = "human-centered-summarization/financial-summarization-pegasus"  
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

        st.write("Summarizing articles...")

        summaries = []
        for article in articles:
            input_ids = tokenizer.encode(article, return_tensors='pt').to(device)
            output = model.generate(input_ids, max_length=20, num_beams=2, early_stopping=True)
            summary = tokenizer.decode(output[0], skip_special_tokens=True)
            summaries.append(summary)
        return summaries

    def create_output_array(self, summaries, scores, urls):
        """
        Create output array.
        """
        st.write("Creating output...")
        output = []
        for ticker in urls:
            for counter in range(len(summaries[ticker])):
                output_this = [
                    ticker,
                    summaries[ticker][counter],
                    scores[ticker][counter]['label'],
                    scores[ticker][counter]['score'],
                    urls[ticker][counter]
                ]
                output.append(output_this)
        return output


# Run the application, main entry point.
if __name__ == "__main__":
    bond_app = BondApp()
    bond_app.run()
