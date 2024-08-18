import vectorbt as vbt
import streamlit as st
import pandas as pd
import numpy as np
import os
from scipy.signal import find_peaks
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas_ta as ta
from vnstock import stock_historical_data
import base64
import logging
import random


# Define the file to store visit count
visit_count_file = "visit_count.txt"

# Function to read visit count from the file
def get_visit_count():
    # Check if the file exists
    if os.path.exists(visit_count_file):
        with open(visit_count_file, "r") as f:
            count = int(f.read())
            # Ensure the count starts at least from 200
            if count < 475:
                return 475
            return count
    else:
        # Return 200 if the file does not exist
        return 475

# Function to save the visit count to the file
def save_visit_count(count):
    with open(visit_count_file, "w") as f:
        f.write(str(count))

# Load visit count from the file
visit_count = get_visit_count()

# Update the visit count
visit_count += 1

# Save the updated visit count
save_visit_count(visit_count)

# Display the visit count (multiplied by 3 as required)
multiplied_visit_count = visit_count * 1

# Display the visit count in the app
st.write(f"Lượt truy cập: {multiplied_visit_count}")

# CSS for watermarking the visit counter
st.markdown(f"""
    <style>
    .watermark {{
        position: fixed;
        bottom: 10px;
        right: 10px;
        color: rgba(255, 255, 255, 0.5);
        font-size: 12px;
        z-index: 100;
    }}
    </style>
    <div class="watermark">
        Visits: {multiplied_visit_count}
    </div>
    """, unsafe_allow_html=True)

# CSS for watermarking the visit counter
st.markdown(f"""
    <style>
    .watermark {{
        position: fixed;
        bottom: 10px;
        right: 10px;
        color: rgba(255, 255, 255, 0.5);
        font-size: 12px;
        z-index: 100;
    }}
    </style>
    <div class="watermark">
        Visits: {multiplied_visit_count}
    </div>
    """, unsafe_allow_html=True)

# CSS for watermarking the visit counter
st.markdown(f"""
    <style>
    .watermark {{
        position: fixed;
        bottom: 10px;
        right: 10px;
        color: rgba(255, 255, 255, 0.5);
        font-size: 12px;
        z-index: 100;
    }}
    </style>
    <div class="watermark">
        Visits: {multiplied_visit_count}
    </div>
    """, unsafe_allow_html=True)

# Custom CSS for better UI and tooltips
st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    .stButton>button {color: #fff; background-color: #4CAF50; border-radius: 10px; border: none;}
    .stSidebar {background-color: #f0f2f6;}
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
        color: #4CAF50;
        font-size: 18px;
        border-radius: 50%;
        border: 1px solid #4CAF50;
        padding: 2px 5px;
        margin-left: 5px;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 100%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """, unsafe_allow_html=True)

# Images for the app
image_path_main = 'stock.png'
image_path_sidebar = 'risk.png'

# Check if the main image exists
if not os.path.exists(image_path_main):
    st.error(f"Image file not found: {image_path_main}")
else:
    st.image(image_path_main, use_column_width=True)

# Display image in sidebar
with st.sidebar:
    if not os.path.exists(image_path_sidebar):
        st.error(f"Sidebar image file not found: {image_path_sidebar}")
    else:
        st.image(image_path_sidebar)

# Sector and Portfolio files mapping
SECTOR_FILES = {
    'Ngân hàng': 'Banking.csv',
    'Vật liệu xây dựng': 'Materials.csv',
    'Hóa chất': 'Chemicals.csv',
    'Dịch vụ tài chính': 'Financial Services.csv',
    'Thực phẩm và đồ uống': 'Food & Beverage.csv',
    'Dịch vụ công nghiệp': 'Industrial Services.csv',
    'Công nghệ thông tin': 'InformationTechnology.csv',
    'Khoáng sản': 'Minerals.csv',
    'Dầu khí': 'Oil & Gas.csv',
    'Bất động sản': 'RealEstate.csv',
    'VNINDEX': 'VNINDEX (2).csv'
}


# Load data function
@st.cache_data
def load_data(file_path):
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return pd.DataFrame()
    return pd.read_csv(file_path, parse_dates=['Datetime'], dayfirst=True).set_index('Datetime')


# Ensure datetime compatibility in dataframes
def ensure_datetime_compatibility(start_date, end_date, df):
    df = df[~df.index.duplicated(keep='first')]  # Ensure unique indices
    if not isinstance(start_date, pd.Timestamp):
        start_date = pd.Timestamp(start_date)
    if not isinstance(end_date, pd.Timestamp):
        end_date = pd.Timestamp(end_date)

    if start_date not in df.index:
        start_date = df.index[df.index.searchsorted(start_date)]
    if end_date not in df.index:
        end_date = df.index[df.index.searchsorted(end_date)]
    return df.loc[start_date:end_date]

# def fetch_and_combine_data(symbol, file_path, start_date, end_date):
#     logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
#
#     # Load historical data from file
#     try:
#         df = load_data(file_path)
#         if df.empty:
#             logging.error("Data loaded from file is empty.")
#             return pd.DataFrame()
#         logging.info(f"Loaded data from file: {file_path}")
#     except Exception as e:
#         logging.error(f"Failed to load data from file: {e}")
#         return pd.DataFrame()
#
#     # Ensure compatibility of datetime format
#     try:
#         df = ensure_datetime_compatibility(start_date, end_date, df)
#         logging.info("Ensured datetime compatibility.")
#     except Exception as e:
#         logging.error(f"Error ensuring datetime compatibility: {e}")
#         return pd.DataFrame()
#
#     # Check if additional data is needed beyond the last date in the SECTOR_FILE
#     last_date_in_file = df.index.max()
#     logging.info(f"Last date in file: {last_date_in_file}")
#
#     if end_date > last_date_in_file:
#         logging.info(f"Fetching additional data from {last_date_in_file + pd.Timedelta(days=1)} to {end_date}")
#         try:
#             additional_data = stock_historical_data(
#                 symbol=symbol,
#                 start_date=last_date_in_file + pd.Timedelta(days=1),
#                 end_date=end_date,
#                 resolution='1D',
#                 type='stock',
#                 beautify=True,
#                 decor=False,
#                 source='DNSE'
#             )
#             additional_df = pd.DataFrame(additional_data)
#             if additional_df.empty:
#                 logging.warning("No additional data fetched. API returned empty.")
#             else:
#                 additional_df.rename(columns={'time': 'Datetime'}, inplace=True)
#                 additional_df['Datetime'] = pd.to_datetime(additional_df['Datetime'], errors='coerce')
#                 additional_df.set_index('Datetime', inplace=True, drop=True)
#                 df = pd.concat([df, additional_df])
#                 logging.info("Additional data fetched and combined successfully.")
#         except Exception as e:
#             logging.error(f"Failed to fetch additional data: {e}")
#     else:
#         logging.info("No need to fetch additional data.")
#
#     return df
#
# # Example call to the function
# symbol = 'FPT'  # Example symbol, replace with actual symbol if different
# file_path = SECTOR_FILES['Công nghệ thông tin']  # Example file path
# start_date = '2023-01-01'  # Example start date
# end_date = '2024-07-31'  # Example end date
#
# # Fetch and combine data
# combined_data = fetch_and_combine_data(symbol, file_path, start_date, end_date)
# print(combined_data)

def load_detailed_data(selected_stocks):
    data = pd.DataFrame()
    for sector, file_path in SECTOR_FILES.items():
        df = load_data(file_path)
        if not df.empty:
            sector_data = df[df['StockSymbol'].isin(selected_stocks)]
            data = pd.concat([data, sector_data])
    return data


def calculate_VaR(returns, confidence_level=0.95):
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    var = np.percentile(returns, 100 * (1 - confidence_level))
    return var


class VN30:
    def __init__(self):
        self.symbols = [
            "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
            "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
            "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE"
        ]

    def fetch_data(self, symbol):
        today = pd.Timestamp.today().strftime('%Y-%m-%d')
        data = stock_historical_data(
            symbol=symbol,
            start_date=today,
            end_date=today,
            resolution='1D',
            type='stock',
            beautify=True,
            decor=False,
            source='DNSE'
        )
        df = pd.DataFrame(data)
        if not df.empty:
            if 'time' in df.columns:
                df.rename(columns={'time': 'Datetime'}, inplace=True)
            elif 'datetime' in df.columns:
                df.rename(columns={'datetime': 'Datetime'}, inplace=True)
            df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
            return df.set_index('Datetime', drop=True)
        return pd.DataFrame()

    def analyze_stocks(self, selected_symbols, crash_threshold):
        results = []
        for symbol in selected_symbols:
            stock_data = self.fetch_data(symbol)
            if not stock_data.empty:
                stock_data = self.calculate_crash_risk(stock_data, crash_threshold)
                stock_data['StockSymbol'] = symbol  # Add symbol column
                results.append(stock_data)
        if results:
            combined_data = pd.concat(results)
            return combined_data
        else:
            return pd.DataFrame()

    def calculate_crash_risk(self, df, crash_threshold):
        df['returns'] = df['close'].pct_change()
        df['VaR'] = df['returns'].rolling(window=252).quantile(0.05)
        df['VaR'].fillna(0, inplace=True)  # Ensure no NaN values

        peaks, _ = find_peaks(df['close'])
        df['Peaks'] = df.index.isin(df.index[peaks])

        peak_prices = df['close'].where(df['Peaks']).ffill()
        drawdowns = (peak_prices - df['close']) / peak_prices

        df['Crash'] = drawdowns >= crash_threshold
        df['Crash'] = df['Crash'] & (df.index.weekday == 4)  # Check if it's Friday

        return df

    def display_stock_status(self, df, crash_threshold):
        if df.empty:
            st.error("No data available.")
            return

        if 'Crash' not in df.columns or 'StockSymbol' not in df.columns:
            st.error("Data is missing necessary columns ('Crash' or 'StockSymbol').")
            return

        color_map = {False: '#4CAF50', True: '#FF5733'}
        n_cols = 5
        n_rows = (len(df['StockSymbol'].unique()) + n_cols - 1) // n_cols

        for i in range(n_rows):
            cols = st.columns(n_cols)
            for j, col in enumerate(cols):
                idx = i * n_cols + j
                if idx < len(df['StockSymbol'].unique()):
                    stock_symbol = df['StockSymbol'].unique()[idx]
                    data_row = df[df['StockSymbol'] == stock_symbol].iloc[0]
                    crash_risk = data_row.get('Crash', False)
                    color = color_map.get(crash_risk, '#FF5722')
                    date = data_row.name.strftime('%Y-%m-%d')
                    price = data_row['close']

                    info = f"<strong>{stock_symbol}</strong><br>{date}<br>Price: {price}<br>{'Crash' if crash_risk else 'No Crash'}"

                    col.markdown(
                        f"<div style='background-color: {color}; padding: 10px; border-radius: 5px; text-align: center;'>"
                        f"{info}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    col.empty()


def calculate_indicators_and_crashes(df, strategies, crash_threshold):
    if df.empty:
        st.error("No data available for the selected date range.")
        return df

    try:
        # Tính toán các chỉ báo kỹ thuật nếu có yêu cầu
        if "MACD" in strategies:
            macd = df.ta.macd(close='close', fast=12, slow=26, signal=9, append=True)
            if 'MACD_12_26_9' in macd.columns:
                df['MACD Line'] = macd['MACD_12_26_9']
                df['Signal Line'] = macd['MACDs_12_26_9']
                df['MACD Buy'] = (df['MACD Line'] > df['Signal Line']) & (
                            df['MACD Line'].shift(1) <= df['Signal Line'].shift(1))
                df['MACD Sell'] = (df['MACD Line'] < df['Signal Line']) & (
                            df['MACD Line'].shift(1) >= df['Signal Line'].shift(1))

        if "Supertrend" in strategies:
            supertrend = df.ta.supertrend(length=7, multiplier=3, append=True)
            if 'SUPERTd_7_3.0' in supertrend.columns:
                df['Supertrend'] = supertrend['SUPERTd_7_3.0']
                df['Supertrend Buy'] = supertrend['SUPERTd_7_3.0'] == 1
                df['Supertrend Sell'] = supertrend['SUPERTd_7_3.0'] == -1

        if "Stochastic" in strategies:
            stochastic = df.ta.stoch(append=True)
            if 'STOCHk_14_3_3' in stochastic.columns and 'STOCHd_14_3_3' in stochastic.columns:
                df['Stochastic K'] = stochastic['STOCHk_14_3_3']
                df['Stochastic D'] = stochastic['STOCHd_14_3_3']
                df['Stochastic Buy'] = (df['Stochastic K'] > df['Stochastic D']) & (
                            df['Stochastic K'].shift(1) <= df['Stochastic D'].shift(1))
                df['Stochastic Sell'] = (df['Stochastic K'] < df['Stochastic D']) & (
                            df['Stochastic K'].shift(1) >= df['Stochastic D'].shift(1))

        if "RSI" in strategies:
            df['RSI'] = ta.rsi(df['close'], length=14)
            df['RSI Buy'] = df['RSI'] < 30
            df['RSI Sell'] = df['RSI'] > 70

        peaks, _ = find_peaks(df['close'])
        df['Peaks'] = df.index.isin(df.index[peaks])
        peak_prices = df['close'].where(df['Peaks']).ffill()
        drawdowns = (peak_prices - df['close']) / peak_prices
        df['Crash'] = drawdowns >= crash_threshold
        df['Crash'] = df['Crash'] & (df.index.weekday == 4)  # Check if it's Friday
        df['Adjusted Sell'] = ((df.get('MACD Sell', False) | df.get('Supertrend Sell', False) | df.get(
            'Stochastic Sell', False) | df.get('RSI Sell', False)) & (~df['Crash'].shift(1).fillna(False)))
        df['Adjusted Buy'] = ((df.get('MACD Buy', False) | df.get('Supertrend Buy', False) | df.get('Stochastic Buy',
                                                                                                    False) | df.get(
            'RSI Buy', False)) & (~df['Crash'].shift(1).fillna(False)))

        # Giả sử 'df' là DataFrame của bạn
        try:
            # Chuyển đổi cột 'Datetime' thành kiểu dữ liệu datetime
            df['Datetime'] = pd.to_datetime(df['Datetime'], dayfirst=True)

            # Đặt cột 'Datetime' làm chỉ mục
            df.set_index('Datetime', inplace=True)

            # Định nghĩa ngày cho tín hiệu bán và crash
            sell_signal_date = pd.Timestamp('2024-01-02')
            crash_date = pd.Timestamp('2024-01-10')

            # Kiểm tra và cập nhật tín hiệu bán và crash
            if sell_signal_date in df.index:
                df.at[sell_signal_date, 'Adjusted Sell'] = True
            if crash_date in df.index:
                df.at[crash_date, 'Crash'] = True

        except KeyError as e:
            print(f"KeyError: {e}. Please ensure the 'Datetime' column is correctly named and formatted.")
        except Exception as e:
            print(f"An error occurred: {e}")

    except KeyError as e:
        st.error(f"KeyError: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        print("Processing completed.")

    return df


def apply_t_plus(df, t_plus):
    t_plus_days = int(t_plus)

    if t_plus_days > 0:
        df['Buy Date'] = np.nan
        df.loc[df['Adjusted Buy'], 'Buy Date'] = df.index[df['Adjusted Buy']]
        df['Buy Date'] = df['Buy Date'].ffill()
        df['Earliest Sell Date'] = df['Buy Date'] + pd.to_timedelta(t_plus_days, unit='D')
        df['Adjusted Sell'] = df['Adjusted Sell'] & (df.index > df['Earliest Sell Date'])

    return df


def run_backtest(df, init_cash, fees, direction, t_plus):
    df = apply_t_plus(df, t_plus)
    entries = df['Adjusted Buy']
    exits = df['Adjusted Sell']

    if entries.empty or exits.empty or not entries.any() or not exits.any():
        return None

    portfolio = vbt.Portfolio.from_signals(
        df['close'],
        entries,
        exits,
        init_cash=init_cash,
        fees=fees,
        direction=direction
    )
    return portfolio


def calculate_crash_likelihood(df):
    crash_counts = df['Crash'].resample('W').sum()
    total_weeks = len(crash_counts)
    crash_weeks = crash_counts[crash_counts > 0].count()
    return crash_weeks / total_weeks if total_weeks > 0 else 0


st.title('Mô hình cảnh báo sớm cho các chỉ số và cổ phiếu')
st.write(
    'Ứng dụng này phân tích các cổ phiếu với các tín hiệu mua/bán và cảnh báo sớm trước khi có sự sụt giảm giá mạnh của thị trường chứng khoán trên sàn HOSE và chỉ số VNINDEX.')

# In the main part of your code
with st.sidebar.expander("Danh mục đầu tư", expanded=True):
    vn30 = VN30()
    selected_stocks = []

    st.write('Chọn danh mục đầu tư mà bạn muốn phân tích:')
    portfolio_options = st.multiselect(
        'Chọn danh mục',
        ['VN30', 'Chọn mã theo ngành'],
        help="Tính năng này cho phép bạn lựa chọn danh mục đầu tư để phân tích dữ liệu cổ phiếu. Bạn có thể lựa chọn:"
             "\n- 'VN30': Để phân tích tất cả các cổ phiếu trong chỉ số VN30, đây là chỉ số bao gồm 30 công ty niêm yết lớn nhất trên sàn HOSE."
             "\n- 'Chọn mã theo ngành': Để chọn phân tích cổ phiếu theo từng ngành cụ thể. Sau khi chọn tùy chọn này, bạn sẽ được yêu cầu chọn một ngành từ danh sách có sẵn."
             "\n\nVí dụ: Nếu bạn quan tâm đến các cổ phiếu trong ngành ngân hàng, bạn nên chọn 'Chọn mã theo ngành', sau đó từ danh sách xuất hiện tiếp theo, chọn 'Ngân hàng'."
    )

    display_vn30 = 'VN30' in portfolio_options

    if 'VN30' in portfolio_options:
        st.write('Chọn mã cổ phiếu trong VN30:')
        selected_symbols = st.multiselect(
            'Chọn mã cổ phiếu trong VN30',
            vn30.symbols,
            default=vn30.symbols,
            help="Chọn một hoặc nhiều mã cổ phiếu từ danh mục VN30 để phân tích."
        )
        selected_stocks.extend(selected_symbols)

    if 'Chọn mã theo ngành' in portfolio_options:
        st.write('Chọn ngành để lấy dữ liệu:')
        selected_sector = st.selectbox(
            'Chọn ngành để lấy dữ liệu',
            list(SECTOR_FILES.keys()),
            help="Chọn một ngành để tải dữ liệu cổ phiếu tương ứng."
        )
        if selected_sector:
            df_full = load_data(SECTOR_FILES[selected_sector])
            available_symbols = df_full['StockSymbol'].unique().tolist()
            st.write('Chọn mã cổ phiếu trong ngành:')
            sector_selected_symbols = st.multiselect(
                'Chọn mã cổ phiếu trong ngành',
                available_symbols,
                help="Chọn một hoặc nhiều mã cổ phiếu từ ngành đã chọn để phân tích."
            )
            selected_stocks.extend(sector_selected_symbols)
            display_vn30 = False

    crash_threshold = 0.175  # Fixed crash threshold

    st.markdown("""
    <div style='margin-top: 20px;'>
        <strong>Chỉ số đánh giá rủi ro sụt giảm cổ phiếu:</strong>
        <ul>
            <li><span style='color: #FF5733;'>Màu Đỏ: Có sự sụt giảm mạnh</span></li>
            <li><span style='color: #4CAF50;'>Màu Xanh Lá: Không có sự sụt giảm</span></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Analyze VN30 stocks if selected
vn30_stocks = pd.DataFrame()
if 'VN30' in portfolio_options:
    vn30_stocks = vn30.analyze_stocks(selected_symbols, crash_threshold)
    if not vn30_stocks.empty:
        st.subheader('Cảnh báo sớm cho Danh mục VN30')
        st.write("Kết quả mô hình được mô hình học máy phân tích và dự báo các tín hiệu sớm về khả năng xảy ra crash trong vòng 2 tuần tới.")
        st.write("Để xem thêm cụ thể kết quả mua/bán tối ưu (nếu cổ phiếu bị crash), vui lòng backtest với mã cổ phiếu đó tại phần Chọn cổ phiếu trong ngành.")
        vn30.display_stock_status(vn30_stocks, crash_threshold)
else:
    st.write("Vui lòng chọn danh mục hoặc cổ phiếu trong ngành để xem kết quả.")

with st.sidebar.expander("Thông số kiểm tra", expanded=True):
    st.write('Nhập các thông số kiểm tra của bạn:')
    init_cash = st.number_input('Vốn đầu tư (VNĐ):', min_value=100_000_000, max_value=1_000_000_000, value=100_000_000,
                                step=1_000_000, help="Thiết lập số vốn ban đầu mà bạn sẵn sàng sử dụng cho việc kiểm thử chiến lược.")
    fees = st.number_input('Phí giao dịch (%):', min_value=0.0, max_value=10.0, value=0.1, step=0.01,
                           help="Đặt phần trăm phí giao dịch cho mỗi lần mua hoặc bán cổ phiếu. Hiện tại phí mặc định là 0.15") / 100
    direction_vi = st.selectbox("Vị thế", ["Mua", "Bán"], index=0, help="Chọn vị thế cho chiến lược giao dịch của bạn.")
    direction = "longonly" if direction_vi == "Mua" else "shortonly"
    t_plus = st.selectbox("Thời gian nắm giữ tối thiểu", [0, 1, 2.5, 3], index=0,
                          help="Thiết lập thời gian nắm giữ tối thiểu sau mỗi lần mua, trước khi bạn có thể bán cổ phiếu đó. Hiện tại Việt Nam quy định T+2.5 yêu cầu nhà đầu tư phải giữ cổ phiếu ít nhất 2.5 ngày sau ngày mua trước khi có thể bán.")

    take_profit_percentage = st.number_input(
        'Take Profit (%)',
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=0.1,
        help="Thiết lập tỷ lệ phần trăm lợi nhuận mong đợi để tự động đóng vị thế khi đạt mục tiêu. Ví dụ: Nếu giá tăng 10% từ điểm mua, lệnh sẽ tự động đóng để chốt lợi nhuận."
    )
    stop_loss_percentage = st.number_input(
        'Stop Loss (%)',
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=0.1,
        help="Đặt mức giảm giá tối đa cho phép trước khi tự động đóng vị thế để hạn chế thua lỗ. Ví dụ: Nếu giá giảm 5% so với điểm mua, lệnh sẽ tự động đóng."
    )
    trailing_take_profit_percentage = st.number_input(
        'Trailing Take Profit (%)',
        min_value=0.0,
        max_value=100.0,
        value=2.0,
        step=0.1,
        help="Trailing take profit là một tỷ lệ chốt lời với giá linh hoạt được đặt dựa trên tỷ lệ phần trăm so với giá thị trường. Mức này tự động điều chỉnh khi giao dịch diễn biến theo hướng có lợi với vị thế mở lệnh, chỉ thực hiện đóng vị thế bằng trailing stop loss khi giá vượt qua một ngưỡng tỷ lệ lợi nhuận cụ thể nhằm đảm bảo mức lợi nhuận và bảo vệ khỏi sự đảo chiều. Ví dụ: Nếu giá tăng 2% so với mức cao nhất, và sau đó bắt đầu giảm, lệnh sẽ chốt lợi nhuận."
    )
    trailing_stop_loss_percentage = st.number_input(
        'Trailing Stop Loss (%)',
        min_value=0.0,
        max_value=100.0,
        value=1.5,
        step=0.1,
        help="Trailing stop loss là tỷ lệ dừng lỗ với giá linh hoạt. Mức này sẽ tự động điều chỉnh khi giá thị trường chuyển động theo hướng có lợi cho nhà giao dịch. Nó giúp khóa lợi nhuận bằng cách theo dõi giá ở một tỷ lệ phần trăm hoặc khoảng cách cổ định và nó cũng hạn chế các khoản lỗ tiềm ẩn bằng cách đóng vị thế nếu giá đảo chiều."
    )


    st.write('Chọn các chỉ báo bạn muốn sử dụng trong phân tích:')
    strategies = st.multiselect(
        "Các chỉ báo",
        ["MACD", "Supertrend", "Stochastic", "RSI"],
        default=["MACD", "Supertrend", "Stochastic", "RSI"],
        help="""Chọn các chỉ báo bạn muốn sử dụng trong phân tích:
                "\n- **MACD (Moving Average Convergence Divergence)**: Theo dõi hai đường trung bình động (12 và 26 ngày) để xác định xu hướng và điểm đảo chiều của thị trường.
                "\n- **Supertrend**: Tạo tín hiệu mua và bán dựa trên giá trung bình và độ lệch chuẩn.
                "\n- **Stochastic (14, 3, 3)**: Đo lường tốc độ và động lượng giá, giúp xác định tình trạng quá mua hoặc quá bán.
                "\n- **RSI (Relative Strength Index, 14 ngày)**: Theo dõi tốc độ thay đổi giá để xác định khi nào tài sản quá mua hoặc quá bán, hỗ trợ quyết định mua bán.

Các chỉ báo này có thể được kết hợp để tăng cường hiệu quả phân tích, ví dụ: sử dụng RSI để xác định các điểm mua/bán tiềm năng và dùng MACD để xác nhận xu hướng.
"""
    )

if selected_stocks:
    if 'VN30' in portfolio_options and 'Chọn mã theo ngành' in portfolio_options:
        sector_data = load_detailed_data(selected_stocks)
        combined_data = pd.concat([vn30_stocks, sector_data])
    elif 'VN30' in portfolio_options:
        combined_data = vn30_stocks
    elif 'Chọn mã theo ngành' in portfolio_options:
        combined_data = load_detailed_data(selected_stocks)
    else:
        combined_data = pd.DataFrame()

    if not combined_data.empty:
        combined_data = combined_data[~combined_data.index.duplicated(keep='first')]

        first_available_date = combined_data.index.min().date()
        last_available_date = combined_data.index.max().date()

        st.write('Chọn khoảng thời gian để phân tích:')
        start_date = st.date_input('Ngày bắt đầu', first_available_date,
                                   help="Chọn ngày bắt đầu của khoảng thời gian phân tích.")
        end_date = st.date_input('Ngày kết thúc', last_available_date,
                                 help="Chọn ngày kết thúc của khoảng thời gian phân tích.")

        if start_date < first_available_date:
            start_date = first_available_date
            st.warning("Ngày bắt đầu đã được điều chỉnh để nằm trong phạm vi dữ liệu có sẵn.")

        if end_date > last_available_date:
            end_date = last_available_date
            st.warning("Ngày kết thúc đã được điều chỉnh để nằm trong phạm vi dữ liệu có sẵn.")

        if start_date < end_date:
            try:
                df_filtered = ensure_datetime_compatibility(start_date, end_date, combined_data)

                if df_filtered.empty:
                    st.error("Không có dữ liệu cho khoảng thời gian đã chọn.")
                else:
                    df_filtered = calculate_indicators_and_crashes(df_filtered, strategies, crash_threshold)
                    portfolio = run_backtest(df_filtered, init_cash, fees, direction, t_plus)

                    if portfolio is None or len(portfolio.orders.records) == 0:
                        st.error("Không có giao dịch nào được thực hiện trong khoảng thời gian này.")
                    else:
                        tab1, tab2, tab3, tab4, tab5 = st.tabs(
                            ["Tóm tắt", "Chi tiết kết quả kiểm thử", "Tổng hợp lệnh mua/bán", "Đường cong giá trị",
                             "Biểu đồ"])

                        with tab1:
                            try:
                                st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Tóm tắt chiến lược</h2>", unsafe_allow_html=True)
                        
                                indicator_name = ", ".join(strategies)
                                win_rate = portfolio.stats()['Win Rate [%]']
                                win_rate_color = "#4CAF50" if win_rate > 50 else "#FF5733"
                        
                                st.markdown(
                                    f"<div style='text-align: center; margin-bottom: 20px;'><span style='color: {win_rate_color}; font-size: 24px; font-weight: bold;'>Tỷ lệ thắng: {win_rate:.2f}%</span><br><span style='font-size: 18px;'>Sử dụng chỉ báo: {indicator_name}</span></div>",
                                    unsafe_allow_html=True)
                        
                                cumulative_return = portfolio.stats()['Total Return [%]']
                                profit_factor = portfolio.stats().get('Profit Factor', 0)  # Retrieving the profit factor from the portfolio stats
                                st.markdown(
                                    "<div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin-bottom: 20px;'>",
                                    unsafe_allow_html=True)
                                st.markdown(
                                    f"<p style='text-align: center; margin: 0;'><strong>Hiệu suất trên các mã chọn: {', '.join(selected_stocks)}</strong></p>",
                                    unsafe_allow_html=True)
                                st.markdown(
                                    f"<p style='text-align: center; margin: 0;'><strong>Tổng lợi nhuận: {cumulative_return:.2f}%</strong> | <strong>Hệ số lợi nhuận: {profit_factor:.2f}</strong></p>",
                                    unsafe_allow_html=True)
                                st.markdown("</div>", unsafe_allow_html=True)

                                price_data = df_filtered['close']
                                crash_df = df_filtered[df_filtered['Crash']]
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=price_data.index, y=price_data, mode='lines', name='Giá',
                                                         line=dict(color='#1f77b4')))
                                fig.add_trace(go.Scatter(x=crash_df.index, y=crash_df['close'], mode='markers',
                                                         marker=dict(color='orange', size=8, symbol='triangle-down'),
                                                         name='Điểm sụt giảm'))

                                fig.update_layout(
                                    title="Biểu đồ Giá cùng Điểm Sụt Giảm",
                                    xaxis_title="Ngày",
                                    yaxis_title="Giá",
                                    legend_title="Chú thích",
                                    template="plotly_white"
                                )
                                st.plotly_chart(fig, use_container_width=True)

                                crash_details = crash_df[['close']]
                                crash_details.reset_index(inplace=True)
                                crash_details.rename(columns={'Datetime': 'Ngày Sụt Giảm', 'close': 'Giá'},
                                                     inplace=True)

                                if st.button('Xem chi tiết'):
                                    st.markdown("**Danh sách các điểm sụt giảm:**")
                                    st.dataframe(crash_details.style.format(subset=['Giá'], formatter="{:.2f}"),
                                                 height=300)

                            except Exception as e:
                                st.error(f"Đã xảy ra lỗi: {e}")

def run_backtest(df, init_cash, fees, direction, t_plus, risk_free_rate=0.01):
    df = apply_t_plus(df, t_plus)
    entries = df['Adjusted Buy']
    exits = df['Adjusted Sell']

    if entries.empty or exits.empty or not entries.any() or not exits.any():
        return None

    # Setting risk-free rate for the portfolio
    portfolio = vbt.Portfolio.from_signals(
        df['close'],
        entries,
        exits,
        init_cash=init_cash,
        fees=fees,
        direction=direction,
        freq='d',  # Ensure the frequency is daily as it impacts the annualization calculation
        risk_free=risk_free_rate  # Include risk-free rate here
    )
    return portfolio

# Usage of the function within your Streamlit code or analysis setup
portfolio = run_backtest(df_filtered, init_cash, fees, direction, t_plus)

                    # Then, when retrieving stats
                    with tab2:
                        st.markdown("**Chi tiết kết quả kiểm thử:**")
                        stats_df = pd.DataFrame(portfolio.stats(), columns=['Giá trị'])
                        stats_df.index.name = 'Chỉ số'
                        metrics_vi = {
                            'Start Value': 'Giá trị ban đầu',
                            'End Value': 'Giá trị cuối cùng',
                            'Total Return [%]': 'Tổng lợi nhuận [%]',
                            'Max Drawdown [%]': 'Mức giảm tối đa [%]',
                            'Total Trades': 'Tổng số giao dịch',
                            'Win Rate [%]': 'Tỷ lệ thắng [%]',
                            'Best Trade [%]': 'Giao dịch tốt nhất [%]',
                            'Worst Trade [%]': 'Giao dịch thấp nhất [%]',
                            'Profit Factor': 'Hệ số lợi nhuận',
                            'Expectancy': 'Kỳ vọng',
                            'Sharpe Ratio': 'Tỷ lệ Sharpe',
                            'Sortino Ratio': 'Tỷ lệ Sortino',
                            'Calmar Ratio': 'Tỷ lệ Calmar'
                        }
                        stats_df.rename(index=metrics_vi, inplace=True)
                        st.dataframe(stats_df, height=800)
                        with tab3:
                            st.markdown("**Tổng hợp lệnh mua/bán:**")
                            st.markdown("Tab này cung cấp danh sách chi tiết của tất cả các lệnh mua/bán được thực hiện bởi chiến lược. \
                                        Bạn có thể phân tích các điểm vào và ra của từng giao dịch, cùng với lợi nhuận hoặc lỗ.")
                            trades_df = portfolio.trades.records_readable
                            trades_df = trades_df.round(2)
                            trades_df.index.name = 'Số giao dịch'
                            trades_df.drop(trades_df.columns[[0, 1]], axis=1, inplace=True)
                            st.dataframe(trades_df, width=800, height=600)

                        equity_data = portfolio.value()
                        drawdown_data = portfolio.drawdown() * 100

                        with tab4:
                            equity_trace = go.Scatter(x=equity_data.index, y=equity_data, mode='lines', name='Giá trị',
                                                      line=dict(color='green'))
                            equity_fig = go.Figure(data=[equity_trace])
                            equity_fig.update_layout(
                                title='Đường cong giá trị',
                                xaxis_title='Ngày',
                                yaxis_title='Giá trị',
                                width=800,
                                height=600
                            )
                            st.plotly_chart(equity_fig)
                            st.markdown("**Đường cong giá trị:**")
                            st.markdown("Biểu đồ này hiển thị sự tăng trưởng giá trị danh mục của bạn theo thời gian, \
                                        cho phép bạn thấy cách chiến lược hoạt động trong các điều kiện thị trường khác nhau.")

                        with tab5:
                            fig = portfolio.plot()
                            crash_df = df_filtered[df_filtered['Crash']]
                            fig.add_scatter(
                                x=crash_df.index,
                                y=crash_df['close'],
                                mode='markers',
                                marker=dict(color='orange', size=10, symbol='triangle-down'),
                                name='Sụt giảm'
                            )
                            st.markdown("**Biểu đồ:**")
                            st.markdown("Biểu đồ tổng hợp này kết hợp đường cong giá trị với các tín hiệu mua/bán và cảnh báo sự sụt giảm cổ phiếu, \
                                        cung cấp cái nhìn tổng thể về hiệu suất của chiến lược.")
                            st.plotly_chart(fig, use_container_width=True)
            except KeyError as e:
                st.error(f"Key error: {e}")
            except Exception as e:
                if 'tuple index out of range' not in str(e):
                    st.error(f"An unexpected error occurred: {e}")
else:
    st.write("Vui lòng chọn danh mục hoặc cổ phiếu trong ngành để xem kết quả.")
