import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


def perform_ddos_analysis():
    print("Starting DDoS Regression Analysis...")

    log_data = []
    try:
        with open('server.log', 'r') as f:
            for line in f:
                # Regex matches the date and time: YYYY-MM-DD HH:MM:SS
                match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                if match:
                    log_data.append(match.group(1))
    except FileNotFoundError:
        print("Error: server.log not found.")
        return

    df = pd.DataFrame(log_data, columns=['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['count'] = 1

    df_resampled = df.resample('1min', on='timestamp').count().rename(columns={'count': 'actual_count'})
    df_resampled = df_resampled.reset_index()

    X = np.arange(len(df_resampled)).reshape(-1, 1)
    y = df_resampled['actual_count'].values

    model = LinearRegression()
    model.fit(X, y)
    df_resampled['predicted'] = model.predict(X)

    std_dev = np.std(y)
    threshold = df_resampled['predicted'] + (2 * std_dev)
    df_resampled['is_ddos'] = df_resampled['actual_count'] > threshold

    ddos_attacks = df_resampled[df_resampled['is_ddos'] == True]

    print("\n--- IDENTIFIED DDOS ATTACK INTERVALS ---")
    if ddos_attacks.empty:
        print("No DDoS attacks detected. Try lowering threshold to 1.5 standard deviations?")
    else:
        print(ddos_attacks[['timestamp', 'actual_count']])
    print("----------------------------------------\n")

    plt.figure(figsize=(14, 7))
    plt.plot(df_resampled['timestamp'], df_resampled['actual_count'], label='Actual Traffic', color='royalblue')
    plt.plot(df_resampled['timestamp'], df_resampled['predicted'], label='Regression Trend', color='red',
             linestyle='--')

    if not ddos_attacks.empty:
        plt.scatter(ddos_attacks['timestamp'], ddos_attacks['actual_count'], color='orange', label='DDoS Attack')

    plt.title('Traffic Regression Analysis')
    plt.savefig('ddos_plot.png')
    plt.show()


if __name__ == "__main__":
    perform_ddos_analysis()