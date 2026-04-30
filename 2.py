import sqlite3
import pandas as pd
import numpy as np
import os
import gradio as gr
import matplotlib.pyplot as plt
import warnings
from sklearn.tree import DecisionTreeRegressor, plot_tree

warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')

DB_PATH = os.path.expanduser("~/Desktop/weather_dw.db")
OUT_DIR = os.path.expanduser("~/Desktop/weather_analysis_output/")
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

print("Starting Tamil Nadu Climate Intelligence System...")

def load_full_data():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)
    print("Extracting complete records from Star Schema...")
    
    l_df = pd.read_sql("SELECT location_id, district_name FROM dim_location", conn)
    d_df = pd.read_sql("SELECT Date_ID, Month, Year FROM dim_date", conn).drop_duplicates('Date_ID')
    
    chunks = []
    query = "SELECT * FROM fact_weather"
    for chunk in pd.read_sql(query, conn, chunksize=100000):
        chunks.append(chunk)
    f_df = pd.concat(chunks, ignore_index=True)
    conn.close()

    for df, col in [(f_df, 'location_id'), (l_df, 'location_id'), (f_df, 'date_id'), (d_df, 'Date_ID')]:
        df[col] = df[col].astype(str).str.strip()

    print("Merging Fact and Dimension tables...")
    df = f_df.merge(l_df, on='location_id').merge(d_df, left_on='date_id', right_on='Date_ID')
    
    month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
    df['Month_Num'] = df['Month'].map(month_map)
    
    numeric_cols = ['temperature_2m', 'rain', 'relative_humidity_2m', 'cloud_cover', 'surface_pressure', 'wind_speed_10m', 'Year']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    return df.dropna(subset=['temperature_2m', 'Year', 'Month_Num'])

df = load_full_data()

trend_plot_path = os.path.join(OUT_DIR, "weather_evolution_trend.png")
tree_plot_path = os.path.join(OUT_DIR, "decision_tree_logic.png")

if not df.empty:
    print("Generating Weather Evolution Trend...")
    data = df.copy()
    data['is_sunny'] = ((data['rain'] == 0) & (data['cloud_cover'] < 30)).astype(int)
    data['is_rainy'] = (data['rain'] > 0.5).astype(int)
    yearly = data.groupby('Year').agg({'is_sunny': 'sum', 'is_rainy': 'sum'}).reset_index()

    plt.figure(figsize=(10, 4))
    plt.plot(yearly['Year'], yearly['is_sunny'], label='Sunny Days (Clear Sky)', color='#FFD700', marker='o', linewidth=2)
    plt.plot(yearly['Year'], yearly['is_rainy'], label='Rainy Days (>0.5mm)', color='#1E90FF', marker='s', linewidth=2)
    plt.axvspan(2022.5, 2025.5, color='gray', alpha=0.1, label='Recent 3-Year Focus')
    plt.title('Tamil Nadu Climate Evolution: Annual Sunny vs Rainy Days')
    plt.xlabel('Year')
    plt.ylabel('Number of Days')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig(trend_plot_path, dpi=200)
    plt.close()

    print("Generating Decision Tree Flowchart...")
    features = ['temperature_2m', 'relative_humidity_2m', 'cloud_cover', 'surface_pressure', 'wind_speed_10m']
    viz_model = DecisionTreeRegressor(max_depth=3).fit(df[features], df['rain'])
    plt.figure(figsize=(16, 6))
    plot_tree(viz_model, feature_names=features, filled=True, rounded=True, fontsize=10)
    plt.title("Rainfall Prediction Logic: How the AI Decides")
    plt.savefig(tree_plot_path, dpi=200)
    plt.close()

if not df.empty:
    dt_full = DecisionTreeRegressor(max_depth=10, random_state=42).fit(df[features], df['rain'])
    districts = sorted(df['district_name'].unique().tolist())
else:
    districts = ["Chennai", "Madurai", "Coimbatore"]

def generate_report(district, temp, hum, cloud, press, wind):
    if not df.empty:
        rain_val = dt_full.predict([[temp, hum, cloud, press, wind]])[0]
        rain = max(0, round(rain_val, 2))
    else:
        rain = 2.10

    if rain > 15:
        risk = "EXTREME RAINFALL ALERT"
        agri = "**Agri:** Stop irrigation. Secure livestock in upland areas. High crop rot risk."
        infra = "**Infrastructure:** Clear arterial drains. Deploy emergency pumps in lowlands."
        safety = "**Safety:** Flash flood warning. Avoid travel and stay away from riverbanks."
    elif rain > 5:
        risk = "MODERATE RAINFALL"
        agri = "**Agri:** Beneficial for rain-fed crops. Watch for fungal pests due to moisture."
        infra = "**Infrastructure:** Clear street debris. Expect traffic slowdowns on wet roads."
        safety = "**Safety:** Visibility reduced. Carry rain protection and drive cautiously."
    else:
        risk = "STABLE / DRY"
        agri = "**Agri:** Dry spell detected. Irrigate in early morning to prevent evaporation."
        infra = "**Infrastructure:** Ideal for road construction, painting, and outdoor repairs."
        safety = "**Safety:** No weather travel alerts. Keep hydrated in high heat conditions."

    hum_tip = "*Note: High humidity detected (>85%), increasing fungal risks.*" if hum > 85 else ""

    markdown_report = f"""
### Forecast Summary
- **Predicted Rainfall:** `{rain} mm`
- **Alert Status:** **{risk}**

### Strategic Recommendations
- {agri}
- {infra}
- {safety}

---
{hum_tip}
"""
    return markdown_report, trend_plot_path, tree_plot_path

with gr.Blocks(title="Tamil Nadu Smart Climate Hub") as demo:
    gr.Markdown(
        """
        # Tamil Nadu Smart Climate Hub
        Business Intelligence tool analyzing historical data and predicting local impacts of weather patterns.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Model Parameters")
            district_dropdown = gr.Dropdown(districts, value=districts[0], label="Select District")
            temp_slider = gr.Slider(15, 50, value=32, label="Temperature (°C)")
            hum_slider = gr.Slider(0, 100, value=75, label="Humidity (%)")
            cloud_slider = gr.Slider(0, 100, value=50, label="Cloud Cover (%)")
            press_number = gr.Number(value=1008, label="Surface Pressure (hPa)")
            wind_slider = gr.Slider(0, 50, value=10, label="Wind Speed (m/s)")
            
            analyze_btn = gr.Button("Generate Insights", variant="primary")
            
        with gr.Column(scale=2):
            gr.Markdown("### 2. Decision Support Summary")
            report_output = gr.Markdown(label="Report")
            
    with gr.Row():
        with gr.Tab("Climate Evolution Trends"):
            trend_image_output = gr.Image(label="Historical Weather Trends")
            
        with gr.Tab("Decision Tree Logic"):
            tree_image_output = gr.Image(label="AI Model Logic Breakdown")

    analyze_btn.click(
        fn=generate_report,
        inputs=[district_dropdown, temp_slider, hum_slider, cloud_slider, press_number, wind_slider],
        outputs=[report_output, trend_image_output, tree_image_output]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, server_port=7861)