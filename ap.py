import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ---------- Load data ----------

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "Statcast_2021.csv"   # CSV is in the same folder

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

df = load_data(DATA_PATH)

# ---------- Datapage construction ----------
st.title("Predicting Pitch Usage Based Off Situation")
st.subheader("Using MLB statcast data from 2021, I will attempt to visualize different pitch usage based on the following factors:")
st.write("1. Runners on Base (ROB)  2. Outs When Up  3. Amount of Balls  4. Amount of Strikes")
st.header("1. Introduction to data")

# Goal explanation
st.write(
    "The goal of this Streamlit site is to analyze how the amount of runners on base, "
    "count, and outs can influence a pitcher's choice to throw a given pitch."
)

# ----- Create runners_on_base -----
df["runners_on_base"] = df[["on_1b", "on_2b", "on_3b"]].notna().sum(axis=1)

# Keep only columns we care about
chart_data = df[["pitch_type", "runners_on_base", "outs_when_up", "balls", "strikes"]]

st.write("Pitch type, runners on base per pitch, outs, amount of balls, and amount of strikes in a given at bat:")
st.dataframe(chart_data.head(50))

# ---------- Section 2: Frequency of Pitch for a given situation ----------

st.header("2. Frequency of Pitch for a Given Situation (ROB, Outs, Count)")

# --- Sidebar filters ---
runners_options = sorted(df["runners_on_base"].dropna().unique())
outs_options = sorted(df["outs_when_up"].dropna().unique())
balls_options = sorted(df["balls"].dropna().unique())
strikes_options = sorted(df["strikes"].dropna().unique())

selected_runners = st.sidebar.selectbox(
    "Runners on base",
    runners_options,
    index=0
)

selected_outs = st.sidebar.selectbox(
    "Outs",
    outs_options,
    index=0
)

selected_balls = st.sidebar.selectbox(
    "Balls in count",
    balls_options,
    index=0
)

selected_strikes = st.sidebar.selectbox(
    "Strikes in count",
    strikes_options,
    index=0
)

# --- Filter data for that game state ---
subset = df[
    (df["runners_on_base"] == selected_runners) &
    (df["outs_when_up"] == selected_outs) &
    (df["balls"] == selected_balls) &
    (df["strikes"] == selected_strikes)
]

st.write(
    f"Frequency of a given pitch with {selected_runners} runner(s) on base, "
    f"{selected_outs} outs, and a {selected_balls}-{selected_strikes} count: "
    f"**{len(subset)}**"
)

if subset.empty:
    st.warning("No pitches found for this combination of runners, outs, and count.")
else:
    # Count pitch types
    pitch_counts = subset["pitch_type"].value_counts()

    # --- Pie chart with matplotlib ---
    fig, ax = plt.subplots()
    ax.pie(
        pitch_counts.values,
        labels=pitch_counts.index,
        autopct="%1.1f%%"
    )
    ax.set_title(
        f"Pitch Type Frequency\n"
        f"Runners on base: {selected_runners}, Outs: {selected_outs}, "
        f"Count: {selected_balls}-{selected_strikes}"
    )

    st.pyplot(fig)

# ---------- Section 3: Analytic Prediction / Scatter ----------

st.header("3. Analytic Prediction Given this Information")
st.write(
    "From observing the trends in each different frequency of pitch given a certain situation, "
    "I want to test whether there is a relationship between fastball frequency and "
    "the sum of the indices of the situation (ROB, outs, and balls). "
    "We define ROBOB = runners_on_base + outs_when_up + balls."
)

# ROBOB as a numeric sum
df["ROBOB"] = df["runners_on_base"] + df["outs_when_up"] + df["balls"]

# Define fastball using pitch_name containing 'Fastball'
fastball_mask = df["pitch_name"].str.contains("Fastball", na=False)
# If you prefer pitch_type codes instead, you could do:
# fastball_mask = df["pitch_type"].isin(["FF", "FA", "SI", "FT"])

df["is_fastball"] = fastball_mask

# Group by ROBOB and compute average fastball percentage
fastball_vs_robob = (
    df.groupby("ROBOB")["is_fastball"]
      .mean()
      .mul(100)        # convert to %
      .reset_index()
      .sort_values("ROBOB")
)

# Keep only ROBOB values from 0 to 10
fastball_vs_robob = fastball_vs_robob[
    (fastball_vs_robob["ROBOB"] >= 0) & (fastball_vs_robob["ROBOB"] <= 10)
]

fastball_vs_robob.rename(columns={"is_fastball": "fastball_pct"}, inplace=True)

st.write("Average fastball percentage for each ROBOB value (0–10):")
st.dataframe(fastball_vs_robob)

# Scatter plot: ROBOB vs average fastball %
# x = ROBOB, y = average fastball percentage
x = fastball_vs_robob["ROBOB"].values
y = fastball_vs_robob["fastball_pct"].values

# ---- Linear fit ----
m_lin, b_lin = np.polyfit(x, y, 1)
y_pred_lin = m_lin * x + b_lin
ss_res_lin = np.sum((y - y_pred_lin) ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
r2_lin = 1 - ss_res_lin / ss_tot

# ---- Log fit: y = a * ln(ROBOB + 1) + b ----
x_log = np.log(x + 1)  # avoid ln(0)
a_log, b_log = np.polyfit(x_log, y, 1)
y_pred_log = a_log * x_log + b_log
ss_res_log = np.sum((y - y_pred_log) ** 2)
r2_log = 1 - ss_res_log / ss_tot

# ---- Power / exponential-in-log fit: y = A * (ROBOB + 1)^B ----
# Only use points with y > 0 for log transform
mask_pos = y > 0
x_pos = x[mask_pos]
y_pos = y[mask_pos]

x_pos_log = np.log(x_pos + 1)
y_pos_log = np.log(y_pos)

B_pow, logA_pow = np.polyfit(x_pos_log, y_pos_log, 1)
A_pow = np.exp(logA_pow)

# Predicted y for original x (where y > 0)
y_pred_pow = A_pow * (x_pos + 1) ** B_pow
ss_res_pow = np.sum((y_pos - y_pred_pow) ** 2)
ss_tot_pos = np.sum((y_pos - y_pos.mean()) ** 2)
r2_pow = 1 - ss_res_pow / ss_tot_pos

st.write(f"Linear fit R²: **{r2_lin:.3f}**")
st.write(f"Log fit R²: **{r2_log:.3f}**")
st.write(f"Power (exp-in-log) fit R²: **{r2_pow:.3f}**")

# ---- Interactive controls ----

fit_choice = st.multiselect(
    "Which trendline(s) to display?",
    ["Linear", "Log", "Power (exp-in-log)"],
    default=["Linear", "Power (exp-in-log)"]
)

# x-axis range slider (ROBOB)
robob_min = int(x.min())
robob_max = int(x.max())
x_range = st.slider(
    "ROBOB range to display",
    min_value=robob_min,
    max_value=robob_max,
    value=(robob_min, robob_max),
    step=1
)

# Mask data to selected ROBOB range
mask_range = (fastball_vs_robob["ROBOB"] >= x_range[0]) & (fastball_vs_robob["ROBOB"] <= x_range[1])
x_plot = fastball_vs_robob.loc[mask_range, "ROBOB"].values
y_plot = fastball_vs_robob.loc[mask_range, "fastball_pct"].values

fig, ax = plt.subplots()

# Scatter points
ax.scatter(x_plot, y_plot, label="Data", alpha=0.8)

# Points where we’ll draw the lines
x_line = np.linspace(x_range[0], x_range[1], 200)

# Linear line
if "Linear" in fit_choice:
    y_line_lin = m_lin * x_line + b_lin
    ax.plot(x_line, y_line_lin, label=f"Linear fit (R²={r2_lin:.3f})")

# Log line
if "Log" in fit_choice:
    x_line_log = np.log(x_line + 1)
    y_line_log = a_log * x_line_log + b_log
    ax.plot(x_line, y_line_log, label=f"Log fit (R²={r2_log:.3f})")

# Power / exp-in-log line
if "Power (exp-in-log)" in fit_choice:
    y_line_pow = A_pow * (x_line + 1) ** B_pow
    ax.plot(x_line, y_line_pow, label=f"Power fit (R²={r2_pow:.3f})")

ax.set_xlabel("ROBOB = Runners on base + Outs + Balls")
ax.set_ylabel("Average fastball percentage (%)")
ax.set_title("Average Fastball % vs ROBOB (Model Comparison)")
ax.set_ylim(30, 55)
ax.legend()

st.pyplot(fig)
st.write(" The chart displays a linear regression fit to the relationship between ROBOB and fastball percentage with a R^2 value of 0.513. Although this is a significantly low R^2, value, the chart still displays a visual relationship between ROBAB and fastball percentage.")

st.header("4.Conclusion")
st.write(" Both the exponential and linear trendlines do not effeciiently model the relationship between ROBAB and fastball percentage. However, we can see visualy that larger ROBAB values do correlate with higher fastball percentages. This means that ROBOB can be thought of as a general indicator for MLB hitters of the pitch they are going to recieve. Furthermore, whe a pitcher is regulated to throwing only fastballs, they are generally less effective. Therefore, ROBAB should be used as a indicator of success and MLB teams can strategize in order to increase there ROBOB. ")