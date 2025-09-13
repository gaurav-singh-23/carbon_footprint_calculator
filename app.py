import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
from functions import *

st.set_page_config(layout="wide", page_title="Carbon Footprint Calculator", page_icon="./media/favicon.ico")

# ===================
# Dark Theme with CSS
# ===================
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #121212 !important;
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
    }
    h1, h2, h3, h4, h5, h6, label, p, span, div {
        color: #FFFFFF !important;
    }
    .stButton button {
        background-color: #2E7D32 !important;
        color: white !important;
        border-radius: 8px;
    }
    .stButton button:hover {
        background-color: #388E3C !important;
        color: #fff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===================
# Input Tabs
# ===================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["👴 Personal","🚗 Travel","🗑️ Waste","⚡ Energy","💸 Consumption"])

def component():
    tab1col1, tab1col2 = tab1.columns(2)
    height = tab1col1.number_input("Height (cm)", min_value=0, max_value=250, value=160)
    weight = tab1col2.number_input("Weight (kg)", min_value=0, max_value=250, value=70)
    calculation = weight / (height/100)**2 if height > 0 else 1
    body_type = "underweight" if (calculation < 18.5) else \
                "normal" if (calculation < 25) else \
                "overweight" if (calculation < 30) else "obese"

    sex = tab1.selectbox('Gender', ["female", "male"])
    diet = tab1.selectbox('Diet', ['omnivore', 'pescatarian', 'vegetarian', 'vegan'])
    social = tab1.selectbox('Social Activity', ['never', 'sometimes', 'often'])

    transport = tab2.selectbox('Transportation', ['public', 'private', 'walk/bicycle'])
    if transport == "private":
        vehicle_type = tab2.selectbox('Vehicle Type', ['petrol', 'diesel', 'hybrid', 'lpg', 'electric'])
    else:
        vehicle_type = "None"

    vehicle_km = 0 if transport == "walk/bicycle" else tab2.number_input(
        'Monthly distance traveled by vehicle (km)', min_value=0, max_value=5000, step=50, value=0
    )

    air_travel = tab2.radio('How often did you fly last month?', ['never', 'rarely', 'frequently', 'very frequently'])

    waste_bag = tab3.selectbox('Waste bag size', ['small', 'medium', 'large', 'extra large'])
    waste_count = tab3.number_input('Waste bags per week', min_value=0, max_value=10, value=2)
    recycle = tab3.multiselect('Do you recycle?', ['Plastic', 'Paper', 'Metal', 'Glass'])

    heating_energy = tab4.selectbox('Heating energy source', ['natural gas', 'electricity', 'wood', 'coal'])
    for_cooking = tab4.multiselect('Cooking systems', ['microwave', 'oven', 'grill', 'airfryer', 'stove'])
    energy_efficiency = tab4.radio('Do you consider energy efficiency?', ['No', 'Yes', 'Sometimes'])
    daily_tv_pc = tab4.number_input('Daily PC/TV use (hours)', min_value=0, max_value=24, value=4)
    internet_daily = tab4.number_input('Daily internet use (hours)', min_value=0, max_value=24, value=6)

    shower = tab5.radio('How often do you shower?', ['daily', 'twice a day', 'more frequently', 'less frequently'])
    grocery_bill = tab5.number_input('Monthly grocery spending ($)', min_value=0, max_value=500, step=10, value=150)
    clothes_monthly = tab5.number_input('Clothes bought per month', min_value=0, max_value=30, value=3)

    data = {'Body Type': body_type, "Sex": sex, 'Diet': diet, "How Often Shower": shower,
            "Heating Energy Source": heating_energy, "Transport": transport, "Social Activity": social,
            'Monthly Grocery Bill': grocery_bill, "Frequency of Traveling by Air": air_travel,
            "Vehicle Monthly Distance Km": vehicle_km, "Waste Bag Size": waste_bag,
            "Waste Bag Weekly Count": waste_count, "How Long TV PC Daily Hour": daily_tv_pc,
            "Vehicle Type": vehicle_type, "How Many New Clothes Monthly": clothes_monthly,
            "How Long Internet Daily Hour": internet_daily, "Energy efficiency": energy_efficiency}

    data.update({f"Cooking_with_{x}": 1 for x in for_cooking})
    data.update({f"Do You Recyle_{x}": 1 for x in recycle})

    return pd.DataFrame(data, index=[0])

df = component()
data = input_preprocessing(df)

sample_df = pd.DataFrame(data=sample, index=[0])
sample_df[sample_df.columns] = 0
sample_df[data.columns] = data

# ===================
# Show results only when user clicks Calculate
# ===================
if st.button("🚀 Calculate My Footprint", use_container_width=True):
    ss = pickle.load(open("./models/scale.sav", "rb"))
    model = pickle.load(open("./models/model.sav", "rb"))
    prediction = round(model.predict(ss.transform(sample_df))[0])

    st.subheader("🌍 Your Carbon Footprint Summary")
    st.metric("Estimated Monthly Emission", f"{prediction} kgCO₂e")

    avg_emission = 1000
    global_target = 500
    if prediction > avg_emission:
        st.error(f"⚠️ {prediction - avg_emission} kgCO₂e above average")
    else:
        st.success(f"✅ {avg_emission - prediction} kgCO₂e below average")

    st.progress(min(int(prediction / avg_emission * 100), 100))

    tree_count = round(prediction / 411.4)
    st.markdown(
        f"🌳 You need to plant **{tree_count} tree{'s' if tree_count > 1 else ''}** monthly to offset."
    )

    # 📊 Donut Chart + Bar Chart Side by Side
    st.subheader("📊 Emission Breakdown & Comparison")
    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(2.5, 2.5))   # smaller donut
        ax1.pie(
            [prediction, avg_emission],
            labels=["You", "Avg"],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops=dict(width=0.35)
        )
        ax1.set_aspect("equal")
        st.pyplot(fig1)

    with col2:
        categories = ["You", "Average", "Global Target"]
        values = [prediction, avg_emission, global_target]

        fig2, ax2 = plt.subplots(figsize=(3, 2.5))  # smaller bar chart
        bars = ax2.bar(categories, values, color=["#2E7D32", "#0277BD", "#FBC02D"])
        ax2.set_ylabel("kgCO₂e / month")
        ax2.set_ylim(0, max(values) + 300)

        for bar in bars:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                     f"{int(bar.get_height())}", ha="center", color="white", fontsize=8)

        st.pyplot(fig2)


# ===================
# Footer
# ===================
st.markdown(
    """
    <div style="text-align: center; margin-top: 20px; color: #aaa; font-size: 14px;">
        🌑 Dark Theme | Made with ❤️ by Gaurav Kumar Singh
    </div>
    """,
    unsafe_allow_html=True
)
