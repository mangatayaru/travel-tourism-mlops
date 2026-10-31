import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import streamlit as st
import pandas as pd
import requests
from src.travel_ml.recommender import HotelRecommender

st.set_page_config(page_title="Travel ML Explorer", page_icon="✈️", layout="wide")
st.title("Travel & Tourism ML Explorer")
st.caption("Flight price estimates and hotel recommendations from the capstone datasets.")
tab1, tab2, tab3 = st.tabs(["Flight price API", "Hotel recommendations", "Dataset overview"])
with tab1:
    st.subheader("Predict a flight price")
    api_url=st.text_input("Flask API URL","http://127.0.0.1:5000")
    c1,c2=st.columns(2)
    with c1:
        origin=st.text_input("Origin","New York")
        destination=st.text_input("Destination","London")
        flight_type=st.text_input("Flight type","firstClass")
        agency=st.text_input("Agency","CloudFy")
    with c2:
        duration=st.number_input("Duration (dataset units)",min_value=0.0,value=7.5)
        distance=st.number_input("Distance (dataset units)",min_value=0.0,value=5585.0)
        date=st.date_input("Travel date")
    if st.button("Get price estimate"):
        payload={"from":origin,"to":destination,"flightType":flight_type,"agency":agency,
                 "time":duration,"distance":distance,"date":str(date)}
        try:
            r=requests.post(api_url.rstrip("/")+"/predict",json=payload,timeout=15)
            if r.ok:
                data=r.json(); st.metric("Estimated price",f'{data["predicted_price"]:,.2f}')
                st.caption(data.get("currency",""))
            else: st.error(f"API returned {r.status_code}: {r.text}")
        except requests.RequestException as e: st.error(f"Cannot reach Flask API: {e}")
with tab2:
    st.subheader("Hotel recommendations")
    hotels_path=ROOT/"data"/"hotels.csv"
    if hotels_path.exists():
        rec=HotelRecommender(hotels_path)
        place=st.text_input("Filter by place (optional)")
        max_price=st.number_input("Maximum average daily price (0 = no limit)",min_value=0.0,value=0.0)
        limit=st.slider("Number of recommendations",3,20,10)
        results=rec.recommend(place or None,max_price if max_price>0 else None,limit)
        st.dataframe(results, use_container_width=True)
        if not results.empty:
            st.bar_chart(results.set_index("name")["bookings"])
    else: st.warning("Place data/hotels.csv in the repository root.")
with tab3:
    for fname in ["flights.csv","hotels.csv","users.csv"]:
        path=ROOT/"data"/fname
        if path.exists():
            df=pd.read_csv(path)
            with st.expander(f"{fname}: {len(df):,} rows × {len(df.columns)} columns"):
                st.write("Missing values by column")
                st.dataframe(df.isna().sum().rename("missing").to_frame())
                st.dataframe(df.head(10),use_container_width=True)
        else: st.warning(f"Missing {path}")
