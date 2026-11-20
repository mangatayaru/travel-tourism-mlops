
import os
import sys
from pathlib import Path

import requests
import streamlit as st

# Make project root available for importing src modules
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.travel_ml.recommender import HotelRecommender

st.set_page_config(
    page_title="Travel & Tourism Dashboard",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Travel & Tourism Dashboard")
st.caption("Flight Price Prediction and Hotel Recommendations")

# --------------------------------------------------
# Sidebar: Flask API status
# --------------------------------------------------

api_url = st.sidebar.text_input(
    "Flask API URL",
    os.getenv("FLASK_API_URL", "http://127.0.0.1:5000")
).rstrip("/")

try:
    health = requests.get(f"{api_url}/health", timeout=4)

    if health.ok and health.json().get("model_loaded"):
        st.sidebar.success("Flask API is online")
    else:
        st.sidebar.error("API is not ready")

except requests.RequestException:
    st.sidebar.warning("Start Flask API for flight predictions.")

# --------------------------------------------------
# Load Hotel Recommender
# --------------------------------------------------

@st.cache_resource
def load_recommender():
    return HotelRecommender(
        PROJECT_ROOT / "data" / "hotels.csv"
    )

try:
    recommender = load_recommender()
    recommender_ready = True
except Exception as exc:
    recommender_ready = False
    st.error(f"Could not load hotel recommender: {exc}")

# --------------------------------------------------
# Application tabs
# --------------------------------------------------

flight_tab, hotel_tab = st.tabs([
    "✈️ Flight Price Prediction",
    "🏨 Hotel Recommendations"
])

# --------------------------------------------------
# Flight Price Prediction
# --------------------------------------------------

with flight_tab:

    st.header("Flight Price Prediction")

    with st.form("flight_form"):

        col1, col2 = st.columns(2)

        origin = col1.text_input("From", "Delhi")
        destination = col2.text_input("To", "Mumbai")

        flight_type = st.selectbox(
            "Flight type",
            ["economy", "business", "first"]
        )

        time_value = st.text_input(
            "Time",
            "morning",
            help="Use the same format used in your training data."
        )

        distance = st.number_input(
            "Distance",
            min_value=1.0,
            value=1150.0,
            step=10.0
        )

        agency = st.text_input("Agency", "Airline")
        travel_date = st.date_input("Travel date")

        submitted = st.form_submit_button(
            "Predict flight price",
            type="primary"
        )

    if submitted:

        payload = {
            "from": origin,
            "to": destination,
            "flightType": flight_type,
            "time": str(time_value),
            "distance": float(distance),
            "agency": agency,
            "date": travel_date.isoformat()
        }

        try:
            response = requests.post(
                f"{api_url}/predict",
                json=payload,
                timeout=30
            )

            if response.ok:

                result = response.json()
                predictions = result.get("predictions", [])

                if predictions:
                    predicted_price = predictions[0]["predicted_price"]

                    st.metric(
                        "Predicted Flight Price",
                        f"{predicted_price:,.2f}"
                    )

                    st.caption(
                        f"Currency: {result.get('currency', 'Not specified')}"
                    )

                else:
                    st.warning("No prediction returned by API.")

            else:
                st.error(
                    f"API error ({response.status_code}): {response.text}"
                )

        except requests.RequestException as exc:
            st.error(f"Could not reach Flask API: {exc}")

    with st.expander("Model/API details"):
        try:
            info = requests.get(
                f"{api_url}/model-info",
                timeout=4
            )

            if info.ok:
                st.json(info.json())
            else:
                st.write(info.text)

        except requests.RequestException as exc:
            st.write(str(exc))

# --------------------------------------------------
# Hotel Recommendations
# --------------------------------------------------

with hotel_tab:

    st.header("🏨 Hotel Recommendations")

    st.write(
        "Discover popular hotels based on booking frequency, "
        "destination and daily price."
    )

    if recommender_ready:

        places = sorted(
            recommender.summary["place"].dropna().unique().tolist()
        )

        col1, col2, col3 = st.columns(3)

        selected_place = col1.selectbox(
            "Select destination",
            ["All destinations"] + places
        )

        max_price = col2.number_input(
            "Maximum daily price",
            min_value=0.0,
            value=1000.0,
            step=50.0
        )

        limit = col3.number_input(
            "Number of recommendations",
            min_value=1,
            max_value=9,
            value=9
        )

        if st.button("Get Hotel Recommendations", type="primary"):

            place_filter = (
                None if selected_place == "All destinations"
                else selected_place
            )

            recommendations = recommender.recommend(
                place=place_filter,
                max_daily_price=max_price,
                limit=int(limit)
            )

            if recommendations.empty:

                st.warning(
                    "No hotels match the selected filters. "
                    "Try increasing the maximum daily price."
                )

            else:

                st.success(
                    f"Found {len(recommendations)} hotel recommendations."
                )

                st.dataframe(
                    recommendations.rename(columns={
                        "name": "Hotel",
                        "place": "Destination",
                        "bookings": "Bookings",
                        "avg_daily_price": "Average Daily Price",
                        "avg_total": "Average Total Price",
                        "score": "Popularity Score"
                    }).style.format({
                        "Average Daily Price": "{:.2f}",
                        "Average Total Price": "{:.2f}",
                        "Popularity Score": "{:.2%}"
                    }),
                    use_container_width=True,
                    hide_index=True
                )

                st.subheader("Top Recommended Hotel")

                top = recommendations.iloc[0]

                c1, c2, c3 = st.columns(3)

                c1.metric("Hotel", top["name"])
                c2.metric("Destination", top["place"])
                c3.metric("Bookings", f"{int(top['bookings']):,}")

                st.caption(
                    "Ranking is based on historical booking frequency. "
                    "Prices are dataset averages; currency is unspecified."
                )

    else:
        st.error("Hotel recommender could not be initialized.")
