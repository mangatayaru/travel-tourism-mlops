from pathlib import Path
import pandas as pd

class HotelRecommender:
    """Simple popularity recommender with optional location and price filters."""
    def __init__(self, hotels_path):
        df = pd.read_csv(hotels_path)
        needed = {"name","place","price","total","days","userCode"}
        if not needed.issubset(df.columns):
            raise ValueError(f"Hotel file must contain {sorted(needed)}")
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["total"] = pd.to_numeric(df["total"], errors="coerce")
        df["days"] = pd.to_numeric(df["days"], errors="coerce")
        self.hotels = df.dropna(subset=["name","place","price"]).copy()
        self.summary = (self.hotels.groupby(["name","place"], dropna=False)
                        .agg(avg_daily_price=("price","mean"), avg_total=("total","mean"),
                             bookings=("userCode","count"), avg_days=("days","mean"))
                        .reset_index())
        self.summary["score"] = self.summary["bookings"] / (1 + self.summary["bookings"].max())

    def recommend(self, place=None, max_daily_price=None, limit=10):
        result = self.summary.copy()
        if place and place.strip():
            mask = result["place"].astype(str).str.contains(place.strip(), case=False, na=False)
            result = result[mask]
        if max_daily_price is not None:
            result = result[result["avg_daily_price"] <= float(max_daily_price)]
        result = result.sort_values(["bookings","avg_daily_price"], ascending=[False,True]).head(int(limit))
        return result.reset_index(drop=True)
