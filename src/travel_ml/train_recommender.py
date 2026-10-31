import argparse, json
from pathlib import Path
from src.travel_ml.recommender import HotelRecommender

def main(hotels, users, output):
    # Validate both input files and preserve a clear relationship check for documentation.
    import pandas as pd
    h=pd.read_csv(hotels); u=pd.read_csv(users)
    if "userCode" not in h or "code" not in u:
        raise ValueError("Expected hotels.userCode and users.code")
    known=set(u["code"])
    matched=int(h["userCode"].isin(known).sum())
    rec=HotelRecommender(hotels)
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    payload={"hotel_rows":int(len(h)),"user_rows":int(len(u)),"hotel_user_matches":matched,
             "unique_hotel_place_pairs":int(len(rec.summary)),
             "approach":"Popularity ranking with optional place and daily-price filters"}
    (out/"recommender_metadata.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--hotels",default="data/hotels.csv"); p.add_argument("--users",default="data/users.csv"); p.add_argument("--output",default="artifacts")
    a=p.parse_args(); main(a.hotels,a.users,a.output)
