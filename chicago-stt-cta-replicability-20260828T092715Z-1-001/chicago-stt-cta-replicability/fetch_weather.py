"""
fetch_weather.py — Download hourly weather data for the Chicago Loop area
from the Open-Meteo Archive API and save as CSV.

Covers: 2025-01-01 to 2025-12-31, hourly resolution.
Features: temperature_2m (°C), precipitation (mm).

The Loop centroid is used (single point query covers the small 3km² area).

Dependencies:
    pip install openmeteo-requests requests-cache retry-requests
"""

import os
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry

import config

# ── Loop centroid (center of the bounding box) ──
LAT = (41.86731 + 41.89063) / 2   # ≈ 41.8790
LON = (-87.63829 + -87.60246) / 2  # ≈ -87.6204

def fetch_weather():
    """Fetch hourly weather from Open-Meteo Archive API and save as CSV."""
    print("Fetching hourly weather data from Open-Meteo …")

    # Setup API client with cache and retry
    cache_session = requests_cache.CachedSession(
        os.path.join(config.DATA_DIR, '.weather_cache'),
        expire_after=-1
    )
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": "2025-01-01",
        "end_date": "2025-12-31",
        "hourly": ["temperature_2m", "precipitation"],
        "timezone": "America/Chicago",
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    print(f"  Coordinates: {response.Latitude():.4f}°N {response.Longitude():.4f}°E")
    print(f"  Elevation: {response.Elevation()} m asl")

    hourly = response.Hourly()
    temperature = hourly.Variables(0).ValuesAsNumpy()
    precipitation = hourly.Variables(1).ValuesAsNumpy()

    # Build DataFrame
    time_range = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )
    # Convert to Chicago local time (matches our traffic/TNP data)
    time_range = time_range.tz_convert("America/Chicago").tz_localize(None)

    df = pd.DataFrame({
        "datetime": time_range,
        "temperature_2m": temperature,
        "precipitation": precipitation,
    })

    # Save
    out_path = os.path.join(config.DATA_DIR, "weather_loop_2025.csv")
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows to {out_path}")
    print(f"  Temperature range: [{df['temperature_2m'].min():.1f}, "
          f"{df['temperature_2m'].max():.1f}] °C")
    print(f"  Max precipitation: {df['precipitation'].max():.1f} mm")
    return df


if __name__ == "__main__":
    df = fetch_weather()
    print("\nPreview:")
    print(df.head(10))
    print(f"\nShape: {df.shape}")
