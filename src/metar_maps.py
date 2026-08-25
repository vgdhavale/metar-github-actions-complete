# -*- coding: utf-8 -*-
"""
METAR scraper and Indian meteorological maps

Outputs:
    metar_dataframe.csv
    metar_station_observations.png
    metar_pressure_contours.png
    metar_temperature_contours.png
    metar_dewpoint_contours.png
    metar_wind_speed_barbs.png
    metar_visibility.png
    metar_current_weather.png
"""

import re
import warnings
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

from scipy.interpolate import griddata

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from metpy.units import units
from metpy.plots import StationPlot, StationPlotLayout, sky_cover


warnings.filterwarnings("ignore", message="Unverified HTTPS request")


# ============================================================
# Configuration
# ============================================================

URL = (
    "https://olbs.amsschennai.gov.in/"
    "nsweb/FlightBriefing/showmetars.php"
)

OUTPUT_CSV = "metar_dataframe.csv"

MAP_EXTENT = [66.0, 100.0, 5.0, 38.0]
PROJECTION = ccrs.PlateCarree()


# ============================================================
# Station coordinates
# ============================================================

# Format:
# ICAO: (latitude, longitude, station name)

STATION_COORDS = {
    # Western India
    "VAAH": (23.0772, 72.6347, "Ahmedabad"),
    "VAAM": (20.5937, 78.9629, "Amravati"),
    "VAAU": (19.8647, 75.3959, "Aurangabad"),
    "VABB": (19.0887, 72.8679, "Mumbai"),
    "VABJ": (23.1778, 79.9658, "Jabalpur"),
    "VABO": (22.3358, 73.2250, "Vadodara"),
    "VABP": (23.2875, 77.3374, "Bhopal"),
    "VABV": (20.8123, 76.6974, "Bhavnagar"),
    "VADU": (22.0156, 70.7794, "Rajkot Hirasar"),
    "VAGD": (21.1070, 70.3312, "Keshod"),
    "VAGO": (15.2891, 73.9632, "Goa Dabolim"),
    "VAHS": (17.4519, 78.4283, "Hyderabad Begumpet"),
    "VAID": (22.7217, 75.8011, "Indore"),
    "VAJB": (23.8824, 78.2250, "Jalgaon"),
    "VAJL": (25.7533, 71.3967, "Jalor"),
    "VAJM": (22.4641, 70.0108, "Jamnagar"),
    "VAKE": (21.4322, 70.0894, "Kandla"),
    "VAKJ": (23.5133, 74.3167, "Khajuraho"),
    "VAKP": (16.7047, 74.2831, "Kolhapur"),
    "VAKS": (21.4114, 72.6369, "Keshod Junagadh"),
    "VAND": (21.0922, 79.0472, "Nagpur"),
    "VANM": (20.0125, 73.9142, "Nashik"),
    "VANP": (21.1500, 79.0833, "Nagpur Sonegaon"),
    "VAOZ": (20.1194, 73.9142, "Ozar Nashik"),
    "VAPO": (18.5821, 73.9197, "Pune"),
    "VAPR": (21.6481, 69.6583, "Porbandar"),
    "VARK": (22.2656, 70.7792, "Rajkot"),
    "VARP": (21.1717, 72.7411, "Surat"),
    "VASD": (24.5833, 73.6833, "Shirdi"),
    "VASU": (21.1702, 72.7414, "Surat Magdalla"),
    "VAUD": (24.6175, 73.7178, "Udaipur"),
    "VASL": (17.6250, 75.9361, "Solapur"),
    "VAMA": (20.9443, 77.7289, "Amravati"),
    "VAJJ": (19.0981, 72.8342, "Mumbai Juhu"),

    # Eastern and northeastern India
    "VEAB": (23.4914, 87.3786, "Asansol"),
    "VEAT": (24.0622, 85.0506, "Agartala"),
    "VEAY": (27.4883, 94.8975, "Along"),
    "VEBD": (23.3644, 85.3217, "Ranchi Mall"),
    "VEBI": (25.2506, 87.0150, "Bhagalpur"),
    "VEBN": (25.4497, 82.8594, "Varanasi"),
    "VEBS": (20.2444, 85.8178, "Bhubaneswar"),
    "VEBU": (22.1833, 86.7167, "Burnpur"),
    "VECC": (22.6547, 88.4467, "Kolkata"),
    "VECO": (21.5422, 84.0544, "Cooch Behar"),
    "VEDB": (27.4839, 95.0181, "Dibrugarh"),
    "VEDG": (25.8333, 89.9667, "Durgapur"),
    "VEDH": (26.1167, 89.9833, "Dhubri"),
    "VEDO": (23.6231, 87.2458, "Durgapur Andal"),
    "VEGK": (26.7397, 83.4497, "Gorakhpur"),
    "VEGT": (26.1065, 91.5855, "Guwahati"),
    "VEGY": (25.3176, 83.0101, "Gaya"),
    "VEIM": (24.7600, 93.8967, "Imphal"),
    "VEJH": (22.4167, 86.4167, "Jharsuguda"),
    "VEJR": (26.7303, 94.1756, "Jorhat"),
    "VEJT": (23.4000, 85.3167, "Jamshedpur"),
    "VEKI": (26.8833, 94.1167, "Kailashahar"),
    "VEKU": (25.8239, 93.7719, "Dimapur"),
    "VELP": (26.1833, 91.7333, "Lakhimpur"),
    "VELR": (27.2833, 94.1000, "Lilabari"),
    "VEMN": (25.0333, 92.0000, "Malda"),
    "VEMR": (25.7500, 93.4833, "Margherita"),
    "VEPG": (24.1000, 86.3000, "Pantnagar"),
    "VEPT": (25.5908, 85.0878, "Patna"),
    "VEPU": (22.8131, 86.1644, "Purulia"),
    "VEPY": (27.2306, 88.5878, "Pakyong"),
    "VERC": (23.3142, 85.3217, "Ranchi"),
    "VERU": (22.2536, 84.8464, "Rourkela"),
    "VETJ": (26.7028, 92.7981, "Tezpur"),
    "VETZ": (27.5894, 95.5564, "Tezu"),
    "VEHO": (24.0500, 85.4167, "Hazaribagh"),
    "VEJS": (22.5119, 86.1622, "Jamshedpur"),
    "VERP": (24.0624, 84.0669, "Daltonganj"),

    # Northern India
    "VIAG": (27.1558, 77.9606, "Agra"),
    "VIAM": (27.4925, 77.6736, "Aligarh"),
    "VIAR": (31.7074, 74.7973, "Amritsar"),
    "VIAX": (29.1833, 75.7167, "Adampur"),
    "VIBK": (24.2631, 82.7164, "Bikaner"),
    "VIBR": (28.3666, 79.4522, "Bareilly"),
    "VIBT": (30.1558, 74.9458, "Bhatinda"),
    "VIBY": (26.5000, 80.5000, "Bareilly Military"),
    "VICG": (30.6733, 76.7886, "Chandigarh"),
    "VICX": (24.5833, 73.6833, "Kanpur Chakeri"),
    "VIDD": (28.6139, 77.2089, "Delhi Safdarjung"),
    "VIDN": (28.6667, 77.3167, "Dehradun"),
    "VIDP": (28.5665, 77.1031, "Delhi IGI"),
    "VIDX": (28.7041, 77.1025, "Hindan"),
    "VIGG": (32.2250, 76.2633, "Gagal Kangra"),
    "VIGR": (26.2300, 78.2272, "Gwalior"),
    "VIHR": (29.1800, 75.7200, "Hisar"),
    "VIJO": (26.2514, 73.0489, "Jodhpur"),
    "VIJP": (26.8242, 75.8122, "Jaipur"),
    "VIJR": (26.9000, 70.9000, "Jaisalmer"),
    "VIJU": (32.6889, 74.8375, "Jammu"),
    "VIKG": (32.2252, 76.2634, "Kangra"),
    "VIKO": (26.4025, 80.4125, "Kanpur"),
    "VILD": (30.9500, 75.8500, "Ludhiana"),
    "VILH": (34.1359, 77.5464, "Leh"),
    "VILK": (26.7606, 80.8893, "Lucknow"),
    "VIPK": (29.0333, 79.4833, "Pantnagar"),
    "VIND": (25.1833, 75.8500, "Kota"),
    "VIPT": (29.0333, 79.4833, "Pantnagar"),
    "VISM": (31.0833, 77.1667, "Shimla"),
    "VISR": (33.9872, 74.7742, "Srinagar"),
    "VEKO": (24.8187, 79.9164, "Khajuraho"),
    "VEJP": (26.8242, 75.8122, "Jaipur"),

    # Southern India
    "VOAT": (10.9000, 76.2333, "Agatti"),
    "VOBL": (12.9499, 77.6978, "Bengaluru Kempegowda"),
    "VOBM": (12.9613, 77.5655, "Bengaluru HAL"),
    "VOBG": (12.9500, 77.6682, "Bengaluru HAL"),
    "VOBR": (15.8592, 74.6178, "Belgaum"),
    "VOBZ": (15.1633, 76.8844, "Bellary"),
    "VOCB": (11.0300, 77.0434, "Coimbatore"),
    "VOCX": (11.0253, 77.0434, "Coimbatore"),
    "VOCI": (10.1520, 76.4019, "Kochi"),
    "VOCL": (11.8889, 75.9556, "Kozhikode"),
    "VOCP": (14.5083, 77.4667, "Cuddapah"),
    "VOML": (12.8049, 74.8911, "Mangalore"),
    "VOKN": (13.0456, 75.2422, "Mangalore Bajpe"),
    "VOMD": (11.6667, 78.1167, "Madurai"),
    "VOTR": (13.6891, 79.5429, "Tirupati"),
    "VOSR": (13.6289, 79.4192, "Tirupati"),
    "VOTV": (8.4821, 76.9200, "Thiruvananthapuram"),
    "VOGA": (15.4869, 73.8350, "Goa"),
    "VOGO": (15.3801, 73.8333, "Goa Dabolim"),
    "VOAR": (14.2831, 78.4332, "Kadapa"),
    "VOJV": (16.9831, 73.3331, "Ratnagiri"),
    "VOPB": (11.6413, 92.7297, "Port Blair"),
    "VOPC": (11.9333, 79.8105, "Puducherry"),
    "VOHB": (15.8643, 78.0256, "Kurnool"),
    "VOHS": (17.2405, 78.4294, "Hyderabad"),
    "VOHY": (17.4531, 78.4676, "Hyderabad Begumpet"),
    "VOKU": (12.5012, 76.0354, "Kushalnagar"),
    "VOMM": (12.9941, 80.1708, "Chennai"),
    "VOMY": (12.3214, 76.5925, "Mysore"),
    "VORY": (16.9830, 81.8175, "Rajahmundry"),
    "VOSM": (11.6775, 78.1642, "Salem"),
    "VOTK": (10.3712, 76.2131, "Thrissur"),
    "VOTP": (9.0514, 76.5358, "Pathanamthitta"),
    "VOVO": (15.1500, 74.0000, "Canacona"),
    "VOVZ": (17.7208, 83.2244, "Visakhapatnam"),
    "VOGB": (17.3082, 76.9652, "Kalaburagi"),
    "VOSH": (13.8580, 75.6189, "Shivamogga"),
    "VOVI": (17.9761, 83.5039, "Bhogapuram"),
    "VERK": (22.2566, 84.8152, "Rourkela"),
    "VERW": (20.9167, 85.1333, "Angul"),
}


# ============================================================
# Download and HTML processing
# ============================================================

def download_page(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/120 Safari/537.36"
        )
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=60,
        verify=False
    )

    response.raise_for_status()
    response.encoding = (
        response.apparent_encoding or response.encoding
    )

    return response.text


def html_to_text(html):
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text("\n")

    except ImportError:
        text = re.sub(
            r"<script.*?</script>",
            "",
            html,
            flags=re.IGNORECASE | re.DOTALL
        )

        text = re.sub(
            r"<style.*?</style>",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

        text = re.sub(r"<[^>]+>", "\n", text)

    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)

    return text


def extract_metar_records(html):
    text = html_to_text(html)

    records = re.findall(
        r"\bMETAR\b.*?=",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    cleaned = []

    for record in records:
        record = record.replace("\r", " ")
        record = record.replace("\n", " ")
        record = re.sub(r"\s+", " ", record).strip()

        if (
            record.upper().startswith("METAR")
            and record.endswith("=")
        ):
            cleaned.append(record)

    return list(dict.fromkeys(cleaned))


# ============================================================
# METAR decoding
# ============================================================

def decode_temperature(value):
    if value.startswith("M"):
        return -float(value[1:])

    return float(value)


def parse_wind(wind_token):
    result = {
        "wind_direction_deg": np.nan,
        "wind_speed_kt": np.nan,
        "wind_gust_kt": np.nan,
        "wind_variable": False,
    }

    if wind_token is None:
        return result

    pattern = (
        r"^(?P<direction>\d{3}|VRB)"
        r"(?P<speed>\d{2,3})"
        r"(?:G(?P<gust>\d{2,3}))?KT$"
    )

    match = re.match(pattern, wind_token)

    if match is None:
        return result

    direction = match.group("direction")

    if direction == "VRB":
        result["wind_variable"] = True
    else:
        result["wind_direction_deg"] = float(direction)

    result["wind_speed_kt"] = float(
        match.group("speed")
    )

    if match.group("gust"):
        result["wind_gust_kt"] = float(
            match.group("gust")
        )

    return result


def parse_visibility(tokens):
    if "CAVOK" in tokens:
        return 10000.0

    for token in tokens:
        if re.fullmatch(r"\d{4}", token):
            return float(token)

    return np.nan


def parse_pressure(token):
    if token is None:
        return np.nan

    if token.startswith("Q"):
        return float(token[1:])

    if token.startswith("A"):
        pressure_inhg = float(token[1:]) / 100.0
        return pressure_inhg * 33.8638866667

    return np.nan


def parse_weather(tokens):
    weather_pattern = (
        r"^(?:RE)?(?:[-+]|VC)?"
        r"(?:MI|BC|PR|DR|BL|SH|TS|FZ)?"
        r"(?:DZ|RA|SN|SG|IC|PL|GR|GS|UP|BR|FG|FU|VA|DU|SA|HZ|PY|PO|SQ|FC|SS|DS)+$"
    )

    weather = []

    for token in tokens:
        if re.fullmatch(weather_pattern, token):
            weather.append(token)

    return " ".join(weather) if weather else "NSW"


def parse_cloud_cover(tokens):
    cloud_codes = {
        "FEW": 2,
        "SCT": 4,
        "BKN": 6,
        "OVC": 8,
        "VV": 8,
    }

    for token in tokens:
        match = re.match(
            r"^(FEW|SCT|BKN|OVC|VV)\d{3}",
            token
        )

        if match:
            return int(
                cloud_codes[match.group(1)]
            )

    return np.nan


def parse_datetime(time_token):
    if time_token is None:
        return pd.NaT

    match = re.fullmatch(
        r"(\d{2})(\d{2})(\d{2})Z",
        time_token
    )

    if match is None:
        return pd.NaT

    day = int(match.group(1))
    hour = int(match.group(2))
    minute = int(match.group(3))

    now_utc = datetime.now(timezone.utc)

    try:
        return pd.Timestamp(
            datetime(
                now_utc.year,
                now_utc.month,
                day,
                hour,
                minute,
                tzinfo=timezone.utc
            )
        )

    except ValueError:
        return pd.NaT


def parse_metar(report):
    report = report.rstrip("=").strip()
    tokens = report.split()

    station_index = None

    for index, token in enumerate(tokens):
        if re.fullmatch(r"[A-Z]{4}", token):
            if token != "METAR":
                station_index = index
                break

    if station_index is None:
        return None

    station = tokens[station_index]
    body = tokens[station_index + 1:]

    time_token = next(
        (
            token for token in body
            if re.fullmatch(r"\d{6}Z", token)
        ),
        None
    )

    date_time = parse_datetime(time_token)

    wind_token = next(
        (
            token for token in body
            if re.fullmatch(
                r"(?:\d{3}|VRB)\d{2,3}"
                r"(?:G\d{2,3})?KT",
                token
            )
        ),
        None
    )

    wind = parse_wind(wind_token)

    visibility_m = parse_visibility(body)

    temperature_c = np.nan
    dew_point_c = np.nan

    temperature_token = next(
        (
            token for token in body
            if re.fullmatch(
                r"M?\d{2}/M?\d{2}",
                token
            )
        ),
        None
    )

    if temperature_token:
        temperature, dew_point = (
            temperature_token.split("/")
        )

        temperature_c = decode_temperature(
            temperature
        )

        dew_point_c = decode_temperature(
            dew_point
        )

    pressure_token = next(
        (
            token for token in body
            if re.fullmatch(
                r"(?:Q\d{4}|A\d{4})",
                token
            )
        ),
        None
    )

    pressure_hpa = parse_pressure(
        pressure_token
    )

    weather = parse_weather(body)
    cloud_coverage = parse_cloud_cover(body)

    latitude = np.nan
    longitude = np.nan
    station_name = station

    if station in STATION_COORDS:
        latitude, longitude, station_name = (
            STATION_COORDS[station]
        )

    return {
        "date_time": date_time,
        "station": station,
        "station_name": station_name,
        "latitude": latitude,
        "longitude": longitude,
        "temperature_C": temperature_c,
        "dew_point_C": dew_point_c,
        "weather": weather,
        "visibility_m": visibility_m,
        "pressure_hPa": pressure_hpa,
        "cloud_coverage": cloud_coverage,
        "wind_speed_kt": wind["wind_speed_kt"],
        "wind_gust_kt": wind["wind_gust_kt"],
        "wind_direction_deg": wind[
            "wind_direction_deg"
        ],
        "wind_variable": wind["wind_variable"],
        "raw_metar": report + " =",
    }


# ============================================================
# Common map functions
# ============================================================

def get_ist_time():
    return datetime.now(
        ZoneInfo("Asia/Kolkata")
    )


def add_map_background(axes):
    axes.set_extent(
        MAP_EXTENT,
        crs=PROJECTION
    )

    axes.add_feature(
        cfeature.LAND,
        facecolor="whitesmoke",
        zorder=0
    )

    axes.add_feature(
        cfeature.OCEAN,
        facecolor="lightblue",
        zorder=0
    )

    axes.add_feature(
        cfeature.LAKES,
        facecolor="lightblue",
        edgecolor="gray",
        linewidth=0.4,
        zorder=1
    )

    axes.add_feature(
        cfeature.RIVERS,
        edgecolor="skyblue",
        linewidth=0.4,
        zorder=1
    )

    axes.add_feature(
        cfeature.COASTLINE,
        linewidth=0.8,
        zorder=2
    )

    axes.add_feature(
        cfeature.BORDERS,
        linewidth=0.7,
        edgecolor="black",
        zorder=2
    )

    gridlines = axes.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--"
    )

    gridlines.top_labels = False
    gridlines.right_labels = False

    gridlines.xlabel_style = {"size": 8}
    gridlines.ylabel_style = {"size": 8}


def get_interpolation_grid(
    dataframe,
    variable,
    grid_resolution=0.25
):
    columns = [
        "longitude",
        "latitude",
        variable
    ]

    valid = dataframe[columns].dropna().copy()

    valid = valid.drop_duplicates(
        subset=["longitude", "latitude"]
    )

    if len(valid) < 3:
        raise ValueError(
            f"At least three valid stations are required "
            f"for {variable} interpolation."
        )

    longitude = valid["longitude"].to_numpy(
        dtype=float
    )

    latitude = valid["latitude"].to_numpy(
        dtype=float
    )

    values = valid[variable].to_numpy(
        dtype=float
    )

    grid_longitudes = np.arange(
        MAP_EXTENT[0],
        MAP_EXTENT[1] + grid_resolution,
        grid_resolution
    )

    grid_latitudes = np.arange(
        MAP_EXTENT[2],
        MAP_EXTENT[3] + grid_resolution,
        grid_resolution
    )

    grid_lon, grid_lat = np.meshgrid(
        grid_longitudes,
        grid_latitudes
    )

    points = np.column_stack(
        [longitude, latitude]
    )

    grid_values = griddata(
        points,
        values,
        (grid_lon, grid_lat),
        method="linear"
    )

    nearest_values = griddata(
        points,
        values,
        (grid_lon, grid_lat),
        method="nearest"
    )

    grid_values = np.where(
        np.isnan(grid_values),
        nearest_values,
        grid_values
    )

    return grid_lon, grid_lat, grid_values, valid


# ============================================================
# Scalar contour maps (NO INTERPOLATION - ACTUAL VALUES)
# ============================================================

def plot_contour_field(
    dataframe,
    variable,
    title,
    colorbar_label,
    output_file,
    cmap,
    contour_levels=15,
    contour_format="%.1f"
):
    """
    MODIFIED: Plot actual station values without interpolation.
    Added ICAO station codes next to each point.
    """
    
    columns = [
        "longitude",
        "latitude",
        variable,
        "station"
    ]

    valid = dataframe[columns].dropna().copy()
    valid = valid.drop_duplicates(
        subset=["longitude", "latitude"]
    )

    if len(valid) < 3:
        raise ValueError(
            f"At least three valid stations are required "
            f"for {variable} plotting."
        )

    figure = plt.figure(
        figsize=(15, 11)
    )

    axes = plt.axes(
        projection=PROJECTION
    )

    add_map_background(axes)

    # Get values for coloring
    values = valid[variable].to_numpy(dtype=float)
    
    # Create scatter plot with actual values (NO INTERPOLATION)
    scatter = axes.scatter(
        valid["longitude"],
        valid["latitude"],
        c=values,
        s=150,  # Larger markers for better visibility
        cmap=cmap,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.5,
        transform=PROJECTION,
        zorder=5
    )

    # Add value labels and ICAO codes next to each station
    for _, row in valid.iterrows():
        value = row[variable]
        station = row["station"]
        
        # Value label
        axes.text(
            row["longitude"] + 0.15,
            row["latitude"] + 0.10,
            f"{value:.0f}",
            transform=PROJECTION,
            fontsize=7,
            ha="left",
            va="bottom",
            color="black",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            zorder=6
        )
        
        # ICAO code label
        axes.text(
            row["longitude"],
            row["latitude"] - 0.25,
            station,
            transform=PROJECTION,
            fontsize=6,
            ha="center",
            va="top",
            color="black",
            zorder=6
        )

    colorbar = figure.colorbar(
        scatter,
        ax=axes,
        pad=0.02,
        shrink=0.85
    )

    colorbar.set_label(colorbar_label)

    current_time = get_ist_time()

    axes.set_title(
        f"{title} (Actual Station Values)\n"
        f"METAR observations: "
        f"{current_time:%Y-%m-%d %H:%M} IST",
        fontsize=13
    )

    figure.savefig(
        output_file,
        dpi=220,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(figure)

    print(f"Saved: {output_file}")


# ============================================================
# Wind map (NO INTERPOLATION - ACTUAL VALUES)
# ============================================================

def plot_wind_map(dataframe, output_file):
    """
    MODIFIED: Plot actual wind values without interpolation.
    Added ICAO station codes next to each point.
    """
    
    wind_data = dataframe[
        [
            "longitude",
            "latitude",
            "wind_speed_kt",
            "wind_direction_deg",
            "station"
        ]
    ].dropna().copy()

    wind_data = wind_data.drop_duplicates(
        subset=["longitude", "latitude"]
    )

    if len(wind_data) < 3:
        raise ValueError(
            "At least three stations with valid wind "
            "data are required."
        )

    figure = plt.figure(
        figsize=(15, 11)
    )

    axes = plt.axes(
        projection=PROJECTION
    )

    add_map_background(axes)

    # Get wind speed for coloring (NO INTERPOLATION)
    wind_speed = wind_data["wind_speed_kt"].to_numpy()
    
    # Scatter plot colored by wind speed
    scatter = axes.scatter(
        wind_data["longitude"],
        wind_data["latitude"],
        c=wind_speed,
        s=140,
        cmap="turbo",
        alpha=0.82,
        edgecolors="black",
        linewidths=0.5,
        transform=PROJECTION,
        zorder=5
    )

    # Meteorological direction is FROM direction.
    direction_rad = np.deg2rad(
        wind_data["wind_direction_deg"].to_numpy()
    )

    u = -wind_speed * np.sin(direction_rad)
    v = -wind_speed * np.cos(direction_rad)

    # Add wind barbs at actual station locations
    axes.barbs(
        wind_data["longitude"].to_numpy(),
        wind_data["latitude"].to_numpy(),
        u,
        v,
        length=5.5,
        linewidth=0.7,
        barbcolor="black",
        flagcolor="black",
        transform=PROJECTION,
        zorder=6
    )

    # Add wind speed labels and ICAO codes
    for _, row in wind_data.iterrows():
        speed = row["wind_speed_kt"]
        station = row["station"]
        
        # Wind speed label
        axes.text(
            row["longitude"] + 0.15,
            row["latitude"] + 0.10,
            f"{speed:.0f}kt",
            transform=PROJECTION,
            fontsize=7,
            ha="left",
            va="bottom",
            color="black",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            zorder=7
        )
        
        # ICAO code label
        axes.text(
            row["longitude"],
            row["latitude"] - 0.25,
            station,
            transform=PROJECTION,
            fontsize=6,
            ha="center",
            va="top",
            color="black",
            zorder=7
        )

    colorbar = figure.colorbar(
        scatter,
        ax=axes,
        pad=0.02,
        shrink=0.85
    )

    colorbar.set_label("Wind speed (kt)")

    current_time = get_ist_time()

    axes.set_title(
        f"Indian METAR Wind Speed and Barbs (Actual Values)\n"
        f"METAR observations: "
        f"{current_time:%Y-%m-%d %H:%M} IST",
        fontsize=13
    )

    figure.savefig(
        output_file,
        dpi=220,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(figure)

    print(f"Saved: {output_file}")


# ============================================================
# Visibility map (NEW)
# ============================================================

def plot_visibility_map(dataframe, output_file):
    """
    NEW: Plot actual visibility values without interpolation.
    Added ICAO station codes next to each point.
    """
    
    vis_data = dataframe[
        [
            "longitude",
            "latitude",
            "visibility_m",
            "station"
        ]
    ].dropna().copy()

    vis_data = vis_data.drop_duplicates(
        subset=["longitude", "latitude"]
    )

    if len(vis_data) < 3:
        raise ValueError(
            "At least three stations with valid visibility "
            "data are required."
        )

    figure = plt.figure(
        figsize=(15, 11)
    )

    axes = plt.axes(
        projection=PROJECTION
    )

    add_map_background(axes)

    # Get visibility for coloring (NO INTERPOLATION)
    visibility = vis_data["visibility_m"].to_numpy()
    
    # Scatter plot colored by visibility
    scatter = axes.scatter(
        vis_data["longitude"],
        vis_data["latitude"],
        c=visibility,
        s=140,
        cmap="YlGnBu",
        alpha=0.82,
        edgecolors="black",
        linewidths=0.5,
        transform=PROJECTION,
        zorder=5
    )

    # Add visibility labels and ICAO codes
    for _, row in vis_data.iterrows():
        vis = row["visibility_m"]
        station = row["station"]
        
        # Visibility label (in km if >= 1000m, else in m)
        if vis >= 1000:
            vis_text = f"{vis/1000:.1f}km"
        else:
            vis_text = f"{vis:.0f}m"
        
        axes.text(
            row["longitude"] + 0.15,
            row["latitude"] + 0.10,
            vis_text,
            transform=PROJECTION,
            fontsize=7,
            ha="left",
            va="bottom",
            color="black",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            zorder=6
        )
        
        # ICAO code label
        axes.text(
            row["longitude"],
            row["latitude"] - 0.25,
            station,
            transform=PROJECTION,
            fontsize=6,
            ha="center",
            va="top",
            color="black",
            zorder=6
        )

    # Add threshold lines for aviation visibility categories
    axes.axhline(
        y=0,
        color="red",
        linestyle="--",
        linewidth=0.5,
        alpha=0.3,
        label="5000m threshold"
    )

    colorbar = figure.colorbar(
        scatter,
        ax=axes,
        pad=0.02,
        shrink=0.85
    )

    colorbar.set_label("Visibility (m)")

    current_time = get_ist_time()

    axes.set_title(
        f"Indian METAR Visibility (Actual Values)\n"
        f"METAR observations: "
        f"{current_time:%Y-%m-%d %H:%M} IST",
        fontsize=13
    )

    figure.savefig(
        output_file,
        dpi=220,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(figure)

    print(f"Saved: {output_file}")


# ============================================================
# Current weather map
# ============================================================

def weather_category(weather_text):
    weather_text = str(weather_text).upper()

    if weather_text in ["NSW", "NAN", ""]:
        return "NSW"

    if any(
        code in weather_text
        for code in ["TS", "SQ", "FC"]
    ):
        return "SEVERE"

    if any(
        code in weather_text
        for code in [
            "RA", "DZ", "SN", "SG",
            "IC", "PL", "GR", "GS"
        ]
    ):
        return "PRECIPITATION"

    if any(
        code in weather_text
        for code in [
            "FG", "BR", "FU", "DU",
            "SA", "HZ", "VA"
        ]
    ):
        return "REDUCED_VISIBILITY"

    return "OTHER"


def plot_current_weather_map(dataframe, output_file):
    valid = dataframe[
        [
            "longitude",
            "latitude",
            "station",
            "weather"
        ]
    ].dropna(
        subset=["longitude", "latitude"]
    ).copy()

    if valid.empty:
        raise ValueError(
            "No valid stations available for "
            "current-weather plotting."
        )

    valid["weather"] = valid["weather"].fillna("NSW")

    valid["weather_category"] = valid[
        "weather"
    ].apply(weather_category)

    category_colors = {
        "NSW": "limegreen",
        "PRECIPITATION": "royalblue",
        "SEVERE": "red",
        "REDUCED_VISIBILITY": "orange",
        "OTHER": "purple",
    }

    figure = plt.figure(
        figsize=(15, 11)
    )

    axes = plt.axes(
        projection=PROJECTION
    )

    add_map_background(axes)

    for category, color in category_colors.items():

        subset = valid[
            valid["weather_category"] == category
        ]

        if subset.empty:
            continue

        axes.scatter(
            subset["longitude"],
            subset["latitude"],
            s=70,
            color=color,
            edgecolors="black",
            linewidths=0.6,
            label=category.replace(
                "_",
                " "
            ).title(),
            transform=PROJECTION,
            zorder=5
        )

    for _, row in valid.iterrows():

        weather_text = str(row["weather"])

        if len(weather_text) > 16:
            weather_text = weather_text[:16]

        axes.text(
            row["longitude"] + 0.18,
            row["latitude"] + 0.12,
            f'{row["station"]}\n{weather_text}',
            transform=PROJECTION,
            fontsize=6,
            ha="left",
            va="bottom",
            color="black",
            zorder=7
        )

    axes.legend(
        loc="lower left",
        fontsize=8,
        framealpha=0.9
    )

    current_time = get_ist_time()

    axes.set_title(
        f"Current Weather over India from METAR\n"
        f"METAR observations: "
        f"{current_time:%Y-%m-%d %H:%M} IST",
        fontsize=13
    )

    figure.savefig(
        output_file,
        dpi=220,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(figure)

    print(f"Saved: {output_file}")


# ============================================================
# Optional station plot
# ============================================================

def prepare_station_plot_data(data):
    data = data.copy()

    direction_rad = np.deg2rad(
        data["wind_direction_deg"].fillna(0.0).to_numpy()
    )

    speed = data[
        "wind_speed_kt"
    ].fillna(0.0).to_numpy()

    data["eastward_wind"] = (
        -speed * np.sin(direction_rad)
    )

    data["northward_wind"] = (
        -speed * np.cos(direction_rad)
    )

    cloud_values = (
        data["cloud_coverage"]
        .fillna(0)
        .astype(np.int64)
        .to_numpy()
    )

    return {
        "longitude": (
            data["longitude"].to_numpy()
            * units.degree
        ),
        "latitude": (
            data["latitude"].to_numpy()
            * units.degree
        ),
        "eastward_wind": (
            data["eastward_wind"].to_numpy()
            * units.knots
        ),
        "northward_wind": (
            data["northward_wind"].to_numpy()
            * units.knots
        ),
        "air_temperature": (
            data["temperature_C"].to_numpy()
            * units.degC
        ),
        "dew_point_temperature": (
            data["dew_point_C"].to_numpy()
            * units.degC
        ),
        "slp": (
            data["pressure_hPa"].to_numpy()
            * units.hPa
        ),
        "cloud_coverage": cloud_values,
    }


def plot_station_observations(dataframe, output_file):
    """
    RESTORED: Original version without SE corner station names.
    """
    
    dataframe = dataframe.dropna(
        subset=["latitude", "longitude"]
    ).copy()

    if dataframe.empty:
        return

    plot_data = prepare_station_plot_data(
        dataframe
    )

    figure = plt.figure(
        figsize=(16, 13)
    )

    axes = plt.axes(
        projection=PROJECTION
    )

    add_map_background(axes)

    layout = StationPlotLayout()

    layout.add_barb(
        "eastward_wind",
        "northward_wind",
        units="knots"
    )

    layout.add_value(
        "NW",
        "air_temperature",
        fmt=".0f",
        units="degC",
        color="darkred"
    )

    layout.add_value(
        "SW",
        "dew_point_temperature",
        fmt=".0f",
        units="degC",
        color="darkgreen"
    )

    layout.add_value(
        "NE",
        "slp",
        fmt=".0f",
        units="mbar",
        color="blue"
    )

    layout.add_symbol(
        "C",
        "cloud_coverage",
        sky_cover
    )

    station_plot = StationPlot(
        axes,
        plot_data["longitude"],
        plot_data["latitude"],
        clip_on=True,
        transform=PROJECTION,
        fontsize=8
    )

    layout.plot(
        station_plot,
        plot_data
    )

    # Add station ICAO labels below the station plots
    for _, row in dataframe.iterrows():

        axes.text(
            row["longitude"],
            row["latitude"] - 0.35,
            row["station"],
            transform=PROJECTION,
            fontsize=6,
            ha="center",
            va="top",
            color="black"
        )

    current_time = get_ist_time()

    axes.set_title(
        f"Indian METAR Station Observations\n"
        f"{current_time:%Y-%m-%d %H:%M} IST",
        fontsize=13
    )

    figure.savefig(
        output_file,
        dpi=220,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(figure)

    print(f"Saved: {output_file}")


# ============================================================
# Main program
# ============================================================

def main():
    print("Downloading METAR webpage...")

    try:
        html = download_page(URL)

    except requests.RequestException as error:
        print("Unable to download METAR webpage:")
        print(error)
        return

    print("Extracting METAR reports...")

    reports = extract_metar_records(html)

    if not reports:
        print("No METAR reports found.")
        return

    print(
        f"METAR reports found: {len(reports)}"
    )

    rows = []

    for report in reports:

        try:
            parsed = parse_metar(report)

            if parsed is not None:
                rows.append(parsed)

        except Exception as error:
            print("Error parsing report:")
            print(report)
            print(error)

    dataframe = pd.DataFrame(rows)

    if dataframe.empty:
        print("No valid observations decoded.")
        return

    dataframe = dataframe.drop_duplicates(
        subset=[
            "station",
            "date_time",
            "raw_metar"
        ]
    )

    dataframe = dataframe.sort_values(
        ["date_time", "station"],
        na_position="last"
    ).reset_index(drop=True)

    dataframe.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print(dataframe.to_string(index=False))
    print()
    print(f"CSV saved as: {OUTPUT_CSV}")

    # Station plot (RESTORED - no SE corner box)
    plot_station_observations(
        dataframe,
        "metar_station_observations.png"
    )

    # Pressure (ACTUAL VALUES + ICAO codes)
    plot_contour_field(
        dataframe,
        variable="pressure_hPa",
        title="Mean Sea-Level Pressure",
        colorbar_label="Pressure (hPa)",
        output_file="metar_pressure_contours.png",
        cmap="viridis",
        contour_levels=15,
        contour_format="%.0f"
    )

    # Temperature (ACTUAL VALUES + ICAO codes)
    plot_contour_field(
        dataframe,
        variable="temperature_C",
        title="Surface Temperature",
        colorbar_label="Temperature (°C)",
        output_file="metar_temperature_contours.png",
        cmap="RdYlBu_r",
        contour_levels=15,
        contour_format="%.0f"
    )

    # Dew-point (ACTUAL VALUES + ICAO codes)
    plot_contour_field(
        dataframe,
        variable="dew_point_C",
        title="Surface Dew-Point Temperature",
        colorbar_label="Dew point (°C)",
        output_file="metar_dewpoint_contours.png",
        cmap="YlGnBu",
        contour_levels=15,
        contour_format="%.0f"
    )

    # Wind-speed (ACTUAL VALUES + ICAO codes)
    plot_wind_map(
        dataframe,
        "metar_wind_speed_barbs.png"
    )

    # Visibility (NEW MAP + ICAO codes)
    plot_visibility_map(
        dataframe,
        "metar_visibility.png"
    )

    # Current-weather map
    plot_current_weather_map(
        dataframe,
        "metar_current_weather.png"
    )

    print()
    print("All maps have been generated successfully.")


if __name__ == "__main__":
    main()