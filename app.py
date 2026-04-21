import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from langdetect import detect, LangDetectException

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify EDA 2024",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS (Spotify dark theme) ──────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #121212; color: #FFFFFF; }
    .metric-card {
        background: #1A1A1A; border: 1px solid #1DB954;
        border-radius: 10px; padding: 20px; text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #1DB954; }
    .metric-label { font-size: 0.9rem; color: #B3B3B3; }
    h1, h2, h3 { color: #1DB954 !important; }
    .stSelectbox label, .stMultiSelect label,
    .stSlider label { color: #B3B3B3 !important; }
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("spotify_2024.csv", encoding="latin-1")
    stream_cols = [
        "Spotify Streams", "YouTube Views", "TikTok Views",
        "Pandora Streams", "Soundcloud Streams"
    ]
    for col in stream_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ""), errors="coerce"
            )
    if "Release Date" in df.columns:
        df["Release Date"] = pd.to_datetime(df["Release Date"], errors="coerce")
        df["Release Year"]  = df["Release Date"].dt.year
        df["Release Month"] = df["Release Date"].dt.month
        df["Release Day"]   = df["Release Date"].dt.day_name()
    return df

df = load_data()

# ── SIDEBAR FILTERS ──────────────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Spotify_logo_without_text.svg/168px-Spotify_logo_without_text.svg.png",
    width=60
)
st.sidebar.title("🎛️ Filters")

years = sorted(df["Release Year"].dropna().unique().astype(int).tolist())
selected_years = st.sidebar.slider(
    "Release Year", min_value=min(years), max_value=max(years),
    value=(2018, 2024)
)

all_artists = sorted(df["Artist"].dropna().unique().tolist())
selected_artists = st.sidebar.multiselect(
    "Filter by Artist (optional)", all_artists, default=[]
)

min_streams = st.sidebar.number_input(
    "Min Spotify Streams (M)", min_value=0, value=0, step=50
) * 1_000_000

# Apply filters
filtered = df[
    (df["Release Year"] >= selected_years[0]) &
    (df["Release Year"] <= selected_years[1]) &
    (df["Spotify Streams"] >= min_streams)
]
if selected_artists:
    filtered = filtered[filtered["Artist"].isin(selected_artists)]

st.sidebar.markdown(f"**{len(filtered):,}** songs shown")

# ── HEADER ───────────────────────────────────────────────────────────────────
st.title("🎵 Most Streamed Spotify Songs 2024")
st.markdown(
    "**Interactive EDA Dashboard** · Symbiosis Institute of Technology · "
    "Drishti Heda · Jayvee Shah · Jui Dixit · Krish Joshi"
)
st.divider()

# ── KPI CARDS ────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{len(filtered):,}</div>
        <div class="metric-label">Songs</div></div>""",
        unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{filtered['Artist'].nunique():,}</div>
        <div class="metric-label">Unique Artists</div></div>""",
        unsafe_allow_html=True)
with k3:
    total = filtered["Spotify Streams"].sum()
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{total/1e12:.2f}T</div>
        <div class="metric-label">Total Streams</div></div>""",
        unsafe_allow_html=True)
with k4:
    top = filtered.nlargest(1, "Spotify Streams")
    top_song = top["Track"].values[0] if len(top) else "N/A"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value" style="font-size:1rem">{top_song}</div>
        <div class="metric-label">Top Song</div></div>""",
        unsafe_allow_html=True)

st.divider()

# ── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Top Songs", "📊 Platform Battle",
    "📈 Era Timeline", "🌍 Language", "❤️ Loyalty Index"
])

# ── TAB 1: TOP SONGS ─────────────────────────────────────────────────────────
with tab1:
    st.subheader("Top Most Streamed Songs")
    n = st.slider("How many songs to show?", 5, 25, 10)
    top_songs = (
        filtered.nlargest(n, "Spotify Streams")[["Track", "Artist", "Spotify Streams"]]
        .reset_index(drop=True)
    )
    top_songs["Streams (B)"] = (top_songs["Spotify Streams"] / 1e9).round(2)

    fig, ax = plt.subplots(figsize=(10, n * 0.5 + 1))
    fig.patch.set_facecolor("#121212")
    ax.set_facecolor("#1A1A1A")
    bars = ax.barh(
        top_songs["Track"] + " — " + top_songs["Artist"],
        top_songs["Streams (B)"], color="#1DB954"
    )
    for bar, val in zip(bars, top_songs["Streams (B)"]):
        ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height()/2,
                f"{val}B", va="center", color="white", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Spotify Streams (Billions)", color="#B3B3B3")
    ax.tick_params(colors="#B3B3B3")
    ax.spines[:].set_color("#333333")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.dataframe(
        top_songs[["Track", "Artist", "Streams (B)"]],
        use_container_width=True, hide_index=True
    )

# ── TAB 2: PLATFORM BATTLE ───────────────────────────────────────────────────
with tab2:
    st.subheader("Platform Battle — Total Streams Comparison")
    platform_cols = {
        "Spotify": "Spotify Streams",
        "YouTube": "YouTube Views",
        "TikTok":  "TikTok Views",
        "Pandora": "Pandora Streams",
    }
    available = {k: v for k, v in platform_cols.items() if v in filtered.columns}
    totals = {k: filtered[v].sum() / 1e9 for k, v in available.items()}

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("#121212")
        ax.set_facecolor("#1A1A1A")
        colors = ["#1DB954", "#FF0000", "#69C9D0", "#0057FF"]
        bars = ax.bar(list(totals.keys()), list(totals.values()), color=colors[:len(totals)])
        for bar, val in zip(bars, totals.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f"{val:.0f}B", ha="center", color="white", fontsize=10)
        ax.set_ylabel("Streams / Views (B)", color="#B3B3B3")
        ax.tick_params(colors="#B3B3B3")
        ax.spines[:].set_color("#333333")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#121212")
        ax.pie(
            list(totals.values()), labels=list(totals.keys()),
            colors=colors[:len(totals)], autopct="%1.1f%%",
            textprops={"color": "white"}
        )
        ax.set_facecolor("#121212")
        fig.patch.set_facecolor("#121212")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ── TAB 3: ERA TIMELINE ──────────────────────────────────────────────────────
with tab3:
    st.subheader("Era Evolution — Platform Dominance Over Time")
    era_cols = {
        "Spotify": "Spotify Streams",
        "YouTube": "YouTube Views",
        "TikTok":  "TikTok Views",
    }
    era_avail = {k: v for k, v in era_cols.items() if v in filtered.columns}
    era_data = (
        filtered.groupby("Release Year")[list(era_avail.values())]
        .sum().div(1e9).reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#121212")
    ax.set_facecolor("#1A1A1A")
    colors_era = {"Spotify": "#1DB954", "YouTube": "#FF0000", "TikTok": "#69C9D0"}
    for name, col in era_avail.items():
        ax.plot(era_data["Release Year"], era_data[col],
                marker="o", label=name, color=colors_era[name], linewidth=2.5)
    ax.set_xlabel("Release Year", color="#B3B3B3")
    ax.set_ylabel("Total Streams (B)", color="#B3B3B3")
    ax.legend(facecolor="#1A1A1A", labelcolor="white")
    ax.tick_params(colors="#B3B3B3")
    ax.spines[:].set_color("#333333")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── TAB 4: LANGUAGE ──────────────────────────────────────────────────────────
with tab4:
    st.subheader("Language Distribution")
    if "Language Code" not in filtered.columns:
        st.info("Language column not found. Add a 'Language Code' column to your CSV for this chart.")
    else:
        lang_counts = filtered["Language Code"].value_counts().head(10)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        for ax in [ax1, ax2]:
            fig.patch.set_facecolor("#121212")
            ax.set_facecolor("#1A1A1A")
        ax1.pie(lang_counts.values, labels=lang_counts.index,
                autopct="%1.1f%%", textprops={"color": "white"},
                colors=sns.color_palette("Greens_r", len(lang_counts)))
        ax1.set_title("Track Share by Language", color="white")
        lang_streams = (
            filtered.groupby("Language Code")["Spotify Streams"]
            .sum().sort_values(ascending=False).head(10).div(1e9)
        )
        ax2.barh(lang_streams.index[::-1], lang_streams.values[::-1], color="#1DB954")
        ax2.set_xlabel("Total Streams (B)", color="#B3B3B3")
        ax2.set_title("Streams by Language", color="white")
        ax2.tick_params(colors="#B3B3B3")
        ax2.spines[:].set_color("#333333")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ── TAB 5: LOYALTY INDEX ─────────────────────────────────────────────────────
with tab5:
    st.subheader("Artist Loyalty Index")
    st.markdown("*Composite score = Spotify playlist adds ÷ streams + YouTube likes ÷ views + TikTok likes ÷ views*")

    from sklearn.preprocessing import MinMaxScaler

    req = ["Spotify Streams", "Spotify Playlist Count",
           "YouTube Views", "YouTube Likes",
           "TikTok Views", "TikTok Likes"]
    if all(c in filtered.columns for c in req):
        ldf = filtered.copy()
        for col in req:
            ldf[col] = pd.to_numeric(ldf[col], errors="coerce")
        ldf = ldf.dropna(subset=req)
        ldf["sp_ratio"] = ldf["Spotify Playlist Count"] / (ldf["Spotify Streams"] + 1)
        ldf["yt_ratio"] = ldf["YouTube Likes"]          / (ldf["YouTube Views"]   + 1)
        ldf["tt_ratio"] = ldf["TikTok Likes"]           / (ldf["TikTok Views"]    + 1)
        scaler = MinMaxScaler()
        ldf[["sp_n","yt_n","tt_n"]] = scaler.fit_transform(ldf[["sp_ratio","yt_ratio","tt_ratio"]])
        ldf["loyalty"] = 0.5*ldf["sp_n"] + 0.3*ldf["yt_n"] + 0.2*ldf["tt_n"]

        min_songs = st.slider("Min charting songs per artist", 1, 10, 3)
        artist_loyalty = (
            ldf.groupby("Artist")
            .filter(lambda x: len(x) >= min_songs)
            .groupby("Artist")["loyalty"].mean()
            .sort_values(ascending=False).head(15)
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#121212")
        ax.set_facecolor("#1A1A1A")
        ax.barh(artist_loyalty.index[::-1], artist_loyalty.values[::-1], color="#1DB954")
        ax.set_xlabel("Loyalty Index (0=low, 1=high)", color="#B3B3B3")
        ax.tick_params(colors="#B3B3B3")
        ax.spines[:].set_color("#333333")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Some required columns are missing from the filtered dataset.")
