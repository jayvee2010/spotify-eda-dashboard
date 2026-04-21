import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import re

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify EDA 2024",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
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
    .stTabs [data-baseweb="tab"] { color: #B3B3B3; }
    .stTabs [aria-selected="true"] { color: #1DB954 !important; }
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("spotify_2024.csv", encoding="latin-1")

    # Clean all numeric stream/view columns
    num_cols = [
        "Spotify Streams", "YouTube Views", "TikTok Views",
        "Pandora Streams", "Soundcloud Streams",
        "Spotify Playlist Count", "YouTube Likes", "TikTok Likes",
        "Apple Music Playlist Count", "Shazam Counts"
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "").str.strip(),
                errors="coerce"
            )

    # Parse dates
    if "Release Date" in df.columns:
        df["Release Date"]  = pd.to_datetime(df["Release Date"], errors="coerce")
        df["Release Year"]  = df["Release Date"].dt.year
        df["Release Month"] = df["Release Date"].dt.month
        df["Release Day"]   = df["Release Date"].dt.day_name()

    # Flag collaborations using regex on track name
    if "Track" in df.columns:
        df["Is Collab"] = df["Track"].str.contains(
            r"\b(feat\.?|ft\.?|with|&)\b", flags=re.IGNORECASE, regex=True
        )

    return df

df = load_data()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Spotify_logo_without_text.svg/168px-Spotify_logo_without_text.svg.png",
    width=60
)
st.sidebar.title("🎛️ Filters")
st.sidebar.markdown("*Filters apply to all tabs*")

years = sorted(df["Release Year"].dropna().unique().astype(int).tolist())
selected_years = st.sidebar.slider(
    "Release Year Range",
    min_value=int(min(years)), max_value=int(max(years)),
    value=(2018, 2024)
)

all_artists = sorted(df["Artist"].dropna().unique().tolist())
selected_artists = st.sidebar.multiselect(
    "Filter by Artist (optional)", all_artists, default=[]
)

min_streams = st.sidebar.number_input(
    "Min Spotify Streams (Millions)", min_value=0, value=0, step=50
) * 1_000_000

# Apply global filters
filtered = df[
    (df["Release Year"] >= selected_years[0]) &
    (df["Release Year"] <= selected_years[1]) &
    (df["Spotify Streams"].fillna(0) >= min_streams)
].copy()
if selected_artists:
    filtered = filtered[filtered["Artist"].isin(selected_artists)]

st.sidebar.markdown(f"---\n**{len(filtered):,}** songs in current filter")

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("🎵 Most Streamed Spotify Songs 2024")
st.markdown(
    "**Interactive EDA Dashboard** · Symbiosis Institute of Technology, Pune  \n"
    "Drishti Heda · Jayvee Shah · Jui Dixit · Krish Joshi · *Faculty: Dr. Snehal Bhosale*"
)
st.divider()

# ── KPI CARDS ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
total_streams = filtered["Spotify Streams"].sum()
top_row = filtered.nlargest(1, "Spotify Streams")
top_song_name = top_row["Track"].values[0] if len(top_row) else "N/A"
top_song_artist = top_row["Artist"].values[0] if len(top_row) else ""

with k1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{len(filtered):,}</div>
        <div class="metric-label">Songs</div></div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{filtered['Artist'].nunique():,}</div>
        <div class="metric-label">Unique Artists</div></div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{total_streams/1e12:.2f}T</div>
        <div class="metric-label">Total Streams</div></div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value" style="font-size:1.1rem">{top_song_name}</div>
        <div class="metric-label">Top Song · {top_song_artist}</div></div>""",
        unsafe_allow_html=True)

st.divider()

# ── PLOT STYLE HELPER ─────────────────────────────────────────────────────────
def dark_fig(figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#121212")
    ax.set_facecolor("#1A1A1A")
    ax.tick_params(colors="#B3B3B3")
    ax.spines[:].set_color("#333333")
    return fig, ax

def dark_fig2(figsize=(12, 5)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor("#121212")
    for ax in [ax1, ax2]:
        ax.set_facecolor("#1A1A1A")
        ax.tick_params(colors="#B3B3B3")
        ax.spines[:].set_color("#333333")
    return fig, ax1, ax2

# ── TABS ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🏆 Top Songs",
    "📊 Platform Battle",
    "📈 Era Timeline",
    "🤝 Collaborations",
    "💿 Top Albums",
    "📅 Release Timing",
    "🔍 Shazam Discovery",
    "🎯 Artist Deep Dive",
    "🥊 Artist vs Artist",
    "📉 Stream Distribution",
])
(tab_top, tab_platform, tab_era, tab_collab,
 tab_albums, tab_timing, tab_shazam,
 tab_artist, tab_vs, tab_dist) = tabs

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TOP SONGS
# ══════════════════════════════════════════════════════════════════════════════
with tab_top:
    st.subheader("🏆 Top Most Streamed Songs")
    n = st.slider("Number of songs to show", 5, 30, 10, key="top_n")

    top_songs = (
        filtered.nlargest(n, "Spotify Streams")[["Track", "Artist", "Spotify Streams"]]
        .reset_index(drop=True)
    )
    top_songs["Streams (B)"] = (top_songs["Spotify Streams"] / 1e9).round(3)
    labels = top_songs["Track"] + "  —  " + top_songs["Artist"]

    fig, ax = dark_fig(figsize=(11, max(4, n * 0.45 + 1)))
    bars = ax.barh(labels[::-1], top_songs["Streams (B)"][::-1], color="#1DB954")
    for bar, val in zip(bars, top_songs["Streams (B)"][::-1]):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val}B", va="center", color="white", fontsize=9)
    ax.set_xlabel("Spotify Streams (Billions)", color="#B3B3B3")
    ax.set_xlim(0, top_songs["Streams (B)"].max() * 1.15)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.dataframe(
        top_songs[["Track", "Artist", "Streams (B)"]],
        use_container_width=True, hide_index=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PLATFORM BATTLE
# ══════════════════════════════════════════════════════════════════════════════
with tab_platform:
    st.subheader("📊 Platform Battle — Total Volume Comparison")

    platform_map = {
        "Spotify":     "Spotify Streams",
        "YouTube":     "YouTube Views",
        "TikTok":      "TikTok Views",
        "Pandora":     "Pandora Streams",
        "Soundcloud":  "Soundcloud Streams",
    }
    available = {k: v for k, v in platform_map.items() if v in filtered.columns}
    totals    = {k: filtered[v].sum() / 1e9 for k, v in available.items()}
    pal       = ["#1DB954", "#FF0000", "#69C9D0", "#0057FF", "#FF5500"]

    fig, ax1, ax2 = dark_fig2(figsize=(13, 5))

    bars = ax1.bar(list(totals.keys()), list(totals.values()),
                   color=pal[:len(totals)])
    for bar, val in zip(bars, totals.values()):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f"{val:.0f}B", ha="center", color="white", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Streams / Views (Billions)", color="#B3B3B3")
    ax1.set_title("Total by Platform", color="white")

    ax2.pie(list(totals.values()), labels=list(totals.keys()),
            colors=pal[:len(totals)], autopct="%1.1f%%",
            textprops={"color": "white"}, startangle=140)
    ax2.set_title("Market Share", color="white")
    fig.patch.set_facecolor("#121212")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Per-song averages
    st.markdown("#### Average per Song")
    avgs = {k: filtered[v].mean() / 1e6 for k, v in available.items()}
    cols = st.columns(len(avgs))
    for col, (name, avg) in zip(cols, avgs.items()):
        col.metric(name, f"{avg:.0f}M")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ERA TIMELINE
# ══════════════════════════════════════════════════════════════════════════════
with tab_era:
    st.subheader("📈 Era Evolution — Platform Dominance Over Time")

    era_map = {
        "Spotify":  "Spotify Streams",
        "YouTube":  "YouTube Views",
        "TikTok":   "TikTok Views",
    }
    era_avail = {k: v for k, v in era_map.items() if v in filtered.columns}
    era_data  = (
        filtered.groupby("Release Year")[list(era_avail.values())]
        .sum().div(1e9).reset_index()
    )
    era_colors = {"Spotify": "#1DB954", "YouTube": "#FF0000", "TikTok": "#69C9D0"}

    fig, ax = dark_fig(figsize=(11, 4.5))
    for name, col in era_avail.items():
        ax.plot(era_data["Release Year"], era_data[col],
                marker="o", label=name, color=era_colors[name],
                linewidth=2.5, markersize=6)

    # Era shading
    ax.axvspan(2018, 2019.5, alpha=0.08, color="#4472C4", label="_Pre-TikTok")
    ax.axvspan(2019.5, 2021,  alpha=0.08, color="#FF0000", label="_TikTok Rise")
    ax.axvspan(2021,  2024,   alpha=0.08, color="#1DB954", label="_Multi-Platform")

    ax.text(2018.2, ax.get_ylim()[1] * 0.92, "Pre-TikTok", color="#4472C4", fontsize=9)
    ax.text(2019.7, ax.get_ylim()[1] * 0.92, "TikTok Rise", color="#FF0000", fontsize=9)
    ax.text(2021.2, ax.get_ylim()[1] * 0.92, "Multi-Platform", color="#1DB954", fontsize=9)

    ax.set_xlabel("Release Year", color="#B3B3B3")
    ax.set_ylabel("Total Streams / Views (Billions)", color="#B3B3B3")
    ax.legend(facecolor="#1A1A1A", labelcolor="white")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.info("💡 TikTok's explosive growth from 2020 onwards confirms no single platform holds dominance after 2021.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — COLLABORATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab_collab:
    st.subheader("🤝 Biggest Collaborations")

    collabs = filtered[filtered["Is Collab"] == True].copy()
    n_collab = st.slider("Number of collabs to show", 5, 25, 15, key="collab_n")

    if len(collabs) == 0:
        st.warning("No collaboration tracks found in current filter.")
    else:
        top_collabs = collabs.nlargest(n_collab, "Spotify Streams")[
            ["Track", "Artist", "Spotify Streams"]
        ].reset_index(drop=True)
        top_collabs["Streams (B)"] = (top_collabs["Spotify Streams"] / 1e9).round(3)

        fig, ax = dark_fig(figsize=(11, max(4, n_collab * 0.42 + 1)))
        colors_c = plt.cm.Greens(
            np.linspace(0.4, 0.9, len(top_collabs))
        )[::-1]
        bars = ax.barh(
            top_collabs["Track"][::-1],
            top_collabs["Streams (B)"][::-1],
            color=colors_c
        )
        for bar, val in zip(bars, top_collabs["Streams (B)"][::-1]):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val}B", va="center", color="white", fontsize=9)
        ax.set_xlabel("Spotify Streams (Billions)", color="#B3B3B3")
        ax.set_xlim(0, top_collabs["Streams (B)"].max() * 1.15)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        total_c = len(collabs)
        pct_c   = total_c / len(filtered) * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Collabs Found", f"{total_c:,}")
        c2.metric("% of All Songs", f"{pct_c:.1f}%")
        c3.metric("Avg Streams (Collab)", f"{collabs['Spotify Streams'].mean()/1e6:.0f}M")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — TOP ALBUMS
# ══════════════════════════════════════════════════════════════════════════════
with tab_albums:
    st.subheader("💿 Top Albums by Total Spotify Streams")

    if "Album Name" not in filtered.columns:
        st.info("No 'Album Name' column found in your CSV.")
    else:
        n_alb = st.slider("Number of albums to show", 5, 25, 15, key="alb_n")
        album_streams = (
            filtered.groupby("Album Name")["Spotify Streams"]
            .sum().sort_values(ascending=False).head(n_alb).div(1e9)
        )

        fig, ax = dark_fig(figsize=(11, max(4, n_alb * 0.42 + 1)))
        colors_a = plt.cm.Greens(np.linspace(0.35, 0.9, len(album_streams)))[::-1]
        bars = ax.barh(album_streams.index[::-1], album_streams.values[::-1],
                       color=colors_a)
        for bar, val in zip(bars, album_streams.values[::-1]):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}B", va="center", color="white", fontsize=9)
        ax.set_xlabel("Total Spotify Streams (Billions)", color="#B3B3B3")
        ax.set_xlim(0, album_streams.max() * 1.18)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown(f"**{filtered['Album Name'].nunique():,}** unique albums analysed")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — RELEASE TIMING
# ══════════════════════════════════════════════════════════════════════════════
with tab_timing:
    st.subheader("📅 Release Timing Analysis")
    st.markdown("*Which days and months produce the most-streamed songs?*")

    day_order   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    day_avg = (
        filtered.groupby("Release Day")["Spotify Streams"]
        .mean().reindex(day_order).div(1e6)
    )
    month_count = (
        filtered.groupby("Release Month")["Track"]
        .count().reindex(range(1, 13))
    )
    month_count.index = month_names

    fig, ax1, ax2 = dark_fig2(figsize=(13, 4.5))

    bar_colors = ["#1DB954" if d == "Friday" else "#535353" for d in day_order]
    b1 = ax1.bar(day_order, day_avg.values, color=bar_colors)
    for bar, val in zip(b1, day_avg.values):
        if not np.isnan(val):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                     f"{val:.0f}M", ha="center", color="white", fontsize=8)
    ax1.set_ylabel("Avg Streams (Millions)", color="#B3B3B3")
    ax1.set_title("Avg Streams by Release Day", color="white")
    ax1.tick_params(axis="x", rotation=30)

    b2 = ax2.bar(month_names, month_count.values, color="#1DB954")
    ax2.set_ylabel("Number of Tracks Released", color="#B3B3B3")
    ax2.set_title("Tracks Released per Month", color="white")

    fig.patch.set_facecolor("#121212")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    fri_avg = filtered[filtered["Release Day"] == "Friday"]["Spotify Streams"].mean()
    all_avg = filtered["Spotify Streams"].mean()
    if fri_avg and all_avg:
        uplift = ((fri_avg - all_avg) / all_avg) * 100
        st.success(f"🎯 **Friday Effect:** Songs released on Friday average "
                   f"**{uplift:.0f}% more streams** than the overall average — "
                   f"driven by New Music Friday playlist adds.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — SHAZAM DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════
with tab_shazam:
    st.subheader("🔍 Shazam Discovery Rate")
    st.markdown("*High Shazam + High Spotify = organically discovered AND deeply streamed*")

    if "Shazam Counts" not in filtered.columns:
        st.info("No 'Shazam Counts' column found in your CSV.")
    else:
        sdf = filtered.dropna(subset=["Shazam Counts", "Spotify Streams"]).copy()
        sdf = sdf[(sdf["Shazam Counts"] > 0) & (sdf["Spotify Streams"] > 0)]

        # Cap at 95th percentile to remove extreme outliers for cleaner plot
        sp_cap  = sdf["Spotify Streams"].quantile(0.95)
        sh_cap  = sdf["Shazam Counts"].quantile(0.95)
        sdf     = sdf[(sdf["Spotify Streams"] <= sp_cap) &
                      (sdf["Shazam Counts"] <= sh_cap)]

        sp_med = sdf["Spotify Streams"].median()
        sh_med = sdf["Shazam Counts"].median()

        # Quadrant labels
        def quadrant(row):
            if row["Spotify Streams"] >= sp_med and row["Shazam Counts"] >= sh_med:
                return "Viral + Discovered"
            elif row["Spotify Streams"] < sp_med and row["Shazam Counts"] >= sh_med:
                return "Shazam Only"
            elif row["Spotify Streams"] >= sp_med and row["Shazam Counts"] < sh_med:
                return "Playlist Driven"
            else:
                return "Niche"

        sdf["Quadrant"] = sdf.apply(quadrant, axis=1)
        qcolors = {
            "Viral + Discovered": "#1DB954",
            "Shazam Only":        "#FF5500",
            "Playlist Driven":    "#4472C4",
            "Niche":              "#888888",
        }

        fig, ax = dark_fig(figsize=(11, 5))
        for qname, qcolor in qcolors.items():
            mask = sdf["Quadrant"] == qname
            ax.scatter(
                sdf.loc[mask, "Spotify Streams"] / 1e6,
                sdf.loc[mask, "Shazam Counts"]   / 1e3,
                c=qcolor, alpha=0.5, s=15, label=qname
            )

        # Median lines
        ax.axvline(sp_med / 1e6, color="#FFFFFF", linewidth=0.8,
                   linestyle="--", alpha=0.4)
        ax.axhline(sh_med / 1e3, color="#FFFFFF", linewidth=0.8,
                   linestyle="--", alpha=0.4)

        ax.set_xlabel("Spotify Streams (Millions)", color="#B3B3B3")
        ax.set_ylabel("Shazam Counts (Thousands)", color="#B3B3B3")
        ax.legend(facecolor="#1A1A1A", labelcolor="white", markerscale=2)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # Quadrant counts
        q_counts = sdf["Quadrant"].value_counts()
        cols = st.columns(4)
        for col, (qname, qcolor) in zip(cols, qcolors.items()):
            count = q_counts.get(qname, 0)
            col.markdown(
                f"<div style='background:#1A1A1A;border:1px solid {qcolor};"
                f"border-radius:8px;padding:12px;text-align:center'>"
                f"<div style='color:{qcolor};font-weight:bold;font-size:1.1rem'>{qname}</div>"
                f"<div style='color:white;font-size:1.4rem;font-weight:bold'>{count:,}</div>"
                f"<div style='color:#B3B3B3;font-size:0.8rem'>songs</div></div>",
                unsafe_allow_html=True
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — ARTIST DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
with tab_artist:
    st.subheader("🎯 Artist Deep Dive")
    st.markdown("*Pick any artist and explore all their songs across platforms*")

    # Build list of artists with most songs first
    artist_counts = filtered["Artist"].value_counts()
    artist_list   = artist_counts.index.tolist()

    selected = st.selectbox("Choose an artist", artist_list, key="deep_artist")
    adf = filtered[filtered["Artist"] == selected].sort_values(
        "Spotify Streams", ascending=False
    )

    if len(adf) == 0:
        st.warning("No songs found for this artist in the current filter.")
    else:
        # KPIs
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Songs in Dataset", len(adf))
        a2.metric("Total Streams",    f"{adf['Spotify Streams'].sum()/1e9:.2f}B")
        a3.metric("Avg Streams/Song", f"{adf['Spotify Streams'].mean()/1e6:.0f}M")
        best = adf.iloc[0]["Track"]
        a4.metric("Top Track", best[:25] + ("…" if len(best) > 25 else ""))

        st.divider()

        # Songs bar chart
        fig, ax = dark_fig(figsize=(11, max(3, len(adf) * 0.4 + 1)))
        vals = adf["Spotify Streams"].values / 1e6
        colors_d = ["#1DB954" if i == 0 else "#158A3E" for i in range(len(adf))]
        bars = ax.barh(adf["Track"].values[::-1], vals[::-1], color=colors_d[::-1])
        for bar, val in zip(bars, vals[::-1]):
            ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f}M", va="center", color="white", fontsize=9)
        ax.set_xlabel("Spotify Streams (Millions)", color="#B3B3B3")
        ax.set_title(f"All songs by {selected}", color="white")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # Cross-platform for this artist
        cross_cols = {
            "Spotify":  "Spotify Streams",
            "YouTube":  "YouTube Views",
            "TikTok":   "TikTok Views",
        }
        cross_avail = {k: v for k, v in cross_cols.items() if v in adf.columns}
        if len(cross_avail) > 1:
            st.markdown("#### Cross-Platform Performance")
            cross_totals = {k: adf[v].sum() / 1e9 for k, v in cross_avail.items()}
            cp_cols = st.columns(len(cross_totals))
            cp_colors = {"Spotify": "#1DB954", "YouTube": "#FF0000", "TikTok": "#69C9D0"}
            for col, (name, val) in zip(cp_cols, cross_totals.items()):
                col.markdown(
                    f"<div style='background:#1A1A1A;border:1px solid "
                    f"{cp_colors.get(name,'#1DB954')};border-radius:8px;"
                    f"padding:14px;text-align:center'>"
                    f"<div style='color:{cp_colors.get(name,'#1DB954')};"
                    f"font-size:1.5rem;font-weight:bold'>{val:.2f}B</div>"
                    f"<div style='color:#B3B3B3'>{name}</div></div>",
                    unsafe_allow_html=True
                )

        # Raw table
        with st.expander("📋 See all songs data"):
            show_cols = [c for c in ["Track", "Album Name", "Release Date",
                                      "Spotify Streams", "YouTube Views", "TikTok Views"]
                         if c in adf.columns]
            st.dataframe(adf[show_cols].reset_index(drop=True),
                         use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 — ARTIST VS ARTIST
# ══════════════════════════════════════════════════════════════════════════════
with tab_vs:
    st.subheader("🥊 Artist vs Artist")
    st.markdown("*Head-to-head comparison across all platforms*")

    col_a, col_b = st.columns(2)
    with col_a:
        artist_a = st.selectbox("Artist A", artist_list, index=0, key="vs_a")
    with col_b:
        default_b = artist_list[1] if len(artist_list) > 1 else artist_list[0]
        artist_b  = st.selectbox("Artist B", artist_list,
                                  index=artist_list.index(default_b), key="vs_b")

    adf_a = filtered[filtered["Artist"] == artist_a]
    adf_b = filtered[filtered["Artist"] == artist_b]

    vs_cols = {
        "Spotify Streams": "Spotify",
        "YouTube Views":   "YouTube",
        "TikTok Views":    "TikTok",
    }
    vs_avail = {v: k for k, v in vs_cols.items() if k in filtered.columns}

    if len(adf_a) == 0 or len(adf_b) == 0:
        st.warning("One of the artists has no songs in the current filter.")
    else:
        totals_a = {label: adf_a[col].sum() / 1e9
                    for col, label in vs_avail.items() if col in adf_a.columns}
        totals_b = {label: adf_b[col].sum() / 1e9
                    for col, label in vs_avail.items() if col in adf_b.columns}

        # Grouped bar chart
        platforms_vs = list(totals_a.keys())
        x = np.arange(len(platforms_vs))
        width = 0.35

        fig, ax = dark_fig(figsize=(9, 4.5))
        bars_a = ax.bar(x - width/2,
                        [totals_a.get(p, 0) for p in platforms_vs],
                        width, label=artist_a, color="#1DB954")
        bars_b = ax.bar(x + width/2,
                        [totals_b.get(p, 0) for p in platforms_vs],
                        width, label=artist_b, color="#FF5500")
        for bar in list(bars_a) + list(bars_b):
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.05,
                        f"{h:.1f}B", ha="center", color="white", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(platforms_vs, color="#B3B3B3")
        ax.set_ylabel("Total Streams / Views (Billions)", color="#B3B3B3")
        ax.legend(facecolor="#1A1A1A", labelcolor="white")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # Stats comparison table
        st.markdown("#### Head-to-Head Stats")
        stats = {
            "Songs in Dataset":  [len(adf_a), len(adf_b)],
            "Avg Streams/Song":  [f"{adf_a['Spotify Streams'].mean()/1e6:.0f}M",
                                   f"{adf_b['Spotify Streams'].mean()/1e6:.0f}M"],
            "Top Track":         [adf_a.nlargest(1,'Spotify Streams')['Track'].values[0][:30],
                                   adf_b.nlargest(1,'Spotify Streams')['Track'].values[0][:30]],
            "Total Spotify (B)": [f"{adf_a['Spotify Streams'].sum()/1e9:.2f}",
                                   f"{adf_b['Spotify Streams'].sum()/1e9:.2f}"],
        }
        compare_df = pd.DataFrame(stats, index=[artist_a, artist_b]).T
        st.dataframe(compare_df, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 10 — STREAM DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_dist:
    st.subheader("📉 Stream Distribution")
    st.markdown("*How are streams spread across all songs? Most songs are mid-tier — a few mega-hits skew everything.*")

    sdata = filtered["Spotify Streams"].dropna()
    sdata = sdata[sdata > 0]

    cap_pct = st.slider("Cap outliers at percentile", 80, 100, 95, key="dist_cap")
    cap_val = sdata.quantile(cap_pct / 100)
    sdata_c = sdata[sdata <= cap_val]

    fig, ax1, ax2 = dark_fig2(figsize=(13, 4.5))

    # Histogram
    ax1.hist(sdata_c / 1e6, bins=40, color="#1DB954", edgecolor="#121212", alpha=0.85)
    ax1.axvline(sdata_c.mean()   / 1e6, color="#FF5500", linewidth=2,
                linestyle="--", label=f"Mean: {sdata_c.mean()/1e6:.0f}M")
    ax1.axvline(sdata_c.median() / 1e6, color="#FFDD00", linewidth=2,
                linestyle="--", label=f"Median: {sdata_c.median()/1e6:.0f}M")
    ax1.set_xlabel("Spotify Streams (Millions)", color="#B3B3B3")
    ax1.set_ylabel("Number of Songs", color="#B3B3B3")
    ax1.set_title(f"Stream Distribution (capped at {cap_pct}th percentile)", color="white")
    ax1.legend(facecolor="#1A1A1A", labelcolor="white")

    # Box plot by year
    year_data = [
        filtered.loc[
            (filtered["Release Year"] == yr) & (filtered["Spotify Streams"] > 0),
            "Spotify Streams"
        ].dropna().values / 1e6
        for yr in range(int(selected_years[0]), int(selected_years[1]) + 1)
    ]
    year_labels = list(range(int(selected_years[0]), int(selected_years[1]) + 1))
    year_data   = [d for d in year_data if len(d) > 0]

    if year_data:
        bp = ax2.boxplot(year_data, labels=year_labels[:len(year_data)],
                         patch_artist=True,
                         medianprops={"color": "#1DB954", "linewidth": 2})
        for patch in bp["boxes"]:
            patch.set_facecolor("#1A3A2A")
            patch.set_edgecolor("#1DB954")
        for whisker in bp["whiskers"]:
            whisker.set_color("#B3B3B3")
        for cap in bp["caps"]:
            cap.set_color("#B3B3B3")
        for flier in bp["fliers"]:
            flier.set(marker="o", color="#535353", alpha=0.3, markersize=3)
        ax2.set_xlabel("Release Year", color="#B3B3B3")
        ax2.set_ylabel("Streams (Millions)", color="#B3B3B3")
        ax2.set_title("Stream Spread by Year", color="white")
        ax2.tick_params(axis="x", rotation=45)

    fig.patch.set_facecolor("#121212")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Stats row
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Mean Streams",   f"{sdata.mean()/1e6:.0f}M")
    s2.metric("Median Streams", f"{sdata.median()/1e6:.0f}M")
    s3.metric("Top 1% Threshold", f"{sdata.quantile(0.99)/1e6:.0f}M")
    pct_under = (sdata < 100e6).sum() / len(sdata) * 100
    s4.metric("Songs Under 100M", f"{pct_under:.0f}%")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#535353;font-size:0.85rem'>"
    "Spotify EDA 2024 · Symbiosis Institute of Technology, Pune · "
    "Drishti Heda · Jayvee Shah · Jui Dixit · Krish Joshi"
    "</div>",
    unsafe_allow_html=True
)
