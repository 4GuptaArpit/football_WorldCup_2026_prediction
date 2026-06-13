import streamlit as st


FLAG_EMOJIS = {
    "Argentina": "🇦🇷",
    "Australia": "🇦🇺",
    "Austria": "🇦🇹",
    "Belgium": "🇧🇪",
    "Brazil": "🇧🇷",
    "Canada": "🇨🇦",
    "Colombia": "🇨🇴",
    "Croatia": "🇭🇷",
    "Curaçao": "🇨🇼",
    "Czech Republic": "🇨🇿",
    "DR Congo": "🇨🇩",
    "Ecuador": "🇪🇨",
    "Egypt": "🇪🇬",
    "England": "🏴",
    "France": "🇫🇷",
    "Germany": "🇩🇪",
    "Haiti": "🇭🇹",
    "Iran": "🇮🇷",
    "Iraq": "🇮🇶",
    "Italy": "🇮🇹",
    "Japan": "🇯🇵",
    "Mexico": "🇲🇽",
    "Morocco": "🇲🇦",
    "Netherlands": "🇳🇱",
    "Norway": "🇳🇴",
    "Panama": "🇵🇦",
    "Paraguay": "🇵🇾",
    "Portugal": "🇵🇹",
    "Qatar": "🇶🇦",
    "Saudi Arabia": "🇸🇦",
    "Scotland": "🏴",
    "Senegal": "🇸🇳",
    "South Africa": "🇿🇦",
    "South Korea": "🇰🇷",
    "Spain": "🇪🇸",
    "Switzerland": "🇨🇭",
    "Tunisia": "🇹🇳",
    "Turkey": "🇹🇷",
    "Ukraine": "🇺🇦",
    "United States": "🇺🇸",
    "Uruguay": "🇺🇾",
    "Uzbekistan": "🇺🇿",
}


def flag(team):
    """Return flag emoji for a team."""
    return FLAG_EMOJIS.get(team, "🏳️")


def team_label(team):
    """Return team name with flag."""
    return f"{flag(team)} {team}"


def apply_global_styles():
    """Apply premium World Cup style CSS with glassmorphism and modern colors."""

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
        
        .stApp {
            background: linear-gradient(135deg, #07111f 0%, #0d1e36 45%, #050d18 100%);
            color: #ffffff;
            font-family: 'Outfit', -apple-system, sans-serif;
        }

        .main-title {
            font-size: 46px;
            font-weight: 800;
            background: linear-gradient(45deg, #ff4d4d, #fca311);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 2px;
            letter-spacing: -1px;
        }

        .sub-title {
            font-size: 19px;
            color: #b0c4de;
            margin-bottom: 30px;
            font-weight: 300;
        }

        /* Glassmorphism Cards */
        .metric-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 18px;
            padding: 22px;
            margin-bottom: 18px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }
        .metric-card:hover {
            border: 1px solid rgba(255, 255, 255, 0.2);
            transform: translateY(-2px);
        }

        .winner-card {
            background: linear-gradient(135deg, #ffd700, #fca311);
            color: #06120f;
            border-radius: 24px;
            padding: 32px;
            text-align: center;
            font-size: 30px;
            font-weight: 800;
            margin: 25px 0;
            box-shadow: 0 10px 40px rgba(252, 163, 17, 0.3);
            border: 1px solid rgba(255, 215, 0, 0.3);
        }

        .team-card {
            background: rgba(255, 255, 255, 0.04);
            backdrop-filter: blur(5px);
            border-radius: 16px;
            padding: 16px 20px;
            margin-bottom: 10px;
            border-left: 6px solid #ff4d4d;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            border-right: 1px solid rgba(255, 255, 255, 0.05);
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            transition: all 0.2s ease;
        }
        .team-card:hover {
            background: rgba(255, 255, 255, 0.07);
            transform: scale(1.01);
        }

        .round-title {
            font-size: 26px;
            font-weight: 700;
            margin-top: 28px;
            margin-bottom: 16px;
            color: #fca311;
            border-left: 4px solid #ff4d4d;
            padding-left: 12px;
        }

        .small-muted {
            color: #a0aec0;
            font-size: 14px;
        }

        /* Customize Streamlit Buttons */
        div.stButton > button:first-child {
            background: linear-gradient(135deg, #ff4d4d, #d62828);
            color: white;
            font-size: 16px;
            font-weight: 600;
            border-radius: 12px;
            border: none;
            padding: 12px 24px;
            box-shadow: 0 4px 15px rgba(214, 40, 40, 0.4);
            transition: all 0.3s ease;
        }
        div.stButton > button:first-child:hover {
            background: linear-gradient(135deg, #ff6666, #ff4d4d);
            box-shadow: 0 6px 20px rgba(214, 40, 40, 0.6);
            transform: translateY(-2px);
        }
        div.stButton > button:first-child:active {
            transform: translateY(1px);
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #050d18;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )



def format_probability(value):
    """Format a probability as a percent string."""
    return f"{value * 100:.1f}%"