import sys, os
from bs4 import BeautifulSoup
sys.path.append('.')
from src.yahoo_reader import YahooFantasyReader

with open('cookie.txt', 'r') as f:
    cookie = f.read().strip()

reader = YahooFantasyReader(league_id='772025', cookie_string=cookie)
html = reader.fetch_url(reader.get_league_base_url() + "?module=matchups")
with open("scoreboard.html", "w") as f:
    f.write(html or "")
    
print("Saved scoreboard.html")
