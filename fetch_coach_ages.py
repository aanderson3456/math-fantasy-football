import json
import urllib.request
import urllib.parse
import re
import time
import os

with open('coaches_list.json', 'r') as f:
    coaches = json.load(f)

coach_birth_years = {}

# Load existing if available to resume
if os.path.exists('coach_birth_years.json'):
    with open('coach_birth_years.json', 'r') as f:
        coach_birth_years = json.load(f)

def get_birth_year(coach_name):
    # Try searching Wikipedia
    search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(coach_name + ' American football')}&utf8=&format=json"
    try:
        req = urllib.request.Request(search_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read())
            search_results = data.get('query', {}).get('search', [])
            if not search_results:
                return None
            title = search_results[0]['title']
            
            # Now fetch page content
            page_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=revisions&rvprop=content&rvslots=main&titles={urllib.parse.quote(title)}&format=json"
            req2 = urllib.request.Request(page_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req2) as response2:
                page_data = json.loads(response2.read())
                pages = page_data.get('query', {}).get('pages', {})
                for page_id, page_info in pages.items():
                    if 'revisions' in page_info:
                        content = page_info['revisions'][0]['slots']['main']['*']
                        # Look for {{Birth date and age|1983|1|24}} or {{birth date|1950|10|31}}
                        match = re.search(r'{{[Bb]irth date(?: and age)?\|(\d{4})\|', content)
                        if match:
                            return int(match.group(1))
                        # Look for | birth_date = [[January 1]], [[1960]]
                        match = re.search(r'birth_date\s*=\s*.*?(19\d{2}|20\d{2})', content)
                        if match:
                            return int(match.group(1))
    except Exception as e:
        print(f"Error for {coach_name}: {e}")
    return None

print(f"Total coaches: {len(coaches)}")
for i, coach in enumerate(coaches):
    if coach in coach_birth_years:
        continue
    by = get_birth_year(coach)
    coach_birth_years[coach] = by
    print(f"[{i+1}/{len(coaches)}] {coach}: {by}")
    time.sleep(0.1) # Be nice to Wikipedia
    
    if i % 10 == 0:
        with open('coach_birth_years.json', 'w') as f:
            json.dump(coach_birth_years, f, indent=4)

with open('coach_birth_years.json', 'w') as f:
    json.dump(coach_birth_years, f, indent=4)

print("Done generating coach birth years.")
