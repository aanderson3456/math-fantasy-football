import json
import wikipedia
import re
import time
import os

with open('coaches_list.json', 'r') as f:
    coaches = json.load(f)

coach_birth_years = {}
if os.path.exists('coach_birth_years.json'):
    with open('coach_birth_years.json', 'r') as f:
        coach_birth_years = json.load(f)

def get_birth_year(coach_name):
    try:
        # Avoid ambiguities by appending 'American football'
        search_term = f"{coach_name} (American football)"
        page = wikipedia.page(search_term, auto_suggest=False)
        content = page.html() if hasattr(page, 'html') else page.content
        summary = page.summary
        match = re.search(r'\(born [^)]*(19\d{2}|20\d{2})[^)]*\)', summary)
        if match:
            return int(match.group(1))
        match2 = re.search(r'born.*?(19\d{2}|20\d{2})', summary[:300])
        if match2:
            return int(match2.group(1))
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            page = wikipedia.page(e.options[0])
            match = re.search(r'\(born [^)]*(19\d{2}|20\d{2})[^)]*\)', page.summary)
            if match:
                return int(match.group(1))
        except:
            pass
    except wikipedia.exceptions.PageError:
        pass
    except Exception as e:
        pass
    return None

for i, coach in enumerate(coaches):
    # Only skip if we already successfully found a number
    if coach_birth_years.get(coach) is not None and isinstance(coach_birth_years.get(coach), int):
        continue
    by = get_birth_year(coach)
    
    # If still None, maybe try without " (American football)"
    if by is None:
        try:
            page = wikipedia.page(coach, auto_suggest=False)
            match = re.search(r'\(born [^)]*(19\d{2}|20\d{2})[^)]*\)', page.summary)
            if match:
                by = int(match.group(1))
        except:
            pass

    coach_birth_years[coach] = by
    print(f"[{i+1}/{len(coaches)}] {coach}: {by}")
    time.sleep(0.1)

with open('coach_birth_years.json', 'w') as f:
    json.dump(coach_birth_years, f, indent=4)
