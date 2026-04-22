# How to Fetch Scripts for ThriveSight Demo

## Scripts Needed

1. **Spider-Man (2002)** — David Koepp screenplay
2. **The Pursuit of Happyness (2006)** — Steve Conrad screenplay

## Where to Get Them

### Spider-Man (2002)
- IMSDB: https://imsdb.com/scripts/Spider-Man.html
- David Koepp's archive: https://davidkoepp.com/script-archive/spider-man/
- Daily Script: https://www.dailyscript.com/scripts/spider_man_koepp.html

### The Pursuit of Happyness (2006)
- IMSDB: https://imsdb.com/scripts/The-Pursuit-of-Happyness.html
- Script Slug: https://www.scriptslug.com/script/the-pursuit-of-happyness-2006

## Steps

1. Open any of the links above in your browser
2. Copy the full screenplay text (Ctrl+A, Ctrl+C on the script page)
3. Save as plain text files in this directory:
   - `scripts/spider_man_2002.txt`
   - `scripts/pursuit_of_happyness_2006.txt`

## Running the Analysis

### Parse only (preview scenes, no LLM calls):
```bash
cd apps/api
python manage.py analyze_screenplay \
    --file ../../scripts/spider_man_2002.txt \
    --title "Spider-Man" \
    --known-chars "PETER,MARY JANE,HARRY,NORMAN,FLASH,AUNT MAY,UNCLE BEN,GREEN GOBLIN" \
    --parse-only
```

### Full multi-lens analysis:
```bash
cd apps/api
python manage.py analyze_screenplay \
    --file ../../scripts/spider_man_2002.txt \
    --title "Spider-Man" \
    --character "PETER" \
    --description "A shy, intelligent high school student who gains spider powers after being bitten by a genetically modified spider. Struggles with identity, responsibility, and the tension between his ordinary life and his role as Spider-Man." \
    --known-chars "PETER,MARY JANE,HARRY,NORMAN,FLASH,AUNT MAY,UNCLE BEN,GREEN GOBLIN" \
    --output ../../datasets/peter_parker.json
```

```bash
cd apps/api
python manage.py analyze_screenplay \
    --file ../../scripts/pursuit_of_happyness_2006.txt \
    --title "The Pursuit of Happyness" \
    --character "CHRIS" \
    --description "A struggling salesman and single father who refuses to give up despite homelessness, financial ruin, and systemic obstacles. Driven by love for his son and an unshakeable belief that he can build a better life." \
    --known-chars "CHRIS,CHRISTOPHER,LINDA,JAY TWISTLE,MARTIN FROHM,WALTER RIBBON" \
    --output ../../datasets/chris_gardner.json
```

## What the Analysis Produces

The multi-lens analyzer applies 5 analytical lenses to every scene:

1. **Dialogue Signals** — Turn-by-turn emotional dynamics (emotion, intensity, reaction, trigger)
2. **Plot Analysis** — Narrative arc position, stakes, power dynamics, scene function
3. **Character Psychology** — Motivations, defense mechanisms, identity conflicts, attachment
4. **Relational Dynamics** — How characters relate, trust evolution, power balance shifts
5. **Contextual Metadata** — Witnesses, build-up patterns, environmental stressors, subtext

The output JSON contains the full character dataset ready for import into ThriveSight.
