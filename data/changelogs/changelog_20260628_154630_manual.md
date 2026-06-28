# Red Bull Editions Processing Changelog
## 2026-06-28T15:46:30.000000

## Summary
- **Trigger:** Manual corrections + targeted `--force` reprocessing of affected countries
- **Countries processed:** 10
- **Editions updated:** 10
- **Editions added:** 0
- **Editions removed:** 0
- **Corrections changed:** 3 flavor rules in `data/corrections.json`

## Corrections Changes (`data/corrections.json`)

### The White Edition — `Coconut` → `Coconut-Blueberry` (now global)
Six per-locale corrections (`de-AT`, `de-DE`, `de-CH`, `en-US`, `en-GB`, `fr-FR`) consolidated into **one global** correction on UUID `1ad9d76b-02ee-462b-a766-13ecb8ac24da` (no locale suffix), so the rule now applies to **all** countries — including Italy and Türkiye, which previously slipped through.

### The Ice Edition — `Iced Gummy Bear` → `Iced Vanilla Berry` (now global)
The `en-GB` and `pl-PL` per-locale corrections consolidated into **one global** correction on UUID `41e97cb8-fb1f-41a3-a380-be2a75123e1a`. Now applies to AT, CH, CZ, DK, EE, SK and Canada's regular Ice Edition. The localized entries `fi-FI` (`Mehujää`) and `fr-FR` (`Mûre givrée et vanille`) are retained, since their raw flavors are not matched by the English search string.

### The Summer Edition — `Curuba` → `Sudachi Lime` (Spain, pre-existing change)
Pre-existing uncommitted correction for Spain (`add1ea51-ee5f-4c6c-a2d2-efd59791e8f8:es-ES`), now applied via reprocessing.

### Fix: invalid trailing comma
Removed an invalid trailing comma before the closing `]` in the working copy of `corrections.json` (would have broken the processor's strict `json.load`). File now validates cleanly (38 corrections).

## Countries Processed

### Austria
#### Updated Editions:
- **The Ice Edition** - Iced Vanilla Berry (was Iced Gummy Bear)

### Canada
#### Updated Editions:
- **The Ice Edition** - Iced Vanilla Berry (was Iced Gummy Bear)

_Note: `The Ice Edition Sugarfree` (separate UUID `5380482f-630c-48b1-b366-38b78582b57b`) was already Iced Vanilla Berry and stays a distinct edition — no duplicate created._

### Czech Republic
#### Updated Editions:
- **The Ice Edition** - Iced Vanilla Berry (was Iced Gummy Bear)

### Denmark
#### Updated Editions:
- **The Ice Edition** - Iced Vanilla Berry (was Iced Gummy Bear)

### Estonia
#### Updated Editions:
- **The Ice Edition** - Iced Vanilla Berry (was Iced Gummy Bear)

### Italy
#### Updated Editions:
- **The White Edition** - Coconut-Blueberry (was Coconut)

### Slovakia
#### Updated Editions:
- **The Ice Edition** - Iced Vanilla Berry (was Iced Gummy Bear)

### Spain
#### Updated Editions:
- **The Summer Edition** - Sudachi Lime (was Curuba-Elderflower)

### Switzerland
#### Updated Editions:
- **The Ice Edition** - Iced Vanilla Berry (was Iced Gummy Bear)

### Türkiye
#### Updated Editions:
- **The White Edition** - Coconut-Blueberry (was Coconut)

## Verification
- Global sweep of `data/redbull_editions_final.json` confirms **no** remaining `Iced Gummy Bear` flavor and **no** White Edition stuck on bare `Coconut`.
- `python3 -c "import json; json.load(open('data/corrections.json'))"` → valid (38 corrections).
