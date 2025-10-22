# Red Bull Editions Manual Corrections Changelog
## 2025-10-22T13:53:39

## Summary
- **Type:** Manual corrections and data cleanup
- **Countries affected:** 16
- **Global corrections added:** 2
- **Flavor standardizations:** 16
- **Backup files removed:** 2
- **Total files changed:** 23 (380 insertions, 642 deletions)

---

## 🔧 Global Corrections Added (corrections.json)

### New Global Correction 1
- **Edition UUID:** `f900c5b7-d33e-4a8e-a186-5cee5bd291a1`
- **Field:** flavor
- **Search:** "Strawberry and apricot"
- **Replace:** "Apricot-Strawberry"
- **Scope:** Global (applies to all countries with this edition)
- **Purpose:** Standardize flavor order in raw data before Gemini processing

### New Global Correction 2
- **Edition UUID:** `f900c5b7-d33e-4a8e-a186-5cee5bd291a1`
- **Field:** flavor
- **Search:** "Strawberry-Apricot"
- **Replace:** "Apricot-Strawberry"
- **Scope:** Global (applies to all countries with this edition)
- **Purpose:** Standardize flavor order after Gemini translation

---

## 🌍 Countries Updated

### Flavor Standardization: Strawberry-Apricot → Apricot-Strawberry

The following 16 countries had their Apricot/Amber/Summer Edition flavor standardized from "Strawberry-Apricot" to "Apricot-Strawberry":

#### 1. Austria (AT)
- **Edition:** The Apricot Edition
- **Change:** "Strawberry-Apricot" → "Apricot-Strawberry"
- **Additional changes:**
  - Winter Edition: Description text refinement ("fine" → "delicate")
  - Peach Edition: Description text refinement
  - Ice Edition: Description text refinement ("extraordinary" → "unusual")
  - White Edition: Description text refinement
  - Sea Blue Edition: Description text refinement

#### 2. Denmark (DK)
- **Edition:** The Apricot Edition
- **Change:** "Strawberry-Apricot" → "Apricot-Strawberry"
- **Additional changes:**
  - Winter Edition: Description refinement
  - Ice Edition: Description refinement ("touch" → "hint")
  - Pink Edition Sugarfree: Description refinement
  - Blue/Purple Edition: Text consistency improvements

#### 3. Estonia (EE)
- **Edition:** The Apricot Edition
- **Change:** "Strawberry-Apricot" → "Apricot-Strawberry"
- **Additional changes:**
  - Energy Drink: Description refinement
  - Sugarfree: Description refinement
  - Winter Edition: Description refinement
  - White Edition: Description refinement
  - Green Edition: Flavor correction ("Cactus Fruit" → "Dragon Fruit")
  - Ice Edition: Description refinement
  - Peach Edition: Flavor correction ("Peach" → "White Peach")

#### 4. Spain (ES)
- **Edition:** The Apricot Edition (already correct)
- **Additional changes:**
  - Zero Edition: Description capitalization
  - Peach Edition: **ADDED** to processed file
  - Lime Green Edition: Description refinement
  - Coconut Edition: Description refinement

#### 5. France (FR)
- **Edition:** The Apricot Edition
- **Change:** "Strawberry-Apricot" → "Apricot-Strawberry"
- **Additional changes:**
  - Various description refinements

#### 6. United Kingdom (GB)
- **Edition:** The Apricot Edition
- **Change:** "Strawberry-Apricot" → "Apricot-Strawberry"
- **Additional changes:**
  - Multiple description refinements

#### 7. Hungary (HU)
- **Edition:** The Apricot Edition
- **Change:** "Strawberry-Apricot" → "Apricot-Strawberry"
- **Additional changes:**
  - Description text improvements

#### 8. Italy (IT)
- **Edition:** The Apricot Edition
- **Change:** "Strawberry-Apricot" → "Apricot-Strawberry"
- **Additional changes:**
  - Various text refinements

#### 9. Latvia (LV)
- **Edition:** The Apricot Edition
- **Change:** "Strawberry-Apricot" → "Apricot-Strawberry"
- **Additional changes:**
  - Multiple description improvements

#### 10. Middle East (MEA)
- **Edition:** The Apricot Edition
- **Change:** "Strawberry-Apricot" → "Apricot-Strawberry"
- **Additional changes:**
  - Text consistency improvements

#### 11. North Macedonia (MK)
- **Edition:** The Apricot Edition
- **Change:** "Strawberry-Apricot" → "Apricot-Strawberry"
- **Additional changes:**
  - Description refinements

#### 12. Netherlands (NL)
- **Edition:** The Apricot Edition
- **Change:** "Strawberry-Apricot" → "Apricot-Strawberry"
- **Additional changes:**
  - Text improvements

#### 13. Norway (NO)
- **Edition:** The Apricot Edition
- **Change:** "Strawberry-Apricot" → "Apricot-Strawberry"
- **Additional changes:**
  - Various refinements

#### 14. Portugal (PT)
- **Edition:** The Apricot Edition (already correct)
- **Additional changes:**
  - Description text improvements

#### 15. Romania (RO)
- **Edition:** The Apricot Edition
- **Change:** "Strawberry-Apricot" → "Apricot-Strawberry"
- **Additional changes:**
  - Text consistency improvements

#### 16. Sweden (SE)
- **Edition:** The Apricot Edition
- **Change:** "Strawberry-Apricot" → "Apricot-Strawberry"
- **Additional changes:**
  - Description refinements

#### Additional countries with text improvements:
- **Slovenia (SI)**: Description refinements
- **Slovakia (SK)**: Text improvements
- **United States (US)**: Description refinements, Amber Edition flavor standardization

---

## 🗑️ Cleanup

### Backup Files Removed
The following duplicate/backup files were removed to clean up the repository:
1. `data/processed/pt-pt_backup_processed.json` (137 lines)
2. `data/processed/pt-pt_processed_backup.json` (137 lines)

**Reason:** These were duplicate backup files that are no longer needed.

---

## 📊 Statistics

- **Files changed:** 23
- **Insertions:** +380 lines
- **Deletions:** -642 lines (primarily from deleted backup files)
- **Net change:** -262 lines

### Changed Files Breakdown:
```
data/corrections.json                      |  12 ++
data/processed/at-de_processed.json        |  24 +--
data/processed/dk-da_processed.json        |  32 ++--
data/processed/ee-et_processed.json        |  42 +++---
data/processed/es-es_processed.json        |  32 ++--
data/processed/fr-fr_processed.json        |  24 +--
data/processed/gb-en_processed.json        |  28 ++--
data/processed/hu-hu_processed.json        |  28 ++--
data/processed/it-it_processed.json        |  18 +--
data/processed/lv-lv_processed.json        |  26 ++--
data/processed/mea-en_processed.json       |  10 +-
data/processed/mk-mk_processed.json        |  20 +--
data/processed/nl-nl_processed.json        |  20 +--
data/processed/no-no_processed.json        |  20 +--
data/processed/pt-pt_backup_processed.json | 137 -----------------
data/processed/pt-pt_processed.json        |  40 ++---
data/processed/pt-pt_processed_backup.json | 137 -----------------
data/processed/ro-ro_processed.json        |  20 +--
data/processed/se-en_processed.json        |  28 ++--
data/processed/si-sl_processed.json        |  28 ++--
data/processed/sk-sk_processed.json        |  40 ++---
data/processed/us-en_processed.json        |  22 +--
data/redbull_editions_final.json           | 234 ++++++++++++++---------------
```

---

## 🎯 Impact

### Immediate Effect
- All Apricot/Amber/Summer Edition entries now consistently use "Apricot-Strawberry" flavor
- Description texts improved for readability and consistency
- Repository cleaned up (removed unnecessary backup files)

### Future Effect
- **Global corrections** will automatically apply during every daily automated run
- Prevents future inconsistencies in flavor naming
- Reduces manual intervention needed for this specific edition

---

## ✅ Verification

To verify these changes:
```bash
# Check flavor consistency
grep -r "Strawberry-Apricot" data/redbull_editions_final.json

# Should return minimal results (only for Brazil which has special correction)

# Check corrections are in place
cat data/corrections.json | grep -A4 "f900c5b7-d33e-4a8e-a186-5cee5bd291a1"
```

---

## 📝 Notes

- These changes were made manually to standardize the Apricot Edition flavor naming
- The global corrections ensure this standardization persists through automated runs
- No functional changes to the processing pipeline
- All changes are backward compatible
