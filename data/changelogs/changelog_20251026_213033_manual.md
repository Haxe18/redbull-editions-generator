# Manual Changelog - 2025-10-26

## Summary
Fixed incorrect flavor and edition name translations for multiple countries through manual corrections and improved Gemini AI prompts.

## Changes Made

### 1. Peach Edition Corrections (Estonia & Norway)
**Issue**: Global correction was converting all "Peach" → "White Peach" for UUID `c55e5804-ce3c-4289-8d89-930b6d678501`, but Estonia and Norway's raw API data shows only "Peach" (not "White Peach").

**Solution**: Added country-specific reverse corrections in `data/corrections.json`:
```json
{
    "id": "c55e5804-ce3c-4289-8d89-930b6d678501:et-EE",
    "field": "flavor",
    "search": "White Peach",
    "replace": "Peach"
},
{
    "id": "c55e5804-ce3c-4289-8d89-930b6d678501:no-NO",
    "field": "flavor",
    "search": "White Peach",
    "replace": "Peach"
}
```

**Result**:
- ✅ Estonia (EE): Flavor = "Peach" (corrected)
- ✅ Norway (NO): Flavor = "Peach" (corrected)
- ✅ 9 other countries: Flavor = "White Peach" (unchanged - AT, BE, CA, DE, FI, GB, IE, ES, TR)

**Files Modified**: `data/corrections.json`, `data/redbull_editions_final.json`

### 2. Winter Edition Fix (Norway)
**Issue**: Norway's Winter Edition was showing incorrect flavor "Apricot-Strawberry" instead of "Apple & Ginger" (raw data: "Eple og ingefær").

**Root Cause**: Cache corruption - the Winter Edition was using cached data from a different edition.

**Solution**: Force reprocessed Norway with `--force` flag to clear cache and regenerate translations.

**Result**:
- ✅ Norway Winter Edition: Flavor = "Apple & Ginger" (corrected from "Apricot-Strawberry")

**Files Modified**: `data/processed/no-no_processed.json`, `data/redbull_editions_final.json`

### 3. Blue Edition Fix (Sweden)
**Issue**: Sweden's "Blue Edition" was being changed to "Blueberry Edition" by Gemini AI, despite raw data clearly showing "Blue Edition".

**Root Cause**: Gemini AI was deriving edition names from flavor values (Flavor = "Blueberry" → Name = "Blueberry Edition"), ignoring existing prompt instructions.

**Solution**: Strengthened Gemini prompt instructions in `processor.py` (lines 1770-1780) with explicit NEVER rules:
```
CRITICAL EDITION NAME RULES - DO NOT MODIFY EDITION NAMES:
- NEVER change "Blue Edition" to "Blueberry Edition" even if flavor is Blueberry
- NEVER change "Pink Edition" to another name even if flavor is Forest Berry or Raspberry
- NEVER change "White Edition" to another name even if flavor is Coconut
- NEVER change "Green Edition" to another name even if flavor is Dragon Fruit
- NEVER derive the edition name from the flavor field
- DO NOT translate or localize edition color names (Blue stays Blue, not Blueberry)
```

**Result**:
- ✅ Sweden: Edition name = "The Blue Edition" (corrected from "The Blueberry Edition")
- ✅ Flavor remains "Blueberry" (correct)

**Files Modified**: `processor.py`, `data/processed/se-en_processed.json`, `data/redbull_editions_final.json`

### 4. Workflow Enhancement
**Issue**: GitHub Actions workflow was not triggering when `data/corrections.json` was modified.

**Solution**: Added `data/corrections.json` to the workflow trigger paths in `.github/workflows/update-editions.yml`.

**Result**: CI/CD pipeline now automatically runs when manual corrections are added.

**Files Modified**: `.github/workflows/update-editions.yml`

## Countries Affected
- 🇪🇪 **Estonia (EE)**: Peach Edition flavor corrected
- 🇳🇴 **Norway (NO)**: Peach Edition flavor + Winter Edition flavor corrected
- 🇸🇪 **Sweden (SE)**: Blue Edition name corrected

## Additional Processed Files Updated
- `data/processed/it-it_processed.json` (Italy)
- `data/processed/pt-pt_processed.json` (Portugal)

## Technical Details

### Correction Strategy: Workaround with Sequential Replacements
For the Peach Edition, we used a "replacement chain" approach:
1. **Global correction** (line 124-128): All countries get `"Peach" → "White Peach"`
2. **Country-specific reverse corrections** (line 129-140): Only EE and NO get `"White Peach" → "Peach"`

This ensures the majority of countries get "White Peach" (correct for their API data) while Estonia and Norway preserve the original "Peach" flavor.

### Testing Performed
- ✅ Estonia force reprocessed and validated
- ✅ Norway force reprocessed and validated
- ✅ Sweden force reprocessed and validated
- ✅ All Blue Edition editions across countries checked for name consistency

## Files Changed
1. `data/corrections.json` - Added 2 new corrections (Peach Edition for EE/NO)
2. `processor.py` - Strengthened Gemini prompt instructions (lines 1770-1780)
3. `.github/workflows/update-editions.yml` - Added corrections.json trigger
4. `data/redbull_editions_final.json` - Updated with corrected data
5. `data/processed/*.json` - Multiple processed files updated after reprocessing

## Notes
- The strengthened Gemini prompts will prevent future edition name errors
- Manual corrections in `corrections.json` are now version-controlled and trigger CI/CD
- Cache issues can be resolved with `--force` flag during reprocessing
- All changes validated against raw API data to ensure accuracy
