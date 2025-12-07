# Manual Changelog - 2025-12-07

## Summary
Manual corrections for flavor normalization and text formatting improvements in Switzerland, Estonia, Latvia, and Poland.

## Changes Made

### 1. Switzerland (CH-DE)
**Issue**: Minor text formatting inconsistencies in Apricot Edition and Winter Edition descriptions.

**Corrections**:
- Apricot Edition: "fruity sweet" → "fruity-sweet" (proper hyphenation)
- Winter Edition: "ginger" → "Ginger" (capitalization consistency)

**Files Modified**: `data/processed/ch-de_processed.json`

### 2. Estonia (EE-ET)
**Issue**: Multiple flavor format inconsistencies and description text improvements.

**Corrections**:
- **Energy Drink**: "travelers" → "travellers" (British English), removed Oxford comma
- **Sugarfree**: Removed redundant article "a"
- **Winter Edition**:
  - Flavor: "Apple and Ginger" → "Fuji Apple & Ginger" (standardized format)
  - Description: "warmly spicy" → "warm spicy"
- **Yellow Edition**: Rephrased description for clarity
- **White Edition**:
  - Flavor: "Coconut and Blueberry" → "Coconut-Blueberry" (standardized format)
  - Description: "Blueberry" → "blueberries"
- **Green Edition**:
  - Flavor: "Dragon fruit" → "Dragon Fruit" (capitalization)
  - Description: Corrected to match flavor ("Dragon Fruit" instead of "Cactus Fruit")
- **Ice Edition**: Removed Oxford comma
- **Sea Blue Edition**: "Juneberry" → "juneberry" (lowercase in description)
- **Apricot Edition**:
  - Flavor: "Apricot and Strawberry" → "Apricot-Strawberry" (standardized format)

**Files Modified**: `data/processed/ee-et_processed.json`

### 3. Poland (PL-PL)
**Issue**: Winter Edition flavor format and description text improvements.

**Corrections**:
- **Energy Drink**: Updated description wording
- **Sugarfree**: Updated description phrasing
- **Winter Edition**:
  - Flavor: "Apple Fuji-Ginger" → "Fuji Apple & Ginger" (standardized format)
  - Description: Removed "The" from edition name reference
- **Winter Edition Sugarfree**: Same flavor standardization
- **Festive Edition**: Removed "The" from edition name reference, lowercase "pomegranate"
- **Ice Edition**: "Blueberry" → "berry" in description
- **Green Edition**: Removed "The" from edition name reference

**Files Modified**: `data/processed/pl-pl_processed.json`

### 4. Latvia (LV-LV)
**Issue**: Green Edition flavor mismatch - showed "Dragon fruit" but description references "Cactus Fruit".

**Corrections**:
- **Green Edition**:
  - Flavor: "Dragon fruit" → "Cactus Fruit" (pinned via corrections.json)
  - Reason: Description says "exotic taste of Cactus Fruit" - flavor must match

**Files Modified**: `data/processed/lv-lv_processed.json`

## Countries Affected
- 🇨🇭 **Switzerland (CH)**: Text formatting improvements
- 🇪🇪 **Estonia (EE)**: Flavor format standardization + text improvements
- 🇱🇻 **Latvia (LV)**: Green Edition flavor correction (Dragon fruit → Cactus Fruit)
- 🇵🇱 **Poland (PL)**: Winter Edition flavor standardization + text improvements

## Technical Details

### Flavor Format Standardization
The following flavor formats were standardized to match the project conventions:
- Combined flavors use hyphens: "Apricot-Strawberry" (not "Apricot and Strawberry")
- Special exceptions preserved: "Fuji Apple & Ginger" (keeps ampersand per conventions)
- Proper capitalization: "Dragon Fruit" (not "Dragon fruit")

### Text Improvements
- Consistent British English spelling (travellers)
- Removed redundant Oxford commas where not needed
- Standardized article usage
- Corrected flavor-description mismatches:
  - Estonia: Description updated to match flavor "Dragon Fruit"
  - Latvia: Flavor corrected to "Cactus Fruit" to match description (pinned via corrections.json)

## Files Changed
1. `data/processed/ch-de_processed.json` - Switzerland corrections
2. `data/processed/ee-et_processed.json` - Estonia corrections
3. `data/processed/lv-lv_processed.json` - Latvia corrections
4. `data/processed/pl-pl_processed.json` - Poland corrections
5. `data/redbull_editions_final.json` - Final consolidated output updated
6. `data/corrections.json` - Latvia Green Edition flavor pinned

## Notes
- All corrections align with the project's normalization rules in CLAUDE.md
- Flavor format standardization ensures consistency across all countries
- Description corrections improve translation quality and accuracy
