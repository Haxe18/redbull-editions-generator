# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Red Bull Editions data collection and processing system that fetches product information from Red Bull's public APIs and normalizes it using Google Gemini AI. The system collects data from 68+ unique countries (with multi-language support) and produces standardized JSON output.

## Architecture

The project follows a pipeline architecture:
1. **collector.py** - Production-ready collector (Pylint 9.92/10) with multi-language support
   - **Clean Architecture**: Dataclasses, NamedTuples, separation of concerns
   - **HTTP Optimization**: Connection pooling with retry strategy (exponential backoff)
   - **Multi-Language**: Merges editions from multiple language versions per country
   - **Language Priority**: English > German/Dutch > Others
   - **Error Handling**: Custom exception hierarchy (CollectorError, APIError, DataProcessingError)
   - **Configuration**: Externalized frozen dataclass for all constants
2. **processor.py** - Processes raw data with AI normalization
   - **Independent Operation**: Works directly with raw files
   - **Single Country Updates**: Directly updates final JSON by default
   - **Alphabetical Sorting**: Countries and editions sorted alphabetically by name for stable diffs
   - **Optimized AI Prompts**: Reduced from ~300 to ~100 lines for faster processing
   - **Intelligent Flavor Matching**: Maps variations like "Grapefruit-Woodruff" → "Woodruff & Pink Grapefruit"
3. **GitHub Actions** - Automates daily collection and processing

## Commands

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Add GEMINI_API_KEY to .env file
```

### Data Collection
```bash
# Test mode (3 countries)
python collector.py --test

# Small batch (10 countries)  
python collector.py --small

# All countries
python collector.py

# Specific number of countries
python collector.py 5

# Development mode (no rate limiting)
python collector.py --no-rate-limit

# Specific country with custom URL (requires --country option)
python collector.py --country Netherlands --custom-url "https://web.archive.org/web/20240827090215/https://www.redbull.com/v3/api/custom/header/v2?locale=nl-nl"
# Note: --custom-url only works with --country option, will error if used alone
```

### Data Processing
```bash
# Basic processing
python processor.py

# Force reprocess all (ignore cache)
python processor.py --force

# Verbose mode with detailed console output
python processor.py --verbose

# Debug mode with API logging to files
python processor.py --debug

# Process single country (updates final JSON)
python processor.py --country Germany

# Process single country to separate file (legacy behavior)
python processor.py --country Germany --separate-file

# Combine options
python processor.py --force --verbose --country Spain
```

### Testing & Quality
```bash
# Python linting
pylint collector.py      # Score: 9.92/10
pylint processor.py      # Check processor score
flake8 collector.py      # PEP 8 compliance

# Type checking
mypy collector.py --strict

# Syntax validation
python -m py_compile collector.py processor.py
```

## Key Features

### Production-Ready Code (collector.py refactored 2025)
- **Pylint Score**: 9.92/10 (exceptional code quality)
- **Architecture Improvements**:
  - Config externalized as frozen dataclass
  - EditionProcessingContext NamedTuple for cleaner method signatures
  - Helper methods extracted to reduce complexity (max 15 lines per method)
- **HTTP Session Management**:
  - Connection pooling with HTTPAdapter
  - Automatic retry with exponential backoff (3 retries per HTTP request)
  - Configurable retry parameters (502, 503, 504 status codes)
- **Country-Level Retry Logic** (NEW):
  - **Automatic Retry**: 1 retry per country (2 attempts total)
  - **Retry Delay**: 5 seconds between country-level retries
  - **Fail-Fast**: Stops immediately on final failure with Exit Code 1
  - **No Partial Results**: Aborts collection entirely if any country fails after retries
  - **Detailed Error Messages**: Shows exact failure reason in console output
- **Error Handling**:
  - Custom exception hierarchy (CollectorError, APIError, DataProcessingError)
  - Multi-level retry strategy (HTTP + Country level)
  - Fail-fast behavior for production/CI pipelines
  - Structured error reporting with Exit Code 1
- **Logging**:
  - Consistent use of logger throughout (no print statements)
  - Appropriate log levels (DEBUG, INFO, WARNING, ERROR)
  - Structured log messages with context
  - Retry progress indicators (🔄) in console output

### Multi-Language Collection (collector.py)
- Merges editions from multiple language versions per country
- Language prioritization: English > German/Dutch > Others
- Intelligent deduplication by product characteristics
- Conservative rate limiting (1.0-2.5s random delay)
- Chrome User-Agent for better compatibility
- **Custom URL Support**: Archive.org snapshots (requires --country option)
- **Automatic Proxy URL Cleaning**: Removes proxy prefixes by detecting second http(s)
- **Improved Methods**:
  - `_retry_request()`: Generic retry logic for all HTTP requests
  - `_process_single_country()`: Extracted from collect_all_data
  - `_initialize_collection_metadata()`: Cleaner initialization
  - `_collect_domain_editions()`: Modular domain processing

### Parallel Processing (processor.py)
- **Independent Processing**: Discovers countries directly from raw files, no collection_summary.json dependency
- **Direct Updates**: Single country processing updates final JSON by default
- **Enhanced Per-Edition Cache Intelligence**: Smart cache reuse for mixed scenarios (cached + new editions)
- **Robust Error Handling**: Automatic validation and retry mechanism for incomplete normalization (up to 2 attempts)
- **Translation Priority System**: Flavor field is SOURCE OF TRUTH, prioritized over description text
- **Intelligent Changelog Creation**: Only creates changelog files when actual changes are made (skips when all cache hits)
- **Legacy Mode**: `--separate-file` flag for backward compatibility (separate output files)
- Processes 3 countries simultaneously using ThreadPoolExecutor
- Manual corrections system via `data/corrections.json`
- Smart caching to minimize Gemini API calls with edition-level granularity
- Enhanced debug mode with detailed cache statistics ("X cached, Y need translation")
- Force reprocess option properly respected for single country processing

### Error Handling & Recovery (processor.py)
- **API Key Expiration Detection**:
  - Detects "expired" in Gemini API error messages
  - Immediately sets abort flag to stop all parallel processing
  - Cancels all pending futures in ThreadPoolExecutor
  - Exits with code 1 without saving any data
  - Clear error message: "Please renew your Gemini API key in .env file"
- **Corrections.json Validation**:
  - Validates JSON syntax on load
  - Hard stop with sys.exit(1) on invalid JSON
  - Shows exact line/column of syntax error
  - Prevents processing with corrupted corrections
- **Abort Flag Mechanism**:
  - Thread-safe `_abort_flag` for critical errors
  - Checked before starting new country processing
  - Prevents data corruption when errors occur
  - Clean shutdown of all parallel tasks

### Data Normalization Rules
- **Pre-processing** (before AI translation):
  - All "Açai" variations (açaí, açaï, açaì, etc.) → "Acai" in raw data
  - Applied to `_raw_flavor` and `_standfirst` fields
  - Ensures consistent handling regardless of source language
- **Names**:
  - "Sugarfree" → "Energy Drink Sugarfree"
  - "Zero" → "Energy Drink Zero"
  - **NEVER**: "The Original Edition" (must be "Energy Drink")
  - **NEVER**: "The Zero Edition" (must be "Energy Drink Zero")
  - **AI Name Validation** (prevents hallucinations like Apple→Apricot):
    - After AI translation, the name is validated against the raw source title (`header_data.content.title`) and the `product_url`
    - If either matches an APPROVED_EDITION, that match acts as ground truth
    - If AI-generated name differs from ground truth → Warning logged, correct name used
    - Log pattern: `⚠️ Edition name mismatch: AI said 'X' but source data indicates 'Y' → using 'Z'`
- **Editions**: Prefixed with "The" (e.g., "The Summer Edition")
  - Exception: Basic Energy Drink variants don't get "The" prefix
- **Flavors**:
  - **Translation Priority**: ALWAYS prioritize direct translation of 'flavor' field over description text
  - Combined flavors use hyphens (e.g., "Coconut-Blueberry")
  - "Sugarfree" kept standalone, removed from combinations
  - **Intelligent variation matching**: "Grapefruit-Woodruff" → "Woodruff & Pink Grapefruit"
  - **Specific translations**:
    - "Waldbeere/Waldbeeren" → "Forest Berry" (NOT "Raspberry")
    - **Forest Berry vs Forest Fruits** (DISTINCT flavors):
      - German "Waldbeere/Waldbeeren" → "Forest Berry"
      - English "Forest fruit/fruits" → "Forest Fruits"
      - These are different flavors and must NOT be interchanged
    - **Apricot Edition**: "Apricot-Strawberry" is the standardized order (NOT "Strawberry-Apricot")
      - Enforced via global corrections in corrections.json
      - Applies to Apricot/Amber/Summer Edition variants
  - Exceptions: "Pear Cinnamon" (space), "Grapefruit & Blossom" (keeps &)
- **Alphabetical Order**: Countries sorted alphabetically; editions within each country sorted by name
- **URLs**: All HTTP converted to HTTPS
- **Text**: Always in English (translated if necessary)
- **Brand**: "Red Bull" removed from names (case-insensitive)
- **JSON**: 4-space indentation

### Enhanced Translation Rules (2025 Update)
- **Strict 1:1 Translation**: AI instructed to translate EXACTLY as written, no interpretation
- **Specific Mappings**: 
  - Maracujá → Maracuja (NOT Passion Fruit)
  - Curuba → Curuba (NOT cuban)
- **Active Validation**: Step 3 validation now actively corrects flavors against APPROVED_FLAVORS list
- **Automatic Corrections**: Invalid flavors are automatically corrected during processing

### Region Emoji System
- **Unique emojis** for regions with "INT" flag code
- **Cached in** `data/region_emojis.json`
- **Priority order**:
  1. Flag mappings (e.g., Europe → 🇪🇺)
  2. Characteristic mappings (e.g., Caribbean → 🌴, Middle East → 🐪)
  3. Dynamic mappings (generated by Gemini for new regions)
  4. Fallback → 📍
- **No duplicates**: Each emoji is unique across all countries/regions
- **Gemini integration**: Automatically generates emojis for new regions

## Data Flow

1. **Input**: Red Bull API endpoints
   - Header API: `https://www.redbull.com/v3/api/custom/header/v2`
   - GraphQL API: `https://www.redbull.com/v3/api/graphql/v1/`

2. **Storage Structure**:
   - `data/raw/` - Raw API responses per country
   - `data/processed/` - Normalized data after Gemini processing
   - `data/corrections.json` - Manual corrections configuration
   - `data/redbull_editions_final.json` - Final consolidated output
   - `data/collection_summary.json` - Statistics and metadata

3. **Output Format**: 
```json
{
  "Country": {
    "flag": "🇦🇱",
    "editions": [
      {
        "name": "Energy Drink",
        "flavor": "Energy Drink",
        "flavor_description": "...",
        "sugarfree": false,
        "color": "#2F4581",
        "image_url": "...",
        "alt_text": "...",
        "product_url": "https://..."
      }
    ],
    "flag_url": "..."
  }
}
```

## Manual Corrections System

Create/edit `data/corrections.json`:
```json
{
  "corrections": [
    {
      "id": "graphql-id:locale",
      "field": "flavor",
      "search": "Original Text",
      "replace": "Corrected Text"
    },
    {
      "id": "f900c5b7-d33e-4a8e-a186-5cee5bd291a1",
      "field": "flavor",
      "search": "Strawberry-Apricot",
      "replace": "Apricot-Strawberry"
    },
    {
      "id": "22f260ac-2e6a-4082-9469-3bba2de2b523:en-ZA",
      "field": "flavor_description",
      "match_mode": "partial",
      "search": "Sea Blue Edition",
      "replace": "Blue Edition"
    }
  ]
}
```

**ID Format (Flexible Matching)**:
- **Country-Specific**: `"id": "UUID:de-DE"` → Applies only to Germany
- **Global (Multi-Country)**: `"id": "UUID"` → Applies to ALL countries with this UUID
- Example: `"id": "c55e5804-ce3c-4289-8d89-930b6d678501"` applies to FI, DE, GB, etc.
- **Real-world example**: The Apricot Edition correction above applies globally to ~16 countries

**Match Modes**:
- **`"exact"` (default)**: Entire field value must match `search` (case-insensitive). Prevents accidental partial matches (e.g. `"Peach"` won't match `"White Peach"`). Omit `match_mode` to use this default.
- **`"partial"`**: `search` is replaced as a substring anywhere in the field value (case-insensitive check, case-sensitive replace). Useful for long text fields like `flavor_description` where the full value may change upstream. A failed partial match is logged to the changelog.

**Field Mapping (Automatic)**:
- Use `"flavor"` in corrections → internally maps to `"_raw_flavor"`
- Use `"flavor_description"` → internally maps to `"_standfirst"`
- This allows intuitive field names while the processor handles the technical details

**Active Global Corrections** (as of 2025-10-22):
- **Apricot/Amber/Summer Edition** (UUID `f900c5b7-d33e-4a8e-a186-5cee5bd291a1`):
  - Two corrections ensure "Apricot-Strawberry" flavor order (not "Strawberry-Apricot")
  - Corrects both raw API data and Gemini-translated data
  - Applied automatically before Gemini processing on every run
  - Affects: AT, DK, EE, ES, FR, GB, HU, IT, LV, MEA, MK, NL, NO, PT, RO, SE, SI, SK, US
  - Details: See `data/changelogs/changelog_20251022_135339_manual.md`

## Intelligent Caching System

### Per-Edition Cache Logic
The processor uses intelligent caching to minimize Gemini API calls:

1. **Hash-Based Change Detection**: Uses SHA256 hash of raw edition data to detect changes
2. **Edition-Level Granularity**: When data changes, only new/modified editions are processed
3. **Cache Reuse**: Existing editions with unchanged data reuse cached translations
4. **Automatic Cache Invalidation**: Stale cache entries are automatically identified and updated

### Cache Behavior Examples
- **New Edition Added**: Only the new edition is translated, existing 7 editions use cache
- **Edition Removed**: Cache automatically adapts, no API calls needed
- **Edition Modified**: Only the modified edition is re-translated
- **Mixed Scenario**: Intelligently handles partial cache hits with new editions
- **No Changes**: Complete cache hit, no API calls

### Enhanced Verbose Output
Use `--verbose` flag to see detailed cache statistics and validation:
```
📊 Cache stats: 7 cached, 1 need translation
♻️ Using cached translation for Energy Drink
🆕 New edition needs translation: Winter Edition
⚠️ Normalization count mismatch: Expected 8, got 7
🔄 Retrying normalization for missing editions (attempt 1/2)
✅ Successfully normalized missing editions. Total: 8
```

### Error Recovery System
The processor includes robust validation and recovery:
- **Automatic Validation**: Ensures all input editions receive normalization
- **Retry Mechanism**: Up to 2 automatic retries for missing normalized data
- **Graceful Degradation**: Continues with partial results if retries fail
- **Detailed Logging**: Shows specific edition names and IDs for missing data

This approach reduces API costs by ~90% for incremental changes while ensuring data accuracy.

### Intelligent Changelog System

The processor now intelligently determines when to create changelog files:

**Changelog is created when:**
- Countries are actually processed (not just cache hits)
- Manual corrections are applied or fail
- Processing errors occur

**Changelog is skipped when:**
- All countries use cached data (no changes)
- No corrections applied
- No errors occurred

**Output messages:**
```
📝 Changelog saved to: data/changelogs/changelog_20250909_143022.md  # Changes made
📝 No changes made - changelog skipped                                # All cache hits
```

This prevents unnecessary changelog files while ensuring important changes and errors are still documented.

### Validation System

The processor includes active Step 3 validation that:
- **Checks all flavors** against APPROVED_FLAVORS list  
- **Automatically corrects** non-approved flavors
- **Prevents translation errors** (e.g., "Passion Fruit" → "Maracuja", "cuban" → "Curuba")
- **Shows corrections** in debug mode with 🔧 icons
- **Respects manual corrections** from corrections.json file

## API Configuration

### Rate Limiting & Retry Configuration
- **HTTP-Level Retries**: 3 automatic retries per request (502, 503, 504 errors)
- **Country-Level Retries**: 1 retry per country (2 total attempts)
  - Configurable via `Config.COUNTRY_MAX_RETRIES` (default: 1)
  - Delay via `Config.COUNTRY_RETRY_DELAY` (default: 5 seconds)
- **GraphQL Delay**: Random 1.0-2.5 seconds between GraphQL requests (conservative 2x)
- **Domain Delay**: 500ms between country fetches
- **Processor Delay**: 7 seconds minimum between Gemini API calls
- **Development Mode**: Disable with `--no-rate-limit` flag

### Collection Date Optimization (2025-10-22)
- **Smart Date Preservation**: `collection_date` only updated when changes detected
- **Prevents Unnecessary Commits**: Avoids git commits when no data changes
- **Implementation**: Reads previous `collection_date` from `collection_summary.json`
  - If `changes_detected` is empty → preserves old date
  - If `changes_detected` has countries → sets new date
- **Benefit**: Cleaner git history with commits only when data actually changes

### Collector Retry Behavior
When a country fails (e.g., API timeout, HTTP error):
1. **First Attempt Fails** → Wait 5 seconds
2. **Second Attempt** (Retry) → Try again
   - **Success** ✅ → Continue with next country
   - **Failure** ❌ → **Stop immediately with Exit Code 1**

**Example Output**:
```
[53/67] Processing Romania...
  ❌ ro-ro: Header API failed (HTTP 502 Bad Gateway)
  🔄 Retrying Romania in 5 seconds (attempt 2/2)...

[Retry] Processing Romania...
  ❌ ro-ro: Header API failed (HTTP 502 Bad Gateway)

❌ COLLECTION FAILED
Error: Romania failed after 2 attempts: API error for Romania
Collection aborted - no partial data saved
```
**Exit Code**: 1 (CI/CD pipeline will fail)

### Gemini AI Integration
- Model: `gemini-2.5-flash`
- Used for text normalization and translation
- Fallback processing when API key unavailable
- Parallel processing with 3 concurrent requests

### Request Headers
```python
"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 (+https://github.com/Haxe18/redbull-editions-generator)"
```

The User-Agent includes the project URL to allow API administrators to identify and contact the project maintainer if needed.

## GitHub Actions Automation

The workflow (`update-editions.yml`) runs daily at 5:11 AM UTC and:
- Uses Python 3.12 and latest GitHub Actions (setup-python@v5)
- Caches dependencies and raw data for faster runs
- Detects changes in multiple files before committing
- Creates releases automatically on every change (not just weekly)
- Uses GitHub CLI for modern release management
- Can be manually triggered with options:
  - `limit`: Number of countries to process
  - `force`: Force reprocess all data (ignores cache)
- Uses `github-actions[bot]` for commits
- Includes changelog files in releases when available
- Dynamic versioning with timestamps (e.g., v2024.12.15-1430)

## Troubleshooting

### Collector Issues
- **No editions found**: Check if country has products on redbull.com
- **API errors**: Verify rate limiting and User-Agent header
- **Collection fails with Exit 1**:
  - Check network connectivity
  - Verify Red Bull API is accessible
  - Review error message for specific country/domain that failed
  - Note: Collector uses fail-fast - stops on first failure after retries
- **Duplicate countries**: Collector V2 automatically merges language versions

### Processor Issues
- **Gemini errors**: Check API key and model availability
- **API Key Expiration**: Renew your Gemini API key in .env file

### Exit Codes
- **0**: Success - all countries processed
- **1**: Failure - at least one country failed after retries (collector stops immediately)

## Dependencies

- `google-genai==1.31.0` - Gemini AI API client
- `requests` - HTTP client for API calls
- `python-dotenv` - Environment variable management
- Python 3.8+ required