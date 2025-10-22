# Red Bull Editions Generator

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)

Automated data collector and processor for Red Bull Energy Drink editions worldwide.

## ✨ Features

- 🌍 Collects edition data from 68 unique countries (with multi-language support)
- 🤖 AI-powered normalization using Google Gemini 2.5 Flash
- 🌐 Intelligent language prioritization (English > German/Dutch > Others)
- 🔄 Parallel processing with threading (3 concurrent workers)
- 📊 Change detection to minimize API calls
- 🛡️ Rate limiting with random delays (0.5-1.5s)
- 📝 Manual corrections system for data overrides
- 💾 Efficient caching of raw data with intelligent per-edition cache invalidation
- 📚 Production-ready code with high Pylint score
- 🔌 HTTP connection pooling with automatic retry strategy
- 🎯 Clean architecture with separation of concerns

## 🏗️ Architecture

```
collector.py  →  raw data  →  processor.py  →  final JSON
(multi-lang)     (cached)     (Gemini AI)
```

### Components

1. **collector.py**: Production-ready data collector
   - **Architecture**: Clean separation of concerns with dataclasses and NamedTuples
   - **HTTP**: Session pooling with automatic retry strategy (exponential backoff)
   - **Multi-language**: Merges editions from multiple language versions per country
   - **Language Priority**: English > German/Dutch > Others
   - **Rate Limiting**: Configurable delays (0.5-1.5s random)
   - **Custom URLs**: Support for Archive.org snapshots (requires --country option)
   - **Error Handling**: Custom exception hierarchy with proper recovery
   - **Logging**: Structured logging with appropriate levels
   - **Configuration**: Externalized config with frozen dataclass
   - **Type Safety**: Full type hints throughout the codebase

2. **processor.py**: Advanced data processor with AI normalization
   - **Independent Operation**: Works directly with raw files, no collection dependency
   - **Direct Updates**: Single country processing updates final JSON by default
   - **Enhanced Per-Edition Caching**: Smart cache reuse for mixed scenarios with detailed statistics
   - **Robust Error Handling**: Automatic validation and retry mechanism for incomplete processing
   - **Translation Priority System**: Flavor field prioritized as source of truth over descriptions
   - **Intelligent Changelog Creation**: Only creates changelog files when actual changes occur (skips pure cache runs)
   - **Legacy Mode**: `--separate-file` flag for backward compatibility
   - Parallel processing (3 countries simultaneously)
   - Manual corrections via `data/corrections.json`
   - Smart flavor name cleaning and standardization
   - Enhanced debug mode with cache hit/miss reporting
   - Force reprocess option properly respected for single country processing
   - Comprehensive docstrings for 40+ methods and all Pydantic models

## ⚙️ Setup

### Local Development

1. Install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Set up Gemini API key:
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

3. Run data collection:
```bash
# Data collection with multi-language support
python collector.py --test              # First 3 countries
python collector.py --small             # First 10 countries
python collector.py                     # All countries
python collector.py --no-rate-limit     # Dev mode without delays
python collector.py --country Germany   # Specific country
python collector.py --debug             # Debug output (verbose console)

# Use custom URL (requires --country option)
python collector.py --country Netherlands --custom-url "https://web.archive.org/web/20240827090215/https://www.redbull.com/v3/api/custom/header/v2?locale=nl-nl"
# Note: --custom-url only works with --country option
```

4. Process data:
```bash
# Basic processing
python processor.py

# Advanced options
python processor.py --force                      # Force reprocess all, ignore cache
python processor.py --verbose                    # Show detailed console output
python processor.py --debug                      # Write API logs to debug files
python processor.py --country Germany            # Process single country (updates final JSON, respects --force)
python processor.py --country Germany --separate-file  # Process to separate file (legacy)
python processor.py --country Germany --force    # Force reprocess single country (ignores cache)
python processor.py --force --verbose --country Spain    # Combine options
```

### Manual Corrections

Create/edit `data/corrections.json` to override specific fields before AI processing:
```json
{
    "corrections": [
        {
            "id": "9f5e826b-3589-4e15-8da7-86759325fc9b:en-GB",
            "field": "flavor",
            "search": "Dragon Fruit",
            "replace": "Curuba-Elderflower"
        },
        {
            "id": "c55e5804-ce3c-4289-8d89-930b6d678501",
            "field": "flavor",
            "search": "Peach",
            "replace": "White Peach"
        },
        {
            "id": "f900c5b7-d33e-4a8e-a186-5cee5bd291a1",
            "field": "flavor",
            "search": "Strawberry-Apricot",
            "replace": "Apricot-Strawberry"
        }
    ]
}
```

**ID Format** (Flexible Matching):
- **Country-Specific**: `"id": "UUID:de-DE"` → Applies only to Germany (example 1 above)
- **Global (Multi-Country)**: `"id": "UUID"` → Applies to ALL countries with this UUID
- Examples:
  - Correction 2 applies to FI, DE, GB, and any other country with that Peach Edition UUID
  - Correction 3 standardizes Apricot Edition flavor order globally across ~16 countries

**Field Mapping**: The processor automatically maps user-friendly field names to internal fields:
- `"flavor"` → `"_raw_flavor"` (original flavor text)
- `"flavor_description"` → `"_standfirst"` (original description text)

This allows you to use intuitive field names in corrections while the processor handles the technical mapping internally.

**Active Corrections** (2025-10-22):
- **Apricot Edition standardization**: Two global corrections ensure "Apricot-Strawberry" flavor order
  - Corrects both raw data ("Strawberry and apricot") and translated data ("Strawberry-Apricot")
  - Applies automatically during daily runs to maintain consistency
  - See `data/changelogs/changelog_20251022_135339_manual.md` for details

### GitHub Actions Setup

1. Add secret `GEMINI_API_KEY` in repository settings
2. Enable GitHub Actions
3. Manual trigger available in Actions tab with options:
   - **limit**: Number of countries to process (optional)
   - **force**: Force reprocess all data ignoring cache (optional)

The workflow runs daily at 5:11 AM UTC and:
- Uses Python 3.12 for better performance
- Caches dependencies and raw data
- Creates releases automatically when changes are detected
- Uses changelog files for release notes when available
- Supports force mode for manual triggers

## 📤 Output Format

```json
{
  "Albania": {
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

### Data Processing Rules

- **Pre-processing**: "Açai" variations (açaí, açaï, etc.) normalized to "Acai" before AI translation
- **Translation Priority**: Flavor field is SOURCE OF TRUTH, prioritized over description text
- **Names**: 
  - "Sugarfree" → "Energy Drink Sugarfree"
  - "Zero" → "Energy Drink Zero"
  - **Never**: "The Original Edition" (always "Energy Drink")
  - **Never**: "The Zero Edition" (always "Energy Drink Zero")
- **Editions**: Automatically prefixed with "The" (e.g., "The Summer Edition")
  - Exception: Basic Energy Drink variants don't get "The" prefix
- **Flavors**: 
  - Combined flavors use hyphens (e.g., "Coconut-Blueberry")
  - Specific translations: "Waldbeere" → "Forest Berry" (NOT "Raspberry")
  - Forest Berry ≠ Forest Fruits (kept distinct)
- **URLs**: All HTTP converted to HTTPS
- **Descriptions**: Always in English (translated if necessary)
- **JSON**: 4-space indentation

## 📁 Data Structure

- `data/raw/`: Raw API responses per country
- `data/processed/`: Processed data with normalization
- `data/redbull_editions_final.json`: Final output
- `data/collection_summary.json`: Collection statistics
- `data/changelogs/`: Changelog files (only created when changes occur)
- `data/latest_changelog.md`: Latest changelog for easy access

## 📝 Changelog Behavior

The processor intelligently creates changelog files:
- **Creates changelog**: When countries are processed, corrections applied, or errors occur
- **Skips changelog**: When all data is current (cache hits only)
- **Output**: Shows "📝 Changelog saved" or "📝 No changes made - changelog skipped"

## ⏱️ Rate Limiting

The collector implements intelligent rate limiting to respect API limits:
- **GraphQL requests**: Random delay of 0.5-1.5 seconds between requests
- **Country collection**: 300ms delay between country fetches
- **Exponential backoff**: Automatic retry with exponential backoff on HTTP errors (502, 503, 504)
- **Development mode**: Use `--no-rate-limit` flag to disable delays for testing

## 🔧 Troubleshooting

### Collector Issues

**No editions found for a country**
- Verify the country has products listed on redbull.com
- Try using `--debug` flag for detailed output
- Check if the country is accessible in your region

**API errors (HTTP 502, 503, 504)**
- The collector automatically retries with exponential backoff
- If persistent, check Red Bull API status
- Verify network connectivity and firewall settings

**Collection fails with Exit Code 1**
- Check error message for specific country/domain that failed
- Verify rate limiting isn't triggering API blocks
- Try using `--no-rate-limit` for testing (not recommended for production)
- Note: Collector uses fail-fast strategy - stops on first unrecoverable failure

### Processor Issues

**Gemini API errors**
- Verify `GEMINI_API_KEY` is set correctly in `.env` file
- Check API key hasn't expired (renew if needed)
- Verify Gemini API quota limits haven't been exceeded
- Review error message for specific API issue

**Incorrect translations or flavors**
- Use `data/corrections.json` for manual overrides
- Check CLAUDE.md for approved flavor names and translation rules
- Use `--force` flag to reprocess with latest corrections
- Use `--verbose` flag to see cache statistics and validation details

**Missing or incomplete data**
- The processor includes automatic validation and retry (up to 2 attempts)
- Check debug logs with `--debug` flag for detailed API communication
- Verify raw data files exist in `data/raw/` directory
- Try `--force` to bypass cache and reprocess all data

**Changelog not created**
- This is normal when all data uses cached translations (no changes)
- Changelogs are only created when actual processing occurs
- Use `--force` to force processing and generate changelog

### GitHub Actions Issues

**Workflow fails to commit**
- Verify `GEMINI_API_KEY` secret is set in repository settings
- Check workflow logs for specific error messages
- Ensure repository permissions allow GitHub Actions to commit

**Releases not created**
- Releases are only created when data changes are detected
- Check if `has_changes` step shows changes
- Verify GitHub token has release creation permissions

## 🏆 Credits

This project was developed with assistance from **Claude Code** (Anthropic), an AI-powered development assistant. The codebase was created through **pair programming** sessions, iterating on user feedback and requirements, contributing to:

- Architecture design and refactoring to achieve production-ready code quality
- Implementation of robust error handling with custom exception hierarchies
- Multi-language data collection and intelligent deduplication logic
- Parallel processing with ThreadPoolExecutor and smart caching mechanisms
- Comprehensive documentation and inline code comments
- Test coverage and code quality improvements

Claude Code helped evolve the project from a monolithic script to a clean, maintainable codebase with proper separation of concerns, type safety, and professional logging practices.

## 🙏 Acknowledgments

- **Red Bull** for the amazing product data and public APIs
- **Google** for the Gemini AI platform powering data normalization
- **GitHub** for the excellent Pages and Actions platforms
- **Anthropic** for providing Claude AI assistance
- **The open-source community** for the tools and libraries used

## 📄 License

This project is licensed under the **WTFPL** (Do What The F*ck You Want To Public License) - see the [LICENSE](LICENSE) file for details.

**TL;DR**: Do whatever you want with this code. Seriously. 🤷‍♂️

---

**Developed with ❤️ using AI-Human collaboration**

---

**Note**: This is a fan project and is not officially affiliated with Red Bull GmbH.
