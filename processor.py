#!/usr/bin/env python3
"""
Red Bull Editions Data Processor

A comprehensive data processing system for Red Bull product editions.
Processes raw data and normalizes it using Google Gemini AI.
"""

import argparse
import hashlib
import json
import logging

# region Imports
# Standard library imports
import os
import re
import sys
import time
import traceback
import unicodedata
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, cast

from babel import Locale, UnknownLocaleError

# Third-party imports
from dotenv import load_dotenv
from google import genai
from google.genai import errors
from google.genai.types import GenerateContentConfig
from pydantic import BaseModel, Field

# Local imports
from lib.logging_utils import setup_basic_logging, setup_logger

# endregion

# region Environment Setup
load_dotenv()

# Configure logging
setup_basic_logging()

# Suppress HTTP request logs from google-genai library
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("google_genai.models").setLevel(logging.WARNING)
# endregion


# region Pydantic Models
class Edition(BaseModel):
    """Represents a normalized Red Bull edition.

    Pydantic model for structured edition data with validation.
    Used for Gemini AI processing and normalization.

    Attributes:
        edition_id: Unique GraphQL ID for mapping back to original edition.
        name: Original edition name (used for mapping).
        flavor: Standardized English flavor name.
        flavor_description: Clean English description (1-2 sentences).
        sugarfree: Whether this edition is sugar-free.
    """

    edition_id: str = Field(description="Unique GraphQL ID - MUST be preserved from input")
    name: str = Field(description="Original edition name for mapping purposes")
    flavor: str = Field(description="Standardized English flavor name")
    flavor_description: str = Field(description="Clean English description (1-2 sentences)")
    sugarfree: bool = Field(description="True if this edition is sugar-free, False otherwise")


class TranslatedEdition(BaseModel):
    """Represents a translated Red Bull edition.

    Pydantic model for editions after translation step.
    Used as intermediate format before normalization.

    Attributes:
        edition_id: Unique GraphQL ID for mapping back to original edition.
        name: Translated English name of the edition.
        flavor: Translated English flavor text.
        description: Translated English description text.
        product_url: Optional product URL for additional context.
    """

    edition_id: str = Field(description="Unique GraphQL ID - MUST be preserved from input")
    name: str = Field(description="Translated English name of the edition")
    flavor: str = Field(description="Translated English flavor text")
    description: str = Field(description="Translated English description text")
    product_url: Optional[str] = Field(default=None, description="Product URL for context")


class ValidationResult(BaseModel):
    """Represents the validation result for an edition.

    Pydantic model for validation step results.
    Used to track and correct issues in normalized data.

    Attributes:
        is_valid: Whether the edition passes validation.
        flavor_in_approved_list: Whether the flavor is in the approved list.
        corrections_needed: List of corrections that should be made.
        corrected_flavor: The corrected flavor name if needed.
        corrected_description: The corrected description if non-English text was found.
    """

    is_valid: bool = Field(description="Whether the edition passes all validation rules")
    flavor_in_approved_list: bool = Field(description="Whether the flavor is in the approved list")
    corrections_needed: List[str] = Field(description="List of issues found that need correction")
    corrected_flavor: Optional[str] = Field(description="The corrected flavor name if correction is needed")
    corrected_description: Optional[str] = Field(description="The corrected description if non-English text was found")


# endregion


class RedBullDataProcessor:
    """Main processor class for Red Bull editions data.

    This class handles the complete pipeline of processing Red Bull product
    data from raw API responses to normalized,  translated output using
    Google Gemini AI. Supports parallel processing, caching, and manual corrections.

    Attributes:
        data_dir: Base directory for data storage.
        raw_dir: Directory containing raw API responses.
        processed_dir: Directory for processed data.
        debug: Enable debug output with timing metrics.
        max_workers: Number of parallel workers (3 for optimal performance).
        client: Google Gemini AI client instance.
        corrections: Manual corrections loaded from corrections.json.
        changelog: Tracks changes during processing.
        print_lock: Thread lock for thread-safe printing.
        last_api_call_time: Timestamp of last API call for rate limiting.
    """

    # region Class Constants
    GEMINI_MODEL = "gemini-2.5-flash-lite"
    MIN_DELAY_BETWEEN_REQUESTS = 6.5  # seconds (60s / 10 RPM = 6.0s minimum, 6.5s buffer)
    MAX_REQUESTS_PER_MINUTE = 10
    MAX_REQUESTS_PER_DAY = 20
    SIMILARITY_THRESHOLD = 0.75  # Minimum similarity for flavor matching

    # APPROVED FLAVOR LIST (Source of Truth)
    APPROVED_FLAVORS = [
        "Apricot-Strawberry",
        "Acai",
        "Blueberry",
        "Blueberry & Vanilla",
        "Cactus Fruit",
        "Cherry Sakura",
        "Cherry & Wild Berries",
        "Coconut-Blueberry",
        "Curuba-Elderflower",
        "Dragon Fruit",
        "Energy Drink",
        "Exotic Passion Fruit",
        "Raspberry",
        "Forest Berry",
        "Forest Fruits",
        "Fuji Apple & Ginger",
        "Glacier Ice",
        "Grapefruit & Blossom",
        "Iced Gummy Bear",
        "Iced Vanilla Berry",
        "Juneberry",
        "Maracuja & Melon",
        "Strawberry & Peach",
        "Peach",
        "Pear Cinnamon",
        "Pomelo",
        "Pomegranate",
        "Sudachi Lime",
        "Sugarfree",
        "Tropical Fruits",
        "Watermelon",
        "White Peach",
        "Wild Berries",
        "Wildflower & Pink Grapefruit",
        "Woodruff & Pink Grapefruit",
        "Zero Sugar",
    ]

    # APPROVED EDITION LIST (Known editions for guidance - not strict enforcement)
    APPROVED_EDITIONS = [
        "Amber Edition",
        "Amora Edition",
        "Apple Edition",
        "Apricot Edition",
        "Berry Edition",
        "Blue Edition",
        "Coconut Edition",
        "Green Edition",
        "Festive Edition",
        "Glacier Edition",
        "Ice Edition",
        "Lilac Edition",
        "Lime Edition",
        "Lime Green Edition",
        "Peach Edition",
        "Pink Edition",
        "Pomelo Edition",
        "Purple Edition",
        "Red Edition",
        "Ruby Edition",
        "Sea Blue Edition",
        "Spring Edition",
        "Summer Edition",
        "Tropical Edition",
        "White Edition",
        "Winter Edition",
        "Yellow Edition",
    ]

    # Flavor combinations that should keep spaces
    SPACE_COMBINATIONS = [
        "pear cinnamon",
        "tropical fruits",
        "arctic berry",
        "beach breeze",
        "winter edition",
    ]

    # Marketing phrases to detect in names
    MARKETING_PHRASES = ["same wings", "new taste", "gives you wings", "vitalizes body"]
    # endregion

    # region Initialization
    def __init__(
        self,
        data_dir: str = "data",
        debug: bool = False,
        max_workers: int = 3,
        verbose: bool = False,
        single_country: bool = False,
    ) -> None:
        """
        Initialize the Red Bull Data Processor.

        Args:
            data_dir: Base directory for data storage
            debug: Enable debug output
            max_workers: Number of parallel workers for multi-country processing (default: 3)
            verbose: Enable verbose mode with full API request/response logging
            single_country: Whether processing only a single country (disables rate limiting)

        Raises:
            ValueError: If GEMINI_API_KEY is not found in environment
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.verbose_dir = self.data_dir / "debug"
        self.processed_dir.mkdir(exist_ok=True)
        self.verbose = verbose
        self.debug = debug
        self.single_country = single_country

        # Thread safety - initialize early since _log_debug uses it
        self.print_lock = Lock()
        self.api_call_lock = Lock()
        self._abort_flag = False  # Flag to abort all processing on critical errors
        self._daily_limit_reached = False  # Flag when daily API quota is exhausted
        self._skipped_due_to_limit: List[str] = []  # Countries skipped due to daily limit

        # Setup logger
        self.logger = setup_logger(self.__class__.__name__, enable_verbose=verbose, debug=debug)
        if self.debug:
            logging.getLogger().setLevel(logging.DEBUG)

        # Create debug directory if debug mode
        if self.verbose:
            self.verbose_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.verbose_log_file = self.verbose_dir / f"debug_{timestamp}.log"
            # Add file handler for verbose mode
            file_handler = logging.FileHandler(self.verbose_log_file, mode="w", encoding="utf-8")
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(file_handler)
            self.logger.info("🔍 DEBUG MODE: Full API logging enabled")
            self.logger.info("   Debug log: %s", self.verbose_log_file)
            self._log_debug("=" * 80)
            self._log_debug(f"Debug session started at {datetime.now().isoformat()}")
            self._log_debug("=" * 80)

        # Set workers for parallel processing
        self.max_workers = max_workers

        # Rate limiting
        self.last_api_call_time: Optional[float] = None

        # Changelog tracking
        self.changelog = {
            "timestamp": datetime.now().isoformat(),
            "countries_processed": [],
            "countries_skipped": [],
            "editions_added": defaultdict(list),
            "editions_updated": defaultdict(list),
            "editions_removed": defaultdict(list),
            "field_changes": defaultdict(list),
            "corrections_applied": [],
            "corrections_failed": [],
            "id_mappings_failed": [],
            "api_calls_made": 0,
            "cache_hits": 0,
            "errors": [],
            "summary": {},
            "countries_skipped_daily_limit": [],
        }

        # Load corrections
        self.corrections, self.id_mappings = self._load_corrections()
        self.corrections_tracking = {}

        # Initialize Gemini client
        self._initialize_gemini_client()

    def _initialize_gemini_client(self) -> None:
        """
        Initialize the Google Gemini AI client.

        Raises:
            ValueError: If GEMINI_API_KEY is not found in environment
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            self.logger.error("❌ No GEMINI_API_KEY found in environment")
            self.logger.error("   Please set the GEMINI_API_KEY environment variable or add it to .env file")
            raise ValueError("GEMINI_API_KEY is required for processing")

        self.client = genai.Client(api_key=api_key)
        self.logger.info("✅ Gemini AI initialized")

        if self.debug:
            self.logger.debug("  📍 API Key: %s...", api_key[:10])
            self.logger.debug("  🤖 Model: %s", self.GEMINI_MODEL)
            self.logger.debug("  🔄 Max parallel workers: %d", self.max_workers)

    def _load_corrections(self) -> tuple[List[Dict], List[Dict]]:
        """Load manual corrections and ID mappings from JSON file.

        Loads corrections.json which contains manual overrides for
        specific editions identified by GraphQL ID and locale, and optional
        id_mappings entries that replace raw fields from another locale.

        Returns:
            Tuple of (corrections list, id_mappings list).
        """
        corrections_file = self.data_dir / "corrections.json"
        if corrections_file.exists():
            try:
                with open(corrections_file, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    corrections = data.get("corrections", [])
                    id_mappings = data.get("id_mappings", [])
                    if corrections and self.debug:
                        self.logger.info("📝 Loaded %d manual corrections", len(corrections))
                    if id_mappings and self.debug:
                        self.logger.info("🔀 Loaded %d ID mappings", len(id_mappings))
                    return corrections, id_mappings
            except json.JSONDecodeError as err:
                self.logger.error("\n❌ INVALID JSON in %s", corrections_file)
                self.logger.error("   Error at line %d, column %d", err.lineno, err.colno)
                self.logger.error("   %s", err.msg)
                self.logger.error("\n   Please fix the JSON syntax in corrections.json")
                sys.exit(1)
        return [], []

    def _get_word_set(self, flavor: str) -> tuple:
        """Extract sorted word tuple for order-insensitive comparison.

        Splits flavor string into individual words, normalizes them,
        and returns as sorted tuple for comparison regardless of word order.
        Example: "Apple Fuji-Ginger" → ('apple', 'fuji', 'ginger')

        Args:
            flavor: Flavor string to extract words from.

        Returns:
            Tuple of sorted lowercase words.
        """
        normalized = flavor.lower().replace("-", " ").replace("&", " ")
        words = sorted([w for w in normalized.split() if w])
        return tuple(words)

    @staticmethod
    def _get_language_priority(domain: str) -> int:
        """Get language priority for domain (lower = higher priority).

        Used to select preferred language version when multiple raw files
        exist for the same country (e.g., ch-de vs ch-fr for Switzerland).

        Priority order:
        1. English (en, gb, us)
        2. German/Dutch (de, nl)
        3. Austrian/Swiss variants (at, ch)
        4. French (fr)
        5. Spanish (es)
        6. Portuguese (pt)
        7. Italian (it)
        99. Other languages

        Args:
            domain: The locale domain (e.g., 'ch-de', 'ch-fr').

        Returns:
            Priority number (lower = higher priority).
        """
        if not domain or "-" not in domain:
            return 999

        lang = domain.split("-")[-1].lower()
        priorities = {
            "en": 1,
            "gb": 1,
            "us": 1,  # English
            "de": 2,
            "nl": 2,  # German/Dutch
            "at": 3,
            "ch": 3,  # Austrian/Swiss German
            "fr": 4,  # French
            "es": 5,
            "pt": 6,
            "it": 7,  # Other European
        }
        return priorities.get(lang, 99)

    def discover_raw_files(self) -> Dict[str, Dict[str, Any]]:
        """Discover and load metadata from all raw files in data/raw/ directory.

        Scans the raw directory for JSON files and extracts country metadata.
        Independent of collection_summary.json for better resilience.

        Returns:
            Dictionary mapping country names to their metadata including:
            - domain: Primary locale domain
            - flag_code: Country flag code
            - file_path: Path to raw data file
        """
        countries = {}
        raw_dir = self.data_dir / "raw"

        if not raw_dir.exists():
            return countries

        for json_file in raw_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as file:
                    data = json.load(file)

                locale_info = data.get("locale_info", {})
                country_name = locale_info.get("country_name")

                if country_name:
                    new_domain = json_file.stem  # e.g., "ch-de", "ch-fr"
                    new_priority = self._get_language_priority(new_domain)

                    # Only update if this domain has higher priority (lower number)
                    if country_name in countries:
                        existing_priority = self._get_language_priority(countries[country_name]["domain"])
                        if new_priority >= existing_priority:
                            continue  # Keep existing entry with higher/equal priority

                    countries[country_name] = {
                        "domain": new_domain,
                        "flag_code": locale_info.get("flag_code", ""),
                        "data_file": json_file.name,
                        "editions_count": len(data.get("editions", [])),
                    }

            except (json.JSONDecodeError, IOError) as err:
                if self.debug:
                    self.logger.debug("⚠️  Failed to read %s: %s", json_file.name, err)

        return countries

    def update_final_json_with_country(self, country_name: str, country_data: Dict[str, Any]) -> bool:
        """Update the final JSON file with a single country's data.

        Directly updates redbull_editions_final.json with the processed
        data for a single country. Preserves order and other countries' data.

        Args:
            country_name: Name of the country to update.
            country_data: Processed country data to insert.

        Returns:
            True if successful, False otherwise.
        """
        final_file = self.data_dir / "redbull_editions_final.json"

        try:
            # Load existing final data
            if final_file.exists():
                with open(final_file, "r", encoding="utf-8") as file:
                    final_data = json.load(file)
            else:
                final_data = {}

            # Remove raw processing fields
            clean_data = dict(country_data)
            clean_data.pop("_raw_hash", None)
            clean_data.pop("_translated_editions", None)

            # Update the country data
            final_data[country_name] = clean_data

            # Apply description prefixes to all countries
            for _, data in final_data.items():
                if "editions" in data:
                    data["editions"] = self.add_description_prefix(data["editions"])

            # Save updated final data (sorted alphabetically by country and edition name)
            with open(final_file, "w", encoding="utf-8") as file:
                json.dump(self._sort_final_data(final_data), file, indent=4, ensure_ascii=False)

            return True

        except json.JSONDecodeError as err:
            error_msg = f"  ❌ Failed to parse JSON file: {err}\n" f"     File: {final_file}\n" f"     Line: {err.lineno}, Column: {err.colno}"
            self.logger.error(error_msg)
            if self.verbose:
                self.logger.error("     Stack trace:")
                traceback.print_exc()
            return False
        except IOError as err:
            error_msg = f"  ❌ Failed to read/write file: {err}\n" f"     File: {final_file}"
            self.logger.error(error_msg)
            if self.verbose:
                self.logger.error("     Stack trace:")
                traceback.print_exc()
            return False
        except KeyError as err:
            error_msg = f"  ❌ Missing required field in data: {err}\n" f"     Country: {country_name}"
            self.logger.error(error_msg)
            if self.verbose:
                self.logger.error("     Stack trace:")
                traceback.print_exc()
            return False
        except Exception as err:
            error_msg = f"  ❌ Unexpected error updating final JSON: {err}\n" f"     Type: {type(err).__name__}"
            self.logger.error(error_msg)
            if self.verbose:
                self.logger.error("     Stack trace:")
                traceback.print_exc()
            return False

    # endregion

    # region Thread-Safe Operations
    def thread_safe_print(self, message: str) -> None:
        """Thread-safe printing to console.

        Ensures output doesn't get interleaved when using parallel processing.

        Args:
            message: Message to print.
        """
        with self.print_lock:
            self.logger.info(message)

    def _log_debug(self, message: str) -> None:
        """Write debug message to log file (thread-safe).

        Writes to debug log file when verbose mode is enabled.

        Args:
            message: Debug message to log.
        """
        if self.verbose:
            with self.print_lock:
                with open(self.verbose_log_file, "a", encoding="utf-8") as file:
                    file.write(f"{message}\n")

    def _log_api_request(self, country: str, step: str, prompt: str) -> None:
        """Log API request details.

        Logs full API prompts and metadata for debugging in verbose mode.

        Args:
            country: Country being processed.
            step: Processing step (translation/normalization/validation).
            prompt: The prompt being sent to API.
        """
        if self.verbose:
            self._log_debug(f"\n{'='*80}")
            self._log_debug(f"API REQUEST - {country} - {step}")
            self._log_debug(f"Timestamp: {datetime.now().isoformat()}")
            self._log_debug(f"Prompt length: {len(prompt)} characters")
            self._log_debug("-" * 40)
            self._log_debug("PROMPT:")
            self._log_debug(prompt)
            self._log_debug("-" * 40)

    def _log_api_response(self, country: str, step: str, response: Any, duration: float) -> None:
        """Log API response details.

        Logs full API responses and timing for debugging in verbose mode.

        Args:
            country: Country being processed.
            step: Processing step.
            response: The API response (Pydantic models or raw data).
            duration: Time taken for API call in seconds.
        """
        if self.verbose:
            self._log_debug(f"\nAPI RESPONSE - {country} - {step}")
            self._log_debug(f"Duration: {duration:.2f} seconds")
            self._log_debug("-" * 40)
            if response:
                try:
                    if hasattr(response, "__iter__") and all(hasattr(r, "model_dump") for r in response):
                        # List of Pydantic models
                        response_data = [r.model_dump() for r in response]
                    elif hasattr(response, "model_dump"):
                        # Single Pydantic model
                        response_data = response.model_dump()
                    else:
                        response_data = response
                    self._log_debug("RESPONSE:")
                    self._log_debug(json.dumps(response_data, indent=4, ensure_ascii=False))
                except (json.JSONDecodeError, TypeError) as err:
                    # Log full response without truncation
                    self._log_debug(f"RESPONSE (raw - JSON serialization failed: {err}):")
                    self._log_debug(str(response))
            else:
                self._log_debug("RESPONSE: Empty/None")
            self._log_debug("=" * 80)

    # endregion

    # region Data Transformation Methods
    def load_region_emojis(self) -> Dict[str, Any]:
        """Load region emoji mappings from cache file.

        Returns:
            Dictionary with flag_mappings, characteristic_mappings, and dynamic_mappings.
        """
        emoji_file = self.data_dir / "region_emojis.json"
        if emoji_file.exists():
            with open(emoji_file, "r", encoding="utf-8") as file:
                return json.load(file)
        return {
            "flag_mappings": {},
            "characteristic_mappings": {},
            "dynamic_mappings": {},
        }

    def save_region_emojis(self, emoji_data: Dict[str, Any]) -> None:
        """Save region emoji mappings to cache file.

        Args:
            emoji_data: Dictionary with emoji mappings to save.
        """
        emoji_file = self.data_dir / "region_emojis.json"
        if "_metadata" not in emoji_data:
            emoji_data["_metadata"] = {}
        emoji_data["_metadata"]["last_updated"] = datetime.now().isoformat()
        with open(emoji_file, "w", encoding="utf-8") as file:
            json.dump(emoji_data, file, indent=4, ensure_ascii=False)

    def get_region_emoji_from_gemini(self, region_name: str, used_emojis: set) -> Optional[str]:
        """Ask Gemini for a unique emoji for the region"""
        if not self.client:
            return None

        try:
            # Apply rate limiting if needed
            self._apply_rate_limiting()

            prompt = f"""
            Suggest ONE emoji for the region "{region_name}".

            Rules:
            1. If this region has its own flag emoji (like 🇪🇺 for Europe), use that
            2. Otherwise choose a characteristic symbol for the region
            3. Do NOT use these already used emojis: {', '.join(sorted(used_emojis))}
            4. Do NOT use the globe emoji 🌍 unless specifically for "Worldwide"
            5. Reply with ONLY the emoji, nothing else

            Examples:
            - Europe → 🇪🇺 (has flag)
            - Caribbean → 🌴 (characteristic)
            - Middle East → 🐪 (characteristic)
            """

            response = self.client.models.generate_content(model=self.GEMINI_MODEL, contents=prompt)

            if response and response.text:
                emoji = response.text.strip()
                # Validate it's a single emoji and not already used
                if emoji and len(emoji) <= 4 and emoji not in used_emojis:
                    return emoji

        except (errors.APIError, ValueError, AttributeError) as err:
            if self.verbose:
                self.thread_safe_print(f"    ⚠️ Gemini emoji request failed: {str(err)[:100]}")

        return None

    def get_unique_region_emoji(self, country_name: str) -> str:
        """Get unique emoji for a region with INT flag code"""
        emoji_data = self.load_region_emojis()

        # Check flag mappings first (e.g., Europe → 🇪🇺)
        if country_name in emoji_data.get("flag_mappings", {}):
            flag_emoji = emoji_data["flag_mappings"][country_name]
            # Convert flag code to emoji if it's a code like "DE"
            if len(flag_emoji) == 2 and flag_emoji.isupper():
                # flag conversion
                return "".join(chr(ord(c) + 127397) for c in flag_emoji.upper())
            return flag_emoji

        # Check characteristic mappings (e.g., Caribbean → 🌴)
        if country_name in emoji_data.get("characteristic_mappings", {}):
            return emoji_data["characteristic_mappings"][country_name]

        # Check dynamic mappings (previously generated by Gemini)
        if country_name in emoji_data.get("dynamic_mappings", {}):
            return emoji_data["dynamic_mappings"][country_name]

        # Need to generate new emoji with Gemini
        if self.client:
            # Collect all used emojis to avoid duplicates
            used_emojis = set()
            used_emojis.update(emoji_data.get("flag_mappings", {}).values())
            used_emojis.update(emoji_data.get("characteristic_mappings", {}).values())
            used_emojis.update(emoji_data.get("dynamic_mappings", {}).values())

            # Try to get emoji from Gemini
            new_emoji = self.get_region_emoji_from_gemini(country_name, used_emojis)

            if new_emoji:
                # Save to dynamic mappings
                if "dynamic_mappings" not in emoji_data:
                    emoji_data["dynamic_mappings"] = {}
                emoji_data["dynamic_mappings"][country_name] = new_emoji
                self.save_region_emojis(emoji_data)

                if self.verbose:
                    self.thread_safe_print(f"    🎨 Generated new emoji for {country_name}: {new_emoji}")

                return new_emoji

        # Fallback to pin emoji if all else fails
        if self.verbose:
            self.thread_safe_print(f"    ⚠️ Using fallback emoji for {country_name}")
        return "📍"

    def convert_flag_code_to_emoji(self, flag_code: str, country_name: str = "") -> str:
        """
        Convert a 2-letter country code to flag emoji, with special handling for regions.

        Args:
            flag_code: Two-letter country code (e.g., "US") or "INT" for regions
            country_name: Country/region name for INT codes

        Returns:
            Flag emoji, region emoji, or default world emoji
        """
        # Handle special INT code for regions
        if flag_code == "INT" and country_name:
            return self.get_unique_region_emoji(country_name)

        # Normal country flags
        if not flag_code or len(flag_code) != 2:
            return "🌍"

        # Handle special region codes that should map to flag emojis
        if flag_code == "EU":
            return "🇪🇺"
        if flag_code == "UN":
            return "🇺🇳"

        return "".join(chr(ord(c) + 127397) for c in flag_code.upper())

    @staticmethod
    def extract_image_url(image_url: str) -> str:
        """
        Transform image URL with configured placeholder.

        Args:
            image_url: Original image URL with placeholder

        Returns:
            Transformed image URL with proper operations
        """
        if not image_url:
            return ""
        return image_url.replace("{op}", "e_trim:1:transparent/c_limit,w_800,h_800/bo_5px_solid_rgb:00000000")

    @staticmethod
    def extract_edition_from_url(url: str) -> Optional[str]:
        """
        Extract edition name from URL if it contains 'edition'.

        Args:
            url: Product URL

        Returns:
            Edition name like "Winter Edition" or None if not an edition URL
        """
        if not url or "edition" not in url.lower():
            return None

        # Extract the last part of the URL path
        path_parts = url.rstrip("/").split("/")
        if not path_parts:
            return None

        # Get the last segment (usually the product identifier)
        last_segment = path_parts[-1].lower()

        # Check if it contains 'edition'
        if "edition" not in last_segment:
            return None

        # Clean up the segment and format it
        # Examples: "red-bull-winter-edition" -> "Winter Edition"
        #          "winter-edition-sugarfree" -> "Winter Edition Sugarfree"
        #          "red-bull-blue-edition-juneberry" -> "Blue Edition Juneberry"

        # Remove common prefixes
        cleaned = last_segment.replace("red-bull-", "").replace("products-", "")

        # Split by hyphen and capitalize
        words = cleaned.split("-")
        formatted_words = []

        for word in words:
            if word in ["red", "bull"]:
                continue  # Skip Red Bull prefix
            formatted_words.append(word.capitalize())

        return " ".join(formatted_words)

    @staticmethod
    def fix_product_url(url: str) -> str:
        """Ensure all product URLs use HTTPS.

        Args:
            url: Original URL.

        Returns:
            URL with HTTPS protocol.
        """
        if not url:
            return url

        if url.startswith("http://"):
            return url.replace("http://", "https://", 1)

        return url

    # endregion

    # region Text Cleaning Methods
    def clean_flavor_name(self, flavor: str) -> str:
        """Clean and standardize flavor names.

        Handles duplicate removal, Sugarfree extraction, slash processing,
        and proper formatting based on approved flavors list.

        Args:
            flavor: Original flavor name.

        Returns:
            Cleaned and standardized flavor name.
        """
        if not flavor:
            return flavor

        # CRITICAL: Strip whitespace FIRST before any processing
        original_flavor = flavor.strip()
        # Remove multiple spaces
        original_flavor = " ".join(original_flavor.split())

        # HARDCODED CORRECTION: Curuba -> Curuba-Elderflower
        if original_flavor == "Curuba":
            return "Curuba-Elderflower"

        # CRITICAL CHECK: If it's in the approved list - RETURN IMMEDIATELY, DO NOT MODIFY
        if original_flavor in self.APPROVED_FLAVORS:
            if self.verbose:
                self.thread_safe_print(f"      ✅ Flavor '{original_flavor}' is in APPROVED_FLAVORS - keeping as is")
            return original_flavor

        # Handle special case: "Tropical/Tropical Fruits" -> "Tropical Fruits"
        if "/" in original_flavor:
            # Split by slash and check each part
            parts = [p.strip() for p in original_flavor.split("/")]
            # Check if any part is in approved list
            for part in parts:
                if part in self.APPROVED_FLAVORS:
                    return part
            # If "Tropical Fruits" is one of the parts, use it
            if "Tropical Fruits" in parts:
                return "Tropical Fruits"

        # Handle standalone Sugarfree - ONLY for basic Energy Drink Sugarfree
        # Not for flavored sugarfree editions like "Sugarfree Apricot-Strawberry"
        if original_flavor.lower() in ["sugarfree", "sugar-free", "sugar free"]:
            return "Sugarfree"

        # Remove Sugarfree from combinations to get the actual flavor
        # e.g., "Sugarfree Apricot-Strawberry" → "Apricot-Strawberry"
        for term in ["Sugarfree", "Sugar-free", "Sugar Free"]:
            flavor = flavor.replace(term, "").strip()

        # Clean up any leading/trailing hyphens or spaces after removing Sugarfree
        flavor = flavor.strip().strip("-").strip()

        # Handle slashes - keep only unique parts
        if "/" in flavor:
            parts = [p.strip() for p in flavor.split("/")]
            unique_parts = []
            for part in parts:
                if part and not any(part.lower() == p.lower() for p in unique_parts):
                    unique_parts.append(part)

            flavor = " ".join(unique_parts) if len(unique_parts) > 1 else (unique_parts[0] if unique_parts else flavor)

        # Fix missing spaces around & (e.g., "Woodruff &Pink" -> "Woodruff & Pink")
        flavor = re.sub(r"&(?=[A-Za-z])", "& ", flavor)  # Add space after & if followed by letter
        # Add space before & if preceded by letter
        flavor = re.sub(r"(?<=[A-Za-z])&", " &", flavor)

        # Check if the cleaned version is in approved list
        if flavor.strip() in self.APPROVED_FLAVORS:
            return flavor.strip()

        # FUZZY MATCHING: Try to match against approved flavors by normalizing both sides
        # This handles cases like "Fuji Apple-Ginger" → "Fuji Apple & Ginger"
        normalized_input = flavor.lower().replace("-", "").replace("&", "").replace(" ", "")

        for approved in self.APPROVED_FLAVORS:
            normalized_approved = approved.lower().replace("-", "").replace("&", "").replace(" ", "")
            if normalized_input == normalized_approved:
                # Found a match! Use the approved version with correct formatting
                if self.verbose:
                    self.thread_safe_print(f"      🔄 Fuzzy match: '{flavor}' → '{approved}' (normalized match)")
                return approved

        # WORD-ORDER MATCHING: Match when same words in any order
        # This handles cases like "Apple Fuji-Ginger" → "Fuji Apple & Ginger"
        input_words = self._get_word_set(flavor)
        for approved in self.APPROVED_FLAVORS:
            if input_words == self._get_word_set(approved):
                if self.verbose:
                    self.thread_safe_print(f"      🔄 Word-order match: '{flavor}' → '{approved}'")
                return approved

        # SIMILARITY MATCHING: Use difflib for partial matches
        # Handles cases like "Apple and Ginger" → "Fuji Apple & Ginger"
        best_match = None
        best_ratio = 0.0

        for approved in self.APPROVED_FLAVORS:
            normalized_approved = approved.lower().replace("-", "").replace("&", "").replace(" ", "")
            ratio = SequenceMatcher(None, normalized_input, normalized_approved).ratio()

            if ratio > best_ratio:
                best_ratio = ratio
                best_match = approved

        if best_match and best_ratio >= self.SIMILARITY_THRESHOLD:
            if self.verbose:
                self.thread_safe_print(f"      🔄 Similarity match: '{flavor}' → '{best_match}' " f"(ratio: {best_ratio:.2f})")
            return best_match

        # For flavors not in the approved list, check if it should keep the &
        # Check against all approved flavors that contain &
        approved_with_ampersand = [f for f in self.APPROVED_FLAVORS if "&" in f]
        should_keep_ampersand = False

        for approved in approved_with_ampersand:
            # Check if the current flavor matches this approved pattern (case-insensitive)
            if approved.lower() in flavor.lower() or flavor.lower() in approved.lower():
                should_keep_ampersand = True
                # Try to use the correct formatting from the approved list
                if flavor.lower().replace(" ", "") == approved.lower().replace(" ", ""):
                    flavor = approved
                break

        # For flavors not in the approved list, use hyphen for combinations unless & should be kept
        if not should_keep_ampersand:
            flavor = flavor.replace(" & ", "-")
            flavor = flavor.replace(" and ", "-")
            flavor = flavor.replace(" And ", "-")

        # Remove consecutive duplicate words
        words = flavor.split()
        cleaned_words = []
        prev_word = None
        for word in words:
            if word.lower() != prev_word:
                cleaned_words.append(word)
                prev_word = word.lower()

        result = " ".join(cleaned_words).strip()
        return result if result else original_flavor

    def clean_description(self, description: str) -> str:
        """Clean and standardize descriptions.

        Removes unwanted characters, fixes non-English phrases,
        normalizes capitalization, and standardizes common terms.

        Args:
            description: Original description text.

        Returns:
            Cleaned and standardized description.
        """
        if not description:
            return description

        # Remove unwanted newlines and normalize whitespace
        description = " ".join(description.split())

        # Handle quotes: escaped quotes become apostrophes, remove normal quotes and asterisks
        description = description.replace('"', "").replace("*", "")

        # Remove hyphen in "taste of X - Y" patterns
        # (e.g., "berry - juneberry" -> "berry juneberry")
        # But keep hyphens in compound flavors like "Coconut-Berry"
        description = re.sub(
            r"(\btaste of [^-]+)\s*-\s*([a-z])",
            r"\1 \2",
            description,
            flags=re.IGNORECASE,
        )

        # Replace "sugars" with "sugar" (but keep "taste" as is)
        description = description.replace("sugars", "sugar")
        description = description.replace("Sugars", "Sugar")

        # Replace all variations of "sugar-free" with "sugarfree" (no hyphen)
        description = description.replace("sugar-free", "sugarfree")
        description = description.replace("Sugar-free", "Sugarfree")
        description = description.replace("Sugar-Free", "Sugarfree")
        description = description.replace("sugar free", "sugarfree")
        description = description.replace("Sugar Free", "Sugarfree")

        # Fix "Sugarfree wings" back to "Wings without sugar" (preferred format)
        description = description.replace("Sugarfree wings", "Wings without sugar")
        description = description.replace("sugarfree wings", "wings without sugar")

        # Normalize CAPSLOCK text - handle both full sentences and individual words
        # First, handle CAPSLOCK words within sentences
        words = description.split()
        normalized_words = []

        for word in words:
            # Handle REDBULL -> Red Bull
            if word.upper() == "REDBULL":
                normalized_words.append("Red Bull")
            # Handle all-caps words (including short ones like FOR)
            elif word.isupper() and len(word) > 1:
                if "EDITION" in word.upper():
                    normalized_words.append(word.capitalize())
                else:
                    normalized_words.append(word.capitalize())
            else:
                normalized_words.append(word)

        description = " ".join(normalized_words)

        # Now handle sentence capitalization
        sentences = description.split(". ")
        normalized_sentences = []

        for sentence in sentences:
            # Ensure first letter is uppercase
            if sentence:
                sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
            normalized_sentences.append(sentence)

        # Rejoin sentences
        description = ". ".join(normalized_sentences)

        # Fix percentages
        description = re.sub(r"(\d+)\s*%", r"\1%", description)

        # Remove trailing period if present
        description = description.strip()
        if description.endswith("."):
            description = description[:-1].strip()

        # Final cleanup: remove any remaining multiple spaces
        description = " ".join(description.split())

        # Capitalize flavor names from the approved list
        description = self.capitalize_flavors_in_description(description)

        return description

    def capitalize_flavors_in_description(self, description: str) -> str:
        """Capitalize flavor names in descriptions based on approved list.

        Args:
            description: Description text.

        Returns:
            Description with properly capitalized flavor names.
        """
        if not description:
            return description

        # Process each flavor from the approved list
        for flavor in self.APPROVED_FLAVORS:
            # Skip generic ones that shouldn't be capitalized everywhere
            if flavor in ["Energy Drink", "Sugarfree", "Zero Sugar"]:
                continue

            # For compound flavors like "Coconut-Blueberry", also process individual parts
            if "-" in flavor:
                parts = flavor.split("-")
                for part in parts:
                    # Case-insensitive replacement while preserving the correct capitalization
                    pattern = re.compile(re.escape(part.lower()), re.IGNORECASE)
                    description = pattern.sub(part, description)

            # Also handle special compound flavors with spaces
            if " & " in flavor:
                parts = flavor.split(" & ")
                for part in parts:
                    pattern = re.compile(re.escape(part.lower()), re.IGNORECASE)
                    description = pattern.sub(part, description)
            elif " " in flavor and flavor not in [
                "Energy Drink",
                "Zero Sugar",
                "Tropical Fruits",
                "Forest Fruits",
                "Forest Berry",
                "Dragon Fruit",
                "Cactus Fruit",
                "White Peach",
                "Pear Cinnamon",
                "Exotic Passion Fruit",
            ]:
                # Skip multi-word flavors that are common phrases
                continue

            # Replace the full flavor name (case-insensitive)
            pattern = re.compile(re.escape(flavor.lower()), re.IGNORECASE)
            description = pattern.sub(flavor, description)

        # Handle specific multi-word flavors that should be capitalized
        multi_word_flavors = [
            "Tropical Fruits",
            "Forest Fruits",
            "Forest Berry",
            "Dragon Fruit",
            "Cactus Fruit",
            "White Peach",
            "Pear Cinnamon",
            "Exotic Passion Fruit",
        ]

        for flavor in multi_word_flavors:
            pattern = re.compile(re.escape(flavor.lower()), re.IGNORECASE)
            description = pattern.sub(flavor, description)

        return description

    @staticmethod
    def fix_edition_spacing(text: str) -> str:
        """Fix spacing issues in edition names.

        Ensures proper spacing for editions like "The Summer Edition".

        Args:
            text: Text to fix spacing in.

        Returns:
            Text with corrected spacing.
        """
        if not text:
            return text

        # Fix Edition spacing (Summeredition → Summer Edition, etc.)
        text = re.sub(r"([a-z])edition\b", r"\1 Edition", text, flags=re.IGNORECASE)
        text = re.sub(r"\b([A-Z][a-z]+)edition\b", r"\1 Edition", text)

        # Capitalize standalone "edition" → "Edition"
        text = re.sub(r"\bedition\b", "Edition", text)

        return text

    @staticmethod
    def _sort_final_data(final_data: Dict) -> Dict:
        """Sort countries alphabetically and editions within each country by name.

        Args:
            final_data: The complete final data dictionary.

        Returns:
            New dict with countries sorted alphabetically and editions sorted by name.
        """
        sorted_data = {}
        for country_name in sorted(final_data.keys()):
            country = dict(final_data[country_name])
            if "editions" in country:
                country["editions"] = sorted(country["editions"], key=lambda e: e.get("name", ""))
            sorted_data[country_name] = country
        return sorted_data

    @staticmethod
    def add_description_prefix(editions: List[Dict]) -> List[Dict]:
        """Add appropriate prefixes to flavor descriptions.

        Adds "Wings with the taste of" or "Wings without sugar, with the taste of"
        based on sugarfree status.

        Args:
            editions: List of edition dictionaries.

        Returns:
            Editions with prefixed descriptions.
        """
        for edition in editions:
            name = edition.get("name", "")
            description = edition.get("flavor_description", "")

            if not description:
                continue

            # Only add prefix to Edition products, not Energy Drink/Sugarfree/Zero
            if "Edition" in name:
                # Extract the edition name for the prefix
                edition_name = name
                if edition_name.startswith("The "):
                    edition_name = edition_name[4:]  # Remove "The " from beginning

                # Check if description already contains "The Red Bull [Edition Name]"
                if f"The Red Bull {edition_name}" in description:
                    # Already has correct format, nothing to do
                    continue
                if f"Red Bull The {edition_name}" in description:
                    # Replace "Red Bull The X Edition" with "The Red Bull X Edition"
                    description = description.replace(
                        f"Red Bull The {edition_name}",
                        f"The Red Bull {edition_name}",
                        1,
                    )
                    edition["flavor_description"] = description
                elif f"Red Bull {edition_name}" in description:
                    # Replace "Red Bull X Edition" with "The Red Bull X Edition"
                    # But only if there's no "The" in the few words before "Red Bull"
                    search_text = f"Red Bull {edition_name}"
                    red_bull_pos = description.find(search_text)

                    # Look at up to 20 characters before "Red Bull" to check for "The"
                    if red_bull_pos > 0:
                        start_pos = max(0, red_bull_pos - 20)
                        text_before = description[start_pos:red_bull_pos].lower()

                        # Split into words and check if any is "the"
                        words_before = text_before.split()
                        if any(word == "the" for word in words_before):
                            # There's a "The" in the recent context, don't add another
                            continue

                    # Safe to add "The" prefix
                    description = description.replace(search_text, f"The {search_text}", 1)
                    edition["flavor_description"] = description
                else:
                    # Fallback: Check for any "Red Bull [Something] Edition" pattern
                    # This handles cases like "Red Bull Watermelon Edition" when
                    # name is "The Red Edition"
                    pattern = r"(Red Bull\s+\w+\s+Edition)"
                    match = re.search(pattern, description)
                    if match:
                        found_text = match.group(1)
                        # Check if it already has "The" before it
                        red_bull_pos = description.find(found_text)
                        if red_bull_pos > 0:
                            start_pos = max(0, red_bull_pos - 20)
                            text_before = description[start_pos:red_bull_pos].lower()
                            words_before = text_before.split()
                            if any(word == "the" for word in words_before):
                                # There's already a "The", don't add another
                                continue

                        # Add "The" prefix to the found pattern
                        description = description.replace(found_text, f"The {found_text}", 1)
                        edition["flavor_description"] = description

        return editions

    def fix_edition_name(self, name: str) -> str:
        """Fix and standardize edition names.

        Handles Red Bull Energy Drink variations, removes marketing phrases,
        and ensures proper formatting.

        Args:
            name: Original edition name.

        Returns:
            Fixed and standardized edition name.
        """
        if not name:
            return name

        name = name.strip()
        # Remove multiple spaces
        name = " ".join(name.split())

        # Check if name is just marketing slogan or starts with "Flavor of" or contains "Flavor"
        if (
            any(phrase in name.lower() for phrase in self.MARKETING_PHRASES)
            or name.lower().startswith("flavor of")
            or name.lower().startswith("flavour of")
            or "flavor" in name.lower()
            or "flavour" in name.lower()
        ):
            # If it's something like "Blueberry Flavor", try to extract from pattern
            if "flavor" in name.lower() or "flavour" in name.lower():
                # Log warning
                if self.verbose:
                    self.thread_safe_print(f"      ⚠️ WARNING: Invalid name with 'Flavor': '{name}' - clearing name")
            return ""  # Return empty to be filled based on flavor later

        # Handle Edition names: Remove "Red Bull" and add "The"
        if "Edition" in name:
            # Remove "Red Bull" from Edition names
            name = re.sub(r"\bred\s*bull\s*", "", name, flags=re.IGNORECASE).strip()

        # Remove hyphens before "Sugarfree" in edition names
        name = name.replace("- Sugarfree", " Sugarfree")
        name = name.replace("-Sugarfree", " Sugarfree")

        # Handle Sugarfree and Zero variants (including edition variants)
        if name.lower() in [
            "sugarfree",
            "sugar-free",
            "sugar free",
            "red bull sugarfree",
            "red bull sugarfree edition",
            "sugarfree edition",
        ]:
            return "Energy Drink Sugarfree"
        if name.lower() in [
            "zero",
            "zero sugar",
            "red bull zero",
            "zero edition",
            "the zero edition",
        ]:
            return "Energy Drink Zero"
        if name.lower() in [
            "energy drink",
            "red bull energy drink",
            "the red bull energy drink",
            "original edition",
            "the original edition",
        ]:
            # Always normalize to "Energy Drink"
            return "Energy Drink"

        # Clean up Edition names to only keep "The X Edition" format
        if "Edition" in name:
            # Check if it has Sugarfree at the end and preserve it
            has_sugarfree = name.endswith(" Sugarfree")

            # Extract just "[Color/Season/etc] Edition" and remove extra text after "Edition"
            # This handles both "The Green Edition" and "Green Edition Summer Passion Fruit"
            match = re.search(r"(\w+(?:\s+\w+)*\s+Edition)", name)
            if match:
                edition_part = match.group(1)
                # Ensure it starts with "The"
                if not edition_part.startswith("The "):
                    name = f"The {edition_part}"
                else:
                    name = edition_part

                if has_sugarfree:
                    name = f"{name} Sugarfree"

                if self.verbose and "Summer Passion" in edition_part:
                    self.thread_safe_print("      🔧 Cleaned edition name: removed text after 'Edition'")
            # If no match but contains Edition, just add "The" if missing
            elif not name.startswith("The "):
                name = f"The {name}"

        # Smart validation for edition names
        if "Edition" in name:
            # Extract the edition part without "The" and "Sugarfree"
            edition_core = name.replace("The ", "").replace(" Sugarfree", "")

            # Check if it's in approved list - if yes, trust it
            if edition_core in self.APPROVED_EDITIONS:
                if self.verbose:
                    self.thread_safe_print(f"      ✅ Edition '{edition_core}' found in approved list")
            else:
                # Check if it follows valid pattern "Word(s) Edition"
                if re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Edition$", edition_core):
                    if self.verbose:
                        self.thread_safe_print(f"      ⚠️  New edition detected: '{edition_core}' (not in approved list but follows valid pattern)")
                else:
                    if self.verbose:
                        self.thread_safe_print(f"      🚨 Suspicious edition name: '{edition_core}' - doesn't match approved list or valid pattern")

        # Final cleanup: remove any multiple spaces
        name = " ".join(name.split())

        return name

    # endregion

    # region Corrections Methods
    def apply_corrections(self, edition: Dict, graphql_id: str) -> Dict:
        """
        Apply manual corrections to an edition before processing.

        Args:
            edition: Edition dictionary to correct
            graphql_id: GraphQL ID for matching corrections

        Returns:
            Corrected edition dictionary
        """
        # First normalize Açai variations in raw data before AI processing
        # This ensures consistent handling regardless of source language
        for field in ["_raw_flavor", "_standfirst"]:
            if field in edition and edition[field]:
                # Normalize all Açai variations to 'Acai'
                # This handles: açai, açaí, açaï, açaì, Açai, Açaí, etc.
                text_normalized = unicodedata.normalize("NFD", edition[field])

                # Pattern matches all variations of açai with different diacritics in NFD form
                # After NFD normalization: ç becomes c + \u0327, í becomes i + \u0301
                pattern = r"[aA][çc]\u0327?[aA][iI]\u0301?"
                text_acai_fix = re.sub(pattern, "Acai", text_normalized, flags=re.IGNORECASE)

                # Normalize back to composed form
                edition[field] = unicodedata.normalize("NFC", text_acai_fix)

                if self.verbose and "acai" in edition[field].lower():
                    self.thread_safe_print(f"      🔧 Normalized Açai in {field}: {edition[field][:50]}...")

        if not self.corrections:
            return edition

        # Map user-friendly field names to actual raw field names
        # This allows corrections.json to use logical field names
        field_mapping = {"flavor": "_raw_flavor", "flavor_description": "_standfirst"}

        # Extract the short ID format for matching
        # Convert "rrn:content:energy-drinks:UUID:locale" to "UUID:locale"
        short_id = graphql_id
        if graphql_id.startswith("rrn:content:energy-drinks:"):
            parts = graphql_id.split(":")
            if len(parts) >= 5:
                # Get UUID and locale parts
                short_id = f"{parts[3]}:{parts[4]}"

        applied = 0
        for correction in self.corrections:
            correction_id = correction.get("id")

            # Extract UUID from short_id for partial matching
            uuid_only = short_id.split(":")[0] if ":" in short_id else short_id

            # Match if:
            # 1. Exact match with full IDs (e.g., "UUID:de-DE"), OR
            # 2. correction_id has no locale AND matches UUID part (e.g., "UUID")
            if correction_id in (graphql_id, short_id) or (":" not in correction_id and correction_id == uuid_only):
                field = correction.get("field")
                search = correction.get("search")
                replace = correction.get("replace")

                # Map the field name if needed
                actual_field = field_mapping.get(field, field)

                # Mark this correction as checked
                correction_key = f"{correction_id}:{field}:{search}"
                if correction_key not in self.corrections_tracking:
                    self.corrections_tracking[correction_key] = {
                        "id": correction_id,
                        "field": field,
                        "search": search,
                        "replace": replace,
                        "applied": False,
                        "attempted": True,
                    }

                if actual_field in edition and search and replace:
                    original = edition[actual_field]
                    match_mode = correction.get("match_mode", "exact")

                    if match_mode == "partial":
                        # Substring-Replace: search muss im Feldwert enthalten sein
                        if search.lower() not in original.lower():
                            self.changelog["corrections_failed"].append(
                                {
                                    "id": correction_id,
                                    "field": field,
                                    "search": search,
                                    "reason": "Text not found in field (partial match)",
                                }
                            )
                            if self.verbose:
                                self.thread_safe_print(
                                    f"      ⚠️ Correction not applied for {correction_id}: "
                                    f"'{search}' not found in {field} (partial match, checking {actual_field})"
                                )
                        else:
                            edition[actual_field] = original.replace(search, replace)

                            # Track that this field was corrected (especially important for flavor)
                            if "_corrected_fields" not in edition:
                                edition["_corrected_fields"] = set()
                            # Track the logical field name (e.g., "flavor" not "_raw_flavor")
                            edition["_corrected_fields"].add(field)

                            # If we corrected _raw_flavor, also set the flavor field immediately
                            if field == "flavor" and actual_field == "_raw_flavor":
                                edition["flavor"] = edition[actual_field]

                            # If we corrected _standfirst, also set the
                            # flavor_description field immediately
                            if field == "flavor_description" and actual_field == "_standfirst":
                                edition["flavor_description"] = edition[actual_field]

                            applied += 1
                            self.corrections_tracking[correction_key]["applied"] = True
                            self.changelog["corrections_applied"].append(
                                {
                                    "id": correction_id,
                                    "field": field,  # Log the user-friendly field name
                                    "search": search,
                                    "replace": replace,
                                }
                            )
                            if self.verbose:
                                self.thread_safe_print(f"      🔧 Applied correction: {field} - " f"'{search}' → '{replace}'")
                    elif original.strip().lower() == search.strip().lower():
                        # EXACT MATCH: Compare case-insensitively but match entire field value
                        # This prevents partial matches (e.g., "Peach" matching inside "White Peach")
                        edition[actual_field] = replace

                        # Track that this field was corrected (especially important for flavor)
                        if "_corrected_fields" not in edition:
                            edition["_corrected_fields"] = set()
                        # Track the logical field name (e.g., "flavor" not "_raw_flavor")
                        edition["_corrected_fields"].add(field)

                        # If we corrected _raw_flavor, also set the flavor field immediately
                        if field == "flavor" and actual_field == "_raw_flavor":
                            edition["flavor"] = edition[actual_field]

                        # If we corrected _standfirst, also set the
                        # flavor_description field immediately
                        if field == "flavor_description" and actual_field == "_standfirst":
                            edition["flavor_description"] = edition[actual_field]

                        applied += 1
                        self.corrections_tracking[correction_key]["applied"] = True
                        self.changelog["corrections_applied"].append(
                            {
                                "id": correction_id,
                                "field": field,  # Log the user-friendly field name
                                "search": search,
                                "replace": replace,
                            }
                        )
                        if self.verbose:
                            self.thread_safe_print(f"      🔧 Applied correction: {field} - " f"'{search}' → '{replace}'")
                    else:
                        # Correction couldn't be applied - text not found
                        self.changelog["corrections_failed"].append(
                            {
                                "id": correction_id,
                                "field": field,  # Log the user-friendly field name
                                "search": search,
                                "reason": "Text not found in field",
                            }
                        )
                        if self.verbose:
                            self.thread_safe_print(
                                f"      ⚠️ Correction not applied for {correction_id}: " f"'{search}' not found in {field} (checking {actual_field})"
                            )

        if applied > 0 and self.verbose:
            self.thread_safe_print(f"      ✅ Applied {applied} corrections for {short_id}")

        return edition

    def _find_edition_by_id(self, edition_id: str) -> Optional[Dict]:
        """Load a raw file on-demand and return the edition matching the given ID.

        Derives the raw filename from the locale embedded in edition_id.
        Example: 'dad80e1f-...:en-GB' → 'data/raw/gb-en.json'

        Args:
            edition_id: Full edition ID in the format 'UUID:locale' (e.g. 'dad80e1f-...:en-GB').

        Returns:
            Edition dictionary from raw data if found, otherwise None.
        """
        if ":" not in edition_id:
            self.logger.warning("ID-Mapping: Cannot derive locale from edition_id '%s' (no colon)", edition_id)
            return None

        locale = edition_id.split(":")[-1]  # e.g. "en-GB"
        locale_parts = locale.split("-")
        if len(locale_parts) != 2:
            self.logger.warning("ID-Mapping: Unexpected locale format '%s' in edition_id '%s'", locale, edition_id)
            return None

        lang, country = locale_parts  # e.g. "en", "GB"
        filename = f"{country.lower()}-{lang.lower()}.json"
        raw_file = self.data_dir / "raw" / filename

        if not raw_file.exists():
            self.logger.warning("ID-Mapping: Raw file '%s' for source_id '%s' not found", filename, edition_id)
            return None

        with raw_file.open(encoding="utf-8") as f:
            raw_data = json.load(f)

        uuid_part = edition_id.split(":")[0]
        for edition in raw_data.get("editions", []):
            raw_id = edition.get("id", "")
            # Support both "UUID:locale" and full "rrn:content:energy-drinks:UUID:locale" format
            if raw_id != edition_id and not raw_id.endswith(f":{uuid_part}:{locale}"):
                continue

            # Normalize raw file structure into the _raw_flavor/_standfirst keys used by the processor
            graphql_data = edition.get("graphql_data", {}).get("data", {})
            edition["_raw_flavor"] = graphql_data.get("flavour", "")
            edition["_standfirst"] = graphql_data.get("standfirst", "")
            return edition

        self.logger.warning("ID-Mapping: Edition '%s' not found in '%s'", edition_id, filename)
        return None

    def _apply_id_mappings(self, editions: List[Dict]) -> List[Dict]:
        """Replace raw fields of target editions with data from a source locale.

        Called after apply_corrections but before Gemini processing.
        Useful when the API for a country returns wrong flavor data, but the
        same edition in another country/locale has correct data.

        Args:
            editions: List of raw edition dicts for the current country.

        Returns:
            The editions list with mapped fields applied in-place.
        """
        if not self.id_mappings:
            return editions

        field_mapping = {"flavor": "_raw_flavor", "flavor_description": "_standfirst"}

        for mapping in self.id_mappings:
            source_id = mapping.get("source_id", "")
            target_id = mapping.get("target_id", "")
            fields = mapping.get("fields", ["flavor", "flavor_description"])

            if not source_id or not target_id:
                self.logger.warning("ID-Mapping: Skipping entry with missing source_id or target_id")
                continue

            # Extract the UUID:locale portion from both IDs for matching
            target_uuid_locale = target_id if ":" in target_id else None

            matched = False
            for edition in editions:
                raw_id = edition.get("_graphql_id", "")
                # Normalize to "UUID:locale" for comparison
                if raw_id.startswith("rrn:content:energy-drinks:"):
                    parts = raw_id.split(":")
                    short_id = f"{parts[3]}:{parts[4]}" if len(parts) >= 5 else raw_id
                else:
                    short_id = raw_id

                if short_id != target_uuid_locale and raw_id != target_id:
                    continue

                matched = True
                source_data = self._find_edition_by_id(source_id)
                if not source_data:
                    self.changelog["id_mappings_failed"].append(
                        {
                            "source_id": source_id,
                            "target_id": target_id,
                            "fields": fields,
                        }
                    )
                    break

                for field in fields:
                    raw_field = field_mapping.get(field, field)
                    source_raw_id = source_data.get("id", "")
                    # Normalize source raw_field lookup (same rrn format possible)
                    src_value = source_data.get(raw_field)
                    if src_value is None:
                        self.logger.warning("ID-Mapping: Field '%s' not found in source edition '%s'", raw_field, source_id)
                        continue

                    old_value = edition.get(raw_field, "")
                    edition[raw_field] = src_value

                    # Keep the logical field in sync too (as apply_corrections does)
                    if field == "flavor":
                        edition["flavor"] = src_value
                    elif field == "flavor_description":
                        edition["flavor_description"] = src_value

                    self.logger.info(
                        "ID-Mapping applied: %s.%s '%s' → '%s' (from %s / %s)",
                        target_id,
                        raw_field,
                        old_value,
                        src_value,
                        source_id,
                        source_raw_id,
                    )
                    if self.verbose:
                        self.thread_safe_print(f"      🔀 ID-Mapping applied: {field} '{old_value}' → '{src_value}' " f"(source: {source_id})")
                break

            if not matched and self.debug:
                self.logger.debug("ID-Mapping: target_id '%s' not found in current country editions", target_id)

        return editions

    # endregion

    # region AI Processing Methods
    def _apply_rate_limiting(self) -> None:
        """Apply rate limiting before API calls.

        Enforces minimum delay between Gemini API requests to avoid quota issues.
        RPM delay is skipped when processing single country, but daily limit is always enforced.

        Raises:
            RuntimeError: When the daily API request limit is exceeded.
        """
        self.changelog["api_calls_made"] += 1

        # Daily limit check – always enforced, even for single country
        if self.changelog["api_calls_made"] > self.MAX_REQUESTS_PER_DAY:
            if not self._daily_limit_reached:
                self._daily_limit_reached = True
                self.logger.warning(
                    "⚠️  Daily API limit reached (%d/%d requests). " "Stopping – results so far will be saved.",
                    self.MAX_REQUESTS_PER_DAY,
                    self.MAX_REQUESTS_PER_DAY,
                )
            raise RuntimeError(f"Daily API limit of {self.MAX_REQUESTS_PER_DAY} requests reached")

        # Warn at 80% of daily limit
        warning_threshold = int(self.MAX_REQUESTS_PER_DAY * 0.8)
        if self.changelog["api_calls_made"] == warning_threshold:
            self.logger.warning(
                "⚠️  API limit warning: %d/%d daily requests used",
                self.changelog["api_calls_made"],
                self.MAX_REQUESTS_PER_DAY,
            )

        # Skip RPM delay when processing a single country
        if self.single_country:
            return

        with self.api_call_lock:
            if self.last_api_call_time:
                time_since_last_call = time.time() - self.last_api_call_time
                if time_since_last_call < self.MIN_DELAY_BETWEEN_REQUESTS:
                    sleep_time = self.MIN_DELAY_BETWEEN_REQUESTS - time_since_last_call
                    if self.verbose:
                        self.thread_safe_print(f"    ⏱️ Rate limiting: waiting {sleep_time:.1f}s " "before next API call")
                    time.sleep(sleep_time)
            self.last_api_call_time = time.time()

    def _gemini_step_1_translate(
        self,
        editions: List[Dict],
        country_name: str,
        source_language: str = "Unknown",
        retry_count: int = 0,
    ) -> List[TranslatedEdition]:
        """Step 1: Translate edition data to English using Gemini.

        Translates product names, flavors, and descriptions to English.
        Handles special cases for Brazil and Romania (keeps original names).

        Args:
            editions: List of editions to translate.
            country_name: Country name for context.
            source_language: Source language name for translation context.
            retry_count: Current retry attempt (for internal use).

        Returns:
            List of TranslatedEdition objects.

        Raises:
            SystemExit: If API quota is exceeded.
            RuntimeError: If translation fails after retries.
        """
        if not self.client or not editions:
            return []

        if self.verbose:
            self.thread_safe_print(f"    🌍 Step 1: Translating {len(editions)} editions " f"for {country_name}...")

        editions_for_ai = []
        for i, edition in enumerate(editions):
            # Try to extract edition name from URL if it contains
            # "edition"
            product_url = edition.get("product_url", "")
            edition_from_url = self.extract_edition_from_url(product_url)

            # Use edition name from URL if available, otherwise use alt_text
            alt_text_value = edition_from_url if edition_from_url else edition.get("_raw_alt_text", edition.get("alt_text", "")).strip()

            editions_for_ai.append(
                {
                    "edition_id": edition.get("_graphql_id", f"edition_{i}"),  # Use GraphQL ID or fallback
                    "name": edition.get("name", "").strip(),
                    "alt_text": alt_text_value,
                    "original_flavor": edition.get("_raw_flavor", "").strip(),
                    "original_description": edition.get("_standfirst", "").strip(),
                    "product_url": product_url,  # Include URL for sugar-free detection
                }
            )

        # Special handling for countries that keep original language names
        special_instruction = ""
        if country_name == "Brazil":
            special_instruction = "- SPECIAL RULE FOR BRAZIL: Do NOT translate the 'name' field - keep it in Portuguese.\n"

        # Convert approved editions list to string for prompt
        approved_editions_str = json.dumps(self.APPROVED_EDITIONS, indent=4)

        # Build language context string
        language_context = f" ({source_language})" if source_language != "Unknown" else ""

        prompt = f"""
        Translate Red Bull edition data from {country_name}{language_context} to English.

        Return for each:
        1. edition_id: PRESERVE EXACTLY
        2. name: English name (check alt_text for hints!)
        3. flavor: FLAVOR NAME ONLY
        4. description: Full marketing description

        APPROVED EDITIONS (use these when matching):
        {approved_editions_str}

        NAME RULES:
        - Extract edition name from alt_text/title, NEVER derive from flavor !
        - Match to APPROVED EDITIONS list when possible
        - "Red Bull X Edition" → "The X Edition" (where X is from APPROVED list)
        - Exception: "Red Bull Sugarfree Edition" → "Energy Drink Sugarfree"
        - CRITICAL: NEVER create "The Original Edition" → use "Energy Drink"
        - CRITICAL: NEVER create "The Zero Edition" → use "Energy Drink Zero"
        - CRITICAL: NEVER use the flavor to create the edition name
        - Keep SHORT: "The X Edition" (not with flavor)
        - Sugarfree editions: "The X Edition Sugarfree"
        - Common edition names: Blue, Green, Pink, Purple, Red, Yellow, Summer, Winter, Sea Blue, Apricot, Berry
        - Examples:
          • RIGHT: alt_text: "Red Bull Sea Blue Edition" → name: "The Sea Blue Edition"
          • RIGHT: alt_text: "Red Bull Apricot Edition" → name: "The Apricot Edition"
          • WRONG: flavor: "Strawberry-Apricot" → name "The Strawberry Edition" - IGNORE the flavor for name finding !
          • WRONG: flavor: "Strawberry-Apricot" → name "The Apricot and Strawberry Edition" - IGNORE the flavor for name finding !
          • RIGHT: alt_text "Red Bull Apricot" → name "The Apricot Edition"
          • RIGHT: alt_text: "Red Bull Green Edition" → name: "The Green Edition"

        TRANSLATION:
        {special_instruction}- Translate EVERYTHING to English
        - NO ALL CAPS (except acronyms)
        - Fix spacing: "Summeredition" → "Summer Edition"
        - Prefer the word "taste", do not active replace with "flavor"
        - Cleanup the sentence to be human readable, remove e.g. * (asterisk) or - (Dash)
        - Covert the "Wiiings" (multiple "i") marketing word to normal "Wings"
        - CRITICAL: PRESERVE ALL NUMBERS AND PERCENTAGES (e.g., "100% Wings", "0% Sugar") - DO NOT remove them!

        CRITICAL TRANSLATION RULES - DO NOT INTERPRET OR IMPROVE:
        - Translate EXACTLY as written - DO NOT interpret or improve flavor names
        - DO NOT convert flavor names to English equivalents unless explicitly listed below
        - Keep original spelling and terminology when possible
        - Do NOT extend/fill the description with generic informations
        - Do NOT remove/replace flavor's in the description, keep them as provided
        - Example: "Maracujá" → "Maracuja" (NOT "Passion Fruit"!)
        - Example: "Curuba" → "Curuba-Elderflower" (NOT "cuban" or any other interpretation!)

        EXAMPLES:
        - name: "The Red Edition", flavor: "Watermelon"
        - name: "Energy Drink Sugarfree", flavor: "Sugarfree"

        FLAVOR TRANSLATIONS (ONLY these specific translations allowed):
        - Waldmeister → Woodruff
        - Waldbeeren → Forest Berry
        - Waldfrüchte → Forest Fruits
        - Forest fruit (English, singular, any case) → Forest Fruits
        - Forest fruits (English, any case) → Forest Fruits
        - Ice/Ľadová → Iced
        - Maracujá → Maracuja (NOT Passion Fruit!)
        - Curuba → Curuba-Elderflower (NOT cuban!)
        - Pompelmoes (Dutch) → Pink Grapefruit (NOT Pomelo!)
        - Keep order: "vanilka-čučoriedky" → "Vanilla-Blueberry"

        CRITICAL: Keep "Forest Berry" and "Forest Fruits" DISTINCT:
        - German "Waldbeeren" → "Forest Berry"
        - English "Forest fruit/fruits" → "Forest Fruits"
        - These are DIFFERENT flavors, do NOT interchange them!

        KEY PHRASES:
        - Alas/Asas/Flügel → Wings
        - sin azúcar/sem açúcar/ohne Zucker → without sugar
        - Keep "Red Bull" as-is

        Return ONLY JSON array.

        Input:
        {json.dumps(editions_for_ai, indent=4)}
        """

        try:
            # Apply rate limiting
            self._apply_rate_limiting()

            # Log the request in verbose mode
            self._log_api_request(country_name, "Translation", prompt)

            if self.verbose:
                self.thread_safe_print(f"    - Sending translation request to Gemini " f"({len(prompt)} chars)...")

            start_time = time.time()

            response = self.client.models.generate_content(
                model=self.GEMINI_MODEL,
                contents=prompt,
                config=GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=list[TranslatedEdition],
                ),
            )

            api_time = time.time() - start_time

            if self.verbose:
                self.thread_safe_print(f"    - Gemini translation response time: {api_time:.2f}s")

            translated_editions = response.parsed

            # Check for None response which causes unclear "NoneType has no len()" errors
            if translated_editions is None:
                raise RuntimeError(
                    "Gemini API returned None for translation request (empty response). "
                    "This usually indicates API parsing failed or returned invalid data."
                )

            # Log the response in verbose mode
            self._log_api_response(country_name, "Translation", translated_editions, api_time)

            if self.debug:
                self.thread_safe_print(f"    🔍 {country_name}: Translated {len(translated_editions)} editions " "(see debug log for details)")

            if self.verbose:
                self.thread_safe_print("    ✅ Step 1: Successfully translated " f"{len(translated_editions)} editions for {country_name}.")
            return translated_editions

        except errors.APIError as err:
            # Handle Gemini API errors with proper error codes

            # Log full error details in debug mode
            if self.verbose:
                self.thread_safe_print(f"    🐛 Debug: APIError caught - Code: {err.code}")
                self.thread_safe_print(f"    🐛 Debug: Error message: {err.message[:300]}")

            # Check error message to distinguish quota/billing from rate limit
            error_msg_lower = err.message.lower()
            is_quota_error = "quota" in error_msg_lower or "billing" in error_msg_lower
            is_rate_limit = err.code == 429 and not is_quota_error

            # Check for quota exceeded error (429)
            if err.code == 429:
                # If we have retries left, and it's a true rate limit (not quota), wait and retry
                if retry_count < 3 and is_rate_limit:
                    wait_time = (retry_count + 1) * 10  # 10, 20, 30 seconds
                    self.thread_safe_print(f"    ⚠️ Rate limit hit for {country_name}, " f"waiting {wait_time}s before retry {retry_count + 1}/3")
                    time.sleep(wait_time)
                    return self._gemini_step_1_translate(editions, country_name, source_language, retry_count + 1)

                # It's either a quota/billing error or we've exhausted retries
                error_type = "QUOTA/BILLING ERROR" if is_quota_error else "RATE LIMIT EXCEEDED"
                self.thread_safe_print(f"    ❌ {error_type}: Translation failed for " f"{country_name} after {retry_count + 1} attempts")
                self.thread_safe_print(f"       Error: {err.message[:200]}")

                if is_quota_error:
                    self.thread_safe_print("    ⛔ Aborting processing to preserve cache. Please check your Gemini API quota and billing.")
                else:
                    self.thread_safe_print("    ⛔ Aborting processing to preserve cache. Rate limits exhausted after 3 retries.")

                raise SystemExit(f"Gemini API {error_type}. Aborting to preserve cache.") from err

            # Check for internal server error (500)
            if err.code == 500:
                if retry_count < 3:
                    wait_time = (retry_count + 1) * 2  # 2, 4, 6 seconds
                    self.thread_safe_print(f"    ⚠️ Internal error for {country_name}, " f"retrying in {wait_time}s (attempt {retry_count + 1}/3)")
                    time.sleep(wait_time)
                    return self._gemini_step_1_translate(editions, country_name, source_language, retry_count + 1)

                self.thread_safe_print(f"    ❌ Translation failed for {country_name} " f"after 3 retries: {err.message[:150]}")
                self.thread_safe_print("       Cannot proceed without AI - critical error")
                raise RuntimeError(f"Translation failed for {country_name} after 3 retries: " f"err.message") from err

            # Check for expired API key - abort immediately
            error_msg = str(err.message) if hasattr(err, "message") else str(err)
            if "expired" in error_msg.lower():
                self._abort_flag = True  # Signal all threads to stop
                self.thread_safe_print(f"    ❌ API KEY EXPIRED for {country_name}")
                self.thread_safe_print("       Please renew your Gemini API key in .env file")
                self.thread_safe_print("       Aborting all processing - retries won't help with expired keys")
                raise RuntimeError("API key expired - aborting all processing") from err

            # Check for bad request error (400) - might be too large input
            if err.code == 400:
                self.thread_safe_print(f"    ❌ Bad request for {country_name}: {err.message[:150]}")
                self.thread_safe_print("       This might be due to too many editions or content size")
                raise RuntimeError(f"Translation failed for {country_name}: Bad request - err.message") from err

            # Other API errors
            self.thread_safe_print(f"    ❌ Translation failed for {country_name}: " f"Code {err.code} - {err.message[:150]}")
            raise RuntimeError(f"Translation failed for {country_name}: {err.code} - {err.message}") from err

        except (ValueError, AttributeError, KeyError, TypeError) as err:
            # Handle non-API errors
            self.thread_safe_print(f"    ❌ Unexpected error for {country_name}: {str(err)[:150]}")
            raise RuntimeError(f"Translation failed for {country_name}: {str(err)}") from err

    def _gemini_step_2_normalize(
        self,
        translated_editions: List[TranslatedEdition],
        country_name: str,
        retry_count: int = 0,
    ) -> List[Edition]:
        """Step 2: Normalize translated data using Gemini.

        Applies standardization rules to flavor names, descriptions,
        and determines sugarfree status based on approved lists.

        Args:
            translated_editions: List of translated editions.
            country_name: Country name for context.
            retry_count: Current retry attempt (for internal use).

        Returns:
            List of normalized Edition objects.

        Raises:
            SystemExit: If API quota is exceeded.
            RuntimeError: If normalization fails after retries.
        """
        if not self.client or not translated_editions:
            return []

        # Add verbose logging for normalization start
        if self.debug:
            self.thread_safe_print(f"    🔧 {country_name}: Normalizing {len(translated_editions)} editions...")
        elif self.verbose:
            self.thread_safe_print(f"    ✨ Step 2: Normalizing {len(translated_editions)} " f"translated editions for {country_name}...")

        # Convert approved lists to strings for prompt
        approved_flavors_str = json.dumps(self.APPROVED_FLAVORS, indent=4)
        approved_editions_str = json.dumps(self.APPROVED_EDITIONS, indent=4)

        prompt = f"""
        Normalize Red Bull edition data from {country_name}.

        Return for each edition:
        1. edition_id: PRESERVE EXACTLY - DO NOT truncate, modify, or omit ANY part
           CRITICAL: Copy the ENTIRE edition_id string character-by-character
           Example: "rrn:content:energy-drinks:c17571d1-556a-4cc7-ad79-8b78ced3adbc:sk-SK"
           → MUST return exactly: "rrn:content:energy-drinks:c17571d1-556a-4cc7-ad79-8b78ced3adbc:sk-SK"
        2. name: PRESERVE the original edition name from input (Pink Edition stays Pink Edition, White Edition stays White Edition, etc.)
        3. flavor: Match to approved list or create normalized
        4. flavor_description: Translated description
        5. sugarfree: Determine based on these indicators:
           - TRUE if name contains "Sugarfree" or "Zero"
           - TRUE if URL contains "sugarfree"
           - TRUE if description contains: "ohne Zucker", "without sugar", "sin azúcar", "sem açúcar", "sans sucre"
           - TRUE if description says "also without sugar" or similar phrases indicating sugar-free
           - FALSE otherwise

        APPROVED EDITIONS (for validation):
        {approved_editions_str}

        APPROVED FLAVORS:
        {approved_flavors_str}

        CRITICAL EDITION ID PRESERVATION:
        - The edition_id field is MANDATORY and MUST be preserved EXACTLY as provided
        - DO NOT truncate long IDs (some IDs are 70+ characters - preserve ALL of them)
        - DO NOT omit any part of the ID (colons, hyphens, country codes, UUIDs)
        - If you receive 9 editions, you MUST return exactly 9 editions with their EXACT IDs
        - Failure to preserve edition_id will cause data corruption

        CRITICAL EDITION NAME RULES - DO NOT MODIFY EDITION NAMES:
        - PRESERVE the original edition name from the 'name' field EXACTLY as provided
        - NEVER change "Blue Edition" to "Blueberry Edition" even if flavor is Blueberry
        - NEVER change "Pink Edition" to another name even if flavor is Forest Berry or Raspberry
        - NEVER change "White Edition" to another name even if flavor is Coconut
        - NEVER change "Green Edition" to another name even if flavor is Dragon Fruit
        - NEVER derive the edition name from the flavor field
        - DO NOT translate or localize edition color names (Blue stays Blue, not Blueberry)
        - Only add "The" prefix if missing: "Blue Edition" → "The Blue Edition"
        - DO NOT change the core edition name: "Blue" must stay "Blue", not "Blueberry"
        - If the original name is in APPROVED EDITIONS list, keep it exactly

        CRITICAL TRANSLATION PRIORITY:
        - ALWAYS prioritize the direct translation of the 'flavor' field over the description
        - Do NOT derive the flavor from the description text
        - Example: If flavor says "Waldbeere", translate to "Forest Berry" (NOT "Raspberry" even if description mentions raspberry)
        - The flavor field is the SOURCE OF TRUTH, description is only supplementary

        FLAVOR HANDLING:
        - Return the translated flavor AS-IS - normalization is handled by post-processing code
        - DO NOT try to match flavors to an approved list - the code will normalize them
        - SPECIAL RULES (these distinctions are important for accurate translation!):
           - curuba ≠ cuban (different things!)
           - Forest Berry ≠ Forest Fruits (keep distinct)
           - Pomelo ≠ Grapefruit & Blossom (keep distinct)
           - Waldbeere/Waldbeeren → Forest Berry (NOT Raspberry)
           - Gletschereis → Glacier Ice (German for Glacier Ice)
           - Yellow + tropical → "Tropical Fruits"
           - Green + cactus/dragon → "Cactus Fruit"/"Dragon Fruit"
           - Ice → Iced
           - Keep "Fruits" plural

        Input:
        {json.dumps([e.model_dump() for e in translated_editions], indent=4)}

        Return ONLY JSON array.
        """

        try:
            # Apply rate limiting
            self._apply_rate_limiting()

            if self.verbose:
                self.thread_safe_print(f"    - Sending normalization request to Gemini " f"({len(prompt)} chars)...")

            start_time = time.time()

            try:
                response = self.client.models.generate_content(
                    model=self.GEMINI_MODEL,
                    contents=prompt,
                    config=GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=list[Edition],
                    ),
                )
            except errors.APIError as err:
                # Log full error details in debug mode
                if self.verbose:
                    self.thread_safe_print(f"    🐛 Debug: APIError caught - Code: {err.code}")
                    self.thread_safe_print(f"    🐛 Debug: Error message: {err.message[:300]}")

                # Check error message to distinguish quota/billing from rate limit
                error_msg_lower = err.message.lower()
                is_quota_error = "quota" in error_msg_lower or "billing" in error_msg_lower
                is_rate_limit = err.code == 429 and not is_quota_error

                # Handle specific API error codes
                if err.code == 429:  # Rate limit or quota
                    if retry_count < 3 and is_rate_limit:
                        wait_time = (retry_count + 1) * 10  # 10, 20, 30 seconds
                        self.thread_safe_print(f"    ⚠️ Rate limit hit for {country_name}, " f"waiting {wait_time}s before retry {retry_count + 1}/3")
                        time.sleep(wait_time)
                        return self._gemini_step_2_normalize(translated_editions, country_name, retry_count + 1)

                    # It's either a quota/billing error or we've exhausted retries
                    error_type = "QUOTA/BILLING ERROR" if is_quota_error else "RATE LIMIT EXCEEDED"
                    self.thread_safe_print(f"    ❌ {error_type}: Normalization failed for " f"{country_name} after {retry_count + 1} attempts")
                    self.thread_safe_print(f"       Error: {err.message[:200]}")

                    if is_quota_error:
                        self.thread_safe_print("    ⛔ Aborting processing to preserve cache. Please check your Gemini API quota and billing.")
                    else:
                        self.thread_safe_print("    ⛔ Aborting processing to preserve cache. Rate limits exhausted after 3 retries.")

                    raise SystemExit(f"Gemini API {error_type}. Aborting to preserve cache.") from err
                if err.code == 500:  # Internal server error
                    if retry_count < 3:
                        wait_time = (retry_count + 1) * 2  # 2, 4, 6 seconds
                        self.thread_safe_print(f"    ⚠️ Server error for {country_name}, " f"retrying in {wait_time}s (attempt {retry_count + 1}/3)")
                        time.sleep(wait_time)
                        return self._gemini_step_2_normalize(translated_editions, country_name, retry_count + 1)

                    self.thread_safe_print(f"    ❌ Server error for {country_name} after 3 retries: " f"{err.message[:150]}")
                    raise

                # Other API errors - retry with backoff
                if retry_count < 3:
                    wait_time = (retry_count + 1) * 3  # 3, 6, 9 seconds
                    self.thread_safe_print(f"    ⚠️ API error {err.code} for {country_name}: {err.message[:100]}")
                    self.thread_safe_print(f"       Retrying in {wait_time}s (attempt {retry_count + 1}/3)")
                    time.sleep(wait_time)
                    return self._gemini_step_2_normalize(translated_editions, country_name, retry_count + 1)

                self.thread_safe_print(f"    ❌ API error {err.code} for {country_name} after 3 retries: " f"{err.message[:150]}")
                raise
            except (ValueError, AttributeError, KeyError, TypeError) as err:
                # Non-API errors (network, parsing, etc.)
                if retry_count < 3:
                    wait_time = (retry_count + 1) * 2  # 2, 4, 6 seconds
                    self.thread_safe_print(f"    ⚠️ Unexpected error for {country_name}: {str(err)[:100]}")
                    self.thread_safe_print(f"       Retrying in {wait_time}s (attempt {retry_count + 1}/3)")
                    time.sleep(wait_time)
                    return self._gemini_step_2_normalize(translated_editions, country_name, retry_count + 1)

                self.thread_safe_print(f"    ❌ Failed for {country_name} after 3 retries: {str(err)[:150]}")
                raise

            if self.verbose:
                api_time = time.time() - start_time
                self.thread_safe_print(f"    - Gemini normalization response time: {api_time:.2f}s")

            normalized_editions = response.parsed

            # Check if response is valid and retry if empty
            if not normalized_editions:
                if retry_count < 3:
                    wait_time = (retry_count + 1) * 2  # 2, 4, 6 seconds
                    self.thread_safe_print(
                        f"    ⚠️ Gemini returned empty response for {country_name}, " f"retrying in {wait_time}s (attempt {retry_count + 1}/3)"
                    )
                    if hasattr(response, "text"):
                        self.thread_safe_print(f"       Raw response: " f"{response.text[:200] if response.text else 'None'}")
                    time.sleep(wait_time)
                    return self._gemini_step_2_normalize(translated_editions, country_name, retry_count + 1)

                self.thread_safe_print(f"    ❌ Gemini returned empty response for {country_name} " f"after 3 retries")
                if hasattr(response, "text"):
                    self.thread_safe_print(f"       Final raw response: " f"{response.text[:200] if response.text else 'None'}")
                return []

            # Validate that we received normalization for all input editions
            if len(normalized_editions) != len(translated_editions):
                self.thread_safe_print(
                    f"    ⚠️ Normalization count mismatch for {country_name}: "
                    f"Expected {len(translated_editions)}, "
                    f"got {len(normalized_editions)}"
                )

                # Debug: Show what we received vs what we expected
                if self.verbose:
                    self.thread_safe_print("    🔍 Debug: Checking edition IDs...")
                    self.thread_safe_print("       Expected editions:")
                    for translated_edition in translated_editions[:5]:  # Show first 5
                        self.thread_safe_print(f"         - {translated_edition.name} (ID: {translated_edition.edition_id})")
                    if len(translated_editions) > 5:
                        self.thread_safe_print(f"         ... and {len(translated_editions) - 5} more")

                    self.thread_safe_print("       Received editions:")
                    for normalized_edition in normalized_editions[:5]:  # Show first 5
                        edition_id = getattr(normalized_edition, "edition_id", "NO_ID")
                        edition_name = getattr(normalized_edition, "name", "NO_NAME")
                        self.thread_safe_print(f"         - {edition_name} (ID: {edition_id})")
                    if len(normalized_editions) > 5:
                        self.thread_safe_print(f"         ... and {len(normalized_editions) - 5} more")

                # Find missing editions - check for None/empty IDs too
                normalized_ids = {ne.edition_id for ne in normalized_editions if hasattr(ne, "edition_id") and ne.edition_id}
                input_ids = {te.edition_id for te in translated_editions if hasattr(te, "edition_id") and te.edition_id}

                # If IDs are missing or unreliable, force retry based on count alone
                if len(normalized_ids) == 0 or len(input_ids) == 0:
                    self.thread_safe_print("    ⚠️ Edition IDs not available for comparison, using count-based retry logic")
                    missing_ids = set()  # Will trigger count-based retry below
                else:
                    missing_ids = input_ids - normalized_ids

                # Retry on ANY mismatch (ID-based or count-based)
                if missing_ids or len(normalized_editions) < len(translated_editions):
                    if missing_ids:
                        self.thread_safe_print(f"    🚨 Missing normalized data for {len(missing_ids)} editions:")
                        for missing_id in list(missing_ids)[:5]:  # Show first 5
                            # Find the edition name for better logging
                            missing_edition = next(
                                (te for te in translated_editions if te.edition_id == missing_id),
                                None,
                            )
                            edition_name = missing_edition.name if missing_edition else "Unknown"
                            self.thread_safe_print(f"       - {edition_name} (ID: {missing_id})")
                        if len(missing_ids) > 5:
                            self.thread_safe_print(f"       ... and {len(missing_ids) - 5} more")
                    else:
                        self.thread_safe_print(
                            f"    🚨 Count mismatch detected, attempting to recover "
                            f"{len(translated_editions) - len(normalized_editions)} "
                            f"missing editions"
                        )

                    # Retry normalization for ALL editions on mismatch
                    if retry_count < 2:
                        self.thread_safe_print(f"    🔄 Retrying normalization for ALL editions " f"due to mismatch (attempt {retry_count + 1}/2)")

                        try:
                            # Retry with ALL editions, not just missing ones
                            retried_normalized = self._gemini_step_2_normalize(translated_editions, country_name, retry_count + 1)
                            if retried_normalized and len(retried_normalized) == len(translated_editions):
                                # Full success - replace with retry results
                                normalized_editions = retried_normalized
                                self.thread_safe_print(f"    ✅ Retry successful! Normalized all " f"{len(normalized_editions)} editions")
                            elif retried_normalized and len(retried_normalized) > len(normalized_editions):
                                # Partial improvement - use the better result
                                normalized_editions = retried_normalized
                                self.thread_safe_print(
                                    f"    ⚠️ Retry improved results: " f"{len(normalized_editions)}/{len(translated_editions)} editions"
                                )
                            else:
                                self.thread_safe_print(
                                    f"    ⚠️ Retry didn't improve results, keeping original " f"{len(normalized_editions)} editions"
                                )
                        except Exception as err:  # pylint: disable=broad-exception-caught
                            self.thread_safe_print(f"    ❌ Retry failed: {str(err)[:150]}")
                            # Continue with partial results rather than failing completely
                    else:
                        self.thread_safe_print(
                            f"    ⚠️ Max retries reached. Continuing with " f"{len(normalized_editions)}/{len(translated_editions)} editions"
                        )

            # Add verbose logging for normalization completion
            if self.debug:
                self.thread_safe_print(f"    🔍 {country_name}: Normalized {len(normalized_editions)} editions " "(see debug log for details)")
                # Log all edition IDs for debugging
                if normalized_editions:
                    self.thread_safe_print("    📋 Normalized edition IDs:")
                    for normalized_edition in normalized_editions:
                        edition_id = getattr(normalized_edition, "edition_id", "NO_ID")
                        edition_name = getattr(normalized_edition, "name", "NO_NAME")
                        # Show truncated ID for readability
                        id_display = edition_id if edition_id != "NO_ID" else "MISSING"
                        if len(id_display) > 60:
                            id_display = id_display[:30] + "..." + id_display[-27:]
                        self.thread_safe_print(f"       - {edition_name}: {id_display}")
            elif self.verbose:
                self.thread_safe_print(f"    ✅ Step 2: Successfully normalized " f"{len(normalized_editions)} editions for {country_name}.")
            return normalized_editions

        except (ValueError, AttributeError, KeyError, TypeError) as err:
            # Handle any other unexpected errors
            self.thread_safe_print(f"    ❌ Unexpected error in normalization for {country_name}: {str(err)[:200]}")
            return []

    def _gemini_step_3_validate(self, editions: List[Dict], country_name: str, retry_count: int = 0) -> List[ValidationResult]:
        """Step 3: Validate the processed editions against approved list and rules.

        Checks flavors against approved list and detects any remaining
        non-English text that needs correction.

        Args:
            editions: List of processed editions to validate.
            country_name: Country name for context.
            retry_count: Current retry attempt (for internal use).

        Returns:
            List of ValidationResult objects.
        """
        if not self.client or not editions:
            return []

        # Add verbose logging for validation start
        if self.debug:
            self.thread_safe_print(f"    ✅ {country_name}: Validating {len(editions)} editions...")
        elif self.verbose:
            self.thread_safe_print(f"    🔍 Step 3: Validating {len(editions)} editions " f"for {country_name} against rules...")

        # Convert approved flavors list to string for prompt
        approved_flavors_str = json.dumps(self.APPROVED_FLAVORS, indent=4)

        # Prepare editions for validation
        editions_to_validate = [
            {
                "name": edition.get("name", "").strip(),
                "flavor": edition.get("flavor", "").strip(),
                "flavor_description": edition.get("flavor_description", "").strip(),
            }
            for edition in editions
        ]

        prompt = f"""
        You are a QUALITY VALIDATOR. Your job is to check if the Red Bull edition data follows all rules correctly.

        APPROVED FLAVORS LIST:
        {approved_flavors_str}

        For each edition, validate:
        1. is_valid: Does it follow ALL the rules below?
        2. flavor_in_approved_list: Is the flavor listed in the approved list?
        3. corrections_needed: List any problems found (empty list if no problems)
        4. corrected_flavor: ONLY if the flavor needs correction, provide the correct one. Otherwise leave this field empty/null
        5. corrected_description: ONLY if the description needs correction, provide the corrected version. Otherwise leave this field empty/null

        CRITICAL: DO NOT INTERPRET OR IMPROVE THE DESCRIPTIONS!
        Only fix SPECIFIC errors listed below. DO NOT replace edition names with flavor names!

        VALIDATION RULES TO CHECK:

        FLAVOR VALIDATION:
        1. Mark flavors as VALID - flavor normalization is handled by post-processing code
           - Do NOT correct flavors, the code will normalize them via similarity matching
           - Exception: Energy Drink base variants need correct flavors:
             - "Energy Drink Sugarfree" → flavor MUST be "Sugarfree"
             - "Energy Drink Zero" → flavor MUST be "Zero Sugar"
             - "Energy Drink" → flavor MUST be "Energy Drink"

        DESCRIPTION VALIDATION (ONLY fix these SPECIFIC issues):
        2. Description should NOT contain "sugars" (should be "sugar")
        3. Description should NOT contain "sugar-free" with hyphen (should be "sugarfree")
        4. PRESERVE EDITION NAMES IN DESCRIPTIONS:
            - The description MUST use the edition name from the "name" field, NOT the flavor!
            - Compare the edition name in the description against the "name" field
            - WRONG: If name="The Green Edition" but description says "Dragon Fruit Edition" → FIX IT!
            - CORRECT: "The Red Bull Green Edition with the taste of Dragon Fruit"
            - Edition names are color-based (Green, Blue, Yellow, Purple, etc.), NOT flavor-based!
            - Example: name="The Green Edition", flavor="Dragon Fruit"
              → Description MUST say "Green Edition", NOT "Dragon Fruit Edition"
        5. LANGUAGE CHECK: Description MUST be ENTIRELY in English!
            Check for non-English words like:
            - Spanish: sin, con, alas, azúcar, azúcares, sabor, edición
            - Portuguese: sem, com, asas, açúcar, sabor, edição
            - German: ohne, mit, flügel, zucker, geschmack
            - French: sans, avec, ailes, sucre, saveur
            - If ANY non-English words are found, translate ONLY those words to English

        NAME CHECKS:
        6. SUGARFREE NAME CHECK: Edition names with sugarfree=true MUST end with "Sugarfree"
        7. Edition names should NEVER start with "Flavor of" or contains Flavor" at all
        8. CRITICAL: NEVER use "The Original Edition" → must be "Energy Drink"
        9. CRITICAL: NEVER use "The Zero Edition" → must be "Energy Drink Zero"

        BE STRICT! Mark as invalid if ANY rule is broken.

        Editions to validate:
        {json.dumps(editions_to_validate, indent=4)}

        Return ONLY a JSON array of ValidationResult objects.
        """

        try:
            # Apply rate limiting
            self._apply_rate_limiting()

            # Log the request in verbose mode
            self._log_api_request(country_name, "Validation", prompt)

            if self.verbose:
                self.thread_safe_print(f"    - Sending validation request to Gemini " f"({len(prompt)} chars)...")

            start_time = time.time()

            response = self.client.models.generate_content(
                model=self.GEMINI_MODEL,
                contents=prompt,
                config=GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=list[ValidationResult],
                ),
            )

            api_time = time.time() - start_time

            if self.verbose:
                self.thread_safe_print(f"    - Gemini validation response time: {api_time:.2f}s")

            validation_results = response.parsed

            # Log the response in verbose mode
            self._log_api_response(country_name, "Validation", validation_results, api_time)

            # Add verbose logging for validation completion
            if self.debug:
                self.thread_safe_print(f"    🔍 {country_name}: Validated {len(validation_results)} editions " "(see debug log for details)")
            elif self.verbose:
                invalid_editions = []
                for i, validation in enumerate(validation_results):
                    if not validation.is_valid and i < len(editions):
                        edition = editions[i]
                        edition_info = {
                            "name": edition.get("name", "Unknown"),
                            "flavor": edition.get("flavor", ""),
                            "corrected": validation.corrected_flavor,
                            "issues": validation.corrections_needed,
                        }
                        invalid_editions.append(edition_info)

                if invalid_editions:
                    self.thread_safe_print(f"    ⚠️  Step 3: Found {len(invalid_editions)} editions needing correction:")
                    for edition_info in invalid_editions:
                        # Build the detail message
                        detail_parts = [f"      - {edition_info['name']}"]
                        if edition_info["corrected"] and edition_info["flavor"] != edition_info["corrected"]:
                            detail_parts.append(f" (flavor: {edition_info['flavor']} → {edition_info['corrected']})")
                        elif edition_info["issues"]:
                            # Show first issue if no flavor correction
                            first_issue = edition_info["issues"][0] if edition_info["issues"] else "validation failed"
                            detail_parts.append(f" (issue: {first_issue})")
                        self.thread_safe_print("".join(detail_parts))
                else:
                    self.thread_safe_print(f"    ✅ Step 3: All {len(validation_results)} editions " f"validated successfully")

            return validation_results

        except (ValueError, AttributeError, KeyError, TypeError) as err:
            # Log error details in debug mode
            if self.verbose:
                if hasattr(err, "code"):
                    self.thread_safe_print(f"    🐛 Debug: Error caught - Code: {err.code}")
                if hasattr(err, "message"):
                    self.thread_safe_print(f"    🐛 Debug: Error message: {err.message[:300]}")

                self.thread_safe_print(f"    🐛 Debug: Error: {str(err)[:300]}")

            if retry_count < 2:
                wait_time = (retry_count + 1) * 2
                self.thread_safe_print(f"    ⚠️ Validation error for {country_name}, " f"retrying in {wait_time}s (attempt {retry_count + 1}/2)")
                time.sleep(wait_time)
                return self._gemini_step_3_validate(editions, country_name, retry_count + 1)

            self.thread_safe_print(f"    ⚠️ Validation failed for {country_name}, continuing without validation")
            return []

    def normalize_with_gemini(
        self,
        editions: List[Dict],
        country_name: str,
        translated_cache: Optional[List[Dict]] = None,
        editions_to_translate: Optional[List[Dict]] = None,
        source_language: str = "Unknown",
    ) -> Tuple[List[Dict], List[Dict]]:
        """Use a multistep Gemini process to translate and normalize edition data.

        Orchestrates the 3-step AI processing pipeline:
        1. Translation to English (with intelligent partial caching)
        2. Normalization and standardization
        3. Validation against approved lists

        Args:
            editions: List of editions to process.
            country_name: Country name for context.
            translated_cache: Optional cached translations.
            editions_to_translate: Optional list of editions that need fresh translation.
            source_language: Source language name for translation context.

        Returns:
            Tuple of (processed editions, translation cache).

        Raises:
            RuntimeError: If processing fails.
        """
        if not self.client or not editions:
            return editions, []

        # Step 1: Translate (using intelligent partial cache)
        translated_editions_models = []

        # First, add any cached translations
        if translated_cache:
            if self.verbose:
                self.thread_safe_print(f"    ✅ Using {len(translated_cache)} cached translations for {country_name}.")
            # Convert dict cache to Pydantic models
            translated_editions_models = [TranslatedEdition(**item) for item in translated_cache]

        # Then, translate any new/changed editions
        if editions_to_translate:
            if self.verbose:
                self.thread_safe_print(f"    🔄 Translating {len(editions_to_translate)} new/changed editions " f"for {country_name}.")
            try:
                new_translations = self._gemini_step_1_translate(editions_to_translate, country_name, source_language)
                translated_editions_models.extend(new_translations)
            except (RuntimeError, ValueError, AttributeError) as err:
                self.thread_safe_print(f"  ❌ Translation error for {country_name}: {str(err)[:200]}")
                raise
        elif not translated_cache:
            # No cache and no specific editions to translate - translate everything
            if self.verbose:
                self.thread_safe_print(f"    🔄 No cache found, performing live translation " f"for {country_name}.")
            try:
                translated_editions_models = self._gemini_step_1_translate(editions, country_name, source_language)
            except (RuntimeError, ValueError, AttributeError) as err:
                self.thread_safe_print(f"  ❌ Translation error for {country_name}: {str(err)[:200]}")
                raise

        if not translated_editions_models:
            self.thread_safe_print(f"  ❌ No translations received for {country_name}")
            self.thread_safe_print("     Cannot proceed without AI")
            raise RuntimeError(f"Translation step produced no results for {country_name}")

        # Map URLs from original editions to translated models
        url_mapping = {}
        for edition in editions:
            edition_id = edition.get("_graphql_id")
            if edition_id:
                url_mapping[edition_id] = edition.get("product_url", "")

        # Add URLs to translated models
        for translated_model in translated_editions_models:
            if translated_model.edition_id in url_mapping:
                translated_model.product_url = url_mapping[translated_model.edition_id]

        # Step 2: Normalize
        try:
            normalized_data = self._gemini_step_2_normalize(translated_editions_models, country_name)
            if not normalized_data:
                self.thread_safe_print(f"  ❌ Normalization step failed for {country_name}, aborting.")
                self.thread_safe_print("     Details: Gemini API returned empty/None response")
                raise RuntimeError(f"Normalization failed for {country_name} - " "Gemini API returned no data (possible API filtering or timeout)")
        except (RuntimeError, ValueError, AttributeError, KeyError) as err:
            # Re-raise the exception with better context
            if "Normalization failed" not in str(err):
                self.thread_safe_print(f"  ❌ Normalization error for {country_name}: {str(err)[:200]}")
            raise

        # Create ID-based mapping for normalized data
        normalized_map = {}
        if normalized_data:
            for norm_edition in normalized_data:
                if hasattr(norm_edition, "edition_id") and norm_edition.edition_id:
                    normalized_map[norm_edition.edition_id] = norm_edition

        # Create ID-based mapping for translated editions
        translated_map = {}
        if translated_editions_models:
            for trans_edition in translated_editions_models:
                if hasattr(trans_edition, "edition_id") and trans_edition.edition_id:
                    translated_map[trans_edition.edition_id] = trans_edition

        # Log mapping info if in debug mode
        if self.verbose and normalized_map:
            self.thread_safe_print(f"    📊 Created ID-based mapping for {len(normalized_map)} normalized editions")
            if len(normalized_map) != len(editions):
                self.thread_safe_print(f"    ⚠️ WARNING: Edition count mismatch! " f"Input: {len(editions)}, Normalized: {len(normalized_map)}")

        # Combine results using ID-based mapping with fallbacks
        for idx, edition in enumerate(editions):
            # Use GraphQL ID for matching
            edition_id = edition.get("_graphql_id", "")
            edition_name = edition.get("name", "")

            # Try to find matching normalized data by ID (primary)
            norm_data = normalized_map.get(edition_id)

            # Fallback 1: Try name-based matching if ID match failed
            if not norm_data and normalized_data:
                for norm_edition in normalized_data:
                    if hasattr(norm_edition, "name") and norm_edition.name and norm_edition.name.strip().lower() == edition_name.strip().lower():
                        norm_data = norm_edition
                        if self.verbose:
                            self.thread_safe_print(f"      🔄 Using NAME-based matching for '{edition_name}' " f"(ID matching failed)")
                        break

            # Fallback 2: Try index-based matching if counts match
            if not norm_data and normalized_data and len(normalized_data) == len(editions):
                if idx < len(normalized_data):
                    norm_data = normalized_data[idx]
                    if self.verbose:
                        self.thread_safe_print(
                            f"      🔢 Using INDEX-based matching for '{edition_name}' " f"at position {idx} (ID and name matching failed)"
                        )

            if norm_data:
                # Only update flavor if it wasn't manually corrected
                corrected_fields = edition.get("_corrected_fields", set())
                if "flavor" not in corrected_fields:
                    # CRITICAL: Check if original flavor (cleaned) was in
                    # APPROVED_FLAVORS
                    original_raw_flavor = edition.get("_raw_flavor", "").strip()
                    if original_raw_flavor in self.APPROVED_FLAVORS:
                        # Original was already approved - DO NOT let AI change it!
                        edition["flavor"] = original_raw_flavor
                        if self.verbose:
                            # Only show preservation message if AI actually tried to change it
                            if norm_data.flavor != original_raw_flavor:
                                self.thread_safe_print(
                                    f"      🛡️ Preserving APPROVED flavor '{original_raw_flavor}' " f"(AI tried to change to '{norm_data.flavor}')"
                                )
                            else:
                                self.thread_safe_print(f"      ✅ AI correctly kept APPROVED flavor '{original_raw_flavor}'")
                    else:
                        edition["flavor"] = self.clean_flavor_name(norm_data.flavor)
                elif self.verbose:
                    self.thread_safe_print(f"      🔒 Keeping corrected flavor: {edition.get('flavor', 'unknown')}")

                # Always update flavor_description unless specifically corrected
                if "flavor_description" not in corrected_fields:
                    edition["flavor_description"] = self.clean_description(norm_data.flavor_description)
                elif self.verbose:
                    self.thread_safe_print(
                        f"      🔒 Keeping corrected flavor_description: " f"{edition.get('flavor_description', 'unknown')[:50]}..."
                    )

                # Use the sugarfree value from AI normalization
                edition["sugarfree"] = norm_data.sugarfree
            else:
                if self.verbose:
                    self.thread_safe_print(f"      ⚠️ No normalized data found for edition: " f"{edition_name} (ID: {edition_id})")
                # This should NEVER happen - raise an error to prevent empty fields
                raise RuntimeError(
                    f"CRITICAL: No normalized data found for edition "
                    f"'{edition_name}' with ID '{edition_id}'. "
                    f"This would result in empty flavor fields which is unacceptable."
                )

            # Update name from translated edition if available
            trans_data = translated_map.get(edition_id)
            if trans_data:
                fixed_name = self.fix_edition_name(trans_data.name)

                # Validate AI name against raw title and product URL as ground truth
                raw_title = edition.get("name", "")
                approved_raw = next((ed for ed in self.APPROVED_EDITIONS if ed.lower() == raw_title.lower()), None)
                url_edition = self.extract_edition_from_url(edition.get("product_url", ""))
                approved_url = next((ed for ed in self.APPROVED_EDITIONS if ed.lower() == (url_edition or "").lower()), None)
                ground_truth = approved_raw or approved_url

                if ground_truth and fixed_name:
                    expected_name = f"The {ground_truth}"
                    if fixed_name.lower() != expected_name.lower():
                        logging.warning(
                            "⚠️ Edition name mismatch: AI said '%s' but source data indicates '%s' → using '%s'",
                            fixed_name,
                            ground_truth,
                            expected_name,
                        )
                        fixed_name = expected_name

                edition["name"] = fixed_name or "Energy Drink"

        # Step 3: Validate and correct flavors against approved list
        try:
            validation_results = self._gemini_step_3_validate(editions, country_name)
            if validation_results:
                # Apply validation corrections
                for i, validation in enumerate(validation_results):
                    if i < len(editions) and not validation.is_valid:
                        edition = editions[i]
                        corrected_fields = edition.get("_corrected_fields", set())

                        # Apply corrected flavor if it's not manually corrected
                        if "flavor" not in corrected_fields and validation.corrected_flavor and validation.corrected_flavor.strip():
                            old_flavor = edition.get("flavor", "")
                            # Apply clean_flavor_name to ensure APPROVED_FLAVORS matching
                            new_flavor = self.clean_flavor_name(validation.corrected_flavor.strip())
                            edition["flavor"] = new_flavor

                            if self.verbose and old_flavor != new_flavor:
                                # Only show correction message if actually different
                                self.thread_safe_print(f"      🔧 Validation corrected flavor: " f"'{old_flavor}' → '{new_flavor}'")

                        # Apply corrected description if needed
                        if (
                            "flavor_description" not in corrected_fields
                            and validation.corrected_description
                            and validation.corrected_description.strip()
                        ):
                            old_description = edition.get("flavor_description", "")
                            new_description = validation.corrected_description.strip()
                            edition["flavor_description"] = new_description

                            if self.verbose and old_description != new_description:
                                # Only show correction message if actually different
                                self.thread_safe_print(f"      🔧 Validation corrected description for " f"{edition.get('name', 'unknown')}")

        except Exception as err:  # pylint: disable=broad-exception-caught
            if self.verbose:
                self.thread_safe_print(f"    ⚠️ Validation step failed for {country_name}: {str(err)[:200]}")
                self.thread_safe_print("    Continuing without validation corrections...")

        return editions, [t.model_dump() for t in translated_editions_models]

    # endregion

    # region Change Tracking Methods
    def _track_country_changes(self, country_name: str, new_data: Dict, processed_file: Path) -> None:
        """Track changes made to a country's data.

        Compares old and new data to track additions, updates, and removals
        for changelog generation.

        Args:
            country_name: Name of the country.
            new_data: New processed data.
            processed_file: Path to processed file for comparison.
        """
        # Check if we have existing data to compare
        if processed_file.exists():
            with open(processed_file, "r", encoding="utf-8") as file:
                old_data = json.load(file)

            old_editions = {edition.get("name", "") + edition.get("flavor", ""): edition for edition in old_data.get("editions", [])}
            new_editions = {edition.get("name", "") + edition.get("flavor", ""): edition for edition in new_data.get("editions", [])}

            # Find added editions
            for key, edition in new_editions.items():
                if key not in old_editions:
                    self.changelog["editions_added"][country_name].append(
                        {
                            "name": edition.get("name", ""),
                            "flavor": edition.get("flavor", ""),
                        }
                    )

            # Find removed editions
            for key, edition in old_editions.items():
                if key not in new_editions:
                    self.changelog["editions_removed"][country_name].append(
                        {
                            "name": edition.get("name", ""),
                            "flavor": edition.get("flavor", ""),
                        }
                    )

            # Find updated editions
            for key, new_edition in new_editions.items():
                if key in old_editions:
                    old_edition = old_editions[key]
                    changes = []

                    # Check each field for changes
                    fields_to_check = [
                        "flavor",
                        "flavor_description",
                        "name",
                        "sugarfree",
                        "color",
                        "image_url",
                        "product_url",
                    ]

                    for field in fields_to_check:
                        if old_edition.get(field) != new_edition.get(field):
                            changes.append(
                                {
                                    "field": field,
                                    "old": old_edition.get(field),
                                    "new": new_edition.get(field),
                                }
                            )

                    if changes:
                        self.changelog["editions_updated"][country_name].append(
                            {
                                "name": new_edition.get("name", ""),
                                "flavor": new_edition.get("flavor", ""),
                                "changes": changes,
                            }
                        )

                        # Also track in field_changes for summary
                        for change in changes:
                            self.changelog["field_changes"][change["field"]].append(
                                {
                                    "country": country_name,
                                    "edition": new_edition.get("name", ""),
                                    "old": change["old"],
                                    "new": change["new"],
                                }
                            )
        else:
            # All editions are new
            for edition in new_data.get("editions", []):
                self.changelog["editions_added"][country_name].append(
                    {
                        "name": edition.get("name", ""),
                        "flavor": edition.get("flavor", ""),
                    }
                )

    def _get_country_from_correction_id(self, correction_id: str) -> Optional[str]:
        """Extract country name from correction ID.

        Correction IDs have format: graphql-id:locale (e.g., 'eb9c22db-6c3d-4a68-b4e3-c915c59b1414:de-AT')
        This maps the locale part to the corresponding country name.

        Args:
            correction_id: The correction ID in format 'graphql-id:locale'

        Returns:
            Country name if found, None otherwise
        """
        if ":" not in correction_id:
            return None

        locale = correction_id.split(":", 1)[1]  # Get the locale part (e.g., 'de-AT')

        # Map locale to file pattern (e.g., 'de-AT' -> 'at-de')
        if "-" in locale:
            parts = locale.split("-")
            if len(parts) == 2:
                country_code, lang_code = parts
                file_pattern = f"{country_code.lower()}-{lang_code.lower()}"

                # Check if this corresponds to any processed country
                available_countries = self.discover_raw_files()
                for country_name, country_info in available_countries.items():
                    if country_info["domain"] == file_pattern:
                        return country_name

        return None

    def _generate_changelog_markdown(self) -> str:
        """Generate a markdown changelog from tracked changes.

        Creates detailed markdown report of all changes made during processing.

        Returns:
            Markdown formatted changelog string.
        """
        lines = ["# Red Bull Editions Processing Changelog", f"## {self.changelog['timestamp']}", "", "## Summary"]

        # Summary section
        total_added = sum(len(editions) for editions in self.changelog["editions_added"].values())
        total_updated = sum(len(editions) for editions in self.changelog["editions_updated"].values())
        total_removed = sum(len(editions) for editions in self.changelog["editions_removed"].values())

        lines.append(f"- **Countries processed:** {len(self.changelog['countries_processed'])}")
        lines.append(f"- **Countries skipped:** {len(self.changelog['countries_skipped'])}")
        lines.append(f"- **Cache hits:** {self.changelog['cache_hits']}")
        lines.append(f"- **API calls made:** {self.changelog['api_calls_made']}")
        lines.append(f"- **Editions added:** {total_added}")
        lines.append(f"- **Editions updated:** {total_updated}")
        lines.append(f"- **Editions removed:** {total_removed}")

        # Corrections summary
        applied_corrections = len(
            set(
                (cast(Dict[str, Any], c)["id"], cast(Dict[str, Any], c)["field"], cast(Dict[str, Any], c)["search"])
                for c in self.changelog["corrections_applied"]
            )
        )
        failed_corrections = len(
            set(
                (cast(Dict[str, Any], c)["id"], cast(Dict[str, Any], c)["field"], cast(Dict[str, Any], c)["search"])
                for c in self.changelog["corrections_failed"]
            )
        )

        lines.append(f"- **Corrections applied:** {applied_corrections}")
        lines.append(f"- **Corrections failed:** {failed_corrections}")
        lines.append("")

        # Warnings section for failed corrections
        if self.changelog["corrections_failed"]:
            lines.append("## ⚠️ Warnings - Corrections Not Applied")
            lines.append("")
            lines.append("The following corrections could not be applied (text not found):")
            lines.append("")

            # Group by unique correction
            seen_corrections = set()
            for correction in self.changelog["corrections_failed"]:
                corr = cast(Dict[str, Any], correction)
                key = (corr["id"], corr["field"], corr["search"])
                if key not in seen_corrections:
                    seen_corrections.add(key)
                    lines.append(f"### Correction ID: `{corr['id']}`")
                    lines.append(f"- **Field:** {corr['field']}")
                    lines.append(f"- **Search text:** \"{corr['search']}\"")
                    lines.append(f"- **Reason:** {corr['reason']}")
                    lines.append("")

        # ID mappings failed section
        if self.changelog["id_mappings_failed"]:
            lines.append("## ⚠️ Warnings - ID Mappings Not Applied")
            lines.append("")
            lines.append("The following ID mappings could not be applied (source edition not found):")
            lines.append("")
            for entry in self.changelog["id_mappings_failed"]:
                m = cast(Dict[str, Any], entry)
                lines.append(f"### Source not found: `{m['source_id']}`")
                lines.append(f"- **Target ID:** `{m['target_id']}`")
                lines.append(f"- **Fields:** {', '.join(m['fields'])}")
                lines.append(f"- **Effect:** Target edition keeps original (possibly incorrect) raw data")
                lines.append("")

        # Unused corrections check - only for processed countries
        relevant_correction_ids = set()
        processed_countries = set(self.changelog["countries_processed"])

        for corr in self.corrections:
            correction_country = self._get_country_from_correction_id(corr["id"])
            # Only include corrections for countries that were actually processed
            if correction_country in processed_countries:
                relevant_correction_ids.add(corr["id"])

        used_correction_ids = set()
        for corr in self.changelog["corrections_applied"]:
            used_correction_ids.add(cast(Dict[str, Any], corr)["id"])
        for corr in self.changelog["corrections_failed"]:
            used_correction_ids.add(cast(Dict[str, Any], corr)["id"])

        unused_corrections = relevant_correction_ids - used_correction_ids
        if unused_corrections:
            lines.append("## 📝 Unused Corrections")
            lines.append("")
            lines.append("The following corrections were not checked (no matching editions found):")
            lines.append("")
            for corr_id in sorted(unused_corrections):
                lines.append(f"- `{corr_id}`")
            lines.append("")

        # Countries processed section
        if self.changelog["countries_processed"]:
            lines.append("## Countries Processed")
            lines.append("")
            for country in sorted(self.changelog["countries_processed"]):
                lines.append(f"### {country}")

                # Track if this country has any edition changes
                has_changes = False

                # Additions
                if country in self.changelog["editions_added"] and self.changelog["editions_added"][country]:
                    lines.append("#### Added Editions:")
                    for edition in self.changelog["editions_added"][country]:
                        edition_dict = cast(Dict[str, Any], edition)
                        lines.append(f"- **{edition_dict['name']}** - {edition_dict['flavor']}")
                    lines.append("")
                    has_changes = True

                # Updates
                if country in self.changelog["editions_updated"] and self.changelog["editions_updated"][country]:
                    lines.append("#### Updated Editions:")
                    for edition in self.changelog["editions_updated"][country]:
                        edition_dict = cast(Dict[str, Any], edition)
                        lines.append(f"- **{edition_dict['name']}** - {edition_dict['flavor']}")
                    lines.append("")
                    has_changes = True

                # Removals
                if country in self.changelog["editions_removed"] and self.changelog["editions_removed"][country]:
                    lines.append("#### Removed Editions:")
                    for edition in self.changelog["editions_removed"][country]:
                        edition_dict = cast(Dict[str, Any], edition)
                        lines.append(f"- **{edition_dict['name']}** - {edition_dict['flavor']}")
                    lines.append("")
                    has_changes = True

                # If no edition changes, indicate this clearly
                if not has_changes:
                    lines.append("- No edition changes")
                    lines.append("")

        # Countries skipped section
        if self.changelog["countries_skipped"]:
            lines.append("## Countries Skipped (Cached)")
            lines.append("")
            for country in sorted(self.changelog["countries_skipped"]):
                lines.append(f"- {country}")
            lines.append("")

        # Daily limit section – countries deferred to next run
        if self.changelog.get("countries_skipped_daily_limit"):
            skipped = self.changelog["countries_skipped_daily_limit"]
            calls_remaining = len(skipped) * 3
            lines.append("## ⚠️ Daily API Limit Reached – Countries Deferred")
            lines.append("")
            lines.append(f"The following {len(skipped)} countries could not be processed today " f"and will be picked up on the next run:")
            lines.append("")
            for country in sorted(skipped):
                lines.append(f"- {country}")
            lines.append("")
            lines.append(f"**API calls today:** {self.changelog['api_calls_made']}/{self.MAX_REQUESTS_PER_DAY}")
            lines.append(f"**Calls still needed:** ~{calls_remaining} ({len(skipped)} countries × 3 steps)")
            lines.append("")

        # Errors section
        if self.changelog["errors"]:
            lines.append("## ❌ Errors")
            lines.append("")
            for error in self.changelog["errors"]:
                lines.append(f"- {error}")
            lines.append("")

        return "\n".join(lines)

    def _should_create_changelog(self) -> bool:
        """
        Determine if a changelog should be created based on actual changes.

        Returns:
            bool: True if changelog should be created, False otherwise
        """
        return (
            len(self.changelog["countries_processed"]) > 0
            or len(self.changelog["corrections_applied"]) > 0  # Actual processing happened
            or len(self.changelog["corrections_failed"]) > 0  # Corrections were applied
            or len(self.changelog["errors"]) > 0  # Errors occurred
            or len(self.changelog.get("countries_skipped_daily_limit", [])) > 0  # Daily limit hit
        )

    def _save_changelog(self) -> Path:
        """
        Save the changelog to a markdown file.

        Returns:
            Path to the saved changelog file
        """
        changelog_dir = self.data_dir / "changelogs"
        changelog_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        changelog_file = changelog_dir / f"changelog_{timestamp}.md"

        markdown_content = self._generate_changelog_markdown()

        with open(changelog_file, "w", encoding="utf-8") as file:
            file.write(markdown_content)

        # Also save as latest changelog for easy access
        latest_file = self.data_dir / "latest_changelog.md"
        with open(latest_file, "w", encoding="utf-8") as file:
            file.write(markdown_content)

        return changelog_file

    # endregion

    # region Country Processing Methods
    def apply_energy_drink_flavor_rules(self, editions: List[Dict]) -> None:
        """
        Apply hard-coded flavor rules for Energy Drink variants.
        These rules are deterministic and should not be left to AI interpretation.
        """
        for edition in editions:
            name = edition.get("name", "").strip()
            name_lower = name.lower()

            # Hard-coded rules for Energy Drink variants (case-insensitive)
            if name_lower == "energy drink sugarfree":
                edition["flavor"] = "Sugarfree"
                edition["sugarfree"] = True
                if self.verbose:
                    self.thread_safe_print(f"      🔧 Applied rule: {name} → flavor: Sugarfree, sugarfree: True")
            elif name_lower == "energy drink zero":
                edition["flavor"] = "Zero Sugar"
                edition["sugarfree"] = True
                if self.verbose:
                    self.thread_safe_print(f"      🔧 Applied rule: {name} → flavor: Zero Sugar, sugarfree: True")
            elif name_lower == "energy drink":
                edition["flavor"] = "Energy Drink"
                if self.verbose:
                    self.thread_safe_print(f"      🔧 Applied rule: {name} → flavor: Energy Drink")

    def enforce_sugarfree_logic(self, editions: List[Dict]) -> None:
        """
        Force sugarfree status based on strong keywords in URL, name, or flavor.
        This overrides AI mistakes where it fails to detect sugarfree status.
        """
        sugarfree_keywords = [
            "sugarfree",
            "sugar-free",
            "zero",
            "zuckerfrei",
            "sem açúcar",
            "sin azúcar",
            "senza zucchero",
            "sokeros",
            "sockerfri",
            "cukormentes",
            "bez cukru",
            "uten sukker",
            "uden sukker",
            "sokeriton",
            "sugarlane",
        ]

        for edition in editions:
            # Skip if already sugarfree
            if edition.get("sugarfree", False):
                continue

            is_sugarfree = False
            match_source = ""

            # Check name
            name_lower = edition.get("name", "").lower()

            # Check flavor
            flavor_lower = edition.get("flavor", "").lower()
            raw_flavor_lower = str(edition.get("_raw_flavor", "")).lower()

            # Check URL
            url_lower = edition.get("product_url", "").lower()

            # Check all fields against keywords
            for keyword in sugarfree_keywords:
                # Word boundary check for some might be better, but substring is safer for now
                # given the variations in URLs and raw data
                if keyword in name_lower:
                    is_sugarfree = True
                    match_source = f"name ({keyword})"
                    break
                if keyword in flavor_lower:
                    is_sugarfree = True
                    match_source = f"flavor ({keyword})"
                    break
                if keyword in raw_flavor_lower:
                    is_sugarfree = True
                    match_source = f"raw_flavor ({keyword})"
                    break
                if keyword in url_lower:
                    is_sugarfree = True
                    match_source = f"url ({keyword})"
                    break

            if is_sugarfree:
                edition["sugarfree"] = True
                if self.verbose:
                    self.thread_safe_print(f"      🔧 Forced sugarfree: {edition.get('name')} (detected in {match_source})")

    def fix_edition_names(self, editions: List[Dict]) -> None:
        """
        Fix common edition naming errors.
        Editions and flavors are independent - never derive edition name from flavor!
        """
        edition_fixes = {
            "Iced Edition": "Ice Edition",  # It's ALWAYS "Ice Edition", never "Iced"
            # Add more known corrections here as needed
        }

        for edition in editions:
            name = edition.get("name", "").strip()

            # Apply all known fixes
            for wrong, correct in edition_fixes.items():
                if wrong in name:
                    fixed_name = name.replace(wrong, correct)
                    edition["name"] = fixed_name
                    if self.verbose:
                        self.thread_safe_print(f"      🔧 Fixed edition name: {name} → {fixed_name}")
                    name = fixed_name  # Update for next iteration

    def process_country(self, country_name: str, domain: str, flag_code: str, force: bool = False) -> Tuple[Optional[Dict], bool]:
        """Process a single country's data.

        Main processing pipeline: load raw data → apply corrections →
        translate → normalize → validate → save.

        Args:
            country_name: Name of the country.
            domain: Domain identifier.
            flag_code: Two-letter country code.
            force: Force reprocessing even if cached.

        Returns:
            Tuple of (processed country data or None, was_actually_processed boolean).
        """
        raw_file = self.raw_dir / f"{domain}.json"

        if not raw_file.exists():
            self.thread_safe_print(f"  ❌ Raw data not found for {country_name}")
            return None, False

        with open(raw_file, "r", encoding="utf-8") as file:
            raw_data = json.load(file)

        # Extract source language from domain (e.g., "br-pt" → "Portuguese (Brazil)")
        source_language = "English"  # Default
        if "-" in domain:
            # Split domain into parts
            parts = domain.split("-")
            try:
                # Convert domain format (br-pt) to Babel format (pt_BR)
                babel_locale = f"{parts[1]}_{parts[0].upper()}"
                locale = Locale.parse(babel_locale)
                source_language = locale.get_language_name("en")
            except (UnknownLocaleError, ValueError):
                # Fallback: just use language code
                source_language = parts[1]

        raw_hash = hashlib.sha256(json.dumps(raw_data["editions"], sort_keys=True).encode()).hexdigest()

        # Check if already processed and unchanged
        processed_file = self.processed_dir / f"{domain}_processed.json"
        translated_cache = None
        editions_to_translate = None

        if not force and processed_file.exists():
            with open(processed_file, "r", encoding="utf-8") as file:
                existing = json.load(file)
                # Check if raw data hasn't changed
                if existing.get("_raw_hash") == raw_hash:
                    self.thread_safe_print(f"  ⏭️  {country_name}: No changes, using cached")
                    self.changelog["cache_hits"] += 1
                    self.changelog["countries_skipped"].append(country_name)
                    return existing, False

                # If raw data changed, intelligently use partial cache
                if existing.get("_translated_editions"):
                    # Build a map of cached translations by ID
                    cached_translations = {item["edition_id"]: item for item in existing.get("_translated_editions", [])}

                    # Check which editions can use cache vs need fresh translation
                    editions_to_translate = []
                    cached_editions = []

                    # Track which edition IDs need fresh translation
                    editions_needing_translation = []
                    for edition in raw_data["editions"]:
                        # Get the edition ID
                        edition_id = edition.get("id", "")

                        if edition_id in cached_translations:
                            # Use cached translation for this edition
                            cached_editions.append(cached_translations[edition_id])
                            if self.verbose:
                                self.thread_safe_print(
                                    f"    ♻️ Using cached translation for {edition.get('header_data', {}).get('content', {}).get('title', 'Unknown')}"
                                )
                        else:
                            # Track this edition ID for later processing
                            editions_needing_translation.append(edition_id)
                            if self.verbose:
                                title = edition.get("header_data", {}).get("content", {}).get("title", "Unknown")
                                self.thread_safe_print(f"    🆕 New edition needs translation: {title}")

                    # Set the partial cache
                    translated_cache = cached_editions if cached_editions else None

                    if self.verbose and cached_editions:
                        self.thread_safe_print(f"    📊 Cache stats: {len(cached_editions)} cached, {len(editions_to_translate)} need translation")

        elif force and self.verbose:
            self.thread_safe_print(f"    🔧 Debug: Forcing reprocess for {country_name}, " "ignoring cache")

        # Check if there are editions to process
        if not raw_data.get("editions"):
            if self.verbose:
                self.thread_safe_print(f"  ⏭️  {country_name}: No editions found, skipping")
            self.changelog["countries_skipped"].append(f"{country_name} (no editions)")
            return None, False

        self.thread_safe_print(f"  🔄 Processing {country_name}...")

        editions_to_process = []

        for edition_raw in raw_data["editions"]:
            if edition_raw is None:
                logging.warning("⚠️  %s: Found None edition entry in raw data!", country_name)
                continue
            header_data = edition_raw.get("header_data", {})
            graphql_data = edition_raw.get("graphql_data", {}).get("data", {})
            graphql_id = edition_raw.get("id", "")

            edition = {
                "name": header_data.get("content", {}).get("title", "Unknown"),
                "flavor": "",
                "flavor_description": "",
                "sugarfree": False,
                "color": graphql_data.get("brandingHexColorCode", "#FFFFFF"),
                "image_url": "",
                "alt_text": "",
                "product_url": "",
                "_raw_flavor": graphql_data.get("flavour", ""),
                "_standfirst": graphql_data.get("standfirst", ""),
                "_graphql_id": graphql_id,
                "_raw_alt_text": "",
            }

            # Extract color
            if edition["color"] == "#FFFFFF" and header_data.get("media", {}).get("mainImage"):
                colors = header_data["media"]["mainImage"].get("imageEssence", {}).get("predominantColors", [])
                if colors:
                    edition["color"] = colors[0].get("hexColorCode", "#FFFFFF")

            # Extract image
            img_url = None
            if graphql_data.get("image", {}).get("imageEssence", {}).get("imageURL"):
                img_url = graphql_data["image"]["imageEssence"]["imageURL"]
                edition["alt_text"] = graphql_data["image"].get("altText", "")
                edition["_raw_alt_text"] = edition["alt_text"]  # Keep for processing
            elif header_data.get("media", {}).get("mainImage", {}).get("imageEssence", {}).get("imageURL"):
                img_url = header_data["media"]["mainImage"]["imageEssence"]["imageURL"]
                edition["alt_text"] = header_data["media"]["mainImage"].get("altText", "")
                edition["_raw_alt_text"] = edition["alt_text"]  # Keep for processing

            if img_url:
                edition["image_url"] = self.extract_image_url(img_url)

            # Extract product URL
            if graphql_data.get("reference", {}).get("externalUrl"):
                edition["product_url"] = self.fix_product_url(graphql_data["reference"]["externalUrl"])
            elif header_data.get("reference", {}).get("externalUrl"):
                edition["product_url"] = self.fix_product_url(header_data["reference"]["externalUrl"])

            # Check sugar-free
            # Sugarfree will be determined by AI, not by simple keyword search
            edition["sugarfree"] = False  # Default, will be updated by AI

            # Apply manual corrections before processing
            edition = self.apply_corrections(edition, graphql_id)

            editions_to_process.append(edition)

        # Apply ID mappings (replace raw fields from another locale before AI processing)
        editions_to_process = self._apply_id_mappings(editions_to_process)

        # If we have per-edition cache logic, build the editions_to_translate list from processed editions
        if "editions_needing_translation" in locals() and editions_needing_translation:
            # Find the fully processed editions that correspond to the IDs needing translation
            editions_to_translate = [edition for edition in editions_to_process if edition.get("_graphql_id", "") in editions_needing_translation]
        else:
            # Fallback to existing logic (no per-edition cache)
            editions_to_translate = editions_to_translate if "editions_to_translate" in locals() else None

        # Normalize with AI
        editions_to_process, translated_editions = self.normalize_with_gemini(
            editions_to_process,
            country_name,
            translated_cache,
            editions_to_translate,
            source_language,
        )

        # Force sugarfree status based on keywords (overrides AI mistakes)
        self.enforce_sugarfree_logic(editions_to_process)

        # Apply hard-coded Energy Drink flavor rules (AFTER AI normalization)
        self.apply_energy_drink_flavor_rules(editions_to_process)

        # Fix edition naming errors (e.g., "Iced Edition" → "Ice Edition")
        self.fix_edition_names(editions_to_process)

        # Ensure Edition names with sugarfree=true end with "Sugarfree"
        for edition in editions_to_process:
            # Fix incorrect "Sugarfree Edition" naming
            name = edition.get("name", "")
            if name == "The Sugarfree Edition":
                edition["name"] = "Energy Drink"  # Will become "Energy Drink Sugarfree" below
                if self.verbose:
                    self.thread_safe_print("      🔧 Fixed 'The Sugarfree Edition' → 'Energy Drink'")

            # AI has already determined sugarfree status correctly
            # We just need to ensure consistent naming
            # Exception: "Energy Drink Zero" is OK
            if edition["sugarfree"]:
                name = edition.get("name", "")
                if "Edition" in name and not name.endswith("Sugarfree") and "Zero" not in name:
                    edition["name"] = f"{name} Sugarfree"
                    if self.verbose:
                        self.thread_safe_print(f"      🔧 Added Sugarfree to edition name: {edition['name']}")

        # Fix Edition spacing issues in all text fields
        for edition in editions_to_process:
            edition["flavor_description"] = self.fix_edition_spacing(edition.get("flavor_description", ""))
            # Also fix flavor field if needed
            edition["flavor"] = self.fix_edition_spacing(edition.get("flavor", ""))

        # Check for suspicious patterns in edition names
        for edition in editions_to_process:
            name = edition.get("name", "")
            flavor = edition.get("flavor", "")

            if "Edition" in name and name != "Energy Drink":
                # Extract edition core
                edition_core = name.replace("The ", "").replace(" Sugarfree", "")

                # Warning: Edition name matches flavor exactly (suspicious)
                if edition_core.replace(" Edition", "") == flavor:
                    # Check if it's a known acceptable case
                    # (like Coconut Edition with Coconut flavor)
                    acceptable_matches = [
                        "Coconut",
                        "Peach",
                        "Watermelon",
                        "Tropical Fruits",
                    ]
                    if flavor not in acceptable_matches:
                        if self.verbose:
                            self.thread_safe_print(f"      🚨 SUSPICIOUS: Edition name '{edition_core}' " f"matches flavor '{flavor}' exactly!")

                # Warning: Edition name contains typical flavor words
                flavor_words = ["Berry", "Fruits", "Fruit", "Taste", "Flavor"]
                for flavor_word in flavor_words:
                    if flavor_word in edition_core and edition_core not in self.APPROVED_EDITIONS:
                        if self.verbose:
                            self.thread_safe_print(
                                f"      ⚠️  Edition name '{edition_core}' contains " f"flavor word '{flavor_word}' - verify this is correct"
                            )

        # Final cleanup: ensure no double spaces in any text field
        for edition in editions_to_process:
            if edition.get("name"):
                edition["name"] = " ".join(edition["name"].split())
            if edition.get("flavor"):
                edition["flavor"] = " ".join(edition["flavor"].split())
            if edition.get("flavor_description"):
                edition["flavor_description"] = " ".join(edition["flavor_description"].split())
            if edition.get("alt_text"):
                edition["alt_text"] = " ".join(edition["alt_text"].split())

        # Clean up temporary fields
        for edition in editions_to_process:
            edition.pop("_raw_flavor", None)
            edition.pop("_standfirst", None)
            edition.pop("_graphql_id", None)
            edition.pop("_raw_alt_text", None)
            edition.pop("_corrected_fields", None)

        # Build final structure
        # Use 'Worldwide' for INT flag code in URLs
        flag_url_code = "Worldwide" if flag_code == "INT" else flag_code
        country_data = {
            "flag": self.convert_flag_code_to_emoji(flag_code, country_name),
            "editions": editions_to_process,
            "flag_url": f"https://rbds-static.redbull.com/@cosmos/foundation/latest/" f"flags/cosmos-flag-{flag_url_code}.svg",
            "_raw_hash": raw_hash,
            "_translated_editions": translated_editions,
        }

        # Track changes for changelog
        self._track_country_changes(country_name, country_data, processed_file)

        # Save processed data
        with open(processed_file, "w", encoding="utf-8") as file:
            json.dump(country_data, file, indent=4, ensure_ascii=False)

        self.changelog["countries_processed"].append(country_name)

        # Show completion message
        self.thread_safe_print(f"  ✅ Processed {country_name}")

        return country_data, True

    def process_country_wrapper(self, args: Tuple[str, str, str, bool]) -> Tuple[str, Optional[Dict], bool]:
        """Wrapper for process_country to work with ThreadPoolExecutor.

        Unpacks tuple arguments for parallel processing compatibility.

        Args:
            args: Tuple of (country_name, domain, flag_code, force).

        Returns:
            Tuple of (country_name, processed_data, was_actually_processed).
        """
        # Check abort flag before starting
        if self._abort_flag:
            country_name = args[0]
            return country_name, None, False

        country_name, domain, flag_code, force = args
        result, was_processed = self.process_country(country_name, domain, flag_code, force)
        return country_name, result, was_processed

    # endregion

    # region Main Processing Method
    def process_all(self, force_reprocess: bool = False) -> Dict[str, Any]:
        """Process all collected data.

        Main entry point for batch processing. Uses parallel processing
        with ThreadPoolExecutor for efficiency. Handles change detection,
        caching, and final JSON generation.

        Args:
            force_reprocess: Force reprocessing even if cached.

        Returns:
            Dictionary of all processed country data.
        """
        self.logger.info("🔄 Starting data processing...")
        if force_reprocess:
            self.logger.info("  ⚡ Force reprocess mode - ignoring cache")
        if self.verbose:
            self.logger.debug("  🔧 Debug mode enabled")
            self.logger.debug("  🔄 Parallel processing with %d workers", self.max_workers)

        # Discover available countries from raw files
        available_countries = self.discover_raw_files()

        if not available_countries:
            self.logger.error("❌ No raw data files found in data/raw/!")
            self.logger.error("   Please run collector.py first to collect data.")
            return {}

        self.logger.info("📊 Found %d countries in raw data", len(available_countries))

        # Try to load collection summary for change detection (optional)
        changes_detected = []
        summary_file = self.data_dir / "collection_summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, "r", encoding="utf-8") as file:
                    summary = json.load(file)
                changes_detected = summary["metadata"].get("changes_detected", [])
                if self.verbose:
                    print(f"  🔧 Debug: Found {len(changes_detected)} changed countries from summary")
            except (IOError, json.JSONDecodeError, KeyError) as err:
                if self.verbose:
                    print(f"  ⚠️  Could not read collection summary: {err}")
                    print("  🔧 Debug: Proceeding without change detection")

        # Load existing final data if exists
        final_file = self.data_dir / "redbull_editions_final.json"
        if final_file.exists():
            with open(final_file, "r", encoding="utf-8") as file:
                final_data = json.load(file)
            # Remove any countries with empty editions from existing data
            final_data = {country: data for country, data in final_data.items() if data.get("editions") and len(data.get("editions", [])) > 0}
        else:
            final_data = {}

        # Determine countries to process
        if force_reprocess:
            countries_to_process = list(available_countries.keys())
            if self.verbose:
                print(f"  🔧 Debug: Force mode - will process all " f"{len(countries_to_process)} countries")
        else:
            countries_to_process = changes_detected if final_data and changes_detected else list(available_countries.keys())
            if self.verbose:
                print(f"  🔧 Debug: Processing {len(countries_to_process)} " "countries (changed or new)")

        # Check if there's actually anything to process
        if not countries_to_process:
            self.logger.info("\n✨ Processing complete!")
            self.logger.info("📊 Statistics:")
            self.logger.info("  • Total countries: %d", len(final_data))
            self.logger.info("  • No countries need updating")
            self.logger.info("  • All data is up-to-date")
            self.logger.info("💾 Data location: %s", final_file)
            return final_data

        print(f"📊 Processing {len(countries_to_process)} countries...")

        # Always show rate limiting info
        api_calls_needed = len(countries_to_process) * 3
        print(f"⏱️  Rate limiting: Max {self.MAX_REQUESTS_PER_MINUTE} req/min, max {self.MAX_REQUESTS_PER_DAY} req/day")
        print(f"   Each country requires 3 API calls → {api_calls_needed} calls needed")
        if api_calls_needed > self.MAX_REQUESTS_PER_DAY:
            max_countries_today = self.MAX_REQUESTS_PER_DAY // 3
            print(
                f"   ⚠️  {api_calls_needed} calls needed but only {self.MAX_REQUESTS_PER_DAY}/day available "
                f"→ max {max_countries_today} countries today, remainder deferred to next run"
            )
        countries_today = min(len(countries_to_process), self.MAX_REQUESTS_PER_DAY // 3)
        estimated_time = (countries_today * 3 * self.MIN_DELAY_BETWEEN_REQUESTS) / 60 / self.max_workers
        print(f"   Estimated processing time: ~{estimated_time:.1f} minutes")

        # Prepare tasks
        tasks = []
        for country_name in countries_to_process:
            country_info = available_countries.get(country_name, {})
            domain = country_info.get("domain")
            flag_code = country_info.get("flag_code")

            if domain:
                tasks.append((country_name, domain, flag_code, force_reprocess))

        # Process countries (parallel if no rate limit, sequential otherwise)
        processed_count = 0  # Count of actually processed (not cached)
        completed_count = 0  # Count of all completed (including cached)
        start_time = time.time()

        # Retry tracking: {country_name: attempt_count}
        retry_attempts = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(self.process_country_wrapper, task): task for task in tasks}

            # Process completed tasks
            while future_to_task:
                done, _ = wait(future_to_task.keys(), return_when=FIRST_COMPLETED)

                for future in done:
                    task = future_to_task.pop(future)
                    country_name = task[0]

                    try:
                        (
                            country_name,
                            processed,
                            was_actually_processed,
                        ) = future.result()
                        if processed:
                            # Skip countries with no editions - don't add them to final_data at all
                            if not processed.get("editions") or len(processed.get("editions", [])) == 0:
                                self.thread_safe_print(f"  ⏭️  Skipping {country_name}: No editions found")
                                # Don't increment completed_count for skipped countries
                                continue

                            # Remove internal fields before adding to final
                            processed_copy = processed.copy()
                            processed_copy.pop("_raw_hash", None)
                            processed_copy.pop("_translated_editions", None)
                            final_data[country_name] = processed_copy

                            # Only increment processed_count if actually processed (not cached)
                            if was_actually_processed:
                                processed_count += 1

                            # Always increment completed_count
                            completed_count += 1

                            # Success - no retry tracking needed with immediate retry logic

                            elapsed = time.time() - start_time
                            # Only calculate meaningful ETA if we have actual processing time
                            remaining = len(tasks) - completed_count

                            # Check if this was a successful retry
                            attempt = retry_attempts.get(country_name, 1)
                            if attempt > 1:
                                self.thread_safe_print(f"  ✅ {country_name} retry successful after {attempt} attempts!")

                            self.thread_safe_print(f"  ✅ Completed {completed_count}/{len(tasks)}: " f"{country_name}")

                            # Only show ETA if there are remaining tasks and we have meaningful processing data
                            if remaining > 0 and processed_count > 0:
                                avg_time_per_processed_country = elapsed / processed_count
                                # Estimate remaining time based on how many countries still need actual processing
                                # This is a rough estimate since we don't know which remaining are cached vs need processing
                                eta = remaining * avg_time_per_processed_country
                                if eta > 5:  # Only show ETA if more than 5 seconds remaining
                                    self.thread_safe_print(f"     ⏱️ ETA: ~{eta/60:.1f} minutes remaining")
                    except (
                        IOError,
                        json.JSONDecodeError,
                        ValueError,
                        RuntimeError,
                    ) as err:
                        # Graceful skip when daily API limit is reached
                        is_daily_limit_error = self._daily_limit_reached and "Daily API limit" in str(err)
                        is_new_skip = country_name not in self._skipped_due_to_limit
                        if is_daily_limit_error and is_new_skip:
                            self._skipped_due_to_limit.append(country_name)
                            self.changelog["countries_skipped_daily_limit"].append(country_name)
                            self.thread_safe_print(f"  ⏭️  {country_name} skipped – daily API limit reached")
                        if is_daily_limit_error:
                            continue  # Don't retry, don't abort – just skip

                        # Check if abort was triggered (e.g., expired API key)
                        if self._abort_flag:
                            self.thread_safe_print(f"  ⏹️  Aborting {country_name} due to critical error")
                            # Cancel all pending futures
                            for pending_future in future_to_task:
                                if not pending_future.done():
                                    pending_future.cancel()
                            break  # Exit the while loop

                        # Track retry attempts
                        attempt = retry_attempts.get(country_name, 0) + 1
                        retry_attempts[country_name] = attempt

                        # Enhanced error logging with clearer messages
                        error_summary = str(err)[:100].replace("\n", " ")
                        self.thread_safe_print(f"  ❌ {country_name} failed (attempt {attempt}/3): {error_summary}")

                        # Log error to changelog
                        self.changelog["errors"].append(f"{country_name} (attempt {attempt}): {error_summary}")

                        if attempt < 3:
                            # Immediate focused retry - submit as new task parallel to others
                            remaining_countries = len([f for f in future_to_task if not f.done()])
                            self.thread_safe_print(
                                f"  🔄 Starting immediate retry for {country_name} " f"(parallel to {remaining_countries} other countries)..."
                            )

                            # Submit retry as new future - runs parallel to other tasks
                            retry_future = executor.submit(self.process_country_wrapper, task)
                            future_to_task[retry_future] = task

                            self.thread_safe_print(f"  ⏱️  {country_name} retry in progress, " f"{remaining_countries} countries still processing...")
                        else:
                            # Failed 3 times - abort entire processing
                            self._abort_flag = True
                            self.thread_safe_print(f"  ❌ {country_name} failed 3 times. Aborting all processing.")
                            self.thread_safe_print(f"     Final error: {error_summary}")
                            # Cancel pending futures
                            for pending_future in future_to_task:
                                if not pending_future.done():
                                    pending_future.cancel()
                            # Save partial changelog before aborting
                            self._save_changelog()
                            # Raise exception to abort processing
                            raise SystemExit(
                                f"Critical failure: {country_name} failed after 3 attempts. "
                                f"Processing aborted to preserve API quota and cache integrity."
                            ) from err

        # Check if processing was aborted due to critical error (e.g., expired API key)
        if self._abort_flag:
            self.thread_safe_print("\n❌ Processing aborted due to critical error")
            self.thread_safe_print("   No data was saved to preserve integrity")
            # Save error changelog
            if self.changelog["errors"]:
                self._save_changelog()
            sys.exit(1)

        # Add 'The Red Bull X Edition:' prefix to descriptions at the very end
        for country_name, country_data in final_data.items():
            if "editions" in country_data:
                country_data["editions"] = self.add_description_prefix(country_data["editions"])

        # Save final data (sorted alphabetically by country and edition name)
        with open(final_file, "w", encoding="utf-8") as file:
            json.dump(self._sort_final_data(final_data), file, indent=4, ensure_ascii=False)

        # Clear changes_detected in summary after successful processing
        if summary.get("metadata", {}).get("changes_detected"):
            summary["metadata"]["changes_detected"] = []
            with open(summary_file, "w", encoding="utf-8") as file:
                json.dump(summary, file, indent=4, ensure_ascii=False)
            if self.verbose:
                self.logger.info("  🧹 Cleared changes_detected list in collection_summary.json")

        # Statistics
        total_editions = sum(len(country["editions"]) for country in final_data.values())

        # Generate and save changelog only if changes were made
        changelog_file = None
        if self._should_create_changelog():
            changelog_file = self._save_changelog()

        self.logger.info("\n✨ Processing complete!")
        self.logger.info("📊 Statistics:")
        self.logger.info("  • Total countries: %d", len(final_data))

        # Check if all were cached
        if processed_count == 0 and len(countries_to_process) > 0:
            print(f"  • Countries checked: {len(countries_to_process)} (all using cache)")
            print("  • No actual updates needed")
        else:
            print(f"  • Countries updated: {processed_count}")
            print(f"  • Countries checked: {len(countries_to_process)}")

        print(f"  • Total editions: {total_editions}")
        print(f"💾 Final data saved to: {final_file}")

        if changelog_file:
            print(f"📝 Changelog saved to: {changelog_file}")
        else:
            print("📝 No changes made - changelog skipped")

        # Print warnings if there are failed corrections
        if self.changelog["corrections_failed"]:
            print(f"\n⚠️  WARNING: {len(self.changelog['corrections_failed'])} corrections " f"could not be applied!")
            if changelog_file:
                print(f"   Check the changelog for details: {self.data_dir / 'latest_changelog.md'}")
            else:
                # This shouldn't happen since corrections_failed triggers changelog creation
                print("   Correction failures should have triggered changelog creation.")

        return final_data

    # endregion


# region Main Entry Point
def main():
    """Main entry point for the processor.

    Handles command-line arguments and initiates processing.
    Supports various modes: full processing, single country,
    force reprocessing, debug mode, and verbose logging.

    Command-line arguments:
        --force: Force reprocess all countries
        --debug: Enable debug mode
        --country: Process specific country
        --separate-file: Save to separate file (legacy)
        --workers: Number of parallel workers
        --verbose: Enable verbose API logging

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(description="Process Red Bull editions data with Gemini AI")
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force reprocess all countries, ignoring cache",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Enable debug mode with full API request/response logging to debug file",
    )
    parser.add_argument(
        "--country",
        "-c",
        type=str,
        help='Process only a specific country (e.g., "Germany")',
    )
    parser.add_argument(
        "--separate-file",
        "-s",
        action="store_true",
        help="Save single country to separate file instead of updating final JSON",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=3,
        help="Number of parallel workers for multi-country processing (default: 3)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose mode with detailed console output",
    )

    args = parser.parse_args()

    try:
        # Create processor with appropriate settings
        # For single country, use 1 worker since we only process 1 country
        processor = RedBullDataProcessor(
            debug=args.verbose,
            max_workers=1 if args.country else args.workers,
            verbose=args.debug,
            single_country=bool(args.country),
        )

        if args.country:
            # Process single country
            print(f"🎯 Processing single country: {args.country}")

            # Discover available countries from raw files
            available_countries = processor.discover_raw_files()

            if args.country in available_countries:
                country_info = available_countries[args.country]
                result, _ = processor.process_country(
                    args.country,
                    country_info["domain"],
                    country_info["flag_code"],
                    force=args.force,  # Use force flag from arguments
                )
                if result:
                    if args.separate_file:
                        # Save to separate file (old behavior)
                        if "editions" in result:
                            result["editions"] = processor.add_description_prefix(result["editions"])

                        output = {args.country: result}
                        output[args.country].pop("_raw_hash", None)
                        output[args.country].pop("_translated_editions", None)

                        single_file = processor.data_dir / f"single_{country_info['domain']}.json"
                        with open(single_file, "w", encoding="utf-8") as file:
                            json.dump(output, file, indent=4, ensure_ascii=False)

                        print(f"✅ Single country result saved to: {single_file}")
                    else:
                        # Update final JSON directly (new default behavior)
                        success = processor.update_final_json_with_country(args.country, result)
                        if success:
                            print(f"✅ {args.country} updated in final JSON: " f"data/redbull_editions_final.json")
                        else:
                            print(f"❌ Failed to update final JSON for {args.country}")
            else:
                print(f"❌ Country '{args.country}' not found in raw data")
                print("Available countries:")
                for country in sorted(available_countries.keys()):
                    print(f"  - {country}")
        else:
            # Process all countries
            processor.process_all(force_reprocess=args.force)

    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        return 1
    except SystemExit as err:
        # If it's just an exit code (numeric), don't print it
        # If it's a message (string), print it with emoji
        if err.code and not isinstance(err.code, int):
            print(f"\n💔 {err.code}")
        return err.code if isinstance(err.code, int) else 1
    except (IOError, json.JSONDecodeError, ValueError, RuntimeError) as err:
        print(f"\n❌ Unexpected error: {err}")
        if args.debug:
            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
# endregion
