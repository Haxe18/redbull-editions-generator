#!/usr/bin/env python3
"""
Red Bull Editions Data Collector

Collects and merges data from multiple language versions per country.
Supports intelligent language prioritization and deduplication.
"""
# Standard library imports
import argparse
import hashlib
import json
import logging
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

# Third-party imports
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Local imports
from lib.logging_utils import setup_basic_logging, setup_logger


# Custom exceptions
class CollectorError(Exception):
    """Base exception for collector errors."""


class APIError(CollectorError):
    """Exception raised for API-related errors."""


class DataProcessingError(CollectorError):
    """Exception raised for data processing errors."""


# Configure logging
setup_basic_logging()


# Configuration constants
@dataclass(frozen=True)
class Config:
    """Configuration constants for the collector."""

    # HTTP Configuration
    USER_AGENT: str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 (+https://github.com/Haxe18/redbull-editions-generator)"
    TIMEOUT: int = 10

    # Retry Configuration
    MAX_RETRIES: int = 3
    BASE_DELAY: float = 1.0
    RETRY_STATUS_CODES: tuple = (502, 503, 504)

    # Country-Level Retry Configuration
    COUNTRY_MAX_RETRIES: int = 1  # 1 Retry = 2 attempts total
    COUNTRY_RETRY_DELAY: int = 5  # Seconds to wait between country retries

    # Rate Limiting
    RATE_LIMIT_MIN: float = 0.5
    RATE_LIMIT_MAX: float = 1.5
    DOMAIN_DELAY: float = 0.3

    # API Endpoints
    HEADER_API_URL: str = "https://www.redbull.com/v3/api/custom/header/v2"
    GRAPHQL_API_URL: str = "https://www.redbull.com/v3/api/graphql/v1/"

    # Connection Pool
    POOL_CONNECTIONS: int = 10
    POOL_MAXSIZE: int = 10
    MAX_RETRIES_ADAPTER: int = 3


# Create global config instance
config = Config()


class EditionProcessingContext(NamedTuple):
    """Context for edition processing to reduce method arguments."""

    edition: Dict
    domain: str
    uuid: str
    lang_priority: int


class RedBullDataCollector:
    """Main data collector for Red Bull editions from multiple locales.

    This collector fetches product data from Red Bull's API endpoints,
    supporting multiple language versions per country with intelligent
    merging and deduplication.

    Attributes:
        output_dir: Base directory for output data.
        raw_dir: Directory for raw API responses.
        rate_limit: Whether rate limiting is enabled.
        logger: Logger instance for output.
        headers: HTTP headers for requests.
        session: Requests session with connection pooling.
    """

    def __init__(
        self,
        output_dir: str = "data",
        rate_limit: bool = True,
        verbose: bool = False,
        debug: bool = False,
    ):
        """Initialize the collector.

        Args:
            output_dir: Directory for output data
            rate_limit: Whether to apply rate limiting
            verbose: Enable verbose output with detailed logging
            debug: Enable debug output for troubleshooting
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.raw_dir = self.output_dir / "raw"
        self.raw_dir.mkdir(exist_ok=True)
        self.rate_limit = rate_limit

        # Setup logger
        self.logger = setup_logger(self.__class__.__name__, enable_verbose=verbose, debug=debug)

        # Set user agent for all requests
        self.headers = {"User-Agent": config.USER_AGENT}

        # Setup session with connection pooling and retry strategy
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create a requests session with connection pooling and retry strategy.

        Returns:
            Configured requests session.
        """
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=config.MAX_RETRIES_ADAPTER,
            status_forcelist=list(config.RETRY_STATUS_CODES),
            allowed_methods=["GET"],
            backoff_factor=config.BASE_DELAY,
        )

        # Create adapter with connection pooling
        adapter = HTTPAdapter(
            pool_connections=config.POOL_CONNECTIONS,
            pool_maxsize=config.POOL_MAXSIZE,
            max_retries=retry_strategy,
        )

        # Mount adapter for both HTTP and HTTPS
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Set default headers
        session.headers.update(self.headers)

        return session

    def _retry_request(self, request_func: Callable, *args, identifier: str = "request", **kwargs) -> Optional[Dict[str, Any]]:
        """Generic retry logic for HTTP requests.

        Args:
            request_func: Function to call for the request.
            *args: Positional arguments for the request function.
            identifier: Identifier for logging purposes.
            **kwargs: Keyword arguments for the request function.

        Returns:
            Response data or None if all retries failed.
        """
        for attempt in range(config.MAX_RETRIES + 1):
            try:
                response = request_func(*args, **kwargs)
                response.raise_for_status()
                return response.json()

            except requests.exceptions.HTTPError as err:
                if err.response.status_code in config.RETRY_STATUS_CODES:
                    if attempt < config.MAX_RETRIES:
                        delay = config.BASE_DELAY * (2**attempt) + random.uniform(0, 1)
                        self.logger.warning(
                            "HTTP %s error for %s (attempt %d/%d), retrying in %.1fs",
                            err.response.status_code,
                            identifier,
                            attempt + 1,
                            config.MAX_RETRIES + 1,
                            delay,
                        )
                        time.sleep(delay)
                        continue
                    self.logger.error(
                        "Error fetching %s: %s %s (all retries failed)",
                        identifier,
                        err.response.status_code,
                        err.response.reason,
                    )
                    return None
                # Non-retryable HTTP error
                self.logger.error(
                    "Error fetching %s: %s %s",
                    identifier,
                    err.response.status_code,
                    err.response.reason,
                )
                return None

            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ) as err:
                if attempt < config.MAX_RETRIES:
                    delay = config.BASE_DELAY * (2**attempt) + random.uniform(0, 1)
                    self.logger.warning(
                        "Connection error for %s (attempt %d/%d), retrying in %.1fs",
                        identifier,
                        attempt + 1,
                        config.MAX_RETRIES + 1,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                self.logger.error(
                    "Error fetching %s: %s (all retries failed)",
                    identifier,
                    str(err)[:50],
                )
                return None

            except (requests.RequestException, ValueError) as err:
                # Non-retryable errors
                self.logger.error("Error fetching %s: %s", identifier, err)
                return None

        return None

    def fetch_initial_locales(self) -> Dict[str, Any]:
        """Fetch the initial API to get all available locales.

        Returns:
            Dict containing available locales and country information.

        Raises:
            requests.HTTPError: If the API request fails.
        """
        params = {"locale": "int-en"}

        result = self._retry_request(
            self.session.get,
            config.HEADER_API_URL,
            params=params,
            timeout=config.TIMEOUT,
            identifier="initial locales",
        )

        if not result:
            raise APIError("Failed to fetch initial locales")
        return result

    def fetch_country_data(self, domain: str, custom_url: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Fetch API data for a specific country/locale.

        Supports custom URLs including Archive.org snapshots.

        Args:
            domain: The locale domain to fetch (e.g., 'us-en', 'de-de').
            custom_url: Optional custom URL to fetch from (e.g., Archive.org URL).

        Returns:
            API response data or None if all retries failed.
        """
        if custom_url:
            url = custom_url
            params = {}  # Archive.org URLs already include parameters
        else:
            url = config.HEADER_API_URL
            params = {"locale": domain}

        return self._retry_request(
            self.session.get,
            url,
            params=params,
            timeout=config.TIMEOUT,
            identifier=domain,
        )

    @staticmethod
    def clean_proxy_url(url: str) -> str:
        """Remove proxy prefixes by finding the second http(s) occurrence.

        Works generically for any proxy service that prepends to the original URL.
        Examples:
        - https://web.archive.org/web/123/https://example.com -> https://example.com
        - https://proxy.com/cache/http://site.com -> http://site.com

        Args:
            url: URL that may contain proxy prefix.

        Returns:
            Cleaned URL with proxy prefix removed, or original if no prefix found.
        """
        if not url:
            return url

        # Find all occurrences of http:// or https://
        matches = list(re.finditer(r"https?://", url))

        if len(matches) >= 2:
            # Return from the second http(s) onwards
            return url[matches[1].start() :]

        return url

    @staticmethod
    def filter_graphql_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter GraphQL data to keep only required fields.

        Reduces API response size by keeping only necessary fields:
        flavour, standfirst, brandingHexColorCode, image, and reference.

        Args:
            data: Raw GraphQL response data.

        Returns:
            Filtered data with only required fields.
        """
        if not data or "data" not in data:
            return data

        filtered = {"data": {}}
        original = data["data"]

        # Copy only needed fields with nested filtering
        if "flavour" in original:
            filtered["data"]["flavour"] = original["flavour"]

        if "standfirst" in original:
            filtered["data"]["standfirst"] = original["standfirst"]

        if "brandingHexColorCode" in original:
            filtered["data"]["brandingHexColorCode"] = original["brandingHexColorCode"]

        # Filter image to keep only what we need
        if "image" in original and original["image"]:
            filtered_image = {}
            if "altText" in original["image"]:
                filtered_image["altText"] = original["image"]["altText"]
            if "imageEssence" in original["image"] and "imageURL" in original["image"]["imageEssence"]:
                filtered_image["imageEssence"] = {"imageURL": original["image"]["imageEssence"]["imageURL"]}
            if filtered_image:
                filtered["data"]["image"] = filtered_image

        # Filter reference to keep only externalUrl
        if "reference" in original and original["reference"] and "externalUrl" in original["reference"]:
            filtered["data"]["reference"] = {"externalUrl": original["reference"]["externalUrl"]}

        return filtered

    def filter_header_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter header data to keep only required fields.

        Reduces header API response by keeping only:
        content.title, media.mainImage, and reference.externalUrl.

        Args:
            data: Raw header API response data.

        Returns:
            Filtered data with only required fields.
        """
        if not data:
            return data

        filtered = {}

        # Keep content.title
        if "content" in data and "title" in data["content"]:
            filtered["content"] = {"title": data["content"]["title"]}

        # Keep media.mainImage fields we need
        if "media" in data and "mainImage" in data["media"]:
            main_image = data["media"]["mainImage"]
            filtered_image = {}

            if "imageEssence" in main_image:
                filtered_image["imageEssence"] = {}
                if "imageURL" in main_image["imageEssence"]:
                    # Clean proxy URL if present
                    filtered_image["imageEssence"]["imageURL"] = self.clean_proxy_url(main_image["imageEssence"]["imageURL"])
                if "predominantColors" in main_image["imageEssence"]:
                    filtered_image["imageEssence"]["predominantColors"] = main_image["imageEssence"]["predominantColors"]

            if "altText" in main_image:
                filtered_image["altText"] = main_image["altText"]

            if filtered_image:
                filtered["media"] = {"mainImage": filtered_image}

        # Keep reference.externalUrl
        if "reference" in data and "externalUrl" in data["reference"]:
            # Clean proxy URL if present
            filtered["reference"] = {"externalUrl": self.clean_proxy_url(data["reference"]["externalUrl"])}

        return filtered

    def fetch_graphql_details(self, graphql_id: str) -> Optional[Dict[str, Any]]:
        """Fetch detailed information for an edition via GraphQL.

        Args:
            graphql_id: The GraphQL resource ID for the edition.

        Returns:
            Filtered GraphQL data or None if request failed.
        """
        params = {"rb3ResourceId": graphql_id, "rb3Schema": "v1:assetInfo"}

        self.logger.debug("Fetching GraphQL for ID: %s", graphql_id)

        data = self._retry_request(
            self.session.get,
            config.GRAPHQL_API_URL,
            params=params,
            timeout=config.TIMEOUT,
            identifier=f"GraphQL {graphql_id[:20]}...",
        )

        if data:
            if data.get("data"):
                flavor = data["data"].get("flavour", "N/A")
                self.logger.debug("Flavor: %s", flavor)

            # Filter to keep only needed fields
            return self.filter_graphql_data(data)

        return None

    @staticmethod
    def get_language_priority(domain: str) -> int:
        """Get language priority (lower = higher priority).

        Determines priority based on language code:
        1. English (en, gb, us)
        2. German/Dutch (de, nl)
        3. Other European languages

        Args:
            domain: The locale domain (e.g., 'us-en', 'de-de').

        Returns:
            Priority number (lower = higher priority).
        """
        if not domain or "-" not in domain:
            return 999

        lang = domain.split("-")[-1].lower()
        # Priority: English first, then German/Dutch, then others
        priorities = {
            "en": 1,  # English
            "gb": 1,  # UK English
            "us": 1,  # US English
            "de": 2,  # German
            "nl": 2,  # Dutch
            "at": 3,  # Austrian German
            "ch": 3,  # Swiss German
            "fr": 4,  # French
            "es": 5,  # Spanish
            "pt": 6,  # Portuguese
            "it": 7,  # Italian
        }
        return priorities.get(lang, 99)

    def get_base_edition_id(self, edition_id: str) -> str:
        """Extract UUID from edition ID.

        Extracts the UUID from full edition IDs in format:
        'rrn:content:energy-drinks:UUID:locale'

        Args:
            edition_id: Full edition ID string.

        Returns:
            Extracted UUID or original ID if extraction fails.
        """
        if edition_id and "energy-drinks:" in edition_id:
            # Extract UUID between 'energy-drinks:' and the last ':'
            parts = edition_id.split(":")
            if len(parts) >= 5:
                # UUID is at position 3 (0-indexed)
                uuid = parts[3]
                self.logger.debug("Extracted UUID %s from %s", uuid, edition_id)
                return uuid
        return edition_id

    def _process_edition_by_language(self, context: EditionProcessingContext, merged: Dict, uuid_lang_priority: Dict) -> None:
        """Process a single edition based on language priority.

        Helper method to reduce complexity of merge_editions.

        Args:
            context: EditionProcessingContext with edition data
            merged: Dictionary of merged editions
            uuid_lang_priority: Dictionary tracking language priorities
        """
        should_use = False
        edition, domain, uuid, lang_priority = context

        # Check if we've seen this UUID before
        if uuid not in merged:
            # First time seeing this UUID
            should_use = True
            uuid_lang_priority[uuid] = lang_priority
            title = edition.get("header_data", {}).get("content", {}).get("title", "N/A")
            self.logger.debug("New UUID %s... - '%s' from %s", uuid[:8], title, domain)
        elif uuid_lang_priority[uuid] > lang_priority:
            # Better language version of same product (lower priority number = higher priority)
            should_use = True
            title = edition.get("header_data", {}).get("content", {}).get("title", "N/A")
            self.logger.debug(
                "Better language for UUID %s... - '%s' from %s (priority %d < %d)",
                uuid[:8],
                title,
                domain,
                lang_priority,
                uuid_lang_priority[uuid],
            )
            uuid_lang_priority[uuid] = lang_priority
        else:
            title = edition.get("header_data", {}).get("content", {}).get("title", "N/A")
            self.logger.debug(
                "Skipping UUID %s... - '%s' from %s (priority %d >= %d)",
                uuid[:8],
                title,
                domain,
                lang_priority,
                uuid_lang_priority[uuid],
            )

        if should_use and uuid:
            edition["_lang_priority"] = lang_priority
            edition["_source_domain"] = domain
            merged[uuid] = edition

    def _deduplicate_by_flavor(self, editions: List[Dict]) -> List[Dict]:
        """Deduplicate editions based on flavor to handle promotional duplicates.

        Helper method to reduce complexity of merge_editions.
        """
        flavor_map = {}
        for edition in editions:
            # Get flavor from GraphQL data
            graphql_data = edition.get("graphql_data", {})
            if graphql_data and "data" in graphql_data:
                flavor = graphql_data["data"].get("flavour")
                title = edition.get("header_data", {}).get("content", {}).get("title", "")

                if flavor:
                    # Create unique key combining flavor and title to keep variants
                    flavor_key = f"{flavor}|{title}"

                    self.logger.debug("Processing flavor key: %s", flavor_key)

                    # Get URL to check if it's a product page
                    url = edition.get("header_data", {}).get("reference", {}).get("externalUrl", "")

                    if flavor_key not in flavor_map:
                        # First time seeing this flavor/title combination
                        self.logger.debug("Adding new flavor/title combination")
                        flavor_map[flavor_key] = edition
                    else:
                        # Duplicate flavor/title found - prefer product URLs over general pages
                        self.logger.debug("Duplicate found, evaluating which to keep")
                        existing_edition = flavor_map[flavor_key]
                        existing_url = existing_edition.get("header_data", {}).get("reference", {}).get("externalUrl", "")

                        # Prefer entries with /products/ in URL (actual product pages)
                        current_has_products = "/products/" in url
                        existing_has_products = "/products/" in existing_url

                        if current_has_products and not existing_has_products:
                            # Current edition has product URL, existing doesn't - replace
                            flavor_map[flavor_key] = edition
                        elif not current_has_products and existing_has_products:
                            # Keep existing (it has product URL, current doesn't)
                            pass
                        else:
                            # Both have or both don't have /products/
                            # Keep the one with shorter URL (usually more specific)
                            if len(url) < len(existing_url):
                                flavor_map[flavor_key] = edition
                else:
                    # No flavor data - keep the edition
                    unique_key = f"no_flavor_{edition.get('id', '')}"
                    flavor_map[unique_key] = edition
            else:
                # No GraphQL data - keep the edition
                unique_key = f"no_graphql_{edition.get('id', '')}"
                flavor_map[unique_key] = edition

        return list(flavor_map.values())

    def _filter_int_editions(self, editions: List[Dict], flag_code: str) -> List[Dict]:
        """Filter out INT editions for non-INT countries.

        Helper method to reduce complexity of merge_editions.
        """
        if not flag_code or flag_code == "INT":
            return editions

        filtered_editions = []
        for edition in editions:
            edition_id = edition.get("id", "")
            # Check if the edition ID ends with -INT (international edition)
            if edition_id and edition_id.endswith("-INT"):
                title = edition.get("header_data", {}).get("content", {}).get("title", "Unknown")
                self.logger.debug(
                    "Filtering out INT edition for %s: %s (%s)",
                    flag_code,
                    title,
                    edition_id,
                )
            else:
                filtered_editions.append(edition)

        if len(filtered_editions) < len(editions):
            self.logger.info(
                "Filtered out %d INT edition(s) for country %s",
                len(editions) - len(filtered_editions),
                flag_code,
            )

        return filtered_editions

    def merge_editions(self, all_editions: List[List[Dict]], domains: List[str], flag_code: str = "") -> List[Dict]:
        """Merge editions from multiple language versions based on UUID.

        Intelligently merges editions from different language versions,
        prioritizing English and German/Dutch versions. Also handles
        deduplication of promotional duplicates while preserving
        product variants like Sugarfree versions.

        For countries with specific flag codes (not INT), filters out
        international editions (-INT suffix) that shouldn't be included
        in local country data.

        Args:
            all_editions: List of edition lists from different locales.
            domains: Corresponding domain names for each edition list.
            flag_code: Country flag code to determine INT filtering.

        Returns:
            Merged and deduplicated list of editions.
        """
        # Track editions by UUID
        merged = {}
        uuid_lang_priority = {}  # Track language priority per UUID

        # Process editions by language priority
        for editions, domain in zip(all_editions, domains):
            lang_priority = self.get_language_priority(domain)

            for edition in editions:
                edition_id = edition.get("id", "")
                uuid = self.get_base_edition_id(edition_id) if edition_id else ""

                # Skip if no valid UUID
                if not uuid or uuid == edition_id:
                    # Fallback: use full ID if UUID extraction failed
                    uuid = edition_id

                # Process edition based on language priority
                context = EditionProcessingContext(edition, domain, uuid, lang_priority)
                self._process_edition_by_language(context, merged, uuid_lang_priority)

        # Clean up internal fields and prepare for flavor deduplication
        result = []
        for edition in merged.values():
            edition.pop("_lang_priority", None)
            edition.pop("_source_domain", None)
            result.append(edition)

        # Additional deduplication based on flavor to handle promotional duplicates
        final_editions = self._deduplicate_by_flavor(result)

        # Filter out INT editions for non-INT countries
        return self._filter_int_editions(final_editions, flag_code)

    def _group_locales_by_country(self, locales: List[Dict]) -> Dict[str, List[Dict]]:
        """Group locales by country and sort by language priority.

        Helper method to reduce complexity of collect_all_data.
        """
        countries_grouped = {}
        for locale in locales:
            country_name = locale.get("countryName", locale.get("label", "Unknown"))
            if country_name not in countries_grouped:
                countries_grouped[country_name] = []
            countries_grouped[country_name].append(locale)

        # Sort each country's locales by language priority
        for country_name, country_locales in countries_grouped.items():
            country_locales.sort(key=lambda x: self.get_language_priority(x.get("domain", "")))

        return countries_grouped

    def _filter_countries_by_name(self, countries_grouped: Dict, country_filter: str) -> Dict:
        """Filter countries by name with partial matching support.

        Helper method to reduce complexity of collect_all_data.
        """
        # Case-insensitive search for country
        matched_country = None
        for country_name in countries_grouped:
            if country_name.lower() == country_filter.lower():
                matched_country = country_name
                break

        if matched_country:
            self.logger.info("🎯 Filtering for country: %s", matched_country)
            return {matched_country: countries_grouped[matched_country]}

        # Try partial match
        partial_matches = [country for country in countries_grouped if country_filter.lower() in country.lower()]

        if len(partial_matches) == 1:
            matched_country = partial_matches[0]
            self.logger.info("🎯 Found partial match: %s", matched_country)
            return {matched_country: countries_grouped[matched_country]}
        if len(partial_matches) > 1:
            self.logger.info("❌ Multiple countries match '%s':", country_filter)
            for country in partial_matches[:10]:  # Show max 10 matches
                self.logger.info("  • %s", country)
            self.logger.info("\nPlease be more specific.")
            return {}
        # No matches found
        self.logger.info("❌ Country '%s' not found.", country_filter)
        self.logger.info("\nAvailable countries:")
        for country in sorted(countries_grouped.keys())[:20]:  # Show first 20
            self.logger.info("  • %s", country)
        if len(countries_grouped) > 20:
            self.logger.info("  ... and %d more", len(countries_grouped) - 20)
        return {}

    def _collect_domain_editions(self, domain: str, country_name: str, custom_url: Optional[str] = None) -> Tuple[List[Dict], bool]:
        """Collect editions from a single domain.

        Args:
            domain: Domain to collect from.
            country_name: Name of the country for logging.
            custom_url: Optional custom URL.

        Returns:
            Tuple of (editions list, success boolean).
        """
        # Fetch country data
        if custom_url:
            country_data = self.fetch_country_data(domain, custom_url)
        else:
            country_data = self.fetch_country_data(domain)

        if not country_data:
            self.logger.error("Country %s marked as FAILED due to header API failure", country_name)
            return [], False

        featured_drinks = country_data.get("featuredEnergyDrinks", [])
        domain_editions = []

        # Show rate limit info once per domain in verbose mode
        if self.rate_limit and featured_drinks:
            self.logger.verbose(
                "  ℹ️  Using random rate limits (%.1f-%.1fs) to avoid API throttling",
                config.RATE_LIMIT_MIN,
                config.RATE_LIMIT_MAX,
            )

        for drink in featured_drinks:
            graphql_id = drink.get("id")

            edition_raw = {
                "id": graphql_id,
                "header_data": self.filter_header_data(drink),
                "graphql_data": None,
            }

            if graphql_id:
                # Show edition name in verbose mode
                edition_title = drink.get("content", {}).get("title", "Unknown")
                self.logger.verbose("  🔍 Fetching: %s", edition_title)

                graphql_data = self.fetch_graphql_details(graphql_id)
                if graphql_data is None:
                    self.logger.error(
                        "Country %s marked as FAILED due to GraphQL API failure",
                        country_name,
                    )
                    return [], False
                edition_raw["graphql_data"] = graphql_data

                # Rate limiting
                if self.rate_limit:
                    sleep_time = random.uniform(config.RATE_LIMIT_MIN, config.RATE_LIMIT_MAX)
                    self.logger.verbose("    ⏱️  Rate limit: sleeping %.1fs", sleep_time)
                    time.sleep(sleep_time)

            domain_editions.append(edition_raw)

        return domain_editions, True

    def _save_country_data(
        self,
        country_name: str,
        merged_editions: List[Dict],
        valid_domains: List[str],
        flag_code: str,
    ) -> Tuple[str, bool]:
        """Save country data to file and check for changes.

        Args:
            country_name: Name of the country.
            merged_editions: Merged edition data.
            valid_domains: List of valid domains.
            flag_code: Country flag code.

        Returns:
            Tuple of (file name, has_changed boolean).
        """
        primary_domain = valid_domains[0] if valid_domains else "unknown"
        country_raw_data = {
            "locale_info": {
                "country_name": country_name,
                "domains": valid_domains,
                "flag_code": flag_code,
            },
            "editions": merged_editions,
            "collection_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Save to file
        raw_file = self.raw_dir / f"{primary_domain}.json"

        # Check for changes
        has_changed = True
        if raw_file.exists():
            with open(raw_file, "r", encoding="utf-8") as file_handle:
                existing = json.load(file_handle)
                existing_hash = hashlib.sha256(json.dumps(existing.get("editions", []), sort_keys=True).encode()).hexdigest()
                new_hash = hashlib.sha256(json.dumps(merged_editions, sort_keys=True).encode()).hexdigest()
                has_changed = existing_hash != new_hash

        if has_changed:
            with open(raw_file, "w", encoding="utf-8") as file_handle:
                json.dump(country_raw_data, file_handle, indent=4, ensure_ascii=False)
            self.logger.info("Changes detected for %s, saved to %s", country_name, raw_file.name)
        else:
            self.logger.info("No changes detected for %s", country_name)

        return raw_file.name, has_changed

    def _print_collection_stats(
        self,
        countries_processed: int,
        failed_countries: List[str],
        changes_detected: List[str],
        skipped_countries: List[str],
    ) -> None:
        """Print collection statistics.

        Args:
            countries_processed: Number of countries processed.
            failed_countries: List of failed countries.
            changes_detected: List of countries with changes.
            skipped_countries: List of countries with no data.
        """
        self.logger.info("\n✨ Collection complete!")
        self.logger.info("📊 Stats:")
        self.logger.info("  • Countries processed: %d", countries_processed)
        self.logger.info(
            "  • Successful countries: %d",
            countries_processed - len(failed_countries) - len(skipped_countries),
        )
        self.logger.info("  • Skipped countries (no data): %d", len(skipped_countries))
        self.logger.info("  • Failed countries: %d", len(failed_countries))

        if skipped_countries:
            skipped_str = ", ".join(skipped_countries[:5])
            if len(skipped_countries) > 5:
                skipped_str += f" and {len(skipped_countries) - 5} more"
            self.logger.info("  • Skipped: %s", skipped_str)

        if failed_countries:
            failed_str = ", ".join(failed_countries[:5])
            if len(failed_countries) > 5:
                failed_str += f" and {len(failed_countries) - 5} more"
            self.logger.info("  • Failed: %s", failed_str)

        self.logger.info("  • Changes detected in: %d countries", len(changes_detected))

        if changes_detected:
            changes_str = ", ".join(changes_detected[:5])
            if len(changes_detected) > 5:
                changes_str += f" and {len(changes_detected) - 5} more"
            self.logger.info("  • Changed countries: %s", changes_str)

    @staticmethod
    def _initialize_collection_metadata(countries_count: int) -> Dict[str, Any]:
        """Initialize metadata for collection results.

        Args:
            countries_count: Number of countries to process

        Returns:
            Dictionary with initialized metadata structure
        """
        return {
            "metadata": {
                "collection_date": datetime.now(timezone.utc).isoformat(),
                "total_countries": countries_count,
            },
            "countries": {},
        }

    def _process_single_country(
        self,
        country_name: str,
        country_locales: List[Dict],
        country_idx: int,
        total: int,
        custom_url: Optional[str],
        country_filter: Optional[str],
    ) -> Tuple[Dict, bool, str]:
        """Process a single country's data collection.

        Args:
            country_name: Name of the country
            country_locales: List of locales for the country
            country_idx: Current country index
            total: Total number of countries
            custom_url: Optional custom URL
            country_filter: Optional country filter

        Returns:
            Tuple of (country_data, success, error_message)
        """
        flag_code = country_locales[0].get("flagCode", "")
        domains = [loc.get("domain") for loc in country_locales if loc.get("domain")]

        self.logger.info("[%d/%d] Processing %s...", country_idx, total, country_name)
        if len(domains) > 1:
            self.logger.info("  🌐 Multiple languages: %s", ", ".join(domains))

        # Collect from all language versions
        all_editions = []
        valid_domains = []

        for domain in domains:
            # Use custom URL only for the specified country
            custom_url_for_domain = custom_url if custom_url and country_filter and country_name == country_filter else None

            domain_editions, success = self._collect_domain_editions(domain, country_name, custom_url_for_domain)

            if not success:
                return {}, False, f"API error for {country_name}"

            if domain_editions:
                all_editions.append(domain_editions)
                valid_domains.append(domain)
                self.logger.verbose("  ✅ %s: Found %d editions", domain, len(domain_editions))
                # In normal mode: only show domain count if multiple
                if len(domains) > 1:
                    self.logger.info("  ✅ %s: Found %d editions", domain, len(domain_editions))

            # Rate limiting between domains
            if self.rate_limit and len(domains) > 1:
                self.logger.verbose("  ⏱️  Domain delay: sleeping %.1fs", config.DOMAIN_DELAY)
                time.sleep(config.DOMAIN_DELAY)

        if not all_editions:
            self.logger.warning("  ⚠️  No data collected for %s", country_name)
            return {}, False, ""  # Empty error means no data but not a failure

        # Merge editions from all languages
        total_before = sum(len(editions) for editions in all_editions)
        self.logger.verbose(
            "  🔄 Merging %d editions from %d language(s)",
            total_before,
            len(all_editions),
        )

        merged_editions = self.merge_editions(all_editions, valid_domains, flag_code)
        self.logger.info("  📦 Total unique editions: %d", len(merged_editions))

        removed_count = total_before - len(merged_editions)
        if removed_count > 0:
            self.logger.verbose(
                "  ♻️  Deduplication removed %d duplicate(s)",
                removed_count,
            )

        # Save raw data
        file_name, has_changed = self._save_country_data(country_name, merged_editions, valid_domains, flag_code)

        # Create country data for summary
        primary_domain = valid_domains[0] if valid_domains else "unknown"
        country_data = {
            "domain": primary_domain,
            "all_domains": valid_domains,
            "flag_code": flag_code,
            "editions_count": len(merged_editions),
            "data_file": file_name,
        }

        return country_data, has_changed, ""

    def _process_single_country_with_retry(
        self,
        country_name: str,
        country_locales: List[Dict],
        country_idx: int,
        total: int,
        custom_url: Optional[str],
        country_filter: Optional[str],
    ) -> Tuple[Dict, bool]:
        """Process a single country with retry logic.

        Attempts to process a country up to COUNTRY_MAX_RETRIES + 1 times.
        Raises APIError if all attempts fail.

        Args:
            country_name: Name of the country
            country_locales: List of locales for the country
            country_idx: Current country index
            total: Total number of countries
            custom_url: Optional custom URL
            country_filter: Optional country filter

        Returns:
            Tuple of (country_data, has_changed boolean)

        Raises:
            APIError: If all retry attempts fail
        """
        last_error = None

        for attempt in range(config.COUNTRY_MAX_RETRIES + 1):
            try:
                country_data, has_changed, error = self._process_single_country(
                    country_name,
                    country_locales,
                    country_idx,
                    total,
                    custom_url,
                    country_filter,
                )

                # Success - no error
                if not error:
                    return country_data, has_changed

                # Error occurred
                last_error = error

                # Check if we should retry
                if attempt < config.COUNTRY_MAX_RETRIES:
                    self.logger.warning(
                        "  🔄 Retrying %s in %d seconds (attempt %d/%d)...",
                        country_name,
                        config.COUNTRY_RETRY_DELAY,
                        attempt + 2,
                        config.COUNTRY_MAX_RETRIES + 1,
                    )
                    time.sleep(config.COUNTRY_RETRY_DELAY)
                    continue

                # All retries exhausted
                raise APIError(f"{country_name} failed after {config.COUNTRY_MAX_RETRIES + 1} " f"attempts: {error}")

            except APIError:
                # Re-raise APIError from nested calls
                raise
            except Exception as unexpected_error:  # pylint: disable=broad-exception-caught
                # Unexpected error - raise immediately
                raise APIError(f"{country_name} unexpected error: {str(unexpected_error)}") from unexpected_error

        # Should never reach here, but just in case
        raise APIError(f"{country_name} failed: {last_error if last_error else 'Unknown error'}")

    def collect_all_data(
        self,
        limit: Optional[int] = None,
        country_filter: Optional[str] = None,
        custom_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Main collection process with intelligent merging.

        Collects data from all available countries, handling multiple
        language versions per country and merging them intelligently.

        Args:
            limit: Optional limit on number of countries to process.
            country_filter: Optional specific country name to collect.
            custom_url: Optional custom URL to fetch data from (e.g., Archive.org).

        Returns:
            Dictionary containing collected data and metadata including:
            - metadata: Collection date, statistics, changes
            - countries: Raw data for each country

        Note:
            Failed countries are tracked and reported in metadata.
        """
        self.logger.info("🚀 Starting Red Bull Editions data collection ...")

        try:
            # Get all locales
            self.logger.info("📍 Fetching available locales...")
            initial_data = self.fetch_initial_locales()
            locales = initial_data.get("selectableLocales", [])
        except APIError as api_error:
            self.logger.error("Failed to fetch initial locales: %s", api_error)
            return {"metadata": {"error": str(api_error)}, "countries": {}}

        # Group locales by country
        countries_grouped = self._group_locales_by_country(locales)

        # Filter by specific country if requested
        if country_filter:
            countries_to_process = self._filter_countries_by_name(countries_grouped, country_filter)
            if not countries_to_process:
                return {
                    "metadata": {"error": "Country not found or multiple matches"},
                    "countries": {},
                }
        # Apply limit if specified
        elif limit:
            countries_to_process = dict(list(countries_grouped.items())[:limit])
        else:
            countries_to_process = countries_grouped

        self.logger.info("📊 Found %d unique countries", len(countries_grouped))
        self.logger.info("✅ Will process %d countries\n", len(countries_to_process))

        all_raw_data = self._initialize_collection_metadata(len(countries_to_process))

        changes_detected = []

        for country_idx, (country_name, country_locales) in enumerate(countries_to_process.items(), 1):
            # Use retry method - raises APIError on failure (stops immediately)
            country_data, has_changed = self._process_single_country_with_retry(
                country_name,
                country_locales,
                country_idx,
                len(countries_to_process),
                custom_url,
                country_filter,
            )

            # Only reached if successful
            all_raw_data["countries"][country_name] = country_data
            if has_changed:
                changes_detected.append(country_name)

        # Save collection summary
        all_raw_data["metadata"]["changes_detected"] = changes_detected
        all_raw_data["metadata"]["successful_countries"] = len(countries_to_process)

        try:
            summary_file = self.output_dir / "collection_summary.json"
            with open(summary_file, "w", encoding="utf-8") as file_handle:
                json.dump(all_raw_data, file_handle, indent=4, ensure_ascii=False)
        except (IOError, OSError) as io_error:
            self.logger.error("Failed to save collection summary: %s", io_error)

        # Print collection statistics
        self.logger.info("\n✨ Collection complete!")
        self.logger.info("📊 Stats:")
        self.logger.info("  • Countries processed: %d", len(countries_to_process))
        self.logger.info("  • Successful countries: %d", len(countries_to_process))
        self.logger.info("  • Changes detected in: %d countries", len(changes_detected))

        if changes_detected:
            changes_str = ", ".join(changes_detected[:5])
            if len(changes_detected) > 5:
                changes_str += f" and {len(changes_detected) - 5} more"
            self.logger.info("  • Changed countries: %s", changes_str)

        return all_raw_data


def main():
    """Main entry point for the Red Bull editions data collector.

    Handles command-line arguments and initializes the collection process.
    Supports various modes including full collection, specific country
    filtering, and adjustable output verbosity.

    Command-line arguments:
        --country: Collect specific country (e.g., --country Germany)
        --no-rate-limit: Disable rate limiting (development mode)
        --verbose: Enable verbose mode with detailed console output
        --debug: Enable debug mode for troubleshooting
        [number]: Collect from specified number of countries

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(description="Collect Red Bull editions data")
    parser.add_argument("limit", type=int, nargs="?", help="Number of countries to process (optional)")
    parser.add_argument(
        "--country",
        "-c",
        type=str,
        help='Collect data for a specific country (e.g., Germany, "United States")',
    )
    parser.add_argument(
        "--custom-url",
        type=str,
        help="Custom URL to fetch data from (requires --country option, " "e.g., Archive.org URL for countries without current data)",
    )
    parser.add_argument(
        "--no-rate-limit",
        "-n",
        action="store_true",
        help="Disable rate limiting (for development)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose mode with detailed console output",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Enable debug mode for troubleshooting",
    )

    args = parser.parse_args()

    # Validate custom-url requires country
    if args.custom_url and not args.country:
        parser.error("--custom-url requires --country option")

    # Determine limit
    limit = args.limit if args.limit else None

    # Create collector with rate limit, verbose and debug settings
    collector = RedBullDataCollector(rate_limit=not args.no_rate_limit, verbose=args.verbose, debug=args.debug)

    # Setup logger for main function
    logger = logging.getLogger("Main")

    if args.no_rate_limit:
        logger.info("⚡ Rate limiting disabled - development mode\n")
    if args.verbose:
        logger.info("📢 Verbose mode enabled\n")
    if args.debug:
        logger.info("🔍 Debug mode enabled\n")

    try:
        # Pass country filter and custom URL if specified
        collector.collect_all_data(limit=limit, country_filter=args.country, custom_url=args.custom_url)
        return 0
    except APIError as api_error:
        logger.error("\n❌ COLLECTION FAILED")
        logger.error("Error: %s", str(api_error))
        logger.error("Collection aborted - no partial data saved")
        return 1
    except CollectorError as collector_error:
        logger.error("\n❌ COLLECTION FAILED")
        logger.error("Error: %s", str(collector_error))
        return 1
    except KeyboardInterrupt:
        logger.info("\n🛑 Collection interrupted by user")
        return 1
    except (ValueError, TypeError, AttributeError) as type_error:
        logger.exception("\n❌ Unexpected error: %s", type_error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
