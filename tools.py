"""
External data tools: Mireye site-siting data + EIA electricity pricing,
exposed to the graph as a single LangChain tool.
"""

import logging
import re
import time

import requests
from langchain_core.tools import tool

from config import MIREYE_TOKEN, EIA_API_KEY

logger = logging.getLogger(__name__)


def call_mireye(endpoint: str, payload: dict, max_retries: int = 2) -> dict:
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(
                f"https://api.mireye.com{endpoint}",
                headers={"Authorization": f"Bearer {MIREYE_TOKEN}"},
                json=payload,
                timeout=60,
            )
            r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            last_exc = e
            if attempt < max_retries:
                logger.warning(
                    "[Mireye] %s failed (attempt %d/%d): %s — retrying",
                    endpoint, attempt + 1, max_retries + 1, e,
                )
                time.sleep(1.5 * (attempt + 1))
            else:
                logger.error("[Mireye] %s failed after %d attempts: %s", endpoint, max_retries + 1, e)
    raise last_exc


def get_electricity_price(state: str) -> dict:
    """Fetch latest monthly retail electricity price for a US state from EIA."""
    url = (
        "https://api.eia.gov/v2/electricity/retail-sales/data/"
        f"?api_key={EIA_API_KEY}"
        "&frequency=monthly"
        "&data[0]=price"
        f"&facets[stateid][]={state.upper()}"
        "&sort[0][column]=period"
        "&sort[0][direction]=desc"
        "&length=1"
    )

    last_exc = None
    response = None
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            break
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            last_exc = e
            if attempt < 2:
                logger.warning("[EIA] request failed (attempt %d/3): %s — retrying", attempt + 1, e)
                time.sleep(1.5 * (attempt + 1))
            else:
                logger.error("[EIA] request failed after 3 attempts: %s", e)
                raise last_exc

    data = response.json()["response"]["data"]
    if not data:
        return {"state": state.upper(), "period": None, "price_cents_per_kwh": None}

    result = data[0]
    return {
        "state": result["stateid"],
        "period": result["period"],
        "price_cents_per_kwh": result["price"],
    }


@tool
def get_datacenter_site_data(address: str) -> dict:
    """
    Analyze a location for data center siting.

    Args:
        address: Full address or location to analyze.

    Returns:
        A dictionary containing the geocoded location, electricity price (if available),
        and Mireye data center siting analysis.
    """
    geo = call_mireye("/v1/geocode", {"address": address})

    lat = geo["lat"]
    lng = geo["lng"]

    # Mireye's /v1/geocode response does not include a `state` field.
    # Extract the 2-letter state abbreviation from `normalized_address`
    # (format: "..., City, XX 12345"), which Geocodio always returns.
    normalized_address = geo.get("normalized_address", "") or ""
    state_match = re.search(r",\s*([A-Z]{2})\s+\d{5}", normalized_address)
    if not state_match:
        raise ValueError(
            f"Could not determine US state from geocode result for "
            f"address={address!r} (normalized_address={normalized_address!r})"
        )
    state_abbr = state_match.group(1)

    dc = call_mireye(
        "/v1/fetch",
        {
            "lat": lat,
            "lng": lng,
            "preset": "data_center_siting",
        },
    )

    electricity = get_electricity_price(state_abbr)

    mireye_fields = dc.get("fields")
    if mireye_fields is None:
        raise ValueError(
            f"Mireye /v1/fetch response for address={address!r} did not "
            f"contain a 'fields' key (keys present: {list(dc.keys())})."
        )

    return {
        "tool_output": {
            "location": {
                "address": geo.get("normalized_address"),
                "lat": lat,
                "lng": lng,
                "state": state_abbr,
            },
            "electricity": electricity,
            "mireye": mireye_fields,
        }
    }


tools = [get_datacenter_site_data]
