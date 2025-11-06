from geopy.geocoders import Nominatim
import overpy
from collections import defaultdict
import json

def describe_location(lat, lon, radius=200, pretty=True, as_markdown=False):
    """
    Given a GNSS coordinate (WGS84), returns:
    - The human-readable address
    - Nearby places grouped by category (school, restaurant, etc.)
    """

    # --- Reverse geocode using OpenStreetMap ---
    geolocator = Nominatim(user_agent="geo_context_finder")
    location = geolocator.reverse((lat, lon), exactly_one=True, language='en')

    address = location.raw.get("address", {})
    info = {
        "latitude": lat,
        "longitude": lon,
        "display_name": location.address,
        "road": address.get("road"),
        "house_number": address.get("house_number"),
        "neighbourhood": address.get("neighbourhood"),
        "city": address.get("city") or address.get("town") or address.get("village"),
        "state": address.get("state"),
        "country": address.get("country"),
        "postcode": address.get("postcode")
    }

    # --- Query nearby points of interest using Overpass API ---
    api = overpy.Overpass()
    query = f"""
    [out:json];
    (
      node(around:{radius},{lat},{lon})[amenity];
      way(around:{radius},{lat},{lon})[amenity];
      relation(around:{radius},{lat},{lon})[amenity];
    );
    out center;
    """
    result = api.query(query)

    grouped = defaultdict(set)
    for node in result.nodes:
        amenity = node.tags.get("amenity")
        name = node.tags.get("name")
        if amenity:
            grouped[amenity].add(name or "(unnamed)")

    # --- Build readable output ---
    if not pretty:
        return {"address": info, "nearby_places": grouped}

    lines = []
    lines.append("**Location Summary**")
    lines.append(f"- **Full name:** {info['display_name']}")
    lines.append(f"- **Street:** {info.get('road', 'N/A')} {info.get('house_number', '')}")
    lines.append(f"- **Neighbourhood:** {info.get('neighbourhood', 'N/A')}")
    lines.append(f"- **City:** {info.get('city', 'N/A')}")
    lines.append(f"- **State:** {info.get('state', 'N/A')}")
    lines.append(f"- **Country:** {info.get('country', 'N/A')}")
    lines.append(f"- **Postal code:** {info.get('postcode', 'N/A')}")
    lines.append("")

    lines.append("🏙️ **Nearby Places (within ~200m)**")
    if not grouped:
        lines.append("_No nearby amenities found._")
    else:
        for amenity_type, names in sorted(grouped.items()):
            readable_type = amenity_type.replace("_", " ").title()
            names_list = ", ".join(sorted(names))
            lines.append(f"- **{readable_type}:** {names_list}")

    if as_markdown:
        return "\n".join(lines)
    else:
        print("\n".join(lines))
        return {"address": info, "nearby_places": grouped}

# Example
if __name__ == "__main__":
    describe_location(40.4168, -3.7038)
