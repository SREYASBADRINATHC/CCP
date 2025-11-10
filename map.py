import folium

geojson_data = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"length_m": 120.0, "confidence": 0.9},
            "geometry": {
                "type": "LineString",
                "coordinates": [[80.25, 13.05], [80.35, 13.05]]
            }
        },
        {
            "type": "Feature",
            "properties": {"length_m": 0.002, "confidence": 0.5},
            "geometry": {
                "type": "LineString",
                "coordinates": [[80.1416008, 12.9531946], [80.1416008, 12.9531946]]
            }
        }
    ]
}

m = folium.Map(location=[13.08, 80.27], zoom_start=15)

folium.GeoJson(
    geojson_data,
    name="Detected Changes",
    popup=folium.GeoJsonPopup(fields=['length_m', 'confidence'], aliases=['Length (m):', 'Confidence:'])
).add_to(m)

m.save("change_map_with_popups.html")
print("Map with popups saved to change_map_with_popups.html")
