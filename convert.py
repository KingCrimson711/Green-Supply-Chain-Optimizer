import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# Load CSV with at least columns: from_lat, from_lon, to_lat, to_lon, distance
df = pd.read_csv("global_shipping_lanes_graph.csv")

# Initialize geolocator with user agent
geolocator = Nominatim(user_agent="global_shipping_lanes_graph.csv")
geocode = RateLimiter(geolocator.reverse, min_delay_seconds=1)

def get_port_name(lat, lon):
    try:
        location = geocode((lat, lon), language='en', exactly_one=True)
        if location and location.raw.get('address'):
            address = location.raw['address']
            return address.get('harbour') or address.get('port') or address.get('town') or address.get('city') or address.get('village') or location.address
        return None
    except:
        return None

# Apply to columns to generate port names for nodes (may take time for large data)
df['node1'] = df.apply(lambda row: get_port_name(row['from_lat'], row['from_lon']), axis=1)
df['node2'] = df.apply(lambda row: get_port_name(row['to_lat'], row['to_lon']), axis=1)

# Select final columns required
result = df[['node1', 'node2', 'distance']]

# Display the first few rows
print(result.head())
