"""Register all data loaders with the EVER registry on import."""
from data import geofm           # registers GeoFMMultiEmbeddingLoader + GeoFMEmbeddingLoader
from data import rgb_specialist  # registers BuildingSpecialistRGBLoader
