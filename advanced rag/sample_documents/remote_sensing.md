# Remote Sensing and Earth Observation

Remote sensing obtains information about an object or area without direct physical contact. Earth-observation satellites measure electromagnetic energy reflected or emitted by the surface. Passive sensors use naturally available energy, usually sunlight or thermal emission. Active sensors, including radar and lidar, transmit energy and measure its return.

Spatial resolution is the ground area represented by one pixel. Spectral resolution describes how finely a sensor separates wavelength bands. Temporal resolution is how often a sensor revisits the same area, while radiometric resolution describes its sensitivity to differences in energy.

The Normalized Difference Vegetation Index (NDVI) is calculated as (NIR - Red) / (NIR + Red), using near-infrared and red reflectance. Healthy green vegetation typically absorbs red light and strongly reflects near-infrared light, producing a higher NDVI. Analysts use NDVI to monitor vegetation vigor, crop condition, drought impacts, and seasonal change. Clouds, bare soil, atmosphere, and sensor differences can affect interpretation, so NDVI should not be treated as a universal measure of crop yield.

Image preprocessing may include atmospheric correction, geometric correction, cloud masking, and mosaicking. Classification converts imagery into thematic land-cover classes. Supervised classification learns from labeled training samples; unsupervised classification groups spectrally similar pixels before an analyst assigns meaning to the groups.

