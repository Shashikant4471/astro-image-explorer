# Install dependencies:
# pip install streamlit astroquery requests Pillow wikipedia-api

import streamlit as st
from astroquery.mast import Observations
import requests
from PIL import Image
import wikipediaapi
from astropy import units as u

# App UI
st.set_page_config(page_title="Astro Image Explorer", layout="wide")
st.title("Astro Image Explorer ⚡")
st.write("Upload a space/satellite image, interact, and get detailed info directly from NASA, JWST (MAST), GIBS, and Wikipedia.")

uploaded_file = st.file_uploader("Upload image...", type=["jpg", "png", "tif"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    # Optionally, let users select a region of interest (ROI), provide coordinates, or use image metadata for queries

    st.subheader("Satellite & Space Data")

    # Example: Querying MAST (JWST)
    st.write("**JWST (MAST) Query**")
    obs = Observations.query_object("Andromeda Galaxy", radius=0.02 * u.deg)
    result_table = obs.to_pandas()
    st.dataframe(result_table.head(3))

    # Example: Getting a tile from NASA GIBS (global coverage)
    st.write("**GIBS Quicklook**")
    gibs_url = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?SERVICE=WMS&REQUEST=GetMap&LAYERS=MODIS_Terra_CorrectedReflectance_TrueColor&FORMAT=image/png&HEIGHT=256&WIDTH=256&CRS=EPSG:4326&BBOX=-10,35,0,45"
    st.image(gibs_url, caption="GIBS Satellite Tile")

    # Example: Wikipedia Integration
    st.write("**Wikipedia Info**")
    wiki_wiki = wikipediaapi.Wikipedia('en')
    page = wiki_wiki.page("Andromeda Galaxy")
    if page.exists():
        st.write(page.summary[:500])
    else:
        st.write("No Wikipedia entry found.")

else:
    st.info("Awaiting image upload.")

st.markdown("""
---
**Tip:** Use the sidebar for API keys and advanced options!
""")
