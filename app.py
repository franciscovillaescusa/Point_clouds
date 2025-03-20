import streamlit as st


st.sidebar.markdown("# Cosmological datasets for geometric deep learning")
st.sidebar.write("\n\n")
st.sidebar.write("Choose your dataset:")
Quijote = st.sidebar.button("Quijote")
CAMELS_SAM = st.sidebar.button("CAMELS-SAM")
CAMELS = st.sidebar.button("CAMELS")


if Quijote:

    st.button("Point clouds: 5000")
    st.button("Point clouds: 10000")
    st.button("Point clouds: 50000")

if CAMELS_SAM:

    st.button("Point clouds: 5000")
    st.button("Point clouds: 10000")
    st.button("Point_clouds: 50000")
    st.button("Merger trees")

if CAMELS:
    
    st.button("Point clouds: 5000")
    st.button("Point clouds: 10000")
    st.button("Point_clouds: 50000")
    st.button("Merger trees")
