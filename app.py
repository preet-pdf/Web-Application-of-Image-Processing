import streamlit as st
from multiapp import MultiApp
from apps import pic, srgan, plant # import your app modules here

app = MultiApp()

st.markdown("""
# Web Application of ImageProcessing
The Code is available at [github/preet-pdf](https://github.com/upraneelnihar/streamlit-multiapps) Blog at [medium.com/preet-parikh](https://medium.com/@u.praneel.nihar).
""")

# Add all your application here
app.add_app("Home", srgan.app)
app.add_app("Image Caption", pic.app)
app.add_app("Plant-Disease-Diagnosis", plant.app)
# The main app
app.run()
