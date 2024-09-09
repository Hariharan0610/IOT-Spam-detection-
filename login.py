import streamlit as st
import pandas as pd


import streamlit as st
import base64

# ================ Background image ===





def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('IOT-Banner-1024x614.jpg')


st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:36px;">{"An efficient spam detection technique for iot devices"}</h1>', unsafe_allow_html=True)



UR1 = st.text_input("Login User Name",key="username")
psslog = st.text_input("Password",key="password",type="password")


col1, col2 = st.columns(2)

with col1:
        
    aa = st.button("LOGIN")
    
    if aa:
        df = pd.read_csv(UR1+'.csv')
        U_P1 = df['User'][0]
        U_P2 = df['Password'][0]
        if str(UR1) == str(U_P1) and str(psslog) == str(U_P2):
            st.success('Successfully Login !!!')  
            import subprocess
            subprocess.run(['python','-m','streamlit','run','MainF.py'])
        # st.success('Successfully Registered !!!')
    # else:
        
        # st.write('Registeration Failed !!!')     

with col2:
        
    aa = st.button("BACK")
    
    if aa:
        import subprocess
        subprocess.run(['streamlit','run','IoTSpam.py'])

