#========================== IMPORT PACKAGES ==========================

import numpy as np 
import pandas as pd
from tkinter.filedialog import askopenfilename
import warnings
warnings.filterwarnings('ignore')
import streamlit as st

#============================ BACKGROUND IMAGE  ==========================

import base64



st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:36px;">{"An efficient spam detection technique for iot devices"}</h1>', unsafe_allow_html=True)


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




st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:20px;">{"Register Here!!!"}</h1>', unsafe_allow_html=True)

import pandas as pd


if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False




UR = st.text_input("Register User Name",key="username1")
FN = st.text_input("Enter First Name",key="first")
LN = st.text_input("Enter Last Name",key="last")


pss1 = st.text_input("First Password",key="password1",type="password")
pss2 = st.text_input("Confirm Password",key="password2",type="password")



if pss1 == pss2 and len(str(pss1)) > 2:
    import pandas as pd
    
  
    import csv 
    
    # field names 
    fields = ['User', 'Password','First Name','Last Name'] 
    

    
    # st.text(temp_user)
    old_row = [[UR,pss1,FN,LN]]
    
    # writing to csv file 
    with open(UR+'.csv', 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
            
        # writing the fields 
        csvwriter.writerow(fields) 
            
        # writing the data rows 
        csvwriter.writerows(old_row)
    

col1, col2 = st.columns(2)

with col1:
        
    aa = st.button("REGISTER")
    
    if aa:
        st.success('Successfully Registered !!!')
    # else:
        
        # st.write('Registeration Failed !!!')     

with col2:
        
    aa = st.button("LOGIN")
    
    if aa:
        import subprocess
        subprocess.run(['python','-m','streamlit','run','login.py'])
        # st.success('Successfully Registered !!!')








