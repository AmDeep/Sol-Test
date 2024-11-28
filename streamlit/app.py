# -*- coding: utf-8 -*-



######################
# Import libraries
######################
import streamlit as st
import pickle
from PIL import Image
import pandas as pd
from rdkit import Chem
import xgboost
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import predefined_models
import base64
from sklearn.impute import SimpleImputer


######################
# Custom function
######################
## Calculate molecular descriptors
def generate(smiles_list, verbose=False):

    selected_columns=['nHBAcc', 'nHBDon', 'nRot', 'nBonds', 'nAromBond', 'nBondsO', 'nBondsS',
           'TopoPSA(NO)', 'TopoPSA', 'LabuteASA', 'bpol', 'nAcid', 'nBase',
           'ECIndex', 'GGI1', 'SLogP', 'SMR', 'BertzCT', 'BalabanJ', 'Zagreb1',
           'ABCGG', 'nHRing', 'naHRing', 'NsCH3', 'NaaCH', 'NaaaC', 'NssssC',
           'SsCH3', 'SdCH2', 'SssCH2', 'StCH', 'SdsCH', 'SaaCH', 'SsssCH', 'SdssC',
           'SaasC', 'SaaaC', 'SsNH2', 'SssNH', 'StN', 'SdsN', 'SaaN', 'SsssN',
           'SaasN', 'SsOH', 'SdO', 'SssO', 'SaaO', 'SsF', 'SdsssP', 'SsSH', 'SdS',
           'SddssS', 'SsCl', 'SsI']
      
    
    # Test Data filter
    test_smiles_list=[]
    test_formula_list=[]
    test_mordred_descriptors=[]
    
    for smiles in smiles_list:
        mol=Chem.MolFromSmiles(smiles)
        mol=Chem.AddHs(mol)
        formula=Chem.rdMolDescriptors.CalcMolFormula(mol)
        formula=formula.replace("+","")
        formula=formula.replace("-","")
    
        test_smiles_list.append(smiles)
        test_formula_list.append(formula)
        test_mordred_descriptors.append(predefined_models.predefined_mordred(mol,"all"))
        
    #get all column names
    column_names=predefined_models.predefined_mordred(Chem.MolFromSmiles("CC"),"all",True)
    
    #create Mordred desc dataframe
    test_df=pd.DataFrame(index=test_formula_list, data=test_mordred_descriptors,columns=column_names)
    
    #Select predefined columns by the model
    selected_data_test = test_df[selected_columns]
    selected_data_test = selected_data_test.apply(pd.to_numeric)

    return selected_data_test


######################
# Page Title
######################


#st.set_page_config(page_title="AqSolPred: Online Solubility Prediction Tool")


st.write("""# AqSolPred: Aqueous Solubility Prediction Tool""")

image = Image.open('solubility-factors.png')
st.image(image, use_column_width=False)


######################
# Input molecules (Side Panel)
######################

st.sidebar.write('**Type SMILES below**')

## Read SMILES input
SMILES_input = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C\nCC(=O)OC1=CC=CC=C1C(=O)O"

SMILES = st.sidebar.text_area('then press ctrl+enter', SMILES_input)
SMILES = SMILES.split('\n')
SMILES = list(filter(None, SMILES))


st.sidebar.write("""---------**OR**---------""")
st.sidebar.write("""**Upload a file with a column named 'SMILES'** (Max:2000)""")

   
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    # data
    SMILES=data["SMILES"]  


# st.header('Input SMILES')
# SMILES[1:] # Skips the dummy first item

# Use only top 300
if len(SMILES)>2000:
    SMILES=SMILES[0:2000]
	
## Calculate molecular descriptors
generated_descriptors = generate(SMILES)

#Import pretrained models
mlp_model_import = pickle.load(open('aqsolpred_mlp_model.pkl', 'rb'))
xgboost_model_import = pickle.load(open('aqsolpred_xgb_model.pkl', 'rb'))
# rf_model_import = pickle.load(open('aqsolpred_rf_model_lite.pkl', 'rb'))




#predict test data (MLP,XGB,RF)
pred_mlp = mlp_model_import.predict(generated_descriptors)   
pred_xgb = xgboost_model_import.predict(generated_descriptors)
# pred_rf = rf_model_import.predict(generated_descriptors)   
#calculate consensus
# pred_consensus=(pred_mlp+pred_xgb+pred_rf)/3
pred_consensus=(pred_mlp+pred_xgb)/2
# predefined_models.get_errors(test_logS_list,pred_enseble)

# results=np.column_stack([pred_mlp,pred_xgb,pred_rf,pred_consensus])
df_results = pd.DataFrame(SMILES, columns=['SMILES'])
df_results["LogS (AqSolPred v1.1s)"]=pred_consensus
df_results=df_results.round(3)

# df_results.to_csv("results/predicted-"+test_data_name+".csv",index=False)

st.header('Predicted LogS values')
df_results # Skips the dummy first item

# download=st.button('Download Results File')
# if download:
csv = df_results.to_csv(index=False)
b64 = base64.b64encode(csv.encode()).decode()  # some strings
linko= f'<a href="data:file/csv;base64,{b64}" download="aqsolpred_predictions.csv">Download csv file</a>'
st.markdown(linko, unsafe_allow_html=True)
 
st.header('Computed molecular descriptors')
generated_descriptors # Skips the dummy first item


st.write("""
# Solubility Predictor

""")



