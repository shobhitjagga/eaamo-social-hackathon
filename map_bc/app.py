from flask import Flask, render_template, request, jsonify
import geopandas as gpd
import pandas as pd
import pickle
import shap
import os

app = Flask(__name__)

fcol = [
    "support_count", "social_assistance_count", "outpatient_consultation_count", "hospitalization_count",
    "carcinogenic_water_sources_count","water_sources_carcinogens_2B","water_sources_carcinogens_1",
    "emission_concentration","marginalization", "electricity_fuel", "firewood_fuel", "oil_fuel",
    "brick_factory", "stationary_risk_sources"
]
currentYear=2022
static_folder_path = app.static_folder
file_path = os.path.join(static_folder_path, 'model', 'features.all.csv')
features = pd.read_csv(file_path)
def process_data(municipality, source_type, count):
    print(municipality)
    print(source_type)
    print(count)
    return f"Processed {count} for {municipality} with source type {source_type}"

def compute_contribution(df,model,explainer,stateCode,munCode):
    print(df[fcol])
    print(df['outpatient_consultation'])
    shap_values = explainer.shap_values(df[fcol])
    expected_value = explainer.expected_value
    temp_df = pd.DataFrame({
        'Source': fcol,
        'Contribution': shap_values[0],
    })
    temp_df = temp_df.reindex(temp_df['Contribution'].abs().sort_values(ascending=False).index)
    expected_value = explainer.expected_value
    expected_value = expected_value if isinstance(expected_value, (int, float)) else expected_value[0]
    prediction = model.predict(df[fcol])
    pred_df = pd.DataFrame({'Source': ['Prediction'], 'Contribution': prediction[0]})
    cont_df = pd.concat([pred_df, temp_df], ignore_index=True)
    cont_df['Contribution'] = cont_df['Contribution'].round(1)
    fname = 'riskprofile.'+str(stateCode)+'.'+str(munCode)+'.'+str(currentYear)+'.csv'
    output_file_path = os.path.join(static_folder_path,'risks', fname)
    cont_df.to_csv(output_file_path, index=False)
    print("changed values for ", stateCode, ",", munCode, ",", currentYear);
    with open(output_file_path, 'r') as f:
        lines = [line for line in f if line.strip()]
    with open(output_file_path, 'w') as f:
        f.writelines(lines)
    return prediction

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        municipality = request.form['municipalitySelect']
        source_type = request.form['healthType']
        count = request.form['count']

        # Call your function
        result = process_data(municipality, source_type, count)
        return render_template('index.html', result=result)

    return render_template('index.html', result=None)

@app.route('/add-health-source', methods=['POST'])
def add_health_source():
    # municipality = request.form['municipalitySelect']
    municipalityCode = request.form['mcode']
    stateCode = request.form['scode']
    # municipality = request.form['municipalitySelect']
    health_source = request.form['healthType']
    count = request.form['count']
    model_path = os.path.join(static_folder_path,'model','random_forest_model.pkl')
    with open(model_path, 'rb') as file:
        rf_model = pickle.load(file)
    df = features[ (features['year']==currentYear) & (features['state_code']==int(stateCode)) & (features['municipality_code']==int(municipalityCode)) ]
    if health_source == 'Support':
        df['support_count'] = df['support_count'] + int(count)
    elif health_source == 'Social assistance':
        df['social_assistance_count'] = df['social_assistance_count'] + int(count)
    elif health_source == 'Outpatient Consultation':
        df['outpatient_consultation'] = df['outpatient_consultation_count'] + int(count)
    else:
        df['hospitalization_count'] = df['hospitalization_count'] + int(count)
    explainer = shap.TreeExplainer(rf_model)
    pred = compute_contribution(df, rf_model, explainer,stateCode,municipalityCode);
    risk_file = 'riskprofile.'+str(currentYear)+'.csv'
    risk_file_path = os.path.join(static_folder_path,'risks',risk_file)
    risk_df = pd.read_csv(risk_file_path)
    print(risk_df[(risk_df['state_code']==int(stateCode)) & (risk_df['municipality_code']==int(municipalityCode))]['risk_prediction'])
    risk_df.loc[(risk_df['state_code']==int(stateCode)) & (risk_df['municipality_code']==int(municipalityCode)),'risk_prediction'] = pred
    print(risk_df[(risk_df['state_code']==int(stateCode)) & (risk_df['municipality_code']==int(municipalityCode))]['risk_prediction'])
    risk_df.to_csv(risk_file_path, index=False)
    print(health_source)
    print(municipalityCode)
    print(stateCode)
    print(count)
    return render_template('index.html', result=None)

@app.route('/add-water-source', methods=['POST'])
def add_water_source():
    # municipality = request.form['municipalitySelect']
    municipalityCode = request.form['mcode']
    stateCode = request.form['scode']
    conc_sure = request.form['iarc1']
    conc_possible = request.form['iarc2']
    model_path = os.path.join(static_folder_path,'model','random_forest_model.pkl')
    with open(model_path, 'rb') as file:
        rf_model = pickle.load(file)
    df = features[ (features['year']==currentYear) & (features['state_code']==int(stateCode)) & (features['municipality_code']==int(municipalityCode)) ]
    df['water_sources_carcinogens_1'] = df['water_sources_carcinogens_1'] + conc_sure
    df['water_sources_carcinogens_2B'] = df['water_sources_carcinogens_2B'] + conc_possible
    df['carcinogenic_water_sources_count'] = df['carcinogenic_water_sources_count'] + 1
    explainer = shap.TreeExplainer(rf_model)
    pred = compute_contribution(df, rf_model, explainer,stateCode,municipalityCode);
    risk_file = 'riskprofile.'+str(currentYear)+'.csv'
    risk_file_path = os.path.join(static_folder_path,'risks',risk_file)
    risk_df = pd.read_csv(risk_file_path)
    risk_df.loc[(risk_df['state_code']==int(stateCode)) & (risk_df['municipality_code']==int(municipalityCode)),'risk_prediction'] = pred
    risk_df.to_csv(risk_file_path, index=False)
    return render_template('index.html', result=None)

@app.route('/add-emission-source', methods=['POST'])
def add_emission_source():
    # municipality = request.form['municipalitySelect']
    municipalityCode = request.form['mcode']
    stateCode = request.form['scode']
    conc_sure = request.form['iarc1']
    concentration = request.form['emission']
    model_path = os.path.join(static_folder_path,'model','random_forest_model.pkl')
    with open(model_path, 'rb') as file:
        rf_model = pickle.load(file)
    df = features[ (features['year']==currentYear) & (features['state_code']==int(stateCode)) & (features['municipality_code']==int(municipalityCode)) ]
    df['emission_concentration'] = df['emission_concentration'] + concentration
    explainer = shap.TreeExplainer(rf_model)
    pred = compute_contribution(df, rf_model, explainer,stateCode,municipalityCode);
    risk_file = 'riskprofile.'+str(currentYear)+'.csv'
    risk_file_path = os.path.join(static_folder_path,'risks',risk_file)
    risk_df = pd.read_csv(risk_file_path)
    risk_df.loc[(risk_df['state_code']==int(stateCode)) & (risk_df['municipality_code']==int(municipalityCode)),'risk_prediction'] = pred
    risk_df.to_csv(risk_file_path, index=False)
    return render_template('index.html', result=None)

@app.route('/add-strisk-source', methods=['POST'])
def add_strisk_source():
    # municipality = request.form['municipalitySelect']
    municipalityCode = request.form['mcode']
    stateCode = request.form['scode']
    risk_source = request.form['stRisk']
    concentration = request.form['emission']
    model_path = os.path.join(static_folder_path,'model','random_forest_model.pkl')
    with open(model_path, 'rb') as file:
        rf_model = pickle.load(file)
    df = features[ (features['year']==currentYear) & (features['state_code']==int(stateCode)) & (features['municipality_code']==int(municipalityCode)) ]
    df['stationary_risk_sources'] = df['stationary_risk_sources'] + 1
    explainer = shap.TreeExplainer(rf_model)
    pred = compute_contribution(df, rf_model, explainer,stateCode,municipalityCode);
    risk_file = 'riskprofile.'+str(currentYear)+'.csv'
    risk_file_path = os.path.join(static_folder_path,'risks',risk_file)
    risk_df = pd.read_csv(risk_file_path)
    risk_df.loc[(risk_df['state_code']==int(stateCode)) & (risk_df['municipality_code']==int(municipalityCode)),'risk_prediction'] = pred
    risk_df.to_csv(risk_file_path, index=False)
    return render_template('index.html', result=None)

# @app.route('/')
# def index():
#     return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)