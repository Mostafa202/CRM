import joblib
import pandas as pd
from preprocess import preprocess_test
from dowhy import gcm

def predict(processed_data):
    process_model = joblib.load('process_duration_model.pkl')
    deal_model = joblib.load('deal_stage_model.pkl')
    scm = joblib.load('causal_graph.pkl')


    process_prediction = process_model.predict(processed_data).astype(int)
    # processed_data['process_duration']= [6,7,96,92,66,11,89,134,95]
    processed_data['process_duration']= process_prediction

    processed_data = processed_data.reindex(columns=['sector','revenue','employees','office_location','product','sales_price','agent_win_rate','process_duration','engage_month'])
    deal_predictions = deal_model.predict(processed_data)

    if deal_predictions==1:
        return ['Won'],processed_data
    else:
        # processed_data['deal_stage'] = [1,1,1,1,0,1,1,0,0]
        processed_data['deal_stage'] = deal_predictions

        attrs = gcm.attribute_anomalies(   # Function to attribute anomalies
        scm,       # Structural causal model
        target_node='deal_stage',    # Target node for anomaly attribution
        anomaly_samples=processed_data # Anomaly samples to analyze
        )

        attrs_dict = {x:y for x,y in zip(attrs.keys(),[x[0] for x in attrs.values()])}
        sorted_attrs = dict(sorted(attrs_dict.items(), key=lambda item: abs(item[1]), reverse=True))

        return list(sorted_attrs.keys())[1:3],processed_data















