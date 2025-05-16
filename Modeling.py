
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LogisticRegression
from dowhy import gcm 
import networkx

def training_models(df):
    x = df.drop(['deal_stage','process_duration'],axis=1)
    y = df['process_duration']
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,shuffle=True)
    model = AdaBoostRegressor()
    model.fit(x_train,y_train)
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    y_pred_train = y_pred_train.astype(int)
    y_pred_test = y_pred_test.astype(int)

    print('pickle process_duraion Model....')
    joblib.dump(model, 'process_duration_model.pkl')


    print('Training Error: ',mean_absolute_error(y_train,y_pred_train))
    print('Testing Error : ',mean_absolute_error(y_test,y_pred_test))

    ## Won/Loss ML Model
    x = df.drop(['deal_stage'],axis=1)
    y = df[['deal_stage']]
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,shuffle=True,stratify=y)
    model = LogisticRegression(class_weight={0:0.62,1:0.38})

    model.fit(x_train,y_train)
    y_pred_train = model.predict(x_train)
    y_pred_test  = model.predict(x_test)

    print('Training Accuracy: ',f1_score(y_train,y_pred_train))
    print('Testing Accuracy : ',f1_score(y_test,y_pred_test))

    print(classification_report(y_test,y_pred_test))

    print('pickle deal_stage model....')
    joblib.dump(model, 'deal_stage_model.pkl')


    ## Causal ML Model

    causal_graph = networkx.DiGraph()
    causal_graph.add_nodes_from = df.columns

    dependencies = {
    "revenue": ["sector", "office_location"],
    "employees":["revenue"],
    "sales_price": ["product"],
    "process_duration": ["product","sector","agent_win_rate","engage_month"],
    "deal_stage": ["sales_price", "agent_win_rate", "process_duration","product","sector","revenue"],
    }
    for child in dependencies:
        for parent in dependencies[child]:
            causal_graph.add_edge(parent, child)

    scm = gcm.StructuralCausalModel(causal_graph)


    # Automatically assign generative models to each node based on the given data
    auto_assignment_summary = gcm.auto.assign_causal_mechanisms(
        scm,
        df, 
        override_models=True,
        quality=gcm.auto.AssignmentQuality.GOOD
    )
    gcm.fit(scm,df)
    print('pickle Causal graph...')
    joblib.dump(scm, 'causal_graph.pkl')









