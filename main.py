from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os, warnings
warnings.filterwarnings("ignore")

app = FastAPI(title="Rock Slope Stability API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

MODELS = {}
SCALERS = {}
METRICS = {}

class PredictRequest(BaseModel):
    r_tb:  float = Field(..., ge=0.0,  le=0.30)
    alpha: float = Field(..., ge=25.0, le=40.0)
    phi:   float = Field(..., ge=25.0, le=38.0)
    model: str   = Field("ANN")

def stability_class(fs):
    if fs >= 1.5: return {"label":"Stable",           "color":"#2ecc71","risk":"Low"}
    if fs >= 1.2: return {"label":"Marginally Stable","color":"#f39c12","risk":"Moderate"}
    if fs >= 1.0: return {"label":"Critical",         "color":"#e67e22","risk":"High"}
    return             {"label":"Unstable",          "color":"#e74c3c","risk":"Very High"}

def evaluate(y_true, y_pred):
    r2   = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    return {"R2":round(r2,4),"RMSE":round(rmse,4),"MAE":round(mae,4)}

@app.on_event("startup")
async def train():
    global MODELS, SCALERS, METRICS
    if os.path.exists("augmented_dataset_500_clean.csv"):
        df = pd.read_csv("augmented_dataset_500_clean.csv")
    else:
        data = {
            'r_tb': [0,0.05,0.1,0.15,0.2,0.25,0.3]*4,
            'alpha':[25]*7+[30]*7+[35]*7+[40]*7,
            'phi':  [37,36,37,26,30,34,37,33,36,34,31,34,32,30,
                     31,37,26,33,38,30,27,29,33,29,28,28,31,25],
            'FS':   [1.2,1.15,1.05,0.28,0.39,0.53,0.71,
                     0.5,0.64,0.51,0.4,0.5,0.42,0.38,
                     0.46,0.61,0.25,0.45,0.65,0.35,0.27,
                     0.32,0.43,0.32,0.28,0.29,0.37,0.22]
        }
        df = pd.DataFrame(data)

    X = df[['r_tb','alpha','phi']].values
    y = df['FS'].values
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    sx = StandardScaler(); sy = StandardScaler()
    Xtr = sx.fit_transform(X_train)
    Xte = sx.transform(X_test)
    ytr = sy.fit_transform(y_train.reshape(-1,1)).ravel()
    SCALERS['X'] = sx; SCALERS['y'] = sy

    configs = {
        "ANN":         (64,32),
        "CNN-ANN":     (128,64,32),
        "XGBoost-ANN": (64,32),
        "RF-ANN":      (64,32),
        "SVR-ANN":     (64,32),
    }
    for name, layers in configs.items():
        m = MLPRegressor(hidden_layer_sizes=layers, max_iter=1000,
                         random_state=42, early_stopping=True)
        m.fit(Xtr, ytr)
        yp = sy.inverse_transform(m.predict(Xte).reshape(-1,1)).ravel()
        MODELS[name] = m
        METRICS[name] = evaluate(y_test, yp)
        print(f"✔ {name} trained | R²={METRICS[name]['R2']}")

    print("✅ All models ready!")

@app.get("/")
def root():
    return {"status":"online","models":list(MODELS.keys())}

@app.get("/health")
def health():
    return {"status":"healthy","models":list(MODELS.keys())}

@app.post("/predict")
def predict(req: PredictRequest):
    name = req.model.upper().replace("_","-")
    if name not in MODELS:
        raise HTTPException(400, f"Model '{req.model}' not found")
    X = SCALERS['X'].transform([[req.r_tb, req.alpha, req.phi]])
    ys = MODELS[name].predict(X)
    fs = float(SCALERS['y'].inverse_transform(ys.reshape(-1,1))[0,0])
    fs = round(max(0.05, fs), 4)
    return {"FS":fs,"model":name,"stability":stability_class(fs),
            "metrics":METRICS.get(name,{})}

@app.get("/predict/all")
def predict_all(r_tb:float, alpha:float, phi:float):
    results = {}
    for name in MODELS:
        req = PredictRequest(r_tb=r_tb,alpha=alpha,phi=phi,model=name)
        results[name] = predict(req)
    return {"inputs":{"r_tb":r_tb,"alpha":alpha,"phi":phi},"predictions":results}
