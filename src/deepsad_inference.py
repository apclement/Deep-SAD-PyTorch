
import os
import torch
import io
import pandas as pd
from networks.mlp import MLP
from params import *

def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MLP(x_dim=N_features, h_dims=[64, 32, 16, 8, 4], rep_dim=2, bias=False).to(device)
    net = torch.nn.DataParallel(net)
    net_dict = torch.load(model_dir + '/model.pth')
    net.load_state_dict(net_dict['state'])
    c = torch.tensor(net_dict['c'], device=device)
    return c, net

def input_fn(request_body, request_content_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        df = pd.read_csv(io.StringIO(request_body))
    except:
        return None    
  
    #one line case
    df = pd.read_csv(io.StringIO(request_body), header=None) if df.shape[0] == 0 else df
    
    # start at 4 index to skip project_hash, ecriture_id, label and target columns and get the feature columns     
    try:
        samples = torch.tensor(df.iloc[:, 4:].to_numpy(), dtype=torch.float32, device=device)   
    except Exception as e:
        print(">>>>>>>>>>>>>>>")
        print(request_body)
        print("############")
        df.info()
        #pd.set_option('display.max_rows', 300)
        pd.set_option('display.max_columns', 300)
        df_slice = df.iloc[:, 4:]
        print(df_slice.shape)
        print(df_slice)
        print("<<<<<<<<<<<<<<<<<")
        raise e
        
    # extract ids
    ids = df.iloc[:, :2]
    return ids, samples    

# Perform prediction on the deserialized object, with the loaded model
def predict_fn(inputs, c_model):
    if inputs is None:
        return None
    ids, samples = inputs
    c, model = c_model
    model.eval()
    with torch.no_grad():
        outputs = model(samples)
        scores = torch.sum((outputs - c) ** 2, dim=1)
        
    ids['score'] = scores.cpu().numpy()
    
    return ids

# Serialize the prediction result into the desired response content type
def output_fn(predictions, response_content_type):
    if predictions is None:
        return ""
    return predictions.to_csv(header=False, index=False)
