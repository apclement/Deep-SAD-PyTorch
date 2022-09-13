
import os
import torch
import io
import pandas as pd
from networks.mlp import MLP
from params import *
import json

def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MLP(x_dim=N_features, h_dims=[64, 32, 16, 8, 4], rep_dim=2, bias=False).to(device)
    net = torch.nn.DataParallel(net)
    net_dict = torch.load(model_dir + '/model.pth')
    net.load_state_dict(net_dict['state'])
    c = torch.tensor(net_dict['c'], device=device)
    return c, net

def input_fn(request_body, content_type):
    print(f"Content-type : {content_type}")
    
    body_json = request_body.decode()
    #print(f">>>>>>>>>{body_json}<<<<<<")
    payload = json.loads(body_json)
    print(f">>>>!!!>>>>{len(payload)}<<<")
    values = [line['features'] for elem in payload for line in elem]  
    #df = pd.DataFrame(values)
    #print(df.info())
    #print(df)
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      
    samples = torch.tensor(values, dtype=torch.float32, device=device)   
           
    return samples      

# Perform prediction on the deserialized object, with the loaded model
def predict_fn(inputs, c_model):
    if inputs is None:
        return None
    samples = inputs
    c, model = c_model
    model.eval()
    with torch.no_grad():
        outputs = model(samples)
        scores = torch.sum((outputs - c) ** 2, dim=1)
        
    scores = pd.DataFrame({'score': scores.cpu().numpy()})
    
    return scores

# Serialize the prediction result into the desired response content type
def output_fn(predictions, response_content_type):
    if predictions is None:
        return ""
    return predictions.to_csv(header=False, index=False)
