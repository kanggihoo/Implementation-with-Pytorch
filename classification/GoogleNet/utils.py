import torch
from pathlib import Path
from tqdm import tqdm
def save_checkpoint(model , optimizer, name , path_dir , **results):
    check_point = {}
    check_point.update({"model_state_dict" : model.state_dict()})
    check_point.update({"optimizer_state_dict" : optimizer.state_dict()})
    check_point.update(results)
    model_dir = Path(path_dir)
    if not model_dir.exists():
        model_dir.mkdir(parents=True , exist_ok=True)
        
    model_path = model_dir / str(name+".pth")
    torch.save(check_point, model_path)

def load_checkpoint(path , model , optimizer):
    check_point = torch.load(path)
    model.load_state_dict(check_point['model_state_dict'])
    optimizer.load_state_dict(check_point['optimizer_state_dict'])
    print("Load checkpoint!")

def predict(model , test_imgs , transform , DEVICE):
    results = []
    model.eval()
    with torch.inference_mode():
        for img in tqdm(test_imgs , desc = "Predict"):
            agment= transform(image=img)
            test_img = agment["image"]
            test_img = test_img.unsqueeze(0).to(DEVICE)
            pred = model(test_img)
            pred = torch.argmax(pred,dim=1).item()
            results.append(pred)
    return results