import torch

datas = ["audiocaps", "fsd50k", "clotho", "macs"]

for data in datas:
    print(data)
    fea_path = str(f"./data/{data}/clap_fea.pt")
    file_content = torch.load(fea_path, map_location="cpu")
    
    for split in file_content.keys():
        print(f"  {split}: {len(file_content[split])}")
    
    print("-----------------------------------")