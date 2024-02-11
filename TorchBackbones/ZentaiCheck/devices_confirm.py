import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("====================================")
print(f"Torch version is: <{torch.__version__}>")
print(f"CUDA device available is {torch.cuda.is_available()}, now could be running on {device}")

print(f"Total GPUs num: {torch.cuda.device_count()}")
print("")

print("Devices info is:")
for i in range(torch.cuda.device_count()):
    print(f"\n{i} : \t\t {torch.cuda.get_device_name(0)}\n "
          f"\t\t\t {torch.cuda.device(0)}")

print("====================================")

__all__ = ["device"]
