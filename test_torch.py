import torch
a = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
try:
    print("Trying torch.tensor(a)")
    res = torch.tensor(a)
    print("Success:", res)
except Exception as e:
    print("Error:", e)

try:
    print("Trying torch.stack(a)")
    res = torch.stack(a)
    print("Success:", res)
except Exception as e:
    print("Error:", e)
