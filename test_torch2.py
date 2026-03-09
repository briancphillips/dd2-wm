import torch

try:
    t1 = torch.tensor(1.0)
    t2 = torch.tensor(0.0)
    res = (t1 == 1.0) and (t2 == 0.0)
    print("AND test:", res, type(res))
except Exception as e:
    print("AND Error:", e)

try:
    a = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    res = torch.tensor(a)
    print("torch.tensor(a) test:", res)
except Exception as e:
    print("torch.tensor(a) Error:", e)
