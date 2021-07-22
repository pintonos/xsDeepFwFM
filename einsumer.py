import numpy as np
from opt_einsum import contract
import torch

from timeit import default_timer as timer


x = np.random.rand(39, 2048, 10)
x_t = torch.rand(39, 2048, 10)

start = timer()
np.einsum('kij,lij->klij', x, x)
end = timer()
print(end - start)

start = timer()
torch.einsum('kij,lij->klij', x_t, x_t)
end = timer()
print(end - start)

start = timer()
contract('kij,lij->klij', x, x)
end = timer()
print(end - start)


