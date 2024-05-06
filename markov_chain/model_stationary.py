import torch

# vector de probabilidades iniciales
v = torch.tensor([[0.5, 0.5]])

# matriz de transicion
P = torch.tensor([
    [0.8, 0.2],
    [0.4, 0.6]
])

# Probabilidades en el segundo dia de la serie
w = torch.matmul(v, P)
print(f'w = {w}')

# Probabilidades en el tercer dia de la serie
u = torch.matmul(w, P)
print(f'u = {u}')

# Probabilidades en el cuarto dia de la serie
t = torch.matmul(u, P)
print(f't = {t}')
