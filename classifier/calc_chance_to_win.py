
def calc(k):
  print(k)

  out = []
  for d_size in range(10, max_dataset_size):
    for mistake_1 in range(int(d_size * (1.0 - k - 0.0001)) - 2, int(d_size * (1.0 - k + 0.0001)) + 2):
      res_1 = (d_size - mistake_1) / d_size
      if abs(res_1 - k) < d:
        print(d_size, mistake_1, res_1, abs(res_1 - k), '<--------------------')
        out.append(d_size)
  return out

k1 = 0.990655825205
k2 = 0.915699669349
k3 = 0.453512612049
d =  0.00000000001
d =  0.000000000001


max_dataset_size = 1000000

data = []

data.append(calc(k1))
data.append(calc(k2))
data.append(calc(k3))

print('__________')
res = set(data[0])

for i in range(1, len(data)):
  res = res.intersection(data[i])
  print(res)

print(res)
