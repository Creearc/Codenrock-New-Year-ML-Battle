##
##for j in range(1, 50):
##  for k in range(1, 50):
##    for i in range(100, 10000):
##      res1 = (i - j) / i
##      res2 = (i - j - k) / i
##      if res1 > 0.9908 or res2 > 0.9812:
##        break
##
##      if res1 > 0.99069 and res2 > 0.9810:
##        print(i, j, j+k, res1, res2)

k1 = 0.990655825205
k2 = 0.915699669349
k3 = 0.453512612049
d =  0.00000000001

k = k1

print(k)
max_dataset_size = 100000
for d_size in range(10, max_dataset_size):
  for mistake_1 in range(int(d_size * (1 - k)), 1, -1):
    res_1 = (d_size - mistake_1) / d_size
    if abs(res_1 - k) < d:
      print(d_size, mistake_1, res_1, abs(res_1 - k), '<--------------------')
