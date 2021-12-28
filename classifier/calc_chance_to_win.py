
for j in range(1, 50):
  for i in range(100, 10000):
    res = (i - j) / i
    if res > 0.9908:
      break
    if res > 0.99069:
      print(i, j, res)
