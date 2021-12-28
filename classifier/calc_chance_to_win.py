
for j in range(1, 50):
  for k in range(1, 50):
    for i in range(100, 10000):
      res1 = (i - j) / i
      res2 = (i - j - k) / i
      if res1 > 0.9908 or res2 > 0.9812:
        break

      if res1 > 0.99069 and res2 > 0.9810:
        print(i, j, j+k, res1, res2)
