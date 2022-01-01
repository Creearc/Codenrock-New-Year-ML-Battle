
def compare(files):
  res = []

  for file in files:
    res.append(set())
    f = open(file, 'r')
    for s in f:
      res[-1].add(s)
    f.close()

  for i in range(len(res)):
    for j in range(i+1, len(res)):
      if i == j:
        continue

      out = list(res[i].difference(res[j]))
      for s in out:
        print(s)

      print('_____________________________')
  
  


compare(['1.txt',
         '2.txt'])

compare(['1.txt',
         '3.txt'])

compare(['1.txt',
         '4.txt'])

compare(['2.txt',
         '4.txt'])
