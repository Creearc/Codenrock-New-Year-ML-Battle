from ftplib import FTP
import sys

PATH = '\\'.join(sys.argv[0].split('\\')[:-1])

ftp = FTP()
HOST = '192.168.68.201'
PORT = 21

ftp.connect(HOST, PORT)

print(ftp.login(user='alexandr', passwd='9'))

ftp.cwd('Codenrock-New-Year-ML-Battle/classifier2/files_for_nikita')

for i in ['2_net.csv']:
  with open(i, 'wb') as f:
      ftp.retrbinary('RETR ' + i, f.write)
