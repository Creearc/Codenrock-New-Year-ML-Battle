from ftplib import FTP
import sys

PATH = '\\'.join(sys.argv[0].split('\\')[:-1])

ftp = FTP()
HOST = '192.168.68.206'
PORT = 21

ftp.connect(HOST, PORT)

print(ftp.login(user='alexandr', passwd='9'))

ftp.cwd('Codenrock-New-Year-ML-Battle/classifier/results')

for i in ['m_7_q.tflite']:
  with open(i, 'wb') as f:
      ftp.retrbinary('RETR ' + i, f.write)
