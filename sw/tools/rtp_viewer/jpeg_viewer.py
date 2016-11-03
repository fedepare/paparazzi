#! /usr/bin/python

import cv2
from socket import *
import sys
import select

# Create a named window and add a mouse callback
cv2.namedWindow('image stream')

host="192.168.4.255"
port = 4242
s = socket(AF_INET,SOCK_DGRAM)
s.bind((host,port))

addr = (host,port)
buf=65536
data = []

while(1):
  try:
    s.settimeout(2)
    datain,addr = s.recvfrom(buf)
    data = [data, datain]
    
    print datain

    # Show the image in a window
#    cv2.imshow('image stream', data)
  except timeout:
    s.close()
