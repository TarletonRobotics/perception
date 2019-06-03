from datetime import datetime
import tarfile
import numpy as np
from os import listdir
from io import StringIO
import time
import cv2
from datetime import datetime


def extract_files(work_dir):
	tar = tarfile.open(work_dir + '/trail_data.tar.gz', 'r:gz')
	tar.extractall(work_dir)
	tar.close()
	

def process():
	# Extract the files
	#extract_files('./data')

	f = open('./data/inputs.csv')

	steering = np.genfromtxt(StringIO(f.read()),delimiter=',', dtype='|U8')
	file_names = [f for f in listdir('./data/trail_data')]
	images = []

	for s in steering:
		steering_time = s[0]
		for img_file_name in file_names:
			img_time = img_file_name.split(' ')[3]
			
			# sh_time = int(steering_time.split(':')[0])
			# ih_time = int(img_time.split(':')[0])
			# sm_time = int(steering_time.split(':')[1])
			# im_time = int(img_time.split(':')[1])
			# ss_time = int(steering_time.split(':')[2])
			# is_time = int(img_time.split(':')[2])
			
			if img_time == steering_time: 
				#print('img time', img_time, 'steering_time', steering_time)
		
				img = cv2.imread('./data/trail_data/' + img_file_name)
				images.append(img)

	return images, s[2]


