import os
import time

def main():
	path = os.getcwd()
	directory = './package_file'
	try:
		if not os.path.exists(directory):
			os.makedirs(directory)
	except OSError:
		print ('Error: Creating directory. ' + directory)

	directory = './package_file.bak'
	try:
		if not os.path.exists(directory):
			os.makedirs(directory)
	except OSError:
		print ('Error: Creating directory. ' + directory)

	while True:
		# create pcap
		filename = str(time.time()) + '.pcap'
		cmd = "sudo tcpdump -c 10 -w " + path +"/package_file/" + filename
		os.system(cmd)

		#delete pcap for 1 hr past file
		filenames = os.listdir(path + "/package_file.bak/")
		for x in filenames:
			temp = x[:-5]
			if float(temp) + 3600 < time.time():
				os.remove(path + "/package_file/" + x)


if __name__ == '__main__':
	main()