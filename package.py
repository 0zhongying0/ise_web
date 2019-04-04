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
	while True:
		# create pcap
		filename = str(time.time()) + '.pcap'
		cmd = "sudo tcpdump -c 5 -w " + path +"/package_file/" + filename
		os.system(cmd)

		#delete pcap for 10 min past file
		filenames = os.listdir(path + "/package_file/")
		for x in filenames:
			temp = x[:-5]
			if float(temp) + 600 < time.time():
				os.remove(path + "/package_file/" + x)


if __name__ == '__main__':
	main()