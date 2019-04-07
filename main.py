import os


def main():
	directory = './package_file_temp'
	try:
		if not os.path.exists(directory):
			os.makedirs(directory)
	except OSError:
		print ('Error: Creating directory. ' + directory)

	directory = './package_file_temp_out'
	try:
		if not os.path.exists(directory):
			os.makedirs(directory)
	except OSError:
		print ('Error: Creating directory. ' + directory)

	cmd = "sudo cp -a ./package_file/. ./package_file_temp"
	os.system(cmd)

	os.chdir('CICFlowMeter-4.0/bin')
	cmd = "sudo ./cfm ../../package_file ../../package_file_temp_out"
	os.system(cmd)

	os.chdir('../../ALAD')
	os.system('python3 main.py alad cicids2017 testing')


if __name__ == '__main__':
	main()