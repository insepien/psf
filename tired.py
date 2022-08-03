import pickle
import argparse
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dictfile", type=str)
	args = parser.parse_args()
	data = pickle.load(open(args.dictfile, "rb"))
	print(data)
