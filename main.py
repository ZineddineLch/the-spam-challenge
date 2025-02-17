import os
import sys

# Make sure the current directory (ml_project) is added to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import train
import predict
import preprocess
def main():
    print(" preprocessing....")
    preprocess.run()
    print("train.....")
    train.run()
    print("the predection....")
    predict.run()
if __name__ == "__main__":
   main()