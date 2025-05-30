import fileinput
import sys
import os

def change_batch_size(batch_size):
    with fileinput.FileInput("alias.py", inplace=True) as f:
        for line in f:
            if line.strip().startswith("BATCH_SIZE ="):
                print("BATCH_SIZE = %d" % batch_size)
            else:
                print(line, end='')

if __name__ == "__main__":
    change_batch_size(512)
    print("start batch size 512")
    os.system("python run_mesh.py > log/512.txt")
    os.system("python run_mesh.py 0 > log/512_lazy.txt")

    change_batch_size(256)
    print("start batch size 256")
    os.system("python run_mesh.py > log/256.txt")
    os.system("python run_mesh.py 0 > log/256_lazy.txt")

    change_batch_size(128)
    print("start batch size 128")
    os.system("python run_mesh.py > log/128.txt")
    os.system("python run_mesh.py 0 > log/128_lazy.txt")

    change_batch_size(64)
    print("start batch size 64")
    os.system("python run_mesh.py > log/64.txt")
    os.system("python run_mesh.py 0 > log/64_lazy.txt")

    change_batch_size(32)
    print("start batch size 32")
    os.system("python run_mesh.py > log/32.txt")
    os.system("python run_mesh.py 0 > log/32_lazy.txt")

    change_batch_size(16)
    print("start batch size 16")
    os.system("python run_mesh.py > log/16.txt")
    os.system("python run_mesh.py 0 > log/16_lazy.txt")

