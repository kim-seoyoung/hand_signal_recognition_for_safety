import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="need to fiil in")

    parser.add_argument('--input', type=str, metavar='FILENAME', help='input file name')
    parser.add_argument('--output-dir', type=str, metavar='PATH', help='output dirctory name')
    parser.add_argument('--savegif', default=False,type=bool, metavar='save', help='save 3d plot or not')



    args = parser.parse_args()

    return args
