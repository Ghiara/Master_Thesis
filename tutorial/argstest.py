import argparse

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--A', type=int,help='input int num A', default=15)
    parser.add_argument('--B', type=str,help='input str B by default hello world', default='hello world.')

    args = parser.parse_args()

    return print(args.A, args.B)


if __name__=='__main__':
    main()

