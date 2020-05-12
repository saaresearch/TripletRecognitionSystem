import os


def main():
    os.system('python train.py')
    os.system('rm -r data')
    os.system('python classifier_train.py')
    os.system('python trace.py')
    os.system('find . ! -name "trace_pddmodel.pt" ! -name "classes2.txt" -delete')


if __name__ == "__main__":

    main()
