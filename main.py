import util
from experiment import train, test
from analysis import analyze


if __name__=='__main__':
    # parse options and make directories
    argv = util.option.parse()

    # run and analyze experiment
    if not argv.no_train: train(argv)
    if not argv.no_test: test(argv)
    if not argv.no_analysis and argv.roi=='schaefer': analyze(argv)
    exit(0)
