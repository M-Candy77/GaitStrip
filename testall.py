from datetime import datetime
import numpy as np
import argparse

from model.initialization import initialization
from model.utils import evaluation
from config import conf


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'
def de_diag(acc, each_angle=False):
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / 10.0
    if not each_angle:
        result = np.mean(result)
    return result

if __name__ == '__main__':

    iterall=[50000,60000,70000,80000]
    # iter = 10000
    # while iter<=80000:
    for iter in iterall:
        parser = argparse.ArgumentParser(description='Test')

        parser.add_argument('--batch_size', default='1', type=int,
                            help='batch_size: batch size for parallel test. Default: 1')
        parser.add_argument('--cache', default=False, type=boolean_string,
                            help='cache: if set as TRUE all the test data will be loaded at once'
                                 ' before the transforming start. Default: FALSE')
        opt = parser.parse_args()

        # Exclude identical-view cases

        m = initialization(conf, test=opt.cache)[0]

        # load model checkpoint of iteration opt.iter
        print('Loading the model of iteration %d...' % iter)
        m.load(iter)
        print('Transforming...')
        time = datetime.now()
        test = m.transform('test', opt.batch_size)
        print('Evaluating...')
        acc = evaluation(test, conf['data'])
        print('Evaluation complete. Cost:', datetime.now() - time)

        # Print rank-1 accuracy of the best model
        # e.g.
        # ===Rank-1 (Include identical-view cases)===
        # NM: 95.405,     BG: 88.284,     CL: 72.041
        # for i in range(1):
        #     print('===Rank-%d (Include identical-view cases)===' % (i + 1))
        #     print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
        #         np.mean(acc[0, :, :, i]),
        #         np.mean(acc[1, :, :, i]),
        #         np.mean(acc[2, :, :, i])))

        # Print rank-1 accuracy of the best model excluding identical-view cases
        # e.g.
        # ===Rank-1 (Exclude identical-view cases)===
        # NM: 94.964,     BG: 87.239,     CL: 70.355
        for i in range(1):
            print('===Rank-%d (Exclude identical-view cases)===' % (i + 1))
            print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
                de_diag(acc[0, :, :, i]),
                de_diag(acc[1, :, :, i]),
                de_diag(acc[2, :, :, i])))


        np.set_printoptions(precision=2, floatmode='fixed')
        for i in range(1):
            print('===Rank-%d of each angle (Exclude identical-view cases)===' % (i + 1))
            print('NM:', de_diag(acc[0, :, :, i], True))
            print('BG:', de_diag(acc[1, :, :, i], True))
            print('CL:', de_diag(acc[2, :, :, i], True))
        # iter +=10000