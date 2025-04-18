import numpy as np
def print_summary(acc_taw, acc_tag, forg_taw, forg_tag):
    """Print summary of results"""
    for name, metric in zip(['TAw Acc', 'TAg Acc', 'TAw Forg', 'TAg Forg'], [acc_taw, acc_tag, forg_taw, forg_tag]):
        print('*' * 108)
        print(name)
        for i in range(metric.shape[0]):
            print('\t', end='')
            for j in range(metric.shape[1]):
                print('{:5.2f}% '.format(100 * metric[i, j]), end='')
            if np.trace(metric) == 0.0:
                if i > 0:
                    print('\tAvg.:{:5.2f}% '.format(100 * metric[i, :i].mean()), end='')
            else:
                print('\tAvg.:{:5.2f}% '.format(100 * metric[i, :i + 1].mean()), end='')
            print()
    print('*' * 108)
