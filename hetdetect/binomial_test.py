import scipy


def main():
    # the test for DP=12 and AD=2
    binom12_2 = scipy.stats.binomtest(2, n=12, p=0.5, alternative='two-sided')

    # the test for DP=10 and AD=2
    binom10_2 = scipy.stats.binomtest(2, n=10, p=0.5, alternative='two-sided')

    # print results
    print('Binomial Test P-Values')
    print(f'AD = 2, DP = 12: {binom12_2.pvalue}')
    print(f'AD = 2, DP = 10: {binom10_2.pvalue}')


if __name__ == '__main__':
    main()
