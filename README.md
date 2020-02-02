# Multi-armed bandits Implementation & Evaluation

This project implements famous MAB algorithms and evaulates them on the basis of their performance.

## Offline evalution for algorithms without parameter tuning

``
....
# Run offline evaluation for algorithms
    # EpsilonGreedy
    mab = EpsGreedy(10, params['eps_greedy'])
    results_EpsGreedy = offlineEvaluate(mab, arms, rewards, contexts, T)
    # UCB
    mab = UCB(10, params['ucb'])
    results_UCB = offlineEvaluate(mab, arms, rewards, contexts, T)
    # BetaThompson
    mab = BetaThompson(10, params['beta_thompson'][0], params['beta_thompson'][1])
    results_BetaThompson = offlineEvaluate(mab, arms, rewards, contexts, T)
    # LinUCB
    mab = LinUCB(10, 10, params['lin_ucb'])
    results_LinUCB = offlineEvaluate(mab, arms, rewards, contexts, T)
    # LinThompson
    mab = LinThompson(10, 10, params['lin_thompson'])
    results_LinThompson = offlineEvaluate(mab, arms, rewards, contexts, T)
....
``

