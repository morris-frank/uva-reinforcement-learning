from random import choice, randint, random


def argmax(listable):
    return choice([i for i, v in enumerate(listable) if v == max(listable)])


S = {
    'A': [([0], 'B'), ([1.5], 'T')],
    'B': [([0,2], 'T'), ([2,0], 'T'), ([2,2], 'T'), ([0,0], 'T')],
    'T': [([0], 'T')]
}


def make_q():
    return {s: [0]*len(S[s]) for s in S.keys()}


def make_sarsa(Q, α=1e-4):
    def _step(s, a, s_, a_, r):
        Q[s][a] = Q[s][a] + α * (r + Q[s_][a_] - Q[s][a])
    return _step


def make_qlearn(Q, α=1e-4, γ=1):
    def _step(s, a, s_, a_, r):
        Q[s][a] = Q[s][a] + α * (r + γ * max(Q[s_]) - Q[s][a])
    return _step


def make_double_qlearn(Q, α=1e-4, γ=1):
    Q1, Q2 = make_q(), make_q()
    def _step(s, a, s_, a_, r):
        if random() < 0.5:
            Q1[s][a] = Q1[s][a] + α * (r + γ*Q2[s_][argmax(Q1[s_])] - Q1[s][a])
        else:
            Q2[s][a] = Q2[s][a] + α * (r + γ*Q1[s_][argmax(Q2[s_])] - Q2[s][a])
        Q[s][a] = Q1[s][a] + Q2[s][a]
    return _step


def ε_greedy(Q, ε=1):
    def choose(s):
        if random() < ε:
            return randint(0, len(S[s])-1)
        else:
            return argmax(Q[s])
    return choose


def run(Q, step, steps=int(1e6)):
    π = ε_greedy(Q)
    for _ in range(steps):
        s = 'A'
        #s = choice(list(S.keys()))
        a = π(s)

        while s != 'T':
            rewards, s_ = S[s][a]
            r = choice(rewards)
            a_ = π(s_)
            step(s, a, s_, a_, r)
            s, a = s_, a_


nrounds = int(1e2)
for make_step in (make_qlearn, make_sarsa, make_double_qlearn):
    Q = make_q()
    for _ in range(nrounds):
        _Q = make_q()
        run(_Q, make_step(_Q, α=0.1), 300)
        for s, qs in _Q.items():
            Q[s] = [q1+q2 for q1, q2 in zip(Q[s], qs)]
    for s, qs in Q.items():
        Q[s] = [q/nrounds for q in Q[s]]
        print(f"{s}: {Q[s]}")
