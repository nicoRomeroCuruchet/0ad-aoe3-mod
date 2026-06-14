import argparse
import numpy as np
from stable_baselines3 import SAC
from rl.gather import ZeroADGatherEnv


def run_episode(env, model, deterministic):
    obs, _ = env.reset()
    total, done = 0.0, False
    info = {}
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, r, term, trunc, info = env.step(action)
        total += r
        done = term or trunc
    return total, info["distance"], term


def evaluate(env, model, episodes, deterministic):
    rewards, dists, reached = [], [], 0
    for ep in range(episodes):
        total, dist, term = run_episode(env, model, deterministic)
        rewards.append(total)
        dists.append(dist)
        reached += int(term)
        print("ep %2d: reward=%.1f dist_final=%.1f alcanzado=%s" % (ep, total, dist, term))
    modo = "determinista" if deterministic else "estocástico"
    print("-- %d episodios (%s): reward medio=%.1f | dist_final media=%.1f | éxito=%d/%d --"
          % (episodes, modo, np.mean(rewards), np.mean(dists), reached, episodes))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="rl/sac_gather")
    p.add_argument("--config", default="rl/reset_config.json")
    p.add_argument("--uri", default="http://localhost:6000")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--mode", choices=["deterministic", "stochastic", "both"], default="both")
    args = p.parse_args()

    env = ZeroADGatherEnv(open(args.config).read(), uri=args.uri)
    model = SAC.load(args.model)

    if args.mode in ("deterministic", "both"):
        evaluate(env, model, args.episodes, deterministic=True)
    if args.mode in ("stochastic", "both"):
        evaluate(env, model, args.episodes, deterministic=False)


if __name__ == "__main__":
    main()
