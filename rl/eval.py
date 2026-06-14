import argparse
import time
import numpy as np
from stable_baselines3 import SAC
from rl.gather import ZeroADGatherEnv
from rl.gather.core import denormalize_action


def run_episode(env, model, deterministic, verbose=False, delay=0.0):
    obs, _ = env.reset()
    total, done, step = 0.0, False, 0
    info = {}
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, r, term, trunc, info = env.step(action)
        total += r
        done = term or trunc
        if delay:
            time.sleep(delay)
        if verbose:
            tx, tz = denormalize_action(action, env.map_size_m)
            print("    step %2d: target=(%.0f,%.0f) dist=%.1f reward=%+.2f"
                  % (step, tx, tz, info["distance"], r))
        step += 1
    return total, info["distance"], term


def evaluate(env, model, episodes, deterministic, verbose=False, delay=0.0):
    rewards, dists, reached = [], [], 0
    for ep in range(episodes):
        if verbose:
            print("  episodio %d:" % ep)
        total, dist, term = run_episode(env, model, deterministic, verbose, delay)
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
    p.add_argument("--verbose", action="store_true",
                   help="traza paso a paso (target, distancia, reward) para analizar el comportamiento")
    p.add_argument("--delay", type=float, default=0.0,
                   help="segundos de pausa entre steps (p.ej. 0.3) para SEGUIR la partida en la ventana de 0 A.D.")
    p.add_argument("--replay", action="store_true",
                   help="guarda un replay de cada episodio para verlo después en 0 A.D. (menú Replays)")
    args = p.parse_args()

    env = ZeroADGatherEnv(open(args.config).read(), uri=args.uri, save_replay=args.replay)
    model = SAC.load(args.model)

    if args.mode in ("deterministic", "both"):
        evaluate(env, model, args.episodes, deterministic=True, verbose=args.verbose, delay=args.delay)
    if args.mode in ("stochastic", "both"):
        evaluate(env, model, args.episodes, deterministic=False, verbose=args.verbose, delay=args.delay)


if __name__ == "__main__":
    main()
