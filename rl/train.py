import argparse
from stable_baselines3 import SAC
from rl.gather import ZeroADGatherEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="rl/reset_config.json")
    p.add_argument("--uri", default="http://localhost:6000")
    p.add_argument("--timesteps", type=int, default=5000)
    p.add_argument("--out", default="rl/sac_gather")
    args = p.parse_args()

    env = ZeroADGatherEnv(open(args.config).read(), uri=args.uri)
    model = SAC("MlpPolicy", env, verbose=1, learning_starts=200,
                buffer_size=50000)
    model.learn(total_timesteps=args.timesteps, log_interval=4)
    model.save(args.out)

    # Evaluacion rapida de la politica entrenada.
    reached = 0
    episodes = 10
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        if term:
            reached += 1
        print("episodio: dist_final=%.2f alcanzado=%s" % (info["distance"], term))
    print("ALCANZADOS %d/%d" % (reached, episodes))


if __name__ == "__main__":
    main()
