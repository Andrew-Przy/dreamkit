import pathlib
import functools
import torch
import torch.distributions as torchd
from dreamkit.utils.tools import set_seed_everywhere, enable_deterministic_run, Logger, load_episodes, simulate, recursively_load_optim_state_dict, recursively_collect_optim_state_dict, make_dataset, to_np, OneHotDist
from dreamkit.agents.models.dreamerv3.dreamer import Dreamer
from dreamkit.utils.parallel import Parallel, Damy
from dreamkit.envs import wrappers

def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))

def make_env(config, mode, id):
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import dreamkit.envs.dmc as dmc

        env = dmc.DeepMindControl(
            task, config.action_repeat, config.size, seed=config.seed + id
        )
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import dreamkit.envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.grayscale,
            noops=config.noops,
            lives=config.lives,
            sticky=config.stickey,
            actions=config.actions,
            resize=config.resize,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "dmlab":
        import dreamkit.envs.dmlab as dmlab

        env = dmlab.DeepMindLabyrinth(
            task,
            mode if "train" in mode else "test",
            config.action_repeat,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from dreamkit.envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import dreamkit.envs.crafter as crafter

        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "minecraft":
        import dreamkit.envs.minecraft as minecraft

        env = minecraft.make_env(task, size=config.size, break_speed=config.break_speed)
        env = wrappers.OneHotAction(env)
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    if suite == "minecraft":
        env = wrappers.RewardObs(env)
    return env

class Agent:
    def __init__(self, config):
        self.config = config
        self._setup()
        self._initialize_envs()
        self._prefill_data()
        self._train_agent()

    def _setup(self):
        set_seed_everywhere(self.config.seed)
        if self.config.deterministic_run:
            enable_deterministic_run()
        
        logdir = pathlib.Path(self.config.logdir).expanduser()
        self.config.traindir = self.config.traindir or logdir / "train_eps"
        self.config.evaldir = self.config.evaldir or logdir / "eval_eps"
        self.config.steps //= self.config.action_repeat
        self.config.eval_every //= self.config.action_repeat
        self.config.log_every //= self.config.action_repeat
        self.config.time_limit //= self.config.action_repeat
        
        print("Logdir", logdir)
        logdir.mkdir(parents=True, exist_ok=True)
        self.config.traindir.mkdir(parents=True, exist_ok=True)
        self.config.evaldir.mkdir(parents=True, exist_ok=True)
        
        self.step = count_steps(self.config.traindir)
        self.logger = Logger(logdir, self.config.action_repeat * self.step)
        self.logdir = logdir

    def _initialize_envs(self):
        print("Create envs.")
        directory = self.config.offline_traindir.format(**vars(self.config)) if self.config.offline_traindir else self.config.traindir
        self.train_eps = load_episodes(directory, limit=self.config.dataset_size)
        directory = self.config.offline_evaldir.format(**vars(self.config)) if self.config.offline_evaldir else self.config.evaldir
        self.eval_eps = load_episodes(directory, limit=1)
        
        make = lambda mode, id: make_env(self.config, mode, id)
        self.train_envs = [make("train", i) for i in range(self.config.envs)]
        self.eval_envs = [make("eval", i) for i in range(self.config.envs)]
        
        if self.config.parallel:
            self.train_envs = [Parallel(env, "process") for env in self.train_envs]
            self.eval_envs = [Parallel(env, "process") for env in self.eval_envs]
        else:
            self.train_envs = [Damy(env) for env in self.train_envs]
            self.eval_envs = [Damy(env) for env in self.eval_envs]
        
        acts = self.train_envs[0].action_space
        print("Action Space", acts)
        self.config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    def _prefill_data(self):
        if not self.config.offline_traindir:
            prefill = max(0, self.config.prefill - count_steps(self.config.traindir))
            print(f"Prefill dataset ({prefill} steps).")
            
            acts = self.train_envs[0].action_space
            if hasattr(acts, "discrete"):
                random_actor = OneHotDist(
                    torch.zeros(self.config.num_actions).repeat(self.config.envs, 1)
                )
            else:
                random_actor = torchd.independent.Independent(
                    torchd.uniform.Uniform(
                        torch.tensor(acts.low).repeat(self.config.envs, 1),
                        torch.tensor(acts.high).repeat(self.config.envs, 1),
                    ),
                    1,
                )
            
            def random_agent(o, d, s):
                action = random_actor.sample()
                logprob = random_actor.log_prob(action)
                return {"action": action, "logprob": logprob}, None
            
            self.state = simulate(
                random_agent,
                self.train_envs,
                self.train_eps,
                self.config.traindir,
                self.logger,
                limit=self.config.dataset_size,
                steps=prefill,
            )
            self.logger.step += prefill * self.config.action_repeat
            print(f"Logger: ({self.logger.step} steps).")
        else:
            self.state = None

    def _train_agent(self):
        print("Simulate agent.")
        train_dataset = make_dataset(self.train_eps, self.config)
        eval_dataset = make_dataset(self.eval_eps, self.config)
        
        self.agent = Dreamer(
            self.train_envs[0].observation_space,
            self.train_envs[0].action_space,
            self.config,
            self.logger,
            train_dataset,
        ).to(self.config.device)
        
        self.agent.requires_grad_(requires_grad=False)
        checkpoint_path = self.logdir / "latest.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            self.agent.load_state_dict(checkpoint["agent_state_dict"])
            recursively_load_optim_state_dict(self.agent, checkpoint["optims_state_dict"])
            self.agent._should_pretrain._once = False
        
        while self.agent._step < self.config.steps + self.config.eval_every:
            self.logger.write()
            if self.config.eval_episode_num > 0:
                print("Start evaluation.")
                eval_policy = functools.partial(self.agent, training=False)
                simulate(
                    eval_policy,
                    self.eval_envs,
                    self.eval_eps,
                    self.config.evaldir,
                    self.logger,
                    is_eval=True,
                    episodes=self.config.eval_episode_num,
                )
                if self.config.video_pred_log:
                    video_pred = self.agent._wm.video_pred(next(eval_dataset))
                    self.logger.video("eval_openl", to_np(video_pred))
            
            print("Start training.")
            self.state = simulate(
                self.agent,
                self.train_envs,
                self.train_eps,
                self.config.traindir,
                self.logger,
                limit=self.config.dataset_size,
                steps=self.config.eval_every,
                state=self.state,
            )
            items_to_save = {
                "agent_state_dict": self.agent.state_dict(),
                "optims_state_dict": recursively_collect_optim_state_dict(self.agent),
            }
            torch.save(items_to_save, checkpoint_path)
        
        for env in self.train_envs + self.eval_envs:
            try:
                env.close()
            except Exception:
                pass
