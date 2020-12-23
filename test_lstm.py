import time
import joblib
import os
import os.path as osp
# import tensorflow as tf
import torch
from spinup import EpochLogger
# from spinup.utils.logx import restore_tf_graph
from core_lstm import userCritic, userActor
from env import create_train_env



device = torch.device('cuda')
def load_pytorch_policy(fpath, itr='', deterministic=False):
    """ Load a pytorch policy saved with Spinning Up Logger."""

    fname = osp.join(fpath, 'pyt_save', 'model' + itr + '.pt')
    print('\n\nLoading from %s.\n\n' % fname)

    model = torch.load(fname)

    # make function for producing an action given a single state
    def get_action(x, hidden):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            # action = model.act(x)
            pi, _, hidden = model(x.to(device), None, hidden)
            action = pi.sample()
        return action.cpu(), hidden

    return get_action


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):
    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    logger = EpochLogger()

    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    hidden = (torch.zeros((1, 512), dtype=torch.float).to(device), torch.zeros((1, 512), dtype=torch.float).to(device))
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        a, hidden = get_action(o, hidden)
        o, r, d, _ = env.step(a.numpy().item())
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d' % (n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            hidden=(torch.zeros((1, 512), dtype=torch.float).to(device), torch.zeros((1, 512), dtype=torch.float).to(device))
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', '-f', type=str, default='./pretrain')
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()
    # env, get_action = load_policy_and_env(args.fpath,
    #                                       args.itr if args.itr >= 0 else 'last',
    #                                       args.deterministic)
    env = create_train_env(1,1, 'complx')
    get_action = load_pytorch_policy(args.fpath)#itr='_50'
    run_policy(env, get_action, args.len, args.episodes, not (args.norender))
