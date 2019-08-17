
import sys
import argparse
import os
from policies.Policy import Policy
from policies import *

from GameConfigurations import PLAYER_INIT_TIME



def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_argument_group('I/O')

    g.add_argument('--log_file', '-l', type=str, default=None,
                   help="a path to which game events are logged. default: game.log")
    g.add_argument('--output_file', '-o', type=str, default=None,
                   help="a path to a file in which game results are written. default: game.out")


    g = p.add_argument_group('Game')
    g.add_argument('--board_size', '-bs', type=str, default='(60,20)', help='a tuple of (height, width)')
    g.add_argument('--obstacle_density', '-od', type=float, default=.04, help='the density of obstacles on the board')
    g.add_argument('--policy_wait_time', '-pwt', type=float, default=0.01,
                   help='seconds to wait for policies to respond with actions')
    g.add_argument('--random_food_prob', '-fp', type=float, default=.2,
                   help='probability of a random food appearing in a round')
    g.add_argument('--max_item_density', '-mid', type=float, default=.25,
                   help='maximum item density in the board (not including the players)')
    g.add_argument('--food_ratio', '-fr', type=float, default=.2,
                   help='the ratio between a corpse and the number of food items it produces')
    g.add_argument('--game_duration', '-D', type=int, default=10000, help='number of rounds in the session')
    g.add_argument('--policy_action_time', '-pat', type=float, default=0.01,
                   help='seconds to wait for agents to respond with actions')
    g.add_argument('--policy_learn_time', '-plt', type=float, default=0.1,
                   help='seconds to wait for agents to improve policy')
    g.add_argument('--player_init_time', '-pit', type=float, default=PLAYER_INIT_TIME,
                   help='seconds to wait for agents to initialize in the beginning of the session')
    g.add_argument('--init_player_size', '-is', type=int, default=5, help='player length at start, minimum is 3')


    g = p.add_argument_group('Players')
    g.add_argument('--score_scope', '-s', type=int, default=1000,
                   help='The score is the average reward during the last score_scope rounds of the session')
    g.add_argument('--policies', '-P', type=str, default=None,
                   help='a string describing the policies to be used in the game, of the form: '
                        '<policy_name>(<arg=val>,*);+.\n'
                        'e.g. MyPolicy(layer1=100,layer2=20);YourPolicy(your_params=123)')

    args = p.parse_args()

    # set defaults
    code_path = os.path.split(os.path.abspath(__file__))[0] + os.path.sep

    if args.log_file is None:
        args.__dict__['log_file'] = code_path + 'game.log'
    if args.output_file is None:
        args.__dict__['output_file'] = code_path + 'game.out'


    args.__dict__['board_size'] = [int(x) for x in args.board_size[1:-1].split(',')]
    plcs = []
    if args.policies is not None: plcs.extend(args.policies.split(';'))
    names = args.policies.split(';')
    args.__dict__['policies'] = [build_policies(p) for p in plcs]

    return args, names


POLICIES = {}

def collect_policies():
    """
    internal function for collecting the policies in the folder.
    """
    if POLICIES: return POLICIES # only fill on first function call
    for mname in sys.modules:
        if not mname.startswith('policies.policy'): continue
        mod = sys.modules[mname]
        for cls_name in dir(mod):
            try:
                if cls_name != 'Policy':
                    cls = mod.__dict__[cls_name]
                    if issubclass(cls, Policy.Policy): POLICIES[cls_name] = cls
            except TypeError:
                pass
    return POLICIES
def build_policies(policy_string):
    """
    internal function for building the desired policies when running the game.
    :param policy_string: the policy string entered when running the game.
    """
    available_policies = collect_policies()

    name, args = policy_string.split('(')
    name = name.replace(" ", "")
    if name not in available_policies:
        raise ValueError('no such policy: %s' % name)
    P = available_policies[name]
    kwargs = dict(tuple(arg.split('=')) for arg in args[:-1].split(',') if arg)
    return P, kwargs