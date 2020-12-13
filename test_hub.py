import argparse
import torch

import hubconf


parser = argparse.ArgumentParser()
upstreams = [attr for attr in dir(hubconf) if callable(getattr(hubconf, attr)) and attr[0] != '_']
parser.add_argument('--mode', choices=['list', 'help', 'load'], required=True)
parser.add_argument('--upstream', choices=upstreams)
parser.add_argument('--refresh', action='store_true', help='Whether to re-download upstream contents')
parser.add_argument('--ckpt', help='The PATH/URL/GOOGLE_DRIVE_ID of upstream checkpoint')
args = parser.parse_args()

REPO = 'andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning:benchmark'

if args.mode == 'list':
    print(torch.hub.list(REPO, force_reload=args.refresh))

elif args.mode == 'help':
    print(torch.hub.help(REPO, args.upstream, force_reload=args.refresh))

elif args.mode == 'load':
    print(torch.hub.load(REPO, args.upstream, ckpt=args.ckpt, force_reload=args.refresh, refresh=args.refresh))
