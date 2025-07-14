import logging
import os
import sys
from eratos.creds import AccessTokenCreds

from eratos.errors import CommError
from eratos.adapter import Adapter
from dotenv import load_dotenv
_logger = logging.getLogger(__name__)

load_dotenv()
eratos_key = os.environ['ERATOS_KEY']
eratos_secret = os.environ['ERATOS_SECRET']
ecreds = AccessTokenCreds(eratos_key, eratos_secret)
eratos_adapter = Adapter(ecreds)

# eratos_adapter = Adapter()
eratos_adapter.disable_oapi()
print(sys.argv)
base_dir = sys.argv[1]

# base_dir = '/Users/yuxinma/Documents/GitHub/delivery-fahma/SNAP-model-at-polygon'
if len(sys.argv) > 2:
    space = eratos_adapter.Resource(ern=sys.argv[2])
    space_collection = eratos_adapter.Resource(ern=space.prop('collection'))
else:
    # space = eratos_adapter.Resource(ern='ern:e-pn.io:resource:KW46XR5NGIL4U7GBK4IVJ5RX')
    # space_collection = eratos_adapter.Resource(ern=space.prop('collection'))
    space_collection = None
print(space_collection)

def is_yaml(path):
    elems = os.path.splitext(path)
    if len(elems) < 2:
        return False
    return elems[1] == '.yaml'

def push_resource(yaml_path, owner=None):
    global eratos_adapter
    with open(yaml_path, 'rt', encoding='utf8') as f:
        res = eratos_adapter.Resource(yaml=f.read())
    if owner is not None:
        res.set_prop('@owner', owner)
    if str(res.type()) == 'ern:e-pn.io:schema:dataset':
        try:
            ores = eratos_adapter.Resource(ern=res.ern())
            pndn = ores.prop('pndn', None)
            if pndn is not None:
                res.set_prop('pndn', pndn)
        except CommError as e:
            if e.status_code != 404:
                raise
            pass
    try:
        res.save()
        print('  - Succeeded')
    except BaseException as err:
        print(f'  - Error: {err}\n')
        if res.type() != 'ern:e-pn.io:schema:namespace':
            sys.exit(1)
    return res.ern()

block_dirs = []
for root, subdirs, files in os.walk(base_dir):
    for filename in files:
        abs_path = os.path.join(root, filename)
        # print('abs path: ', abs_path)
        if is_yaml(abs_path) and os.path.basename(abs_path) == 'block_descriptor.yaml':
            print('block descriptor found')
            block_dirs += [os.path.dirname(abs_path)]

for blk_dir in block_dirs:
    res_files = []
    for root, subdirs, files in os.walk(os.path.join(blk_dir, 'resources')):
        for filename in files:
            abs_path = os.path.join(root, filename)
            if is_yaml(abs_path):
                res_files += [abs_path]
    for root, subdirs, files in os.walk(os.path.join(blk_dir, 'operator')):
        for filename in files:
            abs_path = os.path.join(root, filename)
            if is_yaml(abs_path):
                res_files += [abs_path]

    print('Processing Block: %s' % blk_dir)
    blkid = push_resource(os.path.join(blk_dir, 'block_descriptor.yaml'), None)
    if space_collection is not None:
        space_collection.perform_action('AddItems', { "items": [{ 'ern': str(blkid), 'inheritPermissions': True }] })
    print('    - ID: %s' % blkid)
    for res_file in res_files:
        print('  Processing Resource: %s' % res_file)
        push_resource(res_file, blkid)