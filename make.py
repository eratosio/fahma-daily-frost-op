import tarfile
import os
import requests
import json
import yaml
import argparse
import subprocess
import tempfile
from eratos.creds import AccessTokenCreds
from eratos.adapter import Adapter