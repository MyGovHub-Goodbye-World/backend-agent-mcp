"""Small helper to deploy the Serverless service using the Node-based Serverless Framework.

This script does not run anything by default; it's a convenience wrapper you can run locally
if you have Node.js and Serverless installed (or rely on npx).
"""
import subprocess
import shutil
import sys


def main():
    # Prefer npx (ships with recent npm) so users don't need a global install.
    if shutil.which('npx'):
        cmd = ['npx', 'serverless', 'deploy']
    elif shutil.which('serverless'):
        cmd = ['serverless', 'deploy']
    else:
        print('Please install the Serverless Framework (npm i -g serverless) or use npx.')
        sys.exit(2)

    print('Running:', ' '.join(cmd))
    subprocess.run(cmd)


if __name__ == '__main__':
    main()
