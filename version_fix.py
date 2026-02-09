
import sys
import pkg_resources
from packaging.version import Version, InvalidVersion

bad = []

for dist in pkg_resources.working_set:
    try:
        Version(dist.version)
    except InvalidVersion:
        bad.append((dist.project_name, dist.version))

print("Invalid version packages:")
for b in bad:
    print(b)
