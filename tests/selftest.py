"""End-to-end self test on synthetic data: generate, build twice, prove the build is deterministic.

    python tests/selftest.py            # ~1 minute; touches nothing in ./data or ./cache

Use it after any change to src/etl/, and on a new machine before the first real build.
"""

import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def run(cmd, env):
    print("$", " ".join(cmd), flush=True)
    res = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout[-3000:], res.stderr[-3000:])
        sys.exit("FAILED: " + " ".join(cmd))
    return res.stdout


def main():
    tmp = tempfile.mkdtemp(prefix="mtah_selftest_")
    data, out1, out2 = (os.path.join(tmp, d) for d in ("data", "cache1", "cache2"))
    env = dict(os.environ, MTAH_DATA_DIR=data)
    run([sys.executable, "tests/make_synthetic.py", "--out", data], env)
    run([sys.executable, "-m", "src.etl.build", "--out", out1], env)
    run([sys.executable, "-m", "src.etl.build", "--out", out2, "--force-parse"], env)

    def outputs(folder):
        text = open(os.path.join(folder, "report.txt")).read()
        return text.split("OUTPUTS", 1)[1].split()

    def hashes(folder):
        tok = outputs(folder)
        return {tok[i]: tok[i + 4] for i in range(0, len(tok), 5)}

    h1, h2 = hashes(out1), hashes(out2)
    for name in h1:
        print("  {:<18} {}  {}".format(name, h1[name], "same" if h1[name] == h2[name] else "DIFFERENT"))
    if h1 != h2:
        sys.exit("FAILED: two builds of the same input differ")
    out = run([sys.executable, "-m", "src.etl.verify_public", "--cache", out1,
               "--gt", os.path.join(data, "synthetic_gt.csv")], env)
    if "WHO OWNS THE BALL" not in out:
        sys.exit("FAILED: verify_public produced no report")
    report = open(os.path.join(out1, "report.txt")).read()
    if "[FAIL]" in report:
        sys.exit("FAILED: FAIL-level check on synthetic data")
    print("\nOK -- build runs end to end and is deterministic. (WARNs about ball visibility are")
    print("expected on synthetic data; on the real data they should PASS.)")


if __name__ == "__main__":
    main()
