# pcap_to_json.py
# Convert a .pcapng file to JSON using tshark

import shutil
import subprocess
import shlex
import os

def ensure_tshark():
    """Check if tshark is installed; install it if missing."""
    if shutil.which("tshark"):
        print("tshark found:", shutil.which("tshark"))
        return
    print("Installing tshark ...")
    subprocess.run(["apt-get", "update"], check=True)
    subprocess.run(["apt-get", "install", "-y", "tshark"], check=True)
    print("tshark installed:", shutil.which("tshark"))

def convert_pcap_to_json(pcap_path, json_out):
    """Convert a .pcapng file into JSON using tshark."""
    if not os.path.exists(pcap_path):
        raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

    ensure_tshark()

    cmd = f"tshark -r {shlex.quote(pcap_path)} -T json -n"
    print("Running:", cmd)

    with open(json_out, "wb") as fout:
        proc = subprocess.run(cmd, shell=True, check=False, stdout=fout, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            print("tshark error:\n", proc.stderr.decode("utf-8", errors="ignore"))
        else:
            print("Done:", json_out)

if __name__ == "__main__":
    # Modify these paths as needed
    PCAP_PATH = "/content/chatgpt1.pcapng"
    JSON_OUT  = "/content/chatgpt1.json"
    convert_pcap_to_json(PCAP_PATH, JSON_OUT)
