from huggingface_hub import hf_hub_download, HfApi
import json

p = hf_hub_download("willdepueoai/parameter-golf", "manifest.json", subfolder="datasets", repo_type="dataset")
m = json.loads(open(p).read())
print("Datasets:", [d["name"] for d in m.get("datasets", [])])
print("Tokenizers:", [t["name"] for t in m.get("tokenizers", [])])

api = HfApi()
files = [f for f in api.list_repo_files("kevclark/parameter-golf", repo_type="dataset") if "8192" in f]
print("kevclark sp8192 files:", files[:20])
