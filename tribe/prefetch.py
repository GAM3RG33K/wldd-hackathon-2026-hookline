from huggingface_hub import snapshot_download, hf_hub_download
hf_hub_download("facebook/tribev2","best.ckpt")
for r in ["facebook/w2v-bert-2.0","facebook/vjepa2-vitg-fpc64-256"]:
    print("fetch",r); snapshot_download(r, allow_patterns=["*.json","*.safetensors","*.txt","*.model"])
print("done open weights")
