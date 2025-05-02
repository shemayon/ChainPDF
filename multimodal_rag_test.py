
# GPU Detection and Configuration

import torch

def get_optimal_device_config():
    if not torch.cuda.is_available():
        return {"rag_device": "cpu", "vl_device": "cpu"}
    
    num_gpus = torch.cuda.device_count()
    total_memory = {i: torch.cuda.get_device_properties(i).total_memory for i in range(num_gpus)}
        
    if num_gpus >= 2:
        # Use separate GPUs - choose the ones with most available memory
        gpu0, gpu1 = sorted(total_memory.items(), key=lambda kv: kv[1], reverse=True)[:2]
        return {
            "rag_device": f"cuda:{gpu0[0]}", 
            "vl_device": f"cuda:{gpu1[0]}"
        }
    else:
        # Single GPU - use the same device for both
        return {
            "rag_device": "cuda:0",
            "vl_device": "cuda:0"
        }

# Get optimal device configuration
device_config = get_optimal_device_config()


from byaldi import RAGMultiModalModel
# Optionally, you can specify an `index_root`, which is where it'll save the index. It defaults to ".byaldi/".
RAG = RAGMultiModalModel.from_pretrained(
    "vidore/colqwen2-v1.0",
    device=torch.device(device_config["rag_device"])
) 
