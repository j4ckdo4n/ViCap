import torch
import clip
import faiss
import numpy as np
from transformers import ViTFeatureExtractor, AutoTokenizer, CLIPFeatureExtractor, AutoModel, AutoModelForCausalLM
from transformers.models.auto.configuration_auto import AutoConfig
try:
    from src.vision_encoder_decoder import SmallCap, SmallCapConfig
    from src.gpt2 import ThisGPT2Config, ThisGPT2LMHeadModel
    from src.utils import prep_strings, postprocess_preds
except:
    from models.src.vision_encoder_decoder import SmallCap, SmallCapConfig
    from models.src.gpt2 import ThisGPT2Config, ThisGPT2LMHeadModel
    from models.src.utils import prep_strings, postprocess_preds
from tqdm import tqdm
from PIL import Image, ImageFile
import json 
ImageFile.LOAD_TRUNCATED_IMAGES = True
def load_model(checkpoint_path,device):

    config = AutoConfig.from_pretrained(checkpoint_path + '/config.json')
    model = AutoModel.from_pretrained(checkpoint_path)
    model.config = config
    model.eval()
    model.to(device)
    return model
def retrieve_caps(image_embedding, index, k=5):
    xq = image_embedding.astype(np.float32)
    faiss.normalize_L2(xq)
    D, I = index.search(xq, k) 
    return I
class Predictor_SmallCap:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
        tokenizer = AutoTokenizer.from_pretrained("imthanhlv/gpt2news")
        tokenizer.pad_token = '!'
        tokenizer.eos_token = '.'
        self.tokenizer = tokenizer
        self.dataset_name="UIT_VIIC"
        AutoConfig.register("this_gpt2", ThisGPT2Config)
        AutoModel.register(ThisGPT2Config, ThisGPT2LMHeadModel)
        AutoModelForCausalLM.register(ThisGPT2Config, ThisGPT2LMHeadModel)
        AutoConfig.register("smallcap", SmallCapConfig)
        AutoModel.register(SmallCapConfig, SmallCap)
        self.model= load_model("weights/smallcapuit/",self.device)
        self.template = open('models/src/template.txt').read().strip() + ' '
        self.captions = json.load(open('datastoreuit/coco_index_captions_vi.json'))
        self.retrieval_model, self.feature_extractor_retrieval = clip.load("RN50x64", device=self.device)
        retrieval_index = faiss.read_index('datastoreuit/coco_index')
        res = faiss.StandardGpuResources()  
        self.retrieval_index = faiss.index_cpu_to_gpu(res, 0, retrieval_index)
    def predict(self, image_path,dataset_name="VietCap4H"):
        if dataset_name!=self.dataset_name:
            self.dataset_name=dataset_name
            self.load_model_dataset(dataset_name)
        
        image = Image.open(image_path).convert("RGB")
        pixel_values_retrieval = self.feature_extractor_retrieval(image).to(self.device)
        with torch.no_grad():
            image_embedding = self.retrieval_model.encode_image(pixel_values_retrieval.unsqueeze(0)).cpu().numpy()
        nns = retrieve_caps(image_embedding, self.retrieval_index,k=50)[0]
        caps = [self.captions[i] for i in nns][:30]
        #print(caps)
        caps = list(set(caps))
        caps = caps[:4]
        decoder_input_ids = prep_strings('', self.tokenizer, template=self.template, retrieved_caps=caps, k=4, is_test=True)
        caps_list=":".join(caps)
        # generate caption
        pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values
        with torch.no_grad():
            pred = self.model.generate(pixel_values.to(self.device),
                                decoder_input_ids=torch.tensor([decoder_input_ids]).to(self.device),
                                max_new_tokens=200,
                                min_length=1, num_beams=8, eos_token_id=self.tokenizer.eos_token_id)
        cap=postprocess_preds(self.tokenizer.decode(pred[0]), self.tokenizer)
        return cap,caps
    def load_model_dataset(self,dataset_name):
        #coco_index_file
        if dataset_name=="UIT_VIIC":
            checkpoint_path="weights/smallcapuit/"
            coco_index_file="datastoreuit/coco_index"
            caption_retreival="datastoreuit/coco_index_captions_vi.json"
        if dataset_name=="VietCap4H":
            checkpoint_path="weights/smallcap/"
            coco_index_file="datastore/coco_index"
            caption_retreival="datastore/coco_index_captions_vi.json"
        
        self.model= load_model(checkpoint_path,self.device)
        self.captions = json.load(open(caption_retreival))
        self.retrieval_model, self.feature_extractor_retrieval = clip.load("RN50x64", device=self.device)
        retrieval_index = faiss.read_index(coco_index_file)
        res = faiss.StandardGpuResources()  
        self.retrieval_index = faiss.index_cpu_to_gpu(res, 0, retrieval_index)
        
    
def main():
    import time 
    import os 
    predictor = Predictor_SmallCap()
    images_dir="vietcap4h-private-test/images/"
    start_time=time.time()
    for file in tqdm(os.listdir(images_dir)):
        start=time.time()
        image_path=images_dir+file
        cap,caps=predictor.predict(image_path)
        #print("Time predict",time.time()-start)
    end_time=time.time()
    print("Time per image: ",(end_time-start_time)/len(os.listdir(images_dir)))

   
    
if __name__ == "__main__":
    main()

